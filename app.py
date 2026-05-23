import os
# Fix OMP duplicate library error on Windows
# (torch and faiss-cpu both bundle OpenMP runtimes that conflict)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

import logging
import pickle
import uuid
from collections import defaultdict

import faiss
import numpy as np
import timm
import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image

from fastapi import FastAPI, File, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("raaga")

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
UPLOAD_DIR = os.path.join(BASE_DIR, "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)

# ---------------------------------------------------------------------------
# Device
# ---------------------------------------------------------------------------
device = "cuda" if torch.cuda.is_available() else "cpu"
logger.info("Using device: %s", device)

# ---------------------------------------------------------------------------
# DINOv2 Model — loaded ONCE at startup
# ---------------------------------------------------------------------------
logger.info("Loading DINOv2 model …")
model = timm.create_model(
    "vit_base_patch14_dinov2.lvd142m",
    pretrained=True,
    num_classes=0,  # removes classification head → returns 768-d features
)
model = model.to(device)
model.eval()
logger.info("DINOv2 model loaded successfully (dim=768)")

# ---------------------------------------------------------------------------
# Image Preprocessing — DINOv2 native 518×518 with ImageNet normalization
# ---------------------------------------------------------------------------
preprocess = transforms.Compose([
    transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC),
    transforms.CenterCrop(518),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# Resize / crop helpers for patch extraction (applied BEFORE tensor conversion)
_resize = transforms.Resize(518, interpolation=transforms.InterpolationMode.BICUBIC)
_center_crop = transforms.CenterCrop(518)

# ---------------------------------------------------------------------------
# FAISS Index & Metadata — loaded ONCE at startup
# ---------------------------------------------------------------------------
logger.info("Loading FAISS index …")
index = faiss.read_index(os.path.join(BASE_DIR, "textile_index.bin"))
logger.info("FAISS index loaded: %d vectors, dim=%d", index.ntotal, index.d)

with open(os.path.join(BASE_DIR, "image_paths.pkl"), "rb") as f:
    image_paths: list[str] = pickle.load(f)

with open(os.path.join(BASE_DIR, "class_names.pkl"), "rb") as f:
    class_names: list[str] = pickle.load(f)

logger.info(
    "Metadata loaded: %d image paths, %d class names (%d unique)",
    len(image_paths),
    len(class_names),
    len(set(class_names)),
)

# ---------------------------------------------------------------------------
# FastAPI App
# ---------------------------------------------------------------------------
app = FastAPI(title="Raaga Design Similarity Search")

# ---------------------------------------------------------------------------
# Routes MUST be registered BEFORE static-file mounts so that /search and /
# are matched first. StaticFiles acts as a catch-all for its prefix.
# ---------------------------------------------------------------------------


@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.post("/search")
async def search_image(
    request: Request,
    file: UploadFile = File(...),
):
    """
    Accept an uploaded textile/fabric image, extract patches, generate
    DINOv2 embeddings, search the pre-built FAISS index, aggregate scores
    per class, and return the top-5 unique matches.
    """

    # ------------------------------------------------------------------
    # 1. Validate upload
    # ------------------------------------------------------------------
    if not file.content_type or not file.content_type.startswith("image/"):
        return JSONResponse(
            status_code=400,
            content={"error": "Please upload a valid image file (JPEG, PNG, etc.)"},
        )

    # ------------------------------------------------------------------
    # 2. Save upload with unique name
    # ------------------------------------------------------------------
    ext = os.path.splitext(file.filename or "upload.jpg")[1] or ".jpg"
    upload_name = f"{uuid.uuid4().hex}{ext}"
    upload_path = os.path.join(UPLOAD_DIR, upload_name)

    content = await file.read()
    if not content:
        return JSONResponse(
            status_code=400,
            content={"error": "Uploaded file is empty"},
        )

    with open(upload_path, "wb") as f:
        f.write(content)

    # ------------------------------------------------------------------
    # 3. Open and preprocess image
    # ------------------------------------------------------------------
    try:
        image = Image.open(upload_path).convert("RGB")
    except Exception as exc:
        logger.warning("Failed to open uploaded image: %s", exc)
        return JSONResponse(
            status_code=400,
            content={"error": "Could not open the uploaded image. It may be corrupt."},
        )

    # Resize + center crop to 518×518 (PIL Image, for patch extraction)
    image = _resize(image)
    image = _center_crop(image)

    # ------------------------------------------------------------------
    # 4. Extract patches
    # ------------------------------------------------------------------
    patches = _extract_patches(image, patch_size=259)

    # ------------------------------------------------------------------
    # 5. Generate patch embeddings (batched single forward pass)
    # ------------------------------------------------------------------
    patch_embeddings = _get_patch_embeddings(patches)  # (N, 768)

    # ------------------------------------------------------------------
    # 6. Search FAISS for each patch
    # ------------------------------------------------------------------
    k_per_patch = 250
    distances, indices = index.search(patch_embeddings, k_per_patch)

    # ------------------------------------------------------------------
    # 7. Aggregate: best score per class across all patches
    # ------------------------------------------------------------------
    class_best_score: dict[str, float] = defaultdict(float)
    class_image_path: dict[str, str] = {}

    for patch_dists, patch_idxs in zip(distances, indices):
        for score, idx in zip(patch_dists, patch_idxs):
            if idx < 0:  # FAISS returns -1 for missing results
                continue
            cls = class_names[idx]
            s = float(score)
            if s > class_best_score[cls]:
                class_best_score[cls] = s
                # Map Colab-style path → local dataset filename
                filename = os.path.basename(image_paths[idx])
                class_image_path[cls] = f"dataset/{filename}"

    # ------------------------------------------------------------------
    # 8. Rank descending and take top 10
    # ------------------------------------------------------------------
    ranked = sorted(class_best_score.items(), key=lambda x: x[1], reverse=True)[:10]

    if not ranked:
        return JSONResponse(content={"results": []})

    results = [
        {
            "class_name": cls,
            "score": round(score, 4),
            "image_path": class_image_path[cls],
        }
        for cls, score in ranked
    ]

    logger.info("Search completed — top result: %s (%.4f)", results[0]["class_name"], results[0]["score"])
    return JSONResponse(content={"results": results})


# ---------------------------------------------------------------------------
# Static mounts (AFTER route handlers)
# ---------------------------------------------------------------------------
templates = Jinja2Templates(directory=os.path.join(BASE_DIR, "templates"))
app.mount("/static", StaticFiles(directory=os.path.join(BASE_DIR, "static")), name="static")
app.mount("/dataset", StaticFiles(directory=os.path.join(BASE_DIR, "dataset")), name="dataset")


# ===========================================================================
# Helper Functions
# ===========================================================================


def _extract_patches(
    image: Image.Image,
    patch_size: int = 259,
) -> list[Image.Image]:
    """
    Extract non-overlapping patches from a PIL image.
    Also includes the full image as a global-context patch.

    For a 518×518 input with patch_size=259:
    → 2×2 grid = 4 patches + 1 full image = 5 patches total.
    """
    w, h = image.size
    patches: list[Image.Image] = []

    for y in range(0, h - patch_size + 1, patch_size):
        for x in range(0, w - patch_size + 1, patch_size):
            patch = image.crop((x, y, x + patch_size, y + patch_size))
            patches.append(patch)

    # Full image as global-context patch
    patches.append(image)
    return patches


def _get_patch_embeddings(patches: list[Image.Image]) -> np.ndarray:
    """
    Generate L2-normalized DINOv2 embeddings for a list of PIL image patches.

    Each patch is independently preprocessed (resize → center crop → normalize)
    and then batched through the model in a single forward pass.

    Returns: np.ndarray of shape (num_patches, 768), dtype float32.
    """
    tensors = torch.stack([preprocess(p) for p in patches]).to(device)

    with torch.no_grad():
        embeddings = model(tensors)  # (N, 768)

    # L2-normalize so FAISS inner-product search = cosine similarity
    embeddings = F.normalize(embeddings, p=2, dim=1)
    return embeddings.cpu().numpy().astype("float32")