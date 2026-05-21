import os
# Fix OMP duplicate library error on Windows
# (torch and faiss-cpu both bundle OpenMP runtimes that conflict)
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

from fastapi import FastAPI, File, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request

import pickle
import faiss
import torch
import open_clip
import numpy as np

from PIL import Image

app = FastAPI()

# Mount folders
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")
app.mount("/dataset", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "dataset")), name="dataset")

templates = Jinja2Templates(directory="templates")

# Load CLIP
device = "cuda" if torch.cuda.is_available() else "cpu"

model, _, preprocess = open_clip.create_model_and_transforms(
    'ViT-B-32',
    pretrained='openai'
)

model = model.to(device)

# Load FAISS
index = faiss.read_index("faiss_index.bin")

with open("image_paths.pkl", "rb") as f:
    image_paths = pickle.load(f)

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):

    return templates.TemplateResponse(
        "index.html",
        {"request": request}
    )

@app.post("/search")
async def search_image(
    request: Request,
    file: UploadFile = File(...)
):

    upload_path = f"uploads/{file.filename}"

    with open(upload_path, "wb") as f:
        f.write(await file.read())

    # Generate embedding
    image = preprocess(
        Image.open(upload_path).convert("RGB")
    ).unsqueeze(0).to(device)

    with torch.no_grad():
        embedding = model.encode_image(image)

    embedding /= embedding.norm(dim=-1, keepdim=True)

    embedding = embedding.cpu().numpy()

    # Search
    k = 5

    distances, indices = index.search(embedding, k)

    results = []

    for score, idx in zip(distances[0], indices[0]):

        path = image_paths[idx]

        # Extract just the filename and map to local dataset/ folder
        # (pickle stores Colab paths like /content/design/design/D-1.jpg)
        filename = os.path.basename(path)

        class_name = os.path.splitext(filename)[0]

        results.append({
            "class_name": class_name,
            "score": float(score),
            "image_path": "dataset/" + filename
        })

    # Return JSON so the frontend can render results via AJAX
    # without a full page reload
    return JSONResponse(content={"results": results})