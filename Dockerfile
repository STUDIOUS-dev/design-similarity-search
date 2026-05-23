FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (required for FAISS, etc.)
RUN apt-get update && apt-get install -y \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the rest of the app
COPY . .

# Ensure uploads directory exists
RUN mkdir -p /app/uploads

# Hugging Face Spaces exposes port 7860
EXPOSE 7860

# Command to run the application
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "7860"]
