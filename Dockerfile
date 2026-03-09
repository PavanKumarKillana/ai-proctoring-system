# Hugging Face Spaces — MediaPipe-only build (no PyTorch)
# This is the hosted version. Custom MobileNet is disabled (auto-fallback to MediaPipe).
FROM condaforge/miniforge3:latest

ENV PYTHONDONTWRITEBYTECODE=1
ENV PYTHONUNBUFFERED=1
ENV DEBIAN_FRONTEND=noninteractive

# System dependencies for OpenCV + dlib
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .

# Install dlib + face_recognition via conda-forge pre-built binaries
# (avoids 2+ hour C++ compilation that was killing the build)
RUN conda install -y -c conda-forge face_recognition dlib

# Install Python packages (no PyTorch — saves ~3GB and 10+ min build time)
RUN pip install --no-cache-dir -r requirements.txt

# Pre-download YOLOv8n weights so first request doesn't time out
RUN python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"

# Copy project
COPY . .

# Create data dir for face encodings and DB
RUN mkdir -p data && chown -R 1000:1000 /app

USER 1000

ENV PATH="/opt/conda/bin:$PATH"

# Hugging Face Spaces uses port 7860
EXPOSE 7860

# 1 worker (free tier RAM limit), 120s timeout for model warm-up
CMD ["gunicorn", "--timeout", "120", "-w", "1", "-b", "0.0.0.0:7860", "ui.app:app"]
