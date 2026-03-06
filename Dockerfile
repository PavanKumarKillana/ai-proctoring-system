# Use Miniforge, a highly optimized, lightweight Conda environment 
FROM condaforge/miniforge3:latest

# Set environment variables
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1
ENV DEBIAN_FRONTEND=noninteractive

# Install comprehensive system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    ffmpeg \
    libsm6 \
    libxext6 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy requirements
COPY requirements.txt .

# THE "MAGIC" FIX: Install dlib and face_recognition via Conda.
# Conda-forge has pre-compiled binaries for Linux. This completely bypasses
# the massive C++ source compilation that was taking 2+ hours and crashing.
# It takes less than 30 seconds!
RUN conda install -y -c conda-forge face_recognition dlib

# 1. Install CPU-only PyTorch (saves ~3GB and massive RAM) via pip
RUN pip install --no-cache-dir torch torchvision --index-url https://download.pytorch.org/whl/cpu

# 2. Install the rest of the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# The conda image already has a user with UID 1000, so we just reuse it.
# We change ownership of /app to UID 1000 and switch to it.
RUN chown -R 1000:1000 /app
USER 1000

# Ensure the non-root user can execute installed python/conda packages
ENV PATH="/opt/conda/bin:$PATH"

# Expose the Flask port
EXPOSE 7860

# Command to run the application
# CRITICAL: Added --timeout 120 because loading massive PyTorch models 
# on Hugging Face free tier takes longer than Gunicorn's 30s default timeout
CMD ["gunicorn", "--timeout", "120", "-w", "1", "-b", "0.0.0.0:7860", "ui.app:app"]
