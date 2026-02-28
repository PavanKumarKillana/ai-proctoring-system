# Use the official Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies for OpenCV and MediaPipe
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the entire project
COPY . .

# Expose the Flask port
EXPOSE 7860

# Command to run the application (Hugging Face Spaces exposes port 7860 by default)
CMD ["gunicorn", "-b", "0.0.0.0:7860", "ui.app:app"]
