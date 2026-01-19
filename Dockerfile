# Use official TensorFlow 1.15 image (x86_64)
FROM --platform=linux/amd64 tensorflow/tensorflow:1.15.5-py3

LABEL maintainer="White-box-Cartoonization"
LABEL description="Docker image for White-box Cartoonization (TensorFlow 1.x)"

# Set environment variables
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies for OpenCV
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Install additional Python dependencies
RUN pip install --no-cache-dir \
    opencv-python==4.5.5.64 \
    scikit-image==0.14.5 \
    tqdm>=4.0.0

# Copy the application code
COPY . .

# Set working directory to test_code for inference
WORKDIR /app/test_code

# Default command to run cartoonization
CMD ["python", "cartoonize.py"]
