# Use official PyTorch image with CUDA support
FROM pytorch/pytorch:2.0.1-cuda11.7-cudnn8-runtime

# Set working directory
WORKDIR /app

# Copy all project files to container
COPY . /app

# Install Python dependencies
RUN pip install --upgrade pip && \
    pip install -r requirement.txt

# Set default command (modify if needed)
CMD ["python", "run.py"]
