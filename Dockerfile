FROM python:3.11.5-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

RUN pip install ./dlib-19.24.1-cp311-cp311-win_amd64.whl

# Install Python packages with exact versions
RUN pip install \
    face-recognition==1.3.0 \
    opencv-python==4.11.0.86

# Create a volume for syncing code
VOLUME /app

# Copy your application code
COPY . /app

# Command to run your application
CMD ["python", "your_script.py"]
