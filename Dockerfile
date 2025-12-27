FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Install system dependencies (build + runtime)
ARG DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y \
    gcc \
    pkg-config \
    portaudio19-dev \
    libsndfile1-dev \
    ffmpeg \
    libgl1 \
    libglib2.0-0 \
    libportaudio2 \
    && rm -rf /var/lib/apt/lists/*

# Setup python environment
# Upgrade pip to latest version
RUN pip install --upgrade pip

# Copy requirements file
COPY requirements.txt .

# Install dependencies using standard pip
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . .

# Expose port and define environment variables
EXPOSE 50001
ENV FLASK_APP=app.py
ENV NUMBA_CACHE_DIR=/tmp

# Run the application
CMD ["gunicorn", "--workers=2", "--timeout=300", "--bind=0.0.0.0:50001", "app:app"]
