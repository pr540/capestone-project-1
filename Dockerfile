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

# Install uv for fast installation
COPY --from=ghcr.io/astral-sh/uv:latest /uv /uv/bin/uv

# Set UV timeout
ENV UV_HTTP_TIMEOUT=600
ENV PYTHONUNBUFFERED=1

# Copy requirements file
COPY requirements.txt .

# Install dependencies in one go
# Removed manual torch install as it's not used by the app and saves space/time
RUN --mount=type=cache,target=/root/.cache/uv \
    /uv/bin/uv pip install --system -r requirements.txt

# Copy the application code
COPY . .

# Expose port and define environment variables
EXPOSE 5000
ENV FLASK_APP=app.py
ENV NUMBA_CACHE_DIR=/tmp

# Run the application
CMD ["gunicorn", "--workers=2", "--timeout=300", "--bind=0.0.0.0:5000", "app:app"]
