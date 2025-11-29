# Use an official Python runtime as a parent image
FROM python:3.9-slim

# Set the working directory in the container
WORKDIR /app

# Install system dependencies required for PyAudio and Librosa
# portaudio19-dev is needed for PyAudio
# libsndfile1 is needed for Librosa
# ffmpeg is needed for moviepy to process video files
RUN apt-get update && apt-get install -y \
    gcc \
    portaudio19-dev \
    libsndfile1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

# Copy the current directory contents into the container at /app
COPY . /app

# Install any needed packages specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Make port 5900 available to the world outside this container
EXPOSE 5900

# Define environment variable
ENV FLASK_APP=app.py

# Run app.py when the container launches
CMD ["python", "app.py"]
