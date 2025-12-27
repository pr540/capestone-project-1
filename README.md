# Speech Emotion Recognition Application

This application analyzes audio and video files to detect human emotions using machine learning. It supports various file formats and provides a visual interface for uploading and viewing prediction results.

## 🚀 Features
- **Multi-Format Support**: 
  - Audio: WAV, MP3, OGG, FLAC, M4A
  - Video: MP4, AVI, MOV, MKV, WEBM
- **Smart Analysis**: Extracts audio from video files automatically.
- **Visual Feedback**: Displays detected emotions with animated emojis and confidence scores.
- **Improved Accuracy**: Enhanced fusion logic that prioritizes facial expressions (Happy, Surprise, Angry) to prevent audio-based misclassifications.
- **Database Integration**: Automatically saves all prediction results to a SQLite database (`emotions.db`) for tracking.
- **Dual Analysis**: Analyzes facial expressions in videos alongside audio emotion detection for high-precision results.

## 📋 Prerequisites

Before you begin, ensure you have the following installed:
- **Git**: [Download Git](https://git-scm.com/downloads)
- **Docker Desktop** (Recommended): [Download Docker](https://www.docker.com/products/docker-desktop/)

*Optional (for local non-Docker setup):*
- Python 3.9 or higher
- FFmpeg (Required for video processing)

---

## 🛠️ Installation & Running

We recommend using **Docker Compose** for the easiest setup.

### Method 1: Docker Compose (Recommended)

1. **Clone the Repository**
   ```bash
   git clone <your-repo-url>
   cd capestone-project-1
   ```

2. **Build and Run**
   Open your terminal/command prompt in the project directory and run:
   ```bash
   docker-compose up --build
   ```

3. **Access the App**
   Once the server starts, open your browser and go to:
   👉 **http://localhost:50001**

   *Note: The app runs on port 50001 by default when using Docker Compose.*

---

### Method 2: Docker CLI (Manual)

If you prefer using standard Docker commands:

1. **Build the Image**
   ```bash
   docker build -t speech-emotion-app .
   ```

2. **Run the Container**
   ```bash
   docker run -p 50001:50001 speech-emotion-app
   ```

3. **Access the App**
   Go to: **http://localhost:50001**

---

### Method 3: Local Python Setup (Without Docker)

If you want to run it directly on your machine, follow these steps carefully.

1. **Install System Dependencies (FFmpeg)**
   - **Windows**: [Download FFmpeg](https://ffmpeg.org/download.html), extract it, and add the `bin` folder to your System PATH environment variable.
   - **Mac**: `brew install ffmpeg`
   - **Linux**: `sudo apt-get install ffmpeg`

2. **Create a Virtual Environment**
   ```bash
   # Windows
   python -m venv venv
   .\venv\Scripts\activate

   # Mac/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install Python Libraries**
   ```bash
   pip install --upgrade pip
   pip install -r requirements.txt
   ```

4. **Run the Application**
   ```bash
   python app.py
   ```

5. **Access the App**
   Go to: **http://localhost:50001**

---

## 📂 Project Structure

```
capestone-project-1/
├── app.py                 # Main Flask application
├── Dockerfile             # Docker build implementation
├── docker-compose.yaml    # Docker services configuration
├── requirements.txt       # Python dependencies
├── mlp.pkl               # Trained Machine Learning Model
├── templates/             # HTML files (frontend)
└── static/               # CSS, Images, JS
```

## ❓ Troubleshooting

**Issue: "File too large" error**
- The application has a 100MB file limit. Ensure your audio/video file is under 100MB.

**Issue: Docker "port already allocated"**
- If port 50001 is busy, edit `docker-compose.yaml` and change `"50001:5000"` to `"50002:5000"`.

**Issue: "ffmpeg not found" (Local Run)**
- **FIXED**: The application now includes `imageio-ffmpeg` which automatically provides the necessary binaries for Windows, Mac, and Linux. No manual installation is required.
