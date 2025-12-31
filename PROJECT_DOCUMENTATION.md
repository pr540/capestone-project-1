# Project One: Speech Emotion Recognition Application
**Documentation & Status Report**

## 1. Executive Summary
This project is a sophisticated **Speech Emotion Recognition Application** that analyzes audio and video files to detect human emotions using machine learning. It provides a visual interface for uploading files, processing them, and displaying prediction results with animated emojis and confidence scores. The system is designed to be robust, supporting multiple file formats and prioritizing accuracy through multi-modal analysis (audio + facial expressions).

## 2. Key Features
- **Multi-Format Support**:
  - **Audio**: WAV, MP3, OGG, FLAC, M4A
  - **Video**: MP4, AVI, MOV, MKV, WEBM (Automatic audio extraction)
- **Advanced Machine Learning**: Uses a trained Multi-Layer Perceptron (MLP) model (`mlp.pkl`) for emotion classification.
- **Smart Analysis**:
  - **Fusion Logic**: Combines audio emotion detection with facial expression analysis (Happy, Surprise, Angry) for higher accuracy.
  - **Dual Analysis**: Analyzes facial expressions in videos alongside audio.
- **Visual Feedback**:
  - Animated emojis for detected emotions.
  - Bounce animations and gradient designs for a premium UI feel.
- **Reliability & Validation**:
  - **Input Validation**: Checks file types and sizes (100MB limit).
  - **Error Handling**: Graceful handling of invalid inputs and support for unsupported formats.
- **Database Integration**: Results are automatically saved to a SQLite database (`emotions.db`) for historical tracking.
- **Containerization**: Fully Dockerized for consistent deployment across environments.

## 3. System Architecture & Workflow
### How It Works
1.  **User Upload**: The user uploads an audio or video file via the web interface.
2.  **Validation**:
    -   System checks file extension.
    -   System validates file size (< 100MB).
3.  **Preprocessing**:
    -   **Video**: If video, the audio track is extracted using `moviepy`.
    -   **Audio**: Loaded directly using `librosa` or `pydub`.
4.  **Feature Extraction**:
    -   Extracts MFCC (Mel-frequency cepstral coefficients), Mel-spectrogram, and Chroma features from the audio.
5.  **Prediction**:
    -   The extracted features are fed into the MLP model.
    -   Video frames are analyzed for facial expressions (if applicable).
    -   Results are fused to determine the final emotion.
6.  **Output**:
    -   The detected emotion is returned to the frontend.
    -   The result page displays the corresponding animated emoji and emotion label.
    -   Data is logged to the `emotions.db`.

## 4. Technical Stack
-   **Backend**: Python, Flask, Gunicorn
-   **Machine Learning**: TensorFlow (CPU), Scikit-learn, Librosa, FER (Facial Expression Recognition)
-   **Frontend**: HTML5, CSS3, JavaScript (Jinja2 templates)
-   **Database**: SQLite (SQLAlchemy)
-   **Processing**: FFmpeg, MoviePy, Pydub
-   **DevOps**: Docker, Docker Compose

## 5. Recent Fixes & Improvements (as of Dec 2025)
### Video File Support & Stability
-   **Issue**: System crashed on MP4 uploads.
-   **Fix**: Implemented `moviepy` for audio extraction and added robust file type validation. Failed formats now return clear error messages instead of crashing.

### UI/UX Overhaul
-   **Result Page**: Redesigned with a premium gradient look, large animated emojis, and descriptive text.
-   **Scroll Bug**: Fixed a navbar scroll issue on page load.
-   **Upload Form**: Added real-time file size validation and visual feedback.

### Performance & Security
-   **Upload Limit**: Enforced a strict 100MB file limit to prevent server overload.
-   **Error Handling**: Added custom 413 (Payload Too Large) error handlers and detailed logs.

## 6. Installation & Deployment
### Prerequisites
-   Docker Desktop (Recommended)
-   Git

### Running with Docker Compose (Standard)
```bash
git clone <repo-url>
cd capestone-project-1
docker-compose up --build
```
Access the application at: **http://localhost:50001**

## 7. Future Roadmap
-   Add progress bar for large file uploads.
-   Implement loading spinners during prediction processing.
-   Allow users to download or share prediction results.
-   View history of past predictions in the UI.
