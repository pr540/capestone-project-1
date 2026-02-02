# Project Documentation: Emotion Recognition System (Optimization Phase)

## 1. Overview
A premium Flask-based web application for Speech and Visual Emotion Recognition. Optimized specifically for Vercel's "Serverless" environment limit (250MB).

## 2. Technology Stack
*   **Web Framework**: Flask (Python 3.12)
*   **Frontend**: HTML5, Vanilla CSS3 (Glassmorphism), JavaScript (JQuery, Bootstrap)
*   **Database**: SQLite (SQLAlchemy ORM)
*   **Audio Engine**: Pure NumPy (Custom Feature Extraction: MFCC, Chroma, Mel-spec)
*   **Video Engine**: OpenCV (Haar Cascades for Face, Smile, and Eye detection)
*   **Model Execution**: Custom NumPy MLP Class (No scikit-learn dependency)
*   **Deployment**: Vercel (Production), GitHub Actions (CI/CD), Docker (Scaling)

## 3. Core AI Architecture
### Audio Processing (Proprietary NumPy Engine)
Since `librosa` and `scikit-learn` exceed Serverless size limits, the system uses a custom engine:
*   **Extraction**: RAW PCM data via `ffmpeg` subprocess.
*   **Math**: STFT with reflect padding, triangular Mel filters, and DCT-II ortho-normalization.
*   **Inference**: A dense MLP model running on raw matrix multiplication.

### Visual Processing (Heuristic Engine)
Uses `OpenCV` Haar Cascades to perform real-time facial feature analysis:
*   **Smile Detection**: High-sensitivity smile tracking for 'Happy' categorization.
*   **Eye Tracking**: Wide-eye detection for 'Surprise' categorization.
*   **Sampling**: 30-frame temporal variance check.

## 4. API Documentation
### Internal Application Routes
*   `GET /`: Home page.
*   `GET /prediction_page`: File upload interface.
*   `POST /predict`: Main AI endpoint. Accepts `.mp4`, `.wav`, `.mp3`. Returns detection result.
*   `GET /analyze`: View historical prediction data.

## 5. Testing & Validation
*   **Unit Testing**: Verified via `test_mlp.py`. Compare NumPy predictions against original Scikit-learn outputs (Match: 100%).
*   **Feature Tests**: Verified that NumPy MFCC extraction matches Librosa's `atol=1e-5`.
*   **Integration Tests**: Vercel build status and runtime logs check.

## 6. Deployment Optimization
*   **Size Reduction**: Removed 600MB of dependencies (TensorFlow, TF-Lite, Scikit-learn, Librosa, Scipy).
*   **Bundle Size**: Current unzipped size ~140MB (Limit: 250MB).
*   **Asset Routing**: Configured `vercel.json` to bypass rewrites for `/static` to fix UI caching and broken links.
