import os
import pickle
import numpy as np
import subprocess
import imageio_ffmpeg
from mlp_numpy import NumpyMLP
from audio_features_numpy import extract_features_combined

try:
    import cv2
except ImportError:
    cv2 = None

detector = None
model = None
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def get_detector():
    global detector
    # Use standard Haar Cascade as a lightweight alternative to FER
    if detector is None and cv2:
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        detector = cv2.CascadeClassifier(cascade_path)
    return detector

def get_model():
    global model
    if model is None:
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp_weights.npz')
            model = NumpyMLP(model_path)
        except Exception as e:
            print(f"[ERROR] Loading model failed: {e}")
    return model

def analyze_video_faces(video_path):
    face_cascade = get_detector()
    if not face_cascade or not cv2: return "neutral", 0.0
    
    cap = cv2.VideoCapture(video_path)
    # Since we don't have a specific video-emotion model anymore (it was inside FER),
    # we'll use a simplified neutral fallback for video on serverless,
    # or return neutral if face is detected.
    found_face = False
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 4)
        if len(faces) > 0:
            found_face = True
            break
    cap.release()
    return ("neutral", 0.5) if found_face else (None, 0.0)

def predict_audio_emotion(audio_data, sr):
    m = get_model()
    if not m: return "neutral", 0.0, [0]*len(emotions)
    
    try:
        features = extract_features_combined(audio_data, sr).reshape(1, -1)
        probs = m.predict_proba(features)[0]
        pred = m.predict(features)[0]
        return pred, np.max(probs), probs
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return "neutral", 0.0, [0]*len(emotions)

def warmup():
    print("[INFO] Warmup (Static Mode)")
    # No-op for now to save memory
    pass
