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

detector = {}
model = None
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def get_detector():
    global detector
    if not detector and cv2:
        try:
            base = cv2.data.haarcascades
            detector['face'] = cv2.CascadeClassifier(base + 'haarcascade_frontalface_default.xml')
            detector['smile'] = cv2.CascadeClassifier(base + 'haarcascade_smile.xml')
            detector['eye'] = cv2.CascadeClassifier(base + 'haarcascade_eye.xml')
        except Exception:
            detector = None
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
    dets = get_detector()
    if not dets or not cv2: return "neutral", 0.0, {}
    
    cap = cv2.VideoCapture(video_path)
    stats = {'happy': 0, 'surprise': 0, 'neutral': 0, 'sad': 0}
    total_frames = 0
    detected_faces = 0
    
    while cap.isOpened() and total_frames < 60: # Sample 60 frames (approx 2s)
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1
        if total_frames % 3 != 0: continue # Skip
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dets['face'].detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            detected_faces += 1
            roi_gray = gray[y:y+h, x:x+w]
            
            # Improved heuristic detection
            smiles = dets['smile'].detectMultiScale(roi_gray, 1.7, 12)
            eyes = dets['eye'].detectMultiScale(roi_gray, 1.1, 8)
            
            if len(smiles) > 0:
                stats['happy'] += 1
            elif len(eyes) > 2:
                stats['surprise'] += 1
            elif len(eyes) < 2:
                stats['sad'] += 1
            else:
                stats['neutral'] += 1
    
    cap.release()
    # If no faces found at all
    if detected_faces == 0: return "neutral", 0.0, stats
    
    dominant = max(stats, key=stats.get)
    confidence = stats[dominant] / sum(stats.values()) if sum(stats.values()) > 0 else 0.0
    return dominant, confidence, stats

def predict_audio_emotion(audio_data, sr):
    m = get_model()
    if not m: return "neutral", 0.0, [0]*len(emotions)
    
    try:
        features = extract_features_combined(audio_data, sr).reshape(1, -1)
        probs = m.predict_proba(features)[0]
        # Fixed: m.predict already returns the string label, indexing [0] was truncating it
        pred = m.predict(features)
        return pred, np.max(probs), probs
    except Exception as e:
        print(f"Feature extraction failed: {e}")
        return "neutral", 0.0, [0]*len(emotions)

def warmup():
    print("[INFO] Warmup (Static Mode)")
    # No-op for now to save memory
    pass
