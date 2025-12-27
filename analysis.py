import os
import cv2
import pickle
import numpy as np
import librosa
try:
    from fer import FER
except ImportError:
    try:
        from fer.fer import FER
    except ImportError:
        FER = None

detector = None
model = None
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def get_detector():
    global detector
    if detector is None and FER:
        detector = FER(mtcnn=False)
    return detector

def get_model():
    global model
    if model is None:
        try:
            with open('mlp.pkl', 'rb') as f:
                model = pickle.load(f)
        except Exception as e:
            print(f"[ERROR] Loading model failed: {e}")
    return model

def analyze_video_faces(video_path):
    det = get_detector()
    if not det: return None, 0.0
    cap = cv2.VideoCapture(video_path)
    emotions_list = []
    frame_count = 0
    skip_frames = 5
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret: break
        frame_count += 1
        if frame_count % skip_frames == 0:
            result = det.detect_emotions(frame)
            if result: emotions_list.append(result[0]['emotions'])
    cap.release()
    if not emotions_list: return "neutral", 0.0
    
    total_scores = {}
    for ems in emotions_list:
        for k, v in ems.items():
            total_scores[k] = total_scores.get(k, 0) + v
    dominant = max(total_scores, key=total_scores.get)
    total_sum = sum(total_scores.values())
    return dominant, total_scores[dominant] / total_sum if total_sum > 0 else 0.0

def predict_audio_emotion(audio_data, sr):
    m = get_model()
    if not m: return "neutral", 0.0, [0]*len(emotions)
    
    stft = np.abs(librosa.stft(audio_data))
    chromas = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mfccs = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40).T, axis=0)
    mels = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128).T, axis=0)
    features = np.hstack((chromas, mfccs, mels)).reshape(1, -1)
    
    probs = m.predict_proba(features)[0]
    pred = m.predict(features)[0]
    return pred, np.max(probs), probs

def warmup():
    print("[INFO] Warming up...")
    y = np.random.randn(22050).astype(np.float32)
    librosa.feature.mfcc(y=y, sr=22050, n_mfcc=40)
    print("[INFO] Warmup complete.")
Warmup = warmup # reference for app.py
