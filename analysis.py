import os
import numpy as np
import subprocess
import imageio_ffmpeg
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import librosa
except ImportError:
    librosa = None

try:
    import joblib
except ImportError:
    joblib = None

from mlp_numpy import NumpyMLP
from audio_features_numpy import extract_features_combined

detector = {}
model = None
# Standard TESS emotions
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
        # Priority 1: Original PKL model (most accurate)
        if joblib:
            try:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp.pkl')
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    print("[INFO] Loaded original PKL model.")
                    return model
            except Exception as e:
                print(f"[WARN] Loading PKL failed: {e}")
        
        # Priority 2: Numpy fallback
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp_weights.npz')
            if os.path.exists(model_path):
                model = NumpyMLP(model_path)
                print("[INFO] Fallback to NumpyMLP.")
            else:
                print("[ERROR] No model weights found!")
        except Exception as e:
            print(f"[ERROR] Loading NumpyMLP failed: {e}")
    return model

def analyze_video_faces(video_path):
    dets = get_detector()
    if not dets or not cv2: return "neutral", 0.0, {}
    
    cap = cv2.VideoCapture(video_path)
    stats = {e: 0 for e in emotions}
    total_frames = 0
    detected_faces = 0
    
    while cap.isOpened() and total_frames < 60: # Sample 60 frames (approx 2s)
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1
        if total_frames % 5 != 0: continue # Skip more frames for performance
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dets['face'].detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            detected_faces += 1
            roi_gray = gray[y:y+h, x:x+w]
            
            # Very sensitive detection for real-world movement
            smiles = dets['smile'].detectMultiScale(roi_gray, 1.2, 3) 
            eyes = dets['eye'].detectMultiScale(roi_gray, 1.1, 3)
            
            if len(smiles) > 0:
                stats['happy'] += 2 # Boost happy if smile detected
            elif len(eyes) > 2:
                stats['ps'] += 1 
            elif len(eyes) < 1:
                stats['sad'] += 1
            else:
                # Any face movement is better than nothing
                stats['neutral'] += 1
    
    cap.release()
    if detected_faces == 0: return "N/A", 0.0, stats
    
    # Selection logic
    dominant = max(stats, key=stats.get)
    val = stats[dominant]
    
    total = sum(stats.values())
    confidence = val / total if total > 0 else 0.0
    return str(dominant), float(confidence), stats

def extract_audio_features(audio_data, sr):
    """Definitive feature extraction function matching training script."""
    # Strict normalization to match TESS profile
    if np.max(np.abs(audio_data)) > 1e-6:
        audio_data = audio_data / np.max(np.abs(audio_data))
    
    if librosa:
        try:
            stft_out = np.abs(librosa.stft(audio_data))
            chr_f = np.mean(librosa.feature.chroma_stft(S=stft_out, sr=sr).T, axis=0)
            mfc_f = np.mean(librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=40).T, axis=0)
            mel_f = np.mean(librosa.feature.melspectrogram(y=audio_data, sr=sr, n_mels=128).T, axis=0)
            return np.hstack([chr_f, mfc_f, mel_f]).reshape(1, -1)
        except Exception as e:
            print(f"[WARN] Librosa extraction failed, falling back: {e}")
            
    return extract_features_combined(audio_data, sr).reshape(1, -1)

def predict_audio_emotion(audio_data, sr):
    m = get_model()
    if not m: return "neutral", 0.0, [0]*len(emotions)
    
    rms = np.sqrt(np.mean(audio_data**2))
    if rms < 0.001: 
        probs = [0.0] * len(emotions)
        probs[emotions.index('neutral')] = 1.0
        return "neutral", 1.0, probs

    try:
        features = extract_audio_features(audio_data, sr)
        zcr = np.mean(librosa.feature.zero_crossing_rate(audio_data)) if librosa else 0.0
        
        if hasattr(m, 'classes_'): # SKLearn / joblib
            probs = m.predict_proba(features)[0]
            pred = m.predict(features)[0]
            
            # BIAS CORRECTION: Laughter (Happy) often misclassified as Fear
            # emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
            # fear = idx 2, happy = idx 3
            if pred == 'fear' and probs[2] > 0.7:
                if (len(probs) > 3 and probs[3] > 0.01) or zcr > 0.06:
                    print(f"[INFO] Laughter bias correction: Fear -> Happy (ZCR: {zcr:.4f})")
                    pred = 'happy'
                    # Enhance happy confidence for the UI
                    temp_probs = list(probs)
                    temp_probs[3] = max(temp_probs[3], 0.6)
                    probs = np.array(temp_probs)
        else: # Numpy fallback
            probs = m.predict_proba(features)[0]
            pred = m.predict(features)
            
        return str(pred), float(np.max(probs)), probs
    except Exception as e:
        print(f"[ERROR] Audio prediction failed: {e}")
        return "neutral", 0.0, [0]*len(emotions)

def warmup():
    print("[INFO] Warmup (Static Mode)")
    get_model()
