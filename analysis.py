import os
import numpy as np
import subprocess
try:
    import cv2
except ImportError:
    cv2 = None

try:
    import imageio_ffmpeg
except ImportError:
    imageio_ffmpeg = None

try:
    import joblib
except ImportError:
    joblib = None

from mlp_numpy import NumpyMLP
from audio_features_numpy import extract_features_combined, numpy_zcr

detector = {}
model = None
# Matches PKL order exactly: ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

def get_detector():
    global detector
    if not detector and cv2:
        try:
            # Multi-path cascade search
            paths = []
            if hasattr(cv2, 'data') and hasattr(cv2.data, 'haarcascades'):
                paths.append(cv2.data.haarcascades)
            paths.extend(['/usr/share/opencv/haarcascades/', '/usr/local/share/opencv/haarcascades/'])
            
            base = ""
            for p in paths:
                if os.path.exists(os.path.join(p, 'haarcascade_frontalface_default.xml')):
                    base = p
                    break
            
            detector['face'] = cv2.CascadeClassifier(os.path.join(base, 'haarcascade_frontalface_default.xml'))
            detector['smile'] = cv2.CascadeClassifier(os.path.join(base, 'haarcascade_smile.xml'))
            detector['eye'] = cv2.CascadeClassifier(os.path.join(base, 'haarcascade_eye.xml'))
        except Exception as e:
            print(f"[WARN] Detector init failed: {e}")
            detector = {} # Empty but not None to stop retrying
    return detector

def get_model():
    global model
    if model is None:
        # Priority 1: Numpy fallback (Most compatible with Vercel)
        try:
            model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp_weights.npz')
            if os.path.exists(model_path):
                model = NumpyMLP(model_path)
                print("[INFO] Loaded NumpyMLP (Production Mode).")
                return model
        except Exception as e:
            print(f"[ERROR] Loading NumpyMLP failed: {e}")

        # Priority 2: Original PKL model
        if joblib:
            try:
                model_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'mlp.pkl')
                if os.path.exists(model_path):
                    model = joblib.load(model_path)
                    print("[INFO] Loaded PKL model.")
                    return model
            except Exception as e:
                print(f"[WARN] Loading PKL failed: {e}")
        
        print("[ERROR] No model weights found in any format!")
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
            
            # Strict smile filter
            if len(smiles) > 0 and len(smiles) < 3:
                # Only trust smile if eyes are also active (Duchenne marker)
                if len(eyes) > 0:
                     stats['happy'] += 0.5 
                else:
                     stats['happy'] += 0.2
            elif len(eyes) > 2:
                stats['ps'] += 1 
            elif len(eyes) < 1:
                stats['sad'] += 1
            else:
                stats['disgust'] += 0.5 # Default negative interpretation for no-smile
                stats['neutral'] += 0.5
    
    cap.release()
    if detected_faces == 0: return "N/A", 0.0, stats
    
    # Selection logic
    dominant = max(stats, key=stats.get)
    val = stats[dominant]
    
    total = sum(stats.values())
    confidence = val / total if total > 0 else 0.0
    return str(dominant), float(confidence), stats

def extract_audio_features(audio_data, sr):
    """Definitive feature extraction using safe NumPy implementation."""
    return extract_features_combined(audio_data, sr).reshape(1, -1)

def predict_audio_emotion(audio_data, sr):
    m = get_model()
    if not m: return "neutral", 0.0, [0]*len(emotions), []
    
    # Accuracy safety: If signal is extremely low energy, it's silence
    max_val = np.max(np.abs(audio_data))
    if max_val < 0.001: 
        probs = [0.0] * len(emotions)
        probs[emotions.index('neutral')] = 1.0
        return "neutral", 1.0, probs, []

    try:
        # Segmented Analysis: Split into 3s chunks to catch peak emotions/laughter
        chunk_size = sr * 3
        segments = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if len(chunk) < sr: continue # Too short
            segments.append(chunk)
        
        if not segments: segments = [audio_data]
        
        chunk_results = []
        for chunk in segments:
            features = extract_audio_features(chunk, sr)
            if hasattr(m, 'classes_'): 
                p = m.predict_proba(features)[0]
                label = str(m.predict(features)[0])
            else:
                p = m.predict_proba(features)[0]
                label = str(m.predict(features))
            
            # Feature Correction: Disgust/Sad often confused for Happy by the model
            # Happy = High Energy + High ZCR (Laughter). Disgust/Sad = Low ZCR.
            if label == 'happy':
                try:
                    zcr = numpy_zcr(chunk)
                    rms = np.sqrt(np.mean(chunk**2))
                    
                    # Aggressive filter: Real laughter usually has ZCR > 0.1
                    # If ZCR is lower, it's likely speech/grunt (Disgust) or crying (Sad)
                    if zcr < 0.09: 
                         if rms < 0.005: 
                             label = 'sad'
                             p = np.zeros_like(p)
                             p[6] = 0.9
                         else:
                             label = 'disgust'
                             p = np.zeros_like(p)
                             p[1] = 0.9
                except Exception: 
                    pass
                
            chunk_results.append((label, p))
            
        # Decision: Use the Most Confident Expressive Chunk
        # Standard approach: Pick the segment the AI is most certain about
        non_neutral = [r for r in chunk_results if r[0] != 'neutral']
        
        if non_neutral:
            # Pick the one with highest confidence
            final_r = max(non_neutral, key=lambda x: np.max(x[1]))
            return final_r[0], float(np.max(final_r[1])), final_r[1], chunk_results
        else:
            # Fallback to first chunk (likely neutral)
            return chunk_results[0][0], float(np.max(chunk_results[0][1])), chunk_results[0][1], chunk_results

    except Exception as e:
        print(f"[ERROR] Audio prediction failed: {e}")
        return "neutral", 0.0, [0]*len(emotions), []

def warmup():
    print("[INFO] Warmup (Static Mode)")
    get_model()
