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

def analyze_image_emotion(image_path):
    """Analyze a single static image for emotional features."""
    dets = get_detector()
    if not dets or not cv2: return "N/A", 0.0, {}
    
    img = cv2.imread(image_path)
    if img is None: return "N/A", 0.0, {}
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = dets['face'].detectMultiScale(gray, 1.1, 5)
    stats = {e: 0 for e in emotions}
    
    if len(faces) == 0: return "N/A", 0.0, stats
    
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        smiles = dets['smile'].detectMultiScale(roi_gray, 1.2, 3)
        eyes = dets['eye'].detectMultiScale(roi_gray, 1.1, 3)
        
        if len(smiles) > 0:
            stats['happy'] += 1
        elif len(eyes) > 2:
            stats['ps'] += 1
        elif len(eyes) < 1:
            stats['sad'] += 1
        else:
            stats['neutral'] += 1
            
    dominant = max(stats, key=stats.get)
    total = sum(stats.values())
    conf = stats[dominant] / total if total > 0 else 0.0
    return str(dominant), float(conf), stats

def analyze_video_faces(video_path):
    """Analyze video frames for heuristic expression markers."""
    dets = get_detector()
    if not dets or not cv2: return "N/A", 0.0, {}
    
    cap = cv2.VideoCapture(video_path)
    stats = {e: 0 for e in emotions}
    total_frames = 0
    detected_faces = 0
    
    # Analyze up to 100 frames with variable skipping for high temporal resolution
    while cap.isOpened() and total_frames < 100:
        ret, frame = cap.read()
        if not ret: break
        total_frames += 1
        if total_frames % 4 != 0: continue 
        
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = dets['face'].detectMultiScale(gray, 1.3, 5)
        
        for (x, y, w, h) in faces:
            detected_faces += 1
            roi_gray = gray[y:y+h, x:x+w]
            
            smiles = dets['smile'].detectMultiScale(roi_gray, 1.2, 4) 
            eyes = dets['eye'].detectMultiScale(roi_gray, 1.1, 3)
            
            if len(smiles) > 0:
                stats['happy'] += 1.2 # Weight happiness higher in fusion
            elif len(eyes) > 2:
                stats['ps'] += 1 
            elif len(eyes) < 1:
                stats['sad'] += 1
            else:
                stats['neutral'] += 0.5
    
    cap.release()
    if detected_faces == 0: return "N/A", 0.0, stats
    dominant = max(stats, key=stats.get)
    total = sum(stats.values())
    return str(dominant), float(stats[dominant]/total if total > 0 else 0.0), stats

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
        chunk_size = int(sr * 1.5) # 1.5s segments for double detail
        segments = []
        for i in range(0, len(audio_data), chunk_size):
            chunk = audio_data[i:i+chunk_size]
            if len(chunk) < int(sr * 0.5): continue # Keep segments > 0.5s
            segments.append(chunk)
        
        if not segments: segments = [audio_data]
        
        chunk_results = []
        for chunk in segments:
            features = extract_audio_features(chunk, sr)
            p = m.predict_proba(features)[0]
            # Handle both sklearn (array) and NumpyMLP (string) predictions
            raw_pred = m.predict(features)
            if hasattr(raw_pred, '__getitem__') and not isinstance(raw_pred, str):
                label = str(raw_pred[0])
            else:
                label = str(raw_pred)
            
            # Result Aggregation
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
