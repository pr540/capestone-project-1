import os
import subprocess
import imageio_ffmpeg
import tempfile
import numpy as np
try:
    import librosa
except ImportError:
    librosa = None
from flask import Flask, render_template, request, jsonify, send_from_directory, redirect
from werkzeug.utils import secure_filename
from database import db, PredictionResult
from utils import allowed_file, is_video_file, extract_audio_from_video
from analysis import analyze_video_faces, predict_audio_emotion, warmup
from concurrent.futures import ThreadPoolExecutor
import time
import traceback
from audio_features_numpy import extract_features_combined

# Standard Emotion Set
EMOTIONS_ORDER = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
EMOJI_MAP = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'ps': '🤩',
    'Pleasant Surprise': '🤩',
    'surprise': '😲',
    'sad': '😢'
}
LABEL_MAP = {
    'ps': 'Pleasant Surprise',
    'Pleasant Surprise': 'Pleasant Surprise',
    'surprise': 'Surprise'
}


# App Setup
BASE_DIR = os.path.abspath(os.path.dirname(__file__))
app = Flask(__name__, static_folder=os.path.join(BASE_DIR, 'static'), static_url_path='/static')
executor = ThreadPoolExecutor(max_workers=2)

# Use /tmp for SQLite on Vercel since the root is read-only
if os.environ.get('VERCEL'):
    db_path = '/tmp/emotions.db'
else:
    db_path = os.path.join(BASE_DIR, 'instance', 'emotions.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

app.config.update(
    MAX_CONTENT_LENGTH=100 * 1024 * 1024, # 100MB limit (Vercel actual limit is lower)
    SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_path}',
    SQLALCHEMY_TRACK_MODIFICATIONS=False,
    TEMPLATES_AUTO_RELOAD=True
)
db.init_app(app)
with app.app_context(): db.create_all()
warmup()

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                               'logo1.png', mimetype='image/png')

@app.route('/')
def home(): return render_template('index.html', title="Home")

@app.route('/about')
def about(): return render_template('about.html', title="About")

@app.route('/prediction_page')
def prediction_page(): return render_template('prediction.html', title="Predict")

@app.route('/analyze')
def analyze():
    # Limit to 50 for 'fast' store loading
    preds = PredictionResult.query.order_by(PredictionResult.timestamp.desc()).limit(50).all()
    return render_template('history.html', predictions=preds, title="History")

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        PredictionResult.query.delete()
        db.session.commit()
    except Exception as e:
        db.session.rollback()
        print(f"Error clearing history: {e}")
    return redirect('/analyze')


@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['audio_file']
    if not file or not allowed_file(file.filename): return jsonify({'error': 'Invalid file'}), 400

    # Save temp and load
    ext = os.path.splitext(file.filename)[1]
    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp:
        file.save(tmp.name)
        tmp_path = tmp.name

    # Initialize variables for template
    final_emo, final_conf, note = "neutral", 0.0, "Analysis pending"
    audio_emo, vis_emo = "Unknown", "N/A"
    all_emotions_data = []

    try:
        start_time = time.time()
        vis_conf = 0.0
        audio_path = tmp_path
        
        # Parallel analysis for speed
        future_vis = None
        if is_video_file(file.filename):
            future_vis = executor.submit(analyze_video_faces, tmp_path)
            audio_path = extract_audio_from_video(tmp_path)
        
        # Prepare audio stream - skip first 0.5s of potential silence
        sr = 22050
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', audio_path,
            '-ss', '0.5', '-t', '15', # Skip 0.5s, Take 15s
            '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(sr), '-ac', '1', '-'
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = process.communicate()
        X = np.frombuffer(out, dtype=np.float32)
        
        if future_vis:
            vis_emo, vis_conf, vis_stats = future_vis.result()

        if len(X) < 1000:
             audio_emo, au_conf, note = "Silent/Short", 0.0, "Audio stream too short"
             final_emo, final_conf = (vis_emo if vis_emo != 'N/A' else "neutral"), vis_conf
        else:
            rms = np.sqrt(np.mean(X**2))
            audio_emo, au_conf, probs = predict_audio_emotion(X, sr)
            
            if rms < 0.01: # Consistent threshold with analysis.py
                if vis_emo and vis_emo != 'N/A' and vis_conf > 0.1:
                    final_emo, final_conf, note = vis_emo, vis_conf, "Visual analysis (Audio is silent)"
                else:
                    final_emo, final_conf, note = "neutral", 0.9, "Silence detected"
            else:
                # Weighted Fusion Logic
                if vis_emo and vis_emo != 'N/A' and vis_conf > 0.1:
                    # If audio is neutral but video is expressive, trust video
                    if audio_emo == 'neutral' and vis_emo != 'neutral':
                        final_emo, final_conf, note = vis_emo, vis_conf, "Expressive visual over neutral audio"
                    elif au_conf > 0.95: 
                        final_emo, final_conf, note = audio_emo, au_conf, f"Strong audio {audio_emo} detected"
                    elif vis_conf > au_conf + 0.3:
                        final_emo, final_conf, note = vis_emo, vis_conf, "Visual evidence dominant"
                    else:
                        final_emo, final_conf, note = audio_emo, au_conf, "Audio analysis prioritized"
                else:
                    # Pure Audio Logic
                    final_emo, final_conf, note = audio_emo, au_conf, "Audio-only analysis"

            # Unique Fingerprint using MFCCs
            from audio_features_numpy import extract_features_combined
            feats = extract_features_combined(X, sr)
            # Use small subset of features for fingerprint
            raw_hint = ", ".join([f"{v:.1f}" for v in feats[12:17]]) 

            # Prepare all emotions for breakdown
            probs_list = list(probs) if 'probs' in locals() and probs is not None else []
            
            for i, emo_id in enumerate(EMOTIONS_ORDER):
                prob = 0.0
                if i < len(probs_list):
                    prob = float(probs_list[i])
                elif 'vis_emo' in locals() and vis_emo == emo_id:
                    # If audio is silent, but visual detected this emotion
                    prob = 1.0 
                
                all_emotions_data.append({
                    'id': emo_id, 
                    'name': LABEL_MAP.get(emo_id, emo_id).capitalize(),
                    'emoji': EMOJI_MAP.get(emo_id, '❓'), 
                    'prob': round(prob * 100, 1)
                })
            all_emotions_data = sorted(all_emotions_data, key=lambda x: x['prob'], reverse=True)

        # DB Storage with explicit validation
        try:
            # Ensure emotions are strings and not numpy objects
            db_audio = str(audio_emo) if audio_emo else "Unknown"
            db_visual = str(vis_emo) if vis_emo else "N/A"
            db_final = str(final_emo) if final_emo else "neutral"
            
            res = PredictionResult(
                filename=secure_filename(file.filename), 
                audio_emotion=db_audio,
                visual_emotion=db_visual, 
                final_emotion=db_final, 
                confidence=float(final_conf)
            )
            db.session.add(res)
            db.session.commit()
            print(f"[INFO] Analysis completed and stored in DB. ID: {res.id}")
        except Exception as e:
            print(f"[ERROR] DB Save failed: {e}")
            db.session.rollback()
            # Still continue to return result even if DB fails

    except Exception as e:
        print(f"[ERROR] Analysis crash: {e}")
        traceback.print_exc()
        final_emo, final_conf, note = "Error", 0.0, f"System failure: {str(e)}"
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        if 'audio_path' in locals() and audio_path != tmp_path and os.path.exists(audio_path): 
            try: os.unlink(audio_path)
            except: pass

        # Prepare feat_hint for display
        feat_hint = locals().get('raw_hint', "")
        if not feat_hint and 'X' in locals() and len(X) > 0:
            feat_hint = ", ".join([f"{v:.2f}" for v in X[:5]]) 
            
        return render_template('result.html', predicted_emotion=final_emo, confidence=round(final_conf*100,1),
                             visual_emotion=vis_emo, audio_emotion=audio_emo, note=note,
                             all_emotions=all_emotions_data, vis_stats=locals().get('vis_stats', {}),
                             feat_hint=feat_hint)

if __name__ == '__main__':
    try:
        import socket
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        local_ip = s.getsockname()[0]
        s.close()
        print(f" * Running on http://{local_ip}:50005 (Press CTRL+C to quit)")
    except Exception:
        print(" * Could not determine local IP")
    
    app.run(host='0.0.0.0', debug=False, port=50005)