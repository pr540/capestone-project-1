import os
import subprocess
import imageio_ffmpeg
import tempfile
import numpy as np
try:
    import librosa
except ImportError:
    librosa = None
from flask import Flask, render_template, request, jsonify, send_from_directory
from werkzeug.utils import secure_filename
from database import db, PredictionResult
from utils import allowed_file, is_video_file, extract_audio_from_video
from analysis import analyze_video_faces, predict_audio_emotion, warmup

from concurrent.futures import ThreadPoolExecutor
import time

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

from flask import redirect

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

    try:
        start_time = time.time()
        vis_emo, vis_conf = None, 0.0
        audio_path = tmp_path
        
        # Parallel analysis for speed
        future_vis = None
        if is_video_file(file.filename):
            future_vis = executor.submit(analyze_video_faces, tmp_path)
            audio_path = extract_audio_from_video(tmp_path)
        
        # Load audio using ffmpeg - limit to 10s for 'fast' processing
        sr = 22050
        cmd = [
            imageio_ffmpeg.get_ffmpeg_exe(), '-y', '-i', audio_path,
            '-ss', '0', '-t', '10', # Fast extraction: first 10 seconds
            '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(sr), '-ac', '1', '-'
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = process.communicate()
        X = np.frombuffer(out, dtype=np.float32)
        
        if future_vis:
            vis_emo, vis_conf = future_vis.result()

        if len(X) == 0:
             audio_emo, au_conf, note = "Silent/Error", 0.0, "Could not extract audio"
             final_emo, final_conf = vis_emo or "N/A", vis_conf
        else:
            rms = np.sqrt(np.mean(X**2))
            audio_emo, au_conf, probs = predict_audio_emotion(X, sr)
            
            # Logic to prevent 'Pleasant Surprise' bias on quiet/ambient audio
            if rms < 0.01:
                if vis_emo:
                    final_emo, final_conf, note = vis_emo, vis_conf, "Based on face (audio too quiet)"
                else:
                    final_emo, final_conf, note = "neutral", 0.9, "Silent audio detected"
            else:
                # Weighted Fusion / Priority
                if vis_emo:
                    # Trap Emotions: 'disgust' and 'ps' are common model biases on noisy data
                    trap_emotions = ["disgust", "ps", "Pleasant Surprise"]
                    
                    if audio_emo in trap_emotions and au_conf > 0.8:
                        if vis_emo != "neutral" and vis_conf > 0.3:
                            final_emo, final_conf, note = vis_emo, vis_conf, f"Visual ({vis_emo}) overrides biased audio ({audio_emo})"
                        elif vis_emo == "neutral" and vis_conf > 0.5:
                            final_emo, final_conf, note = "neutral", (vis_conf + au_conf)/2, "Visual neutral overrides audio bias"
                        else:
                            final_emo, final_conf, note = audio_emo, au_conf, "Audio prioritized (weak visual signal)"
                    elif vis_conf > au_conf + 0.15:
                        final_emo, final_conf, note = vis_emo, vis_conf, "Based on higher visual confidence"
                    else:
                        final_emo, final_conf, note = audio_emo, au_conf, "Based on audio evidence"
                else:
                    final_emo, final_conf, note = audio_emo, au_conf, "Audio-only analysis"

        # Fast DB Storage
        res = PredictionResult(filename=secure_filename(file.filename), audio_emotion=audio_emo,
                               visual_emotion=vis_emo or "N/A", final_emotion=final_emo, confidence=final_conf)
        db.session.add(res)
        db.session.commit()
        
        print(f"[INFO] Analysis completed in {time.time() - start_time:.2f}s")

        return render_template('result.html', predicted_emotion=final_emo, confidence=round(final_conf*100,1),
                             visual_emotion=vis_emo, audio_emotion=audio_emo, note=note)
    finally:
        if os.path.exists(tmp_path): os.unlink(tmp_path)
        if 'audio_path' in locals() and audio_path != tmp_path and os.path.exists(audio_path): os.unlink(audio_path)

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