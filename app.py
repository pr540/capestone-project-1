import os
import subprocess
import imageio_ffmpeg
import tempfile
import numpy as np
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

# Use /tmp for SQLite on Vercel
if os.environ.get('VERCEL'):
    db_path = '/tmp/emotions.db'
else:
    db_path = os.path.join(BASE_DIR, 'instance', 'emotions.db')
    os.makedirs(os.path.dirname(db_path), exist_ok=True)

app.config.update(
    MAX_CONTENT_LENGTH=50 * 1024 * 1024, # 50MB
    SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_path}',
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)
db.init_app(app)

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
    preds = []
    try:
        preds = PredictionResult.query.order_by(PredictionResult.timestamp.desc()).limit(100).all()
        print(f"[INFO] Analysis history loaded: {len(preds)} rows.")
    except Exception as e:
        print(f"[ERROR] History query failed: {e}")
        try:
            with app.app_context(): db.create_all()
            preds = []
        except: pass
    return render_template('history.html', predictions=preds, title="History")

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        num_deleted = PredictionResult.query.delete()
        db.session.commit()
        print(f"[INFO] Cleared {num_deleted} records from history.")
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Clear history failed: {e}")
    return redirect('/analyze')

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files: return jsonify({'error': 'No file'}), 400
    file = request.files['audio_file']
    if not file or not allowed_file(file.filename): return jsonify({'error': 'Invalid file'}), 400

    # Safe defaults
    final_emo, final_conf, note = "neutral", 0.0, "Analysis Complete"
    audio_emo, vis_emo = "Unknown", "N/A"
    all_emotions_data = []
    audio_segments = []
    vis_stats = {}
    probs = [0.0] * len(EMOTIONS_ORDER)
    X = np.array([])
    sr = 22050
    raw_hint = "N/A"
    tmp_path = None

    try:
        # Load model lazily
        try: warmup()
        except: pass

        # Save to /tmp
        temp_dir = "/tmp" if os.environ.get('VERCEL') else None
        ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=temp_dir) as tmp:
            file.save(tmp.name)
            tmp_path = tmp.name

        audio_path = tmp_path
        
        # Process sequential for stability
        if is_video_file(file.filename):
            try:
                vis_emo, vis_conf, vis_stats = analyze_video_faces(tmp_path)
                audio_path = extract_audio_from_video(tmp_path)
            except Exception as e:
                print(f"[WARN] Video failed: {e}")
        
        # Audio extraction
        try:
            try:
                ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            except:
                ffmpeg_exe = "ffmpeg"

            cmd = [
                ffmpeg_exe, '-y', '-i', audio_path,
                '-t', '15',
                '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(sr), '-ac', '1', '-'
            ]
            process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            out, _ = process.communicate()
            if out:
                X = np.frombuffer(out, dtype=np.float32)
            
            if len(X) >= 500:
                rms = np.sqrt(np.mean(X**2))
                res_audio = predict_audio_emotion(X, sr)
                if len(res_audio) == 4:
                    audio_emo, au_conf, probs, audio_segments = res_audio
                else:
                    audio_emo, au_conf, probs = res_audio[:3]
                    audio_segments = []

                if rms < 0.002:
                    if vis_emo != 'N/A' and vis_conf > 0.05:
                        final_emo, final_conf, note = vis_emo, vis_conf, "Visual only (Silent audio)"
                    else:
                        final_emo, final_conf, note = "neutral", 0.9, "Silence detected"
                else:
                    # Logic: Trust Audio for "Hard" emotions (Disgust/Sad/Fear) unless video is extremely confident
                    # Video "Happy" is often a false positive (grimace)
                    if audio_emo in ['disgust', 'sad', 'fear', 'angry']:
                        final_emo, final_conf, note = audio_emo, au_conf, f"Priority Audio {audio_emo}"
                    elif vis_emo != 'N/A' and vis_conf > 0.05:
                        if vis_conf > au_conf + 0.1:
                            final_emo, final_conf, note = vis_emo, vis_conf, "Visual evidence dominant"
                        else:
                            final_emo, final_conf, note = audio_emo, au_conf, "Segmented audio prioritization"
                    else:
                        final_emo, final_conf, note = audio_emo, au_conf, "Deep Segmented AI Analysis"
            else:
                final_emo, final_conf, note = (vis_emo if vis_emo != 'N/A' else "neutral"), vis_conf, "Short/Silent clip"

        except Exception as e:
            note = f"Audio engine error: {str(e)}"

        # Prepare breakdown
        p_list = list(probs)
        for i, emo_id in enumerate(EMOTIONS_ORDER):
            p = float(p_list[i]) if i < len(p_list) else (0.0)
            if vis_emo == emo_id: p = max(p, 0.5) # Heuristic for visual presence
            
            all_emotions_data.append({
                'id': emo_id, 
                'name': LABEL_MAP.get(emo_id, emo_id).capitalize(),
                'emoji': EMOJI_MAP.get(emo_id, '❓'), 
                'prob': round(p * 100, 1)
            })
        all_emotions_data.sort(key=lambda x: x['prob'], reverse=True)

        # Store in DB
        try:
            res = PredictionResult(
                filename=secure_filename(file.filename), 
                audio_emotion=str(audio_emo), 
                visual_emotion=str(vis_emo), 
                final_emotion=str(final_emo), 
                confidence=float(final_conf)
            )
            db.session.add(res)
            db.session.commit()
            print(f"[INFO] Saved prediction {res.id} for {file.filename}")
        except Exception as dbe:
            print(f"[ERROR] DB Save Error: {dbe}")
            db.session.rollback()

    except Exception as e:
        note = f"Critical Failure: {str(e)}"
    finally:
        if tmp_path and os.path.exists(tmp_path): 
            try: os.unlink(tmp_path)
            except: pass
        if 'audio_path' in locals() and audio_path != tmp_path and audio_path and os.path.exists(audio_path):
            try: os.unlink(audio_path)
            except: pass

    # Last resort rendering safety
    try:
        return render_template('result.html', predicted_emotion=final_emo, confidence=round(final_conf*100,1),
                             visual_emotion=vis_emo, audio_emotion=audio_emo, note=note,
                             all_emotions=all_emotions_data, vis_stats=vis_stats,
                             feat_hint=raw_hint, audio_segments=audio_segments)
    except Exception as e:
        return f"Template Error: {str(e)}", 500

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', debug=True, port=10000)