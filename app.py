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
from audio_features_numpy import extract_features_combined

# Standard Emotion Set
EMOTIONS_ORDER = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
PLEASANT_SURPRISE = 'Pleasant Surprise'

EMOJI_MAP = {
    'angry': '😠',
    'disgust': '🤢',
    'fear': '😨',
    'happy': '😊',
    'neutral': '😐',
    'ps': '🤩',
    PLEASANT_SURPRISE: '🤩',
    'surprise': '😲',
    'sad': '😢'
}
LABEL_MAP = {
    'ps': PLEASANT_SURPRISE,
    PLEASANT_SURPRISE: PLEASANT_SURPRISE,
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
    MAX_CONTENT_LENGTH=4 * 1024 * 1024, # 4MB (Vercel limit)
    SQLALCHEMY_DATABASE_URI=f'sqlite:///{db_path}',
    SQLALCHEMY_TRACK_MODIFICATIONS=False
)
db.init_app(app)

@app.route('/favicon.ico')
def favicon():
    return send_from_directory(os.path.join(app.root_path, 'static', 'images'),
                               'logo1.png', mimetype='image/png')

@app.route('/')
def home(): 
    return render_template('index.html', title="Home")

@app.route('/about')
def about(): 
    return render_template('about.html', title="About")

@app.route('/prediction_page')
def prediction_page(): 
    return render_template('prediction.html', title="Predict")

@app.route('/analyze')
def analyze():
    preds = []
    try:
        preds = PredictionResult.query.order_by(PredictionResult.timestamp.desc()).limit(100).all()
    except Exception as e:
        print(f"[ERROR] History query failed: {e}")
        try:
            with app.app_context():
                db.create_all()
        except Exception as e2:
            print(f"[ERROR] Database creation failed: {e2}")
    return render_template('history.html', predictions=preds, emoji_map=EMOJI_MAP, title="History", db_status="Online")

@app.route('/clear_history', methods=['POST'])
def clear_history():
    try:
        num_deleted = PredictionResult.query.delete()
        db.session.commit()
        print(f"[INFO] Cleared {num_deleted} records.")
    except Exception as e:
        db.session.rollback()
        print(f"[ERROR] Clear history failed: {e}")
    return redirect('/analyze')

def _call_ffmpeg(audio_path, sr):
    """Helper for FFmpeg process management."""
    try:
        try:
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg_exe = "ffmpeg"

        cmd = [
            ffmpeg_exe, '-y', '-i', audio_path,
            '-t', '7',
            '-f', 'f32le', '-acodec', 'pcm_f32le', '-ar', str(sr), '-ac', '1', '-'
        ]
        process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        out, _ = process.communicate()
        return out
    except Exception:
        return None

def _get_audio_results(audio_path, sr):
    """Extract and predict from audio path."""
    out = _call_ffmpeg(audio_path, sr)
    if not out:
        return "neutral", 0.0, [0.0]*len(EMOTIONS_ORDER), []
    
    X = np.frombuffer(out, dtype=np.float32)
    if len(X) < 500:
        return "neutral", 0.1, [0.0]*len(EMOTIONS_ORDER), []
    
    rms = np.sqrt(np.mean(X**2))
    res_audio = predict_audio_emotion(X, sr)
    # Ensure 4-tuple return
    if len(res_audio) < 4:
        return res_audio[0], res_audio[1], res_audio[2], []
    return res_audio + (rms,)

def _fuse_emotions(audio_data, vis_data):
    """Adaptive Cognitive Fusion Engine with Signal-to-Label Mapping."""
    a_emo, a_conf, a_probs, _, rms = audio_data
    v_emo, v_conf, v_stats = vis_data

    # 1. Handle Silence/Ambient Background
    if rms < 0.0012:
        if v_emo != 'N/A' and v_conf > 0.05:
            return v_emo, v_conf, "Visual Dominant (Acoustic Noise Floor)"
        return "neutral", 0.9, "Steady-State Ambient Signal"
    
    # Technical Identity for the user (MFCC/MLE Proof)
    mle_profile = "v1.4_Neural_MLP"
    tech_note = f"Identity: {mle_profile} | Signal Energy: {rms:.4f} | Acc: {a_conf*100:.1f}%"

    # 2. Logic Gates (Fixing Bias Shifts)
    # Happy is often confused with Angry or Fear if pitch is high. 
    # Use Visual Smile marker as the ultimate anchor.
    if v_emo == 'happy' and v_conf > 0.1:
        return 'happy', max(a_conf, v_conf, 0.75), f"Visual Smile Anchoring | {tech_note}"

    # 3. High-Confidence Acoustic Overrides
    if a_conf > 0.90:
        return a_emo, a_conf, f"Primary Acoustic {a_emo.upper()} Signal Detected"

    # 4. Standard Fusion Logic
    if v_emo != 'N/A' and v_conf > a_conf + 0.15:
        return v_emo, v_conf, f"Visual Feature Priority Override | {tech_note}"

    return a_emo, a_conf, f"Integrated Cognitive Path | {tech_note}"

@app.route('/predict', methods=['POST'])
def predict():
    if 'audio_file' not in request.files: 
        return jsonify({'error': 'No file'}), 400
    file = request.files['audio_file']
    if not file or not allowed_file(file.filename): 
        return jsonify({'error': 'Invalid file'}), 400

    # Defaults
    res_vars = {
        'f_emo': 'neutral', 'f_conf': 0.0, 'note': 'Analysis Complete',
        'a_emo': 'Unknown', 'v_emo': 'N/A', 'v_stats': {},
        'a_segs': [], 'a_data': [], 'probs': [0.0]*len(EMOTIONS_ORDER),
        'tmp': None
    }

    try:
        try: warmup()
        except Exception as e_warm: 
            print(f"[DEBUG] Warmup skip/fail: {e_warm}")

        t_dir = "/tmp" if os.environ.get('VERCEL') else None
        ext = os.path.splitext(file.filename)[1]
        with tempfile.NamedTemporaryFile(suffix=ext, delete=False, dir=t_dir) as tmp:
            file.save(tmp.name)
            res_vars['tmp'] = tmp.name

        a_path = res_vars['tmp']
        v_conf = 0.0
        if is_video_file(file.filename):
            try:
                res_vars['v_emo'], v_conf, res_vars['v_stats'] = analyze_video_faces(res_vars['tmp'])
                a_path = extract_audio_from_video(res_vars['tmp'])
            except Exception: pass
        elif is_image_file(file.filename):
            try:
                res_vars['v_emo'], v_conf, res_vars['v_stats'] = analyze_image_emotion(res_vars['tmp'])
                a_path = None # No audio in static images
            except Exception: pass
        
        # Audio Analysis
        if a_path:
            a_res = _get_audio_results(a_path, 22050)
            res_vars['a_emo'], a_conf, res_vars['probs'] = a_res[0], a_res[1], a_res[2]
            res_vars['a_segs'] = a_res[3]
            
            # Fusion
            res_vars['f_emo'], res_vars['f_conf'], res_vars['note'] = _fuse_emotions(a_res, (res_vars['v_emo'], v_conf, res_vars['v_stats']))
        else:
            # Image-only logic
            res_vars['a_emo'] = "N/A"
            res_vars['f_emo'] = res_vars['v_emo']
            res_vars['f_conf'] = v_conf
            res_vars['note'] = "Static Signal Analysis (Image Only)"
            res_vars['probs'] = [0.0]*len(EMOTIONS_ORDER)
            if res_vars['v_emo'] in EMOTIONS_ORDER:
                res_vars['probs'][EMOTIONS_ORDER.index(res_vars['v_emo'])] = v_conf

        # Prepare Data
        for i, eid in enumerate(EMOTIONS_ORDER):
            p = float(res_vars['probs'][i]) if i < len(res_vars['probs']) else 0.0
            if res_vars['v_emo'] == eid: p = max(p, 0.5)
            res_vars['a_data'].append({
                'id': eid, 'name': LABEL_MAP.get(eid, eid).title(),
                'emoji': EMOJI_MAP.get(eid, '❓'), 'prob': round(p * 100, 1)
            })
        res_vars['a_data'].sort(key=lambda x: x['prob'], reverse=True)

        # DB
        try:
            db_res = PredictionResult(
                filename=secure_filename(file.filename), audio_emotion=str(res_vars['a_emo']),
                visual_emotion=str(res_vars['v_emo']), final_emotion=str(res_vars['f_emo']),
                confidence=float(res_vars['f_conf'])
            )
            db.session.add(db_res)
            db.session.commit()
        except Exception:
            db.session.rollback()

    except Exception as e:
        res_vars['note'] = f"Error: {e}"
    finally:
        for p in [res_vars['tmp'], (a_path if 'a_path' in locals() and a_path != res_vars['tmp'] else None)]:
            if p and os.path.exists(p):
                try: os.unlink(p)
                except Exception: pass

    return render_template('result.html', predicted_emotion=res_vars['f_emo'], 
                          confidence=round(res_vars['f_conf']*100,1),
                          visual_emotion=res_vars['v_emo'], audio_emotion=res_vars['a_emo'], 
                          note=res_vars['note'], all_emotions=res_vars['a_data'], 
                          vis_stats=res_vars['v_stats'], audio_segments=res_vars['a_segs'],
                           title="Analysis Result")

if __name__ == '__main__':
    with app.app_context():
        db.create_all()
    app.run(host='0.0.0.0', debug=True, port=10000)