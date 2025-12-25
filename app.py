from flask import Flask, render_template, request, redirect, jsonify
import pyaudio
import numpy as np
import librosa
import pickle
import os
import tempfile
from moviepy.editor import VideoFileClip
import subprocess
import shutil
from werkzeug.utils import secure_filename
import cv2
try:
    from fer import FER
except ImportError:
    try:
        from fer.fer import FER
    except ImportError:
        print("[WARNING] Could not import FER from 'fer' or 'fer.fer'. Facial analysis will be disabled.")
        FER = None
from collections import Counter

app = Flask(__name__)

# Lazy loader for FER detector
detector = None

def get_detector():
    global detector
    if detector is None:
        if FER is None:
            print("[ERROR] FER class is not available. Facial analysis cannot be initialized.")
            return None
        try:
            print("[INFO] Loading FER detector...")
            # Use MTCNN for better face detection if possible, else default haarcascade
            detector = FER(mtcnn=False) 
            print("[INFO] FER detector loaded.")
        except Exception as e:
            print(f"[ERROR] Failed to load FER detector: {e}")
            detector = None
    return detector

def analyze_video_faces(video_path):
    """
    Analyzes facial expressions in a video file.
    Returns: (dominant_emotion, confidence)
    """
    try:
        det = get_detector()
        if det is None:
            return None, 0.0

        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print("Error opening video")
            return None, 0.0

        emotions_list = []
        frame_count = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Process every 10th frame to save time
        skip_frames = 10 
        
        print(f"[INFO] Analyzing video faces: {total_frames} frames")

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_count += 1
            if frame_count % skip_frames != 0:
                continue

            # Detect emotions in the frame
            # FER returns a list of dictionaries, e.g. [{'box':..., 'emotions': {'angry': 0.1, ...}}]
            result = det.detect_emotions(frame)
            
            if result:
                # Get the dominant emotion for the largest face
                # Usually the first one is the main one or we sort by box size
                # Let's simple take the first face detected
                emotions = result[0]['emotions']
                # Find max emotion
                max_emotion = max(emotions, key=emotions.get)
                max_score = emotions[max_emotion]
                
                # Filter low confidence
                if max_score > 0.3:
                    emotions_list.append(max_emotion)
        
        cap.release()
        
        if not emotions_list:
            return "neutral", 0.0 # Default if no faces found
            
        # Find most frequent emotion
        emotion_counts = Counter(emotions_list)
        dominant_emotion = emotion_counts.most_common(1)[0][0]
        
        # Calculate confidence as (count of dominant / total detections)
        confidence = emotion_counts[dominant_emotion] / len(emotions_list)
        
        print(f"[INFO] Visual Analysis: {dominant_emotion} ({confidence:.2f})")
        return dominant_emotion, confidence

    except Exception as e:
        print(f"[ERROR] Facial analysis failed: {e}")
        return None, 0.0

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

# Configure Flask app
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024 * 1024  # 100GB max file size

# Warmup librosa/numba to prevent timeout on first request
try:
    print("[INFO] Warming up librosa...")
    import librosa.core.audio
    # Create a synthetic audio signal instead of downloading a file
    # This is much faster and doesn't require internet connection
    sr = 22050  # Sample rate
    duration = 1  # 1 second
    y = np.random.randn(sr * duration).astype(np.float32) * 0.1  # Random noise
    
    # Trigger JIT compilation for common operations
    librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40)
    librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    librosa.feature.chroma_stft(y=y, sr=sr)
    
    print("[INFO] Librosa warmup complete.")
except Exception as e:
    print(f"[WARNING] Librosa warmup failed: {e}")

# Open the PKL file in binary mode
with open('mlp.pkl', 'rb') as f:
    # Load the object from the file
    model = pickle.load(f)

# Define the emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

# Set up Pyaudio
CHUNKSIZE = 1024  # fixed chunk size
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 44100
RECORD_SECONDS = 3  # set default recording time to 3 seconds



# Define a function to record audio for a given duration
def record_audio(duration):
    p = pyaudio.PyAudio()
    stream = p.open(format=FORMAT,
                    channels=CHANNELS,
                    rate=RATE,
                    input=True,
                    frames_per_buffer=CHUNKSIZE)
    frames = []
    for i in range(int(RATE / CHUNKSIZE * duration)):
        data = stream.read(CHUNKSIZE)
        frames.append(data)
    stream.stop_stream()
    stream.close()
    p.terminate()
    audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio


def allowed_file(filename):
    """Check if the file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def is_video_file(filename):
    """Check if the file is a video file"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_VIDEO_EXTENSIONS


def extract_audio_from_video(video_path):
    dest_path = None
    try:
        # Create a temporary file for the extracted audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        dest_path = temp_audio_path
        
        # Use ffmpeg directly for lightning-fast extraction
        # -y: overwrite output
        # -i: input file
        # -ss: start time
        # -t: duration (5 seconds limit)
        # -vp: no video
        # -acodec: copy (if possible) or pcm_s16le
        # ar: sample rate 
        
        # We use pcm_s16le and 22050Hz to ensure compatibility and speed
        command = [
            'ffmpeg', '-y', 
            '-i', video_path,
            '-ss', '00:00:00',
            '-t', '5',
            '-vn',
            '-acodec', 'pcm_s16le',
            '-ar', '22050',
            '-ac', '1',
            temp_audio_path
        ]
        
        # Suppress output unless error
        subprocess.check_call(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        return temp_audio_path
    except Exception as e:
        if dest_path and os.path.exists(dest_path):
            os.unlink(dest_path)
        # Fallback to moviepy if ffmpeg fails (though ffmpeg is installed in Docker)
        print(f"[WARNING] ffmpeg failed, falling back to moviepy: {e}")
        try:
             # Extract audio from video
            video = VideoFileClip(video_path)
            # Limit to 5 seconds to prevent timeout
            if video.duration > 5:
                video = video.subclip(0, 5)
                
            video.audio.write_audiofile(temp_audio_path, logger=None)
            video.close()
            return temp_audio_path
        except Exception as e2:
             raise Exception(f"Error extracting audio from video: {str(e2)}")


def load_audio_file(file_storage):
    """Load audio from FileStorage object, handling both audio and video files"""
    # Create a temporary file to save the uploaded file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_storage.filename)[1])
    temp_input_path = temp_input.name
    temp_input.close()
    
    # Save the uploaded file
    file_storage.save(temp_input_path)
    
    temp_audio_path = None
    visual_emotion = None
    visual_confidence = 0.0
    
    try:
        # Check if it's a video file
        if is_video_file(file_storage.filename):
            # 1. OPTIONAL: Run Visual Analysis first or in parallel
            try:
                print("[INFO] Starting visual analysis...")
                visual_emotion, visual_confidence = analyze_video_faces(temp_input_path)
            except Exception as ve:
                print(f"[WARNING] Visual analysis failed: {ve}")

            # 2. Extract audio from video
            temp_audio_path = extract_audio_from_video(temp_input_path)
            audio_path_to_load = temp_audio_path
        else:
            audio_path_to_load = temp_input_path
        
        # Load the audio file with librosa 
        # optimizations: limit duration to 5s and use faster resampling to prevent timeouts
        X, sample_rate = librosa.load(audio_path_to_load, res_type='kaiser_fast', duration=5)
        
        return X, sample_rate, visual_emotion, visual_confidence
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


@app.route('/real')
def real():
    return render_template('real.html')



# Define the home page route
@app.route('/')
def home():
    return render_template('index.html')

@app.route('/about')
def about():
    return render_template('about.html')

@app.route('/aboutus')
def aboutus():
    return render_template('about.html')

@app.route('/analyze')
def analyze():
    return render_template('index.html')

@app.route('/pred')
def pred():
    # Record audio for 3 seconds
    audio = record_audio(RECORD_SECONDS)
    # Convert audio data to floating-point format
    audio = audio.astype(np.float32)
    # Extract audio features
    stft = np.abs(librosa.stft(audio))
    chromas = np.mean(librosa.feature.chroma_stft(S=stft, sr=RATE).T, axis=0)
    mfccs = np.mean(librosa.feature.mfcc(y=audio, sr=RATE, n_mfcc=40).T, axis=0)
    mels = np.mean(librosa.feature.melspectrogram(y=audio, sr=RATE, n_mels=128).T, axis=0)
    features = np.hstack((chromas, mfccs, mels))
    # Make a prediction using the trained model
    X_test = features.reshape(1, -1)
    prediction = model.predict(X_test)[0]
    # Convert predicted emotion string to its index in the emotions list
    prediction_index = emotions.index(prediction)
    return str(prediction_index)


# Define the prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Check if file was uploaded
        if 'audio_file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        audio_file = request.files['audio_file']
        
        # Check if file is empty
        if audio_file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Check if file type is allowed
        if not allowed_file(audio_file.filename):
            return jsonify({
                'error': f'File type not supported. Allowed types: {", ".join(ALLOWED_EXTENSIONS)}'
            }), 400

        # Check file size (additional validation beyond Flask's MAX_CONTENT_LENGTH)
        audio_file.seek(0, 2)  # Seek to end of file
        file_size = audio_file.tell()  # Get current position (file size)
        audio_file.seek(0)  # Reset to beginning
        
        max_size = 100 * 1024 * 1024 * 1024  # 100GB
        if file_size > max_size:
            return jsonify({
                'error': f'File too large. Maximum size is {max_size / (1024 * 1024):.0f}MB. Your file is {file_size / (1024 * 1024):.2f}MB'
            }), 400

        # Debug logging
        print(f"[DEBUG] Received file: {audio_file.filename}")
        print(f"[DEBUG] File size: {file_size / (1024 * 1024):.2f}MB")
        print(f"[DEBUG] Is video file: {is_video_file(audio_file.filename)}")
        
        # Load audio file (handles both audio and video files)
        # Returns: Audio Array, Sample Rate, Visual Emotion (if video), Visual Confidence
        X, sample_rate, visual_emotion, visual_confidence = load_audio_file(audio_file)
        
        # Check for silence or very low audio
        rms = librosa.feature.rms(y=X)
        mean_rms = np.mean(rms)
        print(f"[DEBUG] Audio RMS Energy: {mean_rms}")
        
        # Threshold for silence (adjustable)
        SILENCE_THRESHOLD = 0.001
        
        if mean_rms < SILENCE_THRESHOLD:
            # If silence, but we have visual emotion, fallback to visual!
            if visual_emotion and visual_confidence > 0.4:
                print(f"[DEBUG] Silence detected, but visual emotion found: {visual_emotion}")
                return render_template('result.html', 
                                     predicted_emotion=visual_emotion, 
                                     confidence=round(visual_confidence * 100, 1), 
                                     note=f"Audio silent, result based on facial expression.",
                                     visual_emotion=visual_emotion,
                                     audio_emotion="Neutral (Silent)")
            
            print("[DEBUG] Detected silence or near-silence. Defaulting to 'neutral'.")
            return render_template('result.html', predicted_emotion='neutral', confidence=0.0, note="Audio level too low (Silence detected)")

        # Extract features from the audio
        result = np.array([])
        stft = np.abs(librosa.stft(X))
        chromas = np.mean(librosa.feature.chroma_stft(S=stft, sr=sample_rate).T, axis=0)
        result = np.hstack((result, chromas))
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
        result = np.hstack((result, mfccs))
        mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128).T, axis=0)
        result = np.hstack((result, mels))
        
        # Make a prediction using the trained model
        X_test = result.reshape(1, -1)
        
        # Get probabilities
        try:
            probabilities = model.predict_proba(X_test)[0]
            prediction = model.predict(X_test)
            audio_emotion = prediction[0]
            
            # Find max probability
            audio_confidence = np.max(probabilities)
            print(f"[DEBUG] Audio Predicted: {audio_emotion}, Confidence: {audio_confidence:.2f}")
            
            # DECISION LOGIC: Combine Audio and Visual
            final_emotion = audio_emotion
            final_confidence = audio_confidence
            note = None
            
            if visual_emotion:
                print(f"[DEBUG] Fusion - Audio: {audio_emotion} ({audio_confidence:.2f}), Visual: {visual_emotion} ({visual_confidence:.2f})")
                
                # If audio is "neutral" or "calm" but visual is strong "happy" (smile), prefer visual
                if audio_emotion in ['neutral', 'calm'] and visual_emotion in ['happy', 'happy', 'surprise'] and visual_confidence > 0.5:
                     final_emotion = visual_emotion
                     final_confidence = visual_confidence
                     note = f"Audio was neutral, but facial expression indicated {visual_emotion}."
                
                # If visual confidence is very high and audio is low, prefer visual
                elif visual_confidence > 0.8 and audio_confidence < 0.5:
                    final_emotion = visual_emotion
                    final_confidence = visual_confidence
                    note = "Result based primarily on facial expression due to higher confidence."
                    
            
        except AttributeError:
            # Fallback if model doesn't support predict_proba
            prediction = model.predict(X_test)
            final_emotion = prediction[0]
            final_confidence = 1.0 # Mock confidence
            audio_emotion = final_emotion
        
        return render_template('result.html', 
                             predicted_emotion=final_emotion, 
                             confidence=round(final_confidence * 100, 1),
                             visual_emotion=visual_emotion,
                             audio_emotion=audio_emotion,
                             note=note)
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error in /predict: {str(e)}")
        import traceback
        traceback.print_exc()
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500


# Error handler for file size limit
@app.errorhandler(413)
def request_entity_too_large(error):
    return jsonify({
        'error': 'File too large. Maximum upload size is 100GB.'
    }), 413


if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5000)