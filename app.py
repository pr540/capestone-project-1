from flask import Flask, render_template, request, redirect, jsonify
import pyaudio
import numpy as np
import librosa
import pickle
import os
import tempfile
from moviepy.editor import VideoFileClip
from werkzeug.utils import secure_filename

app = Flask(__name__)

# Allowed file extensions
ALLOWED_AUDIO_EXTENSIONS = {'wav', 'mp3', 'ogg', 'flac', 'm4a'}
ALLOWED_VIDEO_EXTENSIONS = {'mp4', 'avi', 'mov', 'mkv', 'webm'}
ALLOWED_EXTENSIONS = ALLOWED_AUDIO_EXTENSIONS | ALLOWED_VIDEO_EXTENSIONS

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
    """Extract audio from video file and return path to temporary audio file"""
    try:
        # Create a temporary file for the extracted audio
        temp_audio = tempfile.NamedTemporaryFile(suffix='.wav', delete=False)
        temp_audio_path = temp_audio.name
        temp_audio.close()
        
        # Extract audio from video
        video = VideoFileClip(video_path)
        video.audio.write_audiofile(temp_audio_path, logger=None)
        video.close()
        
        return temp_audio_path
    except Exception as e:
        raise Exception(f"Error extracting audio from video: {str(e)}")


def load_audio_file(file_storage):
    """Load audio from FileStorage object, handling both audio and video files"""
    # Create a temporary file to save the uploaded file
    temp_input = tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file_storage.filename)[1])
    temp_input_path = temp_input.name
    temp_input.close()
    
    # Save the uploaded file
    file_storage.save(temp_input_path)
    
    temp_audio_path = None
    try:
        # Check if it's a video file
        if is_video_file(file_storage.filename):
            # Extract audio from video
            temp_audio_path = extract_audio_from_video(temp_input_path)
            audio_path_to_load = temp_audio_path
        else:
            audio_path_to_load = temp_input_path
        
        # Load the audio file with librosa
        X, sample_rate = librosa.load(audio_path_to_load, res_type='kaiser_fast')
        
        return X, sample_rate
    finally:
        # Clean up temporary files
        if os.path.exists(temp_input_path):
            os.unlink(temp_input_path)
        if temp_audio_path and os.path.exists(temp_audio_path):
            os.unlink(temp_audio_path)


@app.route('/')
def index():
    return render_template('index.html')


@ app.route('/real')
def real():
       return render_template('real.html')



# Define the home page route
@app.route('/')
def home():
    return render_template('HOME.html')

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
        
        # Load audio file (handles both audio and video files)
        X, sample_rate = load_audio_file(audio_file)
        
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
        prediction = model.predict(X_test)
        
        # Convert predicted emotion string to its index in the emotions list
        predicted_index = emotions.index(prediction[0])
        
        # Print the predicted emotion
        predicted_emotion = emotions[predicted_index]
        
        return render_template('result.html', predicted_emotion=predicted_emotion)
    
    except Exception as e:
        # Log the error for debugging
        print(f"Error in /predict: {str(e)}")
        return jsonify({'error': f'Error processing file: {str(e)}'}), 500



if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True, port=5900)