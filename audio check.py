from flask import Flask, render_template
import pyaudio
import numpy as np
import librosa
import pickle

app = Flask(__name__)

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
p = pyaudio.PyAudio()


# Define a function to record audio for a given duration
def record_audio(duration):
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
    audio = np.frombuffer(b''.join(frames), dtype=np.int16)
    return audio


@app.route('/')
def index():
    return render_template('real.html')


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


if __name__ == '__main__':
    app.run(debug=True,port=5100)