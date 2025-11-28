
import librosa
import numpy as np
import pandas as pd
import joblib

import pickle

# Open the PKL file in binary mode
with open('mlp.pkl', 'rb') as f:
    # Load the object from the file
    model = pickle.load(f)
# Define the emotions
emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']

# Load the audio file and extract features
audio_file = input('Enter the path to the audio file: ')
X, sample_rate = librosa.load(audio_file, res_type='kaiser_fast')
result = np.array([])

stft = np.abs(librosa.stft(X))
chromas = np.mean(librosa.feature.chroma_stft
                  (S=stft, sr=sample_rate).T, axis=0)
result = np.hstack((result, chromas))

mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=40).T, axis=0)
result = np.hstack((result, mfccs))

mels = np.mean(librosa.feature.melspectrogram(y=X, sr=sample_rate, n_mels=128).T, axis=0)
result = np.hstack((result, mels))

# Make a prediction using the trained model
X_test = result.reshape(1, -1)
prediction = model.predict(X_test)

# Convert predicted emotion string to its index in the emotions list
predicted_index = emotions.index(prediction)

# Print the predicted emotion
print(f'The predicted emotion is: {emotions[predicted_index]}')