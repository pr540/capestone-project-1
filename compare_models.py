import os
import librosa
import numpy as np
import joblib
from audio_features_numpy import extract_features_combined as extract_numpy
from mlp_numpy import NumpyMLP

def extract_librosa(y, sr):
    stft = np.abs(librosa.stft(y))
    chromas = np.mean(librosa.feature.chroma_stft(S=stft, sr=sr).T, axis=0)
    mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=40).T, axis=0)
    mels = np.mean(librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128).T, axis=0)
    return np.hstack([chromas, mfccs, mels])

# White noise
sr = 22050
duration = 2
y = np.random.uniform(-0.1, 0.1, sr * duration).astype(np.float32)

print("--- Feature Extraction Comparison ---")
feat_lib = extract_librosa(y, sr)
feat_np = extract_numpy(y, sr)

print(f"Librosa feat shape: {feat_lib.shape}")
print(f"Numpy feat shape:   {feat_np.shape}")
print(f"Feature difference (mean abs): {np.mean(np.abs(feat_lib - feat_np))}")
print(f"First 5 librosa: {feat_lib[:5]}")
print(f"First 5 numpy:   {feat_np[:5]}")

print("\n--- Model Comparison ---")
model_pkl = joblib.load('mlp.pkl')
model_np = NumpyMLP('mlp_weights.npz')

# Test with Librosa features in PKL model
prob_pkl = model_pkl.predict_proba(feat_lib.reshape(1, -1))[0]
pred_pkl = model_pkl.predict(feat_lib.reshape(1, -1))[0]
print(f"PKL Predict (Librosa feats): {pred_pkl}")
print(f"PKL Probs: {prob_pkl}")

# Test with Numpy features in Numpy model
prob_np = model_np.predict_proba(feat_np.reshape(1, -1))[0]
pred_np = model_np.predict(feat_np.reshape(1, -1))
print(f"Numpy Predict (Numpy feats): {pred_np}")
print(f"Numpy Probs: {prob_np}")

# Test if PKL model is biased towards Fear
y_silence = np.zeros(sr * duration, dtype=np.float32)
feat_silence = extract_librosa(y_silence, sr)
pred_silence = model_pkl.predict(feat_silence.reshape(1, -1))[0]
print(f"\nSilence Prediction (Librosa + PKL): {pred_silence}")
print(f"Silence Probs: {model_pkl.predict_proba(feat_silence.reshape(1, -1))[0]}")
