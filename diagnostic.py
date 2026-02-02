import numpy as np
import audio_features_numpy
from mlp_numpy import NumpyMLP

def diagnostic():
    nm = NumpyMLP('mlp_weights.npz')
    
    # 1. Zero signal
    y_zero = np.zeros(22050 * 3)
    feat_zero = audio_features_numpy.extract_features_combined(y_zero, 22050).reshape(1, -1)
    pred_zero = nm.predict(feat_zero)
    probs_zero = nm.predict_proba(feat_zero)
    
    # 2. Random noise
    y_noise = np.random.uniform(-1, 1, 22050 * 3)
    feat_noise = audio_features_numpy.extract_features_combined(y_noise, 22050).reshape(1, -1)
    pred_noise = nm.predict(feat_noise)
    probs_noise = nm.predict_proba(feat_noise)
    
    # 3. Sine wave (high pitch)
    t = np.linspace(0, 3, 22050 * 3)
    y_sine = np.sin(2 * np.pi * 1000 * t)
    feat_sine = audio_features_numpy.extract_features_combined(y_sine, 22050).reshape(1, -1)
    pred_sine = nm.predict(feat_sine)
    probs_sine = nm.predict_proba(feat_sine)

    print(f"Zero Signal: {pred_zero} (probs max: {np.max(probs_zero):.4f})")
    print(f"Noise Signal: {pred_noise} (probs max: {np.max(probs_noise):.4f})")
    print(f"Sine Signal: {pred_sine} (probs max: {np.max(probs_sine):.4f})")
    
    # Check scaling
    print(f"Feat Zero Mean/Std: {np.mean(feat_zero):.4f} / {np.std(feat_zero):.4f}")
    print(f"Model Mean Range: {np.min(nm.mean):.4f} to {np.max(nm.mean):.4f}")

diagnostic()
