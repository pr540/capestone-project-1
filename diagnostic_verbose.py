import numpy as np
import audio_features_numpy
from mlp_numpy import NumpyMLP

def diagnostic():
    nm = NumpyMLP('mlp_weights.npz')
    
    # helper
    def test_signal(name, y):
        feat = audio_features_numpy.extract_features_combined(y, 22050).reshape(1, -1)
        probs = nm.predict_proba(feat)
        pred = nm.predict(feat)
        print(f"--- {name} ---")
        print(f"Prediction: {pred}")
        print(f"Max Prob: {np.max(probs):.4f}")
        print(f"Probs: {probs[0]}")
        print(f"Feature mean: {np.mean(feat):.4f}, std: {np.std(feat):.4f}")
        print(f"Feature min/max: {np.min(feat):.4f} / {np.max(feat):.4f}")

    test_signal("Zero", np.zeros(22050 * 2))
    test_signal("Noise", np.random.uniform(-1, 1, 22050 * 2))
    
    # Sine wave
    t = np.linspace(0, 2, 22050 * 2)
    test_signal("Sine 440Hz", np.sin(2 * np.pi * 440 * t))

diagnostic()
