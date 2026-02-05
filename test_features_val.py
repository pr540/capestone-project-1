import numpy as np
from audio_features_numpy import extract_features_combined

def test_features():
    sr = 22050
    # Create 1 second of random noise at full amplitude
    y = np.random.uniform(-1, 1, sr)
    feats = extract_features_combined(y, sr)
    print("MFCC[0] (Energy):", feats[12])
    print("Mel[0] (Low Freq):", feats[52])

test_features()
