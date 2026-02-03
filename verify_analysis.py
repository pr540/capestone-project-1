import numpy as np
import analysis

sr = 22050
y = np.random.uniform(-0.05, 0.05, sr * 2).astype(np.float32)

print("Testing with white noise...")
emo, conf, probs = analysis.predict_audio_emotion(y, sr)
print(f"Result: {emo} | Conf: {conf}")

print("\nTesting switch to silence check...")
y_silent = np.random.uniform(-0.005, 0.005, sr * 2).astype(np.float32)
emo_s, conf_s, probs_s = analysis.predict_audio_emotion(y_silent, sr)
print(f"Result (Silent): {emo_s} | Conf: {conf_s}")
