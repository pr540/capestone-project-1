import numpy as np
from analysis import predict_audio_emotion, analyze_video_faces

# Mock audio data (silence)
sr = 22050
X = np.random.uniform(-0.1, 0.1, sr * 2).astype(np.float32)

print("Testing audio prediction...")
audio_emo, au_conf, probs = predict_audio_emotion(X, sr)
print(f"Audio Emotion: {audio_emo} (type: {type(audio_emo)})")
print(f"Confidence: {au_conf}")
print(f"Probs: {probs}")

emotions_order = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'ps', 'sad']
for i, p in enumerate(probs):
    print(f"  {emotions_order[i]}: {p}")
