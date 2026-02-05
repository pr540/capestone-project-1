import numpy as np
try:
    import librosa
    LIBROSA_AVAILABLE = True
except ImportError:
    LIBROSA_AVAILABLE = False

from audio_features_numpy import extract_features_combined

def compare():
    sr = 22050
    # Use a fixed seed for Reproducibility
    np.random.seed(42)
    y = np.random.uniform(-1, 1, sr)
    
    my_feats = extract_features_combined(y, sr)
    print("MY MFCC[0]:", my_feats[12])
    print("MY Mel[0]:", my_feats[52])
    
    if LIBROSA_AVAILABLE:
        # chroma_stft
        c = np.mean(librosa.feature.chroma_stft(y=y, sr=sr, n_fft=2048, hop_length=512).T, axis=0)
        # mfcc
        m_mels = librosa.feature.melspectrogram(y=y, sr=sr, n_fft=2048, hop_length=512, n_mels=128)
        m_db = librosa.power_to_db(m_mels, ref=1.0)
        mfcc = np.mean(librosa.feature.mfcc(S=m_db, n_mfcc=40).T, axis=0)
        # mel
        mel = np.mean(m_mels.T, axis=0)
        
        print("LIBROSA MFCC[0]:", mfcc[0])
        print("LIBROSA Mel[0]:", mel[0])

compare()
