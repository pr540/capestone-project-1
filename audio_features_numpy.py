import numpy as np

def stft(y, n_fft=2048, hop_length=512):
    window = np.hanning(n_fft)
    frames = np.array([y[i:i+n_fft] for i in range(0, len(y)-n_fft, hop_length)])
    return np.fft.rfft(frames * window, axis=1)

def melspectrogram(stft_output, sr=22050, n_fft=2048, n_mels=128):
    power_spec = np.abs(stft_output)**2
    
    # Mel Filterbank
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hertz = 700.0 * (10.0**(mels / 2595.0) - 1.0)
    bins = np.floor((n_fft + 1) * hertz / sr).astype(int)
    
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        filters[i, bins[i]:bins[i+1]] = np.linspace(0, 1, bins[i+1] - bins[i])
        filters[i, bins[i+1]:bins[i+2]] = np.linspace(1, 0, bins[i+2] - bins[i+1])
    
    mel_spec = power_spec @ filters.T
    return mel_spec

def mfcc(mel_spec, n_mfcc=40):
    log_mel_spec = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    n_mels = mel_spec.shape[1]
    dct_matrix = np.cos(np.pi * np.arange(n_mfcc)[:, None] * (np.arange(n_mels) + 0.5) / n_mels)
    return log_mel_spec @ dct_matrix.T

def chroma_stft(stft_output, sr=22050, n_chroma=12):
    # Simplified: map STFT bins to 12 chroma bins
    freqs = np.linspace(0, sr/2, stft_output.shape[1])
    chroma = np.zeros((stft_output.shape[0], 12))
    mag = np.abs(stft_output)
    for i, f in enumerate(freqs):
        if f > 0:
            semitone = 12 * np.log2(f / 440.0)
            bin_idx = int(np.round(semitone)) % 12
            chroma[:, bin_idx] += mag[:, i]
    return chroma

def extract_features_combined(y, sr):
    if len(y) < 2048: # Min length for one frame
        y = np.pad(y, (0, 2048 - len(y)))
        
    s = stft(y)
    m = melspectrogram(s, sr=sr)
    
    chr_feat = np.mean(chroma_stft(s, sr=sr), axis=0) # 12
    mfcc_feat = np.mean(mfcc(m), axis=0) # 40
    mel_feat = np.mean(10.0 * np.log10(np.maximum(m, 1e-10)), axis=0) # 128
    
    return np.hstack([chr_feat, mfcc_feat, mel_feat])
