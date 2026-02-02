import numpy as np

def stft(y, n_fft=2048, hop_length=512):
    # Center padding (reflect)
    pad = n_fft // 2
    y = np.pad(y, pad, mode='reflect')
    window = np.hanning(n_fft)
    # Framing
    n_frames = 1 + (len(y) - n_fft) // hop_length
    frames = np.lib.stride_tricks.as_strided(y, shape=(n_frames, n_fft), 
                                           strides=(y.strides[0] * hop_length, y.strides[0]))
    return np.fft.rfft(frames * window, axis=1)

def melspectrogram(stft_output, sr=22050, n_fft=2048, n_mels=128):
    power_spec = np.abs(stft_output)**2
    
    # Mel Filterbank (HTK formula)
    fmin, fmax = 0.0, sr / 2.0
    mel_min = 2595.0 * np.log10(1.0 + fmin / 700.0)
    mel_max = 2595.0 * np.log10(1.0 + fmax / 700.0)
    mels = np.linspace(mel_min, mel_max, n_mels + 2)
    hertz = 700.0 * (10.0**(mels / 2595.0) - 1.0)
    fft_freqs = np.linspace(0, sr / 2, n_fft // 2 + 1)
    
    filters = np.zeros((n_mels, n_fft // 2 + 1))
    for i in range(n_mels):
        # Triangular filters
        lower = hertz[i]
        center = hertz[i+1]
        upper = hertz[i+2]
        
        # Linear interpolation
        filters[i] = np.maximum(0, np.minimum((fft_freqs - lower) / (center - lower), (upper - fft_freqs) / (upper - center)))
    
    # Area normalization (slaney)
    enorm = 2.0 / (hertz[2:n_mels+2] - hertz[:n_mels])
    filters *= enorm[:, None]
    
    mel_spec = power_spec @ filters.T
    return mel_spec

def mfcc(mel_spec, n_mfcc=40):
    # Log power
    log_mel = 10.0 * np.log10(np.maximum(mel_spec, 1e-10))
    # DCT-II with ortho normalization
    n_mels = mel_spec.shape[1]
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    dct_matrix = np.cos(np.pi * k * (n + 0.5) / n_mels)
    # Ortho normalization factors
    dct_matrix[0] *= np.sqrt(1.0 / n_mels)
    dct_matrix[1:] *= np.sqrt(2.0 / n_mels)
    
    return log_mel @ dct_matrix.T

def chroma_stft(stft_output, sr=22050, n_chroma=12):
    # Map frequencies to semitones
    freqs = np.linspace(0, sr / 2.0, stft_output.shape[1])
    # Avoid log(0)
    freqs[0] = 1e-10 
    semitones = 12.0 * np.log2(freqs / 440.0)
    bins = np.round(semitones).astype(int) % 12
    
    mag = np.abs(stft_output)
    chroma = np.zeros((stft_output.shape[0], 12))
    for b in range(12):
        chroma[:, b] = np.sum(mag[:, bins == b], axis=1)
    
    # Normalize
    norm = np.linalg.norm(chroma, axis=1, keepdims=True)
    return chroma / (norm + 1e-10)

def mfcc_from_db(db_mel, n_mfcc=40):
    # DCT-II with ortho normalization
    n_mels = db_mel.shape[1]
    n = np.arange(n_mels)
    k = np.arange(n_mfcc)[:, None]
    dct_matrix = np.cos(np.pi * k * (n + 0.5) / n_mels)
    # Ortho normalization factors
    dct_matrix[0] *= np.sqrt(1.0 / n_mels)
    dct_matrix[1:] *= np.sqrt(2.0 / n_mels)
    
    return db_mel @ dct_matrix.T

def extract_features_combined(y, sr):
    if len(y) < 2048:
        y = np.pad(y, (0, 2048 - len(y)))
    
    # Normalize signal to standard float range [-1, 1]
    if np.max(np.abs(y)) > 1e-8:
        y = y / np.max(np.abs(y))
    
    # STFT and Mel
    s = stft(y)
    m = melspectrogram(s, sr=sr)
    
    # Use dB conversion with dynamic reference (peak)
    # This prevents absolute volume from biasing the results
    ref = np.max(m)
    db_m = 10.0 * np.log10(np.maximum(m, 1e-10) / (ref + 1e-10))
    # Standard speech range is usually -80dB to 0dB relative to peak
    db_m = np.clip(db_m, -80, 0)
    
    chr_feat = np.mean(chroma_stft(s, sr=sr), axis=0)
    mfcc_feat = np.mean(mfcc_from_db(db_m), axis=0)
    mel_feat = np.mean(db_m, axis=0)
    
    return np.hstack([chr_feat, mfcc_feat, mel_feat])
