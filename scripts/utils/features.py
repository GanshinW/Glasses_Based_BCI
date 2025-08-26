# utils/features.py

import numpy as np
from scipy.signal import welch, spectrogram, iirnotch, filtfilt, butter


def preprocess_signal(
    sig: np.ndarray,
    fs: float,
    notch_freq: float = 50.0,
    notch_bw: float = 2.0,
    hp_cutoff: float = 1.0,
    lp_cutoff: float = 40.0,
) -> np.ndarray:
    """
    Apply 50 Hz notch + 1–40 Hz bandpass + z-score to a 1D signal.
    """
    # 1) Notch filter at mains frequency
    Q = notch_freq / notch_bw
    b_notch, a_notch = iirnotch(notch_freq, Q, fs)
    sig = filtfilt(b_notch, a_notch, sig)

    # 2) Bandpass (butterworth)
    nyquist = fs / 2
    if hp_cutoff >= nyquist:
        hp_cutoff = nyquist * 0.99
    if lp_cutoff >= nyquist:
        lp_cutoff = nyquist * 0.99
        
    b_hp, a_hp = butter(4, hp_cutoff / nyquist, btype='high')
    sig = filtfilt(b_hp, a_hp, sig)
    b_lp, a_lp = butter(4, lp_cutoff / nyquist, btype='low')
    sig = filtfilt(b_lp, a_lp, sig)

    # 3) Z-score
    return (sig - sig.mean()) / (sig.std() + 1e-6)


def extract_time_domain(
    X: np.ndarray,
    fs: float,
    notch_freq: float = 50.0,
    notch_bw: float = 2.0,
    hp_cutoff: float = 1.0,
    lp_cutoff: float = 40.0,
) -> np.ndarray:
    """
    X: (n_trials, n_channels, n_samples)
    returns: same shape, filtered + z-scored waveforms
    """
    n_trials, n_ch, n_samples = X.shape
    X_time = np.empty_like(X, dtype=np.float32)
    
    print(f"  Preprocessing {n_trials} trials with {n_ch} channels...")
    
    for i in range(n_trials):
        if i % 20 == 0:  # Progress indicator
            print(f"    Processing trial {i+1}/{n_trials}")
        for ch in range(n_ch):
            X_time[i, ch, :] = preprocess_signal(
                X[i, ch, :], fs,
                notch_freq, notch_bw,
                hp_cutoff, lp_cutoff
            )
    return X_time


def extract_freq_domain(
    X_time: np.ndarray,
    fs: float,
    bands: list = None
) -> np.ndarray:
    """
    X_time: (n_trials, n_channels, n_samples)
    returns: (n_trials, n_channels, n_bands) matrix of log‐bandpower
    """
    if bands is None:
        bands = [(4, 8), (8, 13), (13, 30)]  # theta, alpha, beta
    
    n_trials, n_ch, n_samples = X_time.shape
    n_bands = len(bands)
    X_freq = np.zeros((n_trials, n_ch, n_bands), dtype=np.float32)

    print(f"  Extracting frequency features from {n_bands} bands: {bands}")
    
    # Determine appropriate nperseg for Welch's method
    nperseg = min(512, n_samples // 4)
    if nperseg < 64:
        nperseg = min(256, n_samples // 2)
    
    for i in range(n_trials):
        if i % 20 == 0:
            print(f"    Processing trial {i+1}/{n_trials}")
            
        for ch in range(n_ch):
            # Compute PSD via Welch
            try:
                f, Pxx = welch(
                    X_time[i, ch], fs=fs,
                    nperseg=nperseg,
                    noverlap=nperseg//2
                )
                
                for b, (low, high) in enumerate(bands):
                    mask = (f >= low) & (f <= high)
                    if np.any(mask):
                        # Average power across freq-bins, then log
                        band_pow = Pxx[mask].mean()
                        X_freq[i, ch, b] = np.log(band_pow + 1e-6)
                    else:
                        X_freq[i, ch, b] = -10  # Default value
                        
            except Exception as e:
                print(f"    Warning: Error processing trial {i}, channel {ch}: {e}")
                X_freq[i, ch, :] = -10  # Fill with default values
                
    return X_freq


def extract_timefreq_images(
    X_time: np.ndarray,
    fs: float,
    nperseg: int = None,
    noverlap: int = None,
    channel_method: str = 'average'
) -> np.ndarray:
    """
    X_time: (n_trials, n_channels, n_samples)
    returns: (n_trials, 1, n_freqs, n_times) spectrogram images
    
    channel_method: 'average', 'first', or 'pca'
    """
    n_trials, n_ch, n_samples = X_time.shape
    
    # Adaptive window size based on signal length
    if nperseg is None:
        nperseg = min(256, n_samples // 8)
        nperseg = max(64, nperseg)  # Minimum window size
    if noverlap is None:
        noverlap = nperseg // 2

    print(f"  Generating spectrograms (nperseg={nperseg}, noverlap={noverlap})...")
    
    # Choose channel combination method
    if channel_method == 'average':
        # Average across all channels
        X_combined = np.mean(X_time, axis=1)  # (n_trials, n_samples)
    elif channel_method == 'first':
        # Use first channel only
        X_combined = X_time[:, 0, :]  # (n_trials, n_samples)
    elif channel_method == 'pca':
        # Use PCA to get dominant component (more complex, optional)
        from sklearn.decomposition import PCA
        X_combined = np.zeros((n_trials, n_samples))
        for i in range(n_trials):
            pca = PCA(n_components=1)
            X_combined[i] = pca.fit_transform(X_time[i].T).flatten()
    else:
        raise ValueError(f"Unknown channel_method: {channel_method}")
    
    # Compute dimensions on first trial
    try:
        f, t, Sxx = spectrogram(
            X_combined[0], fs=fs,
            nperseg=nperseg, noverlap=noverlap
        )
        n_freqs, n_times = Sxx.shape
        
        # Limit frequency range to meaningful EEG bands (up to 40 Hz)
        freq_mask = f <= 40
        f_limited = f[freq_mask]
        n_freqs_limited = len(f_limited)
        
        X_img = np.zeros((n_trials, 1, n_freqs_limited, n_times), dtype=np.float32)
        
        for i in range(n_trials):
            if i % 20 == 0:
                print(f"    Processing spectrogram {i+1}/{n_trials}")
                
            try:
                _, _, Sxx = spectrogram(
                    X_combined[i], fs=fs,
                    nperseg=nperseg, noverlap=noverlap
                )
                
                # Limit to EEG frequency range
                Sxx_limited = Sxx[freq_mask, :]
                
                # Log transform and normalize
                img = np.log(Sxx_limited + 1e-6)
                img_mean = img.mean()
                img_std = img.std()
                if img_std > 1e-6:
                    img = (img - img_mean) / img_std
                else:
                    img = img - img_mean  # Just center if no variation
                
                X_img[i, 0, :, :] = img
                
            except Exception as e:
                print(f"    Warning: Error generating spectrogram for trial {i}: {e}")
                # Fill with zeros or noise pattern
                X_img[i, 0, :, :] = np.random.normal(0, 0.1, (n_freqs_limited, n_times))
        
        print(f"  Spectrogram shape: {X_img.shape}, frequency range: 0-{f_limited[-1]:.1f} Hz")
        
    except Exception as e:
        print(f"  Error generating spectrograms: {e}")
        print(f"  Falling back to simple time-frequency representation...")
        
        # Fallback: create simple time-frequency representation
        n_freqs_limited = 64  # Fixed size
        n_times = n_samples // 10  # Downsample time
        X_img = np.zeros((n_trials, 1, n_freqs_limited, n_times), dtype=np.float32)
        
        for i in range(n_trials):
            # Simple approach: reshape and downsample
            signal = X_combined[i]
            # Reshape to approximate time-frequency representation
            reshaped = signal[:n_times * (len(signal) // n_times)].reshape(-1, n_times)
            if reshaped.shape[0] >= n_freqs_limited:
                X_img[i, 0, :, :] = reshaped[:n_freqs_limited, :]
            else:
                # Pad if needed
                X_img[i, 0, :reshaped.shape[0], :] = reshaped
        
        print(f"  Fallback spectrogram shape: {X_img.shape}")

    return X_img


def extract_all_features(dataset, config):
    """
    Extract all multimodal features from a dataset
    
    Parameters:
    -----------
    dataset : dict
        Loaded dataset from load_dataset()
    config : object
        Configuration object with bands, sampling_rate, etc.
        
    Returns:
    --------
    features : dict
        Dictionary containing all extracted features
    """
    print("\n=== Feature Extraction ===")
    
    # Get parameters
    fs = dataset.get('sampling_rate', config.sampling_rate)
    bands = getattr(config, 'bands', [(4, 8), (8, 13), (13, 30)])
    
    features = {}
    
    # Extract features for each split
    for split in ['train', 'val', 'test']:
        X_key = f'X_{split}'
        if X_key not in dataset:
            continue
            
        print(f"\nProcessing {split} set...")
        X_raw = dataset[X_key]
        
        # Time-domain features (preprocessed signals)
        print(f"1. Time-domain features...")
        X_time = extract_time_domain(X_raw, fs)
        features[f'X_time_{split}'] = X_time
        
        # Frequency-domain features (band power)
        print(f"2. Frequency-domain features...")
        X_freq = extract_freq_domain(X_time, fs, bands)
        features[f'X_freq_{split}'] = X_freq
        
        # Time-frequency images (spectrograms)
        if getattr(config, 'use_img', True):
            print(f"3. Time-frequency images...")
            X_img = extract_timefreq_images(X_time, fs)
            features[f'X_img_{split}'] = X_img
        else:
            features[f'X_img_{split}'] = None
    
    # Add metadata
    features['sampling_rate'] = fs
    features['bands'] = bands
    features['use_img'] = getattr(config, 'use_img', True)
    
    print(f"\nFeature extraction completed!")
    print(f"  Time-domain: {features['X_time_train'].shape}")
    print(f"  Frequency-domain: {features['X_freq_train'].shape}")
    if features['X_img_train'] is not None:
        print(f"  Time-frequency images: {features['X_img_train'].shape}")
    
    return features