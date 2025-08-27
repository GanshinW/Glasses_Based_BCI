# utils/features.py
import numpy as np
from scipy.signal import welch, spectrogram

def bandpower_welch(X, fs, bands, nperseg=None, noverlap=None):
    """
    X: (N, C, T)
    bands: list of (low, high)
    return: BP (N, C, B) in log10 scale
    """
    N, C, T = X.shape
    B = len(bands)
    BP = np.zeros((N, C, B), dtype=np.float32)
    for n in range(N):
        for c in range(C):
            f, Pxx = welch(
                X[n, c],
                fs=fs,
                nperseg=nperseg,
                noverlap=noverlap,
                scaling='density'
            )
            for b, (lo, hi) in enumerate(bands):
                mask = (f >= lo) & (f < hi)
                if np.any(mask):
                    # integrate PSD within band
                    BP[n, c, b] = np.trapz(Pxx[mask], f[mask])
    # log transform for stability
    return np.log10(BP + 1e-8)

def spectrogram_images(
    X, fs,
    nperseg=None, noverlap=None,
    fmax=40.0,
    channel_method='average'
):
    """
    X: (N, C, T)
    return: imgs (N, 1, H, W), log + z-norm
    """
    N, C, T = X.shape
    imgs = []
    for n in range(N):
        if channel_method == 'first':
            x = X[n, 0]
        elif channel_method == 'pca':
            # PCA-1 over channels
            Xc = X[n] - X[n].mean(axis=1, keepdims=True)
            U, S, Vt = np.linalg.svd(Xc, full_matrices=False)
            x = (U[:, 0] * S[0]) @ Vt[0]
        else:
            # channel average
            x = X[n].mean(axis=0)

        f, t, Sxx = spectrogram(
            x, fs=fs,
            nperseg=nperseg,
            noverlap=noverlap,
            scaling='density',
            mode='psd'
        )
        mask = f <= fmax
        S = Sxx[mask]
        S = np.log1p(S)                                # log
        S = (S - S.mean()) / (S.std() + 1e-8)          # z-norm
        imgs.append(S[np.newaxis, ...].astype(np.float32))  # (1,H,W)
    return np.stack(imgs, axis=0)

def extract_features(
    X_time_train, X_time_val, X_time_test, fs,
    bands=((4, 8), (8, 13), (13, 30)),
    welch_nperseg=None, welch_noverlap=None,
    make_images=True, spec_nperseg=None, spec_noverlap=None, fmax=40.0,
    channel_method='average'
):
    """
    Return dict with time, freq (bandpower), and optional spectrogram images.
    """
    features = {
        'X_time_train': X_time_train.astype(np.float32),
        'X_time_val'  : X_time_val.astype(np.float32),
        'X_time_test' : X_time_test.astype(np.float32),
        'sampling_rate': float(fs),
    }

    # frequency-domain bandpower
    features['X_freq_train'] = bandpower_welch(
        X_time_train, fs, bands, welch_nperseg, welch_noverlap
    )
    features['X_freq_val'] = bandpower_welch(
        X_time_val, fs, bands, welch_nperseg, welch_noverlap
    )
    features['X_freq_test'] = bandpower_welch(
        X_time_test, fs, bands, welch_nperseg, welch_noverlap
    )

    # time-frequency images
    if make_images:
        features['X_img_train'] = spectrogram_images(
            X_time_train, fs, spec_nperseg, spec_noverlap, fmax, channel_method
        )
        features['X_img_val'] = spectrogram_images(
            X_time_val, fs, spec_nperseg, spec_noverlap, fmax, channel_method
        )
        features['X_img_test'] = spectrogram_images(
            X_time_test, fs, spec_nperseg, spec_noverlap, fmax, channel_method
        )
    else:
        features['X_img_train'] = None
        features['X_img_val'] = None
        features['X_img_test'] = None

    return features
