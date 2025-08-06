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
    Apply 50 Hz notch + 1–40 Hz bandpass + z-score to a 1D signal.
    """
    # 1) Notch filter at mains frequency
    b_notch, a_notch = iirnotch(notch_freq, Q=notch_freq / notch_bw, fs=fs)
    sig = filtfilt(b_notch, a_notch, sig)

    # 2) Bandpass (butterworth)
    b_hp, a_hp = butter(4, hp_cutoff / (fs / 2), btype='high')
    sig = filtfilt(b_hp, a_hp, sig)
    b_lp, a_lp = butter(4, lp_cutoff / (fs / 2), btype='low')
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
    n_trials, n_ch, _ = X.shape
    X_time = np.empty_like(X, dtype=np.float32)
    for i in range(n_trials):
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
    bands: list[tuple[float, float]] = [(4,8), (8,13), (13,30)]
) -> np.ndarray:
    """
    X_time: (n_trials, n_channels, n_samples)
    returns: (n_trials, n_channels, n_bands) matrix of log‐bandpower
    """
    n_trials, n_ch, _ = X_time.shape
    n_bands = len(bands)
    X_freq = np.zeros((n_trials, n_ch, n_bands), dtype=np.float32)

    for i in range(n_trials):
        for b, (low, high) in enumerate(bands):
            # compute PSD via Welch
            f, Pxx = welch(
                X_time[i], fs=fs,
                nperseg=int(fs)  # 1 s windows
            )
            mask = (f >= low) & (f <= high)
            # average power across freq‐bins, then log
            band_pow = Pxx[:, mask].mean(axis=1)
            X_freq[i, :, b] = np.log(band_pow + 1e-6)
    return X_freq


def extract_timefreq_images(
    X_time: np.ndarray,
    fs: float,
    nperseg: int | None = None,
    noverlap: int | None = None
) -> np.ndarray:
    """
    X_time: (n_trials, n_channels, n_samples)
    returns: (n_trials, 1, n_freqs, n_times) spectrogram images using CH0
    """
    if nperseg is None:
        nperseg = int(fs)       # 1 second window
    if noverlap is None:
        noverlap = nperseg // 2

    n_trials = X_time.shape[0]
    # compute dims on first trial
    f, t, Sxx = spectrogram(
        X_time[0, 0], fs=fs,
        nperseg=nperseg, noverlap=noverlap
    )
    n_freqs, n_times = Sxx.shape

    X_img = np.zeros((n_trials, 1, n_freqs, n_times), dtype=np.float32)
    for i in range(n_trials):
        _, _, Sxx = spectrogram(
            X_time[i, 0], fs=fs,
            nperseg=nperseg, noverlap=noverlap
        )
        img = np.log(Sxx + 1e-6)
        img = (img - img.mean()) / (img.std() + 1e-6)
        X_img[i, 0, :, :] = img

    return X_img
