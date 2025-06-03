# utils/preprocess.py

import numpy as np
from scipy.signal import butter, lfilter

def bandpass_filter(data, lowcut=1.0, highcut=40.0, fs=1024, order=4):
    """
    Apply a band-pass filter to a single trial (channels × samples).
    Parameters:
        data (np.ndarray): EEG data of shape (n_channels, n_samples)
        lowcut (float): Lower cut-off frequency in Hz
        highcut (float): Upper cut-off frequency in Hz
        fs (float): Sampling frequency in Hz
        order (int): Filter order
    Returns:
        filtered (np.ndarray): Filtered data, same shape as input
    """
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered = lfilter(b, a, data, axis=1)
    return filtered

def normalize_data(data):
    """
    Z-score normalize a single trial’s EEG data (channels × samples).
    Each channel is normalized independently.
    Parameters:
        data (np.ndarray): EEG data of shape (n_channels, n_samples)
    Returns:
        normed (np.ndarray): Z-score normalized data, same shape
    """
    mean = np.mean(data, axis=1, keepdims=True)
    std = np.std(data, axis=1, keepdims=True) + 1e-8
    return (data - mean) / std

def preprocess_batch(trials, fs=1024):
    """
    Apply band-pass filtering and z-score normalization to a batch of trials.
    Parameters:
        trials (np.ndarray): Array of shape (n_trials, n_channels, n_samples)
        fs (float): Sampling frequency in Hz
    Returns:
        processed (np.ndarray): Preprocessed data, same shape as input
    """
    n_trials, n_ch, n_samp = trials.shape
    processed = np.zeros_like(trials)
    for i in range(n_trials):
        # 1. Band-pass filter each trial
        filtered = bandpass_filter(trials[i], lowcut=1.0, highcut=40.0, fs=fs, order=4)
        # 2. Z-score normalize each channel
        processed[i] = normalize_data(filtered)
    return processed
