#!/usr/bin/env python3
"""
Unified Signal Preprocessor
Applies same preprocessing to all signals regardless of task type
- 1-40 Hz bandpass filter
- 50 Hz notch filter
- Z-score normalization per channel
"""

import numpy as np
from scipy import signal
from scipy.signal import butter, filtfilt, iirnotch


class UnifiedPreprocessor:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.lowcut = 1.0
        self.highcut = 40.0
        self.notch_freq = 50.0
        self.filter_order = 4
        
    def apply_bandpass_filter(self, data):
        """
        Apply 1-40 Hz bandpass filter
        Args:
            data: numpy array of shape (n_channels, n_samples) or (n_samples,)
        Returns:
            filtered_data: same shape as input
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_channel = True
        else:
            single_channel = False
            
        nyquist = 0.5 * self.sampling_rate
        low = self.lowcut / nyquist
        high = self.highcut / nyquist
        
        if high >= 1.0:
            high = 0.99
            
        b, a = butter(self.filter_order, [low, high], btype='band')
        
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch] = filtfilt(b, a, data[ch])
            
        if single_channel:
            return filtered_data[0]
        return filtered_data
        
    def apply_notch_filter(self, data):
        """
        Apply 50 Hz notch filter to remove power line interference
        Args:
            data: numpy array of shape (n_channels, n_samples) or (n_samples,)
        Returns:
            filtered_data: same shape as input
        """
        if data.ndim == 1:
            data = data.reshape(1, -1)
            single_channel = True
        else:
            single_channel = False
            
        quality_factor = 30.0
        b, a = iirnotch(self.notch_freq, quality_factor, self.sampling_rate)
        
        filtered_data = np.zeros_like(data)
        for ch in range(data.shape[0]):
            filtered_data[ch] = filtfilt(b, a, data[ch])
            
        if single_channel:
            return filtered_data[0]
        return filtered_data
        
    def apply_zscore_normalization(self, data):
        """
        Apply Z-score normalization per channel
        Args:
            data: numpy array of shape (n_channels, n_samples) or (n_samples,)
        Returns:
            normalized_data: same shape as input
        """
        if data.ndim == 1:
            mean_val = np.mean(data)
            std_val = np.std(data)
            if std_val == 0:
                std_val = 1.0
            return (data - mean_val) / std_val
        else:
            normalized_data = np.zeros_like(data)
            for ch in range(data.shape[0]):
                mean_val = np.mean(data[ch])
                std_val = np.std(data[ch])
                if std_val == 0:
                    std_val = 1.0
                normalized_data[ch] = (data[ch] - mean_val) / std_val
            return normalized_data
            
    def preprocess_trial(self, trial_data):
        """
        Apply full preprocessing pipeline to single trial
        Args:
            trial_data: numpy array of shape (n_channels, n_samples)
        Returns:
            processed_data: preprocessed trial data
        """
        # Step 1: Bandpass filter
        filtered_data = self.apply_bandpass_filter(trial_data)
        
        # Step 2: Notch filter
        notched_data = self.apply_notch_filter(filtered_data)
        
        # Step 3: Z-score normalization
        normalized_data = self.apply_zscore_normalization(notched_data)
        
        return normalized_data
        
    def preprocess_batch(self, batch_data):
        """
        Apply preprocessing to batch of trials
        Args:
            batch_data: numpy array of shape (n_trials, n_channels, n_samples)
        Returns:
            processed_batch: preprocessed batch data
        """
        n_trials = batch_data.shape[0]
        processed_batch = np.zeros_like(batch_data)
        
        for trial_idx in range(n_trials):
            processed_batch[trial_idx] = self.preprocess_trial(batch_data[trial_idx])
            
        return processed_batch
        
    def extract_windows(self, trial_data, window_length_sec, overlap_sec=0):
        """
        Extract sliding windows from trial data
        Args:
            trial_data: numpy array of shape (n_channels, n_samples)
            window_length_sec: window length in seconds
            overlap_sec: overlap between windows in seconds
        Returns:
            windows: numpy array of shape (n_windows, n_channels, window_samples)
        """
        window_samples = int(window_length_sec * self.sampling_rate)
        overlap_samples = int(overlap_sec * self.sampling_rate)
        step_samples = window_samples - overlap_samples
        
        n_channels, n_samples = trial_data.shape
        
        if n_samples < window_samples:
            return np.array([trial_data])
            
        windows = []
        start = 0
        while start + window_samples <= n_samples:
            window = trial_data[:, start:start + window_samples]
            windows.append(window)
            start += step_samples
            
        return np.array(windows)
        
    def validate_signal_quality(self, data, amplitude_threshold=5000):
        """
        Basic signal quality validation
        Args:
            data: numpy array of any shape
            amplitude_threshold: maximum allowed amplitude in microvolts
        Returns:
            is_valid: boolean indicating if signal quality is acceptable
            issues: list of detected issues
        """
        issues = []
        
        # Check for amplitude artifacts
        max_amplitude = np.max(np.abs(data))
        if max_amplitude > amplitude_threshold:
            issues.append(f"High amplitude artifact detected: {max_amplitude:.1f} uV")
            
        # Check for flat signals
        for ch in range(data.shape[0] if data.ndim > 1 else 1):
            ch_data = data[ch] if data.ndim > 1 else data
            if np.std(ch_data) < 0.1:
                issues.append(f"Flat signal detected in channel {ch + 1}")
                
        # Check for NaN or infinite values
        if np.any(np.isnan(data)) or np.any(np.isinf(data)):
            issues.append("NaN or infinite values detected")
            
        is_valid = len(issues) == 0
        return is_valid, issues
