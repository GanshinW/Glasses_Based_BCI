#!/usr/bin/env python3
"""
Direct NPZ Dataset Generator
Generates intent detection (100 trials) and emotion recognition (60 trials) datasets
"""

import numpy as np
from datetime import datetime
import os

def generate_realistic_eeg_signal(duration_sec=10.0, fs=250, n_channels=6, signal_type='baseline'):
    """
    Generate realistic EEG signal with physiological characteristics
    """
    n_samples = int(duration_sec * fs)
    t = np.arange(n_samples) / fs
    
    # Initialize signal array
    eeg_signal = np.zeros((n_channels, n_samples))
    
    # Base EEG frequencies and amplitudes
    freq_bands = {
        'delta': (0.5, 4, 20),    # (low_freq, high_freq, amplitude)
        'theta': (4, 8, 15),
        'alpha': (8, 13, 50),
        'beta': (13, 30, 10),
        'gamma': (30, 40, 5)
    }
    
    for ch_idx in range(n_channels):
        signal = np.zeros(n_samples)
        
        # Generate base EEG with multiple frequency components
        for band_name, (f_low, f_high, base_amp) in freq_bands.items():
            # Generate 2 frequency components per band
            freqs = np.linspace(f_low + 0.5, f_high - 0.5, 2)
            
            for freq in freqs:
                amplitude = base_amp * np.random.uniform(0.8, 1.2)
                phase = np.random.uniform(0, 2*np.pi)
                
                # Add slight frequency modulation for realism
                freq_mod = freq + 0.3 * np.sin(2 * np.pi * 0.1 * t)
                component = amplitude * np.sin(2 * np.pi * freq_mod * t + phase)
                
                # Apply amplitude modulation (natural brain oscillations)
                am_freq = np.random.uniform(0.05, 0.15)
                envelope = 1 + 0.2 * np.sin(2 * np.pi * am_freq * t)
                
                signal += component * envelope / len(freqs)
        
        # Add realistic noise
        noise = np.random.normal(0, 2.0, n_samples)  # 2μV RMS noise
        
        # Add 50Hz powerline interference
        powerline = 0.5 * np.sin(2 * np.pi * 50 * t + np.random.uniform(0, 2*np.pi))
        
        eeg_signal[ch_idx] = signal + noise + powerline
    
    # Apply signal-specific modifications
    if signal_type == 'jaw_clench':
        # Add EMG artifact during 4-7 seconds
        emg_start_idx = int(4.0 * fs)
        emg_end_idx = int(7.0 * fs)
        emg_duration = emg_end_idx - emg_start_idx
        emg_t = t[emg_start_idx:emg_end_idx]
        
        # EMG is strongest at FT7, FT8 (channels 1, 4)
        emg_channels = [1, 4]  # FT7, FT8
        for ch_idx in emg_channels:
            # High frequency EMG (20-200 Hz)
            emg_freqs = np.linspace(20, 200, 20)
            emg_signal = np.zeros(emg_duration)
            
            for freq in emg_freqs:
                amplitude = np.random.exponential(200)  # High amplitude EMG
                phase = np.random.uniform(0, 2*np.pi)
                emg_signal += amplitude * np.sin(2 * np.pi * freq * emg_t + phase)
            
            # Apply envelope (gradual onset/offset)
            rise_samples = int(0.1 * fs)
            fall_samples = int(0.2 * fs)
            envelope = np.ones(emg_duration)
            
            if emg_duration > rise_samples:
                envelope[:rise_samples] = np.linspace(0, 1, rise_samples)
            if emg_duration > fall_samples:
                envelope[-fall_samples:] = np.linspace(1, 0, fall_samples)
            
            eeg_signal[ch_idx, emg_start_idx:emg_end_idx] += emg_signal * envelope / len(emg_freqs)
    
    elif signal_type == 'gaze_left':
        # Add EOG artifacts for leftward gaze during 4-7 seconds
        eog_times = [4.0, 4.8, 5.6]  # Multiple eye movements
        
        for eog_start in eog_times:
            eog_start_idx = int(eog_start * fs)
            eog_end_idx = int((eog_start + 0.6) * fs)
            eog_duration = eog_end_idx - eog_start_idx
            eog_t = t[eog_start_idx:eog_end_idx]
            
            if eog_end_idx > n_samples:
                continue
            
            # T7 negative, T8 positive for left gaze
            eog_polarity = {'T7': -1, 'T8': 1}  # channels 2, 5
            channel_map = {'T7': 2, 'T8': 5}
            
            for ch_name, polarity in eog_polarity.items():
                ch_idx = channel_map[ch_name]
                
                # Low frequency EOG signal (2-8 Hz)
                eog_freqs = [2, 3, 5, 7]
                eog_signal = np.zeros(eog_duration)
                
                for freq in eog_freqs:
                    amplitude = 80 * abs(polarity)  # 80μV EOG
                    phase = np.random.uniform(0, 2*np.pi)
                    eog_signal += np.sin(2 * np.pi * freq * eog_t + phase)
                
                # EOG envelope (saccade + drift)
                saccade_samples = int(0.08 * fs)
                if eog_duration > saccade_samples:
                    saccade_env = 1 - np.exp(-5 * np.linspace(0, 1, saccade_samples))
                    drift_env = np.exp(-2 * np.linspace(0, 1, eog_duration - saccade_samples))
                    full_envelope = np.concatenate([saccade_env, drift_env])
                else:
                    full_envelope = 1 - np.exp(-5 * np.linspace(0, 1, eog_duration))
                
                eeg_signal[ch_idx, eog_start_idx:eog_end_idx] += polarity * eog_signal * full_envelope / len(eog_freqs)
    
    elif signal_type == 'gaze_right':
        # Same as left but opposite polarity
        eog_times = [4.0, 4.8, 5.6]
        
        for eog_start in eog_times:
            eog_start_idx = int(eog_start * fs)
            eog_end_idx = int((eog_start + 0.6) * fs)
            eog_duration = eog_end_idx - eog_start_idx
            eog_t = t[eog_start_idx:eog_end_idx]
            
            if eog_end_idx > n_samples:
                continue
            
            # T7 positive, T8 negative for right gaze
            eog_polarity = {'T7': 1, 'T8': -1}
            channel_map = {'T7': 2, 'T8': 5}
            
            for ch_name, polarity in eog_polarity.items():
                ch_idx = channel_map[ch_name]
                
                eog_freqs = [2, 3, 5, 7]
                eog_signal = np.zeros(eog_duration)
                
                for freq in eog_freqs:
                    amplitude = 80 * abs(polarity)
                    phase = np.random.uniform(0, 2*np.pi)
                    eog_signal += np.sin(2 * np.pi * freq * eog_t + phase)
                
                saccade_samples = int(0.08 * fs)
                if eog_duration > saccade_samples:
                    saccade_env = 1 - np.exp(-5 * np.linspace(0, 1, saccade_samples))
                    drift_env = np.exp(-2 * np.linspace(0, 1, eog_duration - saccade_samples))
                    full_envelope = np.concatenate([saccade_env, drift_env])
                else:
                    full_envelope = 1 - np.exp(-5 * np.linspace(0, 1, eog_duration))
                
                eeg_signal[ch_idx, eog_start_idx:eog_end_idx] += polarity * eog_signal * full_envelope / len(eog_freqs)
    
    elif signal_type == 'positive':
        # Enhanced left frontal activity (F7, FT7 - channels 0, 1)
        left_channels = [0, 1]
        for ch_idx in left_channels:
            # Increase alpha and beta power
            alpha_boost = 25 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
            beta_boost = 15 * np.sin(2 * np.pi * 20 * t + np.random.uniform(0, 2*np.pi))
            eeg_signal[ch_idx] += alpha_boost + beta_boost
    
    elif signal_type == 'negative':
        # Enhanced right frontal + alpha suppression
        right_channels = [3, 4]  # F8, FT8
        for ch_idx in right_channels:
            theta_boost = 20 * np.sin(2 * np.pi * 6 * t + np.random.uniform(0, 2*np.pi))
            beta_boost = 18 * np.sin(2 * np.pi * 22 * t + np.random.uniform(0, 2*np.pi))
            eeg_signal[ch_idx] += theta_boost + beta_boost
        
        # Alpha suppression across all channels
        for ch_idx in range(n_channels):
            alpha_suppression = -8 * np.sin(2 * np.pi * 10 * t + np.random.uniform(0, 2*np.pi))
            eeg_signal[ch_idx] += alpha_suppression
    
    elif signal_type == 'neutral':
        # Slightly enhanced alpha rhythm
        for ch_idx in range(n_channels):
            alpha_enhancement = 12 * np.sin(2 * np.pi * 9.5 * t + np.random.uniform(0, 2*np.pi))
            eeg_signal[ch_idx] += alpha_enhancement
    
    return eeg_signal

def manual_train_test_split(X, y, test_size=0.15, random_state=42):
    """Manual train-test split to avoid sklearn dependency"""
    np.random.seed(random_state)
    n_samples = len(X)
    
    # Get unique classes
    unique_classes = np.unique(y)
    train_indices = []
    test_indices = []
    
    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        n = len(cls_indices)
        
        n_test = max(1, int(n * test_size))
        test_indices.extend(cls_indices[:n_test])
        train_indices.extend(cls_indices[n_test:])
    
    return X[train_indices], X[test_indices], y[train_indices], y[test_indices]

def create_intent_detection_dataset(n_trials=100):
    """Create intent detection dataset with 100 trials"""
    
    print("Generating Intent Detection Dataset...")
    print(f"Trials: {n_trials} total (25 per class)")
    
    # Set random seed for reproducibility
    np.random.seed(42)
    
    # Intent classes
    intent_classes = {
        'baseline': 0,
        'jaw_clench': 1, 
        'gaze_left': 2,
        'gaze_right': 3
    }
    
    trials_per_class = n_trials // 4
    trial_duration = 10.0  # seconds
    fs = 250
    channels = ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8']
    
    all_trials = []
    all_labels = []
    
    for intent_name, label in intent_classes.items():
        print(f"  Generating {trials_per_class} trials for {intent_name}...")
        
        for trial_idx in range(trials_per_class):
            signal = generate_realistic_eeg_signal(
                duration_sec=trial_duration,
                fs=fs,
                n_channels=6,
                signal_type=intent_name
            )
            
            all_trials.append(signal)
            all_labels.append(label)
    
    # Convert to numpy arrays
    X_data = np.array(all_trials)  # Shape: (trials, channels, samples)
    y_labels = np.array(all_labels)
    
    print(f"  Generated data shape: {X_data.shape}")
    print(f"  Data range: [{X_data.min():.1f}, {X_data.max():.1f}] μV")
    
    # Create train/val/test splits using manual function
    X_train, X_temp, y_train, y_temp = manual_train_test_split(
        X_data, y_labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = manual_train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Create dataset dictionary
    dataset = {
        # Core data arrays
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        
        # Metadata
        'channel_names': np.array(channels),
        'sampling_rate': fs,
        'window_length': 2.0,
        'task_type': 'intent_detection',
        'class_names': np.array(list(intent_classes.keys())),
        'label_map': intent_classes,
        
        # Preprocessing info
        'preprocessed': False,
        'preprocessing_params': {
            'lowcut': 1.0,
            'highcut': 40.0,
            'notch_freq': 50.0,
            'filter_order': 4
        },
        
        # Additional metadata
        'trial_duration': trial_duration,
        'created_date': datetime.now().isoformat(),
        'description': 'Simulated intent detection dataset for glasses-based BCI (100 trials)'
    }
    
    return dataset

def create_emotion_recognition_dataset(n_trials=60):
    """Create emotion recognition dataset with 60 trials"""
    
    print("Generating Emotion Recognition Dataset...")
    print(f"Trials: {n_trials} total (20 per class)")
    
    # Set random seed for reproducibility
    np.random.seed(123)
    
    # Emotion classes
    emotion_classes = {
        'negative': 0,
        'neutral': 1,
        'positive': 2
    }
    
    trials_per_class = n_trials // 3
    trial_duration = 10.0
    fs = 250
    channels = ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8']
    
    all_trials = []
    all_labels = []
    all_ratings = []
    
    # Rating ranges for each emotion
    rating_ranges = {
        'negative': (1, 3),
        'neutral': (4, 6), 
        'positive': (7, 9)
    }
    
    for emotion_name, label in emotion_classes.items():
        print(f"  Generating {trials_per_class} trials for {emotion_name}...")
        
        for trial_idx in range(trials_per_class):
            signal = generate_realistic_eeg_signal(
                duration_sec=trial_duration,
                fs=fs,
                n_channels=6,
                signal_type=emotion_name
            )
            
            # Generate corresponding rating
            rating_min, rating_max = rating_ranges[emotion_name]
            rating = np.random.randint(rating_min, rating_max + 1)
            
            all_trials.append(signal)
            all_labels.append(label)
            all_ratings.append(rating)
    
    # Convert to numpy arrays
    X_data = np.array(all_trials)
    y_labels = np.array(all_labels)
    ratings = np.array(all_ratings)
    
    print(f"  Generated data shape: {X_data.shape}")
    print(f"  Data range: [{X_data.min():.1f}, {X_data.max():.1f}] μV")
    print(f"  Rating range: {ratings.min()}-{ratings.max()}")
    
    # Create train/val/test splits
    X_train, X_temp, y_train, y_temp = manual_train_test_split(
        X_data, y_labels, test_size=0.3, random_state=42
    )
    X_val, X_test, y_val, y_test = manual_train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42
    )
    
    # Split ratings accordingly
    train_size = len(X_train)
    val_size = len(X_val)
    r_train = ratings[:train_size]
    r_val = ratings[train_size:train_size + val_size]
    r_test = ratings[train_size + val_size:]
    
    # Create dataset dictionary
    dataset = {
        # Core data arrays
        'X_train': X_train,
        'y_train': y_train,
        'X_val': X_val,
        'y_val': y_val,
        'X_test': X_test,
        'y_test': y_test,
        
        # Emotion-specific data
        'ratings_train': r_train,
        'ratings_val': r_val,
        'ratings_test': r_test,
        
        # Metadata
        'channel_names': np.array(channels),
        'sampling_rate': fs,
        'window_length': 2.0,
        'task_type': 'emotion_recognition',
        'class_names': np.array(list(emotion_classes.keys())),
        'label_map': emotion_classes,
        
        # Preprocessing info
        'preprocessed': False,
        'preprocessing_params': {
            'lowcut': 1.0,
            'highcut': 40.0,
            'notch_freq': 50.0,
            'filter_order': 4
        },
        
        # Additional metadata
        'trial_duration': trial_duration,
        'created_date': datetime.now().isoformat(),
        'description': 'Simulated emotion recognition dataset for glasses-based BCI (60 trials)'
    }
    
    return dataset

def save_dataset(dataset, filename):
    """Save dataset to NPZ format"""
    np.savez_compressed(filename, **dataset)
    print(f"\nDataset saved: {filename}")
    
    # Print statistics
    print(f"\n=== Dataset Statistics ===")
    print(f"Task: {dataset['task_type']}")
    print(f"Channels: {list(dataset['channel_names'])}")
    print(f"Sampling rate: {dataset['sampling_rate']} Hz")
    print(f"Classes: {list(dataset['class_names'])}")
    
    print(f"\n=== Data Splits ===")
    print(f"Training: {len(dataset['X_train'])} samples")
    print(f"Validation: {len(dataset['X_val'])} samples")
    print(f"Test: {len(dataset['X_test'])} samples")
    
    print(f"\n=== Data Shape ===")
    print(f"X_train: {dataset['X_train'].shape} (trials, channels, samples)")
    print(f"Data range: [{dataset['X_train'].min():.2f}, {dataset['X_train'].max():.2f}] μV")
    
    # Class distribution
    print(f"\n=== Class Distribution ===")
    for split in ['train', 'val', 'test']:
        y_key = f'y_{split}'
        y_data = dataset[y_key]
        print(f"{split.upper()}:")
        class_names = list(dataset['class_names'])
        label_map = dataset['label_map']
        for class_name in class_names:
            if class_name in label_map:
                class_idx = label_map[class_name]
                count = np.sum(y_data == class_idx)
                percentage = count / len(y_data) * 100 if len(y_data) > 0 else 0
                print(f"  {class_name}: {count} ({percentage:.1f}%)")

def main():
    """Main function to generate both datasets"""
    print("="*70)
    print("EEG DATASET GENERATION")
    print("Glasses-Based BCI Project - Direct NPZ Creation")
    print("="*70)
    
    # Create output directory
    os.makedirs('data', exist_ok=True)
    
    # Generate timestamp for filenames
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Generate Intent Detection Dataset (100 trials)
    print("\n1. INTENT DETECTION DATASET")
    print("-" * 40)
    intent_dataset = create_intent_detection_dataset(n_trials=100)
    intent_filename = f'data/intent_dataset_{timestamp}.npz'
    save_dataset(intent_dataset, intent_filename)
    
    print("\n" + "="*60 + "\n")
    
    # Generate Emotion Recognition Dataset (60 trials)
    print("2. EMOTION RECOGNITION DATASET")
    print("-" * 40)
    emotion_dataset = create_emotion_recognition_dataset(n_trials=60)
    emotion_filename = f'data/emotion_dataset_{timestamp}.npz'
    save_dataset(emotion_dataset, emotion_filename)
    
    print("\n" + "="*70)
    print("DATASET GENERATION COMPLETE!")
    print("="*70)
    
    print(f"\nGenerated files:")
    print(f"  Intent Detection: {intent_filename}")
    print(f"  Emotion Recognition: {emotion_filename}")
    
    print(f"\nDataset specifications:")
    print(f"  Intent Detection: 100 trials (25 per class)")
    print(f"  Emotion Recognition: 60 trials (20 per class)")
    print(f"  Channels: 6 (F7, FT7, T7, F8, FT8, T8)")
    print(f"  Sampling rate: 250 Hz")
    print(f"  Trial duration: 10 seconds (2500 samples)")
    print(f"  Signal features: EEG + EOG + EMG artifacts")
    
    print(f"\nUsage example:")
    print(f"  import numpy as np")
    print(f"  data = np.load('{intent_filename}')")
    print(f"  X_train = data['X_train']  # Shape: (70, 6, 2500)")
    print(f"  y_train = data['y_train']  # Labels: 0-3")
    print(f"  channels = data['channel_names']")
    
    return intent_filename, emotion_filename

if __name__ == "__main__":
    main()