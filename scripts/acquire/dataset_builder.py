#!/usr/bin/env python3
"""
Dataset Builder for Intent and Emotion Recognition
Converts raw acquisition data to training-ready datasets
Applies unified preprocessing and creates NPZ format datasets
"""

import json
import csv
import numpy as np
import os
from datetime import datetime
from unified_preprocessor import UnifiedPreprocessor

# Manual train_test_split to avoid sklearn dependency issues
def manual_train_test_split(X, y, test_size=0.15, random_state=42):
    """Manual implementation with small-sample safety"""
    np.random.seed(random_state)
    n_samples = len(X)
    if n_samples == 0:
        return X, X, y, y

    unique_classes = np.unique(y)
    train_indices, test_indices = [], []

    for cls in unique_classes:
        cls_indices = np.where(y == cls)[0]
        np.random.shuffle(cls_indices)
        n = len(cls_indices)

        # keep at least 1 sample per class in train
        if n <= 2:
            n_cls_test = 0
        else:
            n_cls_test = max(1, int(round(n * test_size)))
            if n - n_cls_test < 1:
                n_cls_test = n - 1

        test_indices.extend(cls_indices[:n_cls_test])
        train_indices.extend(cls_indices[n_cls_test:])

    # if train is empty, fall back to all-train
    if len(train_indices) == 0:
        train_indices = list(range(n_samples))
        test_indices = []

    X_train = X[train_indices]
    y_train = y[train_indices]

    if len(test_indices):
        X_test = X[test_indices]
        y_test = y[test_indices]
    else:
        X_test = np.empty((0,) + X.shape[1:], dtype=X.dtype)
        y_test = np.empty((0,), dtype=y.dtype)

    return X_train, X_test, y_train, y_test


class DatasetBuilder:
    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.preprocessor = UnifiedPreprocessor(sampling_rate)
        self.channel_names = ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8']
        
    def load_intent_data(self, data_dirs):
        """
        Load intent detection data from CSV files and session info
        Args:
            data_dirs: list of data directories containing CSV files and session_info.json
        Returns:
            trials: list of trial dictionaries
        """
        all_trials = []
        
        for data_dir in data_dirs:
            # Load session info
            session_file = os.path.join(data_dir, "session_info.json")
            if not os.path.exists(session_file):
                print(f"No session_info.json found in {data_dir}")
                continue
                
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                
            # Find all intent trial CSV files
            csv_files = [f for f in os.listdir(data_dir) if f.startswith('intent_trial_') and f.endswith('.csv')]
            
            for csv_file in csv_files:
                trial_data = self._load_intent_csv_file(os.path.join(data_dir, csv_file), session_info)
                if trial_data:
                    all_trials.append(trial_data)
                    
            print(f"Loaded {len(csv_files)} intent trials from {data_dir}")
            
        return all_trials
        
    def load_emotion_data(self, data_dirs):
        """
        Load emotion recognition data from CSV files and session info
        Args:
            data_dirs: list of data directories containing CSV files and session_info.json
        Returns:
            trials: list of trial dictionaries
        """
        all_trials = []
        
        for data_dir in data_dirs:
            # Load session info
            session_file = os.path.join(data_dir, "session_info.json")
            if not os.path.exists(session_file):
                print(f"No session_info.json found in {data_dir}")
                continue
                
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                
            # Find all emotion trial CSV files
            csv_files = [f for f in os.listdir(data_dir) if f.startswith('emotion_trial_') and f.endswith('.csv')]
            
            for csv_file in csv_files:
                trial_data = self._load_emotion_csv_file(os.path.join(data_dir, csv_file), session_info)
                if trial_data:
                    all_trials.append(trial_data)
                    
            print(f"Loaded {len(csv_files)} emotion trials from {data_dir}")
            
        return all_trials
        
    def _load_intent_csv_file(self, file_path, session_info):
        """Load single intent trial from CSV file"""
        try:
            # Parse metadata from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('intent_trial_', '').replace('.csv', '').split('_')
            participant_id = parts[0]
            trial_id = int(parts[1])
            action_label = parts[2]
            
            # Load CSV data
            samples = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Skip metadata comment lines
                data_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('timestamp,') or 'timestamp' in line:
                        data_start = i
                        break
                        
                # Read actual data
                import io
                data_content = ''.join(lines[data_start:])
                reader = csv.DictReader(io.StringIO(data_content))
                
                for row in reader:
                    # Skip any remaining comment lines
                    if row['timestamp'].startswith('#') or not row['timestamp'].strip():
                        continue
                        
                    try:
                        sample = {
                            'timestamp': float(row['timestamp']),
                            'phase_time': float(row['phase_time']),
                            'phase': row['phase'],
                            'channels': [float(row[f'ch{i}']) for i in range(1, 7)]
                        }
                        samples.append(sample)
                    except (ValueError, KeyError) as e:
                        print(f"Skipping invalid row in {file_path}: {row}")
                        continue
                    
            # Extract action phase samples only
            action_samples = [s for s in samples if s['phase'] == 'action']
            
            if len(action_samples) > 0:
                print(f"Loaded trial {trial_id} ({action_label}): {len(action_samples)} action samples")
                return {
                    'trial_id': trial_id,
                    'label': action_label,
                    'samples': action_samples,
                    'participant_id': participant_id
                }
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return None
        
    def _load_emotion_csv_file(self, file_path, session_info):
        """Load single emotion trial from CSV file"""
        try:
            # Parse metadata from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('emotion_trial_', '').replace('.csv', '').split('_')
            participant_id = parts[0]
            trial_id = int(parts[1])
            emotion_category = parts[2]
            rating = int(parts[3])
            
            # Load CSV data
            samples = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                
                # Skip metadata comment lines
                data_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('timestamp,') or 'timestamp' in line:
                        data_start = i
                        break
                        
                # Read actual data
                import io
                data_content = ''.join(lines[data_start:])
                reader = csv.DictReader(io.StringIO(data_content))
                
                for row in reader:
                    # Skip any remaining comment lines
                    if row['timestamp'].startswith('#') or not row['timestamp'].strip():
                        continue
                        
                    try:
                        sample = {
                            'timestamp': float(row['timestamp']),
                            'recording_time': float(row['recording_time']),
                            'channels': [float(row[f'ch{i}']) for i in range(1, 7)]
                        }
                        samples.append(sample)
                    except (ValueError, KeyError) as e:
                        print(f"Skipping invalid row in {file_path}: {row}")
                        continue
                    
            if len(samples) > 0:
                print(f"Loaded trial {trial_id} ({emotion_category}, rating {rating}): {len(samples)} samples")
                return {
                    'trial_id': trial_id,
                    'label': emotion_category,
                    'rating': rating,
                    'samples': samples,
                    'participant_id': participant_id
                }
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            import traceback
            traceback.print_exc()
            
        return None
        
    def create_intent_dataset(self, trials, window_length_sec=2.0):
        """
        Create intent detection dataset with sliding windows (RAW DATA - NO PREPROCESSING)
        Args:
            trials: list of trial dictionaries
            window_length_sec: window length for real-time prediction
        Returns:
            dataset: dictionary with raw X, y, and metadata
        """
        X_data = []
        y_labels = []
        
        # Label mapping
        label_map = {'baseline': 0, 'jaw_clench': 1, 'gaze_left': 2, 'gaze_right': 3}
        
        for trial in trials:
            if trial['label'] not in label_map:
                continue
                
            # Convert samples to numpy array (RAW DATA)
            channels_data = np.array([s['channels'] for s in trial['samples']])
            if channels_data.shape[0] < 10:  # Skip if too few samples
                continue
                
            # Transpose to (channels, samples)
            trial_data = channels_data.T
            
            # Extract windows WITHOUT preprocessing (raw signal)
            window_samples = int(window_length_sec * self.sampling_rate)
            overlap_samples = int(0.5 * self.sampling_rate)  # 0.5s overlap
            step_samples = window_samples - overlap_samples
            
            n_channels, n_samples = trial_data.shape
            
            if n_samples < window_samples:
                # If trial shorter than window, use entire trial
                X_data.append(trial_data)
                y_labels.append(label_map[trial['label']])
            else:
                # Extract sliding windows
                start = 0
                while start + window_samples <= n_samples:
                    window = trial_data[:, start:start + window_samples]
                    X_data.append(window)
                    y_labels.append(label_map[trial['label']])
                    start += step_samples
                
        X_data = np.array(X_data)
        y_labels = np.array(y_labels)
        
        # Split dataset manually
        X_train, X_temp, y_train, y_temp = manual_train_test_split(
            X_data, y_labels, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = manual_train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'channel_names': self.channel_names,
            'sampling_rate': self.sampling_rate,
            'window_length': window_length_sec,
            'task_type': 'intent_detection',
            'class_names': list(label_map.keys()),
            'label_map': label_map,
            'preprocessed': False,  # Mark as raw data
            'preprocessing_params': {
                'lowcut': 1.0,
                'highcut': 40.0,
                'notch_freq': 50.0,
                'filter_order': 4
            }
        }
        
        return dataset
        
    def create_emotion_dataset(self, trials, window_length_sec=2.0):
        """
        Create emotion recognition dataset (RAW DATA - NO PREPROCESSING)
        Args:
            trials: list of trial dictionaries  
            window_length_sec: window length for analysis
        Returns:
            dataset: dictionary with raw X, y, and metadata
        """
        X_data = []
        y_labels = []
        ratings = []
        
        # Label mapping
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        for trial in trials:
            if trial['label'] not in label_map:
                continue
                
            # Convert samples to numpy array (RAW DATA)
            channels_data = np.array([s['channels'] for s in trial['samples']])
            if channels_data.shape[0] < 10:  # Skip if too few samples
                continue
                
            # Transpose to (channels, samples)
            trial_data = channels_data.T
            
            # Extract windows WITHOUT preprocessing (raw signal)
            window_samples = int(window_length_sec * self.sampling_rate)
            overlap_samples = int(1.0 * self.sampling_rate)  # 1.0s overlap
            step_samples = window_samples - overlap_samples
            
            n_channels, n_samples = trial_data.shape
            
            if n_samples < window_samples:
                # If trial shorter than window, use entire trial
                X_data.append(trial_data)
                y_labels.append(label_map[trial['label']])
                ratings.append(trial['rating'])
            else:
                # Extract sliding windows
                start = 0
                while start + window_samples <= n_samples:
                    window = trial_data[:, start:start + window_samples]
                    X_data.append(window)
                    y_labels.append(label_map[trial['label']])
                    ratings.append(trial['rating'])
                    start += step_samples
                
        X_data = np.array(X_data)
        y_labels = np.array(y_labels)
        ratings = np.array(ratings)
        
        # Split dataset manually
        X_train, X_temp, y_train, y_temp = manual_train_test_split(
            X_data, y_labels, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = manual_train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )
        
        # Split ratings accordingly (simple approach)
        train_indices = np.isin(np.arange(len(y_labels)), np.where(np.isin(y_labels, y_train))[0][:len(y_train)])
        val_indices = np.isin(np.arange(len(y_labels)), np.where(np.isin(y_labels, y_val))[0][:len(y_val)])
        test_indices = np.isin(np.arange(len(y_labels)), np.where(np.isin(y_labels, y_test))[0][:len(y_test)])
        
        r_train = ratings[:len(y_train)]
        r_val = ratings[:len(y_val)]  
        r_test = ratings[:len(y_test)]
        
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'ratings_train': r_train,
            'ratings_val': r_val,
            'ratings_test': r_test,
            'channel_names': self.channel_names,
            'sampling_rate': self.sampling_rate,
            'window_length': window_length_sec,
            'task_type': 'emotion_recognition',
            'class_names': list(label_map.keys()),
            'label_map': label_map,
            'preprocessed': False,  # Mark as raw data
            'preprocessing_params': {
                'lowcut': 1.0,
                'highcut': 40.0,
                'notch_freq': 50.0,
                'filter_order': 4
            }
        }
        
        return dataset
        
    def save_dataset(self, dataset, filename):
        """
        Save dataset to NPZ format
        Args:
            dataset: dataset dictionary
            filename: output filename
        """
        np.savez_compressed(filename, **dataset)
        print(f"Dataset saved: {filename}")
        
        # Print dataset statistics
        self._print_dataset_stats(dataset)
        
    def _print_dataset_stats(self, dataset):
        """Print dataset statistics"""
        print(f"\n=== Dataset Statistics ===")
        print(f"Task: {dataset['task_type']}")
        print(f"Data type: {'RAW (unprocessed)' if not dataset['preprocessed'] else 'PREPROCESSED'}")
        print(f"Window length: {dataset['window_length']}s")
        print(f"Channels: {len(dataset['channel_names'])} {dataset['channel_names']}")
        print(f"Sampling rate: {dataset['sampling_rate']} Hz")
        print(f"Classes: {dataset['class_names']}")
        
        if not dataset['preprocessed']:
            params = dataset['preprocessing_params']
            print(f"\n=== Recommended Preprocessing ===")
            print(f"Bandpass filter: {params['lowcut']}-{params['highcut']} Hz")
            print(f"Notch filter: {params['notch_freq']} Hz")
            print(f"Filter order: {params['filter_order']}")
            print(f"Z-score normalization: per channel")
        
        print(f"\n=== Data Splits ===")
        print(f"Training: {len(dataset['X_train'])} samples")
        print(f"Validation: {len(dataset['X_val'])} samples") 
        print(f"Test: {len(dataset['X_test'])} samples")
        
        print(f"\n=== Class Distribution ===")
        for split in ['train', 'val', 'test']:
            y_key = f'y_{split}'
            y_data = dataset[y_key]
            print(f"{split.upper()}:")
            for class_name, class_idx in dataset['label_map'].items():
                count = np.sum(y_data == class_idx)
                percentage = count / len(y_data) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
                
        print(f"\n=== Data Shape ===")
        print(f"X_train: {dataset['X_train'].shape} (trials, channels, samples)")
        print(f"Data range: [{dataset['X_train'].min():.2f}, {dataset['X_train'].max():.2f}] µV")
        
        print(f"\n=== Usage Instructions ===")
        print("This dataset contains RAW signals. Apply preprocessing in your DL training:")
        print("1. Load dataset: data = np.load('dataset.npz')")
        print("2. Apply preprocessing: use unified_preprocessor.py")  
        print("3. Train model with preprocessed data")
        print("4. Use SAME preprocessing for real-time prediction")


def main():
    builder = DatasetBuilder()
    
    print("Dataset Builder - Select task type:")
    print("1. Intent Detection")
    print("2. Emotion Recognition")
    print("3. Auto-build from data/ folder")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        data_dirs = input("Enter intent data directory paths (comma-separated): ").strip().split(',')
        data_dirs = [d.strip() for d in data_dirs if d.strip()]
        
        trials = builder.load_intent_data(data_dirs)
        if not trials:
            print("No trials loaded")
            return
            
        dataset = builder.create_intent_dataset(trials)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intent_dataset_{timestamp}.npz"
        builder.save_dataset(dataset, filename)
        
    elif choice == "2":
        data_dirs = input("Enter emotion data directory paths (comma-separated): ").strip().split(',')
        data_dirs = [d.strip() for d in data_dirs if d.strip()]
        
        trials = builder.load_emotion_data(data_dirs)
        if not trials:
            print("No trials loaded")
            return
            
        dataset = builder.create_emotion_dataset(trials)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_dataset_{timestamp}.npz"
        builder.save_dataset(dataset, filename)
        
    elif choice == "3":
        # Auto-detect files in data/ folder
        intent_dir = os.path.join("data", "intent")
        emotion_dir = os.path.join("data", "emotion")
        
        intent_exists = os.path.exists(intent_dir) and os.path.exists(os.path.join(intent_dir, "session_info.json"))
        emotion_exists = os.path.exists(emotion_dir) and os.path.exists(os.path.join(emotion_dir, "session_info.json"))
        
        if not intent_exists and not emotion_exists:
            print("No data directories found. Please run data acquisition first.")
            return
            
        if intent_exists:
            intent_csv_files = [f for f in os.listdir(intent_dir) if f.startswith('intent_trial_') and f.endswith('.csv')]
            print(f"Found {len(intent_csv_files)} intent trial files in {intent_dir}")
            
            if intent_csv_files:
                print("\nBuilding intent dataset...")
                trials = builder.load_intent_data([intent_dir])
                if trials:
                    dataset = builder.create_intent_dataset(trials)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"intent_dataset_{timestamp}.npz"
                    builder.save_dataset(dataset, filename)
                    
        if emotion_exists:
            emotion_csv_files = [f for f in os.listdir(emotion_dir) if f.startswith('emotion_trial_') and f.endswith('.csv')]
            print(f"Found {len(emotion_csv_files)} emotion trial files in {emotion_dir}")
            
            if emotion_csv_files:
                print("\nBuilding emotion dataset...")
                trials = builder.load_emotion_data([emotion_dir])
                if trials:
                    dataset = builder.create_emotion_dataset(trials)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"emotion_dataset_{timestamp}.npz"
                    builder.save_dataset(dataset, filename)
        
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()

    def __init__(self, sampling_rate=250):
        self.sampling_rate = sampling_rate
        self.preprocessor = UnifiedPreprocessor(sampling_rate)
        self.channel_names = ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8']
        
    def load_intent_data(self, data_dirs):
        """
        Load intent detection data from CSV files and session info
        Args:
            data_dirs: list of data directories containing CSV files and session_info.json
        Returns:
            trials: list of trial dictionaries
        """
        all_trials = []
        
        for data_dir in data_dirs:
            # Load session info
            session_file = os.path.join(data_dir, "session_info.json")
            if not os.path.exists(session_file):
                print(f"No session_info.json found in {data_dir}")
                continue
                
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                
            # Find all intent trial CSV files
            csv_files = [f for f in os.listdir(data_dir) if f.startswith('intent_trial_') and f.endswith('.csv')]
            
            for csv_file in csv_files:
                trial_data = self._load_intent_csv_file(os.path.join(data_dir, csv_file), session_info)
                if trial_data:
                    all_trials.append(trial_data)
                    
            print(f"Loaded {len(csv_files)} intent trials from {data_dir}")
            
        return all_trials
        
    def load_emotion_data(self, data_dirs):
        """
        Load emotion recognition data from CSV files and session info
        Args:
            data_dirs: list of data directories containing CSV files and session_info.json
        Returns:
            trials: list of trial dictionaries
        """
        all_trials = []
        
        for data_dir in data_dirs:
            # Load session info
            session_file = os.path.join(data_dir, "session_info.json")
            if not os.path.exists(session_file):
                print(f"No session_info.json found in {data_dir}")
                continue
                
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                
            # Find all emotion trial CSV files
            csv_files = [f for f in os.listdir(data_dir) if f.startswith('emotion_trial_') and f.endswith('.csv')]
            
            for csv_file in csv_files:
                trial_data = self._load_emotion_csv_file(os.path.join(data_dir, csv_file), session_info)
                if trial_data:
                    all_trials.append(trial_data)
                    
            print(f"Loaded {len(csv_files)} emotion trials from {data_dir}")
            
        return all_trials
        
    def _load_intent_csv_file(self, file_path, session_info):
        """Load single intent trial from CSV file"""
        try:
            # Parse metadata from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('intent_trial_', '').replace('.csv', '').split('_')
            participant_id = parts[0]
            trial_id = int(parts[1])
            action_label = parts[2]
            
            # Load CSV data
            samples = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip metadata rows
                    if row['timestamp'].startswith('#'):
                        continue
                        
                    sample = {
                        'timestamp': float(row['timestamp']),
                        'phase_time': float(row['phase_time']),
                        'phase': row['phase'],
                        'channels': [float(row[f'ch{i}']) for i in range(1, 7)]
                    }
                    samples.append(sample)
                    
            # Extract action phase samples only
            action_samples = [s for s in samples if s['phase'] == 'action']
            
            if len(action_samples) > 0:
                return {
                    'trial_id': trial_id,
                    'label': action_label,
                    'samples': action_samples,
                    'participant_id': participant_id
                }
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
        return None
        
    def _load_emotion_csv_file(self, file_path, session_info):
        """Load single emotion trial from CSV file"""
        try:
            # Parse metadata from filename
            filename = os.path.basename(file_path)
            parts = filename.replace('emotion_trial_', '').replace('.csv', '').split('_')
            participant_id = parts[0]
            trial_id = int(parts[1])
            emotion_category = parts[2]
            rating = int(parts[3])
            
            # Load CSV data
            samples = []
            with open(file_path, 'r') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    # Skip metadata rows
                    if row['timestamp'].startswith('#'):
                        continue
                        
                    sample = {
                        'timestamp': float(row['timestamp']),
                        'recording_time': float(row['recording_time']),
                        'channels': [float(row[f'ch{i}']) for i in range(1, 7)]
                    }
                    samples.append(sample)
                    
            if len(samples) > 0:
                return {
                    'trial_id': trial_id,
                    'label': emotion_category,
                    'rating': rating,
                    'samples': samples,
                    'participant_id': participant_id
                }
                
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            
        return None
        
    def create_intent_dataset(self, trials, window_length_sec=2.0):
        """
        Create intent detection dataset with sliding windows
        Args:
            trials: list of trial dictionaries
            window_length_sec: window length for real-time prediction
        Returns:
            dataset: dictionary with X, y, and metadata
        """
        X_data = []
        y_labels = []
        
        # Label mapping
        label_map = {'baseline': 0, 'jaw_clench': 1, 'gaze_left': 2, 'gaze_right': 3}
        
        for trial in trials:
            if trial['label'] not in label_map:
                continue
                
            # Convert samples to numpy array
            channels_data = np.array([s['channels'] for s in trial['samples']])
            if channels_data.shape[0] < 10:  # Skip if too few samples
                continue
                
            # Transpose to (channels, samples)
            trial_data = channels_data.T
            
            # Apply preprocessing
            processed_data = self.preprocessor.preprocess_trial(trial_data)
            
            # Extract windows for real-time simulation
            windows = self.preprocessor.extract_windows(
                processed_data, window_length_sec, overlap_sec=0.5
            )
            
            # Add windows to dataset
            label_idx = label_map[trial['label']]
            for window in windows:
                X_data.append(window)
                y_labels.append(label_idx)
                
        X_data = np.array(X_data)
        y_labels = np.array(y_labels)
        
        # Split dataset
        X_train, X_temp, y_train, y_temp = train_test_split(
            X_data, y_labels, test_size=0.3, random_state=42, stratify=y_labels
        )
        X_val, X_test, y_val, y_test = train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'channel_names': self.channel_names,
            'sampling_rate': self.sampling_rate,
            'window_length': window_length_sec,
            'task_type': 'intent_detection',
            'class_names': list(label_map.keys()),
            'label_map': label_map
        }
        
        return dataset
        
    def create_emotion_dataset(self, trials, window_length_sec=2.0):
        """
        Create emotion recognition dataset
        Args:
            trials: list of trial dictionaries  
            window_length_sec: window length for analysis
        Returns:
            dataset: dictionary with X, y, and metadata
        """
        X_data = []
        y_labels = []
        ratings = []
        
        # Label mapping
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        
        for trial in trials:
            if trial['label'] not in label_map:
                continue
                
            # Convert samples to numpy array
            channels_data = np.array([s['channels'] for s in trial['samples']])
            if channels_data.shape[0] < 10:  # Skip if too few samples
                continue
                
            # Transpose to (channels, samples)
            trial_data = channels_data.T
            
            # Apply preprocessing
            processed_data = self.preprocessor.preprocess_trial(trial_data)
            
            # Extract windows
            windows = self.preprocessor.extract_windows(
                processed_data, window_length_sec, overlap_sec=1.0
            )
            
            # Add windows to dataset
            label_idx = label_map[trial['label']]
            for window in windows:
                X_data.append(window)
                y_labels.append(label_idx)
                ratings.append(trial['rating'])
                
        X_data = np.array(X_data)
        y_labels = np.array(y_labels)
        ratings = np.array(ratings)
        
        # Split dataset
        X_train, X_temp, y_train, y_temp, r_train, r_temp = train_test_split(
            X_data, y_labels, ratings, test_size=0.3, random_state=42, stratify=y_labels
        )
        X_val, X_test, y_val, y_test, r_val, r_test = train_test_split(
            X_temp, y_temp, r_temp, test_size=0.5, random_state=42, stratify=y_temp
        )
        
        dataset = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'ratings_train': r_train,
            'ratings_val': r_val,
            'ratings_test': r_test,
            'channel_names': self.channel_names,
            'sampling_rate': self.sampling_rate,
            'window_length': window_length_sec,
            'task_type': 'emotion_recognition',
            'class_names': list(label_map.keys()),
            'label_map': label_map
        }
        
        return dataset
        
    def save_dataset(self, dataset, filename):
        """
        Save dataset to NPZ format
        Args:
            dataset: dataset dictionary
            filename: output filename
        """
        np.savez_compressed(filename, **dataset)
        print(f"Dataset saved: {filename}")
        
        # Print dataset statistics
        self._print_dataset_stats(dataset)
        
    def _print_dataset_stats(self, dataset):
        """Print dataset statistics"""
        print(f"\n=== Dataset Statistics ===")
        print(f"Task: {dataset['task_type']}")
        print(f"Window length: {dataset['window_length']}s")
        print(f"Channels: {len(dataset['channel_names'])} {dataset['channel_names']}")
        print(f"Sampling rate: {dataset['sampling_rate']} Hz")
        print(f"Classes: {dataset['class_names']}")
        
        print(f"\n=== Data Splits ===")
        print(f"Training: {len(dataset['X_train'])} samples")
        print(f"Validation: {len(dataset['X_val'])} samples") 
        print(f"Test: {len(dataset['X_test'])} samples")
        
        print(f"\n=== Class Distribution ===")
        for split in ['train', 'val', 'test']:
            y_key = f'y_{split}'
            y_data = dataset[y_key]
            print(f"{split.upper()}:")
            for class_name, class_idx in dataset['label_map'].items():
                count = np.sum(y_data == class_idx)
                percentage = count / len(y_data) * 100
                print(f"  {class_name}: {count} ({percentage:.1f}%)")
                
        print(f"\n=== Data Shape ===")
        print(f"X_train: {dataset['X_train'].shape}")
        print(f"Data range: [{dataset['X_train'].min():.2f}, {dataset['X_train'].max():.2f}]")


def main():
    builder = DatasetBuilder()
    
    print("Dataset Builder - Select task type:")
    print("1. Intent Detection")
    print("2. Emotion Recognition")
    print("3. Load from data/ folder automatically")
    
    choice = input("Enter choice (1-3): ").strip()
    
    if choice == "1":
        data_files = input("Enter intent data file paths (comma-separated): ").strip().split(',')
        data_files = [f.strip() for f in data_files if f.strip()]
        
        trials = builder.load_intent_data(data_files)
        if not trials:
            print("No trials loaded")
            return
            
        dataset = builder.create_intent_dataset(trials)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"intent_dataset_{timestamp}.npz"
        builder.save_dataset(dataset, filename)
        
    elif choice == "2":
        data_files = input("Enter emotion data file paths (comma-separated): ").strip().split(',')
        data_files = [f.strip() for f in data_files if f.strip()]
        
        trials = builder.load_emotion_data(data_files)
        if not trials:
            print("No trials loaded")
            return
            
        dataset = builder.create_emotion_dataset(trials)
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"emotion_dataset_{timestamp}.npz"
        builder.save_dataset(dataset, filename)
        
    elif choice == "3":
        # Auto-detect files in data/ folder
        intent_dir = os.path.join("data", "intent")
        emotion_dir = os.path.join("data", "emotion")
        
        intent_exists = os.path.exists(intent_dir) and os.path.exists(os.path.join(intent_dir, "session_info.json"))
        emotion_exists = os.path.exists(emotion_dir) and os.path.exists(os.path.join(emotion_dir, "session_info.json"))
        
        if not intent_exists and not emotion_exists:
            print("No data directories found. Please run data acquisition first.")
            return
            
        if intent_exists:
            intent_csv_files = [f for f in os.listdir(intent_dir) if f.startswith('intent_trial_') and f.endswith('.csv')]
            print(f"Found {len(intent_csv_files)} intent trial files in {intent_dir}")
            
            if intent_csv_files:
                print("\nBuilding intent dataset...")
                trials = builder.load_intent_data([intent_dir])
                if trials:
                    dataset = builder.create_intent_dataset(trials)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"intent_dataset_{timestamp}.npz"
                    builder.save_dataset(dataset, filename)
                    
        if emotion_exists:
            emotion_csv_files = [f for f in os.listdir(emotion_dir) if f.startswith('emotion_trial_') and f.endswith('.csv')]
            print(f"Found {len(emotion_csv_files)} emotion trial files in {emotion_dir}")
            
            if emotion_csv_files:
                print("\nBuilding emotion dataset...")
                trials = builder.load_emotion_data([emotion_dir])
                if trials:
                    dataset = builder.create_emotion_dataset(trials)
                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"emotion_dataset_{timestamp}.npz"
                    builder.save_dataset(dataset, filename)
        
    else:
        print("Invalid choice")


if __name__ == "__main__":
    main()
