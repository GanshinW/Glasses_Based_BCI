# acquire/dataset_builder.py
"""
Dataset Builder - Auto-build from data/ folder
"""

import json
import csv
import numpy as np
import os
from datetime import datetime

def manual_train_test_split(X, y, test_size=0.15, random_state=42):
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

        if n <= 2:
            n_cls_test = 0
        else:
            n_cls_test = max(1, int(round(n * test_size)))
            if n - n_cls_test < 1:
                n_cls_test = n - 1

        test_indices.extend(cls_indices[:n_cls_test])
        train_indices.extend(cls_indices[n_cls_test:])

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
        self.channel_names = ['F7', 'FT7', 'T7', 'F8', 'FT8', 'T8']
        
    def load_intent_data(self, data_dirs):
        all_trials = []
        failed_files = []
        
        for data_dir in data_dirs:
            # keep your original session check INSIDE loader
            session_file = os.path.join(data_dir, "session_info.json")
            if not os.path.exists(session_file):
                continue
                
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                
            csv_files = [f for f in os.listdir(data_dir) if f.startswith('intent_trial_') and f.endswith('.csv')]
            
            for csv_file in csv_files:
                trial_data = self._load_intent_csv_file(os.path.join(data_dir, csv_file), session_info)
                if trial_data:
                    all_trials.append(trial_data)
                else:
                    failed_files.append(csv_file)
        
        if failed_files:
            print(f"\nFAILED TO LOAD {len(failed_files)} FILES:")
            for filename in failed_files:
                print(f"  - {filename}")
            
        return all_trials
        
    def load_emotion_data(self, data_dirs):
        all_trials = []
        
        for data_dir in data_dirs:
            session_file = os.path.join(data_dir, "session_info.json")
            if not os.path.exists(session_file):
                continue
                
            with open(session_file, 'r') as f:
                session_info = json.load(f)
                
            csv_files = [f for f in os.listdir(data_dir) if f.startswith('emotion_trial_') and f.endswith('.csv')]
            
            for csv_file in csv_files:
                trial_data = self._load_emotion_csv_file(os.path.join(data_dir, csv_file), session_info)
                if trial_data:
                    all_trials.append(trial_data)
            
        return all_trials
        
    def _load_intent_csv_file(self, file_path, session_info):
        try:
            filename = os.path.basename(file_path)
            core_name = filename.replace('intent_trial_', '').replace('.csv', '')
            parts = core_name.split('_')
            if len(parts) < 3:
                return None
                
            participant_id = parts[0]
            trial_id = int(parts[1])
            
            if len(parts) >= 4 and parts[2] in ['jaw', 'gaze']:
                if parts[2] == 'jaw' and parts[3] == 'clench':
                    action_label = 'jaw_clench'
                elif parts[2] == 'gaze' and parts[3] in ['left', 'right']:
                    action_label = f'gaze_{parts[3]}'
                else:
                    action_label = parts[2]
            else:
                action_label = parts[2]
            
            samples = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                data_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('timestamp,') or 'timestamp' in line:
                        data_start = i
                        break
                import io
                data_content = ''.join(lines[data_start:])
                reader = csv.DictReader(io.StringIO(data_content))
                for row in reader:
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
                    except (ValueError, KeyError):
                        continue
            
            # action_samples = [s for s in samples if s['phase'] == 'action']
            # if len(action_samples) > 0:
            #     return {
            #         'trial_id': trial_id,
            #         'label': action_label,
            #         'samples': action_samples,
            #         'participant_id': participant_id
            #     }
            if len(samples) > 0:
                return {
                    'trial_id': trial_id,
                    'label': action_label,
                    'samples': samples,            # <-- all phases
                    'participant_id': participant_id
                }
        except Exception:
            pass
        return None
        
    def _load_emotion_csv_file(self, file_path, session_info):
        try:
            filename = os.path.basename(file_path)
            parts = filename.replace('emotion_trial_', '').replace('.csv', '').split('_')
            participant_id = parts[0]
            trial_id = int(parts[1])
            emotion_category = parts[2]
            rating = int(parts[3])
            
            samples = []
            with open(file_path, 'r') as f:
                lines = f.readlines()
                data_start = 0
                for i, line in enumerate(lines):
                    if line.startswith('timestamp,') or 'timestamp' in line:
                        data_start = i
                        break
                import io
                data_content = ''.join(lines[data_start:])
                reader = csv.DictReader(io.StringIO(data_content))
                for row in reader:
                    if row['timestamp'].startswith('#') or not row['timestamp'].strip():
                        continue
                    try:
                        sample = {
                            'timestamp': float(row['timestamp']),
                            'recording_time': float(row['recording_time']),
                            'channels': [float(row[f'ch{i}']) for i in range(1, 7)]
                        }
                        samples.append(sample)
                    except (ValueError, KeyError):
                        continue
                    
            if len(samples) > 0:
                return {
                    'trial_id': trial_id,
                    'label': emotion_category,
                    'rating': rating,
                    'samples': samples,
                    'participant_id': participant_id
                }
        except Exception:
            pass
        return None
        
    def create_intent_dataset(self, trials, window_length_sec=8.0):
        X_data = []
        y_labels = []
        label_map = {'baseline': 0, 'jaw_clench': 1, 'gaze_left': 2, 'gaze_right': 3}
        target_samples = int(window_length_sec * self.sampling_rate)  # 8s -> 2000 @ 250Hz

        for trial in trials:
            if trial['label'] not in label_map:
                continue
            channels_data = np.array([s['channels'] for s in trial['samples']], dtype=float)  # (N, 6)
            if channels_data.shape[0] < 2:
                continue
            trial_data = channels_data.T  # (6, T)
            n_channels, n_samples = trial_data.shape

            # one window per trial, no sliding
            if n_samples < target_samples:
                padded = np.zeros((n_channels, target_samples), dtype=float)  # zero-pad
                padded[:, :n_samples] = trial_data
                X_data.append(padded)
            else:
                window = trial_data[:, -target_samples:]  # last 8s
                X_data.append(window)
            y_labels.append(label_map[trial['label']])

        if len(X_data) == 0:
            return None

        X_data = np.array(X_data)
        y_labels = np.array(y_labels)

        X_train, X_temp, y_train, y_temp = manual_train_test_split(
            X_data, y_labels, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = manual_train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        dataset = {
            'X_train': X_train, 'y_train': y_train,
            'X_val': X_val,     'y_val': y_val,
            'X_test': X_test,   'y_test': y_test,
            'channel_names': self.channel_names,
            'sampling_rate': self.sampling_rate,
            'window_length': window_length_sec,  # 8.0
            'task_type': 'intent_detection',
            'class_names': list(label_map.keys()),
            'label_map': label_map
        }
        return dataset

        
    def create_emotion_dataset(self, trials, window_length_sec=10.0):
        X_data = []
        y_labels = []
        ratings = []
        label_map = {'negative': 0, 'neutral': 1, 'positive': 2}
        target_samples = int(window_length_sec * self.sampling_rate)

        for trial in trials:
            if trial['label'] not in label_map:
                continue
            channels_data = np.array([s['channels'] for s in trial['samples']], dtype=float)  # (N, 6)
            if channels_data.shape[0] < 2:
                continue
            trial_data = channels_data.T  # (6, T)
            n_channels, n_samples = trial_data.shape

            # one window per trial, no sliding
            if n_samples < target_samples:
                padded = np.zeros((n_channels, target_samples), dtype=float)
                padded[:, :n_samples] = trial_data
                X_data.append(padded)
            else:
                window = trial_data[:, -target_samples:]  # last 10s
                X_data.append(window)
            y_labels.append(label_map[trial['label']])
            ratings.append(trial.get('rating', -1))

        if len(X_data) == 0:
            return None

        X_data = np.array(X_data)
        y_labels = np.array(y_labels)
        ratings = np.array(ratings)

        # keep your original split logic
        X_train, X_temp, y_train, y_temp = manual_train_test_split(
            X_data, y_labels, test_size=0.3, random_state=42
        )
        X_val, X_test, y_val, y_test = manual_train_test_split(
            X_temp, y_temp, test_size=0.5, random_state=42
        )

        # ratings random slice (与你原脚本一致；如果要严格对齐索引，另行可改)
        np.random.seed(42)
        n = len(ratings)
        idx = np.arange(n); np.random.shuffle(idx)
        r_train = ratings[idx[:len(y_train)]]
        r_val = ratings[idx[len(y_train):len(y_train)+len(y_val)]]
        r_test = ratings[idx[len(y_train)+len(y_val):len(y_train)+len(y_val)+len(y_test)]]

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
            'window_length': window_length_sec,  # 10.0
            'task_type': 'emotion_recognition',
            'class_names': list(label_map.keys()),
            'label_map': label_map
        }
        return dataset
        
    def save_dataset(self, dataset, filename):
        # ensure dir exists + print absolute path
        os.makedirs(os.path.dirname(filename), exist_ok=True)  # create data/ if missing
        np.savez_compressed(filename, **dataset)
        print("[saved]", os.path.abspath(filename))  # print absolute path


def main():
    builder = DatasetBuilder()
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(script_dir))
    data_dir = os.path.join(project_root, "data")
    
    intent_dir = os.path.join(data_dir, "intent")
    emotion_dir = os.path.join(data_dir, "emotion")
    
    # change: don't require session_info.json to decide existence
    intent_exists = os.path.isdir(intent_dir) and any(
        f.startswith('intent_trial_') and f.endswith('.csv') for f in os.listdir(intent_dir)
    )
    emotion_exists = os.path.isdir(emotion_dir) and any(
        f.startswith('emotion_trial_') and f.endswith('.csv') for f in os.listdir(emotion_dir)
    )
    
    if not intent_exists and not emotion_exists:
        print("No data directories found. Expecting CSVs under:", os.path.abspath(data_dir))
        return
        
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    if intent_exists:
        intent_csv_files = [f for f in os.listdir(intent_dir) if f.startswith('intent_trial_') and f.endswith('.csv')]
        print(f"\nFound {len(intent_csv_files)} CSV files")
        if intent_csv_files:
            trials = builder.load_intent_data([intent_dir])
            print(f"Successfully loaded {len(trials)} trials")
            label_counts = {}
            for trial in trials:
                label = trial['label']
                label_counts[label] = label_counts.get(label, 0) + 1
            print(f"Label distribution: {label_counts}")
            if trials:
                dataset = builder.create_intent_dataset(trials)
                if dataset:
                    filename = os.path.join(data_dir, f"intent_dataset_{timestamp}.npz")
                    builder.save_dataset(dataset, filename)
                else:
                    print("Intent dataset creation failed - returned None")
                
    if emotion_exists:
        emotion_csv_files = [f for f in os.listdir(emotion_dir) if f.startswith('emotion_trial_') and f.endswith('.csv')]
        if emotion_csv_files:
            trials = builder.load_emotion_data([emotion_dir])
            if trials:
                label_counts = {}
                for trial in trials:
                    label = trial['label']
                    label_counts[label] = label_counts.get(label, 0) + 1
                print(f"Emotion label distribution: {label_counts}")
                dataset = builder.create_emotion_dataset(trials)
                if dataset:
                    filename = os.path.join(data_dir, f"emotion_dataset_{timestamp}.npz")
                    builder.save_dataset(dataset, filename)
                else:
                    print("Emotion dataset creation failed - returned None")


if __name__ == "__main__":
    main()
