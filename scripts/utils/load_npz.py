# utils/load_npz.py

import numpy as np
import os
from datetime import datetime
import glob

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..', 'data'))

def load_dataset(filename=None, task_type='auto'):
    """
    Load dataset from NPZ file - auto-detect generated datasets or specify filename
    
    Parameters:
    -----------
    filename : str, optional
        Path to specific NPZ file. If None, auto-detect latest generated datasets
    task_type : str
        'auto', 'intent', 'emotion', or specific filename
        
    Returns:
    --------
    dataset : dict
        Complete dataset dictionary with all metadata
    """
    data_dir = DATA_DIR
    
    if filename is None:
        # Auto-detect generated datasets
        if task_type == 'auto':
            # Look for both intent and emotion datasets and let user choose
            
            intent_files = glob.glob(os.path.join(data_dir, 'intent_dataset_*.npz'))
            emotion_files = glob.glob(os.path.join(data_dir, 'emotion_dataset_*.npz'))            
            
            print("Available datasets:")
            all_files = []
            if intent_files:
                latest_intent = max(intent_files, key=os.path.getmtime)
                print(f"  1. Intent Detection: {latest_intent}")
                all_files.append(('intent', latest_intent))
            if emotion_files:
                latest_emotion = max(emotion_files, key=os.path.getmtime) 
                print(f"  2. Emotion Recognition: {latest_emotion}")
                all_files.append(('emotion', latest_emotion))
            
            if len(all_files) == 1:
                # Only one type available, use it
                task_type, filename = all_files[0]
            else:
                # Multiple available, use intent by default (or could prompt user)
                task_type, filename = all_files[0]  # Default to first (intent)
                print(f"Using: {filename}")
        
        elif task_type == 'intent':
            intent_files = glob.glob(os.path.join(data_dir, 'intent_dataset_*.npz'))
            if not intent_files:
                raise FileNotFoundError("No intent dataset found")
            filename = max(intent_files, key=os.path.getmtime)
            
        elif task_type == 'emotion':
            emotion_files = glob.glob(os.path.join(data_dir, 'emotion_dataset_*.npz'))
            if not emotion_files:
                raise FileNotFoundError("No emotion dataset found")
            filename = max(emotion_files, key=os.path.getmtime)
    
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {filename}")
    
    # Load data
    data = np.load(filename, allow_pickle=True)
    
    # Convert numpy arrays and objects to proper format
    dataset = {}
    for key in data.files:
        if data[key].ndim == 0 and data[key].dtype == object:
            # Handle scalar objects (like dicts)
            dataset[key] = data[key].item()
        else:
            # Handle regular arrays
            dataset[key] = data[key]
    
    # Verify required keys exist
    required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
                     'channel_names', 'sampling_rate', 'task_type']
    missing_keys = [key for key in required_keys if key not in dataset]
    if missing_keys:
        print(f"Warning: Missing keys {missing_keys}, attempting backwards compatibility...")
        
        # Add missing keys for backwards compatibility
        if 'task_type' not in dataset:
            if 'intent' in filename:
                dataset['task_type'] = 'intent_detection'
            elif 'emotion' in filename:
                dataset['task_type'] = 'emotion_recognition'
            else:
                dataset['task_type'] = 'unknown'
        
        if 'trial_duration' not in dataset and 'sampling_rate' in dataset:
            # Estimate from data shape
            n_samples = dataset['X_train'].shape[2]
            fs = dataset['sampling_rate']
            dataset['trial_duration'] = float(n_samples / fs)
    
    # Ensure channel_names is a list
    if 'channel_names' in dataset:
        if isinstance(dataset['channel_names'], np.ndarray):
            dataset['channel_names'] = [str(ch) for ch in dataset['channel_names']]
    
    # Print dataset info
    print(f"  -> Dataset loaded successfully")
    print(f"  -> Task: {dataset.get('task_type', 'Unknown')}")
    print(f"  -> Channels: {dataset.get('channel_names', 'Unknown')}")
    print(f"  -> Sampling rate: {dataset.get('sampling_rate', 'Unknown')} Hz")
    
    if 'trial_duration' in dataset:
        print(f"  -> Trial duration: {dataset['trial_duration']:.1f} s")
    
    print(f"  -> Train: {dataset['X_train'].shape[0]} samples")
    print(f"  -> Val: {dataset['X_val'].shape[0]} samples")
    print(f"  -> Test: {dataset['X_test'].shape[0]} samples")
    print(f"  -> Data shape: {dataset['X_train'].shape}")
    
    # Verify data consistency
    n_channels, n_samples = dataset['X_train'].shape[1], dataset['X_train'].shape[2]
    if 'sampling_rate' in dataset and 'trial_duration' in dataset:
        expected_samples = int(dataset['sampling_rate'] * dataset['trial_duration'])
        if abs(n_samples - expected_samples) > 10:  # Allow small tolerance
            print(f"  -> Warning: Expected ~{expected_samples} samples, got {n_samples}")
    
    return dataset

def print_dataset_stats(dataset):
    """
    Print detailed statistics about the loaded dataset.
    
    Parameters:
    -----------
    dataset : dict
        Dataset returned by load_dataset()
    """
    print(f"\n=== Dataset Statistics ===")
    print(f"Task: {dataset.get('task_type', 'Unknown')}")
    print(f"Channels: {dataset.get('channel_names', 'Unknown')}")
    print(f"Sampling rate: {dataset.get('sampling_rate', 'Unknown')} Hz")
    
    if 'class_names' in dataset:
        print(f"Classes: {list(dataset['class_names'])}")
    elif 'label_map' in dataset:
        print(f"Classes: {list(dataset['label_map'].keys())}")
    
    for split in ['train', 'val', 'test']:
        X_key = f'X_{split}'
        y_key = f'y_{split}'
        
        if X_key not in dataset or y_key not in dataset:
            continue
            
        X_data = dataset[X_key]
        y_data = dataset[y_key]
        
        print(f"\n{split.upper()} SET:")
        print(f"  Shape: {X_data.shape}")
        print(f"  Data range: [{X_data.min():.2f}, {X_data.max():.2f}] μV")
        
        # Class distribution
        unique_labels, counts = np.unique(y_data, return_counts=True)
        print(f"  Label distribution: {dict(zip(unique_labels, counts))}")
        
        # Channel-wise statistics
        if 'channel_names' in dataset and len(dataset['channel_names']) == X_data.shape[1]:
            print(f"  Channel statistics:")
            for ch_idx, ch_name in enumerate(dataset['channel_names']):
                ch_data = X_data[:, ch_idx, :]
                print(f"    {ch_name}: μ={ch_data.mean():.1f}μV, σ={ch_data.std():.1f}μV")

def get_dataset_info(dataset):
    """
    Get basic dataset information for model configuration
    
    Returns:
    --------
    info : dict
        Basic dataset parameters for model initialization
    """
    info = {
        'task_type': dataset.get('task_type', 'unknown'),
        'n_channels': dataset['X_train'].shape[1],
        'n_samples': dataset['X_train'].shape[2], 
        'sampling_rate': dataset.get('sampling_rate', 250),
        'n_classes': len(np.unique(dataset['y_train'])),
        'channel_names': dataset.get('channel_names', [f'Ch{i}' for i in range(dataset['X_train'].shape[1])]),
    }
    
    # Add class information if available
    if 'class_names' in dataset:
        info['class_names'] = list(dataset['class_names'])
    elif 'label_map' in dataset:
        info['class_names'] = list(dataset['label_map'].keys())
    else:
        # Generate default names based on task and number of classes
        if info['task_type'] == 'intent_detection' and info['n_classes'] == 4:
            info['class_names'] = ['baseline', 'jaw_clench', 'gaze_left', 'gaze_right']
        elif info['task_type'] == 'emotion_recognition' and info['n_classes'] == 3:
            info['class_names'] = ['negative', 'neutral', 'positive']
        else:
            info['class_names'] = [f'Class_{i}' for i in range(info['n_classes'])]
    
    return info