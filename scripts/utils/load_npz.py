# utils/load_npz.py

import numpy as np
import os
from datetime import datetime

def load_dataset(filename='chewing_dataset.npz'):
    """
    Load the simulated chewing dataset from a .npz file.

    Parameters:
    -----------
    filename : str
        Path to the .npz file containing the dataset.

    Returns:
    --------
    dataset : dict
        {
            'X_train': np.ndarray,  # shape (n_train, n_channels, n_samples)
            'y_train': np.ndarray,  # shape (n_train,)
            'X_val':   np.ndarray,  # shape (n_val,   n_channels, n_samples)
            'y_val':   np.ndarray,  # shape (n_val,)
            'X_test':  np.ndarray,  # shape (n_test,  n_channels, n_samples)
            'y_test':  np.ndarray,  # shape (n_test,),
            'channel_names': list[str],
            'sampling_rate': int,
            'trial_duration': float
        }
    """
    # Check if file exists
    if not os.path.exists(filename):
        raise FileNotFoundError(f"Dataset file not found: {filename}")
    
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading dataset from {filename}")
    
    # Load data
    data = np.load(filename)
    
    # Verify required keys exist
    required_keys = ['X_train', 'y_train', 'X_val', 'y_val', 'X_test', 'y_test',
                     'channel_names', 'sampling_rate', 'trial_duration']
    for key in required_keys:
        if key not in data.files:
            raise KeyError(f"Missing required key in NPZ file: {key}")
    
    # Extract dataset
    dataset = {
        'X_train':       data['X_train'],
        'y_train':       data['y_train'],
        'X_val':         data['X_val'],
        'y_val':         data['y_val'],
        'X_test':        data['X_test'],
        'y_test':        data['y_test'],
        'channel_names': list(data['channel_names']),
        'sampling_rate': int(data['sampling_rate']),
        'trial_duration': float(data['trial_duration'])
    }
    
    # Print dataset info
    print(f"  -> Dataset loaded successfully")
    print(f"  -> Channels: {dataset['channel_names']}")
    print(f"  -> Sampling rate: {dataset['sampling_rate']} Hz")
    print(f"  -> Trial duration: {dataset['trial_duration']} s")
    print(f"  -> Train: {dataset['X_train'].shape[0]} samples")
    print(f"  -> Val: {dataset['X_val'].shape[0]} samples") 
    print(f"  -> Test: {dataset['X_test'].shape[0]} samples")
    print(f"  -> Data shape: {dataset['X_train'].shape}")
    
    # Verify data shapes are consistent
    n_channels, n_samples = dataset['X_train'].shape[1], dataset['X_train'].shape[2]
    expected_samples = int(dataset['sampling_rate'] * dataset['trial_duration'])
    
    if n_samples != expected_samples:
        print(f"  -> Warning: Expected {expected_samples} samples, got {n_samples}")
    
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
    
    for split in ['train', 'val', 'test']:
        X_key = f'X_{split}'
        y_key = f'y_{split}'
        
        X_data = dataset[X_key]
        y_data = dataset[y_key]
        
        print(f"\n{split.upper()} SET:")
        print(f"  Shape: {X_data.shape}")
        print(f"  Data range: [{X_data.min():.2f}, {X_data.max():.2f}]")
        print(f"  Labels: {np.bincount(y_data)} (0=baseline, 1=chewing)")
        
        # Channel-wise statistics
        for ch_idx, ch_name in enumerate(dataset['channel_names']):
            ch_data = X_data[:, ch_idx, :]
            print(f"  {ch_name}: mean={ch_data.mean():.2f}, std={ch_data.std():.2f}")