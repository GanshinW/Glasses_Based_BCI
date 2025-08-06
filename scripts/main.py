# main.py

import os
import re
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn, optim
from datetime import datetime

from utils.load_bdf import extract_trials_from_bdf
from utils.load_labels import load_labels
from utils.preprocess import preprocess_batch
from models.cnn_model import EmotionCNN
from utils.load_npz import load_dataset
from utils.features import (
    extract_time_domain,
    extract_freq_domain,
    extract_timefreq_images
)

def gather_all_data(eeg_folder, label_file, trial_duration_s):
    """
    Traverse the EEG folder, extract trials from each .bdf/.mrk pair, 
    and collect the corresponding labels from the common label file.
    Returns:
        X_all: numpy array shape (N_total, n_channels, n_samples)
        y_all: numpy array shape (N_total,)
    """
    all_trials = []
    all_labels = []
    ref_channels = None

    # Pattern to match “Part<subject>_IAPS_SES<session>_EEG_fNIRS_… .bdf”
    pattern = re.compile(r"Part(\d+)_IAPS_SES(\d+)_EEG_fNIRS_.*\.bdf$", re.IGNORECASE)

    for fname in os.listdir(eeg_folder):
        if not fname.lower().endswith(".bdf"):
            continue

        match = pattern.match(fname)
        if not match:
            continue

        subject_idx = int(match.group(1))
        session_idx = int(match.group(2))

        # Only load Part1_SES1 for now
        if subject_idx != 1 or session_idx != 1:
            continue

        bdf_path = os.path.join(eeg_folder, fname)
        mrk_path = bdf_path.replace(".bdf", ".bdf.mrk")
        if not os.path.exists(mrk_path):
            continue

        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Loading Subject {subject_idx}, Session {session_idx}")
        trials = extract_trials_from_bdf(bdf_path, mrk_path, trial_duration_s=trial_duration_s)
        n_trials, n_channels, n_samples = trials.shape
        print(f"  -> Extracted {n_trials} trials (channels={n_channels}, samples={n_samples})")

        labels_all = load_labels(label_file, subject_idx=(session_idx - 1))
        if len(labels_all) < n_trials:
            raise RuntimeError(f"Need {n_trials} labels but found {len(labels_all)} in {label_file}")
        labels = labels_all[:n_trials]
        print(f"  -> Loaded {len(labels)} labels for session {session_idx}")

        all_trials.append(trials)
        all_labels.extend(labels)

    if not all_trials:
        raise RuntimeError("No trials were loaded. Check Part1_SES1 files exist.")

    X_all = np.concatenate(all_trials, axis=0)
    y_all = np.array(all_labels, dtype=np.int64)
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Total trials collected: {X_all.shape[0]}")
    return X_all, y_all


def build_dataset(X, y, batch_size):
    """
    Given raw EEG data (X) and labels (y), preprocess, wrap into PyTorch Dataset,
    and split into train/val DataLoaders.
    Returns:
        train_loader, val_loader, n_train, n_val, n_channels, n_samples
    """
    fs = 1024
    print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Preprocessing data: band-pass + z-score")
    X_processed = preprocess_batch(X, fs=fs)

    X_tensor = torch.tensor(X_processed, dtype=torch.float32).unsqueeze(1)
    y_tensor = torch.tensor(y, dtype=torch.long)

    assert X_tensor.shape[0] == y_tensor.shape[0], "Mismatch between X and y lengths."

    dataset = TensorDataset(X_tensor, y_tensor)
    N_total = X_tensor.shape[0]
    n_train = int(0.8 * N_total)
    n_val = N_total - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    _, _, n_channels, n_samples = X_tensor.shape
    return train_loader, val_loader, n_train, n_val, n_channels, n_samples


def train_model(train_loader, val_loader, n_train, n_val, n_channels, n_samples, num_epochs, lr):
    """
    Build the CNN model, train for num_epochs, and return the trained model.
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(n_channels=n_channels, n_samples=n_samples, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for X_batch, y_batch in train_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            optimizer.zero_grad()
            logits = model(X_batch)
            loss = criterion(logits, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * X_batch.size(0)

        avg_loss = total_loss / n_train
        print(f"[{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}] Epoch {epoch}/{num_epochs}  Train Loss: {avg_loss:.4f}")

        model.eval()
        correct = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch, y_batch = X_batch.to(device), y_batch.to(device)
                logits = model(X_batch)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == y_batch).sum().item()

        val_acc = correct / n_val
        print(f"               Val Acc: {val_acc:.4f}")

    return model


def main():
 # 1. Load raw NPZ
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_file = os.path.join(project_root, 'simulation', 'chewing_dataset.npz')
    ds = load_dataset(dataset_file)
    fs  = ds['sampling_rate']
    dur = ds['trial_duration']

    # 2. Time‐domain (filtered waveforms)
    X_time_train = extract_time_domain(ds['X_train'], fs)
    X_time_val   = extract_time_domain(ds['X_val'],   fs)
    X_time_test  = extract_time_domain(ds['X_test'],  fs)

    # 3. Frequency‐domain (theta/alpha/beta band‐power)
    bands = [(4,8), (8,13), (13,30)]
    X_freq_train = extract_freq_domain(X_time_train, fs, bands)
    X_freq_val   = extract_freq_domain(X_time_val,   fs, bands)
    X_freq_test  = extract_freq_domain(X_time_test,  fs, bands)

    # 4. Time‐frequency images (spectrogram of channel 0)
    X_img_train = extract_timefreq_images(X_time_train, fs)
    X_img_val   = extract_timefreq_images(X_time_val,   fs)
    X_img_test  = extract_timefreq_images(X_time_test,  fs)

    # 5. Now feed (X_time_*, X_freq_*, X_img_*) plus y_* into your Dataset / DataLoader
    #    and then into your three‐branch CNN+LSTM / 1D‐CNN / 2D‐CNN model.


if __name__ == "__main__":
    main()
