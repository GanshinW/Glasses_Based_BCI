# main.py

import os
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset, random_split
from torch import nn, optim

from utils.load_bdf import extract_trials_from_bdf
from utils.load_labels import load_labels
from utils.preprocess import preprocess_batch
from models.cnn_model import EmotionCNN

def main():
    # --- 1. Configuration: file paths and hyperparameters ---
    data_dir = "data"
    bdf_file = os.path.join(data_dir, "Part1_IAPS_SES1_EEG_fNIRS_03082006.bdf")
    mrk_file = os.path.join(data_dir, "Part1_IAPS_SES1_EEG_FNIRS_03082006.bdf.mrk")
    label_file = os.path.join(data_dir, "IAPS_Classes_EEG_FNIRS.txt")
    subject_idx = 0         # Use column 0 for Part1’s labels
    trial_duration_s = 12.5 # Trial length in seconds
    batch_size = 8
    num_epochs = 10
    learning_rate = 1e-3

    # --- 2. Load and split EEG trials from BDF + MRK ---
    print("Loading and splitting EEG trials from BDF + MRK ...")
    trials = extract_trials_from_bdf(bdf_file, mrk_file, trial_duration_s=trial_duration_s)
    # trials shape: (n_trials, n_channels, n_samples)
    n_trials, n_channels, n_samples = trials.shape
    print(f"Loaded {n_trials} trials each with {n_channels} channels × {n_samples} samples")

    # --- 3. Load labels from IAPS_Classes ... ---
    print("Loading emotion labels ...")
    all_labels = load_labels(label_file, subject_idx=subject_idx)
    # Use as many labels as trials
    labels = all_labels[:n_trials]
    labels = np.array(labels, dtype=np.int64)

    # --- 4. Preprocess each trial: band-pass filter + z-score normalize ---
    print("Preprocessing trials: band-pass filter + z-score normalization ...")
    X = preprocess_batch(trials, fs=1024)
    # Convert to torch.Tensor and add channel dimension for CNN
    X_tensor = torch.tensor(X, dtype=torch.float32).unsqueeze(1)  # shape: (n_trials, 1, n_channels, n_samples)
    y_tensor = torch.tensor(labels, dtype=torch.long)            # shape: (n_trials,)
    print("X_tensor.shape =", X_tensor.shape)
    print("y_tensor.shape =", y_tensor.shape)
    
    # --- 5. Create Dataset and DataLoader (80% train / 20% val split) ---
    dataset = TensorDataset(X_tensor, y_tensor)
    n_train = int(0.8 * n_trials)
    n_val = n_trials - n_train
    train_ds, val_ds = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)
    print(f"Training on {n_train} samples, validating on {n_val} samples")

    # --- 6. Initialize model, loss function, optimizer ---
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = EmotionCNN(n_channels=n_channels, n_samples=n_samples, num_classes=3).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    # --- 7. Training loop ---
    for epoch in range(1, num_epochs + 1):
        model.train()
        total_loss = 0.0
        for batch_X, batch_y in train_loader:
            batch_X, batch_y = batch_X.to(device), batch_y.to(device)
            optimizer.zero_grad()
            logits = model(batch_X)                # (batch_size, 3)
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item() * batch_X.size(0)

        avg_loss = total_loss / n_train
        print(f"Epoch {epoch:02d}/{num_epochs}  Training Loss: {avg_loss:.4f}")

        # --- 8. Validation step ---
        model.eval()
        correct = 0
        total = 0
        with torch.no_grad():
            for batch_X, batch_y in val_loader:
                batch_X, batch_y = batch_X.to(device), batch_y.to(device)
                logits = model(batch_X)
                preds = torch.argmax(logits, dim=1)
                correct += (preds == batch_y).sum().item()
                total += batch_y.size(0)
        val_acc = correct / total
        print(f"                Validation Accuracy: {val_acc:.4f}")

    # --- 9. Save trained model weights ---
    torch.save(model.state_dict(), "emotion_cnn.pth")
    print("Model weights saved to emotion_cnn.pth")

if __name__ == "__main__":
    main()
