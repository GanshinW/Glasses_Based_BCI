# main.py

import os
from datetime import datetime
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report

# Import your existing modules
from config import Config
from models.branches import TimeBranch, FreqBranch, ImgBranch
from models.multimodal_model import MultiModalNet

# Import utilities
from utils.load_npz import load_dataset, print_dataset_stats
from utils.features import (
    extract_time_domain,
    extract_freq_domain,
    extract_timefreq_images
)

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('training.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)


class EEGMultimodalDataset(Dataset):
    """PyTorch Dataset for multimodal EEG data"""
    def __init__(self, X_time, X_freq, X_img, y):
        self.X_time = X_time
        self.X_freq = X_freq
        self.X_img  = X_img
        self.y      = y

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        sample = {
            'time': torch.tensor(self.X_time[idx], dtype=torch.float32),
            'freq': torch.tensor(self.X_freq[idx], dtype=torch.float32),
            'label': torch.tensor(int(self.y[idx]), dtype=torch.long),
        }
        if self.X_img is not None:
            sample['img'] = torch.tensor(self.X_img[idx], dtype=torch.float32)
        return sample


def train_epoch(model, loader, criterion, optimizer, device, use_img=True):
    """Train for one epoch"""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch in loader:
        optimizer.zero_grad()
        
        # Prepare inputs for your MultiModalNet
        time = batch['time'].to(device)
        freq = batch['freq'].to(device)
        img = batch.get('img').to(device) if use_img and 'img' in batch else None
        labels = batch['label'].to(device)
        
        # Forward pass through your model
        outputs = model(time, freq, img)
        loss = criterion(outputs, labels)
        
        # Backward and optimize
        loss.backward()
        optimizer.step()
        
        # Statistics
        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def evaluate(model, loader, criterion, device, use_img=True):
    """Evaluate model"""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch in loader:
            time = batch['time'].to(device)
            freq = batch['freq'].to(device)
            img = batch.get('img').to(device) if use_img and 'img' in batch else None
            labels = batch['label'].to(device)
            
            outputs = model(time, freq, img)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item()
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    avg_loss = running_loss / len(loader)
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def main():
    # Load configuration
    config = Config()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # ============= Load Dataset =============
    logger.info("Loading dataset...")
    project_root = os.path.dirname(os.path.dirname(__file__))
    dataset_file = os.path.join(project_root, 'simulation', 'chewing_dataset.npz')
    
    ds = load_dataset(dataset_file)
    print_dataset_stats(ds)
    
    # Get dataset parameters
    fs = ds['sampling_rate']
    n_channels = len(ds['channel_names'])
    n_samples = ds['X_train'].shape[2]
    n_classes = len(np.unique(ds['y_train']))
    
    logger.info(f"Dataset: {n_channels} channels, {n_samples} samples, {n_classes} classes")
    
    # ============= Extract Features =============
    logger.info("Extracting features...")
    
    # Time-domain
    X_time_train = extract_time_domain(ds['X_train'], fs)
    X_time_val   = extract_time_domain(ds['X_val'],   fs)
    X_time_test  = extract_time_domain(ds['X_test'],  fs)
    logger.info(f"  Time-domain shape: {X_time_train.shape}")
    
    # Frequency-domain
    X_freq_train = extract_freq_domain(X_time_train, fs, config.bands)
    X_freq_val   = extract_freq_domain(X_time_val,   fs, config.bands)
    X_freq_test  = extract_freq_domain(X_time_test,  fs, config.bands)
    logger.info(f"  Frequency-domain shape: {X_freq_train.shape}")
    
    # Time-frequency images
    if config.use_img:
        X_img_train = extract_timefreq_images(X_time_train, fs)
        X_img_val   = extract_timefreq_images(X_time_val,   fs)
        X_img_test  = extract_timefreq_images(X_time_test,  fs)
        logger.info(f"  Time-frequency image shape: {X_img_train.shape}")
    else:
        X_img_train = X_img_val = X_img_test = None
    
    # ============= Create DataLoaders =============
    train_dataset = EEGMultimodalDataset(X_time_train, X_freq_train, X_img_train, ds['y_train'])
    val_dataset   = EEGMultimodalDataset(X_time_val,   X_freq_val,   X_img_val,   ds['y_val'])
    test_dataset  = EEGMultimodalDataset(X_time_test,  X_freq_test,  X_img_test,  ds['y_test'])
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, shuffle=True, num_workers=config.num_workers)
    val_loader   = DataLoader(val_dataset,   batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    test_loader  = DataLoader(test_dataset,  batch_size=config.batch_size, shuffle=False, num_workers=config.num_workers)
    
    # ============= Initialize Model =============
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create your MultiModalNet
    model = MultiModalNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_bands=len(config.bands),
        img_out_dim=config.img_out_dim,
        hidden_dim=config.hidden_dim,
        n_classes=n_classes,
        use_img=config.use_img
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', patience=3, factor=0.5)
    
    # ============= Training Loop =============
    logger.info("Starting training...")
    best_val_acc = 0.0
    patience_counter = 0
    
    for epoch in range(1, config.num_epochs + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device, config.use_img)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device, config.use_img)
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Log progress
        logger.info(
            f"Epoch [{epoch:3d}/{config.num_epochs}] | "
            f"Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}% | "
            f"Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.2f}%"
        )
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f"  -> Saved best model (Val Acc: {val_acc:.2f}%)")
        else:
            patience_counter += 1
            if patience_counter >= config.patience:
                logger.info(f"Early stopping at epoch {epoch}")
                break
    
    # ============= Test Evaluation =============
    logger.info("Evaluating on test set...")
    model.load_state_dict(torch.load('best_model.pth'))
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, config.use_img)
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # Get predictions for classification report
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in test_loader:
            time = batch['time'].to(device)
            freq = batch['freq'].to(device)
            img = batch.get('img').to(device) if config.use_img else None
            labels = batch['label']
            
            outputs = model(time, freq, img)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.numpy())
    
    # Print classification report
    class_names = ['Baseline', 'Chewing'] if n_classes == 2 else ['Calm', 'Positive', 'Negative', 'EyeMove', 'JawClench']
    report = classification_report(all_labels, all_preds, target_names=class_names[:n_classes], digits=4)
    logger.info("\nClassification Report:\n" + report)
    
    logger.info("Training completed!")


if __name__ == "__main__":
    main()