# main.py

import os
import sys
from datetime import datetime
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Import your existing modules
from config import Config
from models.branches import TimeBranch, FreqBranch, ImgBranch
from models.multimodal_model import MultiModalNet

# Import utilities
from utils.load_npz import load_dataset, print_dataset_stats, get_dataset_info
from utils.features import extract_all_features

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
        self.X_img = X_img
        self.y = y

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


def get_predictions(model, loader, device, use_img=True):
    """Get predictions for classification report"""
    model.eval()
    all_preds = []
    all_labels = []
    
    with torch.no_grad():
        for batch in loader:
            time = batch['time'].to(device)
            freq = batch['freq'].to(device)
            img = batch.get('img').to(device) if use_img and 'img' in batch else None
            labels = batch['label'].to(device)
            
            outputs = model(time, freq, img)
            _, predicted = outputs.max(1)
            
            all_preds.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    return np.array(all_labels), np.array(all_preds)


def plot_training_history(train_losses, train_accs, val_losses, val_accs, save_path):
    """Plot and save training history"""
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(train_losses, label='Train Loss', color='blue')
    plt.plot(val_losses, label='Val Loss', color='red')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(train_accs, label='Train Accuracy', color='blue')
    plt.plot(val_accs, label='Val Accuracy', color='red')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()  # Don't display, just save


def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """Plot and save confusion matrix"""
    cm = confusion_matrix(y_true, y_pred)
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=class_names,
                yticklabels=class_names)
    plt.title('Confusion Matrix')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()


def save_results(results, filepath):
    """Save training results to JSON"""
    import json
    with open(filepath, 'w') as f:
        json.dump(results, f, indent=2)


def train_single_dataset(dataset, dataset_info, config, output_dir):
    """
    Train model on a single dataset (intent or emotion)
    
    Parameters:
    -----------
    dataset : dict
        Loaded dataset
    dataset_info : dict  
        Dataset information from get_dataset_info()
    config : Config
        Configuration object
    output_dir : str
        Directory to save results
        
    Returns:
    --------
    results : dict
        Training results and metrics
    """
    task_name = dataset_info['task_type'].replace('_', ' ').title()
    print(f"\n{'='*60}")
    print(f"TRAINING {task_name.upper()}")
    print(f"{'='*60}")
    
    # Extract features
    print("Extracting features...")
    features = extract_all_features(dataset, config)
    
    # Get dataset parameters
    fs = dataset_info['sampling_rate']
    n_channels = dataset_info['n_channels']
    n_samples = dataset_info['n_samples']
    n_classes = dataset_info['n_classes']
    class_names = dataset_info['class_names']
    
    logger.info(f"Dataset: {n_channels} channels, {n_samples} samples, {n_classes} classes")
    logger.info(f"Classes: {class_names}")
    
    # Create DataLoaders
    print("Creating data loaders...")
    train_dataset = EEGMultimodalDataset(
        features['X_time_train'], 
        features['X_freq_train'], 
        features['X_img_train'], 
        dataset['y_train']
    )
    val_dataset = EEGMultimodalDataset(
        features['X_time_val'], 
        features['X_freq_val'], 
        features['X_img_val'], 
        dataset['y_val']
    )
    test_dataset = EEGMultimodalDataset(
        features['X_time_test'], 
        features['X_freq_test'], 
        features['X_img_test'], 
        dataset['y_test']
    )
    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, 
                             shuffle=True, num_workers=config.num_workers)
    val_loader = DataLoader(val_dataset, batch_size=config.batch_size, 
                           shuffle=False, num_workers=config.num_workers)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, 
                            shuffle=False, num_workers=config.num_workers)
    
    # Initialize model
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    model = MultiModalNet(
        n_channels=n_channels,
        n_samples=n_samples,
        n_bands=len(config.bands),
        img_out_dim=config.img_out_dim,
        hidden_dim=config.hidden_dim,
        n_classes=n_classes,
        use_img=config.use_img
    ).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', patience=3, factor=0.5, verbose=True
    )
    
    # Training loop
    logger.info("Starting training...")
    best_val_acc = 0.0
    patience_counter = 0
    
    train_losses, train_accs = [], []
    val_losses, val_accs = [], []
    
    for epoch in range(1, config.num_epochs + 1):
        print(f"\nEpoch {epoch}/{config.num_epochs}")
        print("-" * 50)
        
        # Train
        train_loss, train_acc = train_epoch(
            model, train_loader, criterion, optimizer, device, config.use_img
        )
        
        # Validate
        val_loss, val_acc = evaluate(
            model, val_loader, criterion, device, config.use_img
        )
        
        # Update scheduler
        scheduler.step(val_acc)
        
        # Store history
        train_losses.append(train_loss)
        train_accs.append(train_acc)
        val_losses.append(val_loss)
        val_accs.append(val_acc)
        
        logger.info(f"Epoch {epoch:2d} - Train Loss: {train_loss:.4f}, "
                   f"Train Acc: {train_acc:.2f}%, Val Loss: {val_loss:.4f}, "
                   f"Val Acc: {val_acc:.2f}%")
        
        # Early stopping and model saving
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            patience_counter = 0
            
            # Save best model
            model_name = dataset_info['task_type'].replace('_', '')
            model_path = os.path.join(output_dir, f'best_{model_name}_model.pth')
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'config': config.__dict__,
                'dataset_info': dataset_info,
                'model_params': {
                    'n_channels': n_channels,
                    'n_samples': n_samples,
                    'n_bands': len(config.bands),
                    'img_out_dim': config.img_out_dim,
                    'hidden_dim': config.hidden_dim,
                    'n_classes': n_classes,
                    'use_img': config.use_img
                }
            }, model_path)
            logger.info(f"New best model saved: {val_acc:.2f}%")
        else:
            patience_counter += 1
            
        if patience_counter >= config.patience:
            logger.info(f"Early stopping at epoch {epoch}")
            break
    
    # Final evaluation
    print(f"\n{'='*50}")
    print("FINAL EVALUATION")
    print(f"{'='*50}")
    
    # Load best model
    model_path = os.path.join(output_dir, f'best_{model_name}_model.pth')
    checkpoint = torch.load(model_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    # Test evaluation
    test_loss, test_acc = evaluate(model, test_loader, criterion, device, config.use_img)
    logger.info(f"Test Accuracy: {test_acc:.2f}%")
    
    # Get predictions for detailed analysis
    y_true, y_pred = get_predictions(model, test_loader, device, config.use_img)
    
    # Classification report
    print(f"\nClassification Report:")
    report = classification_report(y_true, y_pred, target_names=class_names, digits=4)
    print(report)
    
    # Save plots and results
    task_short = dataset_info['task_type'].replace('_detection', '').replace('_recognition', '')
    
    # Training history plot
    plot_training_history(
        train_losses, train_accs, val_losses, val_accs,
        save_path=os.path.join(output_dir, f'{task_short}_training_history.png')
    )
    
    # Confusion matrix plot
    plot_confusion_matrix(
        y_true, y_pred, class_names,
        save_path=os.path.join(output_dir, f'{task_short}_confusion_matrix.png')
    )
    
    # Compile results
    results = {
        'task': dataset_info['task_type'],
        'task_name': task_name,
        'dataset_info': dataset_info,
        'model_params': total_params,
        'training_config': config.__dict__,
        'best_val_acc': best_val_acc,
        'test_acc': test_acc,
        'epochs_trained': len(train_losses),
        'final_epoch': epoch,
        'train_history': {
            'train_losses': train_losses,
            'train_accs': train_accs,
            'val_losses': val_losses,
            'val_accs': val_accs
        },
        'test_results': {
            'y_true': y_true.tolist(),
            'y_pred': y_pred.tolist(),
            'classification_report': report
        },
        'class_names': class_names
    }
    
    # Save results
    save_results(results, os.path.join(output_dir, f'{task_short}_results.json'))
    
    print(f"\nTraining completed for {task_name}!")
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Test accuracy: {test_acc:.2f}%")
    print(f"Model saved: {model_path}")
    
    return results


def main(task_type='auto'):
    """
    Main training function
    
    Parameters:
    -----------
    task_type : str
        'auto', 'intent', 'emotion', or 'both'
    """
    print("="*80)
    print("GLASSES-BASED BCI TRAINING PIPELINE")
    print("="*80)
    
    # Load configuration
    config = Config()
    
    # Set seeds for reproducibility
    torch.manual_seed(42)
    np.random.seed(42)
    
    # Create output directories
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = f'results_{timestamp}'
    os.makedirs(output_dir, exist_ok=True)
    os.makedirs('models', exist_ok=True)  # Keep models in standard location
    
    logger.info(f"Output directory: {output_dir}")
    
    # Determine which datasets to train
    if task_type == 'both':
        tasks_to_run = ['intent', 'emotion']
    elif task_type in ['intent', 'emotion']:
        tasks_to_run = [task_type]
    else:
        # Auto-detect available datasets
        try:
            # Try to load both and see what's available
            intent_dataset = load_dataset(task_type='intent')
            emotion_dataset = load_dataset(task_type='emotion')
            tasks_to_run = ['intent', 'emotion']
            logger.info("Found both intent and emotion datasets")
        except FileNotFoundError:
            try:
                # Try intent only
                intent_dataset = load_dataset(task_type='intent')
                tasks_to_run = ['intent']
                logger.info("Found intent dataset only")
            except FileNotFoundError:
                try:
                    # Try emotion only
                    emotion_dataset = load_dataset(task_type='emotion')
                    tasks_to_run = ['emotion']
                    logger.info("Found emotion dataset only")
                except FileNotFoundError:
                    logger.error("No datasets found! Please run generate_npz_files.py first")
                    return
    
    all_results = {}
    
    # Train on each available dataset
    for task in tasks_to_run:
        try:
            print(f"\n{'='*80}")
            print(f"LOADING {task.upper()} DATASET")
            print(f"{'='*80}")
            
            # Load dataset
            dataset = load_dataset(task_type=task)
            print_dataset_stats(dataset)
            
            # Get dataset info for model configuration
            dataset_info = get_dataset_info(dataset)
            
            # Train model
            results = train_single_dataset(dataset, dataset_info, config, output_dir)
            all_results[task] = results
            
        except Exception as e:
            logger.error(f"Error training {task} model: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    # Final summary
    print(f"\n{'='*80}")
    print("TRAINING SUMMARY")
    print(f"{'='*80}")
    
    for task, results in all_results.items():
        print(f"\n{results['task_name']}:")
        print(f"  Best validation accuracy: {results['best_val_acc']:.2f}%")
        print(f"  Test accuracy: {results['test_acc']:.2f}%")
        print(f"  Total parameters: {results['model_params']:,}")
        print(f"  Epochs trained: {results['epochs_trained']}")
    
    print(f"\nResults saved to: {output_dir}/")
    print(f"Models saved to: models/")
    
    # Save combined summary
    summary = {
        'timestamp': timestamp,
        'config': config.__dict__,
        'results': all_results,
        'total_tasks': len(all_results)
    }
    save_results(summary, os.path.join(output_dir, 'training_summary.json'))
    
    logger.info("Training pipeline completed successfully!")


if __name__ == "__main__":
    # You can specify which task to run:
    # main('intent')    # Train intent detection only
    # main('emotion')   # Train emotion recognition only  
    # main('both')      # Train both tasks
    # main('auto')      # Auto-detect available datasets
    
    main('auto')  # Default: auto-detect