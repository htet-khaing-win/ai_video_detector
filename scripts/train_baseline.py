import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import json

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
from src.models.baseline_cnn import BaselineCNN
from src.data.preprocess_pytorch import create_dataloader


class EarlyStopping:
    """Early stopping to stop training when validation metric stops improving."""
    def __init__(self, patience=5, mode='max', min_delta=0.0):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, metric):
        score = metric if self.mode == 'max' else -metric
        
        if self.best_score is None:
            self.best_score = score
        elif score < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.counter = 0
        
        return self.early_stop


class TrainingLogger:
    """Logger to track training metrics and create plots."""
    def __init__(self, config):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'lr': []
        }
        self.config = config
        
    def log(self, epoch, train_loss, train_acc, val_loss, val_acc, lr):
        self.history['train_loss'].append(train_loss)
        self.history['train_acc'].append(train_acc)
        self.history['val_loss'].append(val_loss)
        self.history['val_acc'].append(val_acc)
        self.history['lr'].append(lr)
        
    def plot_and_save(self, save_path):
        """Create and save training plots."""
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        
        # Loss plot
        axes[0, 0].plot(self.history['train_loss'], label='Train Loss')
        axes[0, 0].plot(self.history['val_loss'], label='Val Loss')
        axes[0, 0].set_xlabel('Epoch')
        axes[0, 0].set_ylabel('Loss')
        axes[0, 0].set_title('Training and Validation Loss')
        axes[0, 0].legend()
        axes[0, 0].grid(True)
        
        # Accuracy plot
        axes[0, 1].plot(self.history['train_acc'], label='Train Acc')
        axes[0, 1].plot(self.history['val_acc'], label='Val Acc')
        axes[0, 1].set_xlabel('Epoch')
        axes[0, 1].set_ylabel('Accuracy')
        axes[0, 1].set_title('Training and Validation Accuracy')
        axes[0, 1].legend()
        axes[0, 1].grid(True)
        
        # Summary text
        best_val_acc = max(self.history['val_acc'])
        best_epoch = self.history['val_acc'].index(best_val_acc) + 1
        final_train_acc = self.history['train_acc'][-1]
        
        summary_text = f"""
        Training Summary:
        
        Best Val Accuracy: {best_val_acc:.4f}
        Best Epoch: {best_epoch}
        Final Train Accuracy: {final_train_acc:.4f}
        Total Epochs: {len(self.history['train_loss'])}
        """
        
        axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, 
                       verticalalignment='center', family='monospace')
        axes[1, 1].axis('off')
        
        plt.tight_layout()
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.close()
        print(f"Plot saved to: {save_path}")
        
    def save_history(self, save_path):
        """Save training history as JSON."""
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=2)
        print(f"History saved to: {save_path}")


def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    
    pbar = tqdm(loader, desc="Training", leave=False)
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)
        
        optimizer.zero_grad()
        logits = model(frames)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * frames.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
        
        # Update progress bar
        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'acc': f'{correct/total:.4f}'
        })
    
    return total_loss / total, correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    
    for frames, labels in tqdm(loader, desc="Validation", leave=False):
        frames, labels = frames.to(device), labels.to(device)
        logits = model(frames)
        loss = criterion(logits, labels)
        
        total_loss += loss.item() * frames.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    
    return total_loss / total, correct / total


def load_config(config_path):
    """Load configuration from YAML file."""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def main(config_path='config/train_config.yaml'):
    # Load configuration
    config = load_config(config_path)
    print("Configuration loaded:")
    print(yaml.dump(config, default_flow_style=False))
    
    # Setup
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    # Create directories
    Path(config['checkpoint']['save_dir']).mkdir(exist_ok=True, parents=True)
    Path(config['logging']['plot_dir']).mkdir(exist_ok=True, parents=True)
    Path(config['logging']['log_dir']).mkdir(exist_ok=True, parents=True)
    
    # Data loaders
    train_loader = create_dataloader(
        metadata_csv=config['data']['train_csv'],
        cache_root=config['data']['cache_root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        clip_mode=True,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        metadata_csv=config['data']['val_csv'],
        cache_root=config['data']['cache_root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        clip_mode=True,
        shuffle=False
    )
    
    print(f"\nDataset sizes:")
    print(f"  Train: {len(train_loader.dataset)} samples")
    print(f"  Val: {len(val_loader.dataset)} samples")
    
    # Model
    model = BaselineCNN(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        temporal_pool=config['model']['temporal_pool']
    ).to(device)
    
    # Optimizer
    optimizer = optim.Adam(
        model.parameters(),
        lr=config['training']['learning_rate'],
        weight_decay=config['training']['weight_decay']
    )
    
    # Learning rate scheduler
    scheduler = None
    if config['training']['lr_scheduler']['enabled']:
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode=config['training']['lr_scheduler']['mode'],
            factor=config['training']['lr_scheduler']['factor'],
            patience=config['training']['lr_scheduler']['patience'],
            min_lr=config['training']['lr_scheduler']['min_lr'],
            verbose=True
        )
    
    # Loss function
    criterion = nn.CrossEntropyLoss()
    
    # Early stopping
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            mode=config['training']['early_stopping']['mode']
        )
    
    # Training logger
    logger = TrainingLogger(config)
    
    # Training loop
    best_val_acc = 0.0
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print("="*60)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        # Train
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        
        # Validate
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        # Get current learning rate
        current_lr = optimizer.param_groups[0]['lr']
        
        # Log metrics
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        # Print epoch summary
        print(f"Epoch {epoch:2d}/{config['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        # Learning rate scheduler step
        if scheduler is not None:
            if config['training']['lr_scheduler']['monitor'] == 'val_loss':
                scheduler.step(val_loss)
            else:
                scheduler.step(val_acc)
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            
            model_filename = f"{config['model']['name']}_best.pt"
            model_path = Path(config['checkpoint']['save_dir']) / model_filename
            
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, model_path)
            
            print(f"  → Saved best model (val_acc={val_acc:.4f})")
        
        # Early stopping check
        if early_stopping is not None:
            if early_stopping(val_acc):
                print(f"\nEarly stopping triggered at epoch {epoch}")
                break
    
    print("="*60)
    print(f"Training complete! Best val accuracy: {best_val_acc:.4f}")
    
    # Save final model
    final_model_filename = f"{config['model']['name']}_{timestamp}_final.pt"
    final_model_path = Path(config['checkpoint']['save_dir']) / final_model_filename
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    # Save plots
    if config['logging']['save_plot']:
        plot_filename = f"{config['model']['name']}_{timestamp}_training.png"
        plot_path = Path(config['logging']['plot_dir']) / plot_filename
        logger.plot_and_save(plot_path)
        
        # Save history
        history_filename = f"{config['model']['name']}_{timestamp}_history.json"
        history_path = Path(config['logging']['log_dir']) / history_filename
        logger.save_history(history_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                       help='Path to config file')
    args = parser.parse_args()
    
    main(args.config)