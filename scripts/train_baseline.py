# train_baseline.py
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import yaml
import matplotlib.pyplot as plt
from datetime import datetime
import json
import random
import numpy as np

project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from tqdm import tqdm
from src.models.baseline_cnn import BaselineCNN
from src.data.preprocess_pytorch import create_dataloader


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    print(f"Random seed set to {seed}")


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, label_smoothing=0.1):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.label_smoothing = label_smoothing
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.cross_entropy(
            inputs, targets, 
            reduction='none',
            label_smoothing=self.label_smoothing
        )
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


class EarlyStopping:
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
        
    def save_history(self, save_path):
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        with open(save_path, 'w') as f:
            json.dump(self.history, f, indent=4)
        print(f"Training history saved to: {save_path}")
        
    def plot_and_save(self, save_path, save_history_path=None):
        plt.style.use('seaborn-v0_8-whitegrid')
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        plt.rcParams.update({
            "font.family": "serif",
            "font.size": 13,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "legend.fontsize": 11,
            "figure.dpi": 300
        })

        epochs = range(1, len(self.history['train_loss']) + 1)

        axes[0].plot(epochs, self.history['train_loss'], label='Train Loss', color="#1f77b4", linewidth=2)
        axes[0].plot(epochs, self.history['val_loss'], label='Val Loss', color="#ff7f0e", linewidth=2, linestyle="--")
        axes[0].set_xlabel("Epoch")
        axes[0].set_ylabel("Loss")
        axes[0].set_title("Training vs Validation Loss")
        axes[0].legend(frameon=False)
        axes[0].grid(alpha=0.3)

        axes[1].plot(epochs, self.history['train_acc'], label='Train Accuracy', color="#2ca02c", linewidth=2)
        axes[1].plot(epochs, self.history['val_acc'], label='Val Accuracy', color="#d62728", linewidth=2, linestyle="--")
        axes[1].set_xlabel("Epoch")
        axes[1].set_ylabel("Accuracy")
        axes[1].set_title("Training vs Validation Accuracy")
        axes[1].legend(frameon=False)
        axes[1].grid(alpha=0.3)

        best_val_acc = max(self.history['val_acc'])
        best_epoch = self.history['val_acc'].index(best_val_acc) + 1
        final_train_acc = self.history['train_acc'][-1]
        summary_text = (
            f"Training Summary\n"
            f"──────────────────────────────\n"
            f"Best Val Accuracy : {best_val_acc:.4f}\n"
            f"Best Epoch        : {best_epoch}\n"
            f"Final Train Acc   : {final_train_acc:.4f}\n"
            f"Total Epochs      : {len(self.history['train_loss'])}"
        )
        props = dict(boxstyle="round,pad=0.5", facecolor="whitesmoke", edgecolor="gray", alpha=0.9)
        fig.text(0.73, 0.25, summary_text, fontsize=12, family="monospace", bbox=props)

        plt.tight_layout(rect=[0, 0, 0.95, 1])
        fig.savefig(save_path, bbox_inches="tight")
        plt.close()
        print(f"Plot saved to: {save_path}")

        if save_history_path is not None:
            self.save_history(save_history_path)


def mixup_data(x, y, alpha=0.4):
    lam = np.random.beta(alpha, alpha)
    index = torch.randperm(x.size(0)).to(x.device)
    mixed_x = lam * x + (1 - lam) * x[index]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0

    pbar = tqdm(loader, desc="Training", leave=False)
    for frames, labels in pbar:
        frames, labels = frames.to(device), labels.to(device)
        optimizer.zero_grad()

        # --- MIXUP AUGMENTATION (50% of batches) ---
        if torch.rand(1).item() < 0.5:
            frames, y_a, y_b, lam = mixup_data(frames, labels, alpha=0.4)
            logits = model(frames)
            loss = lam * criterion(logits, y_a) + (1 - lam) * criterion(logits, y_b)
        else:
            logits = model(frames)
            loss = criterion(logits, labels)
        # -------------------------------------------

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * frames.size(0)
        preds = logits.argmax(1)
        correct += (preds == labels).sum().item()
        total += labels.size(0)

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
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config


def load_checkpoint(model, optimizer, checkpoint_path, device):
    print(f"\nLoading checkpoint from: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    print(f"  Loaded from epoch {checkpoint['epoch']}")
    print(f"  Previous val_acc: {checkpoint['val_acc']:.4f}")
    print(f"  Previous val_loss: {checkpoint['val_loss']:.4f}")
    
    return checkpoint['epoch'], checkpoint['val_acc']


def main(config_path='config/train_config.yaml', resume_from=None):
    set_seed(42)
    
    config = load_config(config_path)
    print("\n" + "="*60)
    print("Configuration loaded:")
    print("="*60)
    print(yaml.dump(config, default_flow_style=False))
    
    device = config['device'] if torch.cuda.is_available() else 'cpu'
    print(f"\nUsing device: {device}")
    
    Path(config['checkpoint']['save_dir']).mkdir(exist_ok=True, parents=True)
    Path(config['logging']['plot_dir']).mkdir(exist_ok=True, parents=True)
    Path(config['logging']['log_dir']).mkdir(exist_ok=True, parents=True)
    
    train_loader = create_dataloader(
        metadata_csv=config['data']['train_csv'],
        cache_root=config['data']['cache_root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        clip_mode=True,
        shuffle=True,
        augment=True
    )

    val_loader = create_dataloader(
        metadata_csv=config['data']['val_csv'],
        cache_root=config['data']['cache_root'],
        batch_size=config['data']['batch_size'],
        num_workers=config['data']['num_workers'],
        clip_mode=True,
        shuffle=False,
        augment=False
    )
    
    print(f"\nDataset sizes: Train={len(train_loader.dataset)}, Val={len(val_loader.dataset)}")
    
    model = BaselineCNN(
        num_classes=config['model']['num_classes'],
        pretrained=config['model']['pretrained'],
        temporal_pool="attention",
        dropout=0.5
    ).to(device)

    for param in model.parameters():
        param.requires_grad = False
    
    print("\nUnfreezing layer4 and classifier head:")
    for name, param in model.named_parameters():
        if name.startswith('features.6.') or name.startswith('features.7.') or name.startswith('head.') or name.startswith('temporal_attention.'):
            param.requires_grad = True
            print(f"  {name}")
    
    layer3_params = sum(p.numel() for n, p in model.named_parameters() if 'features.6.' in n and p.requires_grad)
    layer4_params = sum(p.numel() for n, p in model.named_parameters() if 'features.7.' in n and p.requires_grad)
    head_params = sum(p.numel() for n, p in model.named_parameters() if 'head.' in n and p.requires_grad)
    attn_params = sum(p.numel() for n, p in model.named_parameters() if 'temporal_attention' in n and p.requires_grad)
    total_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total_params = sum(p.numel() for p in model.parameters())
    
    print(f"\nTrainable parameter breakdown:")
    print(f"  Layer3: {layer3_params:,} params")
    print(f"  Layer4: {layer4_params:,} params")
    print(f"  Head: {head_params:,} params")
    print(f"  Temporal Attention: {attn_params:,} params")
    print(f"  Total: {total_trainable:,} / {total_params:,} ({100*total_trainable/total_params:.1f}%)")
    
    base_lr = config['training']['learning_rate']
    optimizer = optim.AdamW([
        {'params': [p for n, p in model.named_parameters() if 'features.7.' in n and p.requires_grad], 
         'lr': base_lr * 0.1},
        {'params': [p for n, p in model.named_parameters() if 'head.' in n and p.requires_grad], 
         'lr': base_lr},
        {'params': [p for n, p in model.named_parameters() if 'temporal_attention' in n and p.requires_grad], 
         'lr': base_lr}
    ], weight_decay=config['training']['weight_decay'])
    
    print(f"\nOptimizer learning rates:")
    print(f"  Layer4: LR={base_lr * 0.1:.6f}")
    print(f"  Head/Attention: LR={base_lr:.6f}")
    
    start_epoch = 0
    best_val_acc = 0.0
    if resume_from is not None:
        start_epoch, best_val_acc = load_checkpoint(model, optimizer, resume_from, device)
        print(f"\nResuming training from epoch {start_epoch}")
    
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
    
    criterion = FocalLoss(alpha=0.25, gamma=2.0, label_smoothing=0.1)
    
    early_stopping = None
    if config['training']['early_stopping']['enabled']:
        early_stopping = EarlyStopping(
            patience=config['training']['early_stopping']['patience'],
            mode=config['training']['early_stopping']['mode']
        )
    
    logger = TrainingLogger(config)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\nStarting training for {config['training']['epochs']} epochs...")
    print("="*60)
    
    for epoch in range(1, config['training']['epochs'] + 1):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        current_lr = optimizer.param_groups[0]['lr']
        logger.log(epoch, train_loss, train_acc, val_loss, val_acc, current_lr)
        
        print(f"Epoch {epoch:2d}/{config['training']['epochs']} | "
              f"Train Loss: {train_loss:.4f} Acc: {train_acc:.4f} | "
              f"Val Loss: {val_loss:.4f} Acc: {val_acc:.4f} | "
              f"LR: {current_lr:.6f}")
        
        if scheduler is not None:
            monitor_value = val_loss if config['training']['lr_scheduler']['monitor'] == 'val_loss' else val_acc
            scheduler.step(monitor_value)
        
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            model_path = Path(config['checkpoint']['save_dir']) / f"{config['model']['name']}_best.pt"
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
                'val_loss': val_loss,
                'config': config
            }, model_path)
            print(f"  -> Saved best model (val_acc={val_acc:.4f})")
        
        if early_stopping is not None and early_stopping(val_acc):
            print(f"\nEarly stopping triggered at epoch {epoch}")
            break
    
    print("="*60)
    print(f"Training complete! Best val accuracy: {best_val_acc:.4f}")
    
    final_model_path = Path(config['checkpoint']['save_dir']) / f"{config['model']['name']}_{timestamp}_final.pt"
    torch.save({
        'epoch': epoch,
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'val_acc': val_acc,
        'config': config
    }, final_model_path)
    print(f"Final model saved: {final_model_path}")
    
    if config['logging']['save_plot']:
        plot_path = Path(config['logging']['plot_dir']) / f"{config['model']['name']}_{timestamp}_training.png"
        history_path = Path(config['logging']['log_dir']) / f"{config['model']['name']}_{timestamp}_history.json"
        logger.plot_and_save(plot_path, save_history_path=history_path)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config/train_config.yaml',
                        help='Path to config file')
    parser.add_argument('--resume', type=str, default=None,
                        help='Path to checkpoint to resume from')
    args = parser.parse_args()
    
    main(args.config, resume_from=args.resume)