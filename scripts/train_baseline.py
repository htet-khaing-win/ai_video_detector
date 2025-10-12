import sys
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
from tqdm import tqdm
from src.models.baseline_cnn import BaselineCNN
from src.data.preprocess_pytorch import create_dataloader
from utils.vram_monitor import log_and_warn

def train_epoch(model, loader, optimizer, criterion, device):
    model.train()
    total_loss, correct, total = 0, 0, 0
    for frames, labels in tqdm(loader, desc="Training", leave=False):
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

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    
    model = BaselineCNN(num_classes=2, pretrained=True, temporal_pool="avg").to(device)
    
    #Use test_metadata as validation
    train_loader = create_dataloader(
        metadata_csv="data/splits/train_metadata.csv",
        cache_root="data/processed/genbuster_cached",
        batch_size=6,
        num_workers=4,
        clip_mode=True,
        shuffle=True
    )
    
    val_loader = create_dataloader(
        metadata_csv="data/splits/test_metadata.csv",
        cache_root="data/processed/genbuster_cached",
        batch_size=6,
        num_workers=4,
        clip_mode=True,
        shuffle=False
    )
    
    print(f"Train dataset: {len(train_loader.dataset)} samples")
    print(f"Val dataset: {len(val_loader.dataset)} samples")
    
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    criterion = nn.CrossEntropyLoss()
    
    best_val_acc = 0.0
    
    for epoch in range(1, 11):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        print(f"[Epoch {epoch}] Train loss={train_loss:.4f} acc={train_acc:.3f} | Val loss={val_loss:.4f} acc={val_acc:.3f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            Path("checkpoints").mkdir(exist_ok=True)
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_acc': val_acc,
            }, "checkpoints/baseline_cnn_best.pt")
            print(f"  Saved best model (val_acc={val_acc:.3f})")
        
        # Monitor VRAM
        log_and_warn()
    
    print(f"Training complete. Best val acc: {best_val_acc:.3f}")

if __name__ == "__main__":
    main()