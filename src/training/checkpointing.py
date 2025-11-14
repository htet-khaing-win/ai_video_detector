"""
Smart checkpoint management system.

Features:
- Automatic saving of best.pt, latest.pt, global_best.pt
- Global best tracking across all runs
- Keep last N checkpoints to save disk space
- Resume training support
- Supports additional_state for progressive training phases
"""

import torch
from pathlib import Path
import json
from datetime import datetime
import logging

logger = logging.getLogger(__name__)


class CheckpointManager:
    """
    Manages model checkpoints with smart saving logic.
    """
    
    def __init__(self, checkpoint_dir: str, keep_last_n: int = 3):
        self.checkpoint_dir = Path(checkpoint_dir)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        self.keep_last_n = keep_last_n
        self.global_best_file = self.checkpoint_dir.parent / "global_best.json"
        
        self.epoch = 0
        self.global_step = 0
        self.best_val_acc = 0.0
    
    def save_checkpoint(self,
                       epoch: int,
                       model,
                       optimizer,
                       scheduler,
                       scaler,
                       val_acc: float,
                       best_val_acc: float,
                       global_step: int,
                       is_best: bool = False,
                       additional_state: dict = None):
        """
        Save checkpoint with optional additional state.
        """
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict() if scheduler else None,
            'scaler_state_dict': scaler.state_dict(),
            'val_acc': val_acc,
            'best_val_acc': best_val_acc,
            'global_step': global_step,
            'timestamp': datetime.now().isoformat()
        }

        if additional_state is not None:
            checkpoint['additional_state'] = additional_state
        
        # Save latest
        latest_path = self.checkpoint_dir / "latest.pt"
        torch.save(checkpoint, latest_path)
        
        # Save epoch checkpoint
        epoch_path = self.checkpoint_dir / f"epoch_{epoch:03d}.pt"
        torch.save(checkpoint, epoch_path)
        
        # Cleanup old epoch checkpoints
        self._cleanup_old_checkpoints()
        
        # Save best
        if is_best:
            best_path = self.checkpoint_dir / "best.pt"
            torch.save(checkpoint, best_path)
            
            # Check global best
            global_best = self.load_global_best()
            if val_acc > global_best:
                global_best_path = self.checkpoint_dir / "global_best.pt"
                torch.save(checkpoint, global_best_path)
                self.save_global_best(val_acc, epoch)
                logger.info(f" New global best: {val_acc:.2f}%")
    
    def _cleanup_old_checkpoints(self):
        epoch_files = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if len(epoch_files) > self.keep_last_n:
            for old_file in epoch_files[:-self.keep_last_n]:
                old_file.unlink()
    
    def load_checkpoint(self, checkpoint_path: str, model, optimizer, scheduler, scaler):
        checkpoint_path = Path(checkpoint_path)
        if not checkpoint_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")
        
        checkpoint = torch.load(checkpoint_path, map_location='cpu')

        # --- MODEL LOAD ---
        model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        logger.info(f" Loaded model weights from {checkpoint_path}")

        # --- OPTIMIZER LOAD ---
        if optimizer is not None and 'optimizer_state_dict' in checkpoint:
            try:
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                logger.info(" Loaded optimizer state successfully.")
            except ValueError as e:
                logger.warning(
                    f" Optimizer state not loaded due to mismatch ({e}). "
                    "Using freshly initialized optimizer instead."
                )

        # --- SCHEDULER LOAD ---
        if scheduler is not None and 'scheduler_state_dict' in checkpoint:
            try:
                scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                logger.info(" Loaded scheduler state successfully.")
            except Exception as e:
                logger.warning(f" Scheduler state not loaded: {e}")

        # --- SCALER LOAD ---
        if scaler is not None and 'scaler_state_dict' in checkpoint:
            try:
                scaler.load_state_dict(checkpoint['scaler_state_dict'])
                logger.info(" Loaded scaler state successfully.")
            except Exception as e:
                logger.warning(f" Scaler state not loaded: {e}")

        # --- Restore metadata ---
        self.epoch = checkpoint.get('epoch', 0)
        self.global_step = checkpoint.get('global_step', 0)
        self.best_val_acc = checkpoint.get('best_val_acc', 0.0)

        # --- Restore additional_state if exists ---
        if 'additional_state' in checkpoint:
            self.additional_state = checkpoint['additional_state']
        else:
            self.additional_state = {}

        logger.info(f"Resumed from epoch {self.epoch} | Best val acc: {self.best_val_acc:.2f}%")
    
    def find_latest_checkpoint(self) -> Path:
        latest = self.checkpoint_dir / "latest.pt"
        if latest.exists():
            return latest
        epoch_files = sorted(self.checkpoint_dir.glob("epoch_*.pt"))
        if epoch_files:
            return epoch_files[-1]
        raise FileNotFoundError(f"No checkpoints found in {self.checkpoint_dir}")
    
    def save_global_best(self, accuracy: float, epoch: int):
        data = {
            'best_accuracy': accuracy,
            'epoch': epoch,
            'timestamp': datetime.now().isoformat()
        }
        with open(self.global_best_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load_global_best(self) -> float:
        if self.global_best_file.exists():
            try:
                with open(self.global_best_file, 'r') as f:
                    data = json.load(f)
                    return data.get('best_accuracy', 0.0)
            except:
                return 0.0
        return 0.0
