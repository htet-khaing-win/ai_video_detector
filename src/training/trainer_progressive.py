"""
Progressive Trainer with Gradual Unfreezing and Discriminative Learning Rates.
"""

import torch
import torch.nn as nn
from torch.cuda.amp import autocast, GradScaler
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
from tqdm import tqdm
import time
import logging

import sys
project_root = Path("C:/Personal project/ai_video_detector")
sys.path.insert(0, str(project_root))

from src.training.checkpointing import CheckpointManager
from src.training.trainer import Trainer

logger = logging.getLogger(__name__)


class ProgressiveTrainer(Trainer):
    """
    Extended trainer with progressive unfreezing and discriminative LR.
    """
    
    def __init__(self, model, train_loader, val_loader, criterion,
                 optimizer, scheduler, device, config):
        """
        Initialize progressive trainer.
        """
        # Initialize parent trainer
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=scheduler,
            device=device,
            config=config
        )
        
        # Progressive training state
        self.current_phase = 0
        self.last_unfreeze_epoch = -1
        self.phase_start_epoch = 0
        
        # Track best accuracy per phase
        self.phase_best_acc = 0.0
        
        # Phase transition thresholds (for 30 epochs)
        self.phase_epochs = {
            1: (0, 7),    # Phase 1: epochs 0-7
            2: (8, 15),   # Phase 2: epochs 8-15
            3: (16, 23),  # Phase 3: epochs 16-23
            4: (24, 999)  # Phase 4: epochs 24+
        }
    
    def _get_phase_for_epoch(self, epoch):
        """Determine which phase an epoch belongs to."""
        for phase, (start, end) in self.phase_epochs.items():
            if start <= epoch <= end:
                return phase
        return 4  # Default to phase 4 for any epoch > 23
    
    def _format_layer_lrs(self):
        param_groups = self.optimizer.param_groups
        
        # Get block indices that are active (have learning rates)
        block_lrs = []
        for i, group in enumerate(param_groups):
            lr = group['lr']
            # Infer block index from param group order
            # (groups are added in block order in create_discriminative_optimizer)
            block_lrs.append(lr)
        
        if len(block_lrs) == 0:
            return "[?]: N/A"
        
        # Determine active block range
        num_blocks = len(block_lrs)
        if num_blocks == 2:
            # Phase 1: blocks 4,5
            block_range = "[4,5]"
        elif num_blocks == 3:
            # Phase 2: blocks 3,4,5
            block_range = "[3→5]"
        elif num_blocks == 4:
            # Phase 3: blocks 2,3,4,5
            block_range = "[2→5]"
        else:
            # Phase 4: all blocks 0-5 (6 groups)
            block_range = "[0→5]"
        
        # Format LRs compactly
        lr_str = "→".join([f"{lr:.1e}" for lr in block_lrs])
        
        return f"{block_range}: {lr_str}"


    def _get_trainable_percentage(self):
        """
        Calculate percentage of trainable parameters.
        
        Returns float: percentage (0-100)
        """
        total_params = sum(p.numel() for p in self.model.parameters())
        trainable_params = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        
        if total_params == 0:
            return 0.0
        
        return (trainable_params / total_params) * 100
    
    def _maybe_update_phase(self, epoch):
        """
        Check if we need to transition to a new phase and update accordingly.
        """
        new_phase = self._get_phase_for_epoch(epoch)
        
        # Check if phase changed
        if new_phase != self.current_phase:
            logger.info("\n" + "="*80)
            logger.info(f"PHASE TRANSITION: Phase {self.current_phase} → Phase {new_phase}")
            logger.info(f"Epoch {epoch} / {self.config.training.num_epochs}")
            logger.info("="*80)
            
            # Import unfreezing and optimizer functions
            from train_progressive import (
                setup_progressive_unfreezing,
                create_discriminative_optimizer,
                create_scheduler
            )
            
            # 1. Unfreeze layers for new phase
            phase, unfreeze_blocks = setup_progressive_unfreezing(
                self.model, epoch, logger
            )
            
            # 2. Recreate optimizer with discriminative LR for new phase
            logger.info("\nRecreating optimizer with phase-specific learning rates:")
            self.optimizer = create_discriminative_optimizer(
                self.model, epoch, self.config, logger
            )
            
            # 3. Recreate scheduler for remaining training steps
            num_steps_per_epoch = len(self.train_loader) // self.grad_accum_steps
            remaining_epochs = self.config.training.num_epochs - epoch
            remaining_steps = num_steps_per_epoch * remaining_epochs
            
            self.scheduler = create_scheduler(
                self.optimizer, self.config, remaining_steps, logger
            )
            
            # 4. Update phase tracking
            self.current_phase = new_phase
            self.last_unfreeze_epoch = epoch
            self.phase_start_epoch = epoch
            self.phase_best_acc = 0.0  # Reset phase best
            
            logger.info(f"\n✓ Phase {new_phase} initialized")
            logger.info(f"  Remaining epochs: {remaining_epochs}")
            logger.info(f"  Remaining steps: {remaining_steps:,}")
            logger.info("="*80 + "\n")
            
            return True
        
        return False
    
    def _log_phase_progress(self, epoch, val_acc):
        """Log progress within current phase."""
        phase_epoch = epoch - self.phase_start_epoch + 1
        phase_start, phase_end = self.phase_epochs[self.current_phase]
        phase_total = phase_end - phase_start + 1
        
        # Update phase best
        if val_acc > self.phase_best_acc:
            self.phase_best_acc = val_acc
            phase_improved = True
        else:
            phase_improved = False
        
        # Log phase info
        logger.info(
            f"\nPhase {self.current_phase} Progress: "
            f"Epoch {phase_epoch}/{phase_total} | "
            f"Phase Best: {self.phase_best_acc:.2f}%"
            + ("/" if phase_improved else "")
        )
    
    def _save_checkpoint_with_phase(self, epoch, val_acc, is_best):
        """
        Save checkpoint with phase information.
        
        Extended checkpoint includes:
        - Current phase number
        - Phase start epoch
        - Phase best accuracy
        """
        # Use parent's checkpoint manager but add phase info
        self.checkpoint_mgr.save_checkpoint(
            epoch=epoch,
            model=self.model,
            optimizer=self.optimizer,
            scheduler=self.scheduler,
            scaler=self.scaler,
            val_acc=val_acc,
            best_val_acc=self.best_val_acc,
            global_step=self.global_step,
            is_best=is_best,
            # Additional phase info
            additional_state={
                'current_phase': self.current_phase,
                'phase_start_epoch': self.phase_start_epoch,
                'phase_best_acc': self.phase_best_acc,
                'last_unfreeze_epoch': self.last_unfreeze_epoch
            }
        )
    
    def _restore_phase_state(self, checkpoint_path):
        """
        Restore phase state from checkpoint.
        
        This ensures phase transitions happen correctly after resume.
        """
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        
        # Restore phase info if present
        if 'additional_state' in checkpoint:
            state = checkpoint['additional_state']
            self.current_phase = state.get('current_phase', 0)
            self.phase_start_epoch = state.get('phase_start_epoch', 0)
            self.phase_best_acc = state.get('phase_best_acc', 0.0)
            self.last_unfreeze_epoch = state.get('last_unfreeze_epoch', -1)
            
            logger.info(f"\n✓ Phase state restored:")
            logger.info(f"  Current phase: {self.current_phase}")
            logger.info(f"  Phase start epoch: {self.phase_start_epoch}")
            logger.info(f"  Phase best accuracy: {self.phase_best_acc:.2f}%")
        else:
            # Old checkpoint without phase info - infer from epoch
            self.current_phase = self._get_phase_for_epoch(self.current_epoch)
            self.phase_start_epoch = self.phase_epochs[self.current_phase][0]
            logger.info(f"\n No phase state in checkpoint, inferred phase {self.current_phase}")
    
    def train(self, resume_checkpoint: str = None):
        """
        Main progressive training loop with phase transitions.
        
        Workflow:
        1. Resume from checkpoint if provided
        2. Restore phase state
        3. For each epoch:
            a. Check for phase transition
            b. Train epoch
            c. Validate
            d. Save checkpoint with phase info
            e. Log phase progress
        4. Handle early stopping per phase
        
        Args:
            resume_checkpoint: Optional path to checkpoint to resume from
        """
        
        # ====================================================================
        # Resume from Checkpoint
        # ====================================================================
        if resume_checkpoint:
            logger.info(f"\nLoading checkpoint: {resume_checkpoint}")
            
            # Determine resume mode from config
            resume_mode = getattr(self.config.resume, "resume_mode", "full")
            
            if resume_mode == "weights_only":
                # Load ONLY model weights, start fresh training from epoch 0
                logger.info(" Loading weights only — starting fresh fine-tuning from epoch 0")
                
                # Load checkpoint
                checkpoint = torch.load(resume_checkpoint, map_location='cpu')
                self.model.load_state_dict(checkpoint['model_state_dict'])
                
                # Reset all training state
                self.checkpoint_mgr.epoch = 0
                self.checkpoint_mgr.best_val_acc = 0.0
                self.current_epoch = 0
                self.global_step = 0
                self.best_val_acc = 0.0
                
                # Reset phase state
                self.current_phase = 0
                self.phase_start_epoch = 0
                self.phase_best_acc = 0.0
                self.last_unfreeze_epoch = -1
                
                # Keep optimizer/scheduler from __init__ (will be recreated on phase 1 start)
                logger.info(" Weights loaded, starting fresh training")
                
            else:
                # Full resume (resume_mode = "full", "latest", "best", etc.)
                logger.info(f" Resuming full training state (mode: {resume_mode})")
                
                # Load checkpoint using parent's method
                self.checkpoint_mgr.load_checkpoint(
                    resume_checkpoint,
                    self.model,
                    self.optimizer,
                    self.scheduler,
                    self.scaler
                )
                
                # Restore training state
                self.current_epoch = self.checkpoint_mgr.epoch
                self.global_step = self.checkpoint_mgr.global_step
                self.best_val_acc = self.checkpoint_mgr.best_val_acc
                
                # Restore phase state
                self._restore_phase_state(resume_checkpoint)
                
                # Ensure correct unfreezing for current epoch
                self._maybe_update_phase(self.current_epoch)
                
                logger.info(f" Resumed from epoch {self.current_epoch}, phase {self.current_phase}")
        else:
            # Starting fresh - initialize phase 1
            logger.info("\nStarting fresh training")
            self._maybe_update_phase(0)
        
        # ====================================================================
        # Print Training Header
        # ====================================================================
        print("\n" + "="*80)
        print(f"{'PROGRESSIVE FINE-TUNING':^80}")
        print("="*80)
        print(f"  Experiment: {self.config.experiment.name}")
        print(f"  Device: {self.device}")
        print(f"  Batch size: {self.config.training.batch_size} "
              f"(effective: {self.config.training.batch_size * self.grad_accum_steps})")
        print(f"  Total epochs: {self.config.training.num_epochs}")
        print(f"  Current phase: {self.current_phase}/4")
        print(f"  Target: 95%+ validation accuracy")
        print("="*80)
        print("\nPhase Schedule:")
        print("  Phase 1 (epochs  0-7 ):  Blocks [4,5]      | Base LR: 1e-5")
        print("  Phase 2 (epochs  8-15):  Blocks [3,4,5]    | Base LR: 7e-6")
        print("  Phase 3 (epochs 16-23):  Blocks [2,3,4,5]  | Base LR: 5e-6")
        print("  Phase 4 (epochs 24+  ):  All blocks [0-5]  | Base LR: 3e-6")
        print("="*80 + "\n")
        
        # ====================================================================
        # Training Loop
        # ====================================================================
        for epoch in range(self.current_epoch, self.config.training.num_epochs):
            self.current_epoch = epoch
            self.epoch_start_time = time.time()
            self.samples_processed = 0
            
            # Check for phase transition BEFORE training
            phase_changed = self._maybe_update_phase(epoch)
            
            if phase_changed:
                print(f"\n Phase transition completed at epoch {epoch + 1}\n")
            
            self.epoch_start_time = time.time()
            self.samples_processed = 0
            # Train epoch
            train_loss, train_acc = self.train_epoch()
            
            # Validate
            if (epoch + 1) % self.config.validation.frequency == 0:
                val_loss, val_acc = self.validate()
            else:
                val_loss, val_acc = 0.0, 0.0
            
            # Calculate metrics
            epoch_time = time.time() - self.epoch_start_time
            samples_per_sec = self.samples_processed / epoch_time
            current_lr = self.optimizer.param_groups[-1]['lr']  # Classifier LR

            # Calculate AFTER training with current phase state
            lr_display = self._format_layer_lrs()
            trainable_pct = self._get_trainable_percentage()
            
            # Check if best
            is_best = val_acc > self.best_val_acc
            if is_best:
                self.best_val_acc = val_acc
            
            # Save checkpoint with phase info
            if (epoch + 1) % self.config.training.save_frequency == 0:
                self._save_checkpoint_with_phase(epoch + 1, val_acc, is_best)
            
            # Log to TensorBoard
            if self.writer:
                self.writer.add_scalar('Train/Loss', train_loss, epoch)
                self.writer.add_scalar('Train/Accuracy', train_acc, epoch)
                self.writer.add_scalar('Val/Loss', val_loss, epoch)
                self.writer.add_scalar('Val/Accuracy', val_acc, epoch)
                self.writer.add_scalar('LR/Classifier', current_lr, epoch)
                self.writer.add_scalar('LR/Backbone', self.optimizer.param_groups[0]['lr'], epoch)
                self.writer.add_scalar('Throughput', samples_per_sec, epoch)
                self.writer.add_scalar('Phase', self.current_phase, epoch)
            
            # Log to CSV
            metrics = {
                'epoch': epoch + 1,
                'train_loss': train_loss,
                'train_acc': train_acc,
                'val_loss': val_loss,
                'val_acc': val_acc,
                'lr': current_lr,
                'samples_per_sec': samples_per_sec,
                'time_sec': epoch_time
            }
            self._log_metrics(metrics)
            
            lr_display = self._format_layer_lrs()
            trainable_pct = self._get_trainable_percentage()
            # Print epoch summary
            print(
                f"Epoch {epoch + 1:2d}/{self.config.training.num_epochs} "
                f"[Phase {self.current_phase}] | "
                f"Train: {train_loss:.4f}/{train_acc:.2f}% | "
                f"Val: {val_loss:.4f}/{val_acc:.2f}% | "
                f"{samples_per_sec:.0f} samp/s | "
                f"{epoch_time/60:.1f}m | "
                f"LR{lr_display} | "
                f"Params: {trainable_pct:.1f}% | "
                f"T/V Gap: {abs(train_acc - val_acc):.1f}%" 
                + (" 🌟 BEST" if is_best else "")
                + (f" | P-Best: {self.phase_best_acc:.2f}%" if val_acc > 0 else "")
            )
            
            # Log phase progress
            if val_acc > 0:
                self._log_phase_progress(epoch, val_acc)
            
            # Early stopping check
            if self.check_early_stopping(val_acc):
                logger.info(f"\nEarly stopping triggered in Phase {self.current_phase}")
                break
            
       
        
        # Training Complete
        
        print("\n" + "="*80)
        print(f"{'PROGRESSIVE TRAINING COMPLETE':^80}")
        print("="*80)
        print(f"  Final Phase: {self.current_phase}/4")
        print(f"  Best Val Accuracy: {self.best_val_acc:.2f}%")
        print(f"  Global Best: {self.checkpoint_mgr.load_global_best():.2f}%")
        print(f"  Total Epochs Trained: {self.current_epoch + 1}")
        print(f"  Checkpoints: {self.checkpoint_mgr.checkpoint_dir}")
        
        
        if self.writer:
            self.writer.close()