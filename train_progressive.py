"""
Usage:
    python train_progressive.py --config config/finetune_stage3.yaml
    
🔥 FIXED: Model wrapping and GPU allocation bugs
"""

import sys
from pathlib import Path
project_root = Path("C:/Personal project/ai_video_detector")
sys.path.insert(0, str(project_root))

import torch
import torch.nn as nn
from torch.optim import AdamW
import argparse
import random
import numpy as np
import logging
import warnings
warnings.filterwarnings('ignore')

from utils.config import load_config
from src.data.dataset import create_dataloader
from src.models.x3d import create_x3d_model, get_x3d_m_vram_estimate
from src.training.trainer_progressive import ProgressiveTrainer


def setup_logging(config):
    """Setup logging based on config."""
    level = logging.INFO if config.logging.verbose else logging.ERROR
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )


def set_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    
    if torch.cuda.is_available():
        torch.backends.cudnn.deterministic = False
        torch.backends.cudnn.benchmark = True


def setup_cuda_optimizations(config):
    """Apply CUDA optimizations from config."""
    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = config.hardware.cudnn_benchmark
        torch.backends.cudnn.deterministic = config.hardware.cudnn_deterministic
        torch.backends.cuda.matmul.allow_tf32 = config.hardware.tf32_matmul
        torch.backends.cudnn.allow_tf32 = config.hardware.tf32_matmul


def get_base_model(model):
    """
    🔥 FIX: Reliably unwrap model to get base X3D structure.
    
    Handles various wrapping patterns:
    - Direct X3D model
    - X3DWrapper -> X3D
    - DataParallel/DistributedDataParallel wrappers
    """
    base = model
    
    # Unwrap any DataParallel/DDP wrappers
    if hasattr(base, 'module'):
        base = base.module
    
    # Unwrap custom wrappers
    while hasattr(base, 'model'):
        base = base.model
    
    # Verify we have X3D structure
    if not hasattr(base, 'blocks'):
        raise AttributeError(
            f"Cannot find X3D 'blocks' in model structure. "
            f"Model type: {type(base)}, attributes: {list(dir(base))[:10]}..."
        )
    
    return base


def setup_progressive_unfreezing(model, epoch, logger):
    """
    Setup progressive unfreezing based on epoch.
    🔥 FIXED: Use consistent model unwrapping
    """
    # Determine phase and blocks to unfreeze
    if epoch < 8:
        phase = 1
        unfreeze_blocks = [4, 5]
        phase_desc = "Classifier + Late Features"
    elif 8 <= epoch < 16:
        phase = 2
        unfreeze_blocks = [3, 4, 5]
        phase_desc = "Mid to Late Features"
    elif 16 <= epoch < 24:
        phase = 3
        unfreeze_blocks = [2, 3, 4, 5]
        phase_desc = "Early-Mid to Late Features"
    else:
        phase = 4
        unfreeze_blocks = [0, 1, 2, 3, 4, 5]
        phase_desc = "All Layers"

    # 🔥 FIX: Use consistent unwrapping function
    x3d_model = get_base_model(model)

    # ---------- CLEAN BASELINE: freeze everything first ----------
    for param in x3d_model.parameters():
        param.requires_grad = False

    # ---------- SELECTIVELY UNFREEZE requested blocks ----------
    for block_idx in unfreeze_blocks:
        for param in x3d_model.blocks[block_idx].parameters():
            param.requires_grad = True

    # Ensure final classifier projection (if present) is trainable
    try:
        head = x3d_model.blocks[5].proj
        for param in head.parameters():
            param.requires_grad = True
    except Exception:
        pass

    # ---------- Logging: per-block trainable param counts ----------
    # 🔥 FIX: Count from base model, not wrapper
    total_params = sum(p.numel() for p in x3d_model.parameters())
    trainable_params = sum(p.numel() for p in x3d_model.parameters() if p.requires_grad)
    frozen_params = total_params - trainable_params

    # Detailed block breakdown
    block_info_lines = []
    for i, block in enumerate(x3d_model.blocks):
        block_total = sum(p.numel() for p in block.parameters())
        block_trainable = sum(p.numel() for p in block.parameters() if p.requires_grad)
        block_info_lines.append(f"Block {i}: {block_trainable:,}/{block_total:,} trainable")

    logger.info(f"Phase {phase}: Unfreezing blocks {unfreeze_blocks} ({phase_desc})")
    logger.info(f"  Trainable: {trainable_params:,} / {total_params:,} ({trainable_params/total_params*100:.1f}%)")
    logger.info(f"  Frozen: {frozen_params:,}")
    for line in block_info_lines:
        logger.info(f"  {line}")

    return phase, unfreeze_blocks


def create_discriminative_optimizer(model, epoch, config, logger):
    """
    Create optimizer with discriminative learning rates per block.
    🔥 FIXED: Use consistent model unwrapping and proper parameter collection
    """
    # 🔥 FIX: Use consistent unwrapping
    x3d_model = get_base_model(model)
    
    # Determine base LR and active blocks based on phase
    if epoch < 8:
        base_lr = 1e-5
        lr_map = {
            4: base_lr * 0.5,
            5: base_lr * 1.0
        }
    elif 8 <= epoch < 16:
        base_lr = 7e-6
        lr_map = {
            3: base_lr * 0.2,
            4: base_lr * 0.5,
            5: base_lr * 1.0
        }
    elif 16 <= epoch < 24:
        base_lr = 5e-6
        lr_map = {
            2: base_lr * 0.1,
            3: base_lr * 0.2,
            4: base_lr * 0.5,
            5: base_lr * 1.0
        }
    else:
        base_lr = 5e-6
        lr_map = {
            0: base_lr * 0.05,
            1: base_lr * 0.05,
            2: base_lr * 0.1,
            3: base_lr * 0.2,
            4: base_lr * 0.5,
            5: base_lr * 1.0
        }
    
    # 🔥 FIX: Track parameter IDs to detect duplicates
    seen_param_ids = set()
    
    # Group parameters by block
    param_groups = []
    logger.info(f"Discriminative Learning Rates (base={base_lr:.2e}):")
    
    for block_idx in sorted(lr_map.keys()):
        lr = lr_map[block_idx]
        block_params = []
        
        # Collect trainable parameters from this block
        for p in x3d_model.blocks[block_idx].parameters():
            if p.requires_grad:
                param_id = id(p)
                
                # Check for duplicates
                if param_id in seen_param_ids:
                    logger.error(f"❌ DUPLICATE PARAMETER in Block {block_idx}!")
                    raise ValueError(f"Parameter {param_id} appears multiple times!")
                
                seen_param_ids.add(param_id)
                block_params.append(p)
        
        if block_params:
            param_count = sum(p.numel() for p in block_params)
            param_groups.append({
                'params': block_params,
                'lr': lr,
                'weight_decay': config.optimizer.weight_decay
            })
            logger.info(f"  Block {block_idx}: LR={lr:.2e} ({param_count:,} params)")
    
    # 🔥 FIX: Proper verification using base model
    total_params_in_groups = sum(p.numel() for group in param_groups for p in group['params'])
    total_trainable = sum(p.numel() for p in x3d_model.parameters() if p.requires_grad)
    
    logger.info(f"\n✓ Optimizer verification:")
    logger.info(f"  Groups created: {len(param_groups)}")
    logger.info(f"  Unique param tensors: {len(seen_param_ids)}")
    logger.info(f"  Total elements in groups: {total_params_in_groups:,}")
    logger.info(f"  Total trainable elements: {total_trainable:,}")
    
    # 🔥 FIX: Check element count, not tensor count
    if total_params_in_groups != total_trainable:
        logger.error(f"❌ MISMATCH: Groups have {total_params_in_groups:,} params but model has {total_trainable:,} trainable!")
        logger.error(f"   This means some parameters are missing from optimizer!")
        raise ValueError("Parameter counting mismatch - some parameters not being optimized!")
    
    # Create optimizer
    optimizer = AdamW(
        param_groups,
        betas=tuple(config.optimizer.betas),
        eps=config.optimizer.eps
    )
    
    logger.info(f"✓ Optimizer created successfully with {total_params_in_groups:,} parameters")
    
    return optimizer


def create_scheduler(optimizer, config, num_training_steps, logger):
    """Create learning rate scheduler."""
    from torch.optim.lr_scheduler import LambdaLR
    
    scheduler_type = config.scheduler.type.lower()
    
    if scheduler_type == 'cosine':
        warmup_steps = config.scheduler.warmup_epochs * (num_training_steps // config.training.num_epochs)
        
        def lr_lambda(current_step):
            if current_step < warmup_steps:
                return float(current_step) / float(max(1, warmup_steps))
            else:
                progress = float(current_step - warmup_steps) / float(max(1, num_training_steps - warmup_steps))
                return max(0.1, 0.5 * (1.0 + np.cos(np.pi * progress)))
        
        scheduler = LambdaLR(optimizer, lr_lambda)
        logger.info(f"Scheduler: Cosine with {config.scheduler.warmup_epochs} epoch warmup")
        logger.info(f"  Warmup steps: {warmup_steps:,}")
        logger.info(f"  Total steps: {num_training_steps:,}")
    else:
        scheduler = None
        logger.info("Scheduler: None")
    
    return scheduler


def create_criterion(config):
    """Create loss function from config."""
    if config.loss.type.lower() == 'cross_entropy':
        return nn.CrossEntropyLoss(
            label_smoothing=config.loss.label_smoothing
        )
    else:
        raise ValueError(f"Unsupported loss: {config.loss.type}")


def print_system_info():
    """Print system and hardware information."""
    print("\n" + "="*80)
    print(f"{'SYSTEM INFORMATION':^80}")
    print("="*80)
    
    if torch.cuda.is_available():
        print(f"  Device: {torch.cuda.get_device_name(0)}")
        print(f"  CUDA: {torch.version.cuda}")
        print(f"  VRAM: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")
        print(f"  Compute: {torch.cuda.get_device_capability(0)}")
    else:
        print("  Device: CPU only (WARNING: Training will be very slow)")
    
    print(f"  PyTorch: {torch.__version__}")
    print("="*80)


def main():
    """Main progressive fine-tuning function."""
    
    parser = argparse.ArgumentParser(description='Progressive Fine-Tuning for X3D-M')
    parser.add_argument(
        '--config',
        type=str,
        default='config/finetune_progressive.yaml',
        help='Path to config file'
    )
    parser.add_argument(
        '--resume',
        action='store_true',
        help='Resume from latest checkpoint'
    )
    args = parser.parse_args()
    
    # Load configuration
    print(f"\nLoading configuration from: {args.config}")
    config = load_config(args.config)
    
    # Setup
    setup_logging(config)
    logger = logging.getLogger(__name__)
    set_seed(config.seed)
    setup_cuda_optimizations(config)
    
    config.print_summary()
    print_system_info()
    
    # Device
    device = torch.device(config.hardware.device)
    if device.type == 'cuda' and not torch.cuda.is_available():
        print("ERROR: CUDA requested but not available")
        return
    
    # Save config
    exp_dir = Path(config.experiment.output_dir) / config.experiment.name
    exp_dir.mkdir(parents=True, exist_ok=True)
    config.save(exp_dir / 'config.yaml')
    
    # Data Loading
    print("\n" + "="*80)
    print(f"{'LOADING DATA':^80}")
    print("="*80)
    
    train_loader = create_dataloader(config, mode='train')
    val_loader = create_dataloader(config, mode='val')
    
    print(f"\n✓ Data loading complete")
    print(f"  Training batches: {len(train_loader):,}")
    print(f"  Validation batches: {len(val_loader):,}")
    
    # Model Creation
    print("\n" + "="*80)
    print(f"{'CREATING MODEL':^80}")
    print("="*80)
    
    model = create_x3d_model(config)
    
    # 🔥 FIX: CRITICAL - Move model to GPU BEFORE any parameter operations
    logger.info(f"Moving model to {device}...")
    model = model.to(device)
    logger.info(f"✓ Model on {device}")
    
    # 🔥 FIX: Verify model is actually on GPU
    if device.type == 'cuda':
        sample_param = next(model.parameters())
        if not sample_param.is_cuda:
            logger.error("❌ Model parameters are NOT on GPU after .to(device)!")
            raise RuntimeError("Model failed to move to GPU")
        logger.info(f"✓ Verified: Model parameters on {sample_param.device}")
    
    # Estimate VRAM
    estimated_vram = get_x3d_m_vram_estimate(
        config.training.batch_size,
        config.training.use_amp
    )
    print(f"\n  Estimated VRAM usage: {estimated_vram:.2f} GB")
    
    if estimated_vram > 7.8:
        print("  ⚠️  WARNING: May exceed 8GB VRAM. Consider reducing batch_size.")
    
    # Progressive Training Components
    print("\n" + "="*80)
    print(f"{'PROGRESSIVE FINE-TUNING SETUP':^80}")
    print("="*80)
    print("\nPhase Schedule (50 epochs → 95%+ target):")
    print("  Phase 1 (epochs  0-7 ):  Unfreeze blocks [4,5]        - Conservative")
    print("  Phase 2 (epochs  8-15):  Unfreeze blocks [3,4,5]      - Moderate")
    print("  Phase 3 (epochs 16-23):  Unfreeze blocks [2,3,4,5]    - Deep")
    print("  Phase 4 (epochs 24+  ):  Unfreeze all blocks [0-5]    - Full")
    print("\nStrategy:")
    print("  ✓ Gradual unfreezing (prevents catastrophic forgetting)")
    print("  ✓ Discriminative LR (layer-wise learning rates)")
    print("  ✓ Decreasing base LR per phase (stability)")
    print("  ✓ Memory-safe phase transitions (cleanup old optimizer)")
    print("  ✓ Resume-friendly (stop/start anytime)")
    print("="*80)
    
    # Create criterion
    criterion = create_criterion(config)
    
    # 🔥 FIX: Initial unfreezing AFTER model is on GPU
    phase, unfreeze_blocks = setup_progressive_unfreezing(model, 0, logger)
    optimizer = create_discriminative_optimizer(model, 0, config, logger)
    
    # Calculate training steps
    num_steps_per_epoch = len(train_loader) // config.training.gradient_accumulation_steps
    num_training_steps = num_steps_per_epoch * config.training.num_epochs
    
    scheduler = create_scheduler(optimizer, config, num_training_steps, logger)
    
    print(f"\n✓ Training components initialized")
    print(f"  Training steps: {num_training_steps:,}")
    print(f"  Steps per epoch: {num_steps_per_epoch:,}")
    
    # 🔥 FIX: Log initial VRAM state
    if device.type == 'cuda':
        logger.info(f"\n📊 Initial VRAM state:")
        logger.info(f"  Allocated: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
        logger.info(f"  Reserved:  {torch.cuda.memory_reserved() / 1e9:.2f} GB")
    
    # Create trainer
    trainer = ProgressiveTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        criterion=criterion,
        optimizer=optimizer,
        scheduler=scheduler,
        device=device,
        config=config
    )
    
    # Resume handling
    resume_checkpoint = None
    if args.resume or config.resume.enabled:
        if args.resume:
            resume_mode = "full"
            try:
                resume_checkpoint = str(trainer.checkpoint_mgr.find_latest_checkpoint())
                print(f"\n✓ Found checkpoint: {resume_checkpoint}")
            except FileNotFoundError:
                print("\n⚠  No checkpoint found, starting from scratch")
        else:
            resume_mode = config.resume.resume_mode
            if config.resume.checkpoint_path:
                resume_checkpoint = config.resume.checkpoint_path
            else:
                try:
                    resume_checkpoint = str(trainer.checkpoint_mgr.find_latest_checkpoint())
                except FileNotFoundError:
                    print("\n⚠  No checkpoint found")
    
        if resume_checkpoint and hasattr(config.resume, 'resume_mode'):
            config.resume.resume_mode = resume_mode
    
    # Start training
    try:
        trainer.train(resume_checkpoint=resume_checkpoint)
    except KeyboardInterrupt:
        print("\n\n⚠  Training interrupted by user")
        print("✓ Latest checkpoint saved. Resume with --resume flag.")
    except Exception as e:
        print(f"\n\n❌ Training failed with error: {e}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()