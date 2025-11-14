import torch
import torch.nn as nn
from pytorchvideo.models.x3d import create_x3d
import logging

logger = logging.getLogger(__name__)


class X3DWrapper(nn.Module):
    """Wrapper that handles uint8→float32 normalization on GPU."""
    
    def __init__(self, x3d_model):
        super().__init__()
        self.model = x3d_model
    
    def forward(self, x):
        # Normalize uint8 [0-255] to float32 [0-1] on GPU (fast)
        if x.dtype == torch.uint8:
            x = x.float().div_(255.0)
        
        return self.model(x)


def create_x3d_model(config) -> nn.Module:
    """
    Create X3D-M model from config.
    """
    num_classes = config.model.num_classes
    pretrained = config.model.pretrained
    dropout_rate = config.model.dropout_rate
    freeze_backbone = config.model.freeze_backbone
    
    logger.info(f"\nInitializing X3D-M model...")
    logger.info(f"  Pretrained: {pretrained}")
    logger.info(f"  Backbone: {'FROZEN' if freeze_backbone else 'TRAINABLE'}")
    logger.info(f"  Dropout: {dropout_rate}")
    
    # Create X3D-M model
    model = create_x3d(
        input_clip_length=16,
        input_crop_size=224,
        model_num_class=400  # Kinetics-400 pretrained classes
    )
    
    # Load pretrained weights
    if pretrained:
        try:
            logger.info("  Loading Kinetics-400 pretrained weights...")
            checkpoint_url = 'https://dl.fbaipublicfiles.com/pytorchvideo/model_zoo/kinetics/X3D_M.pyth'
            
            checkpoint = torch.hub.load_state_dict_from_url(
                checkpoint_url,
                progress=True,
                map_location='cpu'
            )
            model.load_state_dict(checkpoint, strict=False)
            logger.info("  ✓ Pretrained weights loaded")
        except Exception as e:
            logger.warning(f"  ⚠ Could not load pretrained weights: {e}")
            logger.warning("  Training from scratch...")
    
    # Freeze backbone if requested
    if freeze_backbone:
        logger.info("  Freezing backbone layers...")
        
        # Freeze all parameters except the final projection head
        for name, param in model.named_parameters():
            if 'blocks.5.proj' not in name:  # blocks.5.proj is the final classifier
                param.requires_grad = False
        
        frozen_params = sum(p.numel() for p in model.parameters() if not p.requires_grad)
        logger.info(f"   Frozen {frozen_params:,} parameters")
    
    # Replace final classification head for binary classification
    in_features = model.blocks[5].proj.in_features
    model.blocks[5].proj = nn.Sequential(
        nn.Dropout(p=dropout_rate),
        nn.Linear(in_features, num_classes)
    )
    
    # Always make classifier trainable (even in frozen mode)
    for param in model.blocks[5].proj.parameters():
        param.requires_grad = True
    
    # Model statistics
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    logger.info(f"\n  Model Statistics:")
    logger.info(f"  Total parameters: {total_params:,}")
    logger.info(f"  Trainable parameters: {trainable_params:,}")
    logger.info(f"  Frozen parameters: {total_params - trainable_params:,}")
    logger.info(f"  Trainable ratio: {trainable_params/total_params*100:.1f}%")
    logger.info(f"  Model size: ~{total_params * 4 / 1024**2:.1f} MB (FP32)")
    
    # Wrap model with GPU normalization
    model = X3DWrapper(model)
    logger.info(f"   Wrapped with GPU normalization layer")
    
    return model


def get_x3d_m_vram_estimate(batch_size: int, use_amp: bool = True) -> float:
    """
    Estimate VRAM usage for X3D-M.
    
    Args:
        batch_size: Batch size
        use_amp: Whether using mixed precision
        
    Returns:
        Estimated VRAM in GB
    """
    # X3D-M model size (FP32)
    model_size_mb = 50.7
    
    # Activation memory per sample (estimated for 16 frames @ 224x224)
    activation_mb_per_sample = 250
    activation_mb = activation_mb_per_sample * batch_size
    
    # Mixed precision reduces memory by ~40%
    if use_amp:
        model_size_mb *= 0.6
        activation_mb *= 0.6
    
    # Gradients + optimizer states (AdamW)
    gradient_mb = model_size_mb * 0.5
    optimizer_mb = model_size_mb * 0.3
    
    # Total VRAM
    total_mb = model_size_mb + activation_mb + gradient_mb + optimizer_mb
    
    return total_mb / 1024  # Convert to GB