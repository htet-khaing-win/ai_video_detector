"""
Configuration management system.

Loads YAML config and provides typed access to all parameters.
"""

import yaml
from pathlib import Path
from typing import Any, Dict
from dataclasses import dataclass, field


@dataclass
class ExperimentConfig:
    """Experiment metadata."""
    name: str
    description: str
    output_dir: str


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    name: str
    num_classes: int
    pretrained: bool
    dropout_rate: float
    freeze_backbone: bool


@dataclass
class TrainingConfig:
    """Training loop configuration."""
    num_epochs: int
    batch_size: int
    gradient_accumulation_steps: int
    num_workers: int
    pin_memory: bool
    prefetch_factor: int
    persistent_workers: bool
    use_amp: bool
    compile_model: bool
    max_grad_norm: float
    save_frequency: int
    keep_last_n: int


@dataclass
class OptimizerConfig:
    """Optimizer configuration."""
    type: str
    learning_rate: float
    weight_decay: float
    betas: list
    eps: float


@dataclass
class SchedulerConfig:
    """Learning rate scheduler configuration."""
    type: str
    warmup_epochs: int
    min_lr: float
    step_size: int = 10
    gamma: float = 0.1
    patience: int = 5
    factor: float = 0.5


@dataclass
class LossConfig:
    """Loss function configuration."""
    type: str
    label_smoothing: float


@dataclass
class AugmentationConfig:
    """Data augmentation configuration."""
    horizontal_flip: float
    color_jitter: bool
    brightness: float
    contrast: float
    saturation: float


@dataclass
class DataConfig:
    """Dataset paths and properties."""
    cache_root: str
    train_csv: str
    val_csv: str
    num_frames: int
    resolution: int
    skip_missing_cache: bool
    max_skip_warnings: int
    preload_to_ram: bool


@dataclass
class LoggingConfig:
    """Logging configuration."""
    verbose: bool
    progress_bar: bool
    use_tensorboard: bool
    log_interval: int
    save_metrics_csv: bool


@dataclass
class HardwareConfig:
    """Hardware and CUDA optimization configuration."""
    device: str
    gpu_id: int
    cudnn_benchmark: bool
    cudnn_deterministic: bool
    tf32_matmul: bool
    empty_cache_frequency: int


@dataclass
class ResumeConfig:
    """Resume training configuration."""
    enabled: bool
    checkpoint_path: Any
    resume_mode: str


@dataclass
class ValidationConfig:
    """Validation configuration."""
    frequency: int
    save_predictions: bool


@dataclass
class EarlyStoppingConfig:
    """Early stopping configuration."""
    enabled: bool
    patience: int
    min_delta: float


@dataclass
class Config:
    """
    Complete configuration object.
    
    Single source of truth for all training parameters.
    """
    experiment: ExperimentConfig
    model: ModelConfig
    training: TrainingConfig
    optimizer: OptimizerConfig
    scheduler: SchedulerConfig
    loss: LossConfig
    augmentation: AugmentationConfig
    data: DataConfig
    logging: LoggingConfig
    hardware: HardwareConfig
    resume: ResumeConfig
    validation: ValidationConfig
    early_stopping: EarlyStoppingConfig
    seed: int
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> 'Config':
        """
        Load configuration from YAML file.
        
        Args:
            yaml_path: Path to YAML config file
            
        Returns:
            Config object with all parameters loaded
        """
        yaml_path = Path(yaml_path)
        
        if not yaml_path.exists():
            raise FileNotFoundError(f"Config file not found: {yaml_path}")
        
        with open(yaml_path, 'r', encoding='utf-8') as f:
            config_dict = yaml.safe_load(f)

        
        return cls(
            experiment=ExperimentConfig(**config_dict['experiment']),
            model=ModelConfig(**config_dict['model']),
            training=TrainingConfig(**config_dict['training']),
            optimizer=OptimizerConfig(**config_dict['optimizer']),
            scheduler=SchedulerConfig(**config_dict['scheduler']),
            loss=LossConfig(**config_dict['loss']),
            augmentation=AugmentationConfig(**config_dict['augmentation']),
            data=DataConfig(**config_dict['data']),
            logging=LoggingConfig(**config_dict['logging']),
            hardware=HardwareConfig(**config_dict['hardware']),
            resume=ResumeConfig(**config_dict['resume']),
            validation=ValidationConfig(**config_dict['validation']),
            early_stopping=EarlyStoppingConfig(**config_dict['early_stopping']),
            seed=config_dict['seed']
        )
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary (for saving/logging)."""
        return {
            'experiment': self.experiment.__dict__,
            'model': self.model.__dict__,
            'training': self.training.__dict__,
            'optimizer': self.optimizer.__dict__,
            'scheduler': self.scheduler.__dict__,
            'loss': self.loss.__dict__,
            'augmentation': self.augmentation.__dict__,
            'data': self.data.__dict__,
            'logging': self.logging.__dict__,
            'hardware': self.hardware.__dict__,
            'resume': self.resume.__dict__,
            'validation': self.validation.__dict__,
            'early_stopping': self.early_stopping.__dict__,
            'seed': self.seed
        }
    
    def save(self, output_path: str):
        """Save configuration to YAML file."""
        with open(output_path, 'w') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)
    
    def print_summary(self):
        """Print human-readable configuration summary."""
        print("\n" + "="*80)
        print(f"{'CONFIGURATION SUMMARY':^80}")
        print("="*80)
        
        print(f"\n EXPERIMENT")
        print(f"  Name: {self.experiment.name}")
        print(f"  Description: {self.experiment.description}")
        
        print(f"\n MODEL")
        print(f"  Architecture: {self.model.name.upper()}")
        print(f"  Pretrained: {self.model.pretrained}")
        print(f"  Backbone: {'FROZEN' if self.model.freeze_backbone else 'TRAINABLE'}")
        print(f"  Classes: {self.model.num_classes}")
        
        print(f"\n TRAINING")
        print(f"  Epochs: {self.training.num_epochs}")
        print(f"  Batch size: {self.training.batch_size}")
        print(f"  Gradient accumulation: {self.training.gradient_accumulation_steps}")
        effective_batch = self.training.batch_size * self.training.gradient_accumulation_steps
        print(f"  Effective batch: {effective_batch}")
        print(f"  Mixed precision: {self.training.use_amp}")
        
        print(f"\n  OPTIMIZER")
        print(f"  Type: {self.optimizer.type.upper()}")
        print(f"  Learning rate: {self.optimizer.learning_rate}")
        print(f"  Weight decay: {self.optimizer.weight_decay}")
        
        print(f"\n DATA")
        print(f"  Cache: {self.data.cache_root}")
        print(f"  Skip missing: {self.data.skip_missing_cache}")
        print(f"  Workers: {self.training.num_workers}")
        
        print(f"\n HARDWARE")
        print(f"  Device: {self.hardware.device}")
        print(f"  CUDNN benchmark: {self.hardware.cudnn_benchmark}")
        print(f"  TF32: {self.hardware.tf32_matmul}")
        
        print("="*80 + "\n")


def load_config(config_path: str = "config/train_config.yaml") -> Config:
    """
    Convenience function to load configuration.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Loaded Config object
    """
    return Config.from_yaml(config_path)