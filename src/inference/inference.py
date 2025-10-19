# inference.py - Production inference with optimized threshold
import torch
import numpy as np
from pathlib import Path

from src.models.baseline_cnn import BaselineCNN


class OptimizedClassifier:
    def __init__(self, model_path, threshold_config_path, device='cuda'):
        self.device = device if torch.cuda.is_available() else 'cpu'
        
        self.model = BaselineCNN(
            num_classes=2,
            pretrained=True,
            temporal_pool="attention",
            dropout=0.5
        ).to(self.device)
        
        checkpoint = torch.load(model_path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.model.eval()
        
        threshold_data = np.load(threshold_config_path, allow_pickle=True)
        self.threshold = float(threshold_data['threshold'])
        
        print(f"Loaded model with optimized threshold: {self.threshold:.4f}")
    
    @torch.no_grad()
    def predict(self, frames):
        """
        Predict using optimized threshold.
        
        Args:
            frames: torch.Tensor of shape (B, C, T, H, W)
        
        Returns:
            predictions: numpy array of class predictions (0=Real, 1=AI)
            probabilities: numpy array of AI class probabilities
        """
        frames = frames.to(self.device)
        logits = self.model(frames)
        probs = torch.softmax(logits, dim=1)[:, 1]
        
        predictions = (probs >= self.threshold).cpu().numpy().astype(int)
        probabilities = probs.cpu().numpy()
        
        return predictions, probabilities
    
    def predict_single(self, frames):
        """Convenience method for single video."""
        if len(frames.shape) == 4:
            frames = frames.unsqueeze(0)
        
        preds, probs = self.predict(frames)
        return preds[0], probs[0]


if __name__ == "__main__":
    classifier = OptimizedClassifier(
        model_path='models/baseline_cnn_best.pt',
        threshold_config_path='models/threshold_config.npz'
    )
    
    print("\nClassifier ready for inference")
    print(f"Using threshold: {classifier.threshold:.4f}")
    print("\nUsage:")
    print("  predictions, probabilities = classifier.predict(video_frames)")