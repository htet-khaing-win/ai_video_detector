"""
Comprehensive metrics for AI content detection evaluation.

Metrics:
- Accuracy, Precision, Recall, F1
- AUC-ROC, AUC-PR
- Confusion matrix
- Per-class metrics
"""

import torch
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    roc_auc_score, average_precision_score,
    confusion_matrix, classification_report
)
from typing import Dict, List, Tuple
import logging

logger = logging.getLogger(__name__)


class MetricsCalculator:
    """Calculate and track evaluation metrics."""
    
    def __init__(self, num_classes: int = 2, class_names: List[str] = None):
        """
        Initialize metrics calculator.
        
        Args:
            num_classes: Number of classes
            class_names: Names of classes for reporting
        """
        self.num_classes = num_classes
        self.class_names = class_names or [f"Class {i}" for i in range(num_classes)]
        
        self.reset()
    
    def reset(self):
        """Reset all accumulated metrics."""
        self.all_preds = []
        self.all_labels = []
        self.all_probs = []
    
    def update(self, 
               predictions: torch.Tensor,
               labels: torch.Tensor,
               probabilities: torch.Tensor = None):
        """
        Update metrics with batch results.
        
        Args:
            predictions: Predicted class indices [B]
            labels: Ground truth labels [B]
            probabilities: Class probabilities [B, num_classes] (optional)
        """
        # Convert to numpy
        preds_np = predictions.cpu().numpy()
        labels_np = labels.cpu().numpy()
        
        self.all_preds.extend(preds_np)
        self.all_labels.extend(labels_np)
        
        if probabilities is not None:
            probs_np = probabilities.cpu().numpy()
            self.all_probs.extend(probs_np)
    
    def compute(self) -> Dict[str, float]:
        """
        Compute all metrics.
        
        Returns:
            Dictionary of metric name -> value
        """
        if len(self.all_preds) == 0:
            return {}
        
        preds = np.array(self.all_preds)
        labels = np.array(self.all_labels)
        
        metrics = {}
        
        # Accuracy
        metrics['accuracy'] = accuracy_score(labels, preds)
        
        # Precision, Recall, F1 (macro and per-class)
        precision, recall, f1, support = precision_recall_fscore_support(
            labels, preds, average=None, zero_division=0
        )
        
        # Macro averages
        metrics['precision_macro'] = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )[0]
        metrics['recall_macro'] = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )[1]
        metrics['f1_macro'] = precision_recall_fscore_support(
            labels, preds, average='macro', zero_division=0
        )[2]
        
        # Per-class metrics
        for i, class_name in enumerate(self.class_names):
            metrics[f'precision_{class_name}'] = precision[i]
            metrics[f'recall_{class_name}'] = recall[i]
            metrics[f'f1_{class_name}'] = f1[i]
            metrics[f'support_{class_name}'] = support[i]
        
        # Confusion matrix
        cm = confusion_matrix(labels, preds)
        metrics['confusion_matrix'] = cm.tolist()
        
        # For binary classification, compute additional metrics
        if self.num_classes == 2 and len(self.all_probs) > 0:
            probs = np.array(self.all_probs)
            
            # ROC-AUC
            try:
                metrics['roc_auc'] = roc_auc_score(labels, probs[:, 1])
            except:
                metrics['roc_auc'] = 0.0
            
            # Average Precision (AUC-PR)
            try:
                metrics['avg_precision'] = average_precision_score(labels, probs[:, 1])
            except:
                metrics['avg_precision'] = 0.0
            
            # Compute true/false positives/negatives
            tn, fp, fn, tp = cm.ravel()
            
            # Specificity
            metrics['specificity'] = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            
            # False Positive Rate
            metrics['fpr'] = fp / (fp + tn) if (fp + tn) > 0 else 0.0
            
            # False Negative Rate
            metrics['fnr'] = fn / (fn + tp) if (fn + tp) > 0 else 0.0
        
        return metrics
    
    def get_summary(self) -> str:
        """
        Get formatted summary of metrics.
        
        Returns:
            Formatted string summary
        """
        metrics = self.compute()
        
        if not metrics:
            return "No metrics available"
        
        summary = "\n" + "="*80 + "\n"
        summary += f"{'EVALUATION METRICS':^80}\n"
        summary += "="*80 + "\n\n"
        
        # Overall metrics
        summary += "Overall Performance:\n"
        summary += f"  Accuracy:  {metrics['accuracy']:.4f}\n"
        summary += f"  Precision: {metrics['precision_macro']:.4f}\n"
        summary += f"  Recall:    {metrics['recall_macro']:.4f}\n"
        summary += f"  F1-Score:  {metrics['f1_macro']:.4f}\n"
        
        if 'roc_auc' in metrics:
            summary += f"  ROC-AUC:   {metrics['roc_auc']:.4f}\n"
            summary += f"  Avg Precision: {metrics['avg_precision']:.4f}\n"
        
        # Per-class metrics
        summary += "\nPer-Class Performance:\n"
        for class_name in self.class_names:
            summary += f"\n  {class_name}:\n"
            summary += f"    Precision: {metrics[f'precision_{class_name}']:.4f}\n"
            summary += f"    Recall:    {metrics[f'recall_{class_name}']:.4f}\n"
            summary += f"    F1-Score:  {metrics[f'f1_{class_name}']:.4f}\n"
            summary += f"    Support:   {int(metrics[f'support_{class_name}'])}\n"
        
        # Confusion matrix
        if 'confusion_matrix' in metrics:
            cm = np.array(metrics['confusion_matrix'])
            summary += "\nConfusion Matrix:\n"
            summary += f"  {'Predicted →':>15}"
            for class_name in self.class_names:
                summary += f"{class_name:>12}"
            summary += "\n"
            
            for i, class_name in enumerate(self.class_names):
                summary += f"  {class_name:>15}"
                for j in range(len(self.class_names)):
                    summary += f"{cm[i,j]:>12d}"
                summary += "\n"
        
        summary += "\n" + "="*80 + "\n"
        
        return summary


class AverageMeter:
    """Track running average of a metric."""
    
    def __init__(self, name: str = "metric"):
        self.name = name
        self.reset()
    
    def reset(self):
        """Reset meter."""
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
    
    def update(self, val: float, n: int = 1):
        """
        Update meter with new value.
        
        Args:
            val: New value
            n: Number of samples (for weighted average)
        """
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count > 0 else 0
    
    def __str__(self) -> str:
        return f"{self.name}: {self.avg:.4f}"


class MetricsTracker:
    """Track metrics across epochs."""
    
    def __init__(self):
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': [],
            'val_f1': [],
            'val_roc_auc': [],
            'learning_rate': []
        }
    
    def update(self, epoch: int, metrics: Dict[str, float]):
        """
        Update history with epoch metrics.
        
        Args:
            epoch: Epoch number
            metrics: Dictionary of metrics
        """
        for key, value in metrics.items():
            if key in self.history:
                self.history[key].append(value)
    
    def get_best_epoch(self, metric: str = 'val_acc') -> Tuple[int, float]:
        """
        Get best epoch for given metric.
        
        Args:
            metric: Metric name to optimize
            
        Returns:
            (best_epoch, best_value)
        """
        if metric not in self.history or not self.history[metric]:
            return 0, 0.0
        
        values = self.history[metric]
        best_idx = np.argmax(values)
        return best_idx + 1, values[best_idx]
    
    def save(self, filepath: str):
        """Save history to file."""
        import json
        with open(filepath, 'w') as f:
            json.dump(self.history, f, indent=2)
    
    def load(self, filepath: str):
        """Load history from file."""
        import json
        with open(filepath, 'r') as f:
            self.history = json.load(f)


# ============================================================================
# Utility Functions
# ============================================================================

def compute_accuracy(predictions: torch.Tensor, labels: torch.Tensor) -> float:
    """
    Compute accuracy for a batch.
    
    Args:
        predictions: Predicted class indices [B]
        labels: Ground truth labels [B]
        
    Returns:
        Accuracy as float
    """
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0.0


def compute_topk_accuracy(logits: torch.Tensor, 
                          labels: torch.Tensor, 
                          k: int = 1) -> float:
    """
    Compute top-k accuracy.
    
    Args:
        logits: Model output logits [B, num_classes]
        labels: Ground truth labels [B]
        k: Top-k value
        
    Returns:
        Top-k accuracy as float
    """
    with torch.no_grad():
        _, pred = logits.topk(k, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(labels.view(1, -1).expand_as(pred))
        correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
        return correct_k.item() / labels.size(0)


# ============================================================================
# Testing
# ============================================================================

def test_metrics():
    """Test metrics calculation."""
    print("\n" + "="*80)
    print("Testing Metrics Calculator")
    print("="*80 + "\n")
    
    # Create dummy data
    num_samples = 100
    labels = torch.randint(0, 2, (num_samples,))
    logits = torch.randn(num_samples, 2)
    predictions = logits.argmax(dim=1)
    probabilities = torch.softmax(logits, dim=1)
    
    # Calculate metrics
    calculator = MetricsCalculator(
        num_classes=2,
        class_names=['Real', 'AI-Generated']
    )
    
    calculator.update(predictions, labels, probabilities)
    
    # Print summary
    print(calculator.get_summary())
    
    # Get metrics dict
    metrics = calculator.compute()
    print("\nMetrics Dictionary Keys:")
    for key in sorted(metrics.keys()):
        if key != 'confusion_matrix':
            print(f"  {key}: {metrics[key]:.4f}" if isinstance(metrics[key], float) else f"  {key}: {metrics[key]}")
    
    print("\n Metrics test passed!\n")
    print("="*80 + "\n")


if __name__ == "__main__":
    test_metrics()