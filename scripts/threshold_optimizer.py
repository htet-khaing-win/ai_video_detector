import torch
import numpy as np
from sklearn.metrics import precision_recall_curve, f1_score, classification_report, confusion_matrix
from pathlib import Path
import matplotlib.pyplot as plt
from tqdm import tqdm
import sys
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.models.baseline_cnn import BaselineCNN
from src.data.preprocess_pytorch import create_dataloader


def collect_predictions(model, loader, device):
    """Collect all predictions and probabilities from validation set."""
    model.eval()
    all_probs = []
    all_labels = []
    
    with torch.no_grad():
        for frames, labels in tqdm(loader, desc="Collecting predictions"):
            frames = frames.to(device)
            logits = model(frames)
            probs = torch.softmax(logits, dim=1)[:, 1]
            
            all_probs.append(probs.cpu().numpy())
            all_labels.append(labels.numpy())
    
    return np.concatenate(all_probs), np.concatenate(all_labels)


def find_optimal_threshold(probs, labels, target_recall=0.85, min_precision=0.85):
    """
    Find optimal threshold that maximizes AI recall while maintaining minimum precision.
    
    Args:
        probs: Predicted probabilities for AI class (positive class)
        labels: True labels
        target_recall: Target minimum recall for AI class
        min_precision: Minimum acceptable precision for AI class
    
    Returns:
        optimal_threshold, metrics_dict
    """
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    
    valid_thresholds = []
    for i, (p, r, t) in enumerate(zip(precisions, recalls, thresholds)):
        if r >= target_recall and p >= min_precision:
            valid_thresholds.append({
                'threshold': t,
                'precision': p,
                'recall': r,
                'f1': 2 * (p * r) / (p + r) if (p + r) > 0 else 0
            })
    
    if not valid_thresholds:
        print(f"\nWarning: No threshold meets both recall >= {target_recall} and precision >= {min_precision}")
        print("Relaxing precision constraint...")
        
        for i, (p, r, t) in enumerate(zip(precisions, recalls, thresholds)):
            if r >= target_recall:
                valid_thresholds.append({
                    'threshold': t,
                    'precision': p,
                    'recall': r,
                    'f1': 2 * (p * r) / (p + r) if (p + r) > 0 else 0
                })
    
    if not valid_thresholds:
        raise ValueError(f"Cannot achieve target recall of {target_recall}")
    
    best = max(valid_thresholds, key=lambda x: x['f1'])
    
    return best['threshold'], best


def evaluate_threshold(probs, labels, threshold):
    """Evaluate model performance at given threshold."""
    preds = (probs >= threshold).astype(int)
    
    cm = confusion_matrix(labels, preds)
    tn, fp, fn, tp = cm.ravel()
    
    precision_real = tn / (tn + fn) if (tn + fn) > 0 else 0
    recall_real = tn / (tn + fp) if (tn + fp) > 0 else 0
    f1_real = 2 * (precision_real * recall_real) / (precision_real + recall_real) if (precision_real + recall_real) > 0 else 0
    
    precision_ai = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall_ai = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1_ai = 2 * (precision_ai * recall_ai) / (precision_ai + recall_ai) if (precision_ai + recall_ai) > 0 else 0
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    
    return {
        'accuracy': accuracy,
        'precision_real': precision_real,
        'recall_real': recall_real,
        'f1_real': f1_real,
        'precision_ai': precision_ai,
        'recall_ai': recall_ai,
        'f1_ai': f1_ai,
        'confusion_matrix': cm
    }


def plot_threshold_analysis(probs, labels, optimal_threshold, save_path='threshold_analysis.png'):
    """Plot precision-recall curve and threshold analysis."""
    precisions, recalls, thresholds = precision_recall_curve(labels, probs)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(recalls, precisions, linewidth=2, label='PR Curve')
    axes[0].axvline(x=0.85, color='r', linestyle='--', alpha=0.7, label='Target Recall=0.85')
    axes[0].axhline(y=0.85, color='g', linestyle='--', alpha=0.7, label='Min Precision=0.85')
    axes[0].scatter([evaluate_threshold(probs, labels, optimal_threshold)['recall_ai']], 
                    [evaluate_threshold(probs, labels, optimal_threshold)['precision_ai']], 
                    color='red', s=100, zorder=5, label=f'Optimal (t={optimal_threshold:.3f})')
    axes[0].set_xlabel('Recall (AI class)')
    axes[0].set_ylabel('Precision (AI class)')
    axes[0].set_title('Precision-Recall Curve')
    axes[0].legend()
    axes[0].grid(alpha=0.3)
    
    f1_scores = []
    test_thresholds = np.linspace(0.1, 0.9, 100)
    for t in test_thresholds:
        metrics = evaluate_threshold(probs, labels, t)
        f1_scores.append(metrics['f1_ai'])
    
    axes[1].plot(test_thresholds, f1_scores, linewidth=2, label='F1 Score (AI)')
    axes[1].axvline(x=optimal_threshold, color='r', linestyle='--', alpha=0.7, label=f'Optimal={optimal_threshold:.3f}')
    axes[1].axvline(x=0.5, color='gray', linestyle=':', alpha=0.5, label='Default=0.5')
    axes[1].set_xlabel('Classification Threshold')
    axes[1].set_ylabel('F1 Score')
    axes[1].set_title('F1 Score vs Threshold')
    axes[1].legend()
    axes[1].grid(alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"Analysis plot saved to: {save_path}")


def main():
    print("="*60)
    print("THRESHOLD OPTIMIZATION FOR AI DETECTION")
    print("="*60)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"\nDevice: {device}")
    
    model = BaselineCNN(
        num_classes=2,
        pretrained=True,
        temporal_pool="attention",
        dropout=0.5
    ).to(device)
    
    checkpoint = torch.load('models/baseline_cnn_20251018_231512_final.pt', map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'], strict=False)
    print(f"Loaded model from epoch {checkpoint['epoch']} (val_acc={checkpoint['val_acc']:.4f})")
    
    val_loader = create_dataloader(
        metadata_csv='data/splits/val_crossgen.csv',
        cache_root='data/processed/genbuster_cached',
        batch_size=32,
        num_workers=4,
        clip_mode=True,
        shuffle=False,
        augment=False
    )
    
    print(f"\nCollecting predictions on validation set ({len(val_loader.dataset)} samples)...")
    probs, labels = collect_predictions(model, val_loader, device)
    
    print("\n" + "="*60)
    print("BASELINE PERFORMANCE (threshold=0.5)")
    print("="*60)
    baseline_metrics = evaluate_threshold(probs, labels, 0.5)
    print(f"\nAccuracy: {baseline_metrics['accuracy']:.4f}")
    print(f"\nReal class:")
    print(f"  Precision: {baseline_metrics['precision_real']:.4f}")
    print(f"  Recall: {baseline_metrics['recall_real']:.4f}")
    print(f"  F1: {baseline_metrics['f1_real']:.4f}")
    print(f"\nAI class:")
    print(f"  Precision: {baseline_metrics['precision_ai']:.4f}")
    print(f"  Recall: {baseline_metrics['recall_ai']:.4f}")
    print(f"  F1: {baseline_metrics['f1_ai']:.4f}")
    
    print("\n" + "="*60)
    print("FINDING OPTIMAL THRESHOLD")
    print("="*60)
    print("Target: AI Recall >= 0.85, Min Precision >= 0.85")
    
    optimal_threshold, best_metrics = find_optimal_threshold(
        probs, labels, 
        target_recall=0.85, 
        min_precision=0.85
    )
    
    print(f"\nOptimal threshold: {optimal_threshold:.4f}")
    print(f"  Precision: {best_metrics['precision']:.4f}")
    print(f"  Recall: {best_metrics['recall']:.4f}")
    print(f"  F1: {best_metrics['f1']:.4f}")
    
    print("\n" + "="*60)
    print(f"FULL EVALUATION AT THRESHOLD={optimal_threshold:.4f}")
    print("="*60)
    optimal_metrics = evaluate_threshold(probs, labels, optimal_threshold)
    print(f"\nAccuracy: {optimal_metrics['accuracy']:.4f}")
    print(f"\nReal class:")
    print(f"  Precision: {optimal_metrics['precision_real']:.4f}")
    print(f"  Recall: {optimal_metrics['recall_real']:.4f}")
    print(f"  F1: {optimal_metrics['f1_real']:.4f}")
    print(f"\nAI class:")
    print(f"  Precision: {optimal_metrics['precision_ai']:.4f}")
    print(f"  Recall: {optimal_metrics['recall_ai']:.4f}")
    print(f"  F1: {optimal_metrics['f1_ai']:.4f}")
    
    print(f"\nConfusion Matrix:")
    print(optimal_metrics['confusion_matrix'])
    print("       Pred: Real  AI")
    print(f"Real:   {optimal_metrics['confusion_matrix'][0, 0]:5d}  {optimal_metrics['confusion_matrix'][0, 1]:5d}")
    print(f"AI:     {optimal_metrics['confusion_matrix'][1, 0]:5d}  {optimal_metrics['confusion_matrix'][1, 1]:5d}")
    
    print("\n" + "="*60)
    print("PERFORMANCE COMPARISON")
    print("="*60)
    print(f"                    Baseline (0.5)  Optimized ({optimal_threshold:.3f})  Change")
    print(f"Accuracy:           {baseline_metrics['accuracy']:.4f}         {optimal_metrics['accuracy']:.4f}       {optimal_metrics['accuracy'] - baseline_metrics['accuracy']:+.4f}")
    print(f"AI Precision:       {baseline_metrics['precision_ai']:.4f}         {optimal_metrics['precision_ai']:.4f}       {optimal_metrics['precision_ai'] - baseline_metrics['precision_ai']:+.4f}")
    print(f"AI Recall:          {baseline_metrics['recall_ai']:.4f}         {optimal_metrics['recall_ai']:.4f}       {optimal_metrics['recall_ai'] - baseline_metrics['recall_ai']:+.4f}")
    print(f"AI F1:              {baseline_metrics['f1_ai']:.4f}         {optimal_metrics['f1_ai']:.4f}       {optimal_metrics['f1_ai'] - baseline_metrics['f1_ai']:+.4f}")
    
    plot_threshold_analysis(probs, labels, optimal_threshold, 'plots/threshold_analysis.png')
    
    np.savez('models/threshold_config.npz', 
             threshold=optimal_threshold,
             metrics=optimal_metrics)
    print(f"\nThreshold configuration saved to: models/threshold_config.npz")
    
    print("\n" + "="*60)
    print("RECOMMENDATION")
    print("="*60)
    print(f"\nDeploy with threshold = {optimal_threshold:.4f}")
    print("This achieves:")
    print(f"  - AI Recall: {optimal_metrics['recall_ai']:.1%} (meets target)")
    print(f"  - AI Precision: {optimal_metrics['precision_ai']:.1%} (maintains quality)")
    print(f"  - Overall Accuracy: {optimal_metrics['accuracy']:.1%}")
    
    if optimal_metrics['accuracy'] < baseline_metrics['accuracy']:
        diff = baseline_metrics['accuracy'] - optimal_metrics['accuracy']
        print(f"\nNote: Slight accuracy drop of {diff:.1%} is acceptable trade-off")
        print("      for significantly improved AI detection rate.")


if __name__ == "__main__":
    main()