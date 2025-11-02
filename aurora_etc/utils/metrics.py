"""
Evaluation metrics for encrypted traffic classification
"""

import torch
import numpy as np
from sklearn.metrics import f1_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, Optional


def compute_metrics(
    predictions: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    average: str = "macro",
) -> Dict[str, float]:
    """
    Compute classification metrics.
    
    Args:
        predictions: Predicted labels
        labels: Ground truth labels
        num_classes: Number of classes
        average: Averaging strategy ("macro", "micro", "weighted")
        
    Returns:
        Dictionary of metrics
    """
    predictions = predictions.cpu().numpy()
    labels = labels.cpu().numpy()
    
    # Macro-F1
    macro_f1 = f1_score(labels, predictions, average="macro", zero_division=0)
    
    # Micro-F1
    micro_f1 = f1_score(labels, predictions, average="micro", zero_division=0)
    
    # Per-class metrics
    precision, recall, f1, support = precision_recall_fscore_support(
        labels, predictions, average=None, zero_division=0
    )
    
    # Accuracy
    accuracy = (predictions == labels).mean()
    
    return {
        "accuracy": float(accuracy),
        "macro_f1": float(macro_f1),
        "micro_f1": float(micro_f1),
        "precision": precision.tolist(),
        "recall": recall.tolist(),
        "f1_per_class": f1.tolist(),
        "support": support.tolist(),
    }


def compute_bwt(
    performance_matrix: np.ndarray,
) -> float:
    """
    Compute Backward Transfer (BWT).
    
    Measures the influence of learning new tasks on previous tasks.
    
    Args:
        performance_matrix: Matrix of shape (T, T) where R[i,j] is accuracy
                            on task j after training up to task i
        
    Returns:
        BWT value
    """
    T = performance_matrix.shape[0]
    if T < 2:
        return 0.0
    
    bwt_sum = 0.0
    for i in range(T - 1):
        final_perf = performance_matrix[T - 1, i]
        immediate_perf = performance_matrix[i, i]
        bwt_sum += (final_perf - immediate_perf)
    
    return bwt_sum / (T - 1)


def compute_fwt(
    performance_matrix: np.ndarray,
    baseline_performance: Optional[np.ndarray] = None,
) -> float:
    """
    Compute Forward Transfer (FWT).
    
    Measures the influence of previous tasks on future tasks.
    
    Args:
        performance_matrix: Matrix of shape (T, T) where R[i,j] is accuracy
                            on task j after training up to task i
        baseline_performance: Optional baseline performance for each task
                             (accuracy when trained from scratch)
        
    Returns:
        FWT value
    """
    T = performance_matrix.shape[0]
    if T < 2:
        return 0.0
    
    if baseline_performance is None:
        baseline_performance = np.diag(performance_matrix)
    
    fwt_sum = 0.0
    for i in range(1, T):
        transfer_perf = performance_matrix[i - 1, i]
        baseline_perf = baseline_performance[i]
        fwt_sum += (transfer_perf - baseline_perf)
    
    return fwt_sum / (T - 1)


def compute_ood_auroc(
    in_dist_logits: torch.Tensor,
    ood_logits: torch.Tensor,
) -> float:
    """
    Compute OOD detection AUROC.
    
    Args:
        in_dist_logits: Logits for in-distribution samples
        ood_logits: Logits for out-of-distribution samples
        
    Returns:
        AUROC value
    """
    # Use maximum softmax probability as OOD score
    in_dist_probs = torch.softmax(in_dist_logits, dim=1).max(dim=1)[0]
    ood_probs = torch.softmax(ood_logits, dim=1).max(dim=1)[0]
    
    # Create labels: 1 for in-distribution, 0 for OOD
    y_true = np.concatenate([np.ones(len(in_dist_probs)), np.zeros(len(ood_probs))])
    y_scores = np.concatenate([in_dist_probs.cpu().numpy(), ood_probs.cpu().numpy()])
    
    # Compute AUROC (higher score = more in-distribution)
    auroc = roc_auc_score(y_true, y_scores)
    
    return float(auroc)

