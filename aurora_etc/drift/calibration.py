"""
Expected Calibration Error (ECE) computation
"""

import torch
import numpy as np
from typing import Tuple


def compute_ece(
    logits: torch.Tensor,
    labels: torch.Tensor,
    n_bins: int = 15,
) -> float:
    """
    Compute Expected Calibration Error (ECE).
    
    Args:
        logits: Model logits of shape (batch_size, num_classes)
        labels: Ground truth labels of shape (batch_size,)
        n_bins: Number of confidence bins
        
    Returns:
        ECE value
    """
    # Get predictions and confidences
    probs = torch.softmax(logits, dim=1)
    confidences, predictions = torch.max(probs, dim=1)
    accuracies = (predictions == labels).float()
    
    # Convert to numpy
    confidences = confidences.cpu().numpy()
    accuracies = accuracies.cpu().numpy()
    
    # Bin the confidences
    bin_boundaries = np.linspace(0, 1, n_bins + 1)
    bin_lowers = bin_boundaries[:-1]
    bin_uppers = bin_boundaries[1:]
    
    ece = 0.0
    for bin_lower, bin_upper in zip(bin_lowers, bin_uppers):
        # Find samples in this bin
        in_bin = (confidences > bin_lower) & (confidences <= bin_upper)
        prop_in_bin = in_bin.mean()
        
        if prop_in_bin > 0:
            # Average accuracy and confidence in this bin
            accuracy_in_bin = accuracies[in_bin].mean()
            avg_confidence_in_bin = confidences[in_bin].mean()
            ece += np.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin
    
    return float(ece)

