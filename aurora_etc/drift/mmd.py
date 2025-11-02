"""
Maximum Mean Discrepancy (MMD) computation for drift detection
"""

import numpy as np
from typing import Optional
from scipy.spatial.distance import cdist


def gaussian_kernel(X: np.ndarray, Y: np.ndarray, sigma: float) -> np.ndarray:
    """
    Compute Gaussian RBF kernel matrix.
    
    Args:
        X: First set of points (N, d)
        Y: Second set of points (M, d)
        sigma: Kernel bandwidth
        
    Returns:
        Kernel matrix of shape (N, M)
    """
    # Compute pairwise distances
    dists = cdist(X, Y, metric='euclidean')
    # Apply Gaussian kernel
    K = np.exp(-dists ** 2 / (2 * sigma ** 2))
    return K


def median_heuristic(X: np.ndarray, Y: np.ndarray) -> float:
    """
    Compute median heuristic for kernel bandwidth.
    
    Args:
        X: First set of points (N, d)
        Y: Second set of points (M, d)
        
    Returns:
        Median distance
    """
    # Compute pairwise distances within and across sets
    dists_xx = cdist(X, X, metric='euclidean')
    dists_yy = cdist(Y, Y, metric='euclidean')
    dists_xy = cdist(X, Y, metric='euclidean')
    
    # Get median of all pairwise distances
    all_dists = np.concatenate([
        dists_xx[np.triu_indices_from(dists_xx, k=1)],
        dists_yy[np.triu_indices_from(dists_yy, k=1)],
        dists_xy.flatten(),
    ])
    
    return np.median(all_dists)


def compute_mmd(
    X: np.ndarray,
    Y: np.ndarray,
    sigma: Optional[float] = None,
) -> float:
    """
    Compute Maximum Mean Discrepancy (MMD) between two distributions.
    
    Uses Gaussian RBF kernel: MMD^2 = E[k(x,x')] + E[k(y,y')] - 2*E[k(x,y)]
    
    Args:
        X: Samples from first distribution (N, d)
        Y: Samples from second distribution (M, d)
        sigma: Kernel bandwidth (None for median heuristic)
        
    Returns:
        MMD^2 value
    """
    # Use median heuristic if sigma not provided
    if sigma is None:
        sigma = median_heuristic(X, Y)
        if sigma == 0:
            sigma = 1.0
    
    # Compute kernel matrices
    K_xx = gaussian_kernel(X, X, sigma)
    K_yy = gaussian_kernel(Y, Y, sigma)
    K_xy = gaussian_kernel(X, Y, sigma)
    
    # Compute MMD^2
    n = X.shape[0]
    m = Y.shape[0]
    
    # Unbiased estimator (exclude diagonal terms)
    term1 = (K_xx.sum() - np.trace(K_xx)) / (n * (n - 1)) if n > 1 else 0.0
    term2 = (K_yy.sum() - np.trace(K_yy)) / (m * (m - 1)) if m > 1 else 0.0
    term3 = K_xy.sum() / (n * m)
    
    mmd2 = term1 + term2 - 2 * term3
    
    return max(0.0, mmd2)  # Ensure non-negative

