"""
Tests for drift detection
"""

import torch
import numpy as np
import pytest

from aurora_etc.drift import DriftDetector, compute_mmd, compute_ece


def test_mmd():
    """Test MMD computation."""
    X = np.random.randn(100, 64)
    Y = np.random.randn(100, 64)
    
    mmd_score = compute_mmd(X, Y)
    
    assert mmd_score >= 0
    assert isinstance(mmd_score, float)


def test_ece():
    """Test ECE computation."""
    batch_size = 100
    num_classes = 10
    
    logits = torch.randn(batch_size, num_classes)
    labels = torch.randint(0, num_classes, (batch_size,))
    
    ece = compute_ece(logits, labels, n_bins=15)
    
    assert 0 <= ece <= 1
    assert isinstance(ece, float)


def test_drift_detector():
    """Test drift detector."""
    detector = DriftDetector(tau1=0.5, tau2=0.7)
    
    # Set reference embeddings
    ref_embeddings = torch.randn(100, 512)
    detector.set_reference(ref_embeddings)
    
    # Current embeddings (same distribution - should be stable)
    curr_embeddings = torch.randn(50, 512)
    logits = torch.randn(50, 10)
    labels = torch.randint(0, 10, (50,))
    
    status, score, components = detector.detect_drift(
        curr_embeddings,
        logits,
        labels,
    )
    
    assert status.value in ["stable", "moderate", "severe"]
    assert score >= 0
    assert "mmd" in components

