"""
Unified drift detection module
"""

import torch
import numpy as np
from typing import Optional, Tuple, Dict
from enum import Enum

from aurora_etc.drift.mmd import compute_mmd
from aurora_etc.drift.calibration import compute_ece


class DriftStatus(Enum):
    """Drift status levels"""
    STABLE = "stable"
    MODERATE = "moderate"
    SEVERE = "severe"


class DriftDetector:
    """
    Unified drift detector combining MMD, ECE, uncertainty, and protocol telemetry.
    """
    
    def __init__(
        self,
        tau1: float = 0.5,  # Moderate drift threshold
        tau2: float = 0.7,  # Severe drift threshold
        w1: float = 0.3,    # MMD weight
        w2: float = 0.2,    # Uncertainty weight
        w3: float = 0.2,    # ECE weight
        w4: float = 0.3,    # Protocol telemetry weight
        n_bins: int = 15,   # Number of bins for ECE
        mmd_sigma: Optional[float] = None,  # MMD kernel bandwidth (None = median heuristic)
    ):
        """
        Args:
            tau1: Threshold for moderate drift
            tau2: Threshold for severe drift
            w1: Weight for MMD term
            w2: Weight for uncertainty term
            w3: Weight for ECE term
            w4: Weight for protocol telemetry term
            n_bins: Number of bins for ECE computation
            mmd_sigma: MMD kernel bandwidth (None for median heuristic)
        """
        self.tau1 = tau1
        self.tau2 = tau2
        self.w1 = w1
        self.w2 = w2
        self.w3 = w3
        self.w4 = w4
        self.n_bins = n_bins
        self.mmd_sigma = mmd_sigma
        
        # Reference embeddings for MMD (will be set from validation set)
        self.ref_embeddings: Optional[torch.Tensor] = None
    
    def set_reference(self, ref_embeddings: torch.Tensor):
        """
        Set reference embeddings from validation set.
        
        Args:
            ref_embeddings: Reference embeddings of shape (N, d_model)
        """
        self.ref_embeddings = ref_embeddings.detach().cpu()
    
    def compute_uncertainty(
        self,
        logits: torch.Tensor,
    ) -> float:
        """
        Compute average predictive uncertainty (entropy).
        
        Args:
            logits: Model logits of shape (batch_size, num_classes)
            
        Returns:
            Average uncertainty score
        """
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy.mean().item()
    
    def compute_protocol_telemetry(
        self,
        protocol_counts: Dict[str, int],
        total_flows: int,
    ) -> float:
        """
        Compute normalized protocol telemetry score.
        
        Args:
            protocol_counts: Dictionary with counts for TLS, ECH, QUIC
            total_flows: Total number of flows
            
        Returns:
            Normalized telemetry score [0, 1]
        """
        if total_flows == 0:
            return 0.0
        
        # Normalize protocol ratios
        tls_ratio = protocol_counts.get("TLS", 0) / total_flows
        ech_ratio = protocol_counts.get("ECH", 0) / total_flows
        quic_ratio = protocol_counts.get("QUIC", 0) / total_flows
        
        # Combine into single score (simple average, can be weighted)
        telemetry_score = (tls_ratio + ech_ratio + quic_ratio) / 3.0
        
        return telemetry_score
    
    def detect_drift(
        self,
        embeddings: torch.Tensor,
        logits: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        protocol_counts: Optional[Dict[str, int]] = None,
    ) -> Tuple[DriftStatus, float, Dict[str, float]]:
        """
        Detect drift in current traffic window.
        
        Args:
            embeddings: Current embeddings of shape (batch_size, d_model)
            logits: Model logits of shape (batch_size, num_classes)
            labels: Optional ground truth labels for ECE computation
            protocol_counts: Optional protocol statistics
            
        Returns:
            Tuple of (drift_status, unified_score, component_scores)
        """
        component_scores = {}
        
        # 1. Feature drift (MMD)
        if self.ref_embeddings is not None:
            mmd_score = compute_mmd(
                embeddings.detach().cpu().numpy(),
                self.ref_embeddings.numpy(),
                sigma=self.mmd_sigma,
            )
        else:
            mmd_score = 0.0
        component_scores["mmd"] = mmd_score
        
        # 2. Uncertainty
        uncertainty = self.compute_uncertainty(logits)
        # Normalize uncertainty (max entropy is log(num_classes))
        num_classes = logits.shape[1]
        max_entropy = np.log(num_classes)
        normalized_uncertainty = uncertainty / (max_entropy + 1e-8)
        component_scores["uncertainty"] = normalized_uncertainty
        
        # 3. Calibration error (ECE)
        if labels is not None:
            ece_score = compute_ece(logits, labels, n_bins=self.n_bins)
        else:
            ece_score = 0.0
        component_scores["ece"] = ece_score
        
        # 4. Protocol telemetry
        if protocol_counts is not None:
            total_flows = embeddings.shape[0]
            telemetry_score = self.compute_protocol_telemetry(protocol_counts, total_flows)
        else:
            telemetry_score = 0.0
        component_scores["telemetry"] = telemetry_score
        
        # Unified drift score
        unified_score = (
            self.w1 * mmd_score +
            self.w2 * normalized_uncertainty +
            self.w3 * ece_score +
            self.w4 * telemetry_score
        )
        
        # Determine drift status
        if unified_score >= self.tau2:
            status = DriftStatus.SEVERE
        elif unified_score >= self.tau1:
            status = DriftStatus.MODERATE
        else:
            status = DriftStatus.STABLE
        
        return status, unified_score, component_scores

