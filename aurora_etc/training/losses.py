"""
Loss functions for pretraining and fine-tuning
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional


class ContrastiveLoss(nn.Module):
    """
    InfoNCE contrastive loss for self-supervised pretraining.
    """
    
    def __init__(self, temperature: float = 0.07):
        """
        Args:
            temperature: Temperature scaling parameter
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute InfoNCE loss.
        
        Args:
            z1: Embeddings of first augmented view (batch_size, d_model)
            z2: Embeddings of second augmented view (batch_size, d_model)
            
        Returns:
            Contrastive loss value
        """
        batch_size = z1.shape[0]
        
        # Normalize embeddings
        z1 = F.normalize(z1, p=2, dim=1)
        z2 = F.normalize(z2, p=2, dim=1)
        
        # Compute similarity matrix
        similarity_matrix = torch.matmul(z1, z2.T) / self.temperature
        
        # Labels: positive pairs are on the diagonal
        labels = torch.arange(batch_size, device=z1.device)
        
        # Symmetric loss
        loss_12 = F.cross_entropy(similarity_matrix, labels)
        loss_21 = F.cross_entropy(similarity_matrix.T, labels)
        
        return (loss_12 + loss_21) / 2


class MaskedModelingLoss(nn.Module):
    """
    Masked modeling loss for self-supervised pretraining.
    """
    
    def __init__(self):
        super().__init__()
    
    def forward(
        self,
        predictions: torch.Tensor,
        targets: torch.Tensor,
        mask: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute masked reconstruction loss.
        
        Args:
            predictions: Reconstructed tokens (batch_size, seq_len, d_model)
            targets: Original tokens (batch_size, seq_len, input_dim)
            mask: Mask indicating positions to reconstruct (batch_size, seq_len)
            
        Returns:
            Masked modeling loss
        """
        # Project predictions back to input space (if needed)
        # For simplicity, assuming predictions are already in input space
        # In practice, you'd add a projection layer
        
        # Compute MSE loss only on masked positions
        mse = F.mse_loss(predictions, targets, reduction='none')
        
        # Average over masked positions
        masked_mse = (mse * mask.unsqueeze(-1)).sum() / (mask.sum() + 1e-8)
        
        return masked_mse


class PretrainingLoss(nn.Module):
    """
    Combined pretraining loss (contrastive + masked modeling).
    """
    
    def __init__(
        self,
        alpha: float = 0.5,
        beta: float = 0.5,
        temperature: float = 0.07,
    ):
        """
        Args:
            alpha: Weight for contrastive loss
            beta: Weight for masked modeling loss
            temperature: Temperature for contrastive loss
        """
        super().__init__()
        self.alpha = alpha
        self.beta = beta
        self.contrastive_loss = ContrastiveLoss(temperature)
        self.masked_loss = MaskedModelingLoss()
    
    def forward(
        self,
        z1: torch.Tensor,
        z2: torch.Tensor,
        predictions: Optional[torch.Tensor] = None,
        targets: Optional[torch.Tensor] = None,
        mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Compute combined pretraining loss.
        
        Args:
            z1: Embeddings of first augmented view
            z2: Embeddings of second augmented view
            predictions: Optional reconstruction predictions
            targets: Optional reconstruction targets
            mask: Optional mask for masked modeling
            
        Returns:
            Combined loss
        """
        loss = self.alpha * self.contrastive_loss(z1, z2)
        
        if predictions is not None and targets is not None and mask is not None:
            loss += self.beta * self.masked_loss(predictions, targets, mask)
        
        return loss


class DistillationLoss(nn.Module):
    """
    Knowledge distillation loss for AutoML reconfiguration.
    """
    
    def __init__(self, temperature: float = 4.0):
        """
        Args:
            temperature: Temperature for softmax smoothing
        """
        super().__init__()
        self.temperature = temperature
    
    def forward(
        self,
        student_logits: torch.Tensor,
        teacher_logits: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute distillation loss.
        
        Args:
            student_logits: Student model logits (batch_size, num_classes)
            teacher_logits: Teacher model logits (batch_size, num_classes)
            
        Returns:
            Distillation loss
        """
        # Soft targets with temperature
        teacher_probs = F.softmax(teacher_logits / self.temperature, dim=1)
        student_log_probs = F.log_softmax(student_logits / self.temperature, dim=1)
        
        # KL divergence
        loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Scale by temperature squared
        loss = loss * (self.temperature ** 2)
        
        return loss

