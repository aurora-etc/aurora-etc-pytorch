"""
Training modules for pretraining and online updates
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, List
from tqdm import tqdm
import os

from aurora_etc.models import SessionEncoder, ClassificationHead
from aurora_etc.training.losses import PretrainingLoss, DistillationLoss
from aurora_etc.training.replay_buffer import ReplayBuffer


class Pretrainer:
    """
    Trainer for self-supervised pretraining.
    """
    
    def __init__(
        self,
        model: SessionEncoder,
        device: torch.device,
        loss_fn: Optional[PretrainingLoss] = None,
    ):
        """
        Args:
            model: Session encoder model
            device: Training device
            loss_fn: Pretraining loss function
        """
        self.model = model.to(device)
        self.device = device
        self.loss_fn = loss_fn or PretrainingLoss()
    
    def train_epoch(
        self,
        dataloader: DataLoader,
        optimizer: torch.optim.Optimizer,
    ) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in tqdm(dataloader, desc="Pretraining"):
            features = batch["features"].to(self.device)
            mask = batch["mask"].to(self.device)
            
            # Create two augmented views
            # In practice, augmentation would be applied in dataset
            # For now, assume batch contains both views
            features1 = features
            features2 = features  # TODO: Apply different augmentation
            
            # Forward pass
            z1, seq1 = self.model(features1, mask)
            z2, seq2 = self.model(features2, mask)
            
            # Compute loss
            loss = self.loss_fn(z1, z2)
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        return total_loss / num_batches if num_batches > 0 else 0.0
    
    def save_checkpoint(self, path: str, optimizer: Optional[torch.optim.Optimizer] = None):
        """Save model checkpoint."""
        checkpoint = {
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        }
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save(checkpoint, path)


class OnlineUpdater:
    """
    Trainer for lightweight online updates with LoRA.
    """
    
    def __init__(
        self,
        encoder: SessionEncoder,
        classifier: ClassificationHead,
        device: torch.device,
        replay_buffer: Optional[ReplayBuffer] = None,
    ):
        """
        Args:
            encoder: Session encoder
            classifier: Classification head
            device: Training device
            replay_buffer: Optional replay buffer
        """
        self.encoder = encoder.to(device)
        self.classifier = classifier.to(device)
        self.device = device
        self.replay_buffer = replay_buffer
        
        # Freeze encoder backbone, keep only LoRA trainable
        self.encoder.freeze_backbone()
    
    def compute_uncertainty(self, logits: torch.Tensor) -> torch.Tensor:
        """Compute predictive uncertainty."""
        probs = torch.softmax(logits, dim=1)
        entropy = -torch.sum(probs * torch.log(probs + 1e-8), dim=1)
        return entropy
    
    def select_uncertain_samples(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        threshold: float = 0.8,
    ) -> List[int]:
        """
        Select uncertain samples for active labeling.
        
        Args:
            features: Flow features
            mask: Attention mask
            threshold: Uncertainty threshold
            
        Returns:
            List of indices of uncertain samples
        """
        self.encoder.eval()
        self.classifier.eval()
        
        with torch.no_grad():
            embeddings, _ = self.encoder(features, mask)
            logits = self.classifier(embeddings)
            uncertainties = self.compute_uncertainty(logits)
        
        # Select samples above threshold
        uncertain_indices = (uncertainties > threshold).nonzero(as_tuple=True)[0].tolist()
        
        return uncertain_indices
    
    def train_step(
        self,
        new_features: torch.Tensor,
        new_mask: torch.Tensor,
        new_labels: torch.Tensor,
        optimizer: torch.optim.Optimizer,
        alpha: float = 0.5,  # Weight for replay loss
        beta: float = 0.1,   # Weight for regularization
    ) -> Dict[str, float]:
        """
        Perform one online update step.
        
        Args:
            new_features: New flow features
            new_mask: Attention mask
            new_labels: Labels for new flows
            optimizer: Optimizer
            alpha: Replay loss weight
            beta: Regularization weight
            
        Returns:
            Dictionary of losses
        """
        self.encoder.train()
        self.classifier.train()
        
        # Forward pass on new data
        embeddings, _ = self.encoder(new_features, new_mask)
        logits = self.classifier(embeddings)
        ce_loss = nn.CrossEntropyLoss()(logits, new_labels)
        
        total_loss = ce_loss
        losses = {"ce_loss": ce_loss.item()}
        
        # Add replay loss if buffer is available
        if self.replay_buffer is not None and len(self.replay_buffer) > 0:
            replay_batch = self.replay_buffer.sample(batch_size=len(new_labels))
            replay_features = replay_batch["features"].to(self.device)
            replay_mask = replay_batch["mask"].to(self.device)
            replay_labels = replay_batch["label"].to(self.device)
            
            replay_embeddings, _ = self.encoder(replay_features, replay_mask)
            replay_logits = self.classifier(replay_embeddings)
            replay_loss = nn.CrossEntropyLoss()(replay_logits, replay_labels)
            
            total_loss += alpha * replay_loss
            losses["replay_loss"] = replay_loss.item()
        
        # Add regularization (optional)
        if beta > 0:
            lora_params = self.encoder.get_lora_params()
            if len(lora_params) > 0:
                l2_reg = sum(torch.norm(p) for p in lora_params)
                total_loss += beta * l2_reg
                losses["reg_loss"] = l2_reg.item()
        
        # Backward pass
        optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(
            self.encoder.get_lora_params() + list(self.classifier.parameters()),
            max_norm=1.0,
        )
        optimizer.step()
        
        losses["total_loss"] = total_loss.item()
        
        return losses

