"""
Replay buffer for continual learning
"""

import torch
import numpy as np
from typing import List, Dict, Optional
from collections import defaultdict
import random


class ReplayBuffer:
    """
    Replay buffer for storing representative flows to prevent catastrophic forgetting.
    """
    
    def __init__(
        self,
        capacity: int = 25000,
        selection_strategy: str = "uncertainty",
    ):
        """
        Args:
            capacity: Maximum number of samples in buffer
            selection_strategy: Strategy for selecting samples ("uncertainty", "random", "class_balanced")
        """
        self.capacity = capacity
        self.selection_strategy = selection_strategy
        
        # Storage
        self.buffer: List[Dict] = []
        self.class_counts = defaultdict(int)
        
    def add(
        self,
        features: torch.Tensor,
        mask: torch.Tensor,
        label: torch.Tensor,
        uncertainty: Optional[float] = None,
        embedding: Optional[torch.Tensor] = None,
    ):
        """
        Add samples to buffer.
        
        Args:
            features: Flow features
            mask: Attention mask
            label: Class label
            uncertainty: Optional uncertainty score
            embedding: Optional flow embedding
        """
        sample = {
            "features": features.cpu(),
            "mask": mask.cpu(),
            "label": label.cpu(),
            "uncertainty": uncertainty,
            "embedding": embedding.cpu() if embedding is not None else None,
        }
        
        self.buffer.append(sample)
        self.class_counts[label.item()] += 1
        
        # Remove oldest samples if over capacity
        if len(self.buffer) > self.capacity:
            removed = self.buffer.pop(0)
            self.class_counts[removed["label"].item()] -= 1
    
    def sample(
        self,
        batch_size: int,
        strategy: Optional[str] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Sample a batch from the buffer.
        
        Args:
            batch_size: Number of samples to retrieve
            strategy: Optional override for selection strategy
            
        Returns:
            Dictionary with batched tensors
        """
        strategy = strategy or self.selection_strategy
        
        if len(self.buffer) == 0:
            raise ValueError("Buffer is empty")
        
        if strategy == "random":
            indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
        elif strategy == "uncertainty":
            # Sample based on uncertainty (higher uncertainty = more likely)
            uncertainties = [s.get("uncertainty", 0.0) for s in self.buffer]
            if all(u == 0.0 for u in uncertainties):
                # Fall back to random if no uncertainties
                indices = random.sample(range(len(self.buffer)), min(batch_size, len(self.buffer)))
            else:
                # Weighted sampling by uncertainty
                probs = np.array(uncertainties)
                probs = probs / (probs.sum() + 1e-8)
                indices = np.random.choice(len(self.buffer), size=min(batch_size, len(self.buffer)), p=probs, replace=False)
        elif strategy == "class_balanced":
            # Sample evenly from each class
            num_classes = len(self.class_counts)
            per_class = batch_size // num_classes if num_classes > 0 else batch_size
            
            indices = []
            for class_label in self.class_counts.keys():
                class_samples = [i for i, s in enumerate(self.buffer) if s["label"].item() == class_label]
                if len(class_samples) > 0:
                    selected = random.sample(class_samples, min(per_class, len(class_samples)))
                    indices.extend(selected)
            
            # Fill remaining with random if needed
            if len(indices) < batch_size:
                remaining = batch_size - len(indices)
                remaining_indices = random.sample(range(len(self.buffer)), min(remaining, len(self.buffer)))
                indices.extend(remaining_indices)
        else:
            raise ValueError(f"Unknown selection strategy: {strategy}")
        
        # Retrieve samples
        batch = {
            "features": torch.stack([self.buffer[i]["features"] for i in indices]),
            "mask": torch.stack([self.buffer[i]["mask"] for i in indices]),
            "label": torch.stack([self.buffer[i]["label"] for i in indices]),
        }
        
        return batch
    
    def refresh(
        self,
        new_samples: List[Dict],
        keep_ratio: float = 0.7,
    ):
        """
        Refresh buffer by replacing old samples with new ones.
        
        Args:
            new_samples: List of new samples to potentially add
            keep_ratio: Ratio of old samples to keep
        """
        # Remove oldest samples
        num_keep = int(len(self.buffer) * keep_ratio)
        self.buffer = self.buffer[-num_keep:]
        
        # Reset class counts
        self.class_counts = defaultdict(int)
        for sample in self.buffer:
            self.class_counts[sample["label"].item()] += 1
        
        # Add new samples
        for sample in new_samples:
            self.add(**sample)
    
    def __len__(self) -> int:
        return len(self.buffer)
    
    def get_class_distribution(self) -> Dict[int, int]:
        """Get class distribution in buffer."""
        return dict(self.class_counts)

