"""
AutoML-guided reconfiguration using Bayesian optimization
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Dict, Tuple, Callable
import optuna
from optuna.samplers import TPESampler

from aurora_etc.models import SessionEncoder, ClassificationHead
from aurora_etc.automl.search_space import SearchSpace, ArchitectureConfig
from aurora_etc.training.losses import DistillationLoss


class AutoMLReconfigurator:
    """
    AutoML-guided reconfiguration for severe drift scenarios.
    """
    
    def __init__(
        self,
        search_space: SearchSpace,
        device: torch.device,
        max_trials: int = 30,
        latency_constraint: float = 2.62,  # p99 latency in ms
        memory_constraint: float = 2.4,     # Memory in GB
        throughput_constraint: float = 242.4,  # Flows per second
    ):
        """
        Args:
            search_space: Search space definition
            device: Training device
            max_trials: Maximum number of architecture trials
            latency_constraint: Maximum allowed p99 latency (ms)
            memory_constraint: Maximum allowed memory (GB)
            throughput_constraint: Minimum required throughput (flows/s)
        """
        self.search_space = search_space
        self.device = device
        self.max_trials = max_trials
        self.latency_constraint = latency_constraint
        self.memory_constraint = memory_constraint
        self.throughput_constraint = throughput_constraint
    
    def evaluate_candidate(
        self,
        config: ArchitectureConfig,
        teacher_model: Tuple[SessionEncoder, ClassificationHead],
        train_data: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor],
        num_epochs: int = 5,
    ) -> Dict[str, float]:
        """
        Evaluate a candidate architecture.
        
        Args:
            config: Architecture configuration
            teacher_model: Teacher model (encoder, classifier)
            train_data: Training data
            val_data: Validation data
            num_epochs: Number of training epochs
            
        Returns:
            Dictionary with metrics (f1_score, latency, memory, throughput)
        """
        teacher_encoder, teacher_classifier = teacher_model
        
        # Create student model with new configuration
        student_encoder = self._create_encoder(config, teacher_encoder)
        student_classifier = self._create_classifier(config, teacher_classifier)
        
        student_encoder = student_encoder.to(self.device)
        student_classifier = student_classifier.to(self.device)
        
        # Warm-start from teacher (copy LoRA adapters)
        self._warm_start(student_encoder, teacher_encoder, config)
        
        # Fine-tune with distillation
        self._fine_tune(
            student_encoder,
            student_classifier,
            teacher_encoder,
            teacher_classifier,
            train_data,
            num_epochs,
        )
        
        # Evaluate metrics
        metrics = self._evaluate_metrics(
            student_encoder,
            student_classifier,
            val_data,
        )
        
        # Check constraints
        if not self._satisfies_constraints(metrics):
            metrics["f1_score"] = 0.0  # Penalize constraint violations
        
        return metrics
    
    def _create_encoder(
        self,
        config: ArchitectureConfig,
        base_encoder: SessionEncoder,
    ) -> SessionEncoder:
        """Create student encoder with new configuration."""
        # Create new encoder with updated configuration
        student_encoder = SessionEncoder(
            d_model=config.hidden_dim,
            nhead=config.num_heads,
            use_lora=True,
            lora_rank=config.lora_rank,
        )
        
        # Copy pretrained weights where possible
        student_encoder.load_state_dict(base_encoder.state_dict(), strict=False)
        
        return student_encoder
    
    def _create_classifier(
        self,
        config: ArchitectureConfig,
        base_classifier: ClassificationHead,
    ) -> ClassificationHead:
        """Create student classifier with new configuration."""
        student_classifier = ClassificationHead(
            input_dim=config.hidden_dim,
            num_classes=base_classifier.num_classes,
            hidden_dim=config.head_width if config.head_depth > 1 else None,
            use_cosine=config.use_cosine_classifier,
        )
        
        return student_classifier
    
    def _warm_start(
        self,
        student: SessionEncoder,
        teacher: SessionEncoder,
        config: ArchitectureConfig,
    ):
        """Warm-start student from teacher LoRA adapters."""
        # Copy matching LoRA parameters
        student_state = student.state_dict()
        teacher_state = teacher.state_dict()
        
        for key in student_state:
            if "lora" in key and key in teacher_state:
                # Try to copy if dimensions match
                if student_state[key].shape == teacher_state[key].shape:
                    student_state[key] = teacher_state[key].clone()
        
        student.load_state_dict(student_state)
    
    def _fine_tune(
        self,
        encoder: SessionEncoder,
        classifier: ClassificationHead,
        teacher_encoder: SessionEncoder,
        teacher_classifier: ClassificationHead,
        train_data: Dict[str, torch.Tensor],
        num_epochs: int,
    ):
        """Fine-tune student with knowledge distillation."""
        encoder.train()
        classifier.train()
        
        optimizer = torch.optim.Adam(
            list(encoder.parameters()) + list(classifier.parameters()),
            lr=1e-4,
        )
        
        distillation_loss = DistillationLoss()
        ce_loss = nn.CrossEntropyLoss()
        
        features = train_data["features"].to(self.device)
        mask = train_data["mask"].to(self.device)
        labels = train_data["label"].to(self.device)
        
        for epoch in range(num_epochs):
            # Student forward
            embeddings, _ = encoder(features, mask)
            student_logits = classifier(embeddings)
            
            # Teacher forward (frozen)
            with torch.no_grad():
                teacher_embeddings, _ = teacher_encoder(features, mask)
                teacher_logits = teacher_classifier(teacher_embeddings)
            
            # Combined loss
            sup_loss = ce_loss(student_logits, labels)
            dist_loss = distillation_loss(student_logits, teacher_logits)
            loss = 0.5 * sup_loss + 0.5 * dist_loss
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
    
    def _evaluate_metrics(
        self,
        encoder: SessionEncoder,
        classifier: ClassificationHead,
        val_data: Dict[str, torch.Tensor],
    ) -> Dict[str, float]:
        """Evaluate model metrics including latency and memory."""
        encoder.eval()
        classifier.eval()
        
        features = val_data["features"].to(self.device)
        mask = val_data["mask"].to(self.device)
        labels = val_data["label"].to(self.device)
        
        # Measure latency
        latencies = []
        with torch.no_grad():
            for i in range(min(100, len(features))):  # Sample 100 inferences
                import time
                start = time.time()
                embeddings, _ = encoder(features[i:i+1], mask[i:i+1])
                logits = classifier(embeddings)
                end = time.time()
                latencies.append((end - start) * 1000)  # Convert to ms
        
        p95_latency = np.percentile(latencies, 95)
        p99_latency = np.percentile(latencies, 99)
        
        # Measure memory
        memory_mb = torch.cuda.memory_allocated(self.device) / (1024 ** 2) if torch.cuda.is_available() else 0
        memory_gb = memory_mb / 1024
        
        # Compute accuracy/F1
        with torch.no_grad():
            embeddings, _ = encoder(features, mask)
            logits = classifier(embeddings)
            predictions = torch.argmax(logits, dim=1)
            accuracy = (predictions == labels).float().mean().item()
        
        # Estimate throughput (flows per second)
        avg_latency = np.mean(latencies)
        throughput = 1000.0 / avg_latency if avg_latency > 0 else 0
        
        return {
            "f1_score": accuracy,  # Simplified - use actual F1 in practice
            "accuracy": accuracy,
            "p95_latency": p95_latency,
            "p99_latency": p99_latency,
            "memory_gb": memory_gb,
            "throughput": throughput,
        }
    
    def _satisfies_constraints(self, metrics: Dict[str, float]) -> bool:
        """Check if candidate satisfies resource constraints."""
        return (
            metrics["p99_latency"] <= self.latency_constraint and
            metrics["memory_gb"] <= self.memory_constraint and
            metrics["throughput"] >= self.throughput_constraint
        )
    
    def search(
        self,
        teacher_model: Tuple[SessionEncoder, ClassificationHead],
        train_data: Dict[str, torch.Tensor],
        val_data: Dict[str, torch.Tensor],
    ) -> Tuple[ArchitectureConfig, Dict[str, float]]:
        """
        Search for best architecture configuration.
        
        Args:
            teacher_model: Teacher model
            train_data: Training data
            val_data: Validation data
            
        Returns:
            Tuple of (best_config, best_metrics)
        """
        def objective(trial: optuna.Trial) -> float:
            config = self.search_space.suggest_config(trial)
            metrics = self.evaluate_candidate(config, teacher_model, train_data, val_data)
            return metrics["f1_score"]
        
        study = optuna.create_study(
            direction="maximize",
            sampler=TPESampler(),
        )
        
        study.optimize(objective, n_trials=self.max_trials)
        
        best_config = ArchitectureConfig(**study.best_params)
        best_metrics = {"f1_score": study.best_value}
        
        return best_config, best_metrics

