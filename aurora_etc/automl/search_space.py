"""
Search space definition for AutoML reconfiguration
"""

from dataclasses import dataclass
from typing import List, Dict, Any
import optuna


@dataclass
class ArchitectureConfig:
    """Architecture configuration for a candidate model."""
    lora_rank: int = 8
    hidden_dim: int = 512
    num_heads: int = 8
    head_width: int = 128
    head_depth: int = 2
    use_cosine_classifier: bool = False


class SearchSpace:
    """
    Defines the search space for AutoML reconfiguration.
    """
    
    def __init__(
        self,
        lora_ranks: List[int] = [4, 8, 16],
        hidden_dims: List[int] = [256, 512, 768],
        num_heads_list: List[int] = [4, 8, 12],
        head_widths: List[int] = [64, 128, 256],
        head_depths: List[int] = [1, 2, 3],
    ):
        """
        Args:
            lora_ranks: Candidate LoRA ranks
            hidden_dims: Candidate hidden dimensions
            num_heads_list: Candidate number of attention heads
            head_widths: Candidate classification head widths
            head_depths: Candidate classification head depths
        """
        self.lora_ranks = lora_ranks
        self.hidden_dims = hidden_dims
        self.num_heads_list = num_heads_list
        self.head_widths = head_widths
        self.head_depths = head_depths
    
    def suggest_config(self, trial: optuna.Trial) -> ArchitectureConfig:
        """
        Suggest a configuration from the search space.
        
        Args:
            trial: Optuna trial
            
        Returns:
            Architecture configuration
        """
        return ArchitectureConfig(
            lora_rank=trial.suggest_categorical("lora_rank", self.lora_ranks),
            hidden_dim=trial.suggest_categorical("hidden_dim", self.hidden_dims),
            num_heads=trial.suggest_categorical("num_heads", self.num_heads_list),
            head_width=trial.suggest_categorical("head_width", self.head_widths),
            head_depth=trial.suggest_categorical("head_depth", self.head_depths),
            use_cosine_classifier=trial.suggest_categorical("use_cosine", [True, False]),
        )
    
    def get_default_config(self) -> ArchitectureConfig:
        """Get default architecture configuration."""
        return ArchitectureConfig()

