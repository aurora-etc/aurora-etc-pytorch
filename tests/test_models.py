"""
Tests for model components
"""

import torch
import pytest

from aurora_etc.models import SessionEncoder, ClassificationHead, LoRALinear


def test_session_encoder():
    """Test session encoder forward pass."""
    encoder = SessionEncoder(
        input_dim=3,
        d_model=512,
        nhead=8,
        num_layers=2,  # Small for testing
        use_lora=False,
    )
    
    batch_size = 4
    seq_len = 128
    features = torch.randn(batch_size, seq_len, 3)
    mask = torch.ones(batch_size, seq_len)
    
    embeddings, sequence_output = encoder(features, mask)
    
    assert embeddings.shape == (batch_size, 512)
    assert sequence_output.shape == (batch_size, seq_len, 512)


def test_lora_linear():
    """Test LoRA linear layer."""
    lora_layer = LoRALinear(in_features=128, out_features=64, rank=8, alpha=16)
    
    batch_size = 4
    x = torch.randn(batch_size, 128)
    output = lora_layer(x)
    
    assert output.shape == (batch_size, 64)


def test_classification_head():
    """Test classification head."""
    head = ClassificationHead(
        input_dim=512,
        num_classes=10,
        hidden_dim=128,
    )
    
    batch_size = 4
    embeddings = torch.randn(batch_size, 512)
    logits = head(embeddings)
    
    assert logits.shape == (batch_size, 10)

