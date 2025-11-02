#!/usr/bin/env python
"""
Fine-tuning script for AURORA-ETC
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from aurora_etc.models import SessionEncoder, ClassificationHead
from aurora_etc.data import EncryptedTrafficDataset
from aurora_etc.utils.logging import setup_logger
from aurora_etc.utils.metrics import compute_metrics


def validate(encoder, classifier, val_loader, device):
    """Validate model."""
    encoder.eval()
    classifier.eval()
    
    all_predictions = []
    all_labels = []
    total_loss = 0.0
    num_batches = 0
    
    criterion = torch.nn.CrossEntropyLoss()
    
    with torch.no_grad():
        for batch in val_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            
            embeddings, _ = encoder(features, mask)
            logits = classifier(embeddings)
            
            loss = criterion(logits, labels)
            total_loss += loss.item()
            
            predictions = torch.argmax(logits, dim=1)
            all_predictions.append(predictions)
            all_labels.append(labels)
            
            num_batches += 1
    
    all_predictions = torch.cat(all_predictions)
    all_labels = torch.cat(all_labels)
    
    metrics = compute_metrics(all_predictions, all_labels, num_classes=classifier.num_classes)
    avg_loss = total_loss / num_batches if num_batches > 0 else 0.0
    
    return avg_loss, metrics


def main():
    parser = argparse.ArgumentParser(description="Fine-tune AURORA-ETC")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger(log_file=config["output"]["log_dir"] + "/finetune.log")
    logger.info("Starting fine-tuning...")
    
    # Device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data
    train_dataset = EncryptedTrafficDataset(
        config["data"]["train_path"],
        split="train",
    )
    val_dataset = EncryptedTrafficDataset(
        config["data"]["val_path"],
        split="val",
    )
    test_dataset = EncryptedTrafficDataset(
        config["data"]["test_path"],
        split="test",
    ) if "test_path" in config["data"] else None
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=4,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=4,
    )
    
    # Models
    encoder = SessionEncoder(**config["model"])
    encoder.load_state_dict(torch.load(config["model"]["encoder_checkpoint"], map_location=device))
    encoder = encoder.to(device)
    
    classifier = ClassificationHead(
        input_dim=config["model"]["d_model"],
        num_classes=config["model"]["num_classes"],
        **config["classifier"],
    )
    classifier = classifier.to(device)
    
    # Freeze encoder if using LoRA
    if config["model"].get("use_lora", False):
        encoder.freeze_backbone()
    
    # Loss and optimizer
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = Adam(
        list(encoder.parameters()) + list(classifier.parameters()),
        lr=config["training"]["learning_rate"],
        weight_decay=config["training"]["weight_decay"],
    )
    
    # Learning rate scheduler
    warmup_scheduler = LinearLR(
        optimizer,
        start_factor=0.1,
        end_factor=1.0,
        total_iters=config["training"]["warmup_epochs"],
    )
    cosine_scheduler = CosineAnnealingLR(
        optimizer,
        T_max=config["training"]["num_epochs"] - config["training"]["warmup_epochs"],
    )
    scheduler = SequentialLR(
        optimizer,
        schedulers=[warmup_scheduler, cosine_scheduler],
        milestones=[config["training"]["warmup_epochs"]],
    )
    
    # Training loop
    best_val_f1 = 0.0
    patience_counter = 0
    
    for epoch in range(config["training"]["num_epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        encoder.train()
        classifier.train()
        total_loss = 0.0
        num_batches = 0
        
        for batch in train_loader:
            features = batch["features"].to(device)
            mask = batch["mask"].to(device)
            labels = batch["label"].to(device)
            
            embeddings, _ = encoder(features, mask)
            logits = classifier(embeddings)
            
            loss = criterion(logits, labels)
            
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                list(encoder.parameters()) + list(classifier.parameters()),
                max_norm=config["training"]["gradient_clip"],
            )
            optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        train_loss = total_loss / num_batches if num_batches > 0 else 0.0
        scheduler.step()
        
        # Validate
        val_loss, val_metrics = validate(encoder, classifier, val_loader, device)
        
        logger.info(
            f"Train loss: {train_loss:.4f}, Val loss: {val_loss:.4f}, "
            f"Val Macro-F1: {val_metrics['macro_f1']:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}"
        )
        
        # Save best model
        if val_metrics["macro_f1"] > best_val_f1:
            best_val_f1 = val_metrics["macro_f1"]
            patience_counter = 0
            
            torch.save(
                encoder.state_dict(),
                f"{config['output']['checkpoint_dir']}/best_encoder.pt",
            )
            torch.save(
                classifier.state_dict(),
                f"{config['output']['checkpoint_dir']}/best_classifier.pt",
            )
            logger.info(f"Saved best model with Macro-F1: {best_val_f1:.4f}")
        else:
            patience_counter += 1
            if patience_counter >= config["training"]["early_stopping_patience"]:
                logger.info("Early stopping triggered")
                break
    
    # Test evaluation
    if test_dataset:
        test_loader = DataLoader(test_dataset, batch_size=config["training"]["batch_size"], shuffle=False)
        test_loss, test_metrics = validate(encoder, classifier, test_loader, device)
        logger.info(f"Test Macro-F1: {test_metrics['macro_f1']:.4f}")
    
    logger.info("Fine-tuning completed!")


if __name__ == "__main__":
    main()

