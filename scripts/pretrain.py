#!/usr/bin/env python
"""
Pretraining script for AURORA-ETC session encoder
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader
from torch.optim import Adam
from torch.optim.lr_scheduler import CosineAnnealingLR, LinearLR, SequentialLR

from aurora_etc.models import SessionEncoder
from aurora_etc.data import EncryptedTrafficDataset, TrafficAugmentation, NormalizeTraffic
from aurora_etc.training import Pretrainer, PretrainingLoss
from aurora_etc.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Pretrain AURORA-ETC encoder")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger(log_file=config["output"]["log_dir"] + "/pretrain.log")
    logger.info("Starting pretraining...")
    
    # Device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    logger.info(f"Using device: {device}")
    
    # Data
    augmentation = TrafficAugmentation(**config["augmentation"])
    train_dataset = EncryptedTrafficDataset(
        config["data"]["train_path"],
        augmentation=augmentation,
        split="train",
    )
    val_dataset = EncryptedTrafficDataset(
        config["data"]["val_path"],
        split="val",
    )
    
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
    
    # Model
    model = SessionEncoder(**config["model"])
    model = model.to(device)
    
    # Loss and trainer
    loss_fn = PretrainingLoss(**config["loss"])
    trainer = Pretrainer(model, device, loss_fn)
    
    # Optimizer
    optimizer = Adam(
        model.parameters(),
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
    best_val_loss = float("inf")
    patience_counter = 0
    
    for epoch in range(config["training"]["num_epochs"]):
        logger.info(f"Epoch {epoch+1}/{config['training']['num_epochs']}")
        
        # Train
        train_loss = trainer.train_epoch(train_loader, optimizer)
        scheduler.step()
        
        logger.info(f"Train loss: {train_loss:.4f}, LR: {scheduler.get_last_lr()[0]:.6f}")
        
        # Validation (simplified - implement full validation loop)
        # val_loss = validate(trainer, val_loader)
        # logger.info(f"Val loss: {val_loss:.4f}")
        
        # Early stopping
        # if val_loss < best_val_loss:
        #     best_val_loss = val_loss
        #     patience_counter = 0
        #     trainer.save_checkpoint(
        #         f"{config['output']['checkpoint_dir']}/best_model.pt",
        #         optimizer,
        #     )
        # else:
        #     patience_counter += 1
        #     if patience_counter >= config["training"]["early_stopping_patience"]:
        #         logger.info("Early stopping triggered")
        #         break
    
    # Save final checkpoint
    trainer.save_checkpoint(
        f"{config['output']['checkpoint_dir']}/final_model.pt",
        optimizer,
    )
    logger.info("Pretraining completed!")


if __name__ == "__main__":
    main()

