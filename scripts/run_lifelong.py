#!/usr/bin/env python
"""
Lifelong learning pipeline script
"""

import argparse
import yaml
import torch
from torch.utils.data import DataLoader

from aurora_etc.models import SessionEncoder, ClassificationHead
from aurora_etc.drift import DriftDetector
from aurora_etc.training import OnlineUpdater, ReplayBuffer
from aurora_etc.automl import AutoMLReconfigurator, SearchSpace
from aurora_etc.deployment import DeploymentPipeline
from aurora_etc.utils.logging import setup_logger


def main():
    parser = argparse.ArgumentParser(description="Run AURORA-ETC lifelong learning")
    parser.add_argument("--config", type=str, required=True, help="Path to config file")
    args = parser.parse_args()
    
    # Load config
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Setup logger
    logger = setup_logger(log_file=config["output"]["log_dir"] + "/lifelong.log")
    logger.info("Starting lifelong learning pipeline...")
    
    # Device
    device = torch.device(config["device"] if torch.cuda.is_available() else "cpu")
    
    # Load pretrained models
    encoder = SessionEncoder(**config["model"])
    encoder.load_state_dict(torch.load(config["model"]["encoder_checkpoint"]))
    encoder = encoder.to(device)
    
    classifier = ClassificationHead(
        input_dim=config["model"]["d_model"],
        num_classes=config["model"]["num_classes"],
        **config["classifier"],
    )
    if "classifier_checkpoint" in config["model"]:
        classifier.load_state_dict(torch.load(config["model"]["classifier_checkpoint"]))
    classifier = classifier.to(device)
    
    # Drift detector
    drift_detector = DriftDetector(**config["drift"])
    
    # Replay buffer
    replay_buffer = ReplayBuffer(**config["replay"])
    
    # Online updater
    online_updater = OnlineUpdater(encoder, classifier, device, replay_buffer)
    
    # AutoML reconfigurator
    search_space = SearchSpace(**config["automl"]["search_space"])
    automl = AutoMLReconfigurator(
        search_space,
        device,
        **{k: v for k, v in config["automl"].items() if k != "search_space"},
    )
    
    # Deployment pipeline
    deployment = DeploymentPipeline(**config["deployment"])
    
    logger.info("Lifelong learning pipeline initialized. Ready for deployment.")


if __name__ == "__main__":
    main()

