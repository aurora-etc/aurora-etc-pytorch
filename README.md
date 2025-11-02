# AURORA-ETC: Lifelong Encrypted Traffic Classification

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.4+-orange.svg)](https://pytorch.org/)

Official PyTorch implementation of **AURORA-ETC: Lifelong Encrypted Traffic Classification through Self-Supervised Pretraining and AutoML-Guided Adaptation**.

## Overview

AURORA-ETC is a lifelong learning framework for encrypted traffic classification that combines:
- **Self-supervised pretraining** on large-scale unlabeled TLS 1.3 and QUIC traffic
- **Lightweight online updates** using LoRA (Low-Rank Adaptation) and replay buffers
- **AutoML-guided reconfiguration** for severe drift scenarios
- **Protocol-aware drift sensing** for proactive adaptation

<p align="center">
  <img src="assets/fig-model.png" width="100%" alt="Model Overview">
  <img src="assets/fig-self-supervised-learning.png" width="60%" alt="Self-Supervised Learning">
  <img src="assets/fig-deploy.png" width="60%" alt="Model Overview">
</p>


## Features

- Two-tier adaptation: Fast LoRA updates for moderate drift, AutoML reconfiguration for severe drift
- Continual learning with minimal catastrophic forgetting
- Multi-signal drift detection (MMD, ECE, protocol telemetry)
- Resource-efficient: Low latency, high throughput, minimal memory overhead
- Privacy-preserving: Works on encrypted traffic features without payload inspection

## Pre-Installation
### 1. Download and install Wireshark
* While installing wireshark, CHECK the following option: Install Npcap in WinPcap API-compatible mode
### 2. Set Wireshark to environment variable
* In start menu search environment
* Edit the system environment variables > Environment Variables
* In user variable double-click Path
* Click New
* Paste the Wireshark installation path: (C:\Program Files\Wireshark)
* Click Ok > OK

## Installation

```bash
# Clone the repository
git clone https://github.com/aurora-etc/aurora-etc-pytorch.git
cd aurora-etc

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

<p align="center">
  <img src="assets/fig-pre-performance.png" width="100%" alt="Results">
</p>

## Quick Start

### 1. Pretrain the Session Encoder

```bash
python scripts/pretrain.py --config configs/pretrain.yaml
```

### 2. Fine-tune on Labeled Data

```bash
python scripts/finetune.py --config configs/finetune.yaml --pretrained_checkpoint checkpoints/pretrained/encoder.pt
```

### 3. Run Lifelong Learning Pipeline

```bash
python scripts/run_lifelong.py --config configs/lifelong.yaml
```

<p align="center">
  <img src="assets/results.png" width="100%" alt="Results">
  <img src="assets/legends.png" width="100%" alt="Results">
</p>



## Implementation Status

### ✅ Completed Components

#### 1. Core Models (`aurora_etc/models/`)
- **SessionEncoder**: Transformer-based encoder for processing encrypted traffic flows
  - Supports LoRA integration
  - Flexible pooling strategies (mean, CLS, max)
  - Configurable architecture (d_model, nhead, num_layers, etc.)

- **LoRALinear**: Low-Rank Adaptation implementation
  - Parameter-efficient fine-tuning
  - Configurable rank and alpha
  - Weight merging/unmerging support

- **ClassificationHead**: Classification head with optional cosine similarity
  - Standard and cosine-based classifiers
  - Configurable hidden layers

- **Transformer Components**: Positional encoding and transformer blocks

#### 2. Data Processing (`aurora_etc/data/`)
- **FlowPreprocessor**: Preprocesses packet sequences into feature vectors
- **PacketFeatureExtractor**: Extracts packet-level features (size, direction, IAT)
- **TrafficAugmentation**: Data augmentation for self-supervised learning
  - Random cropping
  - Jittering
  - Masking
- **Dataset Classes**: PyTorch Dataset implementations for encrypted traffic

#### 3. Drift Detection (`aurora_etc/drift/`)
- **DriftDetector**: Unified drift detection module
  - Combines MMD, ECE, uncertainty, and protocol telemetry
  - Configurable thresholds and weights
- **MMD Computation**: Maximum Mean Discrepancy for feature drift
- **ECE Computation**: Expected Calibration Error for confidence drift

#### 4. Training Modules (`aurora_etc/training/`)
- **Loss Functions**:
  - InfoNCE contrastive loss
  - Masked modeling loss
  - Knowledge distillation loss
- **Pretrainer**: Self-supervised pretraining loop
- **OnlineUpdater**: Lightweight online updates with LoRA
- **ReplayBuffer**: Buffer management for continual learning
  - Multiple selection strategies (uncertainty, random, class-balanced)

#### 5. AutoML Reconfiguration (`aurora_etc/automl/`)
- **AutoMLReconfigurator**: Bayesian optimization-based architecture search
  - Resource constraint checking (latency, memory, throughput)
  - Knowledge distillation integration
- **SearchSpace**: Configurable architecture search space

#### 6. Deployment (`aurora_etc/deployment/`)
- **DeploymentPipeline**: Staged deployment (shadow → canary → full)
  - SLO monitoring
  - Automatic rollback

#### 7. Utilities (`aurora_etc/utils/`)
- **Metrics**: Evaluation metrics (macro-F1, BWT, FWT, OOD AUROC)
- **Logging**: Logging utilities

#### 8. Training Scripts (`scripts/`)
- **pretrain.py**: Self-supervised pretraining script
- **finetune.py**: Supervised fine-tuning script
- **run_lifelong.py**: Lifelong learning pipeline script

#### 9. Configuration Files (`configs/`)
- **pretrain.yaml**: Pretraining configuration
- **finetune.yaml**: Fine-tuning configuration
- **lifelong.yaml**: Lifelong learning configuration

#### 10. Testing (`tests/`)
- Unit tests for models and drift detection






## Project Structure

```
aurora-etc/
├── aurora_etc/                    # Main package
│   ├── __init__.py               # Package initialization
│   │
│   ├── models/                   # Model architectures
│   │   ├── __init__.py
│   │   ├── encoder.py           # Session encoder (Transformer-based)
│   │   ├── lora.py              # LoRA (Low-Rank Adaptation) implementation
│   │   ├── classifier.py        # Classification head
│   │   └── transformers.py      # Transformer components
│   │
│   ├── data/                     # Data processing
│   │   ├── __init__.py
│   │   ├── datasets.py          # Dataset classes
│   │   ├── preprocessing.py     # Flow preprocessing
│   │   └── transforms.py        # Data augmentation
│   │
│   ├── drift/                    # Drift detection
│   │   ├── __init__.py
│   │   ├── detector.py          # Unified drift detector
│   │   ├── mmd.py               # Maximum Mean Discrepancy
│   │   └── calibration.py       # Expected Calibration Error (ECE)
│   │
│   ├── training/                 # Training modules
│   │   ├── __init__.py
│   │   ├── losses.py            # Loss functions (contrastive, masked, distillation)
│   │   ├── trainer.py           # Training loops (pretraining, online updates)
│   │   └── replay_buffer.py     # Replay buffer for continual learning
│   │
│   ├── automl/                   # AutoML reconfiguration
│   │   ├── __init__.py
│   │   ├── searcher.py          # Bayesian optimization search
│   │   └── search_space.py      # Architecture search space definition
│   │
│   ├── deployment/               # Deployment pipeline
│   │   ├── __init__.py
│   │   └── pipeline.py          # Shadow/canary/rollout deployment
│   │
│   └── utils/                    # Utilities
│       ├── __init__.py
│       ├── metrics.py           # Evaluation metrics (F1, BWT, FWT, etc.)
│       └── logging.py           # Logging utilities
│
├── scripts/                      # Training and evaluation scripts
│   ├── pretrain.py              # Self-supervised pretraining
│   ├── finetune.py              # Supervised fine-tuning
│   └── run_lifelong.py          # Lifelong learning pipeline
│
├── configs/                      # Configuration files
│   ├── pretrain.yaml            # Pretraining configuration
│   ├── finetune.yaml            # Fine-tuning configuration
│   └── lifelong.yaml            # Lifelong learning configuration
│
├── tests/                        # Unit tests
│   ├── __init__.py
│   ├── test_models.py           # Model tests
│   └── test_drift.py            # Drift detection tests
│
├── examples/                     # Example notebooks and scripts
│   └── (to be added)
│
├── docs/                         # Documentation
│   └── (to be added)
│
├── README.md                     # Main README
├── PROJECT_STRUCTURE.md          # This file
├── LICENSE                       # MIT License
├── requirements.txt              # Python dependencies
├── setup.py                      # Package setup
└── .gitignore                    # Git ignore rules
```


## Download and prepare the datasets
### 1. ISCX-VPN-NonVPN-2016
* Download the following files from the link:
http://205.174.165.80/CICDataset/ISCX-VPN-NonVPN-2016/Dataset/PCAPs/

```
1. NonVPN-PCAPs-01.zip
2. NonVPN-PCAPs-02.zip
3. NonVPN-PCAPs-03.zip
4. VPN-PCAPS-01.zip
5. VPN-PCAPS-02.zip
```
* Extract these following files in folder datasets/ISCX_VPN
* Finally the dataset structure should look like:
```
-datasets
 - ISCX_VPN
  - aim_chat_3a.pcap
  - aim_chat_3b.pcap
	...
  - youtubeHTML5_1.pcap
  - vpn_aim_chat1a.pcap
  - vpn_aim_chat1b.pcap
	...
  - vpn_youtube_A.pcap
```

### 2. ISCX-TOR
* Download page (PCAPs + CSV flows): Canadian Institute for Cybersecurity (UNB) Tor-nonTor dataset page https://www.unb.ca/cic/datasets/tor.html
* Extract into datasets/ISCX_TOR/
* Final folder structure (example):

```
datasets/
  ISCX_TOR/
    tor_browsing.pcap
    tor_email.pcap
    tor_voip.pcap
    nontor_browsing.pcap
    nontor_chat.pcap
    ...
    flows/
      TorNonTor-FlowMeter.csv   # if you use the provided CSV flows
```

### 3. CSTNET-TLS 1.3
* Download CSTNET-TLS 1.3 from the link referenced in ET-BERT's https://github.com/linwhitehat/ET-BERT.
* Place/rename files to match the packet/*.tsv paths used by ET-BERT (its examples call these exact filenames). 
```
datasets/
  CSTNET_TLS13/
    packet/
      train_dataset.tsv
      valid_dataset.tsv
      test_dataset.tsv
      nolabel_test_dataset.tsv   # optional, if provided
```

### 4. CESNET-QUIC22
* Download cesnet-quic22.zip (flows as CSV) and servicemap.csv from official dataset archive https://zenodo.org/records/10728760
* The Zenodo page includes schema, size, and per-week stats. 
* Extract into datasets/CESNET_QUIC22/
* Final folder structure:
```
datasets/
  CESNET_QUIC22/
    cesnet-quic22/
      week-2022-44.csv.gz
      week-2022-45.csv.gz
      week-2022-46.csv.gz
      week-2022-47.csv.gz
      dataset-statistics/...
    servicemap.csv
```

### 5. UCDavisQUIC
* Download Kaggle "UCDavisQUIC" (per-flow PCAPs organized by service) https://www.kaggle.com/datasets/guillaumefraysse/ucdavisquic?utm_source=chatgpt.com
* Extract into datasets/UCDAVIS_QUIC/
* Final folder structure:
```
datasets/
  UCDAVIS_QUIC/
    GoogleDocs/
      flow_00001.pcap
      ...
    GoogleDrive/
    GoogleMusic/
    GoogleSearch/
    YouTube/
```

### 6. WNL TLS
* Dataset datatset from "Novel QoS-aware TLS Dataset for Encrypted Traffic Classification" at https://wnlab.ru/qos-tls-dataset-of-enc-traffic/?utm_source=chatgpt.com
* Describes 3,547 flows across 12 services and 4 traffic types.
* Extract into datasets/WNL_TLS/
* Final folder structure (example):
```
datasets/
  WNL_TLS/
    raw_pcaps/
      service_A_*.pcap
      ...
    labels.csv          # if provided with service/type labels
```


## Component Descriptions

### Models (`aurora_etc/models/`)

- **encoder.py**: Transformer-based session encoder that processes encrypted traffic flows
- **lora.py**: Low-Rank Adaptation implementation for parameter-efficient fine-tuning
- **classifier.py**: Classification head with optional cosine similarity and margin-based loss
- **transformers.py**: Transformer building blocks (positional encoding, transformer blocks)

### Data Processing (`aurora_etc/data/`)

- **datasets.py**: PyTorch Dataset classes for encrypted traffic
- **preprocessing.py**: Flow preprocessing and feature extraction
- **transforms.py**: Data augmentation (cropping, jittering, masking) for self-supervised learning

### Drift Detection (`aurora_etc/drift/`)

- **detector.py**: Unified drift detector combining MMD, ECE, uncertainty, and protocol telemetry
- **mmd.py**: Maximum Mean Discrepancy computation for feature drift detection
- **calibration.py**: Expected Calibration Error for confidence drift detection

### Training (`aurora_etc/training/`)

- **losses.py**: Loss functions including InfoNCE contrastive loss, masked modeling loss, and knowledge distillation
- **trainer.py**: Training loops for pretraining and online updates
- **replay_buffer.py**: Replay buffer implementation for catastrophic forgetting prevention

### AutoML (`aurora_etc/automl/`)

- **searcher.py**: AutoML-guided reconfiguration using Bayesian optimization (Optuna)
- **search_space.py**: Architecture search space definition

### Deployment (`aurora_etc/deployment/`)

- **pipeline.py**: Staged deployment pipeline (shadow → canary → full rollout) with SLO monitoring

### Utilities (`aurora_etc/utils/`)

- **metrics.py**: Evaluation metrics (macro-F1, BWT, FWT, OOD AUROC)
- **logging.py**: Logging setup and utilities

## Workflow

1. **Pretraining** (`scripts/pretrain.py`): Train session encoder on unlabeled traffic using self-supervised learning
2. **Fine-tuning** (`scripts/finetune.py`): Fine-tune encoder and train classifier on labeled data
3. **Lifelong Learning** (`scripts/run_lifelong.py`): Run continuous adaptation pipeline with drift detection and online updates


## Results

<details>
<summary>Performance Table</summary>

<!-- table here -->

</details>


## Citation

If you use this code in your research, please cite:

```bibtex
@article{aurora-etc2025,
  title={AURORA-ETC: Lifelong Encrypted Traffic Classification through Self-Supervised Pretraining and AutoML-Guided Adaptation},
  author={},
  journal={IEEE Transactions on Dependable and Secure Computing},
  year={2025}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

