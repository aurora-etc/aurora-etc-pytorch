# AURORA-ETC Project Structure

This document describes the complete repository structure and the purpose of each component.

## Directory Structure

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

## Key Features

- **Modular Design**: Each component is independently testable and reusable
- **Configuration-Driven**: All hyperparameters in YAML configs
- **Production-Ready**: Includes deployment pipeline and monitoring
- **Extensible**: Easy to add new datasets, models, or metrics

## Next Steps

1. Implement data loaders for specific datasets (ISCX-VPN, CSTNET-TLS, etc.)
2. Add visualization utilities for drift monitoring
3. Create example notebooks demonstrating usage
4. Add comprehensive documentation
5. Implement federated learning extensions (future work)

