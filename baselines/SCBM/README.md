# Stochastic Concept Bottleneck Models (SCBM)

**Reference:** Vandenhirtz et al., "Stochastic Concept Bottleneck Models", NeurIPS 2024

Original repository: [https://github.com/dvandenhirtz/SCBM](https://github.com/dvandenhirtz/SCBM)

## Overview

Stochastic CBMs introduce a distributional parameterization of concept relations, modeling uncertainty in concept predictions through amortized covariance estimation.

## Setup

Requires Python 3.11+, PyTorch, Hydra for config management.

```bash
pip install torch torchvision hydra-core omegaconf wandb scikit-learn
```

## Data Preparation

Dataset configs are in `configs/data/`:
- `configs/data/awa2.yaml`
- `configs/data/cifar100.yaml`
- `configs/data/inet100.yaml`

Update the `data_dir` field in each YAML config to point to your dataset location.

## Running Experiments

### Train SCBM on all datasets (3 seeds)

```bash
bash run.sh
```

### Individual dataset training

```bash
# AwA2 (ResNet-18)
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    +model=SCBM +data=awa2 \
    model.cov_type='amortized' \
    model.reg_precision='l1' \
    model.reg_weight=1 \
    experiment_name="awa2_SCBM_amortized_42" \
    seed=42 \
    logging.project=SCBM \
    logging.mode=offline \
    model.tag=SCBM_experiments \
    model.encoder_arch=resnet18 \
    model.j_epochs=70 model.c_epochs=70 model.t_epochs=70 \
    model.train_batch_size=512 model.val_batch_size=512

# CIFAR100 (ResNet-50)
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    +model=SCBM +data=cifar100 \
    model.cov_type='amortized' \
    model.reg_precision='l1' \
    model.reg_weight=1 \
    experiment_name="cifar100_SCBM_amortized_42" \
    seed=42 \
    logging.project=SCBM \
    logging.mode=offline \
    model.tag=SCBM_experiments \
    model.encoder_arch=resnet50 \
    model.j_epochs=70 model.c_epochs=70 model.t_epochs=70 \
    model.train_batch_size=32 model.val_batch_size=32 \
    model.learning_rate=0.00003

# ImageNet100 (ResNet-50)
CUDA_VISIBLE_DEVICES=0 python -u train.py \
    +model=SCBM +data=inet100 \
    model.cov_type='amortized' \
    model.reg_precision='l1' \
    model.reg_weight=1 \
    experiment_name="inet100_SCBM_amortized_42" \
    seed=42 \
    logging.project=SCBM \
    logging.mode=offline \
    model.tag=SCBM_experiments \
    model.encoder_arch=resnet50 \
    model.j_epochs=70 model.c_epochs=70 model.t_epochs=70 \
    model.train_batch_size=32 model.val_batch_size=32 \
    model.learning_rate=0.00003
```

### Per-dataset scripts

```bash
bash scripts/awa2.sh
bash scripts/cifar100.sh
bash scripts/inet100.sh
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `+model=SCBM` | Model type (SCBM, CBM, CEM, AR) |
| `+data=<dataset>` | Dataset config (awa2, cifar100, inet100) |
| `model.cov_type` | Covariance type (`amortized`) |
| `model.reg_precision` | Regularization type (`l1`) |
| `model.reg_weight` | Regularization weight |
| `model.encoder_arch` | Backbone (`resnet18`, `resnet50`) |
| `model.j_epochs` | Joint training epochs |
| `model.c_epochs` | Concept training epochs |
| `model.t_epochs` | Task training epochs |
| `seed` | Random seed |
| `logging.mode` | Wandb mode (`online`, `offline`, `disabled`) |

## Configuration

- Global config: `configs/config.yaml`
- Model configs: `configs/model/SCBM.yaml`, `configs/model/CBM.yaml`
- Data configs: `configs/data/awa2.yaml`, etc.

Logs are written to `./main_logs/`.

## Citation

```bibtex
@article{vandenhirtz2024stochastic,
  title={Stochastic concept bottleneck models},
  author={Vandenhirtz, Moritz and Laguna, Sonia and Marcinkevi{\v{c}}s, Ri{\v{c}}ards and Vogt, Julia},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={51787--51810},
  year={2024}
}
```
