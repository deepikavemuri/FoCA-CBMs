# Coarse-to-Fine Concept Bottleneck Models (CF-CBM)

**Reference:** Panousis et al., "Coarse-to-Fine Concept Bottleneck Models", NeurIPS 2024

Original repository: [https://github.com/konpanousis/CF-CBMs](https://github.com/konpanousis/CF-CBMs)

## Overview

CF-CBMs propose a multi-level construction that captures both coarse (high-level) and fine (low-level) concepts for interpretable classification. Models are trained using CLIP embeddings with linear layers on top.

## Setup

```bash
conda env create -f clip_env.yml
conda activate <env_name>
```

Requires: PyTorch, CLIP, numpy, scikit-learn

## Project Structure

```
Coarse-To-Fine-CBMs/
├── clip/                    # CLIP model code
├── data/
│   ├── concept_sets_high/   # High-level (class-name) concept sets
│   └── concept_sets_low/    # Low-level (attribute) concept sets
├── scripts/                 # Data preprocessing scripts
├── main.py                  # Entry point
├── networks.py              # Model architectures
├── data_utils.py            # Data loading utilities
└── utils.py                 # General utilities
```

## Data Preparation

1. Download datasets and set paths in `data_utils.py`
2. Low-level concept sets (binary attribute matrices) are provided in `data/concept_sets_low/`:
   - `AwA2/awa2_attrs_per_class_binary_85.npy`
   - `CIFAR100/cifar100_attrs_per_class_binary.npy`
   - `Imagenet100/imagenet100_attrs_per_class_binary_20.npy`

### Optional: Preprocessing Scripts

```bash
# Convert AwA2 images to numpy format
python scripts/convert_awa2_to_npy.py

# Create CIFAR100 attribute annotations
python scripts/create_cifar100_attributes.py

# Reorganize AwA2 into ImageFolder format
python scripts/reorg_awa2_to_imagefolder.py
```

## Running Experiments

### Step 1: Compute CLIP Similarities

Before training, compute and cache CLIP embeddings:

```bash
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --compute_similarities --batch_size 256
CUDA_VISIBLE_DEVICES=0 python main.py --dataset awa2 --compute_similarities --batch_size 256
CUDA_VISIBLE_DEVICES=0 python main.py --dataset inet100 --compute_similarities --batch_size 256
```

### Step 2: Train CF-CBM

```bash
# CIFAR100
CUDA_VISIBLE_DEVICES=0 python main.py --dataset cifar100 --batch_size 256 --epochs 30

# AwA2
CUDA_VISIBLE_DEVICES=0 python main.py --dataset awa2 --batch_size 256 --epochs 30

# ImageNet100
CUDA_VISIBLE_DEVICES=0 python main.py --dataset inet100 --batch_size 256 --epochs 30
```

### Run all datasets

```bash
bash run.sh 0  # Pass GPU id as argument
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset name (`cifar100`, `awa2`, `inet100`) |
| `--compute_similarities` | Flag to compute and cache CLIP embeddings |
| `--batch_size` | Batch size for training/embedding computation |
| `--epochs` | Number of training epochs |

Models are saved to `saved_models/` (created automatically).

## Citation

```bibtex
@inproceedings{panousis2024coarsetofine,
  title={Coarse-to-Fine Concept Bottleneck Models},
  author={Konstantinos P. Panousis and Dino Ienco and Diego Marcos},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
  year={2024}
}
```
