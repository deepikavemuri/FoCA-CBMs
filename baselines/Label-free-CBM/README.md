# Label-free Concept Bottleneck Models

**Reference:** Oikarinen et al., "Label-free Concept Bottleneck Models", ICLR 2023

Original repository: [https://github.com/Trustworthy-ML-Lab/Label-free-CBM](https://github.com/Trustworthy-ML-Lab/Label-free-CBM)

## Overview

Label-free CBMs transform any neural network into an interpretable CBM without requiring labeled concept data. Concepts are derived using LLMs and aligned using CLIP-Dissect.

## Setup

```bash
pip install -r requirements.txt
```

Key dependencies: PyTorch, CLIP, scikit-learn, OpenAI API (for concept generation)

## Data Preparation

1. Download datasets and set paths in `data_utils.py` under `DATASET_ROOTS`
2. Concept sets are provided in `DATA/concepts/` as `.txt` files (one concept per line)

### Optional: Generate Custom Concept Sets

If you want to regenerate concepts using GPT:
1. `GPT_initial_concepts.ipynb` — Generate initial concepts (requires OpenAI API key)
2. `GPT_conceptset_processor.ipynb` — Filter and process concept sets

## Running Experiments

### Train Label-free CBM

```bash
# ImageNet100
CUDA_VISIBLE_DEVICES=0 python train_cbm.py \
    --seed 42 \
    --dataset imagenet100 \
    --backbone resnet50 \
    --concept_set /path/to/DATA/concepts/inet100_concepts.txt \
    --clip_cutoff 0.28 \
    --n_iters 1000 \
    --lam 0.0001

# AwA2
CUDA_VISIBLE_DEVICES=0 python train_cbm.py \
    --seed 42 \
    --dataset awa2 \
    --data_root /path/to/Animals_with_Attributes2 \
    --backbone resnet18 \
    --concept_set /path/to/DATA/concepts/awa2_concepts.txt \
    --clip_cutoff 0.26 \
    --n_iters 1000 \
    --lam 0.0001

# CIFAR100
CUDA_VISIBLE_DEVICES=0 python train_cbm.py \
    --seed 42 \
    --dataset cifar100 \
    --data_root /path/to/cifar100 \
    --backbone resnet50 \
    --concept_set /path/to/DATA/concepts/cifar100_concepts.txt \
    --clip_cutoff 0.26 \
    --n_iters 1000 \
    --lam 0.0001
```

### Run All Datasets (3 seeds)

```bash
# Edit paths in run.sh, then:
bash run.sh
```

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--dataset` | Dataset name (`imagenet100`, `awa2`, `cifar100`) |
| `--backbone` | Backbone architecture (`resnet18`, `resnet50`) |
| `--concept_set` | Path to concept text file |
| `--clip_cutoff` | CLIP similarity threshold for concept filtering |
| `--n_iters` | Number of training iterations for sparse linear layer |
| `--lam` | Sparsity regularization weight |
| `--seed` | Random seed |

## Evaluation

Use `evaluate_cbm.ipynb` to measure accuracy and visualize concept explanations.

## Citation

```bibtex
@inproceedings{oikarinenlabel,
  title={Label-free Concept Bottleneck Models},
  author={Oikarinen, Tuomas and Das, Subhro and Nguyen, Lam M and Weng, Tsui-Wei},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
