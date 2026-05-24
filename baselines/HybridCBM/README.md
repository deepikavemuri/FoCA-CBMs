# Hybrid Concept Bottleneck Models

**Reference:** Liu et al., "Hybrid Concept Bottleneck Models", CVPR 2025

Original repository: [https://github.com/deeplearning-wisc/HybridCBM](https://github.com/deeplearning-wisc/HybridCBM)

## Overview

HybridCBMs introduce a hybrid concept bank comprising both a static concept bank (from text-based concepts) and a dynamic concept bank (learned from data), combining interpretability with expressiveness.

## Setup

```bash
conda create --name hybridcbm python=3.11
conda activate hybridcbm
pip install -r requirements.txt
```

Additionally, install [cuML](https://github.com/rapidsai/cuml) for the linear probe:
```bash
pip install \
    --extra-index-url=https://pypi.nvidia.com \
    "cudf-cu12==24.10.*" "dask-cudf-cu12==24.10.*" "cuml-cu12==24.10.*"
```

Key dependencies: PyTorch, torchvision, CLIP, mmengine, apricot-select, cuML

## Data Preparation

Dataset structure:
```
datasets/<dataset>/
├── images/          # Download and place images here
├── concepts/        # concept.csv (provided)
└── splits/          # train.csv, val.csv, test.csv (provided)
```

Provided datasets with splits and concepts: `awa2`, `CIFAR100`, `inet100`

Download images for each dataset and place in the `images/` directory.

## Running Experiments

### Step 1: Train Translator (optional, pre-trained available)

```bash
bash train_translator.sh
```

Pre-trained RN50 translator: set the path in `config/HybridCBM/base.py`:
```python
translator_path = 'weights/translator/RN50-AUG_True/translator.pt'
```

### Step 2: Train HybridCBM

```bash
# Using the training script (pass dataset and GPU id)
bash script_train.sh awa2 0
bash script_train.sh cifar100 0
bash script_train.sh inet100 0

# Or run directly
CUDA_VISIBLE_DEVICES=0 python trainLinear.py \
    --config config/HybridCBM/awa2/awa2_allshot.py \
    --cfg-options clip_model=RN50 \
    --cfg-options concept_select_fn=submodular \
    --cfg-options num_concept_per_class=10 \
    --cfg-options dynamic_concept_ratio=0.5 \
    --cfg-options lambda_discri_alpha=2 \
    --cfg-options lambda_discri_beta=0.1 \
    --cfg-options lambda_ort=0.1 \
    --cfg-options lambda_align=0.01 \
    --cfg-options seed=42
```

### Step 3: Test

```bash
# Using test script (pass GPU id and experiment path)
bash script_test.sh 0 exp/HybridCBM/<dataset>_Zero_L1

# Or run directly
CUDA_VISIBLE_DEVICES=0 python trainLinear.py --exp_root exp/HybridCBM/<dataset>_Zero_L1 --test
```

## Configuration

Config files are in `config/HybridCBM/<dataset>/`:
- `config/HybridCBM/awa2/awa2_allshot.py`
- `config/HybridCBM/cifar100/cifar100_allshot.py`
- `config/HybridCBM/inet100/inet100_allshot.py`

Base config: `config/HybridCBM/base.py`

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--config` | Path to dataset config file |
| `clip_model` | CLIP model variant (`RN50`, `ViT-L/14`) |
| `concept_select_fn` | Concept selection method (`submodular`) |
| `num_concept_per_class` | Number of concepts per class |
| `dynamic_concept_ratio` | Ratio of dynamic vs static concepts |
| `lambda_discri_alpha/beta` | Discriminative loss weights |
| `lambda_ort` | Orthogonality loss weight |
| `lambda_align` | Alignment loss weight |
| `seed` | Random seed |

## Citation

```bibtex
@inproceedings{liu2025hybrid,
  title={Hybrid Concept Bottleneck Models},
  author={Liu, Yang and Zhang, Tianwei and Gu, Shi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20179--20189},
  year={2025}
}
```
