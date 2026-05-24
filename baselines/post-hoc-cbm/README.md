# Post-hoc Concept Bottleneck Models

**Reference:** Yuksekgonul et al., "Post-hoc Concept Bottleneck Models", ICLR 2023 (Spotlight)

Original repository: [https://github.com/mertyg/post-hoc-cbm](https://github.com/mertyg/post-hoc-cbm)

## Overview

Post-hoc CBMs convert a trained black-box model into a concept bottleneck model using CLIP-based concept activation vectors. This allows interpreting any pre-trained model without retraining from scratch.

## Setup

```bash
pip install torch torchvision clip scikit-learn numpy
```

Requires OpenAI CLIP: `pip install git+https://github.com/openai/CLIP.git`

## Data Preparation

1. Download datasets (ImageNet100, AwA2, CIFAR100) and place in your data directory
2. Update `data_utils/constants.py` with dataset paths
3. Ensure concept lists are available at `DATA/concepts/<dataset>_concepts.txt`

## Running Experiments

The pipeline has three sequential steps:

### Step 1: Learn Concept Bank (via CLIP)

```bash
CUDA_VISIBLE_DEVICES=0 python learn_concepts_multimodal.py \
    --classes=awa2 \
    --backbone-name="clip:RN50" \
    --concept_list="/path/to/DATA/concepts/awa2_concepts.txt" \
    --out-dir=./outputs
```

### Step 2: Train PCBM

```bash
CUDA_VISIBLE_DEVICES=0 python train_pcbm.py \
    --concept-bank="./outputs/multimodal_concept_clip:RN50_awa2_recurse:1.pkl" \
    --dataset=awa2 \
    --backbone-name="clip:RN50" \
    --out-dir=./outputs \
    --data_path=/path/to/awa2/ \
    --lam=2e-4 \
    --seed 42
```

### Step 3: Train PCBM-h (Hybrid, optional)

```bash
CUDA_VISIBLE_DEVICES=0 python train_pcbm_h.py \
    --concept-bank="./outputs/multimodal_concept_clip:RN50_awa2_recurse:1.pkl" \
    --pcbm-path="./outputs/pcbm_awa2__clip:RN50__multimodal_concept_clip:RN50_awa2_recurse:1__lam:0.0002__alpha:0.99__seed:42.ckpt" \
    --out-dir=./outputs \
    --dataset=awa2 \
    --data_path=/path/to/awa2/
```

### Run All Datasets (3 seeds)

```bash
# Edit paths in run.sh, then:
bash run.sh
```

Supported datasets: `awa2`, `inet100`, `cifar100`

## Key Arguments

| Argument | Description |
|----------|-------------|
| `--classes` | Dataset name for concept bank generation |
| `--backbone-name` | CLIP backbone (`clip:RN50`) |
| `--concept_list` | Path to concept text file |
| `--lam` | Regularization strength for sparse linear layer |
| `--seed` | Random seed |

## Citation

```bibtex
@inproceedings{yuksekgonul2023posthoc,
  title={Post-hoc Concept Bottleneck Models},
  author={Mert Yuksekgonul and Maggie Wang and James Zou},
  booktitle={The Eleventh International Conference on Learning Representations},
  year={2023}
}
```
