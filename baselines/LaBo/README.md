# LaBo: Language in a Bottle

**Reference:** Yang et al., "Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification", CVPR 2023

Original repository: [https://github.com/YueYANG1996/LaBo](https://github.com/YueYANG1996/LaBo)

## Overview

LaBo leverages an LLM to define a large space of possible concept bottlenecks and uses submodular optimization to select the best concept set. Concepts are aligned with CLIP to form an interpretable bottleneck layer.

## Setup

```bash
conda create --name labo python=3.9.13
conda activate labo
pip install -r requirements.txt
```

Key dependencies: PyTorch, CLIP, mmcv, apricot-select, wandb, pytorch-lightning

**Note:** You may need to modify the source code of [Apricot](https://github.com/jmschrei/apricot) to run submodular optimization. See [this issue](https://github.com/YueYANG1996/LaBo/issues/1) for details.

## Data Preparation

Dataset-specific files are in `datasets/`:
```
datasets/
├── awa2/
│   ├── images/          # Download and place images here
│   ├── concepts/        # class2concepts.json (provided)
│   └── splits/          # Train/val/test splits (provided)
├── CIFAR100/
│   ├── images/
│   ├── concepts/
│   └── splits/
└── inet100/
    ├── images/
    ├── concepts/
    └── splits/
```

You need to download dataset images and place them in the `images/` subfolder for each dataset.

## Running Experiments

### Training

Train LaBo on all datasets:

```bash
# All datasets sequentially (pass GPU id as argument)
bash run.sh 0

# Individual datasets
CUDA_VISIBLE_DEVICES=0 python main.py \
    --cfg cfg/asso_opt/CIFAR100/CIFAR100_allshot_fac.py \
    --work-dir exp/asso_opt/CIFAR100/CIFAR100_allshot_fac \
    --func asso_opt_main

CUDA_VISIBLE_DEVICES=0 python main.py \
    --cfg cfg/asso_opt/awa2/awa2_allshot_fac.py \
    --work-dir exp/asso_opt/awa2/awa2_allshot_fac \
    --func asso_opt_main

CUDA_VISIBLE_DEVICES=0 python main.py \
    --cfg cfg/asso_opt/inet100/inet100_allshot_fac.py \
    --work-dir exp/asso_opt/inet100/inet100_allshot_fac \
    --func asso_opt_main
```

Training logs are uploaded to wandb. Checkpoints are saved in `exp/asso_opt/<dataset>/`.

### Testing

```bash
# General form
bash labo_test.sh <config_path> <checkpoint_path>

# Example
bash labo_test.sh cfg/asso_opt/awa2/awa2_allshot_fac.py exp/asso_opt/awa2/awa2_allshot_fac/best.ckpt
```

## Configuration

Config files are in `cfg/asso_opt/<dataset>/`:
- `cfg/asso_opt/awa2/awa2_allshot_fac.py`
- `cfg/asso_opt/CIFAR100/CIFAR100_allshot_fac.py`
- `cfg/asso_opt/inet100/inet100_allshot_fac.py`

Modify these files to change hyperparameters, paths, and training settings.

## Citation

```bibtex
@inproceedings{yang2023language,
  title={Language in a bottle: Language model guided concept bottlenecks for interpretable image classification},
  author={Yang, Yue and Panagopoulou, Artemis and Zhou, Shenghao and Jin, Daniel and Callison-Burch, Chris and Yatskar, Mark},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19187--19197},
  year={2023}
}
```
