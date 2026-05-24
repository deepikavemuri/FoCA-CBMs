# Concept Embedding Models (CEM)

**Reference:** Espinosa Zarlenga et al., "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off", NeurIPS 2022

Original repository: [https://github.com/mateoespinosa/cem](https://github.com/mateoespinosa/cem)

## Overview

Concept Embedding Models (CEMs) learn high-dimensional concept embeddings that go beyond the accuracy-explainability trade-off of standard CBMs. They support effective test-time concept interventions while maintaining high task accuracy.

## Setup

```bash
# Install the CEM package
cd cem
python setup.py install
cd ..

# Install additional requirements
pip install -r requirements.txt
```

Key dependencies: PyTorch, PyTorch Lightning, scikit-learn

## Data Preparation

Dataset configs are in `experiments/configs/`:
- `experiments/configs/awa2.yaml`
- `experiments/configs/cifar100.yaml`
- `experiments/configs/inet100.yaml`

Update the `root_dir` parameter in each YAML config to point to your dataset location, or set the `DATASET_DIR` environment variable.

## Running Experiments

### Train CEM on individual datasets

```bash
# AwA2
python experiments/run_experiments.py -c experiments/configs/awa2.yaml

# CIFAR100
python experiments/run_experiments.py -c experiments/configs/cifar100.yaml

# ImageNet100
python experiments/run_experiments.py -c experiments/configs/inet100.yaml
```

### Run all datasets sequentially

```bash
bash run.sh
```

## Output

After execution, results are saved in the output directory specified in the YAML config:
- Trained model checkpoints
- Training logs
- `results.joblib` — dictionary with test/validation metrics summarized over multiple seeds

## Key Config Parameters

Edit the YAML files in `experiments/configs/` to modify:

| Parameter | Description |
|-----------|-------------|
| `n_concepts` | Number of concepts |
| `n_tasks` | Number of output classes |
| `emb_size` | Concept embedding dimension (default: 16) |
| `concept_loss_weight` | Weight for concept prediction loss |
| `learning_rate` | Learning rate |
| `training_intervention_prob` | RandInt probability (default: 0.25) |
| `max_epochs` | Number of training epochs |

## Citation

```bibtex
@article{EspinosaZarlenga2022cem,
  title={Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off},
  author={Espinosa Zarlenga, Mateo and Barbiero, Pietro and Ciravegna, Gabriele and
    Marra, Giuseppe and Giannini, Francesco and Diligenti, Michelangelo and
    Shams, Zohreh and Precioso, Frederic and Melacci, Stefano and
    Weller, Adrian and Lio, Pietro and Jamnik, Mateja},
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```
