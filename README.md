# Repository for FoCA-CBM

### File Structure

`baselines` contains code for few of the baselines used in the paper

`fca4nn` contains code for Vanilla-CBM, MLP-CBM, and both of our methods, FoCA-CBMs 

`DATA` contains the concept set used for all the datasets and lattices that were generated using FCA for each of the datasets.

## How to RUN

### FoCA CBM

In `fca4nn` folder, inside the `scripts` subfolder contains scripts to run FoCA-CBM for each of the datasets. 

Inside the scripts, change the path for the datasets and other required paths.

For example: 
```
python main.py \
    --do_train \
    --do_test \
    --seed 42 \
    --dataset awa2 \
    --model resnet18 \
    --concept_wts 0.01 \
    --cls_wts 0.01 \
    --data_root <path to dataset folder> \
    --concept_file <path to concept files> \
    --lattice_path <path to dataset lattice> \
    --num_clfs 2 \
    --lattice_levels 1 3 \
    --backbone_layer_ids 3 4 \
    --lr 3e-4 \
    --epochs 75 \
    --batch_size 256 \
    --verbose 10 \
    --keep_top_k 2 \
    --clf_special_init \
    --save_model_dir <path to save logs and models>


            OR

# changes according to dataset.
# the above path names need to be changed in the script files
cd fca4nn
bash scripts/foca/inet100.sh 
```

### Baselines
Codes for baselines are taken from their respective repositories. Each of those folders has its own curated `run.sh` files which we execute to get baseline results

- CBMs: "Concept Bottleneck Models"
```
@inproceedings{cbms,
  title={Concept bottleneck models},
  author={Koh, Pang Wei and Nguyen, Thao and Tang, Yew Siang and Mussmann, Stephen and Pierson, Emma and Kim, Been and Liang, Percy},
  booktitle={International conference on machine learning},
  pages={5338--5348},
  year={2020},
  organization={PMLR}
}
```

- Post-hoc CBMs: "Post-hoc Concept Bottleneck Models"
```
@inproceedings{
yuksekgonul2023posthoc,
title={Post-hoc Concept Bottleneck Models},
author={Mert Yuksekgonul and Maggie Wang and James Zou},
booktitle={The Eleventh International Conference on Learning Representations },
year={2023},
url={https://openreview.net/forum?id=nA5AZ8CEyow}
}
```
- Label-free CBMs: "Label-free Concept Bottleneck Models"
```
@inproceedings{oikarinenlabel,
  title={Label-free Concept Bottleneck Models},
  author={Oikarinen, Tuomas and Das, Subhro and Nguyen, Lam M and Weng, Tsui-Wei},
  booktitle={International Conference on Learning Representations},
  year={2023}
}
```
- Concept Embedding Models: "Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off"
```
@article{EspinosaZarlenga2022cem,
  title={Concept Embedding Models: Beyond the Accuracy-Explainability Trade-Off},
  author={
    Espinosa Zarlenga, Mateo and Barbiero, Pietro and Ciravegna, Gabriele and
    Marra, Giuseppe and Giannini, Francesco and Diligenti, Michelangelo and
    Shams, Zohreh and Precioso, Frederic and Melacci, Stefano and
    Weller, Adrian and Lio, Pietro and Jamnik, Mateja
  },
  journal={Advances in Neural Information Processing Systems},
  volume={35},
  year={2022}
}
```

- Language in a Bottle: "Language in a Bottle: Language Model Guided Concept Bottlenecks for Interpretable Image Classification"
```
@inproceedings{yang2023language,
  title={Language in a bottle: Language model guided concept bottlenecks for interpretable image classification},
  author={Yang, Yue and Panagopoulou, Artemis and Zhou, Shenghao and Jin, Daniel and Callison-Burch, Chris and Yatskar, Mark},
  booktitle={Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition},
  pages={19187--19197},
  year={2023}
}
```
- Stochastic CBMs: "Stochastic Concept Bottleneck Models"
```
@article{vandenhirtz2024stochastic,
  title={Stochastic concept bottleneck models},
  author={Vandenhirtz, Moritz and Laguna, Sonia and Marcinkevi{\v{c}}s, Ri{\v{c}}ards and Vogt, Julia},
  journal={Advances in Neural Information Processing Systems},
  volume={37},
  pages={51787--51810},
  year={2024}
}
```

- Probabilistic CBMs: "Probabilistic Concept Bottleneck Models"
```
@article{Kim2023ProbabilisticCB,
  title={Probabilistic Concept Bottleneck Models},
  author={Eunji Kim and Dahuin Jung and Sangha Park and Siwon Kim and Sung-Hoon Yoon},
  journal={ArXiv},
  year={2023},
  volume={abs/2306.01574},
  url={https://api.semanticscholar.org/CorpusID:259063823}
}
```

- Coarse-to-Fine CBMs: "Coarse-to-Fine Concept Bottleneck Models"
```
@inproceedings{
panousis2024coarsetofine,
title={Coarse-to-Fine Concept Bottleneck Models},
author={Konstantinos P. Panousis and Dino Ienco and Diego Marcos},
booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems},
year={2024},
url={https://openreview.net/forum?id=RMdnTnffou}
}
``` 

- Hybrid CBMs: "Hybrid Concept Bottleneck Models"
```
@inproceedings{liu2025hybrid,
  title={Hybrid Concept Bottleneck Models},
  author={Liu, Yang and Zhang, Tianwei and Gu, Shi},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={20179--20189},
  year={2025}
}
```