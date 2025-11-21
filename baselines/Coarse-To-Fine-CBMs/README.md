# Coarse-to-Fine Concept Bottleneck Models

This is the official code implementation for the paper titled "Coarse to Fine Concept Bottleneck Models" published at NeurIPS 2024. We propose a novel framework towards Interpretable Deep Networks using multi-modal models and a novel multilevel construction for capturing low-level details

# Setup 

The file structure to make sure that everything works as intended is the following:

```
── CF-CBMs
│   ├── clip/
│   ├── data/
│   ├── saved_models/
│   ├── README.md
│   ├── main.py
│   ├── networks.py
│   ├── data_utils.py
│   └── utils.py

```
where the `saved_models` folder will be created automatically if it doesn't already exist when running the main script.

 1. Create a venv/conda environment containing all the necessary packages. This can be achieved using the provided .yml file. 
 2. Specifically, run `conda env create -f clip_env.yml`.

When considering CUB and ImageNet, you should set it up with the standard format and provide the correct path in the `data_utils.py` file in the corresponding ImageNet entry.

# Training and Inference 

## CLIP Embeddings 
As described in the main text, the models are trained using the embeddings
arising from a pretrained clip model. To facilitate training and inference speeds, 
we first embed the dataset in the CLIP embedding space and 
use then load the embedded vectors as the dataset to be used. 


For saving the text embeddings of a different dataset, one should use the
following command:

`python main.py --dataset cub --compute_similarities --batch_size 128
`

where you replace the `dataset` argument with the name of your dataset.
For this to work, you need to implement data loding function in the `data_utils.py` file.

This assumes that you use the default concept sets, i.e., cub. To use a different concept set
(even your own), specify the name in the `concept_name` argument, and make sure that
your concept file is in the correct folder, i.e., `data/concept_sets/your_concept_set.txt`.


## Training 
Assuming that you have the embeddings already computed, you can train the linear layers from scratch on 
a given dataset. To train the network on cifar100 with the cifar100 concept set for 300 epochs, the command is:

`python main.py --dataset cub --epochs 300 --batch_size 2048`

