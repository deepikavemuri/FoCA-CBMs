import logging
import numpy as np
import os
import torch

from torchvision.datasets import CIFAR100
from pytorch_lightning import seed_everything
from torchvision.transforms import (
    CenterCrop,
    ColorJitter,
    Compose,
    Normalize,
    RandomHorizontalFlip,
    RandomResizedCrop,
    Resize,
    ToTensor,
)
from torch.utils.data import Dataset, random_split
from glob import glob
from PIL import Image

CIFAR100_CLASSES = [
    "apple",
    "aquarium_fish",
    "baby",
    "bear",
    "beaver",
    "bed",
    "bee",
    "beetle",
    "bicycle",
    "bottle",
    "bowl",
    "boy",
    "bridge",
    "bus",
    "butterfly",
    "camel",
    "can",
    "castle",
    "caterpillar",
    "cattle",
    "chair",
    "chimpanzee",
    "clock",
    "cloud",
    "cockroach",
    "couch",
    "crab",
    "crocodile",
    "cup",
    "dinosaur",
    "dolphin",
    "elephant",
    "flatfish",
    "forest",
    "fox",
    "girl",
    "hamster",
    "house",
    "kangaroo",
    "computer_keyboard",
    "lamp",
    "lawn_mower",
    "leopard",
    "lion",
    "lizard",
    "lobster",
    "man",
    "maple_tree",
    "motorcycle",
    "mountain",
    "mouse",
    "mushroom",
    "oak_tree",
    "orange",
    "orchid",
    "otter",
    "palm_tree",
    "pear",
    "pickup_truck",
    "pine_tree",
    "plain",
    "plate",
    "poppy",
    "porcupine",
    "possum",
    "rabbit",
    "raccoon",
    "ray",
    "road",
    "rocket",
    "rose",
    "sea",
    "seal",
    "shark",
    "shrew",
    "skunk",
    "skyscraper",
    "snail",
    "snake",
    "spider",
    "squirrel",
    "streetcar",
    "sunflower",
    "sweet_pepper",
    "table",
    "tank",
    "telephone",
    "television",
    "tiger",
    "tractor",
    "train",
    "trout",
    "tulip",
    "turtle",
    "wardrobe",
    "whale",
    "willow_tree",
    "wolf",
    "woman",
    "worm",
]

N_CONCEPTS = 700
N_CLASSES = len(CIFAR100_CLASSES)

CONCEPTS_PATH = (
    "/raid/ai24mtech12011/sayanta/fca4nn/DATA/concepts/cifar100_concept_matrix.npy"
)


class Cifar100ConceptDataset(Dataset):
    def __init__(
        self,
        data,
        split="train",
        transform=None,
    ):
        self.split = split
        self.attr_npy = np.load(CONCEPTS_PATH)
        self.data = data

        if transform is None:
            if split == "train":
                self.transforms = Compose(
                    [
                        RandomResizedCrop(224, interpolation=Image.BILINEAR),
                        RandomHorizontalFlip(),
                        ColorJitter(
                            brightness=0.4, contrast=0.4, saturation=0.4, hue=0.2
                        ),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transforms = Compose(
                    [
                        Resize(size=256, interpolation=Image.BILINEAR),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.transforms = transform

    def __len__(self):
        return len(self.data)

    def get_concept_count(self):
        return N_CONCEPTS

    def __getitem__(self, index):
        image, label = self.data[index]
        image = self.transforms(image)
        concept_vector = self.attr_npy[label]

        return image, label, torch.FloatTensor(concept_vector.astype(np.float32))


def generate_data(
    config,
    root_dir,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
):
    concept_group_map = None
    seed_everything(seed)

    full_data = CIFAR100(root=root_dir, train=True, download=True, transform=None)
    test_dataset = CIFAR100(root=root_dir, train=False, download=True, transform=None)

    train_size = int(0.9 * len(full_data))  # 45000
    val_size = len(full_data) - train_size  # 5000

    # Split the dataset into train and validation sets
    train_dataset, val_dataset = random_split(full_data, [train_size, val_size])

    # Load the dataset
    train_dataset = Cifar100ConceptDataset(
        data=train_dataset,
        split="train",
        transform=None,
    )
    val_dataset = Cifar100ConceptDataset(
        data=val_dataset,
        split="val",
        transform=None,
    )
    test_dataset = Cifar100ConceptDataset(
        data=test_dataset,
        split="test",
        transform=None,
    )
    # Create data loaders
    train_dl = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=config["batch_size"],
        shuffle=True,
        num_workers=config["num_workers"],
    )
    val_dl = torch.utils.data.DataLoader(
        val_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )
    test_dl = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=config["batch_size"],
        shuffle=False,
        num_workers=config["num_workers"],
    )

    # Create imbalance
    if config.get("weight_loss", False):
        attribute_count = np.zeros((N_CONCEPTS,))
        samples_seen = 0
        for i in range(len(train_dataset)):
            _, _, attribute = train_dataset[i]
            attribute_count += attribute.numpy()
            samples_seen += 1
        imbalance = samples_seen / attribute_count - 1
    else:
        imbalance = None

    if not output_dataset_vars:
        return train_dl, val_dl, test_dl, imbalance
    return (
        train_dl,
        val_dl,
        test_dl,
        imbalance,
        (N_CONCEPTS, N_CLASSES, concept_group_map),
    )
