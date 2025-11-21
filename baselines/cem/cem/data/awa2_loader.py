import logging
import numpy as np
import os
import torch
import torchvision
import pandas as pd

from pathlib import Path
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
from torch.utils.data import Dataset
from PIL import Image

N_CLASSES = 50
N_CONCEPTS = 85


class AwA2Dataset(Dataset):
    def __init__(
        self,
        data_dir="/data1/ai22resch11001/projects/data/AWA2/Animals_with_Attributes2",
        transform=None,
        split="train",
        apply_corruption=False,
    ):
        predicate_binary_mat = np.array(
            np.genfromtxt(data_dir + "/predicate-matrix-binary.txt", dtype="int")
        )
        self.predicate_binary_mat = predicate_binary_mat
        self.apply_corruption = apply_corruption
        self.data_dir = data_dir
        self.split = split

        self.transform = transform
        if transform is None:
            if split == "train":
                self.transform = Compose(
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
                self.transform = Compose(
                    [
                        Resize(256, interpolation=Image.BILINEAR),
                        CenterCrop(224),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )

        class_to_index = dict()
        # Build dictionary of indices to classes
        with open(data_dir + "/classes.txt") as f:
            index = 0
            for line in f:
                class_name = line.split("\t")[1].strip()
                class_to_index[class_name] = index
                index += 1
        self.class_to_index = class_to_index

        df = pd.read_csv(os.path.join(data_dir, "{}.csv".format(split)))
        self.img_names = df["path"].tolist()
        self.img_index = df["label"].tolist()
        self.num_classes = len(class_to_index.keys())
        self.num_attrs = len(predicate_binary_mat[0])

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index].split("//")[-1])
        im = Image.open(img_path).convert("RGB")
        if self.transform:
            im = self.transform(im)

        im_index = self.img_index[index]
        im_predicate = self.predicate_binary_mat[im_index, :]
        return im, im_index, torch.FloatTensor(im_predicate)

    def __len__(self):
        return len(self.img_names)


def generate_data(
    config,
    root_dir,
    seed=42,
    output_dataset_vars=False,
    rerun=False,
):
    concept_group_map = None
    seed_everything(seed)

    # Load the dataset
    train_dataset = AwA2Dataset(
        data_dir=root_dir,
        split="train",
        transform=None,
        apply_corruption=False,
    )
    val_dataset = AwA2Dataset(
        data_dir=root_dir,
        split="val",
        transform=None,
        apply_corruption=False,
    )
    test_dataset = AwA2Dataset(
        data_dir=root_dir,
        split="test",
        transform=None,
        apply_corruption=False,
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
