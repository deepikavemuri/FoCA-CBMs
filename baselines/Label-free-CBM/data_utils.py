import os
import torch
from torchvision import datasets, transforms, models
from torch.utils.data import Dataset
import pandas as pd
from PIL import Image
import pickle
import random

import clip
from pytorchcv.model_provider import get_model as ptcv_get_model

DATASET_ROOTS = {
    "imagenet100_train": "./../DATA/inet100/train/",
    "imagenet100_val": "./../DATA/inet100/val/",
    "imagenet100_test": "./../DATA/inet100/test_set/",
}

LABEL_FILES = {
    "imagenet100": "./../DATA/classes/inet100_classes.txt",
    "awa2": "./../DATA/classes/awa2_classes.txt",
}


class AwA2Dataset(Dataset):
    def __init__(self, path, split, transform=None):
        super().__init__()
        self.path = path
        self.split = split
        self.transform = transform

        self.data = pd.read_csv(os.path.join(path, split + ".csv"))
        self.targets = self.data["img_index"].values

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        sample = self.data.iloc[idx]
        img_path = os.path.join(self.path, sample["img_name"].split("//")[-1])
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = self.targets[idx]
        return img, label


class CUB200Dataset(Dataset):
    def __init__(self, path, split, transform=None):
        super().__init__()
        self.path = path
        self.split = split
        self.transform = transform
        self.data_list = pickle.load(
            open(os.path.join(path, "CUB112", f"{split}.pkl"), "rb")
        )
        # random shuffle the data list
        self.data_list = random.sample(self.data_list, len(self.data_list))
        self.targets = []
        for sample in self.data_list:
            self.targets.append(int(sample["class_label"]))

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        sample = self.data_list[idx]
        img_path = os.path.join(
            self.path, sample["img_path"].split("/CUB_200_2011/")[-1]
        )
        img = Image.open(img_path).convert("RGB")
        if self.transform:
            img = self.transform(img)
        label = sample["class_label"]
        return img, label


def get_resnet_imagenet_preprocess():
    target_mean = [0.485, 0.456, 0.406]
    target_std = [0.229, 0.224, 0.225]
    preprocess = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean=target_mean, std=target_std),
        ]
    )
    return preprocess


def get_data(dataset_name, preprocess=None, data_root=None):
    if dataset_name == "awa2_train":
        data = AwA2Dataset(data_root, "train", transform=preprocess)
    elif dataset_name == "awa2_val":
        data = AwA2Dataset(data_root, "val", transform=preprocess)
    elif dataset_name == "awa2_test":
        data = AwA2Dataset(data_root, "test", transform=preprocess)
    elif dataset_name == "cub200_train":
        data = CUB200Dataset(data_root, "train", transform=preprocess)
    elif dataset_name == "cub200_val":
        data = CUB200Dataset(data_root, "val", transform=preprocess)
    elif dataset_name == "cub200_test":
        data = CUB200Dataset(data_root, "test", transform=preprocess)

    elif dataset_name in DATASET_ROOTS.keys():
        data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    else:
        raise ValueError("Unknown dataset name: {}".format(dataset_name))

    return data
    # if dataset_name == "cifar100_train":
    #     data = datasets.CIFAR100(
    #         root=os.path.expanduser("~/.cache"),
    #         download=True,
    #         train=True,
    #         transform=preprocess,
    #     )

    # elif dataset_name == "cifar100_val":
    #     data = datasets.CIFAR100(
    #         root=os.path.expanduser("~/.cache"),
    #         download=True,
    #         train=False,
    #         transform=preprocess,
    #     )

    # elif dataset_name == "cifar10_train":
    #     data = datasets.CIFAR10(
    #         root=os.path.expanduser("~/.cache"),
    #         download=True,
    #         train=True,
    #         transform=preprocess,
    #     )

    # elif dataset_name == "cifar10_val":
    #     data = datasets.CIFAR10(
    #         root=os.path.expanduser("~/.cache"),
    #         download=True,
    #         train=False,
    #         transform=preprocess,
    #     )

    # elif dataset_name == "places365_train":
    #     try:
    #         data = datasets.Places365(
    #             root=os.path.expanduser("~/.cache"),
    #             split="train-standard",
    #             small=True,
    #             download=True,
    #             transform=preprocess,
    #         )
    #     except RuntimeError:
    #         data = datasets.Places365(
    #             root=os.path.expanduser("~/.cache"),
    #             split="train-standard",
    #             small=True,
    #             download=False,
    #             transform=preprocess,
    #         )

    # elif dataset_name == "places365_val":
    #     try:
    #         data = datasets.Places365(
    #             root=os.path.expanduser("~/.cache"),
    #             split="val",
    #             small=True,
    #             download=True,
    #             transform=preprocess,
    #         )
    #     except RuntimeError:
    #         data = datasets.Places365(
    #             root=os.path.expanduser("~/.cache"),
    #             split="val",
    #             small=True,
    #             download=False,
    #             transform=preprocess,
    #         )

    # elif dataset_name in DATASET_ROOTS.keys():
    #     data = datasets.ImageFolder(DATASET_ROOTS[dataset_name], preprocess)

    # elif dataset_name == "imagenet_broden":
    #     data = torch.utils.data.ConcatDataset(
    #         [
    #             datasets.ImageFolder(DATASET_ROOTS["imagenet_val"], preprocess),
    #             datasets.ImageFolder(DATASET_ROOTS["broden"], preprocess),
    #         ]
    #     )
    # return data


def get_targets_only(dataset_name, data_root=None):
    pil_data = get_data(dataset_name, data_root=data_root)
    return pil_data.targets


def get_target_model(target_name, device):

    if target_name.startswith("clip_"):
        target_name = target_name[5:]
        model, preprocess = clip.load(target_name, device=device)
        target_model = lambda x: model.encode_image(x).float()

    elif target_name == "resnet18_places":
        target_model = models.resnet18(pretrained=False, num_classes=365).to(device)
        state_dict = torch.load("data/resnet18_places365.pth.tar")["state_dict"]
        new_state_dict = {}
        for key in state_dict:
            if key.startswith("module."):
                new_state_dict[key[7:]] = state_dict[key]
        target_model.load_state_dict(new_state_dict)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name == "resnet18_cub":
        target_model = ptcv_get_model("resnet18_cub", pretrained=True).to(device)
        target_model.eval()
        preprocess = get_resnet_imagenet_preprocess()

    elif target_name.endswith("_v2"):
        target_name = target_name[:-3]
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V2".format(target_name_cap))
        target_model = eval("models.{}(weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    else:
        target_name_cap = target_name.replace("resnet", "ResNet")
        weights = eval("models.{}_Weights.IMAGENET1K_V1".format(target_name_cap))
        target_model = eval("models.{}(weights=weights).to(device)".format(target_name))
        target_model.eval()
        preprocess = weights.transforms()

    return target_model, preprocess
