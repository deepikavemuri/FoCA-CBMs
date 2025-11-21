import os
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Resize, RandomResizedCrop, CenterCrop, Normalize, ToTensor, RandomHorizontalFlip, ColorJitter
from PIL import Image
import numpy as np
import pandas as pd

def get_AwA2_CBM_dataloader(datapath):
    data_dir = os.path.join(datapath, "Animals_with_Attributes2")
    image_datasets = {
        "train": AnimalLoader(
            data_dir=data_dir,
            split='train',
        ),
        "val": AnimalLoader(
            data_dir=data_dir,
            split='val',
        ),
        "test": AnimalLoader(
            data_dir=data_dir,
            split='test',
        ),
    }

    return image_datasets["train"], image_datasets["val"], image_datasets["test"]


class AnimalLoader(Dataset):
    def __init__(
        self,
        data_dir="/DATA/AWA2/Animals_with_Attributes2",
        transform=None,
        split="train",
        apply_corruption=False,
        fraction="full",
        few_shot_train=False
    ):
        self.few_shot_train = few_shot_train
        predicate_binary_mat = np.array(
            np.genfromtxt(os.path.join(data_dir, "predicate-matrix-binary.txt"), dtype="int")
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
        prefix = ""
        if self.few_shot_train:
            prefix = "train"
        # Build dictionary of indices to classes
        with open(os.path.join(self.data_dir, prefix + "classes.txt")) as f:
            index = 0
            for line in f:
                class_name = line.split(" ")[1].strip()
                class_to_index[class_name] = index
                index += 1
        self.class_to_index = class_to_index

        if self.few_shot_train:
            df = pd.read_csv(
                os.path.join(self.data_dir, "{}_fewshot.csv".format(split))
            )  # header=None, names=["id", "path", "label"])
        else:
            df = pd.read_csv(os.path.join(data_dir, "{}_full.csv".format(split)))

        if split == "train":
            if fraction == "half":
                df = pd.read_csv(os.path.join(data_dir, "{}_half.csv".format(split)))
            elif fraction == "quarter":
                df = pd.read_csv(os.path.join(data_dir, "{}_quarter.csv".format(split)))

        self.img_names = df["img_name"].tolist()
        self.img_index = df["class_id"].tolist()
        self.num_classes = len(class_to_index.keys())
        self.num_attrs = len(predicate_binary_mat[0])

    def __getitem__(self, index):
        img_path = os.path.join(self.data_dir, self.img_names[index].split("//")[-1])
        im = Image.open(img_path).convert("RGB")
        if self.transform:
            im = self.transform(im)

        im_index = self.img_index[index]
        im_predicate = self.predicate_binary_mat[im_index, :]
        return {
                    "img_code": img_path,
                    "labels": im_index,
                    "features": im,
                    "concepts": im_predicate,
                }

    def __len__(self):
        return len(self.img_names)