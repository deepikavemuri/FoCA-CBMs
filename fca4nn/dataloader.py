import pandas as pd
import torchvision
from torchvision.transforms import *
from torch.utils.data import Dataset
import os
import numpy as np
from PIL import Image
import ast
import sys
from glob import glob
import json
from collections import OrderedDict
from processing import utils
from collections import defaultdict
import random

from processing.imagenet100_classes import (
    IMAGENET100_CLASS2ID,
    IMAGENET100_CLASSES,
)

from processing.awa2_class_map import AWA2_CLASSES, AWA2_SEEN_CLASSES

DATA_ROOT = "."
NUM_CLASSES = 100


class ImagenetMultiClassLevelsDataset(Dataset):
    def __init__(
        self,
        lattice_levels,
        data_root=DATA_ROOT + "inet100/",
        json_file=DATA_ROOT + "inet100/" + "inet100_classlevel_reduced.json",
        lattice_path=DATA_ROOT + "inet100/" + "inet100_lattice.json",
        split="train",
        transforms=None,
        perlevel_intents=None,
        perlevel_fcs=None,
        fraction="full",
    ):
        self.num_classes = len(IMAGENET100_CLASS2ID.keys())
        self.lattice_levels = lattice_levels
        self.class_concept_dict = dict(
            json.load(open(json_file, "r"), object_pairs_hook=OrderedDict)
        )
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.class_label_map = {
            i: k for i, k in zip(np.arange(0, self.num_classes), self.class_list)
        }
        self.label_class_map = {
            k: i for i, k in zip(np.arange(0, self.num_classes), self.class_list)
        }
        self.concept_label_dict = dict()
        if perlevel_intents is None or perlevel_fcs is None:
            self.perlevel_intents, self.perlevel_fcs = utils.get_info_from_lattice(
                lattice_path, lattice_levels
            )
        else:
            self.perlevel_intents, self.perlevel_fcs = perlevel_intents, perlevel_fcs

        for intent in self.perlevel_intents:
            intent.sort()

        self.dir_idx = {k: v for v, k in enumerate(IMAGENET100_CLASSES.keys())}
        class_dirs = IMAGENET100_CLASSES.keys()

        if transforms is None:
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
            self.transforms = transforms

        self.data = []
        for d in class_dirs:
            label = self.dir_idx[d]
            if split == "train":
                images = glob(os.path.join(data_root, "train", d, "*.JPEG"))
                images.sort()
                if fraction == "half":
                    images = images[: len(images) // 2]
                elif fraction == "quarter":
                    images = images[: len(images) // 4]
                random.shuffle(images)
            elif split == "val":
                images = glob(os.path.join(data_root, "val", d, "*.JPEG"))
            elif split == "test":
                images = glob(os.path.join(data_root, "test_set", d, "*.JPEG"))
            else:
                raise ValueError("Invalid split: {}".format(split))
            self.data.extend(list(zip(images, [label] * len(images))))

        np.random.shuffle(self.data)
        self.attr_to_attrid_mapping = defaultdict(dict)
        for i in range(len(self.lattice_levels)):
            self.attr_to_attrid_mapping[self.lattice_levels[i]] = {
                attr: j for j, attr in enumerate(self.perlevel_intents[i])
            }
        self.fc_class_lists = [
            self.__get_class_list__(fcs) for fcs in self.perlevel_fcs
        ]

        self.get_classes_per_level(self.num_classes)

        class_list = list(range(len(self.class_concept_dict.keys())))
        for cls in class_list:
            attrs = self.class_concept_dict[self.class_label_map[cls]]
            for attr in attrs:
                if attr not in self.concept_label_dict.keys():
                    self.concept_label_dict[attr] = len(self.concept_label_dict.keys())

    def __len__(self):
        return len(self.data)

    def __get_concept_count__(self, level):
        return len(self.attr_to_attrid_mapping[level])

    def __get_class_list__(self, fc_list):
        class_list = []
        for fc in fc_list:
            extent = fc.extent
            class_list.append([self.dir_idx[clss] for clss in extent])
        return class_list

    def get_classes_per_level(self, num_classes):
        labels = list(range(num_classes))
        self.classes_present_perlevel_perlabel = [
            [np.zeros(num_classes) for _ in range(len(self.lattice_levels))]
            for _ in labels
        ]

        for label in labels:
            for i, fc_list in enumerate(self.fc_class_lists):
                for j, lst in enumerate(fc_list):
                    if label in lst:
                        for k in lst:
                            self.classes_present_perlevel_perlabel[label][i][k] = 1

    def __getitem__(self, index):
        image_name, label = self.data[index]
        image = Image.open(image_name).convert("RGB")
        image = self.transforms(image)

        attrs = self.class_concept_dict[self.class_label_map[label]]
        attr_values_perlevel = [[] for _ in range(len(self.lattice_levels))]

        for attr in attrs:
            for i, level in enumerate(self.lattice_levels):
                if attr in self.attr_to_attrid_mapping[level].keys():
                    attr_values_perlevel[i].append(
                        self.attr_to_attrid_mapping[level][attr]
                    )

        concept_vector_perlevel = [
            np.zeros(self.__get_concept_count__(level)) for level in self.lattice_levels
        ]

        for i in range(len(self.lattice_levels)):
            concept_vector_perlevel[i][attr_values_perlevel[i]] = 1

        return (
            image_name,
            image,
            label,
            concept_vector_perlevel,
            self.classes_present_perlevel_perlabel[label],
        )


class Imagenet100ConceptDataset(Dataset):
    def __init__(
        self,
        data_root=DATA_ROOT + "inet100/",
        json_file=DATA_ROOT + "inet100/" + "imagenet_reduced_attr2.json",
        split="train",
        transforms=None,
        fraction="full",
    ):
        self.num_classes = len(IMAGENET100_CLASS2ID.keys())
        self.class_concept_dict = dict(
            json.load(open(json_file, "r"), object_pairs_hook=OrderedDict)
        )
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.num_attrs = len(self.concept_list)
        self.class_label_map = {
            i: k for i, k in zip(np.arange(0, self.num_classes), self.class_list)
        }
        self.label_class_map = {
            k: i for i, k in zip(np.arange(0, self.num_classes), self.class_list)
        }
        self.concept_label_dict = dict()

        self.dir_idx = {k: v for v, k in enumerate(IMAGENET100_CLASSES.keys())}
        class_dirs = IMAGENET100_CLASSES.keys()

        if transforms is None:
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
            self.transforms = transforms

        self.data = []
        for d in class_dirs:
            label = self.dir_idx[d]
            # label = d
            if split == "train":
                images = glob(os.path.join(data_root, "train", d, "*.JPEG"))
                images.sort()
                if fraction == "half":
                    images = images[: len(images) // 2]
                elif fraction == "quarter":
                    images = images[: len(images) // 4]
                random.shuffle(images)
            elif split == "val":
                images = glob(os.path.join(data_root, "val", d, "*.JPEG"))
            elif split == "test":
                images = glob(os.path.join(data_root, "test_set", d, "*.JPEG"))
            else:
                raise ValueError("Invalid split: {}".format(split))
            self.data.extend(list(zip(images, [label] * len(images))))

        np.random.shuffle(self.data)
        class_list = list(range(len(self.class_concept_dict.keys())))

        for cls in class_list:
            attrs = self.class_concept_dict[self.class_label_map[cls]]
            for attr in attrs:
                if attr not in self.concept_label_dict.keys():
                    self.concept_label_dict[attr] = len(self.concept_label_dict.keys())

    def __len__(self):
        return len(self.data)

    def get_concept_count(self):
        return len(self.concept_label_dict)

    def __getitem__(self, index):
        image, label = self.data[index]
        if type(image) == str:
            image = Image.open(image).convert("RGB")
            image = self.transforms(image)

        attrs = self.class_concept_dict[self.class_label_map[label]]
        attr_values = []

        for attr in attrs:
            attr_values.append(self.concept_label_dict[attr])

        concept_vector = np.zeros(self.get_concept_count())
        concept_vector[attr_values] = 1

        return image, label, concept_vector.astype(np.float32)


class Awa2ClassLevelDataset(Dataset):
    def __init__(
        self,
        lattice_levels,
        data_root=DATA_ROOT + "AWA2/Animals_with_Attributes2/",
        json_file=DATA_ROOT + "AWA2/Animals_with_Attributes2/" + "awa2_concepts.json",
        lattice_path=DATA_ROOT + "AWA2/Animals_with_Attributes2/" + "awa2_lattice.json",
        transform=None,
        split="train",
        perlevel_intents=None,
        perlevel_fcs=None,
        fraction="full",
        few_shot_train=False,
    ):
        self.data_dir = data_root
        self.few_shot_train = few_shot_train
        self.class_concept_dict = dict(
            json.load(open(json_file, "r"), object_pairs_hook=OrderedDict)
        )
        self.class_list = self.class_concept_dict.keys()
        self.class_label_map = {
            i: k for i, k in zip(np.arange(0, len(self.class_list)), self.class_list)
        }
        predicate_binary_mat = np.array(
            np.genfromtxt(
                os.path.join(self.data_dir, "predicate-matrix-binary.txt"), dtype="int"
            )
        )
        self.predicate_binary_mat = predicate_binary_mat
        self.transform = transform
        # self.split = "train" if split == "train" else "test"
        self.lattice_levels = lattice_levels

        if perlevel_intents is None or perlevel_fcs is None:
            self.perlevel_intents, self.perlevel_fcs = utils.get_info_from_lattice(
                lattice_path, lattice_levels
            )
        else:
            self.perlevel_intents = perlevel_intents
            self.perlevel_fcs = perlevel_fcs

        for intent in self.perlevel_intents:
            intent.sort()

        # print("fc classes: ", self.perlevel_fcs[0])

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
        prefix = ""
        if self.few_shot_train:
            prefix = "train"
        with open(os.path.join(self.data_dir, prefix + "classes.txt")) as f:
            index = 0
            for line in f:
                class_name = line.split("\t")[
                    1
                ].strip()  # " " for the few shot splits, "\t" for the full split
                class_to_index[class_name] = index
                index += 1
        self.class_to_index = class_to_index
        self.num_classes = len(class_to_index.keys())

        # if self.split == "train":
        #     df = pd.read_csv(os.path.join(self.data_dir , "{}_quarter.csv".format(split)))
        # else:
        if self.few_shot_train:
            df = pd.read_csv(
                os.path.join(self.data_dir, "{}_fewshot.csv".format(split))
            )  # header=None, names=["id", "path", "label"])
        else:
            df = pd.read_csv(os.path.join(self.data_dir, "{}_full.csv".format(split)))

        if split == "train":
            if fraction == "half":
                df = pd.read_csv(
                    os.path.join(self.data_dir, "{}_half.csv".format(split))
                )
            elif fraction == "quarter":
                df = pd.read_csv(
                    os.path.join(self.data_dir, "{}_quarter.csv".format(split))
                )
        # Extract class name from the path (e.g., 'tiger' from '/JPEGImages/tiger/...')
        # df["class"] = df["path"].str.extract(r"/JPEGImages/([^/]+)/")
        # sampled_df = df.groupby("class", group_keys=False).apply(lambda x: x.sample(frac=0.25, random_state=42)).sample(frac=1, random_state=42)
        # sampled_df["id"] = sampled_df["id"].astype(int)
        # sampled_df = sampled_df.drop(columns=["class"])
        # sampled_df.to_csv("/DATA/AWA2/Animals_with_Attributes2/train_quarter.csv", index=False)
        # exit(0)

        self.img_names = df["img_name"].tolist()
        self.class_ids = df["class_id"].tolist()

        self.attr_label_mapping = defaultdict(dict)  # change to attr_to_attrid_mapping
        for i in range(len(self.lattice_levels)):
            self.attr_label_mapping[self.lattice_levels[i]] = {
                attr: j for j, attr in enumerate(self.perlevel_intents[i])
            }
        self.fc_class_lists = [
            self.__get_class_list__(fcs) for fcs in self.perlevel_fcs
        ]
        # print(self.fc_class_lists)
        # print(len(self.fc_class_lists))
        # exit(0)
        self.get_classes_per_level(self.num_classes)

    def create_csv(self):
        import csv

        seen_classes_file = DATA_ROOT + "AWA2/Animals_with_Attributes2/trainclasses.txt"
        with open(seen_classes_file, "r") as f:
            seen_classes = [line.strip() for line in f.readlines()]
        class_to_id = {cls_name: i for i, cls_name in enumerate(seen_classes)}
        image_root = DATA_ROOT + "AWA2/Animals_with_Attributes2/JPEGImages/"

        entries = []
        for cls in seen_classes:
            class_id = class_to_id[cls]
            cls_path = os.path.join(image_root, cls)
            for fname in os.listdir(cls_path):
                if fname.endswith(".jpg") or fname.endswith(".JPEG"):
                    img_path = os.path.join(image_root, cls, fname)
                    entries.append([img_path, cls, class_id])

        random.seed(42)
        random.shuffle(entries)
        val_ratio = 0.2
        split_idx = int(len(entries) * (1 - val_ratio))
        train_imgs = entries[:split_idx]
        val_imgs = entries[split_idx:]

        with open(
            DATA_ROOT + "AWA2/Animals_with_Attributes2/train_fewshot.csv",
            "w",
            newline="",
        ) as f:
            writer = csv.writer(f)
            writer.writerow(["image_path", "class_name", "class_id"])
            writer.writerows(entries)

    def __len__(self):
        return len(self.img_names)

    def __get_concept_count__(self, level):
        return len(self.attr_label_mapping[level])

    def __get_class_list__(self, fc_list):
        def get_id(clss):
            if self.few_shot_train:
                if clss in AWA2_SEEN_CLASSES.keys():
                    return AWA2_SEEN_CLASSES[clss]
            else:
                if clss in AWA2_CLASSES.keys():
                    return AWA2_CLASSES[clss]
            return -1

        class_list = []
        for fc in fc_list:
            extent = fc.extent
            class_list.append([get_id(clss) for clss in extent])
        return class_list

    def get_classes_per_level(self, num_classes):
        labels = list(range(num_classes))
        self.classes_present_perlevel_perlabel = [
            [np.zeros(num_classes) for i in range(len(self.lattice_levels))]
            for label in labels
        ]

        for label in labels:
            for i, fc_list in enumerate(self.fc_class_lists):
                for j, lst in enumerate(fc_list):
                    if label in lst:
                        for k in lst:
                            if k != -1:
                                self.classes_present_perlevel_perlabel[label][i][k] = 1

    def __getitem__(self, index):
        im = Image.open(
            os.path.join(self.data_dir, self.img_names[index].split("//")[-1])
        )
        if im.getbands()[0] == "L":
            im = im.convert("RGB")
        if self.transform:
            im = self.transform(im)

        im_index = self.class_ids[index]
        attrs = self.class_concept_dict[self.class_label_map[im_index]]
        # print("attrs: ", attrs)

        attr_values_perlevel = [[] for i in range(len(self.lattice_levels))]

        for attr in attrs:
            for i, level in enumerate(self.lattice_levels):
                if attr in self.attr_label_mapping[level].keys():
                    attr_values_perlevel[i].append(self.attr_label_mapping[level][attr])

        concept_vector_perlevel = [
            np.zeros(self.__get_concept_count__(level)) for level in self.lattice_levels
        ]

        for i in range(len(self.lattice_levels)):
            concept_vector_perlevel[i][attr_values_perlevel[i]] = 1

        return (
            self.img_names[index].split("//")[-1],
            im,
            im_index,
            concept_vector_perlevel,
            self.classes_present_perlevel_perlabel[im_index],
        )


class AnimalLoader(Dataset):
    def __init__(
        self,
        data_dir=DATA_ROOT + "AWA2/Animals_with_Attributes2",
        transform=None,
        split="train",
        apply_corruption=False,
        fraction="full",
        few_shot_train=False,
    ):
        self.few_shot_train = few_shot_train
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
                self.transform = transforms.Compose(
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
                self.transform = transforms.Compose(
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
        return im, im_index, im_predicate

    def __len__(self):
        return len(self.img_names)


class Cifar100ClassLevelDataset(Dataset):
    def __init__(
        self,
        lattice_levels,
        data_root=DATA_ROOT + "cifar100",
        json_file=DATA_ROOT + "cifar100/cifar100_concepts_filtered_700.json",
        lattice_path=DATA_ROOT + "lattices/cifar100_context_filtered_700.pkl",
        split="train",
        transforms=None,
        perlevel_intents=None,
        perlevel_fcs=None,
    ):
        self.lattice_levels = lattice_levels
        self.class_concept_dict = json.load(
            open(json_file, "r"), object_pairs_hook=OrderedDict
        )
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.num_classes = len(self.class_list)
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.num_attrs = len(self.concept_list)
        self.label_class_map = {
            i: k for i, k in zip(np.arange(0, 100), self.class_list)
        }
        self.class_label_map = {
            k: i for i, k in zip(np.arange(0, 100), self.class_list)
        }
        self.concept_label_dict = {}

        if perlevel_intents is None or perlevel_fcs is None:
            self.perlevel_intents, self.perlevel_fcs = utils.get_info_from_lattice(
                lattice_path, lattice_levels
            )
        else:
            self.perlevel_intents, self.perlevel_fcs = perlevel_intents, perlevel_fcs

        for intent in self.perlevel_intents:
            intent.sort()

        #  train - 308, random crop - 299, test - centre crop
        if transforms is None:
            if split == "train":
                self.transforms = Compose(
                    [
                        Resize((308, 308)),
                        RandomCrop(299),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transforms = Compose(
                    [
                        Resize((308, 308)),
                        CenterCrop(299),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.transforms = transforms

        self.data = torchvision.datasets.CIFAR100(
            root=data_root,
            train=(split == "train"),
            download=False,
            transform=self.transforms,
        )

        self.attr_to_attrid_mapping = defaultdict(dict)
        for i in range(len(self.lattice_levels)):
            self.attr_to_attrid_mapping[self.lattice_levels[i]] = {
                attr: j for j, attr in enumerate(self.perlevel_intents[i])
            }
        self.fc_class_lists = [
            self.__get_class_list__(fcs) for fcs in self.perlevel_fcs
        ]

        self.get_classes_per_level(self.num_classes)

    def __len__(self):
        return len(self.data)

    def __get_concept_count__(self, level):
        return len(self.attr_to_attrid_mapping[level])

    def __get_class_list__(self, fc_list):
        class_list = []
        for fc in fc_list:
            extent = fc.extent
            class_list.append([self.class_label_map[clss] for clss in extent])
        return class_list

    def get_classes_per_level(self, num_classes):
        labels = list(range(num_classes))
        self.classes_present_perlevel_perlabel = [
            [np.zeros(num_classes) for _ in range(len(self.lattice_levels))]
            for _ in labels
        ]

        for label in labels:
            for i, fc_list in enumerate(self.fc_class_lists):
                for j, lst in enumerate(fc_list):
                    if label in lst:
                        for k in lst:
                            self.classes_present_perlevel_perlabel[label][i][k] = 1

    def __getitem__(self, index):
        image, label = self.data[index]
        attrs = self.class_concept_dict[self.label_class_map[label]]
        attr_values_perlevel = [[] for _ in range(len(self.lattice_levels))]

        for attr in attrs:
            for i, level in enumerate(self.lattice_levels):
                if attr in self.attr_to_attrid_mapping[level].keys():
                    attr_values_perlevel[i].append(
                        self.attr_to_attrid_mapping[level][attr]
                    )
        concept_vector_perlevel = [
            np.zeros(self.__get_concept_count__(level)) for level in self.lattice_levels
        ]

        for i in range(len(self.lattice_levels)):
            concept_vector_perlevel[i][attr_values_perlevel[i]] = 1

        return (
            image,  # This is image name in the other loaders. Just doing this to keep consistency
            image,
            label,
            concept_vector_perlevel,
            self.classes_present_perlevel_perlabel[label],
        )


class Cifar100Loader(Dataset):
    def __init__(
        self,
        data_dir=DATA_ROOT + "cifar100",
        json_file=DATA_ROOT + "cifar100/cifar100_concepts_filtered_700.json",
        split="train",
        transforms=None,
    ):
        self.class_concept_dict = json.load(
            open(json_file, "r"), object_pairs_hook=OrderedDict
        )
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.num_classes = len(self.class_list)
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.num_attrs = len(self.concept_list)
        self.class_label_map = {
            i: k for i, k in zip(np.arange(0, 100), self.class_list)
        }
        self.concept_label_dict = {}

        #  train - 308, random crop - 299, test - centre crop
        if transforms is None:
            if split == "train":
                self.transforms = Compose(
                    [
                        Resize((308, 308)),
                        RandomCrop(299),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
            else:
                self.transforms = Compose(
                    [
                        Resize((308, 308)),
                        CenterCrop(299),
                        ToTensor(),
                        Normalize(
                            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
                        ),
                    ]
                )
        else:
            self.transforms = transforms

        self.data = torchvision.datasets.CIFAR100(
            root=data_dir,
            train=(split == "train"),
            download=False,
            transform=self.transforms,
        )

        self.concept_vectors = []
        for cls in self.class_concept_dict.keys():
            attrs = self.class_concept_dict[cls]
            attr_values = []
            for attr in attrs:
                if attr not in self.concept_label_dict.keys():
                    self.concept_label_dict[attr] = len(self.concept_label_dict.keys())
                    attr_values.append(self.concept_label_dict[attr])
            concept_vector = np.zeros(self.get_concept_count())
            concept_vector[attr_values] = 1
            self.concept_vectors.append(concept_vector)

    def __len__(self):
        return len(self.data)

    def get_concept_count(self):
        return len(self.concept_list)

    def __getitem__(self, index):
        image, label = self.data[index]
        attrs = self.concept_vectors[label]

        return image, label, attrs
