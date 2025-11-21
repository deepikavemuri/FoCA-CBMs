"""
CIFAR-100 dataset loader with concept labels. Relies on create_dataset_cifar.py to have generated concept labels.

This module provides a custom DataLoader for the CIFAR-100 dataset, including concept labels for training, validation, and testing.
The dataset is preprocessed with transformations.

Classes:
    CIFAR100_CBM_dataloader: Custom DataLoader for CIFAR-100 with concept labels.

Functions:
    get_CIFAR100_CBM_dataloader: Returns DataLoaders for training, validation, and testing splits.
"""

from torchvision import datasets
import numpy as np
from torch.utils.data import Dataset
import json
from collections import OrderedDict
from torchvision.transforms import Compose, Resize, RandomCrop, CenterCrop, Normalize, ToTensor
import os

def get_CIFAR100_CBM_dataloader(datapath):
    data_dir = os.path.join(datapath, "cifar100")
    json_file = os.path.join(datapath, "concepts", "cifar100_concepts.json")
    image_datasets = {
        "train": Cifar100Loader(
            data_dir=data_dir,
            json_file=json_file,
            split='train',
        ),
        "val": Cifar100Loader(
            data_dir=data_dir,
            json_file=json_file,
            split='val',
        ),
        "test": Cifar100Loader(
            data_dir=data_dir,
            json_file=json_file,
            split='test',
        ),
    }

    return image_datasets["train"], image_datasets["val"], image_datasets["test"]


# class CIFAR100_CBM_dataloader(datasets.CIFAR100):

#     def __init__(self, *args, **kwargs):
#         super(CIFAR100_CBM_dataloader, self).__init__(*args, **kwargs)

#         if kwargs["train"]:
#             self.transform = transforms.Compose(
#                 [
#                     transforms.ColorJitter(brightness=32 / 255, saturation=(0.5, 1.5)),
#                     transforms.Resize(size=(224, 224)),
#                     transforms.RandomHorizontalFlip(),
#                     transforms.ToTensor(),  # implicitly divides by 255
#                     transforms.Normalize(
#                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                     ),
#                 ]
#             )
#             self.concepts = (
#                 torch.load(kwargs["root"] + f"cifar100_train_concept_labels.pt") * 1
#             )
#         else:
#             self.transform = transforms.Compose(
#                 [
#                     transforms.Resize(size=(224, 224)),
#                     transforms.ToTensor(),  # implicitly divides by 255
#                     transforms.Normalize(
#                         mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
#                     ),
#                 ]
#             )
#             self.concepts = (
#                 torch.load(kwargs["root"] + f"cifar100_test_concept_labels.pt") * 1
#             )

#     def __getitem__(self, idx):
#         X, target = super().__getitem__(idx)

#         return {
#             "img_code": idx,
#             "labels": target,
#             "features": X,
#             "concepts": self.concepts[idx],
#         }


class Cifar100Loader(Dataset):
    def __init__(
            self,
            data_dir='/DATA/cifar100',
            json_file='/DATA/cifar100/cifar100_concepts_filtered_700.json', 
            split='train',
            transforms=None,
        ):
        self.class_concept_dict = json.load(open(json_file, 'r'), object_pairs_hook=OrderedDict)
        self.split = split
        self.class_list = self.class_concept_dict.keys()
        self.num_classes = len(self.class_list)
        self.concept_list = []
        for v in self.class_concept_dict.values():
            self.concept_list += v
        self.concept_list = list(set(self.concept_list))
        self.num_attrs = len(self.concept_list)
        self.class_label_map = {i: k for i, k in zip(np.arange(0, 100), self.class_list)}
        self.concept_label_dict = {}

        #  train - 308, random crop - 299, test - centre crop
        if transforms is None:
            if split == 'train':
                self.transforms = Compose([
                    Resize((308, 308)), 
                    RandomCrop(299),    
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
            else:
                self.transforms = Compose([
                    Resize((308, 308)), 
                    CenterCrop(299),    
                    ToTensor(),
                    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                ])
        else:
            self.transforms = transforms

        self.data = datasets.CIFAR100(root=data_dir, train=(split=='train'), download=False, transform=self.transforms)

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
        return {
                "img_code": index,
                "labels": label,
                "features": image,
                "concepts": attrs,
            }
