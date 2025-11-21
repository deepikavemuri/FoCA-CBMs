import os

from torchvision import transforms, datasets
import torch
import torch.multiprocessing
import numpy as np

torch.multiprocessing.set_sharing_strategy("file_system")

base_path = ""
# the paths for the datasets
datasets_paths = {
    "imagenet": base_path + "data/imagenet",
    "cub": "/DATA/ai22resch11001/projects/data/CUB_200_2011",
    "places365": base_path + "data/places365",
    "sun": base_path + "data/SUN",
    "imagenet100": "./../DATA/inet100",
    "awa2": "./../DATA/Animals_with_Attributes2_imagefolder",
    "cifar100": "./../DATA/cifar100",
}

# the class labels for each set
label_paths = {
    "places365": base_path + "data/categories_places365.txt",
    "imagenet": base_path + "data/imagenet_classes.txt",
    "cub": base_path + "data/cub_classes.txt",
    "sun": base_path + "data/sun_classes.txt",
    "imagenet100": "./Coarse-To-Fine-CBMs/data/concept_sets_high/imagenet100_classes.txt",
    "awa2": "./Coarse-To-Fine-CBMs/data/concept_sets_high/AwA2/awa2_classes.txt",
    "cifar100": "./Coarse-To-Fine-CBMs/data/concept_sets_high/CIFAR100/cifar100_classes.txt",
}

concept_paths = {
    "imagenet": base_path + "data/concept_sets_high/imagenet_classes_clean.txt",
    "imagenet_attrs": base_path
    + "data/concept_sets_low/ImageNet/imagenet_attributes_cleaned.txt",
    "imagenet_attrs_inds": base_path
    + "data/concept_sets_low/ImageNet/imagenet_attrs_per_class_binary.npy",
    "cub": base_path + "data/concept_sets_high/cub_classes_cleaned.txt",
    "cub_attrs": base_path + "data/concept_sets_low/CUB/cub_attributes_cleaned.txt",
    "cub_attrs_inds": base_path
    + "data/concept_sets_low/CUB/cub_attrs_per_class_binary_20.npy",
    "sun": base_path + "data/concept_sets_high/sun_classes.txt",
    "sun_attrs": base_path + "data/concept_sets_low/SUN/sun_attributes.txt",
    "sun_attrs_inds": base_path
    + "data/concept_sets_low/SUN/sun_attrs_per_class_binary_0.npy",
    "10k": base_path + "data/concept_sets_low/10k.txt",
    "10k_attrs": base_path + "data/concept_sets_low/10k.txt",
    "imagenet100": "./Coarse-To-Fine-CBMs/data/concept_sets_low/Imagenet100/inet100_attributes.txt",
    "imagenet100_attrs": "./Coarse-To-Fine-CBMs/data/concept_sets_low/Imagenet100/inet100_attributes.txt",
    "imagenet100_attrs_inds": "./Coarse-To-Fine-CBMs/data/concept_sets_low/Imagenet100/imagenet100_attrs_per_class_binary_20.npy",
    "awa2": "./Coarse-To-Fine-CBMs/data/concept_sets_high/AwA2/awa2_classes.txt",
    "awa2_attrs": "./Coarse-To-Fine-CBMs/data/concept_sets_low/AwA2/awa2_attributes.txt",
    "awa2_attrs_inds": "./Coarse-To-Fine-CBMs/data/concept_sets_low/AwA2/awa2_attrs_per_class_binary_85.npy",
    "cifar100": "./Coarse-To-Fine-CBMs/data/concept_sets_high/CIFAR100/cifar100_classes.txt",
    "cifar100_attrs": "./Coarse-To-Fine-CBMs/data/concept_sets_low/CIFAR100/cifar100_attributes.txt",
    "cifar100_attrs_inds": "./Coarse-To-Fine-CBMs/data/concept_sets_low/CIFAR100/cifar100_attrs_per_class_binary.npy",
}


def get_feature_dir(args, val=False):
    save_dir = args.save_dir + "/{}/{}/{}/{}/".format(
        args.dataset,
        args.clip_version.replace("/", "_"),
        "Raw",
        "train" if not val else "val",
    )

    return save_dir


class DualDataset(torch.utils.data.Dataset):
    def __init__(self, data_whole, data_patches):

        self.whole_images = data_whole[0]
        self.whole_labels = data_whole[1]

        self.patches = data_patches[0]
        self.patches_labels = data_patches[1]

    def __getitem__(self, index):
        x_whole = self.whole_images[index]
        y_whole = self.whole_labels[index]

        x_patches = self.patches[index]
        y_patches = self.patches_labels[index]

        return x_whole, y_whole, x_patches, y_patches

    def __len__(self):
        return len(self.whole_images)


def get_loaders(
    args, preprocess=None, patchify=False, batch_size=128, dual_data=False, cs_path=None
):
    """
    Create the loaders for the dataset. The name and other parameters are in the arg variable.
    :param args:
    :param preprocess: torchvision Transform to use instead of the default. For example one could use the CLIP preprocess.

    :return: the train and validation loader, the classes of the dataset and the concept set.
    """
    print(args.dataset)
    if preprocess:
        train_transform = preprocess
        val_transform = preprocess
    # if we load the similarities from file, we cant easily do augmentations
    elif args.dataset not in ["cub", "imagenet", "imagenet100", "sun", "awa2"]:
        train_transform = transforms.Compose([transforms.ToTensor()])

        val_transform = transforms.Compose([transforms.ToTensor()])

    elif args.dataset in ["cub", "imagenet", "imagenet100", "awa2"]:
        print("Using standard imagenet transforms..")
        train_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(224)]
        )
        val_transform = transforms.Compose(
            [transforms.ToTensor(), transforms.Resize(256), transforms.CenterCrop(224)]
        )
    elif args.dataset == "sun":
        train_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                     std=[0.229, 0.224, 0.225]),
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )

        val_transform = transforms.Compose(
            [
                transforms.ToTensor(),
                # transforms.Normalize(mean=[0.485, 0.456, 0.406],
                #                     std=[0.229, 0.224, 0.225]),
                transforms.Resize(256),
                transforms.CenterCrop(224),
            ]
        )
    else:
        raise ValueError("Wrong dataset name..")

    # Load the dataset
    train_data, val_data, classes = data_loader(
        args,
        preprocess_train=train_transform,
        preprocess_val=val_transform,
        load_similarities=not args.compute_similarities,
        patchify=patchify,
        dual_data=dual_data,
        cs_path=cs_path,
    )
    train_sampler = None

    # this shuffle thing is new..
    train_loader = torch.utils.data.DataLoader(
        train_data,
        batch_size=batch_size,
        shuffle=not args.compute_similarities,
        # (not args.compute_similarities),  # (train_sampler is None),
        num_workers=args.num_workers,
        pin_memory=False,
        sampler=None,
    )

    val_loader = torch.utils.data.DataLoader(
        val_data, batch_size=batch_size, shuffle=False, num_workers=1, pin_memory=False
    )

    return train_loader, val_loader, classes


def data_loader(
    args,
    preprocess_train=None,
    preprocess_val=None,
    load_similarities=True,
    patchify=False,
    dual_data=False,
    cs_path=None,
):
    """

    :return: data_train, data_val. Torch tensors. The training and validation data for the chosen dataset.
    """

    name = args.dataset

    if not load_similarities:
        if name == "cifar10":
            data_train = datasets.CIFAR10(
                root=datasets_paths["cifar10"],
                download=True,
                train=True,
                transform=preprocess_train,
            )
            data_val = datasets.CIFAR10(
                root=datasets_paths["cifar10"],
                download=True,
                train=False,
                transform=preprocess_val,
            )

        elif name == "cifar100":
            data_train = datasets.CIFAR100(
                root=datasets_paths["cifar100"],
                download=True,
                train=True,
                transform=preprocess_train,
            )
            data_val = datasets.CIFAR100(
                root=datasets_paths["cifar100"],
                download=True,
                train=False,
                transform=preprocess_val,
            )
        elif name == "places365":
            data_train = datasets.Places365(
                root=datasets_paths["places365"],
                download=False,
                split="train-standard",
                small=True,
                transform=preprocess_train,
            )
            data_val = datasets.Places365(
                root=datasets_paths["places365"],
                download=False,
                split="val",
                small=True,
                transform=preprocess_val,
            )

        # this is for imagenet and cub and inet100
        elif name in datasets_paths.keys():
            print(preprocess_train)
            print(os.getcwd())
            print(datasets_paths[name])
            data_train = datasets.ImageFolder(
                datasets_paths[name] + "/train/", preprocess_train
            )
            data_val = datasets.ImageFolder(
                datasets_paths[name] + "/val/", preprocess_val
            )

        else:
            raise ValueError("Dataset {} not supported (yet?)..".format(name))

    else:

        # if we do not want dual data it will load either the whole image or the patches
        # if we want both we put the High and the low in the same Dataset (in that order)
        confs = [patchify] if not dual_data else [False, True]
        # I need to change this for multiple files that may be present
        train_datasets = []
        val_datasets = []

        print(dual_data)
        print(confs)

        for conf in confs:
            save_dir = get_feature_dir(args)
            print(save_dir)
            save_name_features = save_dir + "image_{}_{}".format(
                "feats",
                (
                    "patches_{}_{}".format(args.patch_size[0], args.patch_size[1])
                    if conf
                    else "whole"
                ),
            )
            save_name_features += ".pt"
            data_tensor, target_tensor = torch.load(save_name_features)
            train_datasets.append(
                (data_tensor.cpu().float(), target_tensor.cpu().float())
            )

            # do the same for validation set
            save_dir = get_feature_dir(args, val=True)
            save_name_features = save_dir + "image_{}_{}".format(
                "feats",
                (
                    "patches_{}_{}".format(args.patch_size[0], args.patch_size[1])
                    if conf
                    else "whole"
                ),
            )
            save_name_features += ".pt"
            data_tensor, target_tensor = torch.load(save_name_features)
            val_datasets.append(
                (data_tensor.cpu().float(), target_tensor.cpu().float())
            )
            # data_val = torch.utils.data.TensorDataset(data_tensor.cpu().float(), target_tensor.cpu().float())

        if not dual_data:
            data_train = torch.utils.data.TensorDataset(*train_datasets[0])
            data_val = torch.utils.data.TensorDataset(*val_datasets[0])
        else:
            data_train = DualDataset(*train_datasets)
            data_val = DualDataset(*val_datasets)

    # read the classes from the label files
    classes = []
    with open((cs_path if cs_path else "") + label_paths[name], "r") as f:
        for line in f:
            # classes.append(line.strip().split()[0][3:])
            classes.append(line.strip())

    return data_train, data_val, classes


def get_concepts(concept_name, patchify, cs_path=None):
    #  read the concepts
    concept_set = []
    with open(
        (cs_path if cs_path else "")
        + concept_paths[concept_name + "_attrs" * patchify],
        "r",
    ) as f:
        for line in f:
            concept_set.append(line.strip())

    return concept_set


def get_concept_indicators(concept_name, cs_path=None):
    binary_inds = torch.tensor(
        np.load(
            (cs_path if cs_path else "") + concept_paths[concept_name + "_attrs_inds"]
        ).astype(np.float32)
    )
    return binary_inds
