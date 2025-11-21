import os
import sys
import json
import random
import argparse

import torch
import numpy as np

from train_foca import train_and_validate
from model import FoCA_CBM_resnet, FoCA_CBM_vit
from dataloader import (
    ImagenetMultiClassLevelsDataset,
    Awa2ClassLevelDataset,
    Cifar100ClassLevelDataset,
)
from processing.utils import get_info_from_lattice

from loguru import logger
from datetime import datetime
import warnings

warnings.filterwarnings("ignore", category=UserWarning, message=".*?.*?")


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def filter_out_specific_info(record):
    # Exclude a specific logger.info message
    if record["level"].name == "INFO" and "Batch" in record["message"]:
        return False  # Exclude this log message
    return True


def main(args, logger=None):
    perlevel_intents, perlevel_fcs = get_info_from_lattice(
        args.lattice_path, args.lattice_levels, args.exclusive_attrs
    )
    if args.dataset == "imagenet100":
        train_dataset = ImagenetMultiClassLevelsDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            split="train",
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
            fraction=args.fraction,
        )
        val_dataset = ImagenetMultiClassLevelsDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            split="val",
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
        )
        test_dataset = ImagenetMultiClassLevelsDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            split="test",
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
        )
        num_classes = train_dataset.num_classes
        cls_list = train_dataset.class_concept_dict.keys()

    elif args.dataset == "awa2":
        train_dataset = Awa2ClassLevelDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            split="train",
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
            fraction=args.fraction,
            few_shot_train=args.do_train_fewshot,
        )
        val_dataset = Awa2ClassLevelDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            split="val",
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
            few_shot_train=args.do_train_fewshot,
        )
        test_dataset = Awa2ClassLevelDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            split="test",
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
            few_shot_train=args.do_train_fewshot,
        )
        num_classes = train_dataset.num_classes
        cls_list = train_dataset.class_to_index.keys()
    elif args.dataset == "cifar100":
        train_dataset = Cifar100ClassLevelDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            split="train",
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
        )
        val_dataset = Cifar100ClassLevelDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            split="val",
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
        )
        test_dataset = Cifar100ClassLevelDataset(
            data_root=args.data_root,
            json_file=args.concept_file,
            split="test",
            lattice_levels=args.lattice_levels,
            lattice_path=args.lattice_path,
            perlevel_intents=perlevel_intents,
            perlevel_fcs=perlevel_fcs,
        )
        num_classes = train_dataset.num_classes
        cls_list = train_dataset.class_concept_dict.keys()

    # print(np.sum(train_dataset.classes_present_perlevel_perlabel[0][0]))
    # print("num clfs:", args.num_clfs)
    # for j in range(args.num_clfs):
    #     cur_level = 0
    #     for i in range(num_classes):
    #         cur_level += np.sum(train_dataset.classes_present_perlevel_perlabel[i][j])
    #     print(cur_level / num_classes)
    # exit(0)

    if args.model_type == "resnet":
        model = FoCA_CBM_resnet(
            perlevel_intents,
            perlevel_fcs,
            args.backbone_layer_ids,
            num_classes,
            pretrained_clfs_path=args.pretrained_clfs_path,
            pretrained_attrs_path=args.pretrained_attrs_path,
            pretrained_backbone_path=args.pretrained_backbone_path,
            backbone_name=args.model,
            exclusive_attrs=args.exclusive_attrs,
        )
    elif args.model_type == "vit":
        model = FoCA_CBM_vit(
            perlevel_intents,
            perlevel_fcs,
            args.backbone_layer_ids,
            num_classes,
            model_name=args.model,
        )
    if args.clf_special_init:
        model.clf_weight_initialization(
            json.load(open(args.concept_file, "r")), cls_list
        )
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)

    train_and_validate(
        args, train_dataset, val_dataset, test_dataset, model, args.num_clfs, logger
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Hierarchical FCA")
    parser.add_argument("--seed", type=int, default=42, help="Specify seed")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name")
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet",
        help="Model type",
        choices=["resnet", "vit"],
    )
    parser.add_argument(
        "--dataset", type=str, default="imagenet100", help="Dataset name"
    )
    parser.add_argument(
        "--data_root", type=str, default="./data/", help="Path to data root"
    )
    parser.add_argument(
        "--concept_file",
        type=str,
        default="./data/concept_files/",
        help="Path to concept file",
    )
    parser.add_argument(
        "--lattice_path",
        type=str,
        default="./data/lattices/",
        help="Path to lattice file",
    )
    parser.add_argument("--epochs", type=int, default=30, help="Number of epochs")
    parser.add_argument("--concept_wts", type=float, default=0.1)
    parser.add_argument("--cls_wts", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument(
        "--varying_lrs",
        action="store_true",
        help="Use different lrs for different levels",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-5, help="Weight decay for optimizer"
    )
    parser.add_argument("--num_clfs", type=int, default=1, help="Number of classifiers")
    parser.add_argument(
        "--lattice_levels", nargs="+", type=int, help="Choose lattice levels"
    )
    parser.add_argument(
        "--backbone_layer_ids",
        nargs="+",
        type=int,
        help="Choose where to place intermediate semantic layers",
    )
    parser.add_argument(
        "--pretrained_clfs_path",
        type=str,
        default=None,
        help="Path to pretrained classifiers",
    )
    parser.add_argument(
        "--pretrained_attrs_path",
        type=str,
        default=None,
        help="Path to pretrained attribute layers",
    )
    parser.add_argument(
        "--pretrained_backbone_path",
        type=str,
        default=None,
        help="Path to pretrained backbone",
    )
    parser.add_argument("--wandb", action="store_true", help="Use wandb for logging")
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="./saved_models/",
        help="Path to save models",
    )
    parser.add_argument(
        "--verbose", type=int, default=None, help="Print every n batches"
    )
    parser.add_argument(
        "--clf_l1_reg",
        action="store_true",
        help="Use L1 regularization for classifiers",
    )
    parser.add_argument(
        "--clf_special_init",
        action="store_true",
        help="Use special initialization for classifiers",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="onecycle",
        help="Scheduler type",
        choices=["cosine", "step", "linear", "onecycle", "constant", "none"],
    )
    parser.add_argument(
        "--validation_freq", type=int, default=2, help="Validation frequency"
    )
    parser.add_argument(
        "--grad_clip", type=float, default=None, help="Gradient clipping value"
    )
    parser.add_argument("--do_train_full", action="store_true", help="Train the model")
    parser.add_argument(
        "--do_train_fewshot",
        action="store_true",
        help="Train the model on only seen classes",
    )
    parser.add_argument("--do_train_attrs", action="store_true", help="Train the model")
    parser.add_argument(
        "--sequential_training", action="store_true", help="Train layer by layer"
    )
    parser.add_argument(
        "--exclusive_attrs",
        action="store_true",
        help="Construct attribute sets that are mutually exclusive",
    )
    parser.add_argument("--do_test", action="store_true", help="Test the model")
    parser.add_argument(
        "--best_model_path",
        type=str,
        default=None,
        help="Path to the best model for testing",
    )
    parser.add_argument(
        "--attr_loss",
        type=str,
        default="bce",
        help="Attribute loss function",
        choices=["focal", "bce", "hierarchical_ce", "dice"],
    )
    parser.add_argument(
        "--pre_final_clf_loss",
        type=str,
        default="bce",
        help="Classifier loss function",
        choices=["focal", "bce", "hierarchical_ce", "dice"],
    )
    parser.add_argument(
        "--final_clf_loss",
        type=str,
        default="ce",
        help="Classifier loss function",
        choices=["focal", "bce", "hierarchical_ce", "dice"],
    )
    parser.add_argument(
        "--opt",
        type=str,
        default="adamw",
        help="Optimizer type",
        choices=["adam", "sgd", "adamw"],
    )
    parser.add_argument(
        "--momentum", type=float, default=0.9, help="Momentum for SGD optimizer"
    )
    parser.add_argument(
        "--keep_top_k",
        type=int,
        default=5,
        help="Keep top k checkpoints based on validation accuracy",
    )
    parser.add_argument(
        "--fraction",
        type=str,
        default="full",
        help="Fraction of the dataset to use for training",
        choices=["full", "half", "quarter"],
    )

    args = parser.parse_args()

    save_model_dir = os.path.join(
        args.save_model_dir,
        args.model_type,
        args.dataset,
        str(args.num_clfs) + "intsem",
        f"exp_{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}",
    )

    args.save_model_dir = save_model_dir
    os.makedirs(save_model_dir, exist_ok=True)
    # save the arguments to a json file
    with open(os.path.join(save_model_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger_name = "log"
    if args.do_train_full:
        logger_name += "_train"
    if args.do_train_attrs:
        logger_name += "_train_attrs"
    if args.do_train_fewshot:
        logger_name += "_train_fewshot"
    if args.do_test:
        logger_name += "_test"
    logger_name += "_{time}.log"

    if args.do_test and not (
        args.do_train_full or args.do_train_attrs or args.do_train_fewshot
    ):
        assert (
            args.best_model_path is not None
        ), "Please provide the path to the best model for testing"
        args.wandb = False

    logger.remove()
    logger.add(
        os.path.join(save_model_dir, logger_name),
        rotation="10 MB",
        retention="10 days",
        level="INFO",
    )
    logger.add(sys.stdout, level="INFO", filter=filter_out_specific_info)
    logger.info("Starting FCA4NN training")

    # log arguments using pretty printer
    logger.info(f"Arguments: {vars(args)}")

    set_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")
    main(args, logger)
