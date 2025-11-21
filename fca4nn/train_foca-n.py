import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import (
    ImagenetMultiClassLevelsDataset,
    Awa2ClassLevelDataset,
    Cifar100ClassLevelDataset,
)
from torch.utils.data import DataLoader
from model import FoCA_CBM_N_resnet
import argparse
from tqdm import tqdm
import os
import random
import numpy as np
from datetime import datetime
from losses import binary_crossentropy
from loguru import logger
import json
import sys
from utils import log_and_print, get_scheduler, get_optimizer
from glob import glob
import heapq
import time
import torch.nn.functional as F
import wandb
from processing.utils import get_info_from_lattice


def filter_out_specific_info(record):
    # Exclude a specific logger.info message
    if record["level"].name == "INFO" and "Batch" in record["message"]:
        return False  # Exclude this log message
    return True


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


@torch.inference_mode()
def validate(args, model, val_dataset, epoch=None, logger=None):
    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )

    model.eval()
    loss = 0
    cls_acc, attr1_acc, attr2_acc = 0, 0, 0
    header = (
        f"Val: Epoch {epoch} - Batch Progress"
        if epoch is not None or epoch == 0
        else f"Test: - Batch Progress"
    )
    for i, data in enumerate(tqdm(valloader, desc=header)):
        _, imgs, cls, attrs_present, classes_present = data

        attr1_pred, attr2_pred, classes = model(imgs.cuda())

        classification_acc = (
            sum(torch.argmax(classes.cpu(), dim=-1) == cls) / imgs.shape[0]
        )
        cls_acc += classification_acc.detach().numpy().mean().item()

        attribute1_acc = (
            sum(torch.round(F.sigmoid(attr1_pred).cpu()) == attrs_present[0])
            / imgs.shape[0]
        )
        attr1_acc += attribute1_acc.detach().numpy().mean().item()
        attribute2_acc = (
            sum(torch.round(F.sigmoid(attr2_pred).cpu()) == attrs_present[1])
            / imgs.shape[0]
        )
        attr2_acc += attribute2_acc.detach().numpy().mean().item()

        attr1_loss = binary_crossentropy(
            F.sigmoid(attr1_pred), attrs_present[0].to(torch.float32).cuda()
        ).mean()
        attr2_loss = binary_crossentropy(
            F.sigmoid(attr2_pred), attrs_present[1].to(torch.float32).cuda()
        ).mean()
        cls_loss = nn.CrossEntropyLoss()(classes, cls.cuda())
        loss += cls_loss + args.concept_wts * (attr1_loss + attr2_loss)

    loss /= len(valloader)
    cls_acc /= len(valloader)
    attr1_acc /= len(valloader)
    attr2_acc /= len(valloader)
    if epoch:
        log_and_print(
            f"Validation {epoch}:  Loss: {loss}, Classifier Acc: {cls_acc}, Attribute Acc: {attr1_acc, attr2_acc}",
            logger,
        )
    else:
        log_and_print(
            f"Test: Loss: {loss}, Classifier Acc: {cls_acc}, Attribute Acc: {attr1_acc, attr2_acc}",
            logger,
        )
    return loss, cls_acc


def train_and_validate(args, model, train_dataset, val_dataset, logger=None):
    top_checkpoints = []  # Min-heap to track top 5 (val_acc, path)

    trainloader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
    )
    # for x in trainloader:
    # print(x[1].shape, x[2], x[3][0][-5], x[4][0][-5])
    # exit(0)

    if args.wandb:
        wandb.init(project="fca_mclcbm", entity="ai21btech11004", config=args)
        wandb.run.name = f"mclcbm_{args.dataset}_{wandb.run.name}"

        wandb.watch(model, log="all", log_freq=args.verbose)
        wandb.config.command = " ".join(sys.argv)
        wandb.config.update(
            {
                "lr": args.lr,
                "epochs": args.epochs,
            }
        )

    opt = get_optimizer(model, args)
    scheduler = get_scheduler(opt, args, trainloader)
    ce_loss = nn.CrossEntropyLoss()
    bce_loss = binary_crossentropy
    best_val_acc = 0
    for epoch in range(args.epochs):
        cls_acc, attr1_acc, attr2_acc = 0, 0, 0
        model.train()
        running_loss, loss = 0, 0
        for i, data in enumerate(
            tqdm(trainloader, desc=f"Train: Epoch {epoch} - Batch Progress")
        ):
            _, imgs, cls, attrs_present, classes_present = data
            attr1_pred, attr2_pred, classes = model(imgs.cuda())

            opt.zero_grad()

            classification_acc = (
                sum(torch.argmax(classes.cpu(), dim=-1) == cls) / imgs.shape[0]
            )
            cls_acc += classification_acc.detach().numpy().mean().item()
            cls_loss = ce_loss(classes, cls.cuda())

            attribute1_acc = (
                sum(torch.round(F.sigmoid(attr1_pred).cpu()) == attrs_present[0])
                / imgs.shape[0]
            )
            attr1_acc += attribute1_acc.detach().numpy().mean().item()
            attribute2_acc = (
                sum(torch.round(F.sigmoid(attr2_pred).cpu()) == attrs_present[1])
                / imgs.shape[0]
            )
            attr2_acc += attribute2_acc.detach().numpy().mean().item()

            attr1_loss = bce_loss(
                F.sigmoid(attr1_pred), attrs_present[0].to(torch.float32).cuda()
            ).mean()
            attr2_loss = bce_loss(
                F.sigmoid(attr2_pred), attrs_present[1].to(torch.float32).cuda()
            ).mean()
            loss = cls_loss + args.concept_wts * (attr1_loss + attr2_loss)

            running_loss += loss.item()
            loss.backward()
            opt.step()
            if args.scheduler != "constant":
                scheduler.step()
            if i % args.verbose == 0:
                log_and_print(
                    f"Epoch {epoch} - Batch {i}: Loss={loss.item()}, Classifier Loss={cls_loss.item()}, Attribute Loss={attr1_loss.item(), attr2_loss.item()}",
                    logger,
                )
            if args.wandb:
                wandb.log(
                    {
                        f"train_clf_loss": cls_loss.item(),
                        f"train_attr1_loss": attr1_loss.item(),
                        f"train_attr2_loss": attr2_loss.item(),
                    }
                )
        running_loss /= len(trainloader)
        cls_acc /= len(trainloader)
        attr1_acc /= len(trainloader)
        attr2_acc /= len(trainloader)
        log_and_print(
            f"Epoch {epoch}: Loss={running_loss}, Classifier Acc={cls_acc}, Attribute Acc={attr1_acc, attr2_acc}",
            logger,
        )
        if epoch % args.validation_freq == 0 or epoch == args.epochs - 1:
            loss, val_acc = validate(args, model, val_dataset, epoch, logger)
            model_path = os.path.join(
                args.save_model_dir,
                f"mclcbm_model_epoch_{epoch}_val_acc_{val_acc:.4f}.pt",
            )
            torch.save(model.state_dict(), model_path)
            log_and_print(
                f"Model saved at epoch {epoch} with val acc {val_acc} at {model_path}",
                logger,
            )
            heapq.heappush(top_checkpoints, (val_acc, -epoch, model_path))
            if len(top_checkpoints) > args.keep_top_k:
                _, _, path_to_remove = heapq.heappop(top_checkpoints)
                if os.path.exists(path_to_remove):
                    os.remove(path_to_remove)
                    log_and_print(
                        f"Deleted old checkpoint: {path_to_remove}",
                        logger,
                    )
            # Update best val acc if needed
            if val_acc > best_val_acc:
                best_val_acc = val_acc

    all_paths = glob(os.path.join(args.save_model_dir, "*.pt"))
    # sort by val_acc and load the best model
    all_paths.sort(key=lambda x: float(x.split("_")[-1].split(".")[1]), reverse=True)
    best_model_path = all_paths[0]
    log_and_print(f"Best model saved at {best_model_path}", logger)
    log_and_print(f"Loading model from {best_model_path}", logger)
    model.load_state_dict(torch.load(best_model_path, weights_only=True), strict=True)
    return model


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42, help="Specify seed")
    parser.add_argument("--model", type=str, default="resnet18", help="Model name")
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
    args.varying_lrs = False

    save_model_dir = os.path.join(
        args.save_model_dir,
        "mclcbm",
        args.dataset,
        f"exp_{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}",
    )
    args.save_model_dir = save_model_dir
    os.makedirs(save_model_dir, exist_ok=True)
    with open(os.path.join(save_model_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger_name = "log"
    if args.do_train_full:
        logger_name += "_train"
    if args.do_train_fewshot:
        logger_name += "_train_fewshot"
    if args.do_test:
        logger_name += "_test"
    logger_name += "_{time}.log"

    logger.remove()
    logger.add(
        os.path.join(save_model_dir, logger_name),
        rotation="10 MB",
        retention="10 days",
        level="INFO",
    )
    logger.add(sys.stdout, level="INFO", filter=filter_out_specific_info)
    logger.info("Starting MCLCBM training")

    # log arguments using pretty printer
    logger.info(f"Arguments: {vars(args)}")

    set_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")

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
    else:
        raise ValueError("Invalid dataset name")

    model = FoCA_CBM_N_resnet(
        perlevel_intents,
        perlevel_fcs,
        num_classes,
        backbone_name=args.model,
    ).cuda()

    if args.do_train_full or args.do_train_fewshot:
        best_model = train_and_validate(
            args, model, train_dataset, val_dataset, logger=logger
        )

    if args.do_test:
        if args.best_model_path is None and not (
            args.do_train_full or args.do_train_fewshot
        ):
            model_paths = glob(
                os.path.join(args.save_model_dir, f"mclcbm_model_epoch_*.pt")
            )
            model_paths.sort(
                key=lambda x: float(x.split("_")[-1].split(".")[0]), reverse=True
            )
            best_model_path = model_paths[0]
            log_and_print(f"Loading model from {best_model_path}", logger)
            best_model = model.load_state_dict(
                torch.load(best_model_path, weights_only=True), strict=True
            )
        elif args.best_model_path is not None and not (
            args.do_train_full or args.do_train_fewshot
        ):
            best_model_path = args.best_model_path
            log_and_print(f"Loading model from {best_model_path}", logger)
            best_model = model.load_state_dict(
                torch.load(best_model_path, weights_only=True), strict=True
            )
        logger.info("========Testing on Test set========")
        loss, test_acc = validate(args, best_model, test_dataset, logger=logger)
        logger.info(f"Test accuracy: {test_acc}, Loss: {loss}")
        logger.info("Testing complete")
    if args.wandb:
        wandb.finish()
