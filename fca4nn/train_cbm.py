import torch
import torch.nn as nn
import torch.optim as optim

from dataloader import (
    AnimalLoader,
    Imagenet100ConceptDataset,
    Cifar100Loader,
)
from torch.utils.data import DataLoader
from model import CBM_resnet, CBM_vit
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
    cls_acc, attr_acc = 0, 0
    header = (
        f"Val: Epoch {epoch} - Batch Progress"
        if epoch is not None or epoch == 0
        else f"Test: - Batch Progress"
    )
    for i, data in enumerate(tqdm(valloader, desc=header)):
        imgs, cls, attrs = data

        concepts, classes = model(imgs.cuda())

        classification_acc = (
            sum(torch.argmax(classes.cpu(), dim=-1) == cls) / imgs.shape[0]
        )
        cls_acc += classification_acc.detach().numpy().mean().item()

        attribute_acc = (
            sum(torch.round(F.sigmoid(concepts).cpu()) == attrs) / imgs.shape[0]
        )
        attr_acc += attribute_acc.detach().numpy().mean().item()
        attr_loss = binary_crossentropy(
            F.sigmoid(concepts), attrs.to(torch.float32).cuda()
        ).mean()
        cls_loss = nn.CrossEntropyLoss()(classes, cls.cuda())
        loss += cls_loss + args.concept_wts * attr_loss

    loss /= len(valloader)
    cls_acc /= len(valloader)
    attr_acc /= len(valloader)
    if epoch:
        log_and_print(
            f"Validation {epoch}:  Loss: {loss}, Classifier Acc: {cls_acc}, Attribute Acc: {attr_acc}",
            logger,
        )
    else:
        log_and_print(
            f"Test: Loss: {loss}, Classifier Acc: {cls_acc}, Attribute Acc: {attr_acc}",
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

    if args.wandb:
        wandb.init(project="fca_intsem", entity="ai22resch11001", config=args)
        wandb.run.name = f"fca4nn_{args.dataset}_{wandb.run.name}"

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
        cls_acc, attr_acc = 0, 0
        model.train()
        running_loss, loss = 0, 0
        for i, data in enumerate(
            tqdm(trainloader, desc=f"Train: Epoch {epoch} - Batch Progress")
        ):
            imgs, cls, attrs = data
            concepts, classes = model(imgs.cuda())

            opt.zero_grad()

            classification_acc = (
                sum(torch.argmax(classes.cpu(), dim=-1) == cls) / imgs.shape[0]
            )
            cls_acc += classification_acc.detach().numpy().mean().item()
            cls_loss = ce_loss(classes, cls.cuda())

            attribute_acc = (
                sum(torch.round(F.sigmoid(concepts).cpu()) == attrs) / imgs.shape[0]
            )
            attr_acc += attribute_acc.detach().numpy().mean().item()
            attr_loss = bce_loss(
                F.sigmoid(concepts), attrs.to(torch.float32).cuda()
            ).mean()
            loss = cls_loss + args.concept_wts * attr_loss
            running_loss += loss.item()
            loss.backward()
            opt.step()
            if args.scheduler != "constant":
                scheduler.step()
            if i % args.verbose == 0:
                log_and_print(
                    f"Epoch {epoch} - Batch {i}: Loss={loss.item()}, Classifier Loss={cls_loss.item()}, Attribute Loss={attr_loss.item()}",
                    logger,
                )
            if args.wandb:
                wandb.log(
                    {
                        f"train_clf_loss": cls_loss.item(),
                        f"train_attr_loss": attr_loss.item(),
                    }
                )
        running_loss /= len(trainloader)
        cls_acc /= len(trainloader)
        attr_acc /= len(trainloader)
        log_and_print(
            f"Epoch {epoch}: Loss={running_loss}, Classifier Acc={cls_acc}, Attribute Acc={attr_acc}",
            logger,
        )
        if epoch % args.val_freq == 0 or epoch == args.epochs - 1:
            loss, val_acc = validate(args, model, val_dataset, epoch, logger)
            model_path = os.path.join(
                args.save_model_dir,
                f"cbm_model_epoch_{epoch}_val_acc_{val_acc:.4f}.pt",
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
    parser.add_argument(
        "--dataset",
        type=str,
        default="imagenet100",
        help="Dataset name",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="./data",
        help="Path to dataset",
    )
    parser.add_argument(
        "--concept_file",
        type=str,
        default="./data/imagenet100_concept.json",
        help="Path to concept file",
    )
    parser.add_argument(
        "--scheduler",
        type=str,
        default="constant",
        help="Scheduler type",
        choices=["cosine", "step", "linear", "onecycle", "constant", "none"],
    )
    parser.add_argument(
        "--save_model_dir",
        type=str,
        default="./saved_models/",
        help="Model save path",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        default="resnet",
        help="Provide model type (Resnet / ViT)",
    )
    parser.add_argument(
        "--model_name",
        type=str,
        default="resnet50",
        help="Model name",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument("--concept_wts", type=float, default=0.01)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--num_workers", type=int, default=12)
    parser.add_argument(
        "--do_train",
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--do_train_fewshot",
        action="store_true",
        help="Whether to run training.",
    )
    parser.add_argument(
        "--weight_decay", type=float, default=5e-5, help="Weight decay for optimizer"
    )
    parser.add_argument(
        "--do_test",
        action="store_true",
        help="Whether to run testing.",
    )
    parser.add_argument("--val_freq", type=int, default=2, help="Validation frequency")
    parser.add_argument(
        "--verbose", type=int, default=None, help="Print every n batches"
    )
    parser.add_argument(
        "--best_model_path",
        type=str,
        default=None,
        help="Path to the best model for testing",
    )
    parser.add_argument(
        "--keep_top_k",
        type=int,
        default=5,
        help="Number of top checkpoints to keep",
    )
    parser.add_argument(
        "--fraction",
        type=str,
        default="full",
        help="Fraction of the dataset to use for training",
        choices=["full", "half", "quarter"],
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

    args = parser.parse_args()
    args.varying_lrs = False

    save_model_dir = os.path.join(
        args.save_model_dir,
        "cbm",
        args.dataset,
        f"exp_{datetime.now().strftime('%Y:%m:%d-%H:%M:%S')}",
    )
    args.save_model_dir = save_model_dir
    os.makedirs(save_model_dir, exist_ok=True)
    with open(os.path.join(save_model_dir, "args.json"), "w") as f:
        json.dump(vars(args), f, indent=4)

    logger_name = "log"
    if args.do_train:
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
    logger.info("Starting Vanilla CBM training")

    # log arguments using pretty printer
    logger.info(f"Arguments: {vars(args)}")

    set_seed(args.seed)
    logger.info(f"Seed set to {args.seed}")

    if args.dataset == "imagenet100":
        train_set = Imagenet100ConceptDataset(
            data_root=args.data_dir,
            json_file=args.concept_file,
            split="train",
            fraction=args.fraction,
        )
        val_set = Imagenet100ConceptDataset(
            data_root=args.data_dir,
            json_file=args.concept_file,
            split="val",
        )
        test_set = Imagenet100ConceptDataset(
            data_root=args.data_dir,
            json_file=args.concept_file,
            split="test",
        )
        num_classes = train_set.num_classes
        num_attrs = train_set.num_attrs
    elif args.dataset == "awa2":
        train_set = AnimalLoader(
            data_dir=args.data_dir,
            split="train",
            fraction=args.fraction,
            few_shot_train=args.do_train_fewshot,
        )
        val_set = AnimalLoader(
            data_dir=args.data_dir,
            split="val",
            few_shot_train=args.do_train_fewshot,
        )
        test_set = AnimalLoader(
            data_dir=args.data_dir,
            split="test",
            few_shot_train=args.do_train_fewshot,
        )
        num_classes = train_set.num_classes
        num_attrs = train_set.num_attrs
    elif args.dataset == "cifar100":
        train_set = Cifar100Loader(
            data_dir=args.data_dir,
            split="train",
        )
        val_set = Cifar100Loader(
            data_dir=args.data_dir,
            split="val",
        )
        test_set = Cifar100Loader(
            data_dir=args.data_dir,
            split="test",
        )
        num_classes = train_set.num_classes
        num_attrs = train_set.num_attrs
    else:
        raise ValueError("Invalid dataset name")

    if args.model_type == "resnet":
        model = CBM_resnet(args.model_name, num_classes, num_attrs).cuda()
    elif args.model_type == "vit":
        model = CBM_vit(
            model_name=args.model_name, num_classes=num_classes, num_concepts=num_attrs
        ).cuda()

    if args.do_train or args.do_train_fewshot:
        best_model = train_and_validate(args, model, train_set, val_set, logger=logger)

    if args.do_test:
        if args.best_model_path is None and not (
            args.do_train or args.do_train_fewshot
        ):
            model_paths = glob(
                os.path.join(args.save_model_dir, f"cbm_model_epoch_*.pt")
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
            args.do_train or args.do_train_fewshot
        ):
            best_model_path = args.best_model_path
            log_and_print(f"Loading model from {best_model_path}", logger)
            best_model = model.load_state_dict(
                torch.load(best_model_path, weights_only=True), strict=True
            )
        logger.info("========Testing on Test set========")
        loss, test_acc = validate(args, best_model, test_set, logger=logger)
        logger.info(f"Test accuracy: {test_acc}, Loss: {loss}")
        logger.info("Testing complete")
    if args.wandb:
        wandb.finish()
