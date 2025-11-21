# -*- coding: utf-8 -*-
import torch
import wandb
import torch.nn as nn
from torch.optim import AdamW, Adam, SGD
from torch.optim.lr_scheduler import (
    LambdaLR,
    CyclicLR,
    CosineAnnealingLR,
    StepLR,
    LinearLR,
    ConstantLR,
)

from losses import *


def get_scheduler(opt, args, trainloader):
    if args.scheduler == "onecycle":
        scheduler = CyclicLR(
            opt,
            base_lr=1e-5,
            max_lr=args.lr,
            step_size_up=2 * len(trainloader),
            mode="triangular2",
        )
    elif args.scheduler == "cosine":
        scheduler = CosineAnnealingLR(
            opt, T_max=args.epochs * len(trainloader), eta_min=1e-5
        )
    elif args.scheduler == "step":
        scheduler = StepLR(opt, step_size=10, gamma=0.1)
    elif args.scheduler == "linear":
        scheduler = LinearLR(
            opt,
            start_factor=1.0,
            end_factor=0.1,
            total_iters=args.epochs * len(trainloader),
        )
    elif args.scheduler == "constant":
        scheduler = ConstantLR(
            opt, factor=1, total_iters=args.epochs * len(trainloader), last_epoch=-1
        )
    elif args.scheduler == "none":
        scheduler = LambdaLR(opt, lr_lambda=lambda epoch: 1)
    else:
        raise ValueError(f"Unknown scheduler {args.scheduler}")
    return scheduler


def get_optimizer(model, args):
    if args.varying_lrs:
        opt = AdamW(
            [
                {"params": model.layers.parameters(), "lr": args.lr},
                {
                    "params": [
                        *list(model.attr_layers[0].parameters()),
                        *list(model.attr_bn_layers[0].parameters()),
                        *list(model.classifier_layers[0].parameters()),
                    ],
                    "lr": args.lr,
                },
                {
                    "params": [
                        *list(model.attr_layers[1].parameters()),
                        *list(model.attr_bn_layers[1].parameters()),
                        *list(model.classifier_layers[1].parameters()),
                    ],
                    "lr": args.lr * 10,
                },
            ]
        )
    elif args.opt == "adamw":
        opt = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "adam":
        opt = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    elif args.opt == "sgd":
        opt = SGD(
            model.parameters(),
            lr=args.lr,
            momentum=args.momentum,
            weight_decay=args.weight_decay,
        )
    return opt


def log_and_print(message, logger=None):
    if logger is not None:
        logger.info(message)
    else:
        print(message)


def complete_logging(
    epoch,
    logger,
    args,
    num_clfs,
    loader,
    loss,
    cls_acc,
    attr_acc,
    cls_01_acc,
    attr_01_acc,
    cls_auc,
    attr_auc,
    mode="train",
):
    if mode == "train":
        logger_name = "Training"
        wandb_name = "train"
        log_and_print(f"Epoch {epoch} Training Completed. Total Loss={loss}", logger)
        if args.wandb:
            wandb.log({"epoch": epoch, "loss": loss})
    elif mode == "val":
        logger_name = "Validation"
        wandb_name = "val"
        log_and_print(f"Validation Loss: {loss}", logger)
        if args.wandb:
            wandb.log({f"val_loss": loss})
    elif mode == "test":
        logger_name = "Test"
        wandb_name = "test"
        log_and_print(f"Test Loss: {loss}", logger)
        if args.wandb:
            wandb.log({f"test_loss": loss})
    else:
        raise ValueError(f"Unknown mode {mode}")

    for i in range(num_clfs):
        log_and_print(
            f"{logger_name} - Classifier {i}: Acc={cls_acc[i] / len(loader)}, Attr Acc={attr_acc[i] / len(loader)}",
            logger,
        )
        if args.wandb:
            wandb.log(
                {
                    f"{wandb_name}_clf_{i}_acc": cls_acc[i] / len(loader),
                    f"{wandb_name}_attr_{i}_acc": attr_acc[i] / len(loader),
                }
            )
    for i in range(num_clfs - 1):
        log_and_print(
            f"{logger_name} - Classifier {i}: 0s Acc={cls_01_acc['0'][i]}, 1s Acc={cls_01_acc['1'][i]}, AUC={cls_auc[i]}",
            logger,
        )
        log_and_print(
            f"{logger_name} - Attributes {i}: 0s Acc={attr_01_acc['0'][i]}, 1s Acc={attr_01_acc['1'][i]}, AUC={attr_auc[i]}",
            logger,
        )
        if args.wandb:
            wandb.log(
                {
                    f"{wandb_name}_clf_{i}_0s_acc": cls_01_acc["0"][i],
                    f"{wandb_name}_clf_{i}_1s_acc": cls_01_acc["1"][i],
                }
            )
            wandb.log(
                {
                    f"{wandb_name}_attr_{i}_0s_acc": attr_01_acc["0"][i],
                    f"{wandb_name}_attr_{i}_1s_acc": attr_01_acc["1"][i],
                }
            )
    log_and_print(
        f"{logger_name} - Attributes {num_clfs-1}: 0s Acc={attr_01_acc['0'][num_clfs-1]}, 1s Acc={attr_01_acc['1'][num_clfs-1]}, AUC={attr_auc[num_clfs-1]}",
        logger,
    )
    if args.wandb:
        wandb.log(
            {
                f"{wandb_name}_attr_{num_clfs-1}_0s_acc": attr_01_acc["0"][
                    num_clfs - 1
                ],
                f"{wandb_name}_attr_{num_clfs-1}_1s_acc": attr_01_acc["1"][
                    num_clfs - 1
                ],
            }
        )


def get_loss(loss_name):
    if loss_name == "bce":
        return binary_crossentropy
    elif loss_name == "focal":
        return focal_loss
    elif loss_name == "hierarchical_ce":
        return hierarchical_ce_loss
    elif loss_name == "dice":
        return dice_loss
    elif loss_name == "bce_with_logits":
        return nn.BCEWithLogitsLoss(pos_weight=2.0)
    else:
        return nn.CrossEntropyLoss()


def calculate_loss_and_accuracy(
    attr_preds,
    class_preds,
    attrs_present,
    classes_present,
    num_clfs,
    batch_size,
    cls_acc,
    attr_acc,
    args,
):
    attr_loss_fn = get_loss(args.attr_loss)
    pre_final_clf_loss = get_loss(args.pre_final_clf_loss)
    final_clf_loss = get_loss(args.final_clf_loss)

    cls_loss, attr_loss = [], []
    for i in range(num_clfs - 1):
        cls_loss.append(
            pre_final_clf_loss(
                class_preds[i], classes_present[i].to(torch.float32).cuda()
            ).mean()
        )
        attr_loss.append(
            attr_loss_fn(
                attr_preds[i], attrs_present[i].to(torch.float32).cuda()
            ).mean()
        )
    cls_loss.append(final_clf_loss(class_preds[-1], classes_present[-1].cuda()).mean())
    attr_loss.append(
        attr_loss_fn(attr_preds[-1], attrs_present[-1].to(torch.float32).cuda()).mean()
    )

    for i in range(num_clfs - 1):
        classification_accuracy = (
            sum(torch.round(class_preds[i].cpu()) == classes_present[i]) / batch_size
        )
        cls_acc[i] += classification_accuracy.detach().numpy().mean().item()
        attribute_accuracy = (
            sum(torch.round(attr_preds[i].cpu()) == attrs_present[i]) / batch_size
        )
        attr_acc[i] += attribute_accuracy.detach().numpy().mean().item()
    classification_accuracy = (
        sum(
            torch.argmax(class_preds[-1].cpu(), dim=-1)
            == torch.argmax(classes_present[-1], dim=-1)
        )
        / batch_size
    )
    cls_acc[-1] += classification_accuracy.detach().numpy().item()
    attribute_accuracy = (
        sum(torch.round(attr_preds[-1].cpu()) == attrs_present[-1]) / batch_size
    )
    attr_acc[-1] += attribute_accuracy.detach().numpy().mean().item()
    return cls_loss, attr_loss, cls_acc, attr_acc
