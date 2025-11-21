import torch
import torch.nn.functional as F

from torch.utils.data import DataLoader
from tqdm import tqdm
import sys
import os
import glob
import wandb
from processing.utils import MetricCalculator
from utils import *
import heapq
import time


def get_cls_probabilities(class_preds, num_clfs):
    for num_clf in range(num_clfs - 1):
        class_preds[num_clf] = F.sigmoid(class_preds[num_clf])
    class_preds[-1] = F.softmax(class_preds[-1], dim=-1)
    return class_preds


@torch.inference_mode()
def validate(epoch, args, val_dataset, model, num_clfs, logger=None, val=True):

    valloader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=False,
        shuffle=False,
    )

    model.eval()
    cls_acc, attr_acc = [0] * num_clfs, [0] * num_clfs
    metric_calc_01 = MetricCalculator(num_clfs)
    loss = 0
    header = (
        f"Val: Epoch {epoch} - Batch Progress" if val else f"Test: - Batch Progress"
    )
    for i, data in enumerate(tqdm(valloader, desc=header)):
        cls_loss, attr_loss = [], []
        _, imgs, cls, attrs_present, classes_present = data
        attr_preds, class_preds = model(imgs.cuda())
        # convert class_preds to probabilities
        class_preds = get_cls_probabilities(class_preds, num_clfs)

        # calculate loss and accuracy
        cls_loss, attr_loss, cls_acc, attr_acc = calculate_loss_and_accuracy(
            attr_preds,
            class_preds,
            attrs_present,
            classes_present,
            num_clfs,
            imgs.shape[0],
            cls_acc,
            attr_acc,
            args,
        )
        metric_calc_01.update(class_preds, attr_preds, classes_present, attrs_present)

        loss += (
            cls_loss[-1]
            + args.concept_wts * sum(attr_loss)
            + args.cls_wts * sum(cls_loss[:-1])
        )
    loss /= len(valloader)
    cls_01_acc, attr_01_acc = metric_calc_01.calculate_accuracy()
    cls_auc, attr_auc = metric_calc_01.calculate_auc()

    complete_logging(
        epoch,
        logger,
        args,
        num_clfs,
        valloader,
        loss.item(),
        cls_acc,
        attr_acc,
        cls_01_acc,
        attr_01_acc,
        cls_auc,
        attr_auc,
        mode="val" if val else "test",
    )

    return sum(cls_acc) / (len(valloader) * num_clfs)


def train_and_validate(
    args, train_dataset, val_dataset, test_dataset, model, num_clfs=1, logger=None
):
    if args.wandb:
        wandb.init(project="fca_intsem", entity="user", config=args)
        wandb.run.name = f"fca4nn_{args.dataset}_{wandb.run.name}"

        wandb.watch(model, log="all", log_freq=args.verbose)
        wandb.config.command = " ".join(sys.argv)
        wandb.config.update(
            {
                "lr": args.lr,
                "epochs": args.epochs,
                "num_clfs": args.num_clfs,
                "lattice_levels": args.lattice_levels,
                "backbone_layer_ids": args.backbone_layer_ids,
            }
        )
    if args.do_train_full or args.do_train_attrs or args.do_train_fewshot:
        top_checkpoints = []  # Min-heap to track top 5 (val_acc, path)

        trainloader = DataLoader(
            train_dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=True,
            drop_last=False,
        )

        opt = get_optimizer(model, args)

        best_val_acc = 0

        scheduler = get_scheduler(opt, args, trainloader)

        metric_caculator = MetricCalculator(num_clfs)

        if args.sequential_training:
            # Freeze clf2 for a couple of epochs
            for param in model.classifier_layers[1].parameters():
                param.requires_grad = False

        for epoch in range(args.epochs):
            model.train()

            if args.sequential_training and epoch == args.epochs // 2:
                for param in model.classifier_layers[1].parameters():
                    param.requires_grad = True

            # Freeze the first classifier once half the epochs are done
            # if epoch == args.epochs // 2:
            #     for param in model.classifier_layers[0].parameters():
            #         param.requires_grad = False

            metric_caculator.reset()
            cls_acc, attr_acc = [0] * num_clfs, [0] * num_clfs
            loss, running_loss = 0, 0
            for idx, data in enumerate(
                tqdm(trainloader, desc=f"Train: Epoch {epoch} - Batch Progress")
            ):
                # cls_loss, attr_loss = [], []
                _, imgs, cls, attrs_present, classes_present = data
                opt.zero_grad()
                attr_preds, class_preds = model(imgs.cuda())
                # convert class_preds to probabilities
                class_preds = get_cls_probabilities(class_preds, num_clfs)
                # calculate loss and accuracy
                cls_loss, attr_loss, cls_acc, attr_acc = calculate_loss_and_accuracy(
                    attr_preds,
                    class_preds,
                    attrs_present,
                    classes_present,
                    num_clfs,
                    imgs.shape[0],
                    cls_acc,
                    attr_acc,
                    args,
                )
                metric_caculator.update(
                    class_preds, attr_preds, classes_present, attrs_present
                )

                if args.do_train_attrs:
                    loss = args.concept_wts * sum(attr_loss)
                else:
                    if args.sequential_training and epoch < args.epochs // 2:
                        loss = cls_loss[0] + args.concept_wts * attr_loss[0]
                    else:
                        loss = (
                            cls_loss[-1]
                            + args.concept_wts * sum(attr_loss)
                            + args.cls_wts * sum(cls_loss[:-1])
                        )  # weighting all the bce loss values with concept_weights
                if args.clf_l1_reg:
                    loss += model.clf_weight_l1_regularize()
                loss.backward()
                # gradient clipping
                if args.grad_clip:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                opt.step()
                running_loss += loss.item()

                scheduler.step()
                curr_lr = scheduler.get_last_lr()[0]
                if args.wandb:
                    wandb.log({"lr": curr_lr})
                # based on args.verbose print loss and log it
                if args.verbose is not None:
                    if idx % args.verbose == 0:
                        if args.do_train_attrs:
                            log_and_print(
                                f"Epoch {epoch} - Batch {idx}: Lr={curr_lr}, Loss={loss.item()}, Attribute Losses={[alos.item() for alos in attr_loss]}",
                                logger,
                            )
                        else:
                            log_and_print(
                                f"Epoch {epoch} - Batch {idx}: Lr={curr_lr}, Loss={loss.item()}, Classifier Losses={[clos.item() for clos in cls_loss]}, Attribute Losses={[alos.item() for alos in attr_loss]}",
                                logger,
                            )
                        if args.wandb:
                            for i in range(num_clfs):
                                wandb.log(
                                    {
                                        f"train_clf_{i}_loss": cls_loss[i].item(),
                                        f"train_attr_{i}_loss": attr_loss[i].item(),
                                    }
                                )
            running_loss /= len(trainloader)
            cls_01_acc, attr_01_acc = metric_caculator.calculate_accuracy()
            cls_auc, attr_auc = metric_caculator.calculate_auc()

            complete_logging(
                epoch,
                logger,
                args,
                num_clfs,
                trainloader,
                running_loss,
                cls_acc,
                attr_acc,
                cls_01_acc,
                attr_01_acc,
                cls_auc,
                attr_auc,
                mode="train",
            )

            if epoch % args.validation_freq == 0 or epoch == args.epochs - 1:
                # validate the model
                val_acc = validate(epoch, args, val_dataset, model, num_clfs, logger)
                if args.do_train_attrs:
                    run_name = f"attrs_{num_clfs}_level_{epoch}_{val_acc:.4f}.pt"
                if args.do_train_full:
                    run_name = f"intsem_{num_clfs}_level_{epoch}_{val_acc:.4f}.pt"
                if args.do_train_fewshot:
                    run_name = (
                        f"intsem_fewshot_{num_clfs}_level_{epoch}_{val_acc:.4f}.pt"
                    )
                model_path = os.path.join(
                    args.save_model_dir,
                    run_name,
                )
                torch.save(model.state_dict(), model_path)
                heapq.heappush(top_checkpoints, (val_acc, -epoch, model_path))

                # Keep only top k; remove the lowest if more
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

                log_and_print(
                    f"Model saved at epoch {epoch} with val acc {val_acc} at {model_path}",
                    logger,
                )

    if args.do_test:

        time.sleep(10)

        log_and_print("========Testing on Test set========", logger)
        if args.best_model_path is not None:
            best_model_path = args.best_model_path
        else:
            if args.do_train_attrs:
                run_name = f"attrs_{num_clfs}_level_*.pt"
            if args.do_train_full:
                run_name = f"intsem_{num_clfs}_level_*.pt"
            if args.do_train_fewshot:
                run_name = f"intsem_fewshot_{num_clfs}_level_*.pt"
            model_paths = glob.glob(os.path.join(args.save_model_dir, run_name))
            model_paths.sort(
                key=lambda x: float(x.split("_")[-1].split(".")[1]), reverse=True
            )
            best_model_path = model_paths[0]
            log_and_print(f"Best model saved at {best_model_path}", logger)
        log_and_print(f"Loading model from {best_model_path}", logger)
        model.load_state_dict(
            torch.load(best_model_path, weights_only=True), strict=True
        )
        test_acc = validate(
            epoch, args, test_dataset, model, num_clfs, logger, val=False
        )
        log_and_print(f"Test accuracy: {test_acc}", logger)
    if args.wandb:
        wandb.finish()
