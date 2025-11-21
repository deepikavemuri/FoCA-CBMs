"""
Utility functions for computing metrics.
"""

import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score
import torch
from torchmetrics import AveragePrecision, CalibrationError


def _roc_auc_score_with_missing(labels, scores):
    # Computes OVR macro-averaged AUROC under missing classes
    aurocs = np.zeros((scores.shape[1],))
    weights = np.zeros((scores.shape[1],))
    for c in range(scores.shape[1]):
        if len(labels[labels == c]) > 0:
            labels_tmp = (labels == c) * 1.0
            aurocs[c] = roc_auc_score(
                labels_tmp, scores[:, c], average="weighted", multi_class="ovr"
            )
            weights[c] = len(labels[labels == c])
        else:
            aurocs[c] = np.NaN
            weights[c] = np.NaN

    # Computing weighted average
    mask = ~np.isnan(aurocs)
    weighted_sum = np.sum(aurocs[mask] * weights[mask])
    average = weighted_sum / len(labels)
    # Regular "macro"
    # average = np.nanmean(aurocs)
    return average


def calc_target_metrics(ys, scores_pred, config, n_decimals=4, n_bins_cal=10):
    """

    :param ys:
    :param scores_pred:
    :param config:
    :return:
    """
    # AUROC
    if config.num_classes == 2:
        auroc = roc_auc_score(ys, scores_pred)
    elif config.num_classes > 2:
        auroc = _roc_auc_score_with_missing(ys, scores_pred)

    # AUPR
    aupr = 0.0
    if config.num_classes == 2:
        aupr = average_precision_score(ys, scores_pred)
    elif config.num_classes > 2:
        ap = AveragePrecision(
            task="multiclass", num_classes=config.num_classes, average="weighted"
        )
        aupr = float(
            ap(torch.tensor(scores_pred), torch.tensor(ys.squeeze()).type(torch.int64))
            .cpu()
            .numpy()
        )

    # Brier score
    if config.num_classes == 2:
        brier = brier_score(ys, np.squeeze(scores_pred))
    else:
        brier = brier_score(ys, scores_pred)

    # ECE
    if config.num_classes == 2:
        ece_fct = CalibrationError(task="binary", n_bins=n_bins_cal, norm="l1")
        tl_ece_fct = CalibrationError(task="binary", n_bins=n_bins_cal, norm="l2")
        ece = float(
            ece_fct(
                torch.tensor(np.squeeze(scores_pred)),
                torch.tensor(ys.squeeze()).type(torch.int64),
            )
            .cpu()
            .numpy()
        )
        tl_ece = float(
            tl_ece_fct(
                torch.tensor(np.squeeze(scores_pred)),
                torch.tensor(ys.squeeze()).type(torch.int64),
            )
            .cpu()
            .numpy()
        )

    else:
        ece_fct = CalibrationError(
            task="multiclass",
            n_bins=n_bins_cal,
            norm="l1",
            num_classes=config.num_classes,
        )
        tl_ece_fct = CalibrationError(
            task="multiclass",
            n_bins=n_bins_cal,
            norm="l2",
            num_classes=config.num_classes,
        )
        ece = float(
            ece_fct(
                torch.tensor(scores_pred), torch.tensor(ys.squeeze()).type(torch.int64)
            )
            .cpu()
            .numpy()
        )
        tl_ece = float(
            tl_ece_fct(
                torch.tensor(scores_pred), torch.tensor(ys.squeeze()).type(torch.int64)
            )
            .cpu()
            .numpy()
        )

    return {
        "AUROC": np.round(auroc, n_decimals),
        "AUPR": np.round(aupr, n_decimals),
        "Brier": np.round(brier, n_decimals),
        "ECE": np.round(ece, n_decimals),
        "TL-ECE": np.round(tl_ece, n_decimals),
    }


def calc_concept_metrics(cs, concepts_pred_probs, config, n_decimals=4, n_bins_cal=10):
    num_concepts = cs.shape[1]

    metrics_per_concept = []

    for j in range(num_concepts):
        # AUROC
        auroc = 0.0
        if len(np.unique(cs[:, j])) == 2:
            auroc = roc_auc_score(
                cs[:, j],
                concepts_pred_probs[j][:, 1],
                average="macro",
                multi_class="ovr",
            )
        elif len(np.unique(cs[:, j])) > 2:
            auroc = roc_auc_score(
                cs[:, j], concepts_pred_probs[j], average="macro", multi_class="ovr"
            )

        # AUPR
        aupr = 0.0
        if len(np.unique(cs[:, j])) == 2:
            aupr = average_precision_score(
                cs[:, j], concepts_pred_probs[j][:, 1], average="macro"
            )
        elif len(np.unique(cs[:, j])) > 2:
            ap = AveragePrecision(
                task="multiclass", num_classes=config.num_classes, average="macro"
            )
            aupr = float(
                ap(torch.tensor(concepts_pred_probs[j]), torch.tensor(cs[:, j]))
                .cpu()
                .numpy()
            )

        # Brier score
        if len(np.unique(cs[:, j])) == 2:
            brier = brier_score(cs[:, j], concepts_pred_probs[j][:, 1])
        else:
            brier = brier_score(cs[:, j], concepts_pred_probs[j])

        # ECE
        ece_fct = CalibrationError(task="binary", n_bins=n_bins_cal, norm="l1")
        tl_ece_fct = CalibrationError(task="binary", n_bins=n_bins_cal, norm="l2")
        # Get the confidence values and targets
        if len(concepts_pred_probs[j].shape) == 1:
            confidences = torch.tensor(concepts_pred_probs[j])
        else:
            confidences = torch.tensor(concepts_pred_probs[j][:, 1])
        
        targets = torch.tensor(cs[:, j].squeeze()).type(torch.int64)
        
        # Validate and clean the data before computing ECE
        # Check for NaN/Inf values
        if torch.isnan(confidences).any() or torch.isinf(confidences).any():
            print(f"Warning: Found NaN/Inf values in concept {j} confidences")
            confidences = torch.nan_to_num(confidences, nan=0.5, posinf=1.0, neginf=0.0)
        
        # Ensure confidences are in [0, 1] range
        confidences = torch.clamp(confidences, min=1e-7, max=1.0 - 1e-7)
        
        # Check for valid targets (should be 0 or 1 for binary)
        if targets.min() < 0 or targets.max() > 1:
            print(f"Warning: Invalid target values for concept {j}, range: [{targets.min()}, {targets.max()}]")
            targets = torch.clamp(targets, min=0, max=1)
        
        # Skip ECE computation if we have empty tensors
        if confidences.numel() == 0 or targets.numel() == 0:
            print(f"Warning: Empty tensors for concept {j}, setting ECE to 0")
            ece = 0.0
            tl_ece = 0.0
        else:
            try:
                ece = float(ece_fct(confidences, targets).cpu().numpy())
                tl_ece = float(tl_ece_fct(confidences, targets).cpu().numpy())
            except Exception as e:
                print(f"Error computing ECE for concept {j}: {e}")
                print(f"Confidences shape: {confidences.shape}, range: [{confidences.min():.4f}, {confidences.max():.4f}]")
                print(f"Targets shape: {targets.shape}, unique values: {torch.unique(targets)}")
                # Set to 0 if computation fails
                ece = 0.0
                tl_ece = 0.0

        metrics_per_concept.append(
            {
                "AUROC": np.round(auroc, n_decimals),
                "AUPR": np.round(aupr, n_decimals),
                "Brier": np.round(brier, n_decimals),
                "ECE": np.round(ece, n_decimals),
                "TL-ECE": np.round(tl_ece, n_decimals),
            }
        )

    auroc = 0.0
    aupr = 0.0
    brier = 0.0
    ece = 0.0
    tl_ece = 0.0
    for j in range(num_concepts):
        auroc += metrics_per_concept[j]["AUROC"]
        aupr += metrics_per_concept[j]["AUPR"]
        brier += metrics_per_concept[j]["Brier"]
        ece += metrics_per_concept[j]["ECE"]
        tl_ece += metrics_per_concept[j]["TL-ECE"]
    auroc /= num_concepts
    aupr /= num_concepts
    brier /= num_concepts
    ece /= num_concepts
    tl_ece /= num_concepts
    metrics_overall = {
        "AUROC": np.round(auroc, n_decimals),
        "AUPR": np.round(aupr, n_decimals),
        "Brier": np.round(brier, n_decimals),
        "ECE": np.round(ece, n_decimals),
        "TL-ECE": np.round(tl_ece, n_decimals),
    }

    return metrics_overall, metrics_per_concept


def brier_score(y_true, y_prob):
    # NOTE:
    # - for multiclass, @y_true must be of dimensionality (n_samples, ) and @y_prob must be (n_samples, n_classes)
    # - for binary, @y_true must be of dimensionality (n_samples, ) and @y_prob must be (n_samples, )

    if len(y_prob.shape) == 2:
        # NOTE: we use the original definition by Brier for categorical variables
        # See the original paper by Brier https://doi.org/10.1175/1520-0493(1950)078<0001:VOFEIT>2.0.CO;2
        sc = 0
        for j in range(y_prob.shape[1]):
            # sc += np.sum(((y_true == j) * 1. - y_prob[j])**2)
            # Correction to multiclass
            sc += np.sum((np.squeeze((y_true == j) * 1.0) - y_prob[:, j]) ** 2)
        sc /= y_true.shape[0]
        return sc
    elif len(y_prob.shape) == 1:
        return np.mean((y_prob - y_true) ** 2)
