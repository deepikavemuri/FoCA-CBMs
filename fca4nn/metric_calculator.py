import argparse
import torch
from torchvision.models import resnet50, ResNet50_Weights, resnet18, ResNet18_Weights
from model import (
    CBM_resnet,
    FoCA_CBM_resnet,
    MLPCBM,
    FoCA_CBM_N_resnet,
    CBM_vit,
    FoCA_CBM_vit,
)
import clip
import os
from metric.calculator import ClusterPurityMetric
from collections import OrderedDict
from processing.utils import get_info_from_lattice
from timm import create_model


def get_dataset_info(dataset_name):
    if dataset_name == "inet100":
        return 700, 100, 400
    if dataset_name == "cifar100":
        return 700, 100, 500
    elif dataset_name == "cub200":
        return 112, 200, 80
    elif dataset_name == "awa2":
        return 85, 50, 60
    else:
        raise ValueError("Invalid dataset name: {}".format(dataset_name))


def main(args):
    # get model seperations
    base, model_name = args.model_name.split("::")
    if base == "CLIP":
        model, _ = clip.load(model_name, device="cpu", download_root=args.metadata_path)
        model = model.visual
        layers = ["layer1", "layer2", "layer3", "layer4"]
    elif base == "PYTORCH":
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        layers = ["layer1", "layer2", "layer3", "layer4"]
    elif base == "VIT":
        model = create_model(model_name, pretrained=True)
        layers = [
            "blocks.0",
            "blocks.1",
            "blocks.2",
            "blocks.3",
            "blocks.4",
            "blocks.5",
            "blocks.6",
            "blocks.7",
            "blocks.8",
            "blocks.9",
            "blocks.10",
            "blocks.11",
        ]
    elif base == "CEM":
        weights = torch.load(args.model_weights)
        new_weights = OrderedDict()
        for k, v in weights.items():
            if "pre_concept_model" in k:
                new_weights[k.replace("pre_concept_model.", "")] = v
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.load_state_dict(new_weights, strict=True)
        layers = ["layer1", "layer2", "layer3", "layer4"]
    elif base == "CBM":
        n_attr, n_classes, _ = get_dataset_info(args.dataset)
        if "vit" in model_name:
            model = CBM_vit(
                num_concepts=n_attr, num_classes=n_classes, model_name=model_name
            )
            model.load_state_dict(torch.load(args.model_weights), strict=True)
            model = model.backbone.vit
            layers = [
                "blocks.0",
                "blocks.1",
                "blocks.2",
                "blocks.3",
                "blocks.4",
                "blocks.5",
                "blocks.6",
                "blocks.7",
                "blocks.8",
                "blocks.9",
                "blocks.10",
                "blocks.11",
            ]
        else:
            model = CBM_resnet(
                model_name=model_name, num_attrs=n_attr, num_classes=n_classes
            )
            model.load_state_dict(torch.load(args.model_weights), strict=True)
            model = model.model
            layers = ["layer1", "layer2", "layer3", "layer4"]
    elif base == "SCBM":
        weights = torch.load(args.model_weights)
        new_weights = OrderedDict()
        for k, v in weights.items():
            if k.startswith("encoder_res."):
                new_weights[k.replace("encoder_res.", "")] = v
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        model.load_state_dict(new_weights, strict=False)
        layers = ["layer1", "layer2", "layer3", "layer4"]
    elif base == "PCBM":
        weights = torch.load(args.model_weights)
        new_weights = OrderedDict()
        for k, v in weights.items():
            if k.startswith("x2c_model."):
                new_weights[k.replace("x2c_model.", "")] = v
        if model_name == "resnet50":
            model = resnet50(weights=ResNet50_Weights.DEFAULT)
        elif model_name == "resnet18":
            model = resnet18(weights=ResNet18_Weights.DEFAULT)
        del new_weights["fc.weight"]
        del new_weights["fc.bias"]
        model.load_state_dict(new_weights, strict=False)
        layers = ["layer1", "layer2", "layer3", "layer4"]
    elif base == "MLPCBM":
        n_attr, n_classes, expand_dim = get_dataset_info(args.dataset)
        model = MLPCBM(
            model_name=model_name,
            num_attrs=n_attr,
            num_classes=n_classes,
            expand_dim=expand_dim,
        )
        model.load_state_dict(torch.load(args.model_weights), strict=True)
        model = model.model
        layers = ["layer1", "layer2", "layer3", "layer4"]
    elif (
        base == "OURS-1FCA"
        or base == "OURS-2FCA"
        or base == "OURS-3FCA"
        or base == "MCLCBM"
    ):
        n_attr, n_classes, _ = get_dataset_info(args.dataset)
        perlevel_intents, perlevel_fcs = get_info_from_lattice(
            args.lattice_path, args.lattice_levels
        )
        if base == "MCLCBM":
            model = FoCA_CBM_N_resnet(
                intent_list=perlevel_intents,
                fc_list=perlevel_fcs,
                num_classes=n_classes,
                backbone_name=model_name,
            )
            model.load_state_dict(torch.load(args.model_weights), strict=True)
            model = model.model
            layers = ["layer1", "layer2", "layer3", "layer4"]
        if "vit" in model_name:
            model = FoCA_CBM_vit(
                intent_list=perlevel_intents,
                fc_list=perlevel_fcs,
                backbone_layer_ids=args.backbone_layer_ids,
                num_classes=n_classes,
                model_name=model_name,
            )
            model.load_state_dict(torch.load(args.model_weights), strict=True)
            layers = [
                "blocks.0",
                "blocks.1",
                "blocks.2",
                "blocks.3",
                "blocks.4",
                "blocks.5",
                "blocks.6",
                "blocks.7",
                "blocks.8",
                "blocks.9",
                "blocks.10",
                "blocks.11",
            ]
            model = model.backbone.vit
        else:
            model = FoCA_CBM_resnet(
                intent_list=perlevel_intents,
                fc_list=perlevel_fcs,
                backbone_layer_ids=args.backbone_layer_ids,
                num_classes=n_classes,
                backbone_name=model_name,
            )
            model.load_state_dict(torch.load(args.model_weights), strict=True)
            layers = ["layer1", "layer2", "layer3", "layer4"]
    else:
        raise ValueError("Invalid model name: {}".format(args.model_name))

    metric_calc = ClusterPurityMetric(
        model=model,
        layers=layers,
        is_vit="vit" in model_name or "deit" in model_name,
        args=args,
    )
    metric_calc.compute_metrics()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cluster Purity Metric Evaluation")

    # General
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["inet100", "cub200", "awa2", "cifar100"],
        help="Dataset to use",
    )
    parser.add_argument(
        "--data_path",
        type=str,
        required=True,
        help="Path to dataset root directory",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default=None,
        help="Path to save extracted embeddings and computed metrics",
    )

    # Clustering
    parser.add_argument(
        "--clustering_method",
        type=str,
        default="kmeans",
        choices=["kmeans", "spectral", "agglomerative"],
        help="Clustering algorithm to use",
    )
    parser.add_argument(
        "--seed", type=int, default=42, help="Random seed for clustering"
    )

    # Purity function
    parser.add_argument(
        "--purity_fn",
        type=str,
        default="gini",
        choices=["gini", "entropy", "impurity"],
        help="Function to compute purity per cluster",
    )

    # Separation score
    parser.add_argument(
        "--separation_score",
        type=str,
        default="silhouette",
        choices=["silhouette", "calinski_harabasz", "davies_bouldin"],
        help="Metric to evaluate cluster separation",
    )

    # Model Extra Args
    parser.add_argument(
        "--model_name",
        type=str,
        required=True,
        help="Name of the model (used in saving embeddings/metrics)",
    )
    parser.add_argument(
        "--model_weights",
        type=str,
        default=None,
        help="Path to model weights (if not using default)",
    )
    parser.add_argument(
        "--lattice_path",
        type=str,
        default="./data/lattices/",
        help="Path to lattice file",
    )
    parser.add_argument(
        "--lattice_levels",
        nargs="+",
        type=int,
        help="Choose lattice levels",
    )
    parser.add_argument(
        "--backbone_layer_ids",
        nargs="+",
        type=int,
        help="Choose where to place intermediate semantic layers",
    )

    args = parser.parse_args()
    args.metadata_path = os.path.join(args.metadata_path, args.dataset)
    os.makedirs(args.metadata_path, exist_ok=True)

    main(args)
