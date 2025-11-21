import os
from tqdm.auto import tqdm
from collections import OrderedDict, Counter
import pickle
import json

from metric.datasets import *
from metric.measures import *

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.cluster import KMeans, SpectralClustering, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    mutual_info_score,
    adjusted_rand_score,
)


def convert_to_builtin_types(obj):
    if isinstance(obj, dict):
        return {k: convert_to_builtin_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_builtin_types(v) for v in obj]
    elif isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    else:
        return obj


class ClusterPurityMetric:
    def __init__(self, model, layers, is_vit=False, args=None):
        self.model = model
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()
        self.is_vit = is_vit
        self.layers = layers
        if self.is_vit:
            self.layers = [
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
        self.dataset = self.get_dataset(args)
        self.n_clusters = self.dataset.num_classes
        self.purity_fn = self.get_purity_fn(args)
        self.save_path = args.metadata_path
        self.dataloader = DataLoader(
            self.dataset, batch_size=1, shuffle=False, pin_memory=True
        )
        self.gap = nn.AdaptiveAvgPool2d((1, 1))
        self.args = args
        self.current_embeddings = OrderedDict()
        self.layerwise_dataset = {}
        self.clustering_method = self.get_clustering_method(args)
        self.separation_score = self.get_separation_score(args)

    def get_dataset(self, args):
        if args.dataset == "inet100":
            return Inet100Dataset(data_root=args.data_path, split="test")
        elif args.dataset == "cub200":
            return CUB112Dataset(data_root=args.data_path, split="test")
        elif args.dataset == "awa2":
            return AwA2Dataset(data_root=args.data_path, split="test")
        elif args.dataset == "cifar100":
            return Cifar100Dataset(data_root=args.data_path, split="test")
        else:
            raise ValueError("Invalid dataset: {}".format(args.dataset))

    def get_separation_score(self, args):
        if args.separation_score == "silhouette":
            return silhouette_score
        elif args.separation_score == "calinski_harabasz":
            return calinski_harabasz_score
        elif args.separation_score == "davies_bouldin":
            return davies_bouldin_score
        else:
            raise ValueError(
                "Invalid separation score: {}".format(args.separation_score)
            )

    def get_purity_fn(self, args):
        args.purity_fn = args.purity_fn.lower()
        if args.purity_fn == "gini":
            return gini_index
        elif args.purity_fn == "entropy":
            return entropy
        elif args.purity_fn == "impurity":
            return impurity
        else:
            raise ValueError("Invalid purity function: {}".format(args.purity_fn))

    def get_clustering_method(self, args):
        args.clustering_method = args.clustering_method.lower()
        if args.clustering_method == "kmeans":
            return KMeans(
                n_clusters=self.n_clusters,
                random_state=args.seed,
            )
        elif args.clustering_method == "spectral":
            return SpectralClustering(
                n_clusters=self.n_clusters,
                random_state=args.seed,
            )
        elif args.clustering_method == "agglomerative":
            return AgglomerativeClustering(
                n_clusters=self.n_clusters,
            )
        else:
            raise ValueError(
                "Invalid clustering method: {}".format(args.clustering_method)
            )

    def get_gap_hook(self, layer_name):
        def hook(module, input, output):
            if self.is_vit:
                pooled = output[:, 0, :]
            else:
                pooled = (
                    self.gap(output).squeeze(-1).squeeze(-1)
                )  # (1, C, 1, 1) → (1, C)
            self.current_embeddings[layer_name] = pooled.detach().cpu().numpy()

        return hook

    def get_embeddings(self):
        # check if embeddings already exist
        if os.path.exists(
            os.path.join(
                self.save_path,
                f"embeddings_{self.args.dataset}_{self.args.model_name}.pkl",
            )
        ):
            self.load_embeddings()
        else:
            handles = []
            if self.is_vit:
                blocks = getattr(self.model, "blocks")
                for layer_name, block in zip(self.layers, blocks):
                    handle = block.register_forward_hook(self.get_gap_hook(layer_name))
                    handles.append(handle)
            else:
                if "OURS" in self.args.model_name:
                    for layer_name, layer in zip(
                        self.layers, self.model.layers.values()
                    ):
                        handle = layer.register_forward_hook(
                            self.get_gap_hook(layer_name)
                        )
                        handles.append(handle)

                else:
                    for layer_name in self.layers:
                        layer = getattr(self.model, layer_name)
                        handle = layer.register_forward_hook(
                            self.get_gap_hook(layer_name)
                        )
                        handles.append(handle)

            with torch.no_grad():
                for images, labels in tqdm(
                    self.dataloader, desc="Extracting embeddings: ", unit="batch"
                ):
                    images = images.to(self.device)
                    self.current_embeddings.clear()

                    _ = self.model(images)

                    label = labels.item()
                    for layer_name, emb in self.current_embeddings.items():
                        if layer_name not in self.layerwise_dataset:
                            self.layerwise_dataset[layer_name] = {
                                "embeddings": [],
                                "labels": [],
                            }
                        self.layerwise_dataset[layer_name]["embeddings"].append(
                            emb.squeeze(0)
                        )
                        self.layerwise_dataset[layer_name]["labels"].append(label)

            if self.save_path:
                self.save_embeddings()

            for handle in handles:
                handle.remove()

    def save_embeddings(self):
        if self.save_path:
            os.makedirs(self.save_path, exist_ok=True)
            with open(
                os.path.join(
                    self.save_path,
                    f"embeddings_{self.args.dataset}_{self.args.model_name}.pkl",
                ),
                "wb",
            ) as f:
                pickle.dump(self.layerwise_dataset, f)

    def load_embeddings(self):
        with open(
            os.path.join(
                self.save_path,
                f"embeddings_{self.args.dataset}_{self.args.model_name}.pkl",
            ),
            "rb",
        ) as f:
            self.layerwise_dataset = pickle.load(f)

    def compute_metrics(self):

        self.get_embeddings()

        results = {
            "purity": {},
            "separation": {},
            "MI": {},
            "ARI": {},
            "avg_classes_per_cluster": {},
        }
        for layer_name, data in self.layerwise_dataset.items():
            embeddings = data["embeddings"]
            labels = data["labels"]

            clusterer = self.clustering_method.fit(embeddings)
            cluster_labels = clusterer.labels_
            cluster_purity = []
            num_classes_per_cluster = []
            for cluster_id in range(self.n_clusters):
                cluster_indices = np.where(cluster_labels == cluster_id)[0]
                within_cluster_labels = np.array(labels)[cluster_indices]
                n_classes = len(np.unique(within_cluster_labels))
                most_common_label = Counter(within_cluster_labels).most_common(1)[0][0]
                purity = self.purity_fn(within_cluster_labels, most_common_label)
                cluster_purity.append(purity)
                num_classes_per_cluster.append(n_classes)
            results["purity"][layer_name] = np.mean(cluster_purity)
            results["avg_classes_per_cluster"][layer_name] = np.mean(
                num_classes_per_cluster
            )
            results["separation"][layer_name] = self.separation_score(
                embeddings, cluster_labels
            )
            results["MI"][layer_name] = mutual_info_score(labels, cluster_labels)
            results["ARI"][layer_name] = adjusted_rand_score(labels, cluster_labels)
        if self.save_path:
            results = convert_to_builtin_types(results)
            os.makedirs(os.path.join(self.save_path, "jsons"), exist_ok=True)
            with open(
                os.path.join(
                    self.save_path,
                    "jsons",
                    f"metrics_{self.args.dataset}_{self.args.model_name}_{self.args.separation_score}_{self.args.clustering_method}.json",
                ),
                "w",
            ) as f:
                json.dump(results, f, indent=4)
        print("Purity and separation scores saved to:", self.save_path)
        return results


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Cluster Purity Metric")
    parser.add_argument("--dataset", type=str, default="inet100")
    parser.add_argument("--model_name", type=str, default="resnet50")
    parser.add_argument(
        "--data_path",
        type=str,
        default="/raid/DATASETS/inet100",
        help="Path to dataset",
    )
    parser.add_argument(
        "--metadata_path",
        type=str,
        default="./metric_metadata",
    )
    parser.add_argument("--purity_fn", type=str, default="gini")
    args = parser.parse_args()
    args.metadata_path = os.path.join(args.metadata_path, args.dataset)
    os.makedirs(args.metadata_path, exist_ok=True)

    # Example model
    from torchvision.models import resnet50, ResNet50_Weights

    model = resnet50(weights=ResNet50_Weights.DEFAULT)
    layers = ["layer1", "layer2", "layer3", "layer4"]
    metric = ClusterPurityMetric(model, layers, args)
    metric.get_embeddings()
