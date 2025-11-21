import pandas as pd
import json
import numpy as np
import pickle
import torch
import os
from collections import defaultdict
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    jaccard_score,
    matthews_corrcoef,
    hamming_loss,
    roc_auc_score,
)
import networkx as nx
import copy

from processing.imagenet10_classes import IMAGENET100_CLASS2ID
from collections import defaultdict

NUM_CLASSES = 100


# Also taken from the official CBM codebase
def create_new_dataset(
    out_dir, field_change, compute_fn, datasets=["train", "val", "test"], data_dir=""
):
    """
    Generic function that given datasets stored in data_dir, modify/ add one field of the metadata in each dataset based on compute_fn
                          and save the new datasets to out_dir
    compute_fn should take in a metadata object (that includes 'img_path', 'class_label', 'attribute_label', etc.)
                          and return the updated value for field_change
    """
    if not os.path.exists(out_dir):
        os.mkdir(out_dir)
    mask = np.array(
        [
            1,
            4,
            6,
            7,
            10,
            14,
            15,
            20,
            21,
            23,
            25,
            29,
            30,
            35,
            36,
            38,
            40,
            44,
            45,
            50,
            51,
            53,
            54,
            56,
            57,
            59,
            63,
            64,
            69,
            70,
            72,
            75,
            80,
            84,
            90,
            91,
            93,
            99,
            101,
            106,
            110,
            111,
            116,
            117,
            119,
            125,
            126,
            131,
            132,
            134,
            145,
            149,
            151,
            152,
            153,
            157,
            158,
            163,
            164,
            168,
            172,
            178,
            179,
            181,
            183,
            187,
            188,
            193,
            194,
            196,
            198,
            202,
            203,
            208,
            209,
            211,
            212,
            213,
            218,
            220,
            221,
            225,
            235,
            236,
            238,
            239,
            240,
            242,
            243,
            244,
            249,
            253,
            254,
            259,
            260,
            262,
            268,
            274,
            277,
            283,
            289,
            292,
            293,
            294,
            298,
            299,
            304,
            305,
            308,
            309,
            310,
            311,
        ]
    )

    for dataset in datasets:
        path = os.path.join(data_dir, dataset + ".pkl")
        if not os.path.exists(path):
            continue
        data = pickle.load(open(path, "rb"))
        new_data = []
        for d in data:
            new_d = copy.deepcopy(d)

            if field_change in new_d:
                old_value = d[field_change]
                new_value = list((np.array(old_value)[mask - 1]))
                assert type(old_value) == type(new_value)
            new_d[field_change] = new_value
            new_data.append(new_d)

        f = open(os.path.join(out_dir, dataset + ".pkl"), "wb")
        pickle.dump(new_data, f)
        f.close()


def attr_id_to_name():
    file_path = "./CUB_200_2011/cub112.json"

    attr_id_to_name = {}
    with open(
        "./CUB_200_2011/filtered_attributes.txt", "r"
    ) as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                attr_id = int(parts[0])
                attr_id_to_name[attr_id] = parts[1]

    with open(file_path, "r") as f:
        class_to_attr_ids = json.load(f)

    class_to_attr_names = {
        cls_name: [
            attr_id_to_name[attr_id]
            for attr_id in attr_ids
            if attr_id in attr_id_to_name
        ]
        for cls_name, attr_ids in class_to_attr_ids.items()
    }

    with open(file_path, "w") as f:
        json.dump(class_to_attr_names, f, indent=2)


def change_class_to_id(
    class_level_json_path="./data/inet100/imagenet100.json",
    class_level_output_json_path="./data/inet100/imagenet100_id2cons.json",
):
    with open(class_level_json_path, "r") as file:
        json_data = json.load(file)

    updated_json = defaultdict(list)
    c1, c2 = 1, 1
    for key, concepts in json_data.items():
        # print(key)
        for class_name in IMAGENET100_CLASS2ID.keys():
            if key in class_name:
                # print("Found: ", class_name, IMAGENET100_CLASS2ID[class_name])
                updated_json[IMAGENET100_CLASS2ID[class_name]] = concepts
                # print("Len: ", len(updated_json.keys()))
                break
    # print(len(updated_json.keys()))
    with open(class_level_output_json_path, "w") as file:
        json.dump(updated_json, file, indent=4)


def get_fc2attrmatrix(fc, attr_l):
    # Convert attr_l to a dict where each attribute is a key and its index is the value
    attr_dict = {attr: i for i, attr in enumerate(attr_l)}
    fc2attrmatrix = torch.zeros((len(fc), len(attr_dict.keys())), dtype=float)
    for con_id, concept in enumerate(fc):
        for elem in concept.intent:
            fc2attrmatrix[con_id, attr_dict[elem]] = 1
    return fc2attrmatrix


def get_info_from_lattice(lattice_path, lattice_levels, make_exclusive=False):
    def toposort_lattice(lattice):
        G = nx.DiGraph()
        for concept in lattice:
            G.add_node(concept)
            for upper in concept.upper_neighbors:
                G.add_edge(concept, upper)
        sorted_concepts = list(nx.topological_sort(G))
        return sorted_concepts

    def compute_levels(concepts):
        level = {}
        for concept in concepts:
            extent, intent = concept
            if str(extent) not in level:
                level[str(extent)] = 0
            for upper in concept.upper_neighbors:
                upper_extent, upper_intent = upper
                if str(upper_extent) not in level:
                    level[str(upper_extent)] = level[str(extent)] + 1
                else:
                    level[str(upper_extent)] = max(
                        level[str(upper_extent)], level[str(extent)] + 1
                    )
        return level

    def compute_hierarchy(concepts, level):
        hierarchy = {}
        for concept in concepts:
            extent, intent = concept
            lvl = level[str(extent)]
            if lvl in hierarchy:
                hierarchy[lvl].append(concept)
            else:
                hierarchy[lvl] = [concept]
        return hierarchy

    lattice_levels = sorted(lattice_levels, reverse=True)
    file_extension = os.path.splitext(lattice_path)[1]

    perlevel_intents = [[] for _ in range(len(lattice_levels))]
    perlevel_fcs = [[] for _ in range(len(lattice_levels))]

    if file_extension == ".json":
        with open(lattice_path, "r") as f:
            lattice = json.load(f)
        maxlevel = int(max([int(i) for i in lattice["hierarchy"].keys()]))
        print("maxlevel: ", maxlevel)

        for fc_id in lattice["hierarchy"][str(maxlevel - 3)].keys():
            l1.extend(lattice["hierarchy"][str(maxlevel - 3)][str(fc_id)]["intent"])
        for fc_id in lattice["hierarchy"]["3"].keys():
            l2.extend(lattice["hierarchy"]["3"][str(fc_id)]["intent"])

        l1 = list(set(l1))
        l2 = list(set(l2))

    elif file_extension == ".pkl":
        with open(lattice_path, "rb") as pkl:
            context = pickle.load(pkl)
            lattice = context.lattice

        sorted_concepts = toposort_lattice(lattice)
        level = compute_levels(sorted_concepts)
        hierarchy = compute_hierarchy(sorted_concepts, level)

        for i in range(len(lattice_levels)):
            # print(lattice_levels[i])
            for concept in hierarchy[lattice_levels[i]]:
                perlevel_intents[i].extend(concept.intent)
                perlevel_fcs[i].append(concept)
            perlevel_intents[i] = set(perlevel_intents[i])
        
        # Making the attribute sets exclusive
        if make_exclusive:
            for i, attrs in enumerate(perlevel_intents[1:], start=1):
                for j in range(0, i):
                    perlevel_intents[i] = perlevel_intents[i].difference(perlevel_intents[j])

        # Make everything a list
        for i in range(len(perlevel_intents)):
            perlevel_intents[i] = list(perlevel_intents[i])
            print(len(perlevel_intents[i]))
    return perlevel_intents, perlevel_fcs


def split_csv():
    df = pd.read_csv(
        "./inet100/inet100_formal_concepts_grouped_bin.csv"
    )

    # df['id'] = range(1, len(df) + 1)
    # df_fc = df[['id', 'intent', 'extent', 'level']]
    # df_fc.to_csv('./data/inet100/inet100_formal_concepts.csv', index=False)

    # Group by `class` and create a list of `id`s for each class
    class2ids = (
        df.groupby("class")["fc_id"].apply(list).reset_index().to_dict(orient="records")
    )

    with open(
        "./data/inet100/inet100_fc2class.json", "w"
    ) as f:
        json.dump(class2ids, f, indent=4)

    print("Splitting and saving to JSON files completed successfully!")


def clean_csv():
    df = pd.read_csv(
        "./inet100/inet100_formal_concepts_full.csv"
    )

    def clean_up(text):
        return text.strip("()")

    df["intent"] = df["intent"].apply(clean_up)
    df["extent"] = df["extent"].apply(clean_up)

    df.to_csv(
        "./inet100/inet100_formal_concepts_full.csv",
        index=False,
    )


def binarize_attributes():
    df = pd.read_csv(
        "./fca4nn/data/formal_concepts/inet100_annotations_instancelevel_reduced_100classes.csv"
    )

    all_attributes = set()
    for intent in df["intent"]:
        if isinstance(intent, str):
            for attribute in intent.split(","):
                all_attributes.add(attribute.strip())

    print("all_attrs: ", all_attributes)
    attribute_to_index = {
        attribute: i for i, attribute in enumerate(sorted(all_attributes))
    }
    print(attribute_to_index)
    num_attributes = len(attribute_to_index)
    print("Num attrs: ", num_attributes)

    def intent_to_binary_vector(intent):
        # print(intent)
        if not isinstance(intent, str):
            vector = np.zeros(num_attributes, dtype=int)
            return ",".join(map(str, vector))

        vector = np.zeros(num_attributes, dtype=int)
        for attribute in intent.split(","):
            if attribute.strip() not in attribute_to_index.keys():
                print("Attribute not found: ", intent)
            index = attribute_to_index[attribute.strip()]
            vector[index] = 1
        return ",".join(map(str, vector))

    df["intent"] = df["intent"].apply(intent_to_binary_vector)

    df.to_csv(
        "./fca4nn/data/formal_concepts/inet100_formal_concepts_bin.csv",
        index=False,
    )


def group_formalconcepts():
    df = pd.read_csv(
        "./data/inet100/inet100_formal_concepts_binclasslevel.csv"
    )
    grouped_df = df.groupby(["level", "class"])
    df["fc_id"] = None

    for idx, (name, group) in enumerate(grouped_df):
        group_index = group.index
        df.loc[group_index, "fc_id"] = idx

    df.to_csv(
        "./data/inet100/inet100_formal_concepts_groupedclasslevel.csv",
        index=False,
    )


def binarize_class_level_attrs():
    with open("./data/inet100/imagenet100.json", "r") as f:
        class_attr_mapping = json.load(f)

    all_attributes = []
    for concepts in class_attr_mapping.values():
        all_attributes.extend(concepts)
    all_attributes = set(all_attributes)

    attribute_to_index = {
        attribute: i for i, attribute in enumerate(sorted(all_attributes))
    }
    num_attributes = len(attribute_to_index)

    class2concepts = {}
    for class_name, concepts in class_attr_mapping.items():
        class_name = class_name.replace(" ", "_")
        vector = np.zeros(num_attributes, dtype=int)
        for concept in concepts:
            index = attribute_to_index[concept]
            vector[index] = 1
        class2concepts[class_name] = ",".join(map(str, vector))
    return class2concepts


def create_class_level_annotations():
    class2binconcepts = binarize_class_level_attrs()
    # print(class2binconcepts)
    df = pd.read_csv(
        "./data/inet100/inet100_formal_concepts_full.csv"
    )
    df["intent"] = df["class"].apply(lambda x: class2binconcepts[x])
    df.to_csv(
        "./data/inet100/inet100_formal_concepts_binclasslevel.csv",
        index=False,
    )


def replace_class_with_id():
    classtoid = dict()
    with open(
        "./data/inet100/inet100_fc2class.json", "r"
    ) as f:
        class_attr_mapping = json.load(f)
    classes = [item["class"] for item in class_attr_mapping]
    for idx, class_name in enumerate(classes):
        classtoid[class_name] = idx

    df = pd.read_csv(
        "./data/inet100/inet100_formal_concepts_grouped_bin.csv"
    )
    df["class_id"] = df["class"].map(classtoid)
    df.to_csv(
        "./inet100/inet100_formal_concepts_grouped_bin.csv",
        index=False,
    )


class MultiLabelMetrics:
    def __init__(self):
        self.y_true_list = []
        self.y_pred_list = []

    def update(self, y_true, y_pred):
        """
        Updates the internal storage with a new batch of predictions and ground truth.
        :param y_true: np.array of shape (batch_size, num_labels)
        :param y_pred: np.array of shape (batch_size, num_labels)
        """
        self.y_true_list.append(y_true)
        self.y_pred_list.append(y_pred)

    def compute(self):
        """
        Computes the final metrics over all stored batches.
        :return: Dictionary containing final metric values.
        """
        y_true = np.vstack(self.y_true_list)
        y_pred = np.vstack(self.y_pred_list)

        metrics = {
            "Hamming Loss": hamming_loss(y_true, y_pred),
            "Subset Accuracy": np.mean(
                np.all(y_true == y_pred, axis=1)
            ),  # Exact match ratio
            "Precision (Macro)": precision_score(
                y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
            ),
            "Recall (Macro)": recall_score(
                y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
            ),
            "F1 Score (Macro)": f1_score(
                y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
            ),
            "Jaccard Index (Macro)": jaccard_score(
                y_true, y_pred, average="macro", zero_division=0, labels=[0, 1]
            ),
            "Accuracy of 1s": np.mean(
                np.sum(y_true * y_pred, axis=1) / np.sum(y_true, axis=1)
            ),
            "Accuracy of 0s": np.mean(
                np.sum((1 - y_true) * (1 - y_pred), axis=1)
                / np.sum((1 - y_true), axis=1)
            ),
        }

        # Compute MCC only if we have more than one label
        if y_true.shape[1] > 1:
            mcc_values = [
                matthews_corrcoef(y_true[:, i], y_pred[:, i])
                for i in range(y_true.shape[1])
            ]
            metrics["MCC (Mean)"] = np.mean(mcc_values)

        return metrics


class MetricCalculator:
    def __init__(self, num_clfs):
        self.num_clfs = num_clfs
        self.cls_01_common = {"0": [0] * (num_clfs - 1), "1": [0] * (num_clfs - 1)}
        self.cls_01_correct = {"0": [0] * (num_clfs - 1), "1": [0] * (num_clfs - 1)}
        self.attr_01_common = {"0": [0] * num_clfs, "1": [0] * num_clfs}
        self.attr_01_correct = {"0": [0] * num_clfs, "1": [0] * num_clfs}

        # For AUC calculation
        self.cls_scores = [[] for _ in range(num_clfs - 1)]
        self.cls_labels = [[] for _ in range(num_clfs - 1)]
        self.attr_scores = [[] for _ in range(num_clfs)]
        self.attr_labels = [[] for _ in range(num_clfs)]

    def reset(self):
        self.cls_01_common = {
            "0": [0] * (self.num_clfs - 1),
            "1": [0] * (self.num_clfs - 1),
        }
        self.cls_01_correct = {
            "0": [0] * (self.num_clfs - 1),
            "1": [0] * (self.num_clfs - 1),
        }
        self.attr_01_common = {"0": [0] * self.num_clfs, "1": [0] * self.num_clfs}
        self.attr_01_correct = {"0": [0] * self.num_clfs, "1": [0] * self.num_clfs}

        self.cls_scores = [[] for _ in range(self.num_clfs - 1)]
        self.cls_labels = [[] for _ in range(self.num_clfs - 1)]
        self.attr_scores = [[] for _ in range(self.num_clfs)]
        self.attr_labels = [[] for _ in range(self.num_clfs)]

    def update(self, cls_preds, attr_preds, classes_present, attrs_present):
        for i in range(self.num_clfs - 1):
            pred_i = torch.round(cls_preds[i]).cpu()
            label_i = classes_present[i]

            self.cls_01_common["0"][i] += torch.sum((1 - pred_i) * (1 - label_i)).item()
            self.cls_01_correct["0"][i] += torch.sum(1 - label_i).item()
            self.cls_01_common["1"][i] += torch.sum(pred_i * label_i).item()
            self.cls_01_correct["1"][i] += torch.sum(label_i).item()

            self.cls_scores[i].extend(cls_preds[i].detach().cpu().numpy().tolist())
            self.cls_labels[i].extend(label_i.detach().cpu().numpy().tolist())

            # attribute update for i-th
            attr_pred_i = torch.round(attr_preds[i]).cpu()
            attr_label_i = attrs_present[i]

            self.attr_01_common["0"][i] += torch.sum(
                (1 - attr_pred_i) * (1 - attr_label_i)
            ).item()
            self.attr_01_correct["0"][i] += torch.sum(1 - attr_label_i).item()
            self.attr_01_common["1"][i] += torch.sum(attr_pred_i * attr_label_i).item()
            self.attr_01_correct["1"][i] += torch.sum(attr_label_i).item()

            self.attr_scores[i].extend(attr_preds[i].detach().cpu().numpy().tolist())
            self.attr_labels[i].extend(attr_label_i.detach().cpu().numpy().tolist())

        # last attribute (final)
        self.attr_01_common["0"][-1] += torch.sum(
            (1 - torch.round(attr_preds[-1]).cpu()) * (1 - attrs_present[-1])
        ).item()
        self.attr_01_correct["0"][-1] += torch.sum(1 - attrs_present[-1]).item()
        self.attr_01_common["1"][-1] += torch.sum(
            torch.round(attr_preds[-1]).cpu() * attrs_present[-1]
        ).item()
        self.attr_01_correct["1"][-1] += torch.sum(attrs_present[-1]).item()

        self.attr_scores[-1].extend(attr_preds[-1].detach().cpu().numpy().tolist())
        self.attr_labels[-1].extend(attrs_present[-1].detach().cpu().numpy().tolist())

    def calculate_accuracy(self):
        cls_01_acc = {"0": [0] * (self.num_clfs - 1), "1": [0] * (self.num_clfs - 1)}
        attr_01_acc = {"0": [0] * self.num_clfs, "1": [0] * self.num_clfs}
        for i in range(self.num_clfs - 1):
            cls_01_acc["0"][i] = (
                self.cls_01_common["0"][i] / self.cls_01_correct["0"][i]
                if self.cls_01_correct["0"][i] > 0
                else 0
            )
            cls_01_acc["1"][i] = (
                self.cls_01_common["1"][i] / self.cls_01_correct["1"][i]
                if self.cls_01_correct["1"][i] > 0
                else 0
            )
            attr_01_acc["0"][i] = (
                self.attr_01_common["0"][i] / self.attr_01_correct["0"][i]
                if self.attr_01_correct["0"][i] > 0
                else 0
            )
            attr_01_acc["1"][i] = (
                self.attr_01_common["1"][i] / self.attr_01_correct["1"][i]
                if self.attr_01_correct["1"][i] > 0
                else 0
            )

        attr_01_acc["0"][-1] = (
            self.attr_01_common["0"][-1] / self.attr_01_correct["0"][-1]
            if self.attr_01_correct["0"][-1] > 0
            else 0
        )
        attr_01_acc["1"][-1] = (
            self.attr_01_common["1"][-1] / self.attr_01_correct["1"][-1]
            if self.attr_01_correct["1"][-1] > 0
            else 0
        )
        return cls_01_acc, attr_01_acc

    def calculate_auc(self):
        cls_auc = []
        attr_auc = []
        for i in range(self.num_clfs - 1):
            try:
                auc = roc_auc_score(self.cls_labels[i], self.cls_scores[i])
            except ValueError:
                auc = float("nan")  # Handle cases with only one class
            cls_auc.append(auc)

        for i in range(self.num_clfs):
            try:
                auc = roc_auc_score(self.attr_labels[i], self.attr_scores[i])
            except ValueError:
                auc = float("nan")
            attr_auc.append(auc)

        return cls_auc, attr_auc
