import numpy as np


def gini_index(y, gt_y=None):
    """
    Calculate the Gini index for a list of labels.
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    counts = counts / len(y)
    gini = 1 - np.sum(counts**2)
    return gini


def entropy(y, gt_y=None):
    """
    Calculate the entropy for a list of labels.
    """
    if len(y) == 0:
        return 0.0
    _, counts = np.unique(y, return_counts=True)
    counts = counts / len(y)
    entropy = -np.sum(counts * np.log(counts + 1e-10))
    return entropy


def impurity(y, gt_y=None):
    """
    Calculate the purity for a list of labels.
    """

    if len(y) == 0:
        return 0.0
    y = np.array(y)
    return 1 - (np.sum(y == gt_y) / len(y))
