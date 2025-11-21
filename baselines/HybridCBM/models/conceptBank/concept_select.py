import torch
import random
from tqdm import tqdm
import numpy as np
from apricot import CustomSelection, MixtureSelection, FacilityLocationSelection
from apricot.optimizers import OPTIMIZERS
from scipy.sparse import csr_matrix
from apricot.utils import _calculate_pairwise_distances


class ModifiedMixtureSelection(MixtureSelection):
    """
    It's a modified version of the MixtureSelection class from the apricot library.
    Pass the concept features that contain the negative values.
    """

    def fit(self, X, y=None, sample_weight=None, sample_cost=None):
        self._X = X
        X = _calculate_pairwise_distances(X, metric=self.metric, n_neighbors=self.n_neighbors)

        allowed_dtypes = list, np.ndarray, csr_matrix

        if not isinstance(X, allowed_dtypes):
            raise ValueError("X must be either a list of lists, a 2D numpy array, or a scipy.sparse.csr_matrix.")
        if isinstance(X, np.ndarray) and len(X.shape) != 2:
            raise ValueError("X must have exactly two dimensions.")
        if self.n_samples > X.shape[0]:
            raise ValueError("Cannot select more examples than the number in the data set.")

        if not self.sparse:
            if X.dtype != 'float64':
                X = X.astype('float64')

        if isinstance(self.optimizer, str):
            optimizer = OPTIMIZERS[self.optimizer](function=self,
                                                   verbose=self.verbose, random_state=self.random_state,
                                                   **self.optimizer_kwds)
        else:
            optimizer = self.optimizer

        self._X = X if self._X is None else self._X
        self._initialize(X)

        if self.verbose:
            self.pbar = tqdm(total=self.n_samples, unit_scale=True)

        optimizer.select(X, self.n_samples, sample_cost=sample_cost)

        if self.verbose == True:
            self.pbar.close()

        self.ranking = np.array(self.ranking)
        self.gains = np.array(self.gains)
        return self


def clip_score(img_feat, concept_feat, n_shots, num_images_per_class):
    num_cls = len(num_images_per_class)
    scores_mean = torch.empty((concept_feat.shape[0], num_cls))
    start_loc = 0
    for i in range(num_cls):
        end_loc = sum(num_images_per_class[:i + 1])
        scores_mean[:, i] = (concept_feat @ img_feat[start_loc:end_loc].t()).mean(dim=-1)
        start_loc = end_loc
    return scores_mean


def mi_score(img_feat, concept_feat, num_images_per_class):
    num_cls = len(num_images_per_class)
    scores_mean = clip_score(img_feat, concept_feat, None, num_images_per_class)  # Sim(c,y)
    normalized_scores = scores_mean / (scores_mean.sum(dim=0) * num_cls)  # Sim_bar(c,y)
    # normalized_scores  = normalized_scores - normalized_scores.min() # normalize to positive
    margin_x = normalized_scores.sum(dim=1)  # sum_y in Y Sim_bar(c,y)
    margin_x = margin_x.reshape(-1, 1).repeat(1, num_cls)
    # compute MI and PMI
    # log Sim_bar(c,y) / sum_y in Y Sim_bar(c,y) / N = log(Sim_bar(c|y))
    pmi = torch.log(normalized_scores / (margin_x * 1 / num_cls))
    mi = normalized_scores * pmi  # Sim_bar(c,y)* log(Sim_bar(c|y))
    mi = mi.sum(dim=1)
    return mi, scores_mean


def submodular_select(img_feat, concept_feat, concept2cls, num_concepts, num_images_per_class, submodular_weights):
    assert num_concepts > 0
    num_cls = len(num_images_per_class)

    all_mi_scores, _ = mi_score(img_feat, concept_feat, num_images_per_class)
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))

    def mi_based_function(X):
        return X[:, 0].sum()

    mi_selector = CustomSelection(num_concepts_per_cls, mi_based_function)
    distance_selector = FacilityLocationSelection(num_concepts_per_cls, metric='cosine')

    mi_score_scale = submodular_weights[0]
    facility_weight = submodular_weights[-1]

    if mi_score_scale == 0:
        submodular_weights = [0, facility_weight]
    else:
        submodular_weights = [1, facility_weight]
    submodular_weights = np.array(submodular_weights)
    # min_feature_value = torch.min(concept_feat)
    # if min_feature_value < 0:
    #     concept_feat += abs(min_feature_value)

    concept2cls = torch.from_numpy(concept2cls).long()
    for i in tqdm(range(num_cls)):
        # class i
        cls_idx = torch.where(concept2cls == i)[0]

        if len(cls_idx) <= num_concepts_per_cls:
            selected_idx.extend(cls_idx)
        else:
            mi_scores = all_mi_scores[cls_idx] * mi_score_scale

            current_concept_features = concept_feat[cls_idx]
            augmented_concept_features = torch.hstack([torch.unsqueeze(mi_scores, 1), current_concept_features]).numpy()
            selector = ModifiedMixtureSelection(num_concepts_per_cls, functions=[mi_selector, distance_selector],
                                                weights=submodular_weights, optimizer='naive', verbose=False)

            selected = selector.fit(augmented_concept_features).ranking
            selected_idx.extend(cls_idx[selected])

    return torch.tensor(selected_idx)


def random_select(_, __, concept2cls, num_concepts, num_images_per_class, submodular_weights):
    assert num_concepts > 0
    num_cls = len(num_images_per_class)
    take_all = False
    selected_idx = []
    num_concepts_per_cls = int(np.ceil(num_concepts / num_cls))
    concept2cls = torch.from_numpy(concept2cls).long()
    for i in tqdm(range(num_cls)):
        cls_idx = torch.where(concept2cls == i)[0]
        if len(cls_idx) == 0:
            continue
        elif len(cls_idx) < num_concepts_per_cls or (take_all and num_cls < 10):
            global_idx = cls_idx

        else:
            global_idx = random.sample(cls_idx.tolist(), num_concepts_per_cls)
        selected_idx.extend(global_idx)
    return torch.tensor(selected_idx)
