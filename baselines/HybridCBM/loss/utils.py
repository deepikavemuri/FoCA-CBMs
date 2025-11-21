import torch


class _Loss:
    def __init__(self, loss_weight=0):
        self.loss_weight = loss_weight

    def compute(self, *args, **kwargs):
        raise NotImplementedError

    def __call__(self, *args, **kwargs):
        if self.loss_weight == 0:
            return torch.tensor(0, device=args[0].device)
        return self.loss_weight * self.compute(*args, **kwargs)


def clip_score(img_feat, concept_feat, n_shots, num_images_per_class):
    num_cls = len(num_images_per_class)
    scores_mean = torch.empty((concept_feat.shape[0], num_cls))
    start_loc = 0
    for i in range(num_cls):
        end_loc = sum(num_images_per_class[:i + 1])
        scores_mean[:, i] = (concept_feat @ img_feat[start_loc:end_loc].t()).mean(dim=-1)
        start_loc = end_loc
    return scores_mean


def mutual_information_score(img_feat, concept_feat, n_shots, num_images_per_class, epsilon=1e-10):
    num_cls = len(num_images_per_class)
    scores_mean = clip_score(img_feat, concept_feat, n_shots, num_images_per_class)  # Sim(c,y)
    # 将余弦相似度平移到非负区间
    scores_mean = scores_mean - scores_mean.min()
    normalized_scores = scores_mean / (scores_mean.sum(dim=0) * num_cls + epsilon)  # Sim_bar(c,y)
    margin_x = normalized_scores.sum(dim=1)  # sum_y in Y Sim_bar(c,y)
    margin_x = margin_x.reshape(-1, 1).repeat(1, num_cls)
    # compute MI and PMI
    pmi = torch.log((normalized_scores + epsilon) / (
            margin_x * 1 / num_cls + epsilon))  # log Sim_bar(c,y) / sum_y in Y Sim_bar(c,y) / N = log(Sim_bar(c|y))
    mi = normalized_scores * pmi  # Sim_bar(c,y)* log(Sim_bar(c|y))
    mi = mi.sum(dim=1)
    return mi, scores_mean
