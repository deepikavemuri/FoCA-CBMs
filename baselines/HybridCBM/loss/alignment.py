import torch
import logging
from geomloss import SamplesLoss

try:
    from .utils import _Loss
except:
    from utils import _Loss


class MaxMeanDiscrepancyLoss(_Loss):
    def __init__(self, loss_weight=1.0, sigma=1.0):
        super(MaxMeanDiscrepancyLoss, self).__init__(loss_weight)
        self.sigma = sigma  # Kernel width
        logging.info(f"Setup Mean Discrepancy Loss with loss_weight: {loss_weight}, sigma: {sigma}")

    def compute(self, concept_vectors, static_vectors):
        # Compute Gaussian kernels
        concept = torch.exp(-1 * (concept_vectors[:, None] - concept_vectors).pow(2).sum(2) / (2 * self.sigma ** 2))
        static = torch.exp(-1 * (static_vectors[:, None] - static_vectors).pow(2).sum(2) / (2 * self.sigma ** 2))
        cross = torch.exp(-1 * (concept_vectors[:, None] - static_vectors).pow(2).sum(2) / (2 * self.sigma ** 2))

        # Compute MMD Loss
        loss = concept.mean() + static.mean() - 2 * cross.mean()
        return loss


class MeanCovarianceAlignmentLoss(_Loss):
    def __init__(self, loss_weight=1.0):
        super(MeanCovarianceAlignmentLoss, self).__init__(loss_weight)
        logging.info(f"Setup CORAL Loss with loss_weight: {loss_weight}")

    def compute(self, concept_vectors, static_vectors):
        # 计算每个数据集的均值
        concept_mean = concept_vectors.mean(0)
        static_mean = static_vectors.mean(0)

        # 计算每个数据集的协方差
        N1 = concept_vectors.size(0)
        N2 = static_vectors.size(0)
        concept_cov = (concept_vectors - concept_mean).T @ (concept_vectors - concept_mean) / (N1 - 1)
        static_cov = (static_vectors - static_mean).T @ (static_vectors - static_mean) / (N2 - 1)

        # 计算均值和协方差的Frobenius范数差异
        mean_diff = torch.norm(concept_mean - static_mean, p=2) ** 2
        cov_diff = torch.norm(concept_cov - static_cov, p='fro') ** 2

        # 组合均值和协方差的差异为总损失
        loss = mean_diff + cov_diff
        return loss


class KLDivergenceLoss(_Loss):
    def __init__(self, loss_weight=1.0):
        super(KLDivergenceLoss, self).__init__(loss_weight)
        logging.info(f"Setup KL Divergence Loss with loss_weight: {loss_weight}")

    def compute(self, concept_vectors, static_vectors):
        """
        :param concept_vectors: N1, d
        :param static_vectors: N2, d
        :return:
        """
        # 计算平均概率分布
        p_mean = torch.nn.functional.softmax(concept_vectors, dim=-1).mean(0)
        q_mean = torch.nn.functional.softmax(static_vectors, dim=-1).mean(0)

        # 为避免对数计算中的问题，添加一个小的常数
        eps = 1e-10
        p_mean = torch.clamp(p_mean, eps, 1.0)
        q_mean = torch.clamp(q_mean, eps, 1.0)

        # 计算 KL 散度
        kl_div = torch.sum(p_mean * torch.log(p_mean / q_mean))

        return kl_div


class SinkhornDistanceLoss(_Loss):
    def __init__(self, loss="sinkhorn", loss_weight=1.0, p=2, blur=0.1, scaling=0.5):
        super(SinkhornDistanceLoss, self).__init__(loss_weight)
        self.p = p
        self.sinkhorn_distance = SamplesLoss(loss=loss, p=2, blur=blur, scaling=scaling)
        logging.info(f"Setup Sinkhorn Distance Loss with loss_weight: {loss_weight}")

    def compute(self, concept_vectors, static_vectors):
        """
        Warning: Sinkhorn distance is not like cosine similarity,
                 do not normalize input vectors
        :param concept_vectors: N1, d
        :param static_vectors: N2, d
        :return:
        """
        # static_vectors = self.avoid_nan(concept_vectors.shape[0], static_vectors)
        loss = self.sinkhorn_distance(concept_vectors, static_vectors)
        assert not torch.isnan(loss).any(), f"Sinkhorn distance loss is nan"
        return loss
