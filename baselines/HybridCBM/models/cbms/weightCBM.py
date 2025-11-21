# -*- coding: utf-8 -*-
import torch
import torch.nn.functional as F
from torch.optim.lr_scheduler import LambdaLR

from .baseCBM import CBM


class WeightMatrix(torch.nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.weight = torch.nn.Parameter(weight)

    def get_weight_matrix(self):
        weight = F.softmax(self.weight, dim=-1)
        return weight

    def forward(self, x):
        return x @ self.get_weight_matrix().T


class WeightCBM(CBM):
    def config_classifier(self):
        del self.scale
        return WeightMatrix(self.init_weight_matrix())

    def get_weight_matrix(self):
        return self.classifier.get_weight_matrix()

    def configure_optimizers(self):
        optimizer_dynamic = torch.optim.Adam([
            {'params': self.conceptbank.dynamic_bank.parameters(), 'lr': self.config.concept_lr},
        ])
        optimizer_classifier = torch.optim.Adam([
            {'params': self.classifier.parameters(), 'lr': self.config.lr},
        ])
        return [optimizer_dynamic, optimizer_classifier], []

    def forward(self, img_feat, concept_features=None):
        if concept_features is None:
            concept_features = self.concept_features
        sim_score = img_feat @ concept_features.T  # B, C
        logits = self.classifier(sim_score) * 100
        return logits
