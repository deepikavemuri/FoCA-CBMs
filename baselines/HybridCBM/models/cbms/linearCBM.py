# -*- coding: utf-8 -*-
import torch
from .baseCBM import CBM


class LinearCBM(CBM):
    # lr = 1e-3
    def config_classifier(self):
        # if not self.config.use_normalize:
        classifier = torch.nn.Linear(self.concept_features.shape[0], self.config.num_class)
        classifier.weight.data = self.init_weight_matrix()
        classifier.bias.data.zero_()
        return classifier
