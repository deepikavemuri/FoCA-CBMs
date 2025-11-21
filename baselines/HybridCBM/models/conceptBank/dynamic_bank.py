# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import torch
from pathlib import Path
from .baseCB import BaseCB
from ..translator import ConceptTranslator
from ..utils import freeze


class DynamicConceptBank(BaseCB):
    def __init__(
        self,
        exp_root,
        clip_model,
        num_concept=0,
        num_class=100,
        feature_dim=None,
        translator_path=None,
    ):
        super().__init__()
        self.exp_root = Path(exp_root)
        # self.translator = ConceptTranslator(clip_model=clip_model)
        # self.translator.load(weight_path=translator_path)
        # freeze(self.translator)
        self.num_concept = num_concept
        # selected concept indices
        self.dynamic_features = torch.nn.Parameter(
            torch.randn(num_concept, feature_dim), requires_grad=True
        )
        self.classid_features = torch.LongTensor(
            sorted([i % num_class for i in range(num_concept)])
        )

    @property
    def delete_modulename(self):
        return "translator"

    @property
    def concept_features(self):
        return self.dynamic_features

    @concept_features.setter
    def concept_features(self, value):
        self.dynamic_features.data = value

    @property
    def concepts(self):
        if hasattr(self, "_concepts"):
            return self._concepts
        if self.num_concept == 0:
            return pd.DataFrame(columns=["concept", "label"])
        # self._concepts = self.translator.decode(self.concept_features, entry_length=30, temperature=1, batch_size=32)
        # self._concepts = pd.DataFrame({'concept': self._concepts,
        #                                'label': self.concepts_label.tolist()}).reset_index(drop=True)
        return pd.DataFrame(columns=["concept", "label"])

    @property
    def concepts_label(self):
        return self.classid_features

    @property
    def concepts_path(self):
        return self.exp_root.joinpath("dynamic_concepts.csv")


class AttentionBank(DynamicConceptBank):
    def __init__(
        self,
        exp_root,
        clip_model,
        static_features_fn,
        num_concept=0,
        num_class=100,
        translator_path=None,
    ):
        super(DynamicConceptBank, self).__init__()
        self.exp_root = Path(exp_root)
        self.translator = ConceptTranslator(clip_model=clip_model)
        self.translator.load(weight_path=translator_path)
        freeze(self.translator)
        self.num_concept = num_concept
        # selected concept indices
        self.static_features_fn = static_features_fn

        if clip_model == "ViT-L/14":
            embedding_dim = 768
        elif clip_model == "ViT-B/32":
            embedding_dim = 512
        else:
            raise ValueError(f"Invalid clip model: {clip_model}")
        self.attn = torch.nn.Parameter(torch.randn(embedding_dim, num_concept))
        self.classid_features = torch.LongTensor(
            sorted([i % num_class for i in range(num_concept)])
        )

    @property
    def concept_features(self):
        value = self.static_features_fn() / self.static_features_fn().norm(
            dim=-1, keepdim=True
        )
        atte = self.attn / self.attn.norm(dim=0, keepdim=True)
        attn_weights = (100 * value @ atte).T.softmax(dim=-1)  # (N2, N1)
        # Apply attention weights to values
        context = attn_weights @ self.static_features_fn()  # (N2, D)
        return context

    @concept_features.setter
    def concept_features(self, value):
        raise NotImplementedError
