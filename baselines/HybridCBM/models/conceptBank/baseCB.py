# -*- coding: utf-8 -*-
import torch
import pandas as pd

class BaseCB(torch.nn.Module):
    @property
    def concept_features(self):
        raise NotImplementedError

    @property
    def concepts(self):
        raise NotImplementedError

    @property
    def concepts_label(self):
        raise NotImplementedError

    @property
    def concepts_path(self):
        raise NotImplementedError

    @property
    def delete_modulename(self):
        return ''

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        model_keys = [key for key in state_dict if self.delete_modulename in key]
        for key in model_keys:
            del state_dict[key]
        return state_dict

    def load_state_dict(self, state_dict, strict=True, assign: bool = False):
        super().load_state_dict(state_dict, strict=False, assign=assign)

    @torch.no_grad()
    def get_concepts(self, class_idx=None, idx=None, dataframe=False):
        concepts = self.concepts
        if class_idx is not None:
            if isinstance(class_idx, torch.Tensor):
                class_idx = class_idx.cpu().numpy().tolist()
            if isinstance(class_idx, int):
                concepts = self.concepts[self.concepts['label'] == class_idx]
            elif isinstance(class_idx, list):
                concepts = self.concepts[self.concepts['label'].isin(class_idx)]
            else:
                raise ValueError('class_idx should be int or list')
        if idx is not None:
            if isinstance(idx, torch.Tensor):
                if idx.dim() == 1:
                    idx = idx.cpu().numpy().tolist()
                    concepts = self.concepts.iloc[idx]
                elif idx.dim() == 2:
                    concepts = []
                    for i, cid in enumerate(idx):
                        concept = self.concepts.iloc[cid]
                        concept.loc[:, 'idx'] = i
                        concepts.append(concept)
                    concepts = pd.concat(concepts, axis=0)
            elif isinstance(idx, int):
                concepts = self.concepts.iloc[idx]
            elif isinstance(idx, list):
                concepts = self.concepts.iloc[idx]
            else:
                raise ValueError('idx should be int or list')
        if dataframe:
            return concepts
        return concepts['concept'].tolist()

    def save_concepts(self):
        if self.concepts is None:
            self.get_concepts(dataframe=True)
        self.concepts.to_csv(self.concepts_path, index=False)

    def compute_score(self, x):
        # x (B, dim)
        return x @ self.concept_features.T
