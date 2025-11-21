# -*- coding: utf-8 -*-
import pandas as pd
import torch

from pathlib import Path

from models.clip import ClipEncoder
from .baseCB import BaseCB
from .concept_select import submodular_select, random_select


def select_fn(method):
    if method == "submodular":
        return submodular_select
    elif method == "random":
        return random_select
    else:
        raise NotImplementedError(f"Unknown method: {method}")


class StaticConceptBank(BaseCB):
    def __init__(self,
                 exp_root,
                 data_root,
                 # concept
                 num_concept,
                 concept_select_fn=None,
                 submodular_weights=None,
                 clip_model: str = 'ViT-B/32',
                 ):
        super().__init__()
        self.exp_root = Path(exp_root)
        self.data_root = Path(data_root)
        self.num_concept = num_concept
        concepts = pd.read_csv(self.data_root.joinpath('concepts/concepts.csv'))
        concepts = concepts.drop_duplicates(subset=['concept']).reset_index(drop=True)
        self.classes = pd.read_csv(self.data_root.joinpath('class.csv'))['class'].unique()
        self.num_classes = len(self.classes)
        self._concepts = concepts[['concept', 'label']]

        self.concept_select_fn = select_fn(concept_select_fn)
        self.submodular_weights = submodular_weights
        self.clip_encoder = ClipEncoder(
            model_name=clip_model,
            use_img_norm=False,
            use_txt_norm=False,
        )
        # selected concept indices
        concept_feature_dir = self.data_root.joinpath("concepts_feat")
        concept_feature_dir.mkdir(parents=True, exist_ok=True)
        self.concept_feature_save_path = concept_feature_dir.joinpath(
            f"{self.clip_encoder.model_name.replace('/', '-')}.pth")
        self.select_idx_save_path = concept_feature_dir.joinpath(
            f"{self.clip_encoder.model_name.replace('/', '-')}_{concept_select_fn}-select_{num_concept}_idx.pth")

        # init weight
        self.cls_sim_save_path = self.exp_root.joinpath('cls_sim.pth')
        self.static_features = None
        self.is_initialized = False
        try:
            self.initialize()
        except FileNotFoundError:
            print('concept feature not found, please run datamodule precompute_txt first')

    @property
    @torch.no_grad()
    def concept_features(self):
        return self.static_features

    @property
    def concepts(self):
        return self._concepts

    @property
    def concepts_label(self):
        return torch.LongTensor(self.concepts['label'].tolist())

    @property
    def concepts_path(self):
        return self.exp_root.joinpath('static_concepts.csv')

    @property
    def delete_modulename(self):
        return 'clip_encoder'

    @property
    def classes_embeddings(self):
        return self.clip_encoder.encode_text(self.classes,
                                             batch_size=128,
                                             normalize=False)

    def initialize(self, img_features=None, num_images_per_class=None):
        if self.is_initialized:
            return
        if self.num_concept > 0:
            static_features = self.prepare_text_feature(num_images_per_class)
            select_cid = self.select_concept(img_features, static_features, num_images_per_class)
            self.static_features = torch.nn.Parameter(static_features.data[select_cid], requires_grad=False)
            self._concepts = self._concepts.iloc[select_cid].reset_index(drop=True)
            self.register_parameter('static_features', self.static_features)
        else:
            self._concepts = pd.DataFrame(columns=['concept', 'class', 'label'])
            self.static_features = torch.nn.Parameter(torch.zeros(0))
            self.register_parameter('static_features', self.static_features)
        self.is_initialized = True

    def select_concept(self, img_features, text_features, num_images_per_class):
        if not self.select_idx_save_path.exists():
            if img_features is None:
                raise FileNotFoundError('img_features should be provided for concept selection')
            print('select concepts')
            select_cid = self.concept_select_fn(img_features.detach().cpu(),
                                                text_features.detach().cpu(),
                                                self.concepts['label'].to_numpy(),
                                                self.num_concept,
                                                num_images_per_class,
                                                self.submodular_weights)
            torch.save(select_cid, self.select_idx_save_path)
            self._concepts.iloc[select_cid].reset_index(drop=True).to_csv(
                self.data_root.joinpath('concepts/selected_concepts.csv'), index=False)
        else:
            select_cid = torch.load(self.select_idx_save_path, weights_only=True)
        return select_cid

    def get_mask_from_img_sim(self, img_features=None, labels=None, n_shots=None):
        if not self.cls_sim_save_path.exists():
            print('generate cls sim mask')
            img_feat = img_features
            label = labels[::n_shots]
            num_cls = len(self.classes)
            img_feat = img_feat / (img_feat.norm(dim=-1, keepdim=True) + 1e-7)
            img_sim = img_feat @ img_feat.T
            class_sim = torch.empty((num_cls, num_cls))
            for i, row_split in enumerate(torch.split(img_sim, n_shots, dim=0)):
                for j, col_split in enumerate(torch.split(row_split, n_shots, dim=1)):
                    class_sim[label[i], label[j]] = torch.mean(col_split)

            good = class_sim >= torch.quantile(class_sim, 0.95, dim=-1)
            final_sim = torch.zeros(class_sim.shape)
            for i in range(num_cls):
                for j in range(num_cls):
                    if i == j:
                        final_sim[i, j] = 1
                    elif good[i, j]:
                        final_sim[i, j] = class_sim[i, j]

            torch.save(final_sim, self.cls_sim_save_path)
        else:
            final_sim = torch.load(self.cls_sim_save_path)
        return final_sim

    def prepare_text_feature(self, num_images_per_class):
        if not self.concept_feature_save_path.exists():
            if num_images_per_class is None:
                raise FileNotFoundError('num_images_per_class should be provided for concept selection')
            print('prepare text features')
            concepts = self.concepts['concept'].tolist()
            concept_features = self.clip_encoder.encode_text(concepts, batch_size=128)
            torch.save(concept_features, self.concept_feature_save_path)
        else:
            concept_features = torch.load(self.concept_feature_save_path, weights_only=True)
        return concept_features