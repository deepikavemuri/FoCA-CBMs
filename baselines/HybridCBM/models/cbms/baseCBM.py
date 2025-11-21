# -*- coding: utf-8 -*-
import logging
import torchmetrics
import torch
import lightning as L
import torch.nn.functional as F
from pathlib import Path
from typing import Mapping, Any
from loss import DiscriminabilityLoss, OrthogonalityLoss, SinkhornDistanceLoss
from metrics.clip import ClipEvaluator


class CBM(L.LightningModule):
    def __init__(self, config, conceptbank) -> None:
        super().__init__()
        self.config = config
        self.conceptbank = conceptbank
        self.train_mode = (
            config.train_mode
        )  # 'joint' or 'concept_{epochs}' or 'concept_stop_{epochs}'
        self.concept_stop_epochs = self.config.max_epochs
        self.cls_start_epochs = 0
        if "concept" in self.train_mode:
            if "stop" in self.train_mode:
                self.concept_stop_epochs = int(self.train_mode.split("_")[-1])
            self.cls_start_epochs = int(self.train_mode.split("_")[-1])

        self.exp_root = Path(config.exp_root)
        # this is the scale factor for the concept score, it can impact the speed of convergence
        self.scale = torch.nn.Parameter(
            torch.tensor(config.scale).float(), requires_grad=False
        )
        self.classifier = self.config_classifier()
        self.config_loss(config)
        self.config_metrics()
        self.save_hyperparameters(ignore=["conceptbank"])
        self.automatic_optimization = False
        try:
            logging.info(
                f"initialized {self.__class__.__name__} logit_scale: {self.scale.data.item()}"
            )
        except:
            pass

        logging.info(
            f"Training mode: {self.train_mode} concept_stop_epochs: {self.concept_stop_epochs}"
            f" classifier start epochs: {self.cls_start_epochs}"
        )

    @property
    def is_train_concept(self):
        return (
            self.current_epoch <= self.concept_stop_epochs
            and (
                self.config.lambda_discri > 0
                or self.config.lambda_ort > 0
                or self.config.lambda_align > 0
            )
            and self.config.num_dynamic_concept > 0
        )

    @property
    def is_train_cls(self):
        return self.current_epoch >= self.cls_start_epochs

    @property
    def classes_embeddings(self):
        if not hasattr(self, "_classes_embeddings"):
            self._classes_embeddings = self.conceptbank.classes_embeddings.to(
                self.device
            )
        return self._classes_embeddings

    def config_classifier(self):
        """
        configure the classifier, including the weight matrix
        :return: classifier model
        """
        raise NotImplementedError

    def config_loss(self, config):
        self.cls_loss = torch.nn.CrossEntropyLoss()
        self.discri_loss = DiscriminabilityLoss(
            loss_weight=config.lambda_discri,
            num_classes=config.num_class,
            alpha=config.lambda_discri_alpha,
            beta=config.lambda_discri_beta,
        )
        self.ortho_loss = OrthogonalityLoss(
            loss_weight=config.lambda_ort, num_classes=config.num_class
        )
        self.align_loss = SinkhornDistanceLoss(
            loss_weight=config.lambda_align, loss="sinkhorn"
        )

    def config_metrics(self):
        self.train_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.config.num_class
        )
        self.valid_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.config.num_class
        )
        self.test_acc = torchmetrics.Accuracy(
            task="multiclass", num_classes=self.config.num_class
        )
        self.clip_evaluator = ClipEvaluator(
            clip_encoder=self.conceptbank.clip_encoder, dataset=self.config.dataset
        )
        self.confusion_matrix = torchmetrics.ConfusionMatrix(
            task="multiclass", num_classes=self.config.num_class
        )

    @property
    def concept_features(self):
        return self.conceptbank.concept_features

    def get_weight_matrix(self):
        """
        get weight matrix, used for interpretability.
        if activation is needed, overwrite this function
        :return:
        """
        return self.weight_matrix

    def init_weight_matrix(self, init_weight=None):
        if init_weight is None:
            init_weight = torch.zeros(
                (self.config.num_class, self.concept_features.shape[0])
            )
        if self.config.weight_init_method == "zero":
            init_weight.data.zero_()
        elif self.config.weight_init_method == "rand":
            torch.nn.init.kaiming_normal_(init_weight)
        else:
            init_weight = self.conceptbank.get_init_weight_from_cls(
                self.config.weight_init_method
            )
        return init_weight

    def configure_optimizers(self):
        if self.is_train_concept:
            logging.info("Initializing optimizer for training concept")
            optimizer_dynamic = torch.optim.Adam(
                [
                    {
                        "params": self.conceptbank.dynamic_bank.parameters(),
                        "lr": self.config.concept_lr,
                    },
                ]
            )
            logging.info("Initializing optimizer for training classifier")
            optimizer_classifier = torch.optim.Adam(
                [
                    {"params": self.classifier.parameters(), "lr": self.config.lr},
                    {"params": self.scale, "lr": self.config.lr},
                ]
            )
            return [optimizer_dynamic, optimizer_classifier], []
        else:
            logging.info("Initializing optimizer for training classifier Only")
            optimizer_classifier = torch.optim.Adam(
                [
                    {"params": self.classifier.parameters(), "lr": self.config.lr},
                    {
                        "params": self.conceptbank.dynamic_bank.parameters(),
                        "lr": self.config.lr,
                    },
                    {"params": self.scale, "lr": self.config.lr},
                ]
            )
            return optimizer_classifier

    def forward(self, img_feat, concept_features=None):
        if concept_features is None:
            concept_features = self.concept_features
        sim_score = self.scale * img_feat @ concept_features.T  # B, C
        logits = self.classifier(sim_score)
        return logits

    def train_concept(self, image, label):
        final_loss = 0
        if self.config.lambda_discri > 0:
            image = image / image.norm(dim=-1, keepdim=True)
            discri_loss = self.discri_loss(
                image, self.conceptbank.dynamic_features, label, self.classes_embeddings
            )
            self.log("discri_loss", discri_loss)
            final_loss += discri_loss
        if self.config.lambda_ort > 0:
            ort_loss = self.ortho_loss(
                self.conceptbank.dynamic_bank.concept_features,
                self.conceptbank.static_bank.concept_features,
            )
            self.log("ort_loss", ort_loss)
            final_loss += ort_loss
        if (
            self.config.lambda_align > 0
            and self.conceptbank.dynamic_features.shape[0] > 0
            and self.conceptbank.static_features.shape[0] > 0
        ):
            # align loss, should not normalize the concept feature
            align_loss = self.align_loss(
                self.conceptbank.dynamic_bank.concept_features,
                self.conceptbank.static_bank.concept_features,
            )
            self.log("align_loss", align_loss)
            final_loss += align_loss
        return final_loss

    def train_classifier(self, image, label):
        if self.config.use_normalize:
            image = image / image.norm(dim=-1, keepdim=True)
        logits = self.forward(image)
        # classification accuracy
        cls_loss = self.cls_loss(logits, label)
        self.log("cls_loss", cls_loss)
        final_loss = cls_loss

        if self.config.lambda_l1 > 0:
            row_l1_norm = torch.linalg.vector_norm(
                self.classifier.weight, ord=1, dim=-1
            ).mean()
            self.log("mean l1 norm", row_l1_norm)
            final_loss += self.config.lambda_l1 * row_l1_norm

        self.train_acc(logits, label)
        self.log("train_acc", self.train_acc, on_step=False, on_epoch=True)
        return final_loss

    def training_step(self, train_batch, batch_idx):
        image, label = train_batch
        # Training
        final_loss = 0
        if self.is_train_concept:
            final_loss += self.train_concept(image, label)
        if self.is_train_cls:
            final_loss += self.train_classifier(image, label)
        self.manual_backward(final_loss)
        # Update optimizer
        opts = self.optimizers()
        if isinstance(opts, torch.optim.Optimizer):
            opt_classifier = opts
        else:
            opt_dynamic, opt_classifier = opts
            opt_dynamic.step()
            opt_dynamic.zero_grad()
        opt_classifier.step()
        opt_classifier.zero_grad()
        self.log("final_loss", final_loss)
        return final_loss

    def validation_step(self, batch, batch_idx):
        image, y = batch
        logits = self.forward(image)
        loss = F.cross_entropy(logits, y)
        self.valid_acc(logits, y)
        self.log("val_loss", loss)
        self.log("val_acc", self.valid_acc, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_start(self) -> None:
        self.conceptbank.save_concepts()
        self.clip_evaluator.classEmbeddings = self.classes_embeddings.cpu()
        self.all_y = []
        self.all_pred = []
        self.all_img_concept_purity = {"static": [], "dynamic": []}
        self.all_img_concept_coverage = {"static": [], "dynamic": []}

    def test_step(self, batch, batch_idx):
        image, y = batch
        logits = self.forward(image)
        loss = F.cross_entropy(logits, y)
        y_pred = logits.argmax(dim=-1)
        self.confusion_matrix(y_pred, y)
        self.all_y.append(y.cpu())
        self.all_pred.append(y_pred.cpu())
        self.test_acc(logits, y)

        # if self.conceptbank.static_features.shape[0] > 0:
        #     static_purity, static_coverage, static_pred = self.clip_evaluator.img_similarity(
        #         image, y,
        #         self.conceptbank.static_features
        #     )
        #     self.all_img_concept_purity['static'].append(static_purity)
        #     self.all_img_concept_coverage['static'].append(static_coverage)
        # if self.conceptbank.dynamic_features.shape[0] > 0:
        #     dynamic_purity, dynamic_coverage, dynamic_pred = self.clip_evaluator.img_similarity(
        #         image, y,
        #         self.conceptbank.dynamic_features
        #     )
        #     self.all_img_concept_purity['dynamic'].append(dynamic_purity)
        #     self.all_img_concept_coverage['dynamic'].append(dynamic_coverage)

        self.log("test_loss", loss)
        self.log("test_batch_acc", self.test_acc, on_step=False, on_epoch=True)
        return loss

    def on_test_epoch_end(self):
        all_y = torch.hstack(self.all_y)
        all_pred = torch.hstack(self.all_pred)
        self.test_total_acc = self.test_acc(all_pred, all_y)
        self.test_trans_dynamic_acc = self.test_trans_static_acc = None
        # image metrics
        for key in self.all_img_concept_purity.keys():
            if len(self.all_img_concept_purity[key]) > 0:
                self.all_img_concept_purity[key] = (
                    torch.cat(self.all_img_concept_purity[key]).mean().item()
                )
            if len(self.all_img_concept_coverage[key]) > 0:
                self.all_img_concept_coverage[key] = (
                    torch.cat(self.all_img_concept_coverage[key]).mean().item()
                )
        self.log("test_total_acc", self.test_total_acc, on_step=False, on_epoch=True)

    def predict_step(self, batch, batch_idx):
        image, y, image_name = batch
        logits = self.forward(image)
        _, y_pred = torch.topk(logits, self.num_pred)

    def get_topk_concepts_for_img(self, img, y, k=10):
        img = (img / img.norm(dim=-1, keepdim=True)).cpu()
        sim = img @ self.concept_features.cpu().T  # B, C
        attention = sim * self.classifier.weight[y].cpu()  # B, C, N
        if self.config.num_static_concept > 0:
            value, idx = torch.topk(
                attention[:, : self.config.num_static_concept], k=k, dim=-1
            )
            static_topk = self.conceptbank.get_static_concepts(idx=idx, dataframe=True)
            static_topk["value"] = value.view(-1).tolist()
            static_topk.to_csv(
                self.exp_root.joinpath("topk_static_img_concepts.csv"), index=False
            )

        if self.config.num_dynamic_concept > 0:
            value, idx = torch.topk(
                attention[:, self.config.num_static_concept :], k=k, dim=-1
            )
            dynamic_topk = self.conceptbank.get_dynamic_concepts(
                idx=idx, dataframe=True
            )
            dynamic_topk["value"] = value.view(-1).tolist()
            dynamic_topk.to_csv(
                self.exp_root.joinpath("topk_dynamic_img_concepts.csv"), index=False
            )

        value, idx = torch.topk(attention, k=k, dim=-1)
        hybrid_topk = self.conceptbank.get_concepts(idx=idx, dataframe=True)
        hybrid_topk["value"] = value.view(-1).tolist()
        hybrid_topk.to_csv(
            self.exp_root.joinpath("topk_hybrid_img_concepts.csv"), index=False
        )

    def save_topk_concepts_for_class(self):
        if self.config.num_static_concept > 0:
            k = min(10, self.config.num_static_concept)
            weight = self.classifier.weight[:, : self.config.num_static_concept].cpu()
            value, static_idx = torch.topk(weight, k=k)
            static_topk = self.conceptbank.get_static_concepts(
                idx=static_idx, dataframe=True
            )
            static_topk["value"] = value.view(-1).tolist()
            static_topk.to_csv(
                self.exp_root.joinpath("topk_static_concepts.csv"), index=False
            )
        if self.config.num_dynamic_concept > 0:
            k = min(10, self.config.num_dynamic_concept)
            weight = self.classifier.weight[:, self.config.num_static_concept :].cpu()
            value, dynamic_idx = torch.topk(weight, k=k)
            dynamic_topk = self.conceptbank.get_dynamic_concepts(
                idx=dynamic_idx, dataframe=True
            )
            dynamic_topk["value"] = value.view(-1).tolist()
            dynamic_topk.to_csv(
                self.exp_root.joinpath("topk_dynamic_concepts.csv"), index=False
            )

        weight = self.classifier.weight.cpu()
        value, idx = torch.topk(weight, k=20)
        hybrid_topk = self.conceptbank.get_concepts(idx=idx)
        hybrid_topk["value"] = value.view(-1).tolist()
        hybrid_topk.to_csv(
            self.exp_root.joinpath("topk_hybrid_concepts.csv"), index=False
        )

    def load_state_dict(
        self, state_dict: Mapping[str, Any], strict: bool = True, *args, **kwargs
    ) -> None:
        return super().load_state_dict(
            state_dict=state_dict, strict=False, *args, **kwargs
        )
