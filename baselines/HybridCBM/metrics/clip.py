import os.path

import torch
import matplotlib.pyplot as plt
from .prompts import get_class_from_dataset
from models.clip import ClipEncoder


def heatmap(
    sim_matrix, path, xlabel="Class Index", ylabel="Concept Index", figsize=(10, 8)
):
    plt.figure(figsize=figsize)
    plt.imshow(sim_matrix, cmap="coolwarm", aspect="auto")
    cbar = plt.colorbar(label="Correlation Coefficient", shrink=0.8)
    cbar.ax.tick_params(labelsize=15)

    # 添加图例和标签
    plt.title("Correlation Heatmap", fontsize=20, weight="bold")
    plt.xlabel(xlabel, fontsize=20)
    plt.ylabel(ylabel, fontsize=20)
    plt.xticks(fontsize=15)
    plt.yticks(fontsize=15)

    # 调整布局使图例和标签显示更整齐
    plt.tight_layout()
    plt.show()
    os.makedirs(os.path.dirname(path), exist_ok=True)
    plt.savefig(path, dpi=300, bbox_inches="tight", transparent=True, format="svg")


class ClipEvaluator:
    def __init__(self, clip_encoder: ClipEncoder, dataset, classEmbeddings=None):
        self.classes, labels = get_class_from_dataset(dataset)
        self.clip_encoder = clip_encoder
        self.class2label = {c: l for c, l in zip(self.classes, labels)}
        self.label2class = {l: c for c, l in zip(self.classes, labels)}
        self._classEmbeddings = classEmbeddings

    @property
    def classEmbeddings(self):
        if self._classEmbeddings is not None:
            return self._classEmbeddings
        else:
            self._classEmbeddings = self.clip_encoder.encode_text(
                [self.label2class[i] for i in range(len(self.classes))],
                batch_size=128,
                normalize=True,
            ).cpu()
            return self._classEmbeddings

    @classEmbeddings.setter
    def classEmbeddings(self, value):
        self._classEmbeddings = value

    @torch.no_grad()
    def evaluate(self, concepts, batch_size=128, to_label=False, argmax=False):
        if isinstance(concepts, str):
            concepts = [concepts]
        if isinstance(concepts, torch.Tensor):
            concepts = concepts / concepts.norm(dim=-1, keepdim=True)
        else:
            concepts = self.clip_encoder.encode_text(
                concepts, batch_size=batch_size, normalize=True
            )
        self.classEmbeddings = self.classEmbeddings.to(concepts.device)
        cos = concepts @ self.classEmbeddings.T
        if argmax:
            _, outputs = cos.max(dim=1)
        else:
            outputs = cos
        if to_label:
            outputs = [self.label2class[p.item()] for p in outputs]
        if to_label:
            return outputs
        return outputs.cpu()

    def data_alignment_score(self, concepts, batch_size=128):
        align_score = self.evaluate(concepts, batch_size=batch_size).max(dim=-1)[0]
        return align_score.float().mean().item()

    def concept_purity(self, concepts, batch_size=128):
        align_score = self.evaluate(concepts, batch_size=batch_size)
        # num_class, num_class
        align_score = align_score.reshape(
            len(self.class2label), -1, align_score.shape[-1]
        ).mean(dim=1)
        # 取对角线，即同类别的对齐分数
        align_score = align_score.diagonal().mean().item()
        return align_score

    def concept_coverage(self, concepts, batch_size=128):
        align_score = self.evaluate(concepts, batch_size=batch_size)
        align_score = align_score.reshape(
            len(self.class2label), -1, align_score.shape[-1]
        ).mean(dim=1)
        align_score = 1 - torch.triu(align_score, diagonal=1).mean().item()
        return align_score

    def text_class_similarity_acc(self, concepts, batch_size=128):
        pred = (
            self.evaluate(concepts, batch_size=batch_size)
            .max(dim=-1)[-1]
            .reshape(len(self.class2label), -1)
        )
        target = (
            torch.arange(len(self.class2label))
            .reshape(-1, 1)
            .expand_as(pred)
            .to(pred.device)
        )
        acc = (pred == target).float().max(-1)[0].mean().item()
        return acc

    def img_similarity(self, image, label, features):
        image = image / image.norm(dim=-1, keepdim=True)
        features = features / features.norm(dim=-1, keepdim=True)
        score = image @ features.T
        print(score.shape, image.shape, features.shape)
        score = score.reshape(score.shape[0], len(self.class2label), -1).mean(
            dim=-1
        )  # shape (batch, 200)
        mask = torch.zeros_like(score, device=score.device, dtype=torch.float)
        mask[torch.arange(score.shape[0]), label] = 1
        purity = score[mask == 1].cpu()
        coverage = 1 - score[mask == 0].reshape(score.shape[0], -1).mean(dim=-1).cpu()
        return purity, coverage, score.argmax(dim=-1).cpu()

    def img_alignment_score(self, dataloader):
        score_list = []
        for image, label in dataloader:
            score, _ = self.img_similarity(image, label, self.classEmbeddings)
            score_list.append(score)
        score = torch.cat(score_list).mean().item()
        return score

    def img_similarity_acc(self, dataloader):
        pred_list = []
        for image, label in dataloader:
            _, pred = self.img_similarity(image, label, self.classEmbeddings)
            pred_list.append(pred)
        pred = torch.cat(pred_list)
        target = torch.cat([label for _, label in dataloader])
        acc = (pred == target).float().mean().item()
        return acc

    def compute_class_metrics(self, features):
        data_alignment_score = self.data_alignment_score(features)
        concept_purity_score = self.concept_purity(features)
        concept_coverage_score = self.concept_coverage(features)
        text_class_similarity_acc = self.text_class_similarity_acc(features)
        return dict(
            data_alignment_score=data_alignment_score,
            concept_purity_score=concept_purity_score,
            concept_coverage_score=concept_coverage_score,
            class_alignment_acc=text_class_similarity_acc,
        )

    def correlation_heatmap(self, exp_root, concepts, prefix=""):
        if concepts.shape[0] == 0:
            return
        sim_matrix = self.evaluate(concepts).reshape(
            len(self.class2label), -1, len(self.class2label)
        )
        sim_matrix_max = sim_matrix.max(dim=1)[0].numpy()
        sim_matrix_min = sim_matrix.min(dim=1)[0].numpy()
        sim_matrix_mean = sim_matrix.mean(dim=1).numpy()
        heatmap(
            sim_matrix.reshape(-1, len(self.class2label)).numpy(),
            os.path.join(exp_root, "heatmap", f"{prefix}-concept-class.svg"),
            xlabel="Class Index",
            ylabel="Concept Index",
        )
        heatmap(
            sim_matrix_max,
            os.path.join(exp_root, "heatmap", f"{prefix}-concept-class-max.svg"),
            xlabel="Class Index",
            ylabel="Concept Index",
        )
        heatmap(
            sim_matrix_min,
            os.path.join(exp_root, "heatmap", f"{prefix}-concept-class-min.svg"),
            xlabel="Class Index",
            ylabel="Concept Index",
        )
        heatmap(
            sim_matrix_mean,
            os.path.join(exp_root, "heatmap", f"{prefix}-concept-class-mean.svg"),
            xlabel="Class Index",
            ylabel="Concept Index",
        )

    def self_correlation_heatmap(self, exp_root, concepts, prefix=""):
        if concepts.shape[0] == 0:
            return
        heatmap(
            concepts @ concepts.T,
            os.path.join(exp_root, "heatmap", f"{prefix}-self-concept.svg"),
            xlabel="Concept Index",
            ylabel="Concept Index",
        )
        heatmap(
            self.classEmbeddings @ self.classEmbeddings.T,
            os.path.join(exp_root, "heatmap", f"{prefix}-self-Class.svg"),
            xlabel="Class Index",
            ylabel="Class Index",
        )
