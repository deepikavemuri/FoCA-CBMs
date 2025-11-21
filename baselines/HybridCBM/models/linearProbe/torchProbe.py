# -*- coding: utf-8 -*- 
"""
@Time : 2024/10/17 16:37
@File :     torchProbe.py 
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning as L
import torchmetrics


class TorchLogisticRegression(L.LightningModule):
    def __init__(self, num_class, in_dim, lr) -> None:
        super().__init__()
        self.probe = nn.Linear(in_dim, num_class)
        self.lr = lr
        self.save_hyperparameters()
        self.train_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)
        self.val_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)
        self.test_acc = torchmetrics.Accuracy(task='multiclass', num_classes=num_class)

    def forward(self, x):
        x = x / x.norm(dim=-1, keepdim=True)
        return self.probe(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(),
                                     lr=self.lr,
                                     betas=(0.9, 0.98),
                                     eps=1e-6,
                                     weight_decay=0.2)
        return optimizer

    def training_step(self, train_batch, batch_idx):
        image, y = train_batch
        y_pred = self.forward(image)
        loss = F.cross_entropy(y_pred, y)
        self.log('train_loss', loss)
        self.train_acc(y_pred, y)
        self.log('train_acc', self.train_acc, on_step=False, on_epoch=True)
        return loss

    def validation_step(self, val_batch, batch_idx):
        image, y = val_batch
        y_pred = self.forward(image)
        loss = F.cross_entropy(y_pred, y)
        self.val_acc(y_pred, y)
        self.log('val_acc', self.val_acc, on_step=False, on_epoch=True)
        self.log('val_loss', loss)
        return loss

    def test_step(self, batch, batch_idx):
        image, y = batch
        y_pred = self.forward(image)
        loss = F.cross_entropy(y_pred, y)
        self.test_acc(y_pred, y)
        self.log('test acc', self.test_acc, on_step=False, on_epoch=True)
        self.log('test loss', loss)
        return loss
