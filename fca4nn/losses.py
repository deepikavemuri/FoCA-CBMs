import torch


def binary_crossentropy(pred, target, eps=1e-8):
    loss = (target * torch.log(pred + eps)) + ((1 - target) * torch.log(1 - pred + eps))
    return -loss.sum(dim=-1)


def focal_loss(pred, target, eps=1e-8, gamma=2):
    weight = (1 - pred) ** gamma
    loss = (
        (target * torch.log(pred + eps)) + ((1 - target) * torch.log(1 - pred + eps))
    ) * weight
    return -loss.sum(dim=-1)


def hierarchical_ce_loss(pred, target, eps=1e-8):
    maxes = pred.max(dim=-1, keepdim=True)[0]
    pred = pred - maxes

    prob_group = torch.exp((pred * target).mean(dim=-1))
    pron_not_group = (torch.exp(pred) * (1 - target)).sum(dim=-1)

    loss = -torch.log(prob_group / (prob_group + pron_not_group + eps))
    return loss


def dice_loss(pred, target, eps=1e-8):
    pred = pred.view(pred.size(0), -1)
    target = target.view(target.size(0), -1)

    intersection = (pred * target).sum(dim=-1)
    loss = 1 - (2.0 * intersection + eps) / (
        pred.sum(dim=-1) + target.sum(dim=-1) + eps
    )
    return loss
