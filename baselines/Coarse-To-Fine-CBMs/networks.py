import torch
import torch.nn as nn

from torch.distributions.bernoulli import Bernoulli
from utils import bin_concrete_sample, concrete_sample


class LinearClassifier(nn.Module):
    """Linear classifier"""

    def __init__(self, feat_dim, num_classes=10):
        super(LinearClassifier, self).__init__()
        self.W = nn.Parameter(torch.Tensor(feat_dim, num_classes))
        self.bias = nn.Parameter(torch.Tensor(num_classes))

        torch.nn.init.xavier_normal_(self.W)
        self.bias.data.fill_(0.1)

    def forward(self, features, mask=None):

        if mask is not None:
            features = features * mask

        out = nn.functional.linear(features, self.W.T, self.bias)

        return out


def bernoulli_kl(p, q, eps=1e-7):
    return (p * ((p + eps).log() - (q + eps).log())) + (1. - p) * ((1. - p + eps).log() - (1. - q + eps).log())

def categorical_kl(p, q, eps = 1e-7):
    p = nn.functional.softmax(p, -1)
    log_q = torch.log(q + eps)
    log_p = torch.log(p + eps)

    kl = torch.sum(p * (log_p - log_q), -1).mean()

    return kl

class DiscoveryMechanism(nn.Module):

    def __init__(self, feat_dim, cdim, prior, mode = 'bernoulli'):
        super(DiscoveryMechanism, self).__init__()
        self.W = nn.Parameter(torch.Tensor(feat_dim, cdim))
        self.bias = nn.Parameter(torch.Tensor(cdim))

        self.mode = mode

        self.register_buffer('temp', torch.tensor(0.1))
        self.register_buffer('temptest', torch.tensor(.01))

        self.register_buffer('prior', torch.tensor(prior))

        # init parameters
        torch.nn.init.xavier_normal_(self.W)
        self.bias.data.fill_(0.0)

    def forward(self, features, probs_only=False):

        logits = nn.functional.linear(features, self.W.T, self.bias)

        kl = 0.

        if self.training:
            if self.mode == 'bernoulli':
                out = bin_concrete_sample(logits, self.temp).clamp(1e-6, 1.-1e-6)
                kl += bernoulli_kl(torch.sigmoid(logits), self.prior).sum(1).mean()
            else:
                out = concrete_sample(logits, self.temp)
                kl += categorical_kl(logits, self.prior)
        else:

            if self.mode == 'bernoulli':
                if probs_only:
                    out = torch.sigmoid(logits)
                else:
                    out = Bernoulli( logits=logits).sample()
            else:
                if probs_only:
                    out = nn.functional.softmax(logits)
                else:
                    out = concrete_sample(logits, self.temptest)

        return out, kl

