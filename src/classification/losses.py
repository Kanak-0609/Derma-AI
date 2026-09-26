"""
losses.py - DermaAI Classification Imbalance-Handling Losses

Two strategies to address the 58x class imbalance in HAM10000:
    1. compute_class_weights - inverse-frequency weights for CrossEntropyLoss
    2. FocalLoss - down-weights easy examples, focuses learning on hard/rare cases
"""

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F


def compute_class_weights(train_csv, class_names, label_col='diagnosis', method='inverse_freq'):
    """
    Computes per-class weights from the TRAINING set only (never val/test - would leak
    information about class balance in those sets into how the model is trained).

    method='inverse_freq': weight_c = total_samples / (num_classes * count_c)
        Standard inverse-frequency weighting - larger classes get smaller weight.
    """
    df = pd.read_csv(train_csv)
    counts = df[label_col].value_counts()

    total = len(df)
    num_classes = len(class_names)

    weights = []
    for name in class_names:
        count_c = counts.get(name, 0)
        if count_c == 0:
            weights.append(0.0)
        else:
            weights.append(total / (num_classes * count_c))

    weights = torch.tensor(weights, dtype=torch.float32)
    return weights


class FocalLoss(nn.Module):
    """
    Focal Loss (Lin et al., 2017): FL(p_t) = -alpha_t * (1 - p_t)^gamma * log(p_t)

    gamma > 0 reduces the loss contribution from easy (well-classified) examples,
    letting training focus on hard/rare examples.
    alpha (optional per-class weights) additionally re-balances class frequency,
    similar to weighted CrossEntropyLoss but combined with the focusing term.
    """
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha = alpha  # per-class weight tensor, or None
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        log_probs = F.log_softmax(logits, dim=1)
        probs = torch.exp(log_probs)

        targets_one_hot = F.one_hot(targets, num_classes=logits.size(1)).float()

        p_t = (probs * targets_one_hot).sum(dim=1)
        log_p_t = (log_probs * targets_one_hot).sum(dim=1)

        focal_weight = (1 - p_t) ** self.gamma

        if self.alpha is not None:
            alpha_t = self.alpha.to(logits.device)[targets]
            loss = -alpha_t * focal_weight * log_p_t
        else:
            loss = -focal_weight * log_p_t

        if self.reduction == 'mean':
            return loss.mean()
        elif self.reduction == 'sum':
            return loss.sum()
        else:
            return loss
