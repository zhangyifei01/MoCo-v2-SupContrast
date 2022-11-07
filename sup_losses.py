# ref PCL
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class SupLoss(nn.Module):
    def __init__(self, temperature=1.0, base_temperature=None, K=128):
        super(SupLoss, self).__init__()
        self.temperature = temperature
        self.base_temperature = temperature if base_temperature is None else base_temperature
        self.K = K

    def forward(self, features, labels=None):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        ss = features.shape[0]
        batch_size = (features.shape[0] - self.K) // 2

        labels = labels.contiguous().view(-1, 1)
        mask = torch.eq(labels[:batch_size], labels.T).float().to(device)

        # TODO: label is -1
        # mask[torch.where(labels < 0)[0]] = 0

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features[:batch_size], features.T),
            self.temperature)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(mask),
            1,
            torch.arange(batch_size).view(-1, 1).to(device),
            0
        )

        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12)

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)

        # loss
        loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.mean()

        return loss
