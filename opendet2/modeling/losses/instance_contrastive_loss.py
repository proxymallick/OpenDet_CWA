import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import numpy as np

class ICLoss(nn.Module):
    """ Instance Contrastive Loss
    """
    def __init__(self,num_known_classes, tau=0.1):
        super().__init__()
        self.tau = tau
        self.num_known_classes = num_known_classes


    def forward(self, features, labels, queue_features, queue_labels,scores,gt_classes):

        
        device = features.device
        mask = torch.eq(labels[:, None], queue_labels[:, None].T).float().to(device)
        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(features, queue_features.T), self.tau)

        # for numerical stability
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()
        logits_mask = torch.ones_like(logits)

        # mask itself
        logits_mask[logits == 0] = 0
        mask = mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask
        log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (mask * log_prob).sum(1) / mask.sum(1)
        
        # loss
        loss = - mean_log_prob_pos.mean()
        #loss =   loss  +  0.1 * anc_loss

        #loss = anc_loss
        # trick: avoid loss nan
        return loss if not torch.isnan(loss) else features.new_tensor(0.0)

