# -*- coding: utf-8 -*-
# @Author  : jingyi
import torch.nn.functional as F

from .base import BaseLoss


class L1Loss(BaseLoss):
    def __init__(self, loss_term_weight=1.0, log_accuracy=False):
        super(L1Loss, self).__init__(loss_term_weight)
        self.log_accuracy = log_accuracy

    def forward(self, predict_height, gt_height):
        """
            logits: [n, c, p]
            labels: [n]
        """
        b, t, _ = predict_height.shape
        # print(predict_height)
        loss = F.l1_loss(predict_height, gt_height.unsqueeze(1).unsqueeze(1).repeat(1, t, 1))
        self.info.update({'loss': loss.detach().clone()})
        if self.log_accuracy:
            pred = predict_height.mean(dim=1).squeeze().detach().clone()  # [n, p]
            gt_height = gt_height.detach().clone()
            min_acc =(pred>=gt_height-0.05)
            max_acc = (pred<=gt_height+0.05)
            accu = (min_acc * max_acc).float().mean()
            self.info.update({'accuracy': accu})
        return loss, self.info
    