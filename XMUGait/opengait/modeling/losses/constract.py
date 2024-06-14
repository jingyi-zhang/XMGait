import torch
import torch.nn.functional as F
from .base import BaseLoss, gather_and_scale_wrapper


class ContrastLoss(BaseLoss):
    def __init__(self, loss_term_weight):
        super(ContrastLoss, self).__init__(loss_term_weight)
    
    # @gather_and_scale_wrapper   
    def forward(self, image_features, logit_scale=0.1):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, image_features,logit_scale)
        labels = torch.arange(logits_per_image.shape[0], device=device,
                              dtype=torch.long)
        total_loss = (
                             F.cross_entropy(logits_per_image, labels) +
                             F.cross_entropy(logits_per_text, labels)
                     ) / 2
        if torch.isnan(total_loss).any():
            print('555')
        self.info.update({"contrastive_loss": total_loss.detach().clone()})
        return total_loss, self.info
    
    def get_logits(self, image_features, text_features, logit_scale):
        # 计算image_features @ text_features.T相似度矩阵
        logits_per_image = logit_scale * image_features @ text_features.T
        logits_per_text = logit_scale * text_features @ image_features.T
        return logits_per_image, logits_per_text
