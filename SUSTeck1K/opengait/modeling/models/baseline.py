import torch

from ..base_model import BaseModel
from ..modules import SetBlockWrapper, HorizontalPoolingPyramid, PackSequenceWrapper, \
    SeparateFCs, SeparateBNNecks, PointNet2Encoder

from einops import rearrange

class Baseline(BaseModel):

    def build_network(self, model_cfg):
        self.Backbone = self.get_backbone(model_cfg['backbone_cfg'])
        self.Backbone = SetBlockWrapper(self.Backbone)
        self.FCs = SeparateFCs(**model_cfg['SeparateFCs'])
        self.BNNecks = SeparateBNNecks(**model_cfg['SeparateBNNecks'])
        self.TP = PackSequenceWrapper(torch.max)
        self.HPP = HorizontalPoolingPyramid(bin_num=model_cfg['bin_num'])
        # self.Pointencoder = PointNet2Encoder()

    def forward(self, inputs):
        ipts, labs, _, _, seqL = inputs
        
        height=ipts[0]
        sils = ipts[1]
        # sils = sils.permute(0,1,4,2,3)
        if len(sils.size()) == 4:
            sils = sils.unsqueeze(1)
        else:
            sils = rearrange(sils, 'n s c h w -> n c s h w')

        warp = ipts[2]
        if len(warp.size()) == 4:
            warp = warp.unsqueeze(1)
        else:
            warp = rearrange(warp, 'n s c h w -> n c s h w')

        del ipts
        
        n,c,s,h,w = sils.shape
        heights_cal = height - 1
        heights_cal=heights_cal.unsqueeze(-1).unsqueeze(-1).unsqueeze(-1).unsqueeze(-1)
        heights_cal = heights_cal.repeat(1,1,s,h,w)
        
        in_ = torch.cat((sils, warp, heights_cal.float()), dim=1)
        outs = self.Backbone(in_)  # [n, c, s, h, w]

        # Temporal Pooling, TP
        outs = self.TP(outs, seqL, options={"dim": 2})[0]  # [n, c, h, w] (B,1,1024)
        # Horizontal Pooling Matching, HPM
        feat = self.HPP(outs)  # [n, c, p]
        # feat = outs.squeeze(1).unsqueeze(-1).repeat(1,1,16)
        embed_1 = self.FCs(feat)  # [n, c, p]
        embed_2, logits = self.BNNecks(embed_1)  # [n, c, p]
        embed = embed_1

        retval = {
            'training_feat': {
                'triplet': {'embeddings': embed_1, 'labels': labs},
                'softmax': {'logits': logits, 'labels': labs}
            },
            'visual_summary': {
                'image/sils': rearrange(sils,'n c s h w -> (n s) c h w')
            },
            'inference_feat': {
                'embeddings': embed
            }
        }
        return retval