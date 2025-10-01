import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms
import torch.nn.functional as F
from dataclasses import dataclass
from typing import List

from src.test.vgg_model import VGGEncoder, calc_mean_std

from src.dataset.types import BatchedExample
from src.model.types import Gaussians
from src.loss import Loss
from src.model.decoder.decoder import DecoderOutput
from src.misc.nn_module_tools import convert_to_buffer
from src.model.encoder.stylizer.vgg import NormalizedVGG
from src.model.encoder.stylizer.stylizer import AdaIN, AdaAttN

@dataclass
class LossAdaAttNCfg:
    lam: float
    content_loss_layers: List[int] 
    style_loss_layers: List[int] 
    style_loss_stats: List[str] 

@dataclass
class LossAdaAttNCfgWrapper:
    adaattn: LossAdaAttNCfg

class PixelLoss(nn.Module):
    """ pixel-wise loss """

    def __init__(self, loss_type='l1', reduction='mean'):
        super(PixelLoss, self).__init__()

        if loss_type == 'l1':
            self.criterion = nn.L1Loss(reduction=reduction)
        elif loss_type == 'smooth_l1':
            self.criterion = nn.SmoothL1Loss(reduction=reduction)
        elif loss_type == 'huber':
            self.criterion = nn.HuberLoss(reduction=reduction)
        elif loss_type == 'mse':
            self.criterion = nn.MSELoss(reduction=reduction)
        else:
            raise NotImplementedError(
                '[ERROR] invalid loss type: {:s}'.format(loss_type)
            )

        self.reduction = reduction

    def forward(self, pred, target=None):
        if target is None:
            target = torch.zeros_like(pred)
        loss = self.criterion(pred, target)
        if self.reduction == 'sum':
            loss /= len(pred)
        return loss

class VGGContentLoss(nn.Module):
    """ VGG content loss """

    def __init__(self, loss_type='l1', layers=[3], norm=None, 
                 reduction='mean'):
        super(VGGContentLoss, self).__init__()
        
        self.criterion = PixelLoss(loss_type, reduction=reduction)

        for l in layers:
            assert l in (1, 2, 3, 4, 5), \
                '[ERROR] invalid VGG layer: {:s}'.format(l)
        self.layers = layers

        self.adain = self.adaattn = None
        if norm is not None:
            if norm == 'adain':
                self.adain = AdaIN()
            elif norm == 'adaattn':
                self.adaattn = AdaAttN()
            else:
                raise NotImplementedError(
                    '[ERROR] invalid content normalization: {:s}'.format(norm)
                )
        self.norm = norm

    def forward(self, pred_feats, content_feats, style_feats=None):
        if self.norm is not None:
            assert style_feats is not None, \
                '[ERROR] style features must be given for AdaAttN evaluation'
            if self.norm == 'adaattn':
                q, k = content_feats[0], style_feats[0]
        
        loss = 0
        for i in range(len(pred_feats)):
            p, c = pred_feats[i], content_feats[i]

            # accumulate query and key features for AdaAttN
            if self.norm == 'adaattn' and i > 0:
                s = style_feats[i]
                q = F.interpolate(
                    q, size=c.shape[-2:], mode='bilinear', align_corners=False
                )
                k = F.interpolate(
                    k, size=s.shape[-2:], mode='bilinear', align_corners=False
                )
                q = torch.cat([q, c], 1)
                k = torch.cat([k, s], 1)
            
            if i + 1 in self.layers:
                if self.norm == 'adain':
                    c = self.adain(c, style_feats[i])
                if self.norm == 'adaattn':
                    c = self.adaattn(q, k, c, style_feats[i])
                loss += self.criterion(p, c)
        return loss

class VGGStyleLoss(nn.Module):
    """ VGG style loss """

    def __init__(self, loss_type='mse', layers=[1, 2, 3], 
                 stats=['mean', 'gram'], reduction='sum'):
        super(VGGStyleLoss, self).__init__()

        self.criterion = PixelLoss(loss_type, reduction=reduction)

        for l in layers:
            assert l in (1, 2, 3, 4, 5), \
                '[ERROR] invalid VGG layer: {:s}'.format(l)
        self.layers = layers

        for s in stats:
            assert s in ('mean', 'std', 'gram'), \
                '[ERROR] invalid style statistic: {:s}'.format(s)
        self.stats = stats

    def _gram(self, x):
        bs, c, h, w = x.size()
        x = x.view(bs, c, h * w)
        gram = torch.bmm(x, x.transpose(2, 1)) / (c * h * w)
        return gram

    def forward(self, pred_feats, style_feats):
        loss = 0
        for l in self.layers:
            p, s = pred_feats[l - 1], style_feats[l - 1]
            if 'mean' in self.stats:
                loss += self.criterion(p.mean((-2, -1)), s.mean((-2, -1)))
            if 'std' in self.stats:
                loss += self.criterion(p.std((-2, -1)), s.std((-2, -1)))
            if 'gram' in self.stats:
                loss += self.criterion(self._gram(p), self._gram(s))
        return loss

# TODO: test
class LossAdaAttN(Loss[LossAdaAttNCfg, LossAdaAttNCfgWrapper]):
    
    def __init__(self, cfg: LossAdaAttNCfgWrapper) -> None:
        super().__init__(cfg)
        
        self.vgg = NormalizedVGG()
        convert_to_buffer(self.vgg, persistent=False)
        
        self.vgg_content_loss = VGGContentLoss(loss_type='l1', layers=self.cfg.content_loss_layers, norm='adaattn', reduction='mean')
        self.vgg_style_loss = VGGStyleLoss(loss_type='mse', layers=self.cfg.style_loss_layers, stats=self.cfg.style_loss_stats, reduction='mean')
            
    def forward(
        self,
        prediction: DecoderOutput,
        batch: BatchedExample,
        gaussians: Gaussians,
        global_step: int,
    ) -> Float[Tensor, ""]:
        
        b, v, _, _, _ = batch["target"]["image"].shape
        target_img = rearrange(batch["target"]["image"], "b v c h w -> (b v) c h w")
        pred_img = rearrange(prediction.color, "b v c h w -> (b v) c h w")
        style_img = batch["style"]["image"] # (b, c, h, w)
        
        # duplicate style images to (b, v, c, h, w)
        style_img = repeat(style_img, 'b c h w -> b v c h w', v=v)
        style_img = rearrange(style_img, 'b v c h w -> (b v) c h w')
        
        pred_feats = self.vgg(pred_img)
        content_feats = self.vgg(target_img)
        style_feats = self.vgg(style_img)
        
        content_loss = self.vgg_content_loss(pred_feats, content_feats, style_feats)
        style_loss = self.vgg_style_loss(pred_feats, style_feats)
                
        return content_loss + self.cfg.lam * style_loss