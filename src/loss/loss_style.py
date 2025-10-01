import torch
import torch.nn as nn
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms
from dataclasses import dataclass
from src.test.vgg_model import VGGEncoder, calc_mean_std

from src.dataset.types import BatchedExample
from src.model.types import Gaussians
from src.loss import Loss
from src.model.decoder.decoder import DecoderOutput
from src.misc.nn_module_tools import convert_to_buffer

@dataclass
class LossStyleCfg:
    style_weight: float

@dataclass
class LossStyleCfgWrapper:
    style: LossStyleCfg

# TODO: test
class LossStyle(Loss[LossStyleCfg, LossStyleCfgWrapper]):
    
    def __init__(self, cfg: LossStyleCfgWrapper) -> None:
        super().__init__(cfg)
        
        self.vgg = VGGEncoder()
        convert_to_buffer(self.vgg, persistent=False)
        
        self.preprocess = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    
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
        
        # [0, 1] -> ImageNet mean/std
        target_img = self.preprocess(target_img)
        pred_img = self.preprocess(pred_img)
        style_img = self.preprocess(style_img)
        
        # duplicate style images to (b, v, c, h, w)
        style_img = repeat(style_img, 'b c h w -> b v c h w', v=v)
        style_img = rearrange(style_img, 'b v c h w -> (b v) c h w')
        
        # extract vgg feature maps for all
        pred_img_features = self.vgg(pred_img)
        target_img_features = self.vgg(target_img)
        style_img_features = self.vgg(style_img)
        
        # calculate content loss with pred_img and target_img
        # pred_img_h4 = self.vgg(pred_img, output_last_feature=True)
        # target_img_h4 = self.vgg(target_img, output_last_feature=True)
        # content_loss = nn.functional.mse_loss(pred_img_h4, target_img_h4)
        
        # use second last layer for content loss, which yields the best results on a single scene for now
        content_loss = 0
        content_loss += nn.functional.mse_loss(pred_img_features[-2], target_img_features[-2])
        content_loss += nn.functional.mse_loss(pred_img_features[-1], target_img_features[-1])
        
        # calculate style loss with pred_img and style_img
        style_loss = 0
        for pred_img_feature, style_img_feature in zip(pred_img_features, style_img_features):
            pred_img_feature_mean, pred_img_feature_std = calc_mean_std(pred_img_feature)
            style_img_feature_mean, style_img_feature_std = calc_mean_std(style_img_feature)
            style_loss += nn.functional.mse_loss(pred_img_feature_mean, style_img_feature_mean) \
                    + nn.functional.mse_loss(pred_img_feature_std, style_img_feature_std)
                
        return content_loss + self.cfg.style_weight * style_loss