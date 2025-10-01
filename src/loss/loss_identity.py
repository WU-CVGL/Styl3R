import torch.nn as nn
from torchvision import transforms
from einops import rearrange, repeat
from jaxtyping import Float
from torch import Tensor

from src.test.vgg_model import VGGEncoder, calc_mean_std
from src.misc.nn_module_tools import convert_to_buffer
from src.model.decoder.decoder import DecoderOutput
from src.dataset.types import BatchedExample
from src.model.types import Gaussians

class IdentityLoss(nn.Module):
    
    def __init__(self, weight_1=70, weight_2=1):
        super().__init__()
        
        self.weight_1 = weight_1
        self.weight_2 = weight_2
        
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
        
        loss_identity1 = nn.functional.mse_loss(pred_img, target_img)
        
        pred_img = self.preprocess(pred_img)
        target_img = self.preprocess(target_img)
        
        pred_img_feats = self.vgg(pred_img)
        target_img_feats = self.vgg(target_img)
        
        loss_identity2 = 0
        for pred_img_feat, target_img_feat in zip(pred_img_feats, target_img_feats):
            loss_identity2 += nn.functional.mse_loss(pred_img_feat, target_img_feat)
        
        return loss_identity1 * self.weight_1 + loss_identity2 * self.weight_2