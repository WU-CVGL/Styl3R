from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from einops import rearrange
from torch import nn


from ..backbone.croco.croco import CroCoNet
from ..backbone.croco.misc import fill_default_args
from ..backbone.croco.patch_embed import get_patch_embed
from ..backbone.croco.blocks import Block, DecoderBlock

inf = float('inf')

croco_params = {
    'ViTLarge_BaseDecoder': {
        'enc_depth': 24,
        'dec_depth': 12,
        'enc_embed_dim': 1024,
        'dec_embed_dim': 768,
        'enc_num_heads': 16,
        'dec_num_heads': 12,
        'pos_embed': 'RoPE100',
        'img_size': (512, 512),
    },
}

@dataclass
class TokenStylizerCfg:
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    pretrained_weights: str = ""
    
class TokenStylizer(CroCoNet):
    """encode a style image, then concat it with content tokens
    and perform self-attention to commute information between them, 
    which outputs stylized gaussian appearance at last
    """
    
    def __init__(self, cfg: TokenStylizerCfg) -> None:
        
        self.patch_embed_cls = cfg.patch_embed_cls
        
        self.croco_args = fill_default_args(croco_params[cfg.model], CroCoNet.__init__)
       
        super().__init__(**croco_params[cfg.model])
    
    def _set_patch_embed(self, img_size=224, patch_size=16, enc_embed_dim=768, in_chans=3):
        self.patch_embed = get_patch_embed(self.patch_embed_cls, img_size, patch_size, enc_embed_dim, in_chans) 
    
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
                
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        # self.dec_blocks = nn.ModuleList([
        #     Block(dec_embed_dim, dec_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
        #     for i in range(dec_depth)]
        # cross attention
        self.dec_blocks = nn.ModuleList([
            DecoderBlock(dec_embed_dim, dec_num_heads, mlp_ratio=mlp_ratio, qkv_bias=True, norm_layer=norm_layer, norm_mem=norm_im2_in_dec, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)
    
    def _set_prediction_head(self, *args, **kwargs):
        """ No prediction head """
        return
    
    def _encode_image(self, image, true_shape):
        # embed the image into patches  (x has size B x Npatches x C)
        x, pos = self.patch_embed(image, true_shape=true_shape)

        # add positional embedding without cls token
        assert self.enc_pos_embed is None

        # now apply the transformer encoder and normalization
        for blk in self.enc_blocks:
            x = blk(x, pos)

        x = self.enc_norm(x)  # com: final norm
        return x, pos, None

    # def _decoder(self, out, pos, feat1, pos1, feat2, pos2):
    #     final_output = [(feat1, feat2)]
        
    #     out = self.decoder_embed(out)
    #     feat1 = self.decoder_embed(feat1)
    #     feat2 = self.decoder_embed(feat2)
        
    #     dec_in = torch.cat((out, feat1, feat2), dim=1)
    #     dec_in_pos = torch.cat((pos, pos1, pos2), dim=1)
        
    #     cutoff = out.shape[1]
        
    #     for blk in self.dec_blocks:
    #         dec_in = blk(dec_in, dec_in_pos)
    #         feat1, feat2 = dec_in[:, cutoff:, :].chunk(2, dim=1)
    #         final_output.append((feat1, feat2))
        
    #     dec_in = self.dec_norm(dec_in)
    #     final_output[-1] = dec_in[:, cutoff:, :].chunk(2, dim=1)
        
    #     return zip(*final_output)
    
    def _decoder(self, style_feat, style_pos, content_feat, content_pos):
        
        b, v, l, c = content_feat.shape
        
        final_output = [content_feat]
        
        content_feat = rearrange(content_feat, "b v l c -> b (v l) c")
        content_pos = rearrange(content_pos, "b v l c -> b (v l) c")
        
        style_feat = self.decoder_embed(style_feat)
        content_feat = self.decoder_embed(content_feat)        
        
        for blk in self.dec_blocks:
            content_feat, _ = blk(content_feat,
                                  style_feat,
                                  content_pos,
                                  style_pos)
            final_output.append(rearrange(content_feat, "b (v l) c -> b v l c", v=v, l=l))
            
        content_feat = self.dec_norm(content_feat)
        final_output[-1] = rearrange(content_feat, "b (v l) c -> b v l c", v=v, l=l)
        
        return final_output
    
    def forward(self,
                style: dict,
                content_feat: torch.tensor,
                content_pos: torch.tensor,
                ):
        b, _, h, w = style['image'].shape
        device = style['image'].device
        
        style_img = style['image']
        
        true_shape = style.get('true_shape', torch.tensor(style_img.shape[-2:])[None].repeat(b, 1))
        
        style_feat, style_pos, _ = self._encode_image(style_img, true_shape)

        dec = self._decoder(style_feat, style_pos, content_feat, content_pos)
        
        dec = list(dec)
        for i in range(len(dec)):
            dec[i] = dec[i][:, :, :-1]
        
        return dec
    
    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024    