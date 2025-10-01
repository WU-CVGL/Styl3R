from copy import deepcopy
from dataclasses import dataclass
from typing import Literal

import torch
from torch import nn
from functools import partial

from ..backbone.croco.pos_embed import get_2d_sincos_pos_embed, RoPE2D
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
class StructureBuilderCfg:
    model: Literal["ViTLarge_BaseDecoder", "ViTBase_SmallDecoder", "ViTBase_BaseDecoder"]  # keep interface for the last two models, but they are not supported
    patch_embed_cls: str = 'PatchEmbedDust3R'  # PatchEmbedDust3R or ManyAR_PatchEmbed
    pretrained_weights: str = ""
    
class StructureBuilder(nn.Module):
    """encode a style image, then concat it with content tokens
    and perform self-attention to commute information between them, 
    which outputs stylized gaussian appearance at last
    """
    
    def __init__(self, cfg: StructureBuilderCfg) -> None:
        
        super(StructureBuilder, self).__init__()
        
        params = croco_params[cfg.model]
        
        pos_embed = params['pos_embed']
        enc_embed_dim = params['enc_embed_dim']
        self.enc_embed_dim = enc_embed_dim
        dec_embed_dim = params['dec_embed_dim']
        dec_num_heads = params['dec_num_heads']
        dec_depth = params['dec_depth']
        mlp_ratio = 4
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        norm_im2_in_dec=True
        
        if pos_embed=='cosine':
            # positional embedding of the encoder 
            enc_pos_embed = get_2d_sincos_pos_embed(enc_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
            self.register_buffer('enc_pos_embed', torch.from_numpy(enc_pos_embed).float())
            # positional embedding of the decoder  
            dec_pos_embed = get_2d_sincos_pos_embed(dec_embed_dim, int(self.patch_embed.num_patches**.5), n_cls_token=0)
            self.register_buffer('dec_pos_embed', torch.from_numpy(dec_pos_embed).float())
            # pos embedding in each block
            self.rope = None # nothing for cosine 
        elif pos_embed.startswith('RoPE'): # eg RoPE100 
            self.enc_pos_embed = None # nothing to add in the encoder with RoPE
            self.dec_pos_embed = None # nothing to add in the decoder with RoPE
            if RoPE2D is None: raise ImportError("Cannot find cuRoPE2D, please install it following the README instructions")
            freq = float(pos_embed[len('RoPE'):])
            self.rope = RoPE2D(freq=freq)
        else:
            raise NotImplementedError('Unknown pos_embed '+pos_embed)
        
        # decoder 
        self._set_decoder(enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec)
        
        # initializer weights
        self.initialize_weights()
    
    def _set_decoder(self, enc_embed_dim, dec_embed_dim, dec_num_heads, dec_depth, mlp_ratio, norm_layer, norm_im2_in_dec):
                
        self.dec_depth = dec_depth
        self.dec_embed_dim = dec_embed_dim
        # transfer from encoder to decoder
        self.decoder_embed = nn.Linear(enc_embed_dim, dec_embed_dim, bias=True)
        # transformer for the decoder
        self.dec_blocks = nn.ModuleList([
            Block(dec_embed_dim, dec_num_heads, mlp_ratio, qkv_bias=True, norm_layer=norm_layer, rope=self.rope)
            for i in range(dec_depth)])
        # final norm layer
        self.dec_norm = norm_layer(dec_embed_dim)
    
    def initialize_weights(self):
        # linears and layer norms
        self.apply(self._init_weights)
    
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            # we use xavier_uniform following official JAX ViT:
            torch.nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _decoder(self, feat1, pos1, feat2, pos2):
        final_output = [(feat1, feat2)]
        
        feat1 = self.decoder_embed(feat1)
        feat2 = self.decoder_embed(feat2)
        
        dec_in = torch.cat((feat1, feat2), dim=1)
        dec_in_pos = torch.cat((pos1, pos2), dim=1)
                
        for blk in self.dec_blocks:
            dec_in = blk(dec_in, dec_in_pos)
            feat1, feat2 = dec_in.chunk(2, dim=1)
            final_output.append((feat1, feat2))
        
        dec_in = self.dec_norm(dec_in)
        final_output[-1] = dec_in.chunk(2, dim=1)
        
        return zip(*final_output)
    
    def forward(self,
                feat1: torch.tensor,
                pos1: torch.tensor,
                feat2: torch.tensor,
                pos2: torch.tensor
                ):

        dec1, dec2 = self._decoder(feat1, pos1, feat2, pos2)
        
        dec1, dec2 = list(dec1), list(dec2)
        for i in range(len(dec1)):
            dec1[i] = dec1[i][:, :-1]
            dec2[i] = dec2[i][:, :-1]
        
        return dec1, dec2
        
    @property
    def patch_size(self) -> int:
        return 16

    @property
    def d_out(self) -> int:
        return 1024    