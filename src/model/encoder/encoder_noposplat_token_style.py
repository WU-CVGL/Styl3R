from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional
from contextlib import nullcontext

import torch
import torch.nn.functional as F
from einops import rearrange
from jaxtyping import Float
from torch import Tensor, nn

from .backbone.croco.misc import transpose_to_landscape
from .heads import head_factory
from ...dataset.shims.bounds_shim import apply_bounds_shim
from ...dataset.shims.normalize_shim import apply_normalize_shim
from ...dataset.shims.patch_shim import apply_patch_shim
from ...dataset.types import BatchedExample, DataShim
from ...geometry.projection import sample_image_grid
from ..types import Gaussians
from .backbone import Backbone, BackboneCfg, get_backbone
from .common.gaussian_adapter import GaussianAdapter, GaussianAdapterCfg, UnifiedGaussianAdapter
from .encoder import Encoder
from .visualization.encoder_visualizer_epipolar_cfg import EncoderVisualizerEpipolarCfg

from .token_stylizer.token_stylizer import TokenStylizerCfg, TokenStylizer
from .token_stylizer.structure_builder import StructureBuilder, StructureBuilderCfg

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatTokenStyleCfg:
    name: Literal["noposplat_token_style", "noposplat", "noposplat_multi", "noposplat_multi_token_style"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    token_stylizer: TokenStylizerCfg
    structure_builder: StructureBuilderCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    gs_sh_head_type: str
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    stylized: bool = False


def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNoPoSplatTokenStyle(Encoder[EncoderNoPoSplatTokenStyleCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatTokenStyleCfg) -> None:
        super().__init__(cfg)

        self.backbone = get_backbone(cfg.backbone, 3)
        
        self.pose_free = cfg.pose_free
        if self.pose_free:
            self.gaussian_adapter = UnifiedGaussianAdapter(cfg.gaussian_adapter)
        else:
            self.gaussian_adapter = GaussianAdapter(cfg.gaussian_adapter)

        self.patch_size = self.backbone.patch_embed.patch_size[0]
        self.raw_gs_dim = 1 + self.gaussian_adapter.d_in  # 1 for opacity

        self.gs_params_head_type = cfg.gs_params_head_type
        
        self.stylized = cfg.stylized
        
        self.structure_builder = StructureBuilder(cfg.structure_builder)
        self.token_stylizer = TokenStylizer(cfg.token_stylizer)
        
        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                           depth_mode=('exp', -inf, inf), conf_mode=None,)
        
        # define gaussian struture head (scale, opacity, covariance)
        self.gaussian_structure_head = head_factory(
                "dpt_gs_sh", 'gs_params', self.structure_builder, has_conf=False, out_nchan=self.raw_gs_dim - 3 * self.gaussian_adapter.d_sh)
        
        # define gaussian appearance head
        self.gs_sh_head_type = cfg.gs_sh_head_type
        if self.gs_sh_head_type == "linear":
            self.gaussian_appearance_head = nn.Sequential(
                # nn.ReLU(),
                nn.Linear(
                    self.token_stylizer.dec_embed_dim,
                    cfg.num_surfaces * self.patch_size ** 2 * 3 * self.gaussian_adapter.d_sh,
                ),
            )
        elif self.gs_sh_head_type == "dpt":
            self.gaussian_appearance_head = head_factory(
                "dpt_gs_sh", 'gs_params', self.token_stylizer, has_conf=False, out_nchan=3 * self.gaussian_adapter.d_sh)
        
    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        # TODO: use structure builder to init downstream head
        self.token_stylizer.depth_mode = depth_mode
        self.token_stylizer.conf_mode = conf_mode
        
        self.structure_builder.depth_mode = depth_mode
        self.structure_builder.conf_mode = conf_mode
        
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.structure_builder, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)

    def map_pdf_to_opacity(
        self,
        pdf: Float[Tensor, " *batch"],
        global_step: int,
    ) -> Float[Tensor, " *batch"]:
        # https://www.desmos.com/calculator/opvwti3ba9

        # Figure out the exponent.
        cfg = self.cfg.opacity_mapping
        x = cfg.initial + min(global_step / cfg.warm_up, 1) * (cfg.final - cfg.initial)
        exponent = 2**x

        # Map the probability density to an opacity.
        return 0.5 * (1 - (1 - pdf) ** exponent + pdf ** (1 / exponent))

    def _downstream_head(self, head_num, decout, img_shape, ray_embedding=None):
        B, S, D = decout[-1].shape
        # img_shape = tuple(map(int, img_shape))
        head = getattr(self, f'head{head_num}')
        return head(decout, img_shape, ray_embedding=ray_embedding)

    def forward(
        self,
        context: dict,
        style: dict,
        global_step: int = 0,
        visualization_dump: Optional[dict] = None,
        distill_only: bool = False,
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        feat1, pos1, feat2, pos2, shape1, shape2, view1, view2 = self.backbone(context)

        if distill_only:
            structure_token1, structure_token2 = self.structure_builder(feat1, pos1, feat2, pos2)
            
            with torch.cuda.amp.autocast(enabled=False):
                res1 = self._downstream_head(1, [tok.float() for tok in structure_token1], shape1)
                res2 = self._downstream_head(1, [tok.float() for tok in structure_token2], shape2)
            
            pts3d1 = res1['pts3d']
            pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
            pts3d2 = res2['pts3d']
            pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
            pts_all = torch.stack((pts3d1, pts3d2), dim=1)
            pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces
            pts_all = pts_all.unsqueeze(-2)
            
            if visualization_dump is not None:
                visualization_dump["means"] = rearrange(
                        pts_all, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
                    )
            return
        
        else:
            structure_token1, structure_token2 = self.structure_builder(feat1, pos1, feat2, pos2)
            stylized_token1, stylized_token2 = self.token_stylizer(style, feat1, pos1, feat2, pos2)
            
            with torch.cuda.amp.autocast(enabled=False):
                
                res1 = self._downstream_head(1, [tok.float() for tok in structure_token1], shape1)
                res2 = self._downstream_head(1, [tok.float() for tok in structure_token2], shape2)
                
                GS_structure1 = self.gaussian_structure_head([tok.float() for tok in structure_token1], shape1[0].cpu().tolist())
                GS_structure1 = rearrange(GS_structure1, "b d h w -> b (h w) d")
                GS_structure2 = self.gaussian_structure_head([tok.float() for tok in structure_token2], shape2[0].cpu().tolist())
                GS_structure2 = rearrange(GS_structure2, "b d h w -> b (h w) d")
                
                if self.gs_sh_head_type == "linear":
                    GS_appearance1 = rearrange_head(self.gaussian_appearance_head(stylized_token1[-1]), self.patch_size, h, w)
                    GS_appearance2 = rearrange_head(self.gaussian_appearance_head(stylized_token2[-1]), self.patch_size, h, w)
                elif self.gs_sh_head_type == "dpt":
                    GS_appearance1 = self.gaussian_appearance_head([tok.float() for tok in stylized_token1], shape1[0].cpu().tolist())
                    GS_appearance1 = rearrange(GS_appearance1, "b d h w -> b (h w) d")
                    GS_appearance2 = self.gaussian_appearance_head([tok.float() for tok in stylized_token2], shape2[0].cpu().tolist())
                    GS_appearance2 = rearrange(GS_appearance2, "b d h w -> b (h w) d")
                
                GS_res1 = torch.cat([GS_structure1, GS_appearance1], dim=-1)
                GS_res2 = torch.cat([GS_structure2, GS_appearance2], dim=-1)
        
            pts3d1 = res1['pts3d']
            pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
            pts3d2 = res2['pts3d']
            pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
            pts_all = torch.stack((pts3d1, pts3d2), dim=1)
            pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces

            depths = pts_all[..., -1].unsqueeze(-1)            
                
            gaussians = torch.stack([GS_res1, GS_res2], dim=1)
            # (1, 2, 65536, 83) ->
            gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=self.cfg.num_surfaces)
            densities = gaussians[..., 0].sigmoid().unsqueeze(-1)

            # Convert the features and depths into Gaussians.
            if self.pose_free:
                gaussians = self.gaussian_adapter.forward(
                    pts_all.unsqueeze(-2),
                    depths,
                    self.map_pdf_to_opacity(densities, global_step),
                    rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                )
            else:
                xy_ray, _ = sample_image_grid((h, w), device)
                xy_ray = rearrange(xy_ray, "h w xy -> (h w) () xy")
                xy_ray = xy_ray[None, None, ...].expand(b, v, -1, -1, -1)

                gaussians = self.gaussian_adapter.forward(
                    rearrange(context["extrinsics"], "b v i j -> b v () () () i j"),
                    rearrange(context["intrinsics"], "b v i j -> b v () () () i j"),
                    rearrange(xy_ray, "b v r srf xy -> b v r srf () xy"),
                    depths,
                    self.map_pdf_to_opacity(densities, global_step),
                    rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
                    (h, w),
                )

            # Dump visualizations if needed.
            if visualization_dump is not None:
                visualization_dump["depth"] = rearrange(
                    depths, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                )
                visualization_dump["scales"] = rearrange(
                    gaussians.scales, "b v r srf spp xyz -> b (v r srf spp) xyz"
                )
                visualization_dump["rotations"] = rearrange(
                    gaussians.rotations, "b v r srf spp xyzw -> b (v r srf spp) xyzw"
                )
                visualization_dump["means"] = rearrange(
                    gaussians.means, "b v (h w) srf spp xyz -> b v h w (srf spp) xyz", h=h, w=w
                )
                visualization_dump['opacities'] = rearrange(
                    gaussians.opacities, "b v (h w) srf s -> b v h w srf s", h=h, w=w
                )

            return Gaussians(
                rearrange(
                    gaussians.means,
                    "b v r srf spp xyz -> b (v r srf spp) xyz",
                ),
                rearrange(
                    gaussians.covariances,
                    "b v r srf spp i j -> b (v r srf spp) i j",
                ),
                rearrange(
                    gaussians.harmonics,
                    "b v r srf spp c d_sh -> b (v r srf spp) c d_sh",
                ),
                rearrange(
                    gaussians.opacities,
                    "b v r srf spp -> b (v r srf spp)",
                ),
            )

    def get_data_shim(self) -> DataShim:
        def data_shim(batch: BatchedExample) -> BatchedExample:
            batch = apply_normalize_shim(
                batch,
                self.cfg.input_mean,
                self.cfg.input_std,
            )

            return batch

        return data_shim
