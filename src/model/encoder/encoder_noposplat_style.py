from copy import deepcopy
from dataclasses import dataclass
from typing import Literal, Optional

import torch
import torch.nn.functional as F
from torchvision import transforms
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
from .pointcloud_encoder.pointcloud_encoder import PointCloudEncoderCfg, PointCloudEncoder
from .pointcloud_encoder.pointnet2_encoder import Pointnet2Encoder
from .pointcloud_encoder.layer import PointUpsample
from .pointcloud_encoder.utils import normalize_pointclouds

from ...misc.nn_module_tools import convert_to_buffer
from ...misc.utils import inverse_normalize
from ...test.vgg_model import VGGEncoder, AdaIN_pcd
from ...test.point_mlp_model import PointWiseMLP

from .stylizer.stylizer import AdaAttN3DStylizer, Linear3DStylizer, AdaIN3DStylizer, CrossAttentionTransformer

inf = float('inf')


@dataclass
class OpacityMappingCfg:
    initial: float
    final: float
    warm_up: int


@dataclass
class EncoderNoPoSplatStyleCfg:
    # nondefault
    name: Literal["noposplat", "noposplat_multi", "noposplat_style"]
    d_feature: int
    num_monocular_samples: int
    backbone: BackboneCfg
    visualizer: EncoderVisualizerEpipolarCfg
    gaussian_adapter: GaussianAdapterCfg
    apply_bounds_shim: bool
    opacity_mapping: OpacityMappingCfg
    gaussians_per_pixel: int
    num_surfaces: int
    gs_params_head_type: str
    pointcloud_encoder: PointCloudEncoderCfg
    # default
    input_mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    input_std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    pretrained_weights: str = ""
    pose_free: bool = True
    stylized: bool = False
    appearance_feature_dim: int = 256
    point_aggregate: bool = False
    direct_opt_sh_res: bool = False
    direct_opt_feat_res: bool = False
    stylizer: str = 'AdaAttN'

def rearrange_head(feat, patch_size, H, W):
    B = feat.shape[0]
    feat = feat.transpose(-1, -2).view(B, -1, H // patch_size, W // patch_size)
    feat = F.pixel_shuffle(feat, patch_size)  # B,D,H,W
    feat = rearrange(feat, "b d h w -> b (h w) d")
    return feat


class EncoderNoPoSplatStyle(Encoder[EncoderNoPoSplatStyleCfg]):
    backbone: nn.Module
    gaussian_adapter: GaussianAdapter

    def __init__(self, cfg: EncoderNoPoSplatStyleCfg) -> None:
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
        self.point_aggregate = cfg.point_aggregate
        self.direct_opt_sh_res = cfg.direct_opt_sh_res
        self.direct_opt_feat_res = cfg.direct_opt_feat_res
        
        assert (self.direct_opt_feat_res and self.direct_opt_sh_res) == False, "shouldn't optimizing feat_res and sh_res together"
        
        self.set_center_head(output_mode='pts3d', head_type='dpt', landscape_only=True,
                           depth_mode=('exp', -inf, inf), conf_mode=None,)
        
        self.appearance_feature_dim = cfg.appearance_feature_dim
        
        self.set_feature_head()
        self.set_attributes_head()

        # NOTE: the in_channels is hardcoded as 256 for now, for future should be configurable
        # self.PointMLP = PointCloudMLP(in_channels=3, out_channels=self.raw_gs_dim)
        self.gs_mlp = PointWiseMLP(in_channels=self.appearance_feature_dim, out_channels=3 * self.gaussian_adapter.d_sh, hidden_dims=[])
        
        if self.stylized:
            if cfg.stylizer == 'AdaAttN':
                self.stylizer = AdaAttN3DStylizer(feats_in_dim=self.appearance_feature_dim)
            elif cfg.stylizer == 'LST':
                self.stylizer = Linear3DStylizer()
            elif cfg.stylizer == "AdaIN":
                self.stylizer = AdaIN3DStylizer()
            elif cfg.stylizer == "CrossAttention":
                self.stylizer = CrossAttentionTransformer(feature_dim=self.appearance_feature_dim)
            else:
                raise NotImplementedError(f'{cfg.stylizer} is not found.')

        if self.point_aggregate:
            # self.pointcloud_encoder = Pointnet2Encoder(in_dim=3)
            self.pointcloud_encoder = PointCloudEncoder(cfg.pointcloud_encoder, in_dim=self.appearance_feature_dim)
            self.pointcloud_upsampler = PointUpsample()
        
        if self.direct_opt_sh_res:
            self.opt_sh_res = nn.Parameter(torch.randn((1, 3 * self.gaussian_adapter.d_sh, 2 * 256 * 256)))
        
        if self.direct_opt_feat_res:
            self.opt_feat_res = nn.Parameter(torch.randn((1, self.appearance_feature_dim, 2 * 256 * 256)))
        
    def set_center_head(self, output_mode, head_type, landscape_only, depth_mode, conf_mode):
        self.backbone.depth_mode = depth_mode
        self.backbone.conf_mode = conf_mode
        # allocate heads
        self.downstream_head1 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))
        self.downstream_head2 = head_factory(head_type, output_mode, self.backbone, has_conf=bool(conf_mode))

        # magic wrapper
        self.head1 = transpose_to_landscape(self.downstream_head1, activate=landscape_only)
        self.head2 = transpose_to_landscape(self.downstream_head2, activate=landscape_only)

    def set_feature_head(self):
        # NOTE: hardcode the output dimension of feature as 256 for now, as previous layer also has 256 dim
        self.feature_head = head_factory('dpt_gs', 'gs_params', self.backbone, has_conf=False, out_nchan=self.appearance_feature_dim)
        self.feature_head2 = head_factory('dpt_gs', 'gs_params', self.backbone, has_conf=False, out_nchan=self.appearance_feature_dim)
    
    def set_attributes_head(self):
        self.attributes_head = head_factory('dpt_gs', 'gs_params', self.backbone, has_conf=False, out_nchan=8)
        self.attributes_head2 = head_factory('dpt_gs', 'gs_params', self.backbone, has_conf=False, out_nchan=8)
        
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
    ) -> Gaussians:
        device = context["image"].device
        b, v, _, h, w = context["image"].shape

        # Encode the context images.
        dec1, dec2, shape1, shape2, view1, view2 = self.backbone(context, return_views=True)
        with torch.cuda.amp.autocast(enabled=False):
            res1 = self._downstream_head(1, [tok.float() for tok in dec1], shape1)
            res2 = self._downstream_head(2, [tok.float() for tok in dec2], shape2)

            # for feature heads
            feat_res1 = self.feature_head([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3], shape1[0].cpu().tolist())
            feat_res1 = rearrange(feat_res1, "b d h w -> b (h w) d")
            feat_res2 = self.feature_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
            feat_res2 = rearrange(feat_res2, "b d h w -> b (h w) d")
            
            attributes_res1 = self.attributes_head([tok.float() for tok in dec1], res1['pts3d'].permute(0, 3, 1, 2), view1['img'][:, :3], shape1[0].cpu().tolist())
            attributes_res1 = rearrange(attributes_res1, "b d h w -> b (h w) d")
            attributes_res2 = self.attributes_head2([tok.float() for tok in dec2], res2['pts3d'].permute(0, 3, 1, 2), view2['img'][:, :3], shape2[0].cpu().tolist())
            attributes_res2 = rearrange(attributes_res2, "b d h w -> b (h w) d")
            
        pts3d1 = res1['pts3d']
        pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
        pts3d2 = res2['pts3d']
        pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
        pts_all = torch.stack((pts3d1, pts3d2), dim=1)
        pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces
        depths = pts_all[..., -1].unsqueeze(-1)
        
        # view1_vgg = self.vgg_preprocess(inverse_normalize(view1['img']))
        # view2_vgg = self.vgg_preprocess(inverse_normalize(view2['img']))
        
        # view1_h1, _, view1_h3, _ = self.vgg_encoder(view1_vgg)
        # view2_h1, _, view2_h3, _ = self.vgg_encoder(view2_vgg)
        
        # _, h3_channels, _, _ = view1_vgg.shape
        
        # view1_h3 = F.interpolate(view1_h3, size=(h, w), mode='bilinear', align_corners=False)  
        # view2_h3 = F.interpolate(view2_h3, size=(h, w), mode='bilinear', align_corners=False)
                
        # LSM prototype test
        feat_res1 = rearrange(feat_res1, "b n d -> b d n")
        feat_res2 = rearrange(feat_res2, "b n d -> b d n")
        feat_res = torch.cat((feat_res1, feat_res2), dim=2)
        
        if self.direct_opt_feat_res:
            feat_res = self.opt_feat_res
            print(f'mean: {self.opt_feat_res.mean()}, std: {self.opt_feat_res.std()}')
        
        # use rgb instead
        # view1_rgb = rearrange(inverse_normalize(view1['img']), "b c h w -> b c (h w)")
        # view2_rgb = rearrange(inverse_normalize(view2['img']), "b c h w -> b c (h w)")
        # feat_res = torch.cat((view1_rgb, view2_rgb), dim=2)
        
        if self.point_aggregate:
            # point aggregation on feat
            # NOTE: is there gradient flow back xyz? is it necessary in our case?
            xyz = torch.cat((pts3d1, pts3d2), dim=1)
            xyz = normalize_pointclouds(xyz)
            xyz = xyz.detach() # this make the training more stable
            enc_dict = self.pointcloud_encoder(xyz, feat_res)        
            feats_list = enc_dict['feats_list']
            feat_res = feats_list[-1]
            xyz_list = enc_dict['xyz_list']
            xyz = xyz_list[-1] ###
        
        # view1_h3 = rearrange(view1_h3, "b c h w -> b c (h w)")
        # view2_h3 = rearrange(view2_h3, "b c h w -> b c (h w)")
        # views_h3 = torch.cat((view1_h3, view2_h3), dim=2)
        
        # blended_views_h3 = AdaIN_pcd(views_h3, style_imgs_h3)
        
        if self.direct_opt_sh_res and global_step == 0:
            with torch.no_grad():  # Disable gradient tracking when modifying the parameter
                self.opt_sh_res.data = self.gs_mlp(feat_res).data
        
        if self.stylized and not self.direct_opt_feat_res and not self.direct_opt_sh_res:
            style_images = style['image']
            feat_res = self.stylizer(style_images, feat_res, torch.cat((pts3d1, pts3d2), dim=1))
            
            vis_param = next(self.stylizer.zipper.parameters()).clone().detach().norm()
            print(f"vis_param of stylizer: {vis_param.item()}")
            
            
        if self.point_aggregate:
            # feat upsample 
            for i in range(len(xyz_list) - 2, -1, -1):
                parent_xyz = xyz_list[i]
                feat_res = self.pointcloud_upsampler(xyz, parent_xyz, feat_res)
                xyz = parent_xyz
        
        sh_res = self.gs_mlp(feat_res) # (b, 3, n)
        
        if self.direct_opt_sh_res:
            sh_res = self.opt_sh_res
            print(f'mean: {self.opt_sh_res.mean()}, std: {self.opt_sh_res.std()}')
        
        sh_res1 = sh_res[..., :h * w]
        sh_res2 = sh_res[..., h * w:]
        sh_res1 = rearrange(sh_res1, "b d n -> b n d")
        sh_res2 = rearrange(sh_res2, "b d n -> b n d")
        
        GS_res1 = torch.cat((attributes_res1, sh_res1), dim=-1)
        GS_res2 = torch.cat((attributes_res2, sh_res2), dim=-1)
        
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
