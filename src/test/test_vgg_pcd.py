import torch
from torchvision import transforms
import torchvision.transforms as tf
from PIL import Image
import numpy as np
import hydra
from jaxtyping import install_import_hook
from omegaconf import DictConfig
from sklearn.decomposition import PCA
import torch.nn.functional as F
from torch.utils.data import DataLoader

import plotly.graph_objects as go
import os
import sys
# print(os.getcwd())
sys.path.append(os.path.join("/run/determined/workdir/home/noposplat_private"))

from src.test.vgg_model import VGGEncoder
from src.test.style_dataset import PreprocessDataset
from src.dataset.data_module import get_data_shim
from src.dataset.types import BatchedExample
from src.misc.weight_modify import checkpoint_filter_fn
from src.misc.utils import inverse_normalize
from einops import rearrange
from src.test.point_mlp_model import PointCloudMLP
from src.model.types import Gaussians

# Configure beartype and jaxtyping.
with install_import_hook(
    ("src",),
    ("beartype", "beartype"),
):
    from src.config import load_typed_root_config
    from src.dataset.data_module import DataModule
    from src.global_cfg import set_cfg
    from src.loss import get_losses
    from src.misc.LocalLogger import LocalLogger
    from src.misc.step_tracker import StepTracker
    from src.misc.wandb_tools import update_checkpoint_path
    from src.model.decoder import get_decoder
    from src.model.encoder import get_encoder
    from src.model.model_wrapper import ModelWrapper

def visualize_colored_pcd_plotly(points1: np.array, points2: np.array,
                                 rgbs1: np.array, rgbs2: np.array, subsample: int=1,postfix: str=None):
    """this function takes in pair of pointclouds, rgbs
    color the pointclouds accordingly then save to html for visualization
    
    Input: 
        points1: [N, 3]
        points2: [N, 3]
        rgbs1: [N, 3] range -> [0, 1]
        rgbs2: [N, 3] range -> [0, 1]
    
    Output:
        None, but save colored pointcloud to html
    """
    
    points1 = points1[::subsample]
    points2 = points2[::subsample]
    rgbs1 = rgbs1[::subsample]
    rgbs2 = rgbs2[::subsample]
    
    assert points1.shape == rgbs1.shape and points2.shape == rgbs2.shape
    rgbs1 = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in rgbs1]
    rgbs2 = [f'rgb({int(r*255)},{int(g*255)},{int(b*255)})' for r, g, b in rgbs2]
    
    # Create Plotly 3D scatter plots with per-point colors
    scatter1 = go.Scatter3d(
        x=points1[:, 0], y=points1[:, 1], z=points1[:, 2],
        mode='markers',
        marker=dict(size=2, color=rgbs1),  # Assign per-point colors
        name='View 1'
    )

    scatter2 = go.Scatter3d(
        x=points2[:, 0], y=points2[:, 1], z=points2[:, 2],
        mode='markers',
        marker=dict(size=2, color=rgbs2),  # Assign per-point colors
        name='View 2'
    )
    
    # Combine into a figure
    fig = go.Figure(data=[scatter1, scatter2])
    fig.update_layout(scene=dict(xaxis_title='X', yaxis_title='Y', zaxis_title='Z'))

    # Save as an interactive HTML file
    fig.write_html(f"colored_pcd_3d_{postfix}.html")
    print(f"Colored pointcloud saved as 'colored_pcd_3d_{postfix}.html'.")    

def vis_vgg_feature(view1: torch.Tensor, view2: torch.Tensor, output_size: tuple):
    """input vgg feature maps from 2 views, apply PCA to compress to 3 channels
    then upsample to the original image size
    
    Input:
        view1: [D, H, W]
        view2: [D, H, W]
        output_size: (H1, W1) tuple indicating image size

    Output:
        view1_up: [3, H1, W1]
        view2_up: [3, H1, W1]
    """
    n_channels, h, w = view1.shape
    view1 = view1.permute(1, 2, 0).reshape(-1, n_channels)
    view2 = view2.permute(1, 2, 0).reshape(-1, n_channels)
    
    # apply PCA on both views altogether
    pca = PCA(n_components=3)
    views_pca = pca.fit_transform(torch.cat((view1, view2), 0).numpy())
    view1_pca = views_pca[:h*w]
    view2_pca = views_pca[h*w:]
    
    view1_pca = (view1_pca - view1_pca.min()) / (view1_pca.max() - view1_pca.min())
    view2_pca = (view2_pca - view2_pca.min()) / (view2_pca.max() - view2_pca.min())
    
    view1_pca = view1_pca.reshape(h, w, 3)  # Shape: (64, 64, 3)
    view1_pca = torch.tensor(view1_pca, dtype=torch.float32).permute(2, 0, 1)  # Shape: (3, 64, 64)
    
    view2_pca = view2_pca.reshape(h, w, 3)  # Shape: (64, 64, 3)
    view2_pca = torch.tensor(view2_pca, dtype=torch.float32).permute(2, 0, 1)  # Shape: (3, 64, 64)
    
    view1_pca = view1_pca.unsqueeze(0)  # Shape: (1, 3, 64, 64)
    view1_pca = F.interpolate(view1_pca, size=output_size, mode='bilinear', align_corners=False)  # Shape: (1, 3, 256, 256)
    view1_pca = view1_pca.squeeze(0)  # Shape: (3, 256, 256)
    
    view2_pca = view2_pca.unsqueeze(0)  # Shape: (1, 3, 64, 64)
    view2_pca = F.interpolate(view2_pca, size=output_size, mode='bilinear', align_corners=False)  # Shape: (1, 3, 256, 256)
    view2_pca = view2_pca.squeeze(0)  # Shape: (3, 256, 256)
    
    return view1_pca, view2_pca

def calc_mean_std(x, eps=1e-8):
    """
    calculating channel-wise instance mean and standard variance
    x: shape of (N,C,*)
    """
    mean = torch.mean(x.flatten(2), dim=-1, keepdim=True) # size of (N, C, 1)
    std = torch.std(x.flatten(2), dim=-1, keepdim=True) + eps # size of (N, C, 1)
    
    return mean, std

def AdaIN_pcd(content_features, style_features):
    """apply AdaIN on the whole pointcloud altogether (refer to StyleGaussian)
    or view-wise
    
    Input:
        content_features [b, c, n] / [b, c, h1, w1]
        style_features [b, c, h2, w2]
    
    Output:
        normalized_features [b, c, n] / [b, c, h1, w1]
    """
    if content_features.dim() == 4:
        output_dim = 4
        b, c, h1, w1 = content_features.shape
        content_features = content_features.reshape(b, c, -1)
    else:
        output_dim = 3
    
    content_mean, content_std = calc_mean_std(content_features)
    style_mean, style_std = calc_mean_std(style_features)
        
    normalized_features = (content_features - content_mean) / content_std
    normalized_features = normalized_features * style_std + style_mean
    
    if output_dim == 4:
        normalized_features = normalized_features.reshape(b, c, h1, w1)
    
    return normalized_features

@hydra.main(
    version_base=None,
    config_path="../../config",
    config_name="main",
)
def test(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    set_cfg(cfg_dict)
    
    # Prepare the checkpoint for loading.
    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)
    
    encoder, encoder_visualizer = get_encoder(cfg.model.encoder)
    data_shim = get_data_shim(encoder)
    encoder.to('cuda')
    
    # Load the encoder weights.
    if cfg.model.encoder.pretrained_weights and cfg.mode == "train":
        weight_path = cfg.model.encoder.pretrained_weights
        ckpt_weights = torch.load(weight_path, map_location='cpu')
        if 'model' in ckpt_weights:
            ckpt_weights = ckpt_weights['model']
            ckpt_weights = checkpoint_filter_fn(ckpt_weights, encoder)
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        elif 'state_dict' in ckpt_weights:
            ckpt_weights = ckpt_weights['state_dict']
            ckpt_weights = {k[8:]: v for k, v in ckpt_weights.items() if k.startswith('encoder.')}
            missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights, strict=False)
        else:
            raise ValueError(f"Invalid checkpoint format: {weight_path}")
    
    data_module = DataModule(
        cfg.dataset,
        cfg.data_loader
    )
    data_loader = data_module.train_dataloader()
    batch = next(iter(data_loader))
    batch: BatchedExample = data_shim(batch)
    _, _, _, h, w = batch["target"]["image"].shape
    # NOTE: style image should also be normalize, go through the same transformation
    # that content image has been through
    context = batch["context"]
    for key in context:
        context[key] = context[key].to('cuda')
    device = context["image"].device
    b, v, _, h, w = context["image"].shape
    
    dec1, dec2, shape1, shape2, view1, view2 = encoder.backbone(context, return_views=True)
    with torch.cuda.amp.autocast(enabled=False):
        res1 = encoder._downstream_head(1, [tok.float() for tok in dec1], shape1)
        res2 = encoder._downstream_head(2, [tok.float() for tok in dec2], shape2)
    
    pts3d1 = res1['pts3d']
    pts3d1 = rearrange(pts3d1, "b h w d -> b (h w) d")
    pts3d2 = res2['pts3d']
    pts3d2 = rearrange(pts3d2, "b h w d -> b (h w) d")
    pts_all = torch.stack((pts3d1, pts3d2), dim=1)
    pts_all = pts_all.unsqueeze(-2)  # for cfg.num_surfaces
    
    depths = pts_all[..., -1].unsqueeze(-1)
    
    view_idx = 0
    
    # Convert PyTorch tensors to NumPy arrays
    points1_np = pts3d1[view_idx].detach().cpu().numpy()
    points2_np = pts3d2[view_idx].detach().cpu().numpy()
    
    # extract VGG feature maps on context images
    # Normalize for VGG (expects RGB input with ImageNet mean/std), before norm should be [0, 1]
    preprocess = transforms.Compose([
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    # [-1, 1] -> [0, 1] -> ImageNet mean/std
    view1_vgg = preprocess(inverse_normalize(view1['img']))
    view2_vgg = preprocess(inverse_normalize(view2['img']))
    
    vgg_encoder = VGGEncoder()
    vgg_encoder.to('cuda')
    view1_h1, _, view1_h3, _ = vgg_encoder(view1_vgg)
    view2_h1, _, view2_h3, _ = vgg_encoder(view2_vgg)
    
    _, h1_channels, _, _ = view1_h1.shape
    _, h3_channels, _, _ = view1_h3.shape
    
    # view1_h1_pca, view2_h1_pca = vis_vgg_feature(view1_h1[view_idx], view2_h1[view_idx], output_size=(256, 256))
    # view1_h3_pca, view2_h3_pca = vis_vgg_feature(view1_h3[view_idx], view2_h3[view_idx], output_size=(256, 256))
    
    # view1_h1_pca = np.reshape(view1_h1_pca.numpy().transpose(1, 2, 0), (-1, 3))
    # view2_h1_pca = np.reshape(view2_h1_pca.numpy().transpose(1, 2, 0), (-1, 3))
    # view1_h3_pca = np.reshape(view1_h3_pca.numpy().transpose(1, 2, 0), (-1, 3))
    # view2_h3_pca = np.reshape(view2_h3_pca.numpy().transpose(1, 2, 0), (-1, 3))
    
    # convert colors from [-1, 1] to [0, 1]
    # colors1_np = inverse_normalize(view1['img'][view_idx]).numpy()
    # colors2_np = inverse_normalize(view2['img'][view_idx]).numpy()
    # colors1_np = np.reshape(colors1_np.transpose(1, 2, 0), (-1, 3))
    # colors2_np = np.reshape(colors2_np.transpose(1, 2, 0), (-1, 3))

    # save as colored pointcloud
    # visualize_colored_pcd_plotly(points1_np, points2_np, colors1_np, colors2_np, postfix="RGB")
    # visualize_colored_pcd_plotly(points1_np, points2_np, view1_h1_pca, view2_h1_pca, postfix="vgg_h1")
    # visualize_colored_pcd_plotly(points1_np, points2_np, view1_h3_pca, view2_h3_pca, postfix="vgg_h3")
    
    # load in style image
    # resize and random crop 256 x 256 (AdaIN training)
    style_images_folder = '/run/determined/workdir/data/wikiart/All_Images'
    style_dataset = PreprocessDataset(style_images_folder)
    
    style_dataloader = DataLoader(style_dataset, batch_size=cfg.data_loader.train.batch_size, shuffle=False)
    style_imgs = next(iter(style_dataloader))
    style_imgs = style_imgs.to('cuda')
    
    style_img_vis = style_imgs[0]
    
    style_imgs_h1, _, style_imgs_h3, _ = vgg_encoder(style_imgs)

    view1_h3_up = F.interpolate(view1_h3, size=(h, w), mode='bilinear', align_corners=False)  
    view2_h3_up = F.interpolate(view2_h3, size=(h, w), mode='bilinear', align_corners=False) 
    
    # apply AdaIN view-wise
    # blended_view1_h3 = AdaIN_pcd(view1_h3_up, style_imgs_h3)
    # blended_view2_h3 = AdaIN_pcd(view2_h3_up, style_imgs_h3)
     
    # apply AdaIN for the whole pointcloud 
    view1_h3_up = view1_h3_up.reshape(b, h3_channels, -1)
    view2_h3_up = view2_h3_up.reshape(b, h3_channels, -1)
    views_h3_up = torch.cat((view1_h3_up, view2_h3_up), dim=2)
    blended_views_h3 = AdaIN_pcd(views_h3_up, style_imgs_h3)
    
    PointMLP = PointCloudMLP(in_channels=h3_channels, out_channels=83)
    PointMLP.to('cuda')
    
    GS_res = PointMLP(blended_views_h3)
    
    b, d, n = GS_res.shape
    gaussians = GS_res.reshape(b, d, n//2, 2).transpose(1, 3)
    gaussians = rearrange(gaussians, "... (srf c) -> ... srf c", srf=encoder.cfg.num_surfaces) # (1, 2, 65536, 1, 83)
    densities = gaussians[..., 0].sigmoid().unsqueeze(-1) # (1, 2, 65536, 1, 1)

    global_step = 1
    
    gaussians = encoder.gaussian_adapter.forward(
                pts_all.unsqueeze(-2),
                depths,
                encoder.map_pdf_to_opacity(densities, global_step), 
                rearrange(gaussians[..., 1:], "b v r srf c -> b v r srf () c"),
            )
    
    gaussians = Gaussians(
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
    
    decoder = get_decoder(cfg.model.decoder)
    decoder.to('cuda')
    
    for key in batch["target"]:
        batch["target"][key] = batch["target"][key].to('cuda')
    
    # NOTE: couldn't get through before, but solved after putting everything in GPU
    output = decoder.forward(
            gaussians,
            batch["target"]["extrinsics"],
            batch["target"]["intrinsics"],
            batch["target"]["near"],
            batch["target"]["far"],
            (h, w),
            depth_mode=cfg.train.depth_mode,
        )
    target_gt = batch["target"]["image"]
    
    # compute content and style loss on target views
    # NOTE: output and target image is within [0, 1],
    # should be normalized before fed into VGG
    
    
if __name__ == "__main__":
    test()
    
    