import os
import json
import hydra
import wandb
from omegaconf import DictConfig
from pathlib import Path
from colorama import Fore
from jaxtyping import Float, UInt8
from einops import pack, rearrange, repeat, einsum
from io import BytesIO
from PIL import Image
from typing import Literal

from tqdm import tqdm
import moviepy.editor as mpy
import numpy as np
import torch
from torch import Tensor, nn
import torchvision.transforms as tf
from torchvision.utils import save_image

from src.config import load_typed_root_config
from src.loss import get_losses
from src.model.encoder import get_encoder
from src.model.decoder import get_decoder
from src.model.model_wrapper import TrajectoryFn
from src.model.ply_export import export_ply
from src.misc.wandb_tools import update_checkpoint_path
from src.misc.cam_utils import camera_normalization, update_pose
from src.misc.image_io import save_image
from src.misc.utils import vis_depth_map, inverse_normalize
from src.geometry.projection import get_fov
from src.dataset.shims.augmentation_shim import apply_style_image_augmentation, apply_style_image_augmentation_larger
from src.dataset.shims.crop_shim import rescale_and_crop
from src.dataset.data_module import get_data_shim
from src.dataset.types import BatchedExample
from src.visualization.layout import vcat, hcat
from src.visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)


def convert_poses(
    poses: Float[Tensor, "batch 18"],
) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
]:
    b, _ = poses.shape

    # Convert the intrinsics to a 3x3 normalized K matrix.
    intrinsics = torch.eye(3, dtype=torch.float32)
    intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
    fx, fy, cx, cy = poses[:, :4].T
    intrinsics[:, 0, 0] = fx
    intrinsics[:, 1, 1] = fy
    intrinsics[:, 0, 2] = cx
    intrinsics[:, 1, 2] = cy

    # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
    w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w",
                 b=b).clone()
    w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
    return w2c.inverse(), intrinsics


def convert_images(
    images: list[UInt8[Tensor,
                       "..."]], ) -> Float[Tensor, "batch 3 height width"]:
    to_tensor = tf.ToTensor()
    torch_images = []
    for image in images:
        image = Image.open(BytesIO(image.numpy().tobytes()))
        torch_images.append(to_tensor(image))
    return torch.stack(torch_images)


def test_step_align(test_cfg, encoder, decoder, losses, batch, gaussians,
                    stylized_gaussians, device):
    encoder.eval()
    # freeze all parameters
    for param in encoder.parameters():
        param.requires_grad = False

    b, v, _, h, w = batch["target"]["image"].shape
    with torch.set_grad_enabled(True):
        cam_rot_delta = nn.Parameter(
            torch.zeros([b, v, 3], requires_grad=True, device=device))
        cam_trans_delta = nn.Parameter(
            torch.zeros([b, v, 3], requires_grad=True, device=device))

        opt_params = []
        opt_params.append({
            "params": [cam_rot_delta],
            "lr": test_cfg.rot_opt_lr,
        })
        opt_params.append({
            "params": [cam_trans_delta],
            "lr": test_cfg.trans_opt_lr,
        })
        pose_optimizer = torch.optim.Adam(opt_params)

        extrinsics = batch["target"]["extrinsics"].clone()

        for i in tqdm(range(test_cfg.pose_align_steps), desc="Pose Alignment"):
            pose_optimizer.zero_grad()

            output = decoder.forward(
                gaussians,
                extrinsics,
                batch["target"]["intrinsics"],
                batch["target"]["near"],
                batch["target"]["far"],
                (h, w),
                cam_rot_delta=cam_rot_delta,
                cam_trans_delta=cam_trans_delta,
            )

            # Compute and log loss.
            total_loss = 0
            for loss_fn in losses:
                loss = loss_fn.forward(output, batch, gaussians, 0)
                total_loss = total_loss + loss

            total_loss.backward()
            with torch.no_grad():
                pose_optimizer.step()
                new_extrinsic = update_pose(
                    cam_rot_delta=rearrange(cam_rot_delta, "b v i -> (b v) i"),
                    cam_trans_delta=rearrange(cam_trans_delta,
                                              "b v i -> (b v) i"),
                    extrinsics=rearrange(extrinsics, "b v i j -> (b v) i j"))
                cam_rot_delta.data.fill_(0)
                cam_trans_delta.data.fill_(0)

                extrinsics = rearrange(new_extrinsic,
                                       "(b v) i j -> b v i j",
                                       b=b,
                                       v=v)

    # Render Gaussians.
    output = decoder.forward(
        gaussians,
        extrinsics,
        batch["target"]["intrinsics"],
        batch["target"]["near"],
        batch["target"]["far"],
        (h, w),
    )

    stylized_output = decoder.forward(
        stylized_gaussians,
        extrinsics,
        batch["target"]["intrinsics"],
        batch["target"]["near"],
        batch["target"]["far"],
        (h, w),
    )

    return output, stylized_output


def get_bound(
    bound: Literal["near", "far"],
    num_views: int,
) -> Float[Tensor, " view"]:

    if bound == 'near':
        value = torch.tensor(0.1)
    elif bound == 'far':
        value = torch.tensor(100.0)
    else:
        raise ValueError('bound not found!')

    return repeat(value, "-> v", v=num_views)


def render_video_generic(
    gaussians,
    stylized_gaussians,
    decoder,
    batch: BatchedExample,
    trajectory_fn: TrajectoryFn,
    name: str,
    device,
    output_dir: Path,
    num_frames: int = 60,
    smooth: bool = True,
    loop_reverse: bool = True,
) -> None:

    t = torch.linspace(0, 1, num_frames, dtype=torch.float32, device=device)
    if smooth:
        t = (torch.cos(torch.pi * (t + 1)) + 1) / 2

    extrinsics, intrinsics = trajectory_fn(t)

    _, _, _, h, w = batch["context"]["image"].shape

    # TODO: Interpolate near and far planes?
    near = repeat(batch["context"]["near"][:, 0], "b -> b v", v=num_frames)
    far = repeat(batch["context"]["far"][:, 0], "b -> b v", v=num_frames)
    stylized_output = decoder.forward(stylized_gaussians, extrinsics,
                                      intrinsics, near, far, (h, w), "depth")
    output = decoder.forward(gaussians, extrinsics, intrinsics, near, far,
                             (h, w), "depth")

    # images = [
    #     hcat(depth, stylized_rgb, rgb) for stylized_rgb, rgb, depth in zip(
    #         stylized_output.color[0], output.color[0],
    #         vis_depth_map(output.depth[0]))
    # ]
    
    images = [
        stylized_rgb for stylized_rgb in stylized_output.color[0]
    ]

    video = torch.stack(images)
    video = (video.clip(min=0, max=1) * 255).type(torch.uint8).cpu().numpy()
    if loop_reverse:
        video = pack([video, video[::-1][1:-1]], "* c h w")[0]
    visualizations = {
        f"video/{name}": wandb.Video(video[None], fps=30, format="mp4")
    }

    for key, value in visualizations.items():
        tensor = value._prepare_video(value.data)
        # Use a default fps (e.g., 30) if '_fps' is not available
        fps = getattr(value, "_fps", 30)
        clip = mpy.ImageSequenceClip(list(tensor), fps=fps)
        clip.write_videofile(str(output_dir / "depth_stylized_rgb.mp4"),
                             logger=None)


def render_video_interpolation(gaussians, stylized_gaussians, decoder,
                               batch: BatchedExample, device,
                               output_dir: Path) -> None:
    _, v, _, _ = batch["context"]["extrinsics"].shape

    def trajectory_fn(t):
        extrinsics = interpolate_extrinsics(
            batch["context"]["extrinsics"][0, 0],
            batch["context"]["extrinsics"][0, -1],
            t,
        )
        intrinsics = interpolate_intrinsics(
            batch["context"]["intrinsics"][0, 0],
            batch["context"]["intrinsics"][0, -1],
            t,
        )
        return extrinsics[None], intrinsics[None]

    return render_video_generic(gaussians, stylized_gaussians, decoder, batch,
                                trajectory_fn, "rgb", device, output_dir)


def cyan(text: str) -> str:
    return f"{Fore.CYAN}{text}{Fore.RESET}"


@hydra.main(
    version_base=None,
    config_path="./config",
    config_name="main",
)
def infer(cfg_dict: DictConfig):
    cfg = load_typed_root_config(cfg_dict)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    output_dir = Path(
        hydra.core.hydra_config.HydraConfig.get()["runtime"]["output_dir"])
    print(cyan(f"Saving outputs to {output_dir}."))

    # ------------- load in model and checkpoint -----------------
    # Prepare the checkpoint for loading.
    # -------- 2view ----------
    # re10k, h3 for content loss
    # cfg.checkpointing.load = 'outputs/exp_re10k_multi-view_tok-sty-stylization/2025-04-30_12-43-53/checkpoints/epoch_0-step_18750.ckpt'
    
    # re10k, h3 + h4 for content loss
    # cfg.checkpointing.load = 'outputs/exp_re10k_multi-view_tok-sty-stylization_content-h3-h4/2025-05-01_23-06-44/checkpoints/epoch_0-step_35000.ckpt'
    
    # re10k without identity loss
    # cfg.checkpointing.load = 'ckpts/epoch_0-step_35000_no-identity.ckpt'
    
    # -------- 4view -----------
    # re10k
    # cfg.checkpointing.load = 'outputs/exp_re10k_4multi-view_tok-sty-stylization/2025-05-04_02-35-28/checkpoints/epoch_0-step_35000.ckpt'
    
    # re10k + dl3dv
    cfg.checkpointing.load = 'outputs/exp_re10k_dl3dv_4multi-view_tok-sty-stylization_b2x3/2025-05-06_19-31-34/checkpoints/epoch_0-step_35000.ckpt'

    checkpoint_path = update_checkpoint_path(cfg.checkpointing.load, cfg.wandb)

    encoder, _ = get_encoder(cfg.model.encoder)
    encoder.eval()
    encoder.to(device)

    ckpt_weights = torch.load(checkpoint_path,
                              map_location='cpu')['state_dict']
    ckpt_weights = {
        k[8:]: v
        for k, v in ckpt_weights.items() if k.startswith('encoder.')
    }
    missing_keys, unexpected_keys = encoder.load_state_dict(ckpt_weights)

    # ------------ read in content images from the scene ---------
    # use the root in cfg
    data_cfg = cfg.dataset[0].re10k_style
    root = data_cfg.roots[0]

    # ---------------------------------------------------------------------------------------------
    # TODO: specify scene_idx, style_image_path, left and rightmost ctx view indices, num of views
    # scene_idx = 'f58e63b78cfc54cc'
    # scene_idx = 'b3954464d00ffcc5'
    # scene_idx = '1259726fc1f8e966'
    scene_idx = '9d496b6788c0f149'
    # scene_idx = '5f38fc5d3910c6d6' # top view house
    # scene_idx = 'a17fdfcb519750eb'
    # scene_idx = 'd4db9d2a539da854' # bedroom
    # scene_idx = 'a17fdfcb519750eb' # refrig
    # scene_idx = '0e2c96cd97e73a38' # lamp
    # scene_idx = 'c4ed784ee6c885e5'
    # scene_idx = '45108618c40e26a7' # bedroom lamp
    # scene_idx = 'a79894dc49ea677a'
    
    # specified_style_image_path = Path('colmap_test_data/styles/starry_night.jpg')
    # specified_style_image_path = Path('colmap_test_data/styles/2020.png')
    specified_style_image_path = Path(
        '/ssdwork/wangpeng/datasets/wikiart/images_combine/train/' \
        'still-life_86e878e3fab4c3024f333afc1f06ef31c.jpg'
    )
    # specified_style_image_path = Path(
    #     '/ssdwork/wangpeng/datasets/wikiart/images_combine/train/' \
    #     'still-life_e901bb48c0d8a9973e918e676838341ec.jpg'
    # )
    # specified_style_image_path = Path(
    #     '/ssdwork/wangpeng/datasets/wikiart/images_combine/train/' \
    #     'genre-painting_fdbcbef6afc35b6f3ceabab995d6a5b3c.jpg'
    # )
    # specified_style_image_path = Path('colmap_test_data/styles/abstract.png')
    # specified_style_image_path = Path(
    #     '/ssdwork/datasets/stylerf_style/' \
    #     '17.jpg'
    # )

    ctx_indices = torch.tensor([0, 245])
    num_ctx_views = 8
    # ----------------------------------------------------------------------------------------------

    to_tensor = tf.ToTensor()

    merged_index = {}

    data_stages = ("test", "train")
    for data_stage in data_stages:

        with (root / data_stage / "index.json").open("r") as f:
            index = json.load(f)
        index = {k: Path(root / data_stage / v) for k, v in index.items()}

        # The constituent datasets should have unique keys.
        assert not (set(merged_index.keys()) & set(index.keys()))

        # Merge the root's index into the main index.
        merged_index = {**merged_index, **index}

    # identify the chunk for the specified scene
    chunk_path = merged_index[scene_idx]

    # identify item that corresponds to the specified scene
    chunk = torch.load(chunk_path)
    item = [x for x in chunk if x["key"] == scene_idx]
    assert len(item) == 1

    example = item[0]
    # read in extrinsiscs and intrinsics
    extrinsics, intrinsics = convert_poses(example['cameras'])

    # get scene key
    scene = example['key']
    print(cyan(f'There are {len(example["images"])} images in {scene}.'))

    if num_ctx_views > 2:
        left, right = ctx_indices.unbind(dim=-1)
        # evenly distribute the additional context views between the left and right views
        ctx_indices = torch.stack(
            [torch.linspace(left, right, num_ctx_views).long()],
            dim=-1,
        ).squeeze(-1)
    else:
        left, right = ctx_indices.unbind(dim=-1)

    target_indices = torch.arange(left, right)
    # exlucding ctx views in target_indices
    mask = ~torch.isin(target_indices, ctx_indices)
    target_indices = target_indices[mask]

    # (optional) skip if FOV is too wide
    assert not (get_fov(intrinsics).rad2deg()
                > data_cfg.max_fov).any(), "FOV too wide!"

    context_images = [example["images"][index.item()] for index in ctx_indices]
    context_images = convert_images(context_images)
    target_images = [
        example["images"][index.item()] for index in target_indices
    ]
    target_images = convert_images(target_images)

    # Skip the example if the images don't have the right shape.
    context_image_invalid = context_images.shape[1:] != (
        3, *data_cfg.original_image_shape)
    target_image_invalid = target_images.shape[1:] != (
        3, *data_cfg.original_image_shape)
    assert not (data_cfg.skip_bad_shape and
                (context_image_invalid or
                 target_image_invalid)), "Images don't have the right shape!"

    # Resize the world to make the baseline 1.
    context_extrinsics = extrinsics[ctx_indices]
    if data_cfg.make_baseline_1:
        a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
        scale = (a - b).norm()
        assert not (scale < data_cfg.baseline_min
                    or scale > data_cfg.baseline_max), "Baseline out of range!"

        extrinsics[:, :3, 3] /= scale
    else:
        scale = 1

    # relative pose normalization
    if data_cfg.relative_pose:
        extrinsics = camera_normalization(extrinsics[ctx_indices][0:1],
                                          extrinsics)

    # read in specified style image
    if specified_style_image_path.exists():
        style_image_name = specified_style_image_path.name
        style_image = Image.open(specified_style_image_path).convert("RGB")
        style_image = to_tensor(style_image)
    else:
        raise FileNotFoundError(
            f"No image found for {specified_style_image_path}")

    style_image = apply_style_image_augmentation(style_image, 'val')
    # style_image = apply_style_image_augmentation_larger(style_image, 'val')

    # crop images to 256 x 256
    # data_cfg.input_image_shape = [352, 640]
    context_images, intrinsics[ctx_indices] = rescale_and_crop(
        context_images, intrinsics[ctx_indices], data_cfg.input_image_shape)
    target_images, intrinsics[target_indices] = rescale_and_crop(
        target_images, intrinsics[target_indices], data_cfg.input_image_shape)

    example = {
        "context": {
            "extrinsics":
            extrinsics[ctx_indices].unsqueeze(0).to(device),
            "intrinsics":
            intrinsics[ctx_indices].unsqueeze(0).to(device),
            "image":
            context_images.unsqueeze(0).to(device),
            "near": (get_bound("near", len(ctx_indices)) /
                     scale).unsqueeze(0).to(device),
            "far": (get_bound("far", len(ctx_indices)) /
                    scale).unsqueeze(0).to(device),
            "index":
            ctx_indices.unsqueeze(0).to(device),
        },
        "target": {
            "extrinsics":
            extrinsics[target_indices].unsqueeze(0).to(device),
            "intrinsics":
            intrinsics[target_indices].unsqueeze(0).to(device),
            "image":
            target_images.unsqueeze(0).to(device),
            "near": (get_bound("near", len(target_indices)) /
                     scale).unsqueeze(0).to(device),
            "far": (get_bound("far", len(target_indices)) /
                    scale).unsqueeze(0).to(device),
            "index":
            target_indices.unsqueeze(0).to(device),
        },
        "scene": scene,
        "style": {
            "image": style_image.unsqueeze(0).to(device),
            "image_name": style_image_name
        },
    }

    # ------------- inference model -----------------
    batch = example

    data_shim = get_data_shim(encoder)
    batch: BatchedExample = data_shim(batch)

    # get non-stylized Gaussians
    b, _, _, h, w = batch["target"]["image"].shape
    assert b == 1

    style = {'image': batch["context"]["image"][:, 0]}

    # XXX: bug during training, input style image range is (0, 1) rather than (-1, 1)
    # batch['style']['image']= batch['style']['image'] * 2 -1
    visualization_dump = {}
    with torch.no_grad():
        gaussians = encoder(batch["context"],
                            style,
                            visualization_dump=visualization_dump)
        stylized_gaussians = encoder(batch["context"], batch["style"])

    # align target view poses with non-stylized Gaussians
    decoder = get_decoder(cfg.model.decoder).to(device)
    losses = get_losses(cfg.loss)
    for loss in losses:
        loss.to(device)

    # render non-stylized and stylized target view images
    output, stylized_output = test_step_align(cfg.test, encoder, decoder,
                                              losses, batch, gaussians,
                                              stylized_gaussians, device)

    # NOTE: save context images and target images, refer to test_step, and vicasplat
    save_image(
        batch["style"]['image'][0],
        output_dir / f"{scene} | {style_image_name[:-4]}" /
        f"{batch['style']['image_name']}")

    for index, color in zip(batch["context"]["index"][0],
                            inverse_normalize(batch["context"]["image"][0])):
        # index = index + 1
        save_image(
            color, output_dir / f"{scene} | {style_image_name[:-4]}" /
            f"context/{index:0>6}.png")
        
    for index, color in zip(batch["target"]["index"][0],
                            batch["target"]["image"][0]):
        # index = index + 1
        save_image(
            color, output_dir / f"{scene} | {style_image_name[:-4]}" /
            f"target/{index:0>6}.png")

    for index, color in zip(batch["target"]["index"][0], output.color[0]):
        # index = index + 1
        save_image(
            color, output_dir / f"{scene} | {style_image_name[:-4]}" /
            f"color/{index:0>6}.png")

    for index, color in zip(batch["target"]["index"][0],
                            stylized_output.color[0]):
        # index = index + 1
        save_image(
            color, output_dir / f"{scene} | {style_image_name[:-4]}" /
            f"stylized_color/{index:0>6}.png")

    # render interpolated video with non-stylized and stylized Gaussians
    render_video_interpolation(gaussians, stylized_gaussians, decoder, batch,
                               device, output_dir)

    # export non-stylized and stylized Gaussians as .ply
    export_ply(
        means=stylized_gaussians.means[0],
        scales=visualization_dump['scales'][0],
        rotations=visualization_dump["rotations"][0],
        harmonics=stylized_gaussians.harmonics[0],
        opacities=stylized_gaussians.opacities[0],
        path=output_dir / "stylized_gaussians.ply",
    )
    export_ply(
        means=gaussians.means[0],
        scales=visualization_dump['scales'][0],
        rotations=visualization_dump["rotations"][0],
        harmonics=gaussians.harmonics[0],
        opacities=gaussians.opacities[0],
        path=output_dir / "gaussians.ply",
    )

    print(f"scene = {batch['scene']}; "
          f"style = {batch['style']['image_name']}")


if __name__ == "__main__":
    infer()
