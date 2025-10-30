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
import imageio.v2 as imageio
from typing import Literal

import cv2
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
from src.dataset.shims.augmentation_shim import apply_style_image_augmentation
from src.dataset.shims.crop_shim import rescale_and_crop
from src.dataset.data_module import get_data_shim
from src.dataset.types import BatchedExample
from src.visualization.layout import vcat, hcat
from src.visualization.camera_trajectory.interpolation import (
    interpolate_extrinsics,
    interpolate_intrinsics,
)

from src.dataset.colmap_parsing_utils import read_cameras_binary, read_cameras_text, read_images_binary, read_images_text, qvec2rotmat
from src.dataset.colmap_utils import parse_colmap_camera_params, auto_orient_and_center_poses


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


def load_undistorted_images(indices, image_paths, camera_ids, params_dict, mapx_dict, mapy_dict, roi_undist_dict):
    
    to_tensor = tf.ToTensor()

    images = []
    
    for index in indices:
        image = imageio.imread(image_paths[index])[..., :3]  
        camera_id = camera_ids[index]
        params = params_dict[camera_id]
        if len(params) > 0:
            # Images are distorted. Undistort them.
            mapx, mapy = (
                mapx_dict[camera_id],
                mapy_dict[camera_id],
            )
            image = cv2.remap(image, mapx, mapy, cv2.INTER_LINEAR)
            x, y, w, h = roi_undist_dict[camera_id]
            image = image[y : y + h, x : x + w]

        images.append(to_tensor(image))

    return torch.stack(images)


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

    # ---------------------------------------------------------------------------------------------
    # TODO: specify scene_idx, style_image_path, left and rightmost ctx view indices, num of views
    # data_dir = Path('colmap_test_data/scenes/room')
    # data_dir = Path('colmap_test_data/scenes/flower') 
    # data_dir = Path('colmap_test_data/scenes/lego')
    # data_dir = Path('colmap_test_data/scenes/fortress')
    # data_dir = Path('/ssdwork/datasets/tnt/scene_256/scene_256/train')
    data_dir = Path('colmap_test_data/scenes/train')

    specified_style_image_path = Path('colmap_test_data/styles/tiger.jpg')
    # specified_style_image_path = Path('colmap_test_data/styles/starry_night.jpg')
    # specified_style_image_path = Path('colmap_test_data/styles/abstract.png')
    # specified_style_image_path = Path('/ssdwork/datasets/stylerf_style/5.jpg')
    # specified_style_image_path = Path('/ssdwork/datasets/stylerf_style/10.jpg')
    # specified_style_image_path = Path('/ssdwork/datasets/stylerf_style/121.jpg')
    
    ctx_indices = torch.tensor([1, 9])
    num_ctx_views = 4
    # ----------------------------------------------------------------------------------------------

    colmap_dir = data_dir / "sparse/0/"
    if not os.path.exists(colmap_dir):
        colmap_dir = data_dir / "sparse"
    assert os.path.exists(colmap_dir), f"COLMAP directory {colmap_dir} does not exist"
    
    if (colmap_dir / "cameras.txt").exists():
        cam_id_to_camera = read_cameras_text(colmap_dir / "cameras.txt")
        im_id_to_image = read_images_text(colmap_dir / "images.txt")
    elif (colmap_dir / "cameras.bin").exists():
        cam_id_to_camera = read_cameras_binary(colmap_dir / "cameras.bin")
        im_id_to_image = read_images_binary(colmap_dir / "images.bin")
    else:
        raise ValueError(f"Could not find cameras.txt or cameras.bin in {colmap_dir}")

    cameras = {}
    # Parse cameras
    for cam_id, cam_data in cam_id_to_camera.items():
        cameras[cam_id] = parse_colmap_camera_params(cam_data)
    
    # Parse frames
    # we want to sort all images based on im_id
    ordered_im_id = sorted(im_id_to_image.keys())
    
    # Extract extrinsic matrices in world-to-camera format.
    # imdata = manager.images
    w2c_mats = []
    camera_ids = []
    Ks_dict = dict()
    params_dict = dict()
    imsize_dict = dict()  # width, height
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for im_id in ordered_im_id:
        im = im_id_to_image[im_id]
        rot = qvec2rotmat(im.qvec)
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    
        # support different camera intrinsics
        camera_id = im.camera_id
        camera_ids.append(camera_id)
        
        # camera intrinsics
        cam = cameras[camera_id]
        # fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
        fx, fy, cx, cy = cam['fl_x'], cam['fl_y'], cam['cx'], cam['cy']

        # NOTE: normalize intrinsics here
        # fx, fy, cx, cy = fx / cam['w'], fy / cam['h'], cx / cam['w'], cy / cam['h']

        K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
        Ks_dict[camera_id] = K
        
        imsize_dict[camera_id] = (cam['w'], cam['h'])
        
        # Get distortion parameters.
        type_ = cam["model"]
        if type_ == 0 or type_ == "SIMPLE_PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        elif type_ == 1 or type_ == "PINHOLE":
            params = np.empty(0, dtype=np.float32)
            camtype = "perspective"
        if type_ == 2 or type_ == "SIMPLE_RADIAL":
            params = np.array([cam['k1'], 0, 0, 0], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 3 or type_ == "RADIAL":
            params = np.array([cam.k1, cam.k2, 0.0, 0.0], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 4 or type_ == "OPENCV":
            params = np.array([cam.k1, cam.k2, cam.p1, cam.p2], dtype=np.float32)
            camtype = "perspective"
        elif type_ == 5 or type_ == "OPENCV_FISHEYE":
            params = np.array([cam.k1, cam.k2, cam.k3, cam.k4], dtype=np.float32)
            camtype = "fisheye"
        assert (
            camtype == "perspective"
        ), f"Only support perspective camera model, got {type_}"
        
        params_dict[camera_id] = params
        
    
    print(
        f"[Parser] {len(im_id_to_image)} images, taken by {len(set(camera_ids))} cameras."
    )
    
    if len(im_id_to_image) == 0:
        raise ValueError("No images found in COLMAP.")
    if not (type_ == 0 or type_ == 1):
        print(f"Warning: COLMAP Camera is not PINHOLE. Images have distortion.")
    
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    image_names = [im_id_to_image[k].name for k in ordered_im_id]

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]
    camera_ids = [camera_ids[i] for i in inds]
    
    colmap_image_dir = os.path.join(data_dir, "images")
    if not os.path.exists(colmap_image_dir):
        raise ValueError(f"Image folder {colmap_image_dir} does not exist.")
    
    image_paths = [os.path.join(colmap_image_dir, f) for f in image_names]
    
    camtoworlds, transform_matrix = auto_orient_and_center_poses(
        camtoworlds, 
        method="up",
        center_method="poses"
    )
    
    # NOTE: auto scale poses by default
    scale_factor = 1.0 
    scale_factor /= float(np.max(np.abs(camtoworlds[:, :3, 3]))) 
    camtoworlds[:, :3, 3] *= scale_factor
    N = camtoworlds.shape[0]
    bottoms = np.repeat(bottom[np.newaxis, :], N, axis=0)
    camtoworlds = np.concatenate((camtoworlds, bottoms), axis=1)
    
    # undistortion
    mapx_dict = dict()
    mapy_dict = dict()
    roi_undist_dict = dict()
    for camera_id in params_dict.keys():
        params = params_dict[camera_id]
        if len(params) == 0:
            continue  # no distortion
        assert camera_id in Ks_dict, f"Missing K for camera {camera_id}"
        assert (
            camera_id in params_dict
        ), f"Missing params for camera {camera_id}"
        K = Ks_dict[camera_id]
        width, height = imsize_dict[camera_id]
        K_undist, roi_undist = cv2.getOptimalNewCameraMatrix(
            K, params, (width, height), 0
        )
        mapx, mapy = cv2.initUndistortRectifyMap(
            K, params, None, K_undist, (width, height), cv2.CV_32FC1
        )
        Ks_dict[camera_id] = K_undist
        mapx_dict[camera_id] = mapx
        mapy_dict[camera_id] = mapy
        roi_undist_dict[camera_id] = roi_undist

    # self.image_names = image_names  # List[str], (num_images,)
    # self.image_paths = image_paths  # List[str], (num_images,)
    # self.camtoworlds = camtoworlds  # np.ndarray, (num_images, 4, 4)
    # self.camera_ids = camera_ids  # List[int], (num_images,)
    # self.Ks_dict = Ks_dict  # Dict of camera_id -> K
    # self.params_dict = params_dict  # Dict of camera_id -> params
    # self.imsize_dict = imsize_dict  # Dict of camera_id -> (width, height)
    # self.transform = transform_matrix  # np.ndarray, (4, 4)

    to_tensor = tf.ToTensor()

    # XXX
    extrinsics = torch.from_numpy(camtoworlds).float()
    intrinsics = [torch.from_numpy(Ks_dict[camera_id]) for camera_id in camera_ids if camera_id in Ks_dict]
    intrinsics = torch.stack(intrinsics).float()

    scene = data_dir.name
    
    left, right = ctx_indices[0], ctx_indices[-1]
    if ctx_indices.shape[0] < num_ctx_views:
        if num_ctx_views > 2:
            # evenly distribute the additional context views between the left and right views
            ctx_indices = torch.stack(
                [torch.linspace(left, right, num_ctx_views).long()],
                dim=-1,
            ).squeeze(-1)
    elif ctx_indices.shape[0] > num_ctx_views:
        raise ValueError('You gave more context indices than required!') 

    target_indices = torch.arange(left, right)
    # exlucding ctx views in target_indices
    mask = ~torch.isin(target_indices, ctx_indices)

    if mask.any():
        target_indices = target_indices[mask]
    else:
        print('No target view, thus render context views as tagets!')

    # (optional) skip if FOV is too wide
    # assert not (get_fov(intrinsics).rad2deg()
    #             > data_cfg.max_fov).any(), "FOV too wide!"

    # Load the images.
    context_images = load_undistorted_images(ctx_indices, image_paths, camera_ids, params_dict, mapx_dict, mapy_dict, roi_undist_dict)
    target_images = load_undistorted_images(target_indices, image_paths, camera_ids, params_dict, mapx_dict, mapy_dict, roi_undist_dict)

    # context_images = [to_tensor(imageio.imread(image_paths[index])[..., :3]) for index in ctx_indices]
    # context_images = torch.stack(context_images)
    # target_images = [to_tensor(imageio.imread(image_paths[index])[..., :3]) for index in target_indices]
    # target_images = torch.stack(target_images)

    # normalize intrinsics
    # fx, fy, cx, cy = fx / cam['w'], fy / cam['h'], cx / cam['w'], cy / cam['h']
    _, _, h, w = context_images.shape
    intrinsics[:, 0 , 0] = intrinsics[:, 0 , 0] / w
    intrinsics[:, 1 , 1] = intrinsics[:, 1 , 1] / h
    intrinsics[:, 0 , -1] = intrinsics[:, 0 , -1] / w
    intrinsics[:, 1 , -1] = intrinsics[:, 1 , -1] / h

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

    # crop images to 256 x 256
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
    save_image(batch["style"]['image'][0],
               output_dir / f"{scene} | {style_image_name[:-4]}" / f"{batch['style']['image_name']}")

    for index, color in zip(batch["context"]["index"][0],
                            inverse_normalize(batch["context"]["image"][0])):
        index = index + 1
        save_image(color, output_dir / f"{scene} | {style_image_name[:-4]}" / f"context/{index:0>6}.png")

    for index, color in zip(batch["target"]["index"][0], output.color[0]):
        index = index + 1
        save_image(color, output_dir / f"{scene} | {style_image_name[:-4]}" / f"color/{index:0>6}.png")

    for index, color in zip(batch["target"]["index"][0],
                            stylized_output.color[0]):
        index = index + 1
        save_image(color, output_dir / f"{scene} | {style_image_name[:-4]}" / f"stylized_color/{index:0>6}.png")

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