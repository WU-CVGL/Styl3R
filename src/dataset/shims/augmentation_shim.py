import torch
from jaxtyping import Float
from torch import Tensor
from torchvision import transforms

from ..types import AnyExample, AnyViews
from .crop_shim import rescale

def reflect_extrinsics(
    extrinsics: Float[Tensor, "*batch 4 4"],
) -> Float[Tensor, "*batch 4 4"]:
    reflect = torch.eye(4, dtype=torch.float32, device=extrinsics.device)
    reflect[0, 0] = -1
    return reflect @ extrinsics @ reflect


def reflect_views(views: AnyViews) -> AnyViews:
    return {
        **views,
        "image": views["image"].flip(-1),
        "extrinsics": reflect_extrinsics(views["extrinsics"]),
    }


def apply_augmentation_shim(
    example: AnyExample,
    generator: torch.Generator | None = None,
) -> AnyExample:
    """Randomly augment the training images."""
    # Do not augment with 50% chance.
    if torch.rand(tuple(), generator=generator) < 0.5:
        return example

    return {
        **example,
        "context": reflect_views(example["context"]),
        "target": reflect_views(example["target"]),
    }
    
def apply_style_image_augmentation(style_image: torch.Tensor, stage) -> torch.Tensor:
    
    # NOTE: no arg for now
    train_transform = transforms.RandomCrop(256)
    val_transform = transforms.CenterCrop(256)
    
    _, H, W = style_image.shape
    if H < W:
        ratio = W / H
        H = 256
        W = int(ratio * H)
    else:
        ratio = H / W
        W = 256
        H = int(ratio * W)
    
    style_image = rescale(style_image, (H, W))
    if stage == "train":
        style_image = val_transform(style_image) # 
    else:
        style_image = val_transform(style_image)
    
    return style_image

def apply_style_image_augmentation_larger(style_image: torch.Tensor, stage) -> torch.Tensor:
    
    # NOTE: no arg for now
    train_transform = transforms.RandomCrop(352)
    val_transform = transforms.CenterCrop(352)
    
    _, H, W = style_image.shape
    if H < W:
        ratio = W / H
        H = 352
        W = int(ratio * H)
    else:
        ratio = H / W
        W = 352
        H = int(ratio * W)
    
    style_image = rescale(style_image, (H, W))
    if stage == "train":
        style_image = val_transform(style_image) # 
    else:
        style_image = val_transform(style_image)
    
    return style_image
