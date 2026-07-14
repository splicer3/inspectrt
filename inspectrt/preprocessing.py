"""Canonical inspection image and mask preprocessing."""

from dataclasses import dataclass
from pathlib import Path

import torch
from PIL import Image
from torch import Tensor
from torchvision.models import ResNet50_Weights
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

PREPROCESSING_PROFILE = "inspectrt_resize256_v1"

_OUTPUT_SIZE = [256, 256]


@dataclass(frozen=True, slots=True)
class PreprocessedImage:
    image: Tensor
    evaluation_mask: Tensor
    original_height: int
    original_width: int
    original_mode: str


def preprocess_image(
    image_path: Path, mask_path: Path | None = None
) -> PreprocessedImage:
    """Decode and preprocess one inspection image and its optional mask."""
    _require_file(image_path, "Image")
    if mask_path is not None:
        _require_file(mask_path, "Mask")

    with Image.open(image_path) as source_image:
        original_width, original_height = source_image.size
        original_mode = source_image.mode
        with source_image.convert("RGB") as rgb_image:
            with F.resize(
                rgb_image,
                _OUTPUT_SIZE,
                interpolation=InterpolationMode.BILINEAR,
                antialias=True,
            ) as resized_image:
                image = F.to_tensor(resized_image)

    normalization = ResNet50_Weights.IMAGENET1K_V2.transforms()
    image = F.normalize(image, normalization.mean, normalization.std).contiguous()

    if mask_path is None:
        evaluation_mask = torch.zeros(_OUTPUT_SIZE, dtype=torch.uint8)
    else:
        evaluation_mask = _decode_mask(
            mask_path, expected_size=(original_width, original_height)
        )

    return PreprocessedImage(
        image=image,
        evaluation_mask=evaluation_mask,
        original_height=original_height,
        original_width=original_width,
        original_mode=original_mode,
    )


def _decode_mask(mask_path: Path, expected_size: tuple[int, int]) -> Tensor:
    with Image.open(mask_path) as source_mask:
        if source_mask.size != expected_size:
            raise ValueError(
                "Image and mask dimensions differ: "
                f"image {expected_size}, mask {source_mask.size}"
            )
        mask = F.pil_to_tensor(source_mask).to(dtype=torch.float32)

    mask = F.resize(
        mask,
        _OUTPUT_SIZE,
        interpolation=InterpolationMode.NEAREST_EXACT,
        antialias=False,
    )
    return mask.ne(0).any(dim=0).to(dtype=torch.uint8).contiguous()


def _require_file(path: Path, description: str) -> None:
    if not path.exists():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_file():
        raise IsADirectoryError(f"{description} is not a file: {path}")
