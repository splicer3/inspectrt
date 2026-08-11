"""Frozen ResNet-50 patch feature extraction."""

import torch
from torch import Tensor, nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50
from torchvision.models.feature_extraction import create_feature_extractor

_RETURN_NODE = "layer2"
_IMAGE_SHAPE = (3, 256, 256)
_LAYER2_SHAPE = (512, 32, 32)
_PATCH_COUNT = 32 * 32


def build_resnet50_layer2_extractor(
    weights: ResNet50_Weights | None = ResNet50_Weights.IMAGENET1K_V2,
) -> nn.Module:
    """Build the frozen ResNet-50 extractor used by the baseline."""
    backbone = resnet50(weights=weights).eval()
    extractor = create_feature_extractor(
        backbone,
        return_nodes={_RETURN_NODE: _RETURN_NODE},
    )
    extractor.requires_grad_(False)
    return extractor.eval()


def extract_patch_embeddings(extractor: nn.Module, images: Tensor) -> Tensor:
    """Extract locally averaged, row-major layer-2 patches from an image batch."""
    _validate_image_batch(images)
    extractor.eval()

    with torch.inference_mode():
        outputs = extractor(images)
        if not isinstance(outputs, dict) or _RETURN_NODE not in outputs:
            raise RuntimeError("Extractor must return a 'layer2' tensor")
        feature_map = outputs[_RETURN_NODE]
        if not isinstance(feature_map, Tensor):
            raise RuntimeError("Extractor output 'layer2' must be a tensor")

        expected_shape = (images.shape[0], *_LAYER2_SHAPE)
        if tuple(feature_map.shape) != expected_shape:
            raise RuntimeError(
                f"Expected layer2 shape {expected_shape}, got {tuple(feature_map.shape)}"
            )
        if feature_map.dtype != torch.float32:
            raise RuntimeError(
                f"Expected layer2 dtype torch.float32, got {feature_map.dtype}"
            )

        pooled, patches = _pool_and_layout_layer2(feature_map)
        if pooled.shape != feature_map.shape:
            raise RuntimeError(
                "Local average changed layer2 shape: "
                f"{tuple(feature_map.shape)} to {tuple(pooled.shape)}"
            )

        return patches


def _pool_and_layout_layer2(feature_map: Tensor) -> tuple[Tensor, Tensor]:
    pooled = F.avg_pool2d(
        feature_map,
        kernel_size=3,
        stride=1,
        padding=1,
        count_include_pad=True,
    )
    patches = (
        pooled.permute(0, 2, 3, 1)
        .reshape(feature_map.shape[0], _PATCH_COUNT, _LAYER2_SHAPE[0])
        .contiguous()
    )
    return pooled, patches


def _validate_image_batch(images: Tensor) -> None:
    if images.ndim != 4:
        raise ValueError(
            "Expected image batch rank 4 with shape [B, 3, 256, 256], "
            f"got rank {images.ndim} and shape {tuple(images.shape)}"
        )
    if images.shape[0] < 1:
        raise ValueError("Image batch must contain at least one image")
    if images.shape[1] != _IMAGE_SHAPE[0]:
        raise ValueError(f"Expected 3 image channels, got {images.shape[1]}")
    if tuple(images.shape[2:]) != _IMAGE_SHAPE[1:]:
        raise ValueError(
            f"Expected image size 256 x 256, got {images.shape[2]} x {images.shape[3]}"
        )
    if images.dtype != torch.float32:
        raise TypeError(f"Expected image batch dtype torch.float32, got {images.dtype}")
