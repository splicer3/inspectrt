"""Validated threshold-free baseline metrics."""

from dataclasses import dataclass

import torch
from sklearn.metrics import average_precision_score, roc_auc_score
from torch import Tensor

_PIXEL_SIZE = (256, 256)
_BINARY_DTYPES = {
    torch.bool,
    torch.uint8,
    torch.int8,
    torch.int16,
    torch.int32,
    torch.int64,
    torch.uint16,
    torch.uint32,
    torch.uint64,
}


@dataclass(frozen=True, slots=True)
class ThresholdFreeMetrics:
    image_auroc: float
    image_average_precision: float
    pixel_auroc: float


def compute_threshold_free_metrics(
    image_labels: Tensor,
    image_scores: Tensor,
    pixel_masks: Tensor,
    anomaly_maps: Tensor,
) -> ThresholdFreeMetrics:
    """Compute image AUROC, image Average Precision, and pixel AUROC."""
    inputs = (
        (image_labels, "Image labels", ()),
        (image_scores, "Image scores", ()),
        (pixel_masks, "Pixel masks", _PIXEL_SIZE),
        (anomaly_maps, "Anomaly maps", _PIXEL_SIZE),
    )
    for tensor, name, tail in inputs:
        expected = f"[N{''.join(f', {size}' for size in tail)}]"
        if tensor.ndim != len(tail) + 1 or tuple(tensor.shape[1:]) != tail:
            raise ValueError(
                f"{name} must have shape {expected}; got {tuple(tensor.shape)}"
            )
        if tensor.shape[0] < 1:
            raise ValueError(f"{name} must contain at least one sample")

    counts = tuple(tensor.shape[0] for tensor, _, _ in inputs)
    if len(set(counts)) != 1:
        raise ValueError(
            "Sample counts must match for image labels, image scores, pixel masks, "
            f"and anomaly maps; got {counts}"
        )
    for tensor, name in ((image_labels, "Image labels"), (pixel_masks, "Pixel masks")):
        if tensor.dtype not in _BINARY_DTYPES:
            raise TypeError(
                f"{name} must use a boolean or integer dtype; got {tensor.dtype}"
            )
    for tensor, name in (
        (image_scores, "Image scores"),
        (anomaly_maps, "Anomaly maps"),
    ):
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must use torch.float32; got {tensor.dtype}")

    labels, scores, masks, maps = (
        tensor.detach().cpu()
        for tensor in (image_labels, image_scores, pixel_masks, anomaly_maps)
    )
    for target, name in ((labels, "Image labels"), (masks, "Pixel masks")):
        if not torch.logical_or(target == 0, target == 1).all().item():
            raise ValueError(f"{name} must contain only 0 and 1")
    for values, name in ((scores, "Image scores"), (maps, "Anomaly maps")):
        if not torch.isfinite(values).all().item():
            raise ValueError(f"{name} must contain only finite values")
    if not labels.eq(1).any().item():
        raise ValueError(
            "Image AUROC is undefined because image labels contain only normal samples"
        )
    if not labels.eq(0).any().item():
        raise ValueError(
            "Image AUROC is undefined because image labels contain only anomalous samples"
        )

    pixel_targets = masks.reshape(-1)
    pixel_scores = maps.reshape(-1)
    if not pixel_targets.eq(1).any().item():
        raise ValueError(
            "Pixel AUROC is undefined because flattened pixel masks contain only "
            "background pixels"
        )
    if not pixel_targets.eq(0).any().item():
        raise ValueError(
            "Pixel AUROC is undefined because flattened pixel masks contain only "
            "foreground pixels"
        )

    image_targets = labels.to(dtype=torch.uint8).numpy()
    pixel_targets = pixel_targets.to(dtype=torch.uint8).numpy()
    image_values = scores.numpy()
    return ThresholdFreeMetrics(
        image_auroc=float(roc_auc_score(image_targets, image_values)),
        image_average_precision=float(
            average_precision_score(image_targets, image_values)
        ),
        pixel_auroc=float(roc_auc_score(pixel_targets, pixel_scores.numpy())),
    )
