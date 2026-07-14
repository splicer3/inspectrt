"""Exact patch-memory retrieval and baseline anomaly scoring."""

from dataclasses import dataclass

import torch
from torch import Tensor
from torch.nn import functional as F

_PATCH_GRID_SIZE = 32
_PATCH_COUNT = _PATCH_GRID_SIZE**2
_FEATURE_DIMENSION = 512
_ANOMALY_MAP_SIZE = (256, 256)


@dataclass(frozen=True, slots=True)
class PatchMemoryScores:
    patch_distances: Tensor
    nearest_bank_indices: Tensor
    image_scores: Tensor
    anomaly_maps: Tensor


def exact_top1_squared_l2(
    queries: Tensor,
    memory_bank: Tensor,
    *,
    bank_chunk_size: int,
) -> tuple[Tensor, Tensor]:
    """Return exact squared-L2 distance and bank index for each query."""
    _validate_chunk_size(bank_chunk_size)
    _validate_retrieval_inputs(queries, memory_bank)

    for start in range(0, memory_bank.shape[0], bank_chunk_size):
        bank_chunk = memory_bank[start : start + bank_chunk_size]
        chunk_distances, chunk_indices = (
            torch.cdist(queries, bank_chunk, p=2).square().min(dim=1)
        )
        chunk_indices = chunk_indices + start
        if start == 0:
            best_distances, best_indices = chunk_distances, chunk_indices
            continue
        replace = chunk_distances < best_distances
        best_distances = torch.where(replace, chunk_distances, best_distances)
        best_indices = torch.where(replace, chunk_indices, best_indices)

    return best_distances.contiguous(), best_indices.contiguous()


def score_patch_embeddings(
    patch_embeddings: Tensor,
    memory_bank: Tensor,
    *,
    bank_chunk_size: int,
) -> PatchMemoryScores:
    """Score canonical ResNet patch embeddings against a nominal bank."""
    _validate_fp32_batch(
        patch_embeddings, "Patch embeddings", (_PATCH_COUNT, _FEATURE_DIMENSION)
    )

    batch_size = patch_embeddings.shape[0]
    distances, indices = exact_top1_squared_l2(
        patch_embeddings.reshape(batch_size * _PATCH_COUNT, _FEATURE_DIMENSION),
        memory_bank,
        bank_chunk_size=bank_chunk_size,
    )
    patch_distances = distances.reshape(batch_size, _PATCH_COUNT).contiguous()
    nearest_bank_indices = indices.reshape(batch_size, _PATCH_COUNT).contiguous()
    return PatchMemoryScores(
        patch_distances=patch_distances,
        nearest_bank_indices=nearest_bank_indices,
        image_scores=patch_distances.max(dim=1).values.contiguous(),
        anomaly_maps=reconstruct_anomaly_maps(patch_distances),
    )


def reconstruct_anomaly_maps(patch_distances: Tensor) -> Tensor:
    """Reconstruct raw maps from row-major squared patch distances."""
    _validate_fp32_batch(patch_distances, "Patch distances", (_PATCH_COUNT,))
    grid = patch_distances.reshape(-1, 1, _PATCH_GRID_SIZE, _PATCH_GRID_SIZE)
    return F.interpolate(
        grid,
        size=_ANOMALY_MAP_SIZE,
        mode="bilinear",
        align_corners=False,
    )[:, 0].contiguous()


def _validate_chunk_size(bank_chunk_size: int) -> None:
    if not isinstance(bank_chunk_size, int) or isinstance(bank_chunk_size, bool):
        raise TypeError("bank_chunk_size must be a positive integer")
    if bank_chunk_size <= 0:
        raise ValueError("bank_chunk_size must be a positive integer")


def _validate_retrieval_inputs(queries: Tensor, memory_bank: Tensor) -> None:
    for name, tensor in (("Queries", queries), ("Memory bank", memory_bank)):
        if tensor.ndim != 2:
            raise ValueError(
                f"{name} must have rank 2; got shape {tuple(tensor.shape)}"
            )
        if 0 in tensor.shape:
            raise ValueError(
                f"{name} must be non-empty; got shape {tuple(tensor.shape)}"
            )
        if tensor.dtype != torch.float32:
            raise TypeError(f"{name} must use torch.float32; got {tensor.dtype}")
    if queries.device != memory_bank.device:
        devices = queries.device, memory_bank.device
        raise ValueError(
            f"Queries and memory bank must use the same device; got {devices}"
        )
    if queries.shape[1] != memory_bank.shape[1]:
        dimensions = queries.shape[1], memory_bank.shape[1]
        raise ValueError(
            f"Queries and memory bank feature dimensions differ: {dimensions}"
        )


def _validate_fp32_batch(tensor: Tensor, name: str, tail: tuple[int, ...]) -> None:
    expected = f"[B, {', '.join(str(size) for size in tail)}]"
    actual = tuple(tensor.shape)
    if tensor.ndim != len(tail) + 1:
        raise ValueError(f"{name} must have shape {expected}; got {actual}")
    if tensor.shape[0] < 1:
        raise ValueError(f"{name} batch must contain at least one item")
    if tuple(tensor.shape[1:]) != tail:
        raise ValueError(f"{name} must have shape {expected}; got {actual}")
    if tensor.dtype != torch.float32:
        raise TypeError(f"{name} must use torch.float32; got {tensor.dtype}")
