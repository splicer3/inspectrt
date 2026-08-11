"""In-process evaluation of one MVTec AD category."""

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import Tensor, nn

from inspectrt.data import MvtecSample, discover_mvtec_samples
from inspectrt.features import extract_patch_embeddings
from inspectrt.metrics import ThresholdFreeMetrics, compute_threshold_free_metrics
from inspectrt.preprocessing import PreprocessedImage, preprocess_image
from inspectrt.retrieval import score_patch_embeddings

_PatchExtractor = Callable[[Tensor], Tensor]


@dataclass(frozen=True, slots=True)
class MvtecSampleObservation:
    sample: MvtecSample
    original_height: int
    original_width: int
    original_mode: str


@dataclass(frozen=True, slots=True)
class CategoryEvaluation:
    category: str
    samples: tuple[MvtecSampleObservation, ...]
    test_samples: tuple[MvtecSampleObservation, ...]
    memory_bank: Tensor
    test_labels: Tensor
    pixel_masks: Tensor
    patch_distances: Tensor
    nearest_bank_indices: Tensor
    image_scores: Tensor
    anomaly_maps: Tensor
    metrics: ThresholdFreeMetrics


def evaluate_mvtec_category(
    dataset_root: Path,
    category: str,
    feature_extractor: nn.Module,
    *,
    device: torch.device | str,
    bank_chunk_size: int,
) -> CategoryEvaluation:
    """Evaluate one MVTec AD category with the feature-memory baseline."""
    samples, nominal_samples, test_samples = _discover_category_samples(
        dataset_root, category
    )
    resolved_device = _resolve_evaluation_device(feature_extractor, device)
    return _evaluate_mvtec_category_with_patch_extractor(
        dataset_root,
        category,
        samples,
        nominal_samples,
        test_samples,
        lambda images: extract_patch_embeddings(feature_extractor, images),
        resolved_device=resolved_device,
        bank_chunk_size=bank_chunk_size,
    )


def _evaluate_mvtec_category_with_patch_extractor(
    dataset_root: Path,
    category: str,
    samples: tuple[MvtecSample, ...],
    nominal_samples: tuple[MvtecSample, ...],
    test_samples: tuple[MvtecSample, ...],
    patch_extractor: _PatchExtractor,
    *,
    resolved_device: torch.device,
    bank_chunk_size: int,
) -> CategoryEvaluation:
    observations: dict[str, MvtecSampleObservation] = {}
    memory_bank = _build_nominal_memory_bank(
        dataset_root,
        nominal_samples,
        patch_extractor,
        resolved_device,
        observations,
    )
    retrieval_bank = _transfer_memory_bank(memory_bank, resolved_device)
    return _score_and_finalize_category(
        dataset_root,
        category,
        samples,
        test_samples,
        patch_extractor,
        resolved_device,
        bank_chunk_size,
        memory_bank,
        retrieval_bank,
        observations,
    )


def _discover_category_samples(
    dataset_root: Path, category: str
) -> tuple[tuple[MvtecSample, ...], tuple[MvtecSample, ...], tuple[MvtecSample, ...]]:
    samples = tuple(discover_mvtec_samples(dataset_root, category))
    nominal_samples = tuple(
        sample
        for sample in samples
        if sample.split == "train"
        and sample.defect_type == "good"
        and not sample.is_anomalous
    )
    test_samples = tuple(sample for sample in samples if sample.split == "test")
    if not nominal_samples:
        raise ValueError(f"No nominal training samples discovered for {category!r}")
    if not test_samples:
        raise ValueError(f"No test samples discovered for {category!r}")
    return samples, nominal_samples, test_samples


def _resolve_evaluation_device(
    feature_extractor: nn.Module, device: torch.device | str
) -> torch.device:
    resolved_device = torch.device(device)
    cuda_unavailable = resolved_device.type == "cuda" and (
        not torch.cuda.is_available()
        or (
            resolved_device.index is not None
            and resolved_device.index >= torch.cuda.device_count()
        )
    )
    if cuda_unavailable:
        raise RuntimeError(f"CUDA device {resolved_device} requested but unavailable")
    feature_extractor.to(resolved_device).eval()
    return resolved_device


def _load_and_extract_sample(
    dataset_root: Path,
    sample: MvtecSample,
    feature_extractor: nn.Module | _PatchExtractor,
    resolved_device: torch.device,
    observations: dict[str, MvtecSampleObservation],
) -> tuple[PreprocessedImage, Tensor]:
    mask_path = dataset_root / sample.mask_relpath if sample.mask_relpath else None
    prepared = preprocess_image(dataset_root / sample.image_relpath, mask_path)
    observations[sample.sample_id] = MvtecSampleObservation(
        sample,
        prepared.original_height,
        prepared.original_width,
        prepared.original_mode,
    )
    images = prepared.image.unsqueeze(0).to(resolved_device)
    patches = (
        extract_patch_embeddings(feature_extractor, images)
        if isinstance(feature_extractor, nn.Module)
        else feature_extractor(images)
    )
    return prepared, patches


def _build_nominal_memory_bank(
    dataset_root: Path,
    nominal_samples: tuple[MvtecSample, ...],
    feature_extractor: nn.Module | _PatchExtractor,
    resolved_device: torch.device,
    observations: dict[str, MvtecSampleObservation],
) -> Tensor:
    return torch.cat(
        [
            _load_and_extract_sample(
                dataset_root,
                sample,
                feature_extractor,
                resolved_device,
                observations,
            )[1][0]
            .detach()
            .cpu()
            for sample in nominal_samples
        ]
    ).contiguous()


def _transfer_memory_bank(memory_bank: Tensor, resolved_device: torch.device) -> Tensor:
    return memory_bank.to(resolved_device)


def _score_and_finalize_category(
    dataset_root: Path,
    category: str,
    samples: tuple[MvtecSample, ...],
    test_samples: tuple[MvtecSample, ...],
    feature_extractor: nn.Module | _PatchExtractor,
    resolved_device: torch.device,
    bank_chunk_size: int,
    memory_bank: Tensor,
    retrieval_bank: Tensor,
    observations: dict[str, MvtecSampleObservation],
) -> CategoryEvaluation:
    masks = []
    test_outputs = []
    for sample in test_samples:
        prepared, patches = _load_and_extract_sample(
            dataset_root,
            sample,
            feature_extractor,
            resolved_device,
            observations,
        )
        scores = score_patch_embeddings(
            patches, retrieval_bank, bank_chunk_size=bank_chunk_size
        )
        masks.append(prepared.evaluation_mask)
        score_tensors = (
            scores.patch_distances,
            scores.nearest_bank_indices,
            scores.image_scores,
            scores.anomaly_maps,
        )
        test_outputs.append(tuple(output.detach().cpu() for output in score_tensors))

    if len(masks) != len(test_samples) or len(test_outputs) != len(test_samples):
        raise RuntimeError("Test sample and output counts are inconsistent")
    test_labels = torch.tensor(
        [int(sample.is_anomalous) for sample in test_samples], dtype=torch.uint8
    ).contiguous()
    pixel_masks = torch.stack(masks).contiguous()
    patch_distances, nearest_bank_indices, raw_image_scores, raw_anomaly_maps = (
        torch.cat([outputs[index] for outputs in test_outputs]).contiguous()
        for index in range(4)
    )

    metrics = compute_threshold_free_metrics(
        test_labels, raw_image_scores, pixel_masks, raw_anomaly_maps
    )
    return CategoryEvaluation(
        category=category,
        samples=tuple(observations[sample.sample_id] for sample in samples),
        test_samples=tuple(observations[sample.sample_id] for sample in test_samples),
        memory_bank=memory_bank,
        test_labels=test_labels,
        pixel_masks=pixel_masks,
        patch_distances=patch_distances,
        nearest_bank_indices=nearest_bank_indices,
        image_scores=raw_image_scores,
        anomaly_maps=raw_anomaly_maps,
        metrics=metrics,
    )
