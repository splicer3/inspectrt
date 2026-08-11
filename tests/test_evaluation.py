from pathlib import Path

import pytest
import torch
from PIL import Image
from torch import Tensor, nn

from inspectrt.data import discover_mvtec_samples
from inspectrt.evaluation import (
    CategoryEvaluation,
    _discover_category_samples,
    _evaluate_mvtec_category_with_patch_extractor,
    evaluate_mvtec_category,
)
from inspectrt.features import extract_patch_embeddings
from inspectrt.metrics import compute_threshold_free_metrics
from inspectrt.preprocessing import preprocess_image
from inspectrt.retrieval import score_patch_embeddings

_CHUNK_SIZE = 4096


class _ControlledExtractor(nn.Module):
    def forward(self, images: Tensor) -> dict[str, Tensor]:
        signal = images[:, :1, ::8, ::8]
        return {"layer2": signal.expand(-1, 512, -1, -1).contiguous()}


def _save(root: Path, relpath: str, image: Image.Image) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    with image:
        image.save(path)


def _make_dataset(root: Path) -> None:
    root = root / "bottle"
    _save(root, "train/good/010.png", Image.new("RGB", (8, 5), (64, 64, 64)))
    _save(root, "train/good/002.png", Image.new("L", (7, 4), 0))
    _save(root, "test/good/001.png", Image.new("RGB", (6, 3), 0))
    _save(root, "test/scratch/001.png", Image.new("L", (5, 7), 255))
    mask = Image.new("L", (5, 7), 0)
    mask.paste(255, (1, 2, 4, 6))
    _save(root, "ground_truth/scratch/001_mask.png", mask)


def _patches(root: Path, relpaths: list[str]) -> Tensor:
    extractor = _ControlledExtractor()
    return torch.cat(
        [
            extract_patch_embeddings(
                extractor, preprocess_image(root / relpath).image.unsqueeze(0)
            )[0]
            for relpath in relpaths
        ]
    )


def _evaluate(root: Path, device: torch.device | str = "cpu") -> CategoryEvaluation:
    return evaluate_mvtec_category(
        root,
        "bottle",
        _ControlledExtractor(),
        device=device,
        bank_chunk_size=_CHUNK_SIZE,
    )


def test_evaluates_category_without_test_leakage_and_preserves_contract(
    tmp_path: Path,
) -> None:
    _make_dataset(tmp_path)
    result = _evaluate(tmp_path)

    discovered = discover_mvtec_samples(tmp_path, "bottle")
    assert [item.sample for item in result.samples] == discovered
    assert [item.sample.sample_id for item in result.test_samples] == [
        "mvtec_ad/bottle/test/good/001.png",
        "mvtec_ad/bottle/test/scratch/001.png",
    ]
    assert result.test_labels.tolist() == [0, 1]
    assert [
        (item.original_height, item.original_width, item.original_mode)
        for item in result.samples
    ] == [(3, 6, "RGB"), (7, 5, "L"), (4, 7, "L"), (5, 8, "RGB")]

    expected_bank = _patches(
        tmp_path,
        ["bottle/train/good/002.png", "bottle/train/good/010.png"],
    )
    assert torch.equal(result.memory_bank, expected_bank)
    assert result.nearest_bank_indices[0].max().item() < 1024
    assert result.patch_distances[1].min().item() > 0

    expected_mask = preprocess_image(
        tmp_path / "bottle/test/scratch/001.png",
        tmp_path / "bottle/ground_truth/scratch/001_mask.png",
    ).evaluation_mask
    assert result.pixel_masks[0].count_nonzero().item() == 0
    assert torch.equal(result.pixel_masks[1], expected_mask)

    tensor_contract = {
        "memory_bank": ((2048, 512), torch.float32),
        "test_labels": ((2,), torch.uint8),
        "pixel_masks": ((2, 256, 256), torch.uint8),
        "patch_distances": ((2, 1024), torch.float32),
        "nearest_bank_indices": ((2, 1024), torch.int64),
        "image_scores": ((2,), torch.float32),
        "anomaly_maps": ((2, 256, 256), torch.float32),
    }
    for name, (shape, dtype) in tensor_contract.items():
        tensor = getattr(result, name)
        assert tensor.shape == shape
        assert tensor.dtype == dtype
        assert tensor.device.type == "cpu"
        assert tensor.is_contiguous()

    test_relpaths = [item.sample.image_relpath for item in result.test_samples]
    for position, patches in enumerate(_patches(tmp_path, test_relpaths).split(1024)):
        direct = score_patch_embeddings(
            patches.unsqueeze(0), result.memory_bank, bank_chunk_size=_CHUNK_SIZE
        )
        for name in (
            "patch_distances",
            "nearest_bank_indices",
            "image_scores",
            "anomaly_maps",
        ):
            torch.testing.assert_close(
                getattr(result, name)[position], getattr(direct, name)[0]
            )
    assert result.metrics == compute_threshold_free_metrics(
        result.test_labels,
        result.image_scores,
        result.pixel_masks,
        result.anomaly_maps,
    )
    repeated = _evaluate(tmp_path, torch.device("cpu"))
    assert repeated.samples == result.samples
    assert repeated.test_samples == result.test_samples
    assert repeated.metrics == result.metrics
    for name in tensor_contract:
        assert torch.equal(getattr(repeated, name), getattr(result, name))


def test_patch_extractor_callable_reuses_complete_category_orchestration(
    tmp_path: Path,
) -> None:
    _make_dataset(tmp_path)
    expected = _evaluate(tmp_path)
    samples, nominal_samples, test_samples = _discover_category_samples(
        tmp_path, "bottle"
    )
    calls = []
    extractor = _ControlledExtractor()

    def extract_patches(images: Tensor) -> Tensor:
        calls.append(
            (
                tuple(images.shape),
                images.dtype,
                images.device,
                images.is_contiguous(),
            )
        )
        return extract_patch_embeddings(extractor, images)

    result = _evaluate_mvtec_category_with_patch_extractor(
        tmp_path,
        "bottle",
        samples,
        nominal_samples,
        test_samples,
        extract_patches,
        resolved_device=torch.device("cpu"),
        bank_chunk_size=_CHUNK_SIZE,
    )

    assert calls == [((1, 3, 256, 256), torch.float32, torch.device("cpu"), True)] * 4
    assert result.samples == expected.samples
    assert result.test_samples == expected.test_samples
    assert result.metrics == expected.metrics
    for name in (
        "memory_bank",
        "test_labels",
        "pixel_masks",
        "patch_distances",
        "nearest_bank_indices",
        "image_scores",
        "anomaly_maps",
    ):
        assert torch.equal(getattr(result, name), getattr(expected, name))


@pytest.mark.parametrize(
    ("empty", "message"),
    [("train", "No nominal training samples"), ("test", "No test samples")],
)
def test_rejects_empty_training_or_test_partitions(
    tmp_path: Path, empty: str, message: str
) -> None:
    _make_dataset(tmp_path)
    for sample in discover_mvtec_samples(tmp_path, "bottle"):
        if sample.split == empty:
            (tmp_path / sample.image_relpath).unlink()

    with pytest.raises(ValueError, match=message):
        _evaluate(tmp_path)


def test_rejects_unavailable_cuda(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _make_dataset(tmp_path)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="CUDA.*unavailable"):
        _evaluate(tmp_path, "cuda")
