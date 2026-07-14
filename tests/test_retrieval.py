import pytest
import torch
from torch.nn import functional as F

from inspectrt.retrieval import (
    exact_top1_squared_l2,
    reconstruct_anomaly_maps,
    score_patch_embeddings,
)


def test_matches_unchunked_reference_for_several_chunk_sizes() -> None:
    generator = torch.Generator().manual_seed(7)
    queries = torch.randn(5, 4, generator=generator)
    memory_bank = torch.randn(7, 4, generator=generator)
    expected_distances, expected_indices = (
        torch.cdist(queries, memory_bank, p=2).square().min(dim=1)
    )

    for chunk_size in (1, 4, 7, 10):
        distances, indices = exact_top1_squared_l2(
            queries, memory_bank, bank_chunk_size=chunk_size
        )
        torch.testing.assert_close(distances, expected_distances, rtol=1e-6, atol=1e-6)
        assert torch.equal(indices, expected_indices)
        assert distances.dtype == torch.float32
        assert indices.dtype == torch.int64
        assert distances.device == queries.device == indices.device
        assert distances.is_contiguous() and indices.is_contiguous()


def test_uses_squared_l2_distance() -> None:
    queries = torch.tensor(((0.0, 0.0), (1.0, 2.0)))
    memory_bank = torch.tensor(((3.0, 2.0), (0.0, 0.0)))

    distances, indices = exact_top1_squared_l2(queries, memory_bank, bank_chunk_size=1)

    assert torch.equal(distances, torch.tensor((0.0, 4.0)))
    assert torch.equal(indices, torch.tensor((1, 0)))


def test_ties_return_lower_global_index() -> None:
    cases = (
        (torch.tensor(((-1.0, 0.0), (1.0, 0.0))), 2, 0),
        (torch.tensor(((9.0, 0.0), (-1.0, 0.0), (1.0, 0.0))), 2, 1),
    )
    for memory_bank, chunk_size, expected_index in cases:
        distances, indices = exact_top1_squared_l2(
            torch.zeros(1, 2), memory_bank, bank_chunk_size=chunk_size
        )
        assert torch.equal(distances, torch.ones(1))
        assert indices.item() == expected_index


def test_scores_canonical_patch_batches() -> None:
    patches = torch.zeros(2, 1024, 512)
    patches[0, 17, 0] = 2
    patches[1, 23, 0] = 3

    result = score_patch_embeddings(patches, torch.zeros(1, 512), bank_chunk_size=3)

    assert result.patch_distances.shape == (2, 1024)
    assert result.nearest_bank_indices.shape == (2, 1024)
    assert result.image_scores.shape == (2,)
    assert result.anomaly_maps.shape == (2, 256, 256)
    assert torch.equal(result.image_scores, result.patch_distances.max(dim=1).values)
    assert torch.equal(result.image_scores, torch.tensor((4.0, 9.0)))
    outputs = (
        result.patch_distances,
        result.nearest_bank_indices,
        result.image_scores,
        result.anomaly_maps,
    )
    assert all(output.device == patches.device for output in outputs)
    assert all(output.is_contiguous() for output in outputs)
    assert result.patch_distances.dtype == result.image_scores.dtype == torch.float32
    assert result.anomaly_maps.dtype == torch.float32
    assert result.nearest_bank_indices.dtype == torch.int64


def test_reconstructs_raw_squared_maps_with_canonical_geometry() -> None:
    scores = torch.arange(1024, dtype=torch.float32).reshape(1, 1024)
    grid = scores.reshape(1, 1, 32, 32)

    maps = reconstruct_anomaly_maps(scores)
    expected = F.interpolate(
        grid, size=(256, 256), mode="bilinear", align_corners=False
    )[:, 0]
    assert torch.equal(maps, expected)
    assert maps[0, 4, 4].item() == 2.0625
    assert maps[0, 0, -1].item() == 31
    assert maps[0, -1, 0].item() == 32 * 31

    squared_scores = torch.zeros(1, 1024)
    squared_scores[0, 1] = 4
    squared_maps = reconstruct_anomaly_maps(squared_scores)
    euclidean_then_square = reconstruct_anomaly_maps(squared_scores.sqrt()).square()
    expected_squared = F.interpolate(
        squared_scores.reshape(1, 1, 32, 32),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    assert torch.equal(squared_maps, expected_squared)
    assert not torch.equal(squared_maps, euclidean_then_square)


def test_rejects_invalid_retrieval_inputs() -> None:
    valid = torch.zeros(1, 2)
    cases = (
        (torch.zeros(2), valid, ValueError, "Queries.*rank 2"),
        (valid, torch.zeros(1, 1, 2), ValueError, "Memory bank.*rank 2"),
        (torch.zeros(0, 2), valid, ValueError, "Queries.*non-empty"),
        (valid, torch.zeros(0, 2), ValueError, "Memory bank.*non-empty"),
        (torch.zeros(1, 0), torch.zeros(1, 0), ValueError, "non-empty"),
        (valid, torch.zeros(1, 3), ValueError, "dimensions differ"),
        (valid.double(), valid, TypeError, "float32"),
        (valid, valid.double(), TypeError, "float32"),
        (valid, torch.zeros(1, 2, device="meta"), ValueError, "same device"),
    )
    for queries, memory_bank, error, message in cases:
        with pytest.raises(error, match=message):
            exact_top1_squared_l2(queries, memory_bank, bank_chunk_size=1)

    for chunk_size, error in (
        (0, ValueError),
        (-1, ValueError),
        (1.5, TypeError),
        (True, TypeError),
    ):
        with pytest.raises(error, match="positive integer"):
            exact_top1_squared_l2(valid, valid, bank_chunk_size=chunk_size)


def test_rejects_invalid_patch_scoring_inputs() -> None:
    cases = (
        ((1024, 512), torch.float32, ValueError, "B, 1024, 512"),
        ((0, 1024, 512), torch.float32, ValueError, "at least one item"),
        ((1, 1023, 512), torch.float32, ValueError, "1024, 512"),
        ((1, 1024, 511), torch.float32, ValueError, "1024, 512"),
        ((1, 1024, 512), torch.float64, TypeError, "float32"),
    )
    for shape, dtype, error, message in cases:
        with pytest.raises(error, match=message):
            score_patch_embeddings(
                torch.empty(shape, dtype=dtype),
                torch.zeros(1, 512),
                bank_chunk_size=1,
            )
