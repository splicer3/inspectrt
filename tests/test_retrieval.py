import pytest
import torch
from torch.nn import functional as F

from inspectrt.retrieval import (
    exact_top1_squared_l2,
    reconstruct_anomaly_maps,
    score_patch_embeddings,
)


def test_exact_squared_l2_is_chunk_invariant_and_breaks_computed_ties_by_index() -> (
    None
):
    queries = torch.tensor(((0.0, 0.0), (1.0, 2.0), (0.0, 0.0)))
    memory_bank = torch.tensor(((3.0, 2.0), (0.0, 0.0), (-1.0, 0.0), (1.0, 0.0)))
    expected_distances, expected_indices = (
        torch.cdist(queries, memory_bank).square().min(dim=1)
    )
    for chunk_size in (1, 3, 4, 9):
        distances, indices = exact_top1_squared_l2(
            queries, memory_bank, bank_chunk_size=chunk_size
        )
        torch.testing.assert_close(distances, expected_distances)
        assert torch.equal(indices, expected_indices)
        assert indices[2].item() == 1

    tie_distances, tie_indices = exact_top1_squared_l2(
        torch.zeros(1, 2),
        torch.tensor(((9.0, 0.0), (-1.0, 0.0), (1.0, 0.0))),
        bank_chunk_size=2,
    )
    assert torch.equal(tie_distances, torch.ones(1))
    assert tie_indices.item() == 1


def test_patch_scores_and_raw_squared_maps_preserve_the_public_contract() -> None:
    patches = torch.zeros(2, 1024, 512)
    patches[0, 17, 0] = 2
    patches[1, 23, 0] = 3
    result = score_patch_embeddings(patches, torch.zeros(1, 512), bank_chunk_size=3)

    assert result.patch_distances.shape == (2, 1024)
    assert result.nearest_bank_indices.shape == (2, 1024)
    assert torch.equal(result.image_scores, torch.tensor((4.0, 9.0)))
    assert result.anomaly_maps.shape == (2, 256, 256)
    assert result.patch_distances.dtype == result.image_scores.dtype == torch.float32
    assert result.nearest_bank_indices.dtype == torch.int64
    assert all(
        value.is_contiguous()
        for value in (
            result.patch_distances,
            result.nearest_bank_indices,
            result.image_scores,
            result.anomaly_maps,
        )
    )

    scores = torch.arange(1024, dtype=torch.float32).reshape(1, 1024)
    expected = F.interpolate(
        scores.reshape(1, 1, 32, 32),
        size=(256, 256),
        mode="bilinear",
        align_corners=False,
    )[:, 0]
    assert torch.equal(reconstruct_anomaly_maps(scores), expected)
    squared = torch.zeros(1, 1024)
    squared[0, 1] = 4
    assert not torch.equal(
        reconstruct_anomaly_maps(squared),
        reconstruct_anomaly_maps(squared.sqrt()).square(),
    )


def test_rejects_invalid_retrieval_and_patch_batch_contracts() -> None:
    valid = torch.zeros(1, 2)
    retrieval_cases = (
        (torch.zeros(2), valid, ValueError, "rank 2"),
        (torch.zeros(0, 2), valid, ValueError, "non-empty"),
        (valid, torch.zeros(1, 3), ValueError, "dimensions differ"),
        (valid.double(), valid, TypeError, "float32"),
        (valid, torch.zeros(1, 2, device="meta"), ValueError, "same device"),
    )
    for queries, bank, error, message in retrieval_cases:
        with pytest.raises(error, match=message):
            exact_top1_squared_l2(queries, bank, bank_chunk_size=1)
    with pytest.raises((TypeError, ValueError), match="positive integer"):
        exact_top1_squared_l2(valid, valid, bank_chunk_size=0)

    for shape, dtype in (
        ((1024, 512), torch.float32),
        ((0, 1024, 512), torch.float32),
        ((1, 1023, 512), torch.float32),
        ((1, 1024, 512), torch.float64),
    ):
        with pytest.raises(
            (TypeError, ValueError), match="B, 1024, 512|at least one|float32"
        ):
            score_patch_embeddings(
                torch.empty(shape, dtype=dtype), torch.zeros(1, 512), bank_chunk_size=1
            )
