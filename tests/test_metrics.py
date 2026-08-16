import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from inspectrt.metrics import compute_threshold_free_metrics


def _valid_inputs() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor((0, 1, 0, 1), dtype=torch.uint8)
    scores = torch.tensor((-4.0, -3.0, -2.0, -1.0))
    masks = torch.zeros(4, 256, 256, dtype=torch.uint8)
    maps = torch.zeros(4, 256, 256)
    masks[1, 0, 0] = 1
    maps[1, 0, 0] = 1
    return labels, scores, masks, maps


def test_metrics_use_raw_scores_and_one_global_pixel_population() -> None:
    labels = torch.tensor((0, 1), dtype=torch.uint8)
    scores = torch.tensor((-4.0, -1.0))
    masks = torch.zeros(2, 256, 256, dtype=torch.uint8)
    masks[1, 0, 0] = 1
    maps = torch.empty(2, 256, 256)
    maps[0].fill_(0.8)
    maps[1].fill_(0.1)
    maps[1, 0, 0] = 0.5
    originals = tuple(value.clone() for value in (labels, scores, masks, maps))

    result = compute_threshold_free_metrics(labels, scores, masks, maps)

    assert result.image_auroc == pytest.approx(roc_auc_score(labels, scores))
    assert result.image_average_precision == pytest.approx(
        average_precision_score(labels, scores)
    )
    assert result.pixel_auroc == pytest.approx(
        roc_auc_score(masks.reshape(-1), maps.reshape(-1))
    )
    assert result.pixel_auroc < roc_auc_score(masks[1].reshape(-1), maps[1].reshape(-1))
    assert all(
        isinstance(value, float)
        for value in (
            result.image_auroc,
            result.image_average_precision,
            result.pixel_auroc,
        )
    )
    for value, original in zip((labels, scores, masks, maps), originals, strict=True):
        assert torch.equal(value, original)


def test_accepts_binary_targets_and_rejects_tensor_contract_violations() -> None:
    labels, scores, masks, maps = _valid_inputs()
    compute_threshold_free_metrics(labels.bool(), scores, masks.bool(), maps)
    compute_threshold_free_metrics(labels.long(), scores, masks.int(), maps)

    cases = (
        (0, torch.empty(4, 1, dtype=torch.uint8), ValueError, "shape"),
        (1, torch.empty(3), ValueError, "counts"),
        (2, torch.empty(4, 255, 256, dtype=torch.uint8), ValueError, "shape"),
        (3, torch.empty(4, 256, 256, dtype=torch.float64), TypeError, "float32"),
        (0, torch.full((4,), 2, dtype=torch.int64), ValueError, "only 0 and 1"),
    )
    for index, replacement, error, message in cases:
        inputs = list(_valid_inputs())
        inputs[index] = replacement
        with pytest.raises(error, match=message):
            compute_threshold_free_metrics(*inputs)


def test_rejects_undefined_and_nonfinite_metrics() -> None:
    for scenario in ("image-class", "pixel-class", "score-nan", "map-inf"):
        labels, scores, masks, maps = _valid_inputs()
        if scenario == "image-class":
            labels.zero_()
            message = "only normal"
        elif scenario == "pixel-class":
            masks.zero_()
            message = "only background"
        elif scenario == "score-nan":
            scores[0] = torch.nan
            message = "finite"
        else:
            maps[0, 0, 0] = torch.inf
            message = "finite"
        with pytest.raises(ValueError, match=message) as raised:
            compute_threshold_free_metrics(labels, scores, masks, maps)
        assert raised.value, scenario
