import pytest
import torch
from sklearn.metrics import average_precision_score, roc_auc_score

from inspectrt.metrics import compute_threshold_free_metrics


def _valid_inputs(
    label_dtype: torch.dtype = torch.bool,
    mask_dtype: torch.dtype = torch.uint8,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    labels = torch.tensor((0, 1, 0, 1), dtype=label_dtype)
    scores = torch.tensor((-4.0, 2.0, -1.0, 3.0))
    masks = torch.zeros(4, 256, 256, dtype=mask_dtype)
    maps = torch.zeros(4, 256, 256)
    masks[1, 0, 0] = 1
    maps[1, 0, 0] = 1
    return labels, scores, masks, maps


def test_perfect_ranking_returns_python_floats_without_mutating_inputs() -> None:
    inputs = _valid_inputs()
    inputs[1].requires_grad_()
    inputs[3].requires_grad_()
    originals = tuple(tensor.detach().clone() for tensor in inputs)

    result = compute_threshold_free_metrics(*inputs)
    values = result.image_auroc, result.image_average_precision, result.pixel_auroc

    assert values[:2] == (1.0, 1.0)
    assert all(isinstance(value, float) for value in values)
    for tensor, original in zip(inputs, originals, strict=True):
        assert torch.equal(tensor.detach(), original)


def test_imperfect_negative_scores_match_sklearn_without_thresholding() -> None:
    labels, _, masks, maps = _valid_inputs()
    scores = torch.tensor((-4.0, -3.0, -2.0, -1.0))

    result = compute_threshold_free_metrics(labels, scores, masks, maps)
    expected_auroc = roc_auc_score(labels.numpy(), scores.numpy())
    expected_ap = average_precision_score(labels.numpy(), scores.numpy())

    assert expected_auroc != roc_auc_score(labels.numpy(), scores.gt(0).numpy())
    assert result.image_auroc == pytest.approx(expected_auroc)
    assert result.image_average_precision == pytest.approx(expected_ap)


def test_pixel_auroc_uses_raw_global_flattening_and_keeps_good_pixels() -> None:
    labels = torch.tensor((0, 1), dtype=torch.uint8)
    scores = torch.tensor((0.0, 1.0))
    masks = torch.zeros(2, 256, 256, dtype=torch.uint8)
    masks[:, 0, 0] = 1
    maps = torch.empty(2, 256, 256)
    maps[0].fill_(0.8)
    maps[1].fill_(0.1)
    maps[0, 0, 0] = 0.9
    maps[1, 0, 0] = 0.2

    result = compute_threshold_free_metrics(labels, scores, masks, maps)
    expected = roc_auc_score(masks.reshape(-1).numpy(), maps.reshape(-1).numpy())

    assert result.pixel_auroc == pytest.approx(expected)
    assert result.pixel_auroc < 1.0  # Each per-image AUROC is 1.0.

    masks.zero_()
    maps.zero_()
    masks[1, 0, 0] = 1
    maps[0].fill_(0.8)
    maps[1, 0, 0] = 0.5

    result = compute_threshold_free_metrics(labels, scores, masks, maps)
    expected = roc_auc_score(masks.reshape(-1).numpy(), maps.reshape(-1).numpy())
    anomaly_only = roc_auc_score(masks[1].reshape(-1), maps[1].reshape(-1))

    assert result.pixel_auroc == pytest.approx(expected)
    assert result.pixel_auroc < anomaly_only == 1.0


@pytest.mark.parametrize(
    "dtype", [torch.bool, torch.uint8, torch.int16, torch.int64, torch.uint32]
)
def test_accepts_boolean_and_integer_binary_targets(dtype: torch.dtype) -> None:
    compute_threshold_free_metrics(*_valid_inputs(dtype, dtype))


def test_rejects_unsupported_target_score_and_map_dtypes() -> None:
    for index, dtype, message in (
        (0, torch.float64, "Image labels.*boolean or integer"),
        (0, torch.bits8, "Image labels.*boolean or integer"),
        (1, torch.float64, "Image scores.*torch.float32"),
        (2, torch.float64, "Pixel masks.*boolean or integer"),
        (3, torch.float64, "Anomaly maps.*torch.float32"),
    ):
        inputs = list(_valid_inputs())
        inputs[index] = torch.empty(inputs[index].shape, dtype=dtype)
        with pytest.raises(TypeError, match=message):
            compute_threshold_free_metrics(*inputs)


def test_rejects_nonbinary_labels_and_masks() -> None:
    for index, location, value, message in (
        (0, (0,), -1, "Image labels.*only 0 and 1"),
        (2, (0, 0, 0), 2, "Pixel masks.*only 0 and 1"),
    ):
        inputs = list(_valid_inputs(torch.int64, torch.int16))
        inputs[index][location] = value
        with pytest.raises(ValueError, match=message):
            compute_threshold_free_metrics(*inputs)


@pytest.mark.parametrize(
    ("target", "value", "message"),
    [
        ("image", 0, "Image AUROC.*only normal"),
        ("image", 1, "Image AUROC.*only anomalous"),
        ("pixel", 0, "Pixel AUROC.*only background"),
        ("pixel", 1, "Pixel AUROC.*only foreground"),
    ],
)
def test_rejects_one_class_targets(target: str, value: int, message: str) -> None:
    labels, scores, masks, maps = _valid_inputs(torch.uint8)
    (labels if target == "image" else masks).fill_(value)

    with pytest.raises(ValueError, match=message):
        compute_threshold_free_metrics(labels, scores, masks, maps)


def test_rejects_wrong_shapes_empty_inputs_and_sample_count_mismatches() -> None:
    cases = (
        (0, torch.empty(4, 1, dtype=torch.uint8), "Image labels.*shape"),
        (0, torch.empty(0, dtype=torch.uint8), "Image labels.*at least one"),
        (1, torch.empty(4, 1), "Image scores.*shape"),
        (2, torch.empty(4, 256, dtype=torch.uint8), "Pixel masks.*shape"),
        (3, torch.empty(4, 256, 256, 1), "Anomaly maps.*shape"),
        (2, torch.empty(4, 255, 256, dtype=torch.uint8), "Pixel masks.*shape"),
        (3, torch.empty(4, 256, 255), "Anomaly maps.*shape"),
        (1, torch.empty(3), "Sample counts must match"),
    )
    for index, replacement, message in cases:
        inputs = list(_valid_inputs())
        inputs[index] = replacement
        with pytest.raises(ValueError, match=message):
            compute_threshold_free_metrics(*inputs)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
@pytest.mark.parametrize(
    ("index", "message"),
    [(1, "Image scores.*finite"), (3, "Anomaly maps.*finite")],
)
def test_rejects_nonfinite_scores_and_maps(
    index: int, message: str, value: float
) -> None:
    inputs = list(_valid_inputs())
    inputs[index].reshape(-1)[0] = value

    with pytest.raises(ValueError, match=message):
        compute_threshold_free_metrics(*inputs)
