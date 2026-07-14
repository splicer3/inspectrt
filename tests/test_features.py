import pytest
import torch
from torch import Tensor, nn

from inspectrt.features import (
    build_resnet50_layer2_extractor,
    extract_patch_embeddings,
)


class _StaticExtractor(nn.Module):
    def __init__(self, feature_map: Tensor) -> None:
        super().__init__()
        self.register_buffer("feature_map", feature_map)

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        return {"layer2": self.feature_map}


class _RecordingExtractor(nn.Module):
    def __init__(self, extractor: nn.Module) -> None:
        super().__init__()
        self.extractor = extractor
        self.feature_map: Tensor | None = None
        self.output_keys: tuple[str, ...] = ()

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        outputs = self.extractor(images)
        self.feature_map = outputs["layer2"]
        self.output_keys = tuple(outputs)
        return outputs


class _NeverCalledExtractor(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        raise AssertionError("invalid input reached the extractor")


@pytest.fixture(scope="module")
def extractor() -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return build_resnet50_layer2_extractor(weights=None)


def test_builds_frozen_evaluation_extractor(extractor: nn.Module) -> None:
    parameters = list(extractor.parameters())

    assert not extractor.training
    assert all(not module.training for module in extractor.modules())
    assert parameters
    assert all(not parameter.requires_grad for parameter in parameters)


def test_extracts_raw_layer2_and_contiguous_patch_shapes(
    extractor: nn.Module,
) -> None:
    recording = _RecordingExtractor(extractor)
    recording.train()
    images = torch.zeros(1, 3, 256, 256, dtype=torch.float32, requires_grad=True)

    patches = extract_patch_embeddings(recording, images)

    assert recording.output_keys == ("layer2",)
    assert recording.feature_map is not None
    assert recording.feature_map.shape == (1, 512, 32, 32)
    assert patches.shape == (1, 1024, 512)
    assert patches.dtype == torch.float32
    assert patches.device.type == "cpu"
    assert patches.is_contiguous()
    assert not patches.requires_grad
    assert not recording.training
    assert all(not module.training for module in recording.modules())


def test_preserves_batches_larger_than_one() -> None:
    images = torch.zeros(2, 3, 256, 256, dtype=torch.float32)
    feature_map = torch.stack(
        (
            torch.zeros(512, 32, 32, dtype=torch.float32),
            torch.ones(512, 32, 32, dtype=torch.float32),
        )
    )

    patches = extract_patch_embeddings(_StaticExtractor(feature_map), images)

    assert patches.shape == (2, 1024, 512)
    assert patches[0].count_nonzero().item() == 0
    torch.testing.assert_close(patches[1, 33], torch.ones(512))


def test_local_average_preserves_grid_and_includes_padded_zeros() -> None:
    feature_map = torch.ones(1, 512, 32, 32, dtype=torch.float32)
    images = torch.zeros(1, 3, 256, 256, dtype=torch.float32)

    patches = extract_patch_embeddings(_StaticExtractor(feature_map), images)

    assert patches.shape[1:] == (32 * 32, 512)
    expected = torch.tensor((4 / 9, 6 / 9, 1.0, 4 / 9))
    torch.testing.assert_close(patches[0, (0, 1, 33, 1023), 0], expected)


def test_flattens_spatial_positions_in_row_major_order() -> None:
    grid = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    feature_map = grid.expand(1, 512, -1, -1).contiguous()
    images = torch.zeros(1, 3, 256, 256, dtype=torch.float32)

    patches = extract_patch_embeddings(_StaticExtractor(feature_map), images)

    torch.testing.assert_close(patches[0, 34], torch.full((512,), 34.0))
    torch.testing.assert_close(patches[0, 65], torch.full((512,), 65.0))


@pytest.mark.parametrize(
    ("shape", "dtype", "error", "message"),
    [
        ((3, 256, 256), torch.float32, ValueError, "rank 4"),
        ((1, 3, 256, 256, 1), torch.float32, ValueError, "rank 4"),
        ((0, 3, 256, 256), torch.float32, ValueError, "at least one"),
        ((1, 1, 256, 256), torch.float32, ValueError, "3 image channels"),
        ((1, 3, 255, 256), torch.float32, ValueError, "256 x 256"),
        ((1, 3, 256, 256), torch.float64, TypeError, "torch.float32"),
    ],
)
def test_rejects_noncanonical_image_batches(
    shape: tuple[int, ...],
    dtype: torch.dtype,
    error: type[Exception],
    message: str,
) -> None:
    images = torch.empty(shape, dtype=dtype)

    with pytest.raises(error, match=message):
        extract_patch_embeddings(_NeverCalledExtractor(), images)


def test_rejects_unexpected_layer2_shape() -> None:
    images = torch.zeros(1, 3, 256, 256, dtype=torch.float32)
    feature_map = torch.zeros(1, 256, 32, 32, dtype=torch.float32)

    with pytest.raises(RuntimeError, match="512, 32, 32"):
        extract_patch_embeddings(_StaticExtractor(feature_map), images)
