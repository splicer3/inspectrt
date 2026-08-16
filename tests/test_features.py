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

    def forward(self, images: Tensor) -> dict[str, Tensor]:
        outputs = self.extractor(images)
        self.feature_map = outputs["layer2"]
        return outputs


class _NeverCalledExtractor(nn.Module):
    def forward(self, images: Tensor) -> Tensor:
        raise AssertionError("invalid input reached the extractor")


@pytest.fixture(scope="module")
def extractor() -> nn.Module:
    with torch.random.fork_rng():
        torch.manual_seed(0)
        return build_resnet50_layer2_extractor(weights=None)


def test_builds_frozen_layer2_extractor_and_canonical_patches(
    extractor: nn.Module,
) -> None:
    assert not extractor.training
    assert all(not parameter.requires_grad for parameter in extractor.parameters())
    recording = _RecordingExtractor(extractor)
    recording.train()
    images = torch.zeros(1, 3, 256, 256, requires_grad=True)

    patches = extract_patch_embeddings(recording, images)

    assert recording.feature_map is not None
    assert recording.feature_map.shape == (1, 512, 32, 32)
    assert patches.shape == (1, 1024, 512)
    assert patches.dtype == torch.float32
    assert patches.is_contiguous() and not patches.requires_grad
    assert not recording.training


def test_pool_includes_padding_and_materializes_batches_row_major() -> None:
    grid = torch.arange(32 * 32, dtype=torch.float32).reshape(1, 1, 32, 32)
    feature_map = grid.expand(1, 512, -1, -1).contiguous()
    batch_map = torch.cat((feature_map, torch.ones_like(feature_map)))
    images = torch.zeros(2, 3, 256, 256)

    patches = extract_patch_embeddings(_StaticExtractor(batch_map), images)

    assert patches.shape == (2, 1024, 512)
    torch.testing.assert_close(patches[0, 34], torch.full((512,), 34.0))
    torch.testing.assert_close(patches[0, 65], torch.full((512,), 65.0))
    expected_padding = torch.tensor((4 / 9, 6 / 9, 1.0, 4 / 9))
    torch.testing.assert_close(patches[1, (0, 1, 33, 1023), 0], expected_padding)


def test_rejects_invalid_image_batch_and_layer2_contracts() -> None:
    cases = (
        (torch.empty(3, 256, 256), ValueError, "rank 4"),
        (torch.empty(0, 3, 256, 256), ValueError, "at least one"),
        (torch.empty(1, 1, 256, 256), ValueError, "3 image channels"),
        (torch.empty(1, 3, 255, 256), ValueError, "256 x 256"),
        (torch.empty(1, 3, 256, 256, dtype=torch.float64), TypeError, "float32"),
    )
    for images, error, message in cases:
        with pytest.raises(error, match=message):
            extract_patch_embeddings(_NeverCalledExtractor(), images)

    images = torch.zeros(1, 3, 256, 256)
    malformed = _StaticExtractor(torch.zeros(1, 256, 32, 32))
    with pytest.raises(RuntimeError, match="512, 32, 32"):
        extract_patch_embeddings(malformed, images)
