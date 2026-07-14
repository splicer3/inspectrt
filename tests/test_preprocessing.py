from dataclasses import FrozenInstanceError
from pathlib import Path

import pytest
import torch
from PIL import Image, UnidentifiedImageError
from torchvision.transforms import InterpolationMode
from torchvision.transforms import functional as F

from inspectrt.preprocessing import PREPROCESSING_PROFILE, preprocess_image

_MEAN = torch.tensor((0.485, 0.456, 0.406))[:, None, None]
_STD = torch.tensor((0.229, 0.224, 0.225))[:, None, None]


def _save(image: Image.Image, path: Path) -> None:
    image.save(path)
    image.close()


def _decoded_values(image: torch.Tensor) -> torch.Tensor:
    return image * _STD + _MEAN


def _pattern_pixel(x: int, y: int) -> tuple[int, int, int]:
    return 255 * (x < 128), 255 * ((x + y) % 3 == 0), 255 * (y >= 64)


def _resize_mask(
    source: torch.Tensor, interpolation: InterpolationMode
) -> torch.Tensor:
    resized = F.resize(source, [256, 256], interpolation=interpolation, antialias=False)
    return resized.squeeze(0).ne(0).to(torch.uint8)


def test_preprocesses_rgb_values_and_records_source_metadata(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    _save(Image.new("RGB", (7, 3), (0, 128, 255)), image_path)

    result = preprocess_image(image_path)

    assert PREPROCESSING_PROFILE == "inspectrt_resize256_v1"
    assert result.image.shape == (3, 256, 256)
    assert result.image.dtype == torch.float32
    assert result.image.device.type == "cpu"
    assert result.image.is_contiguous()
    metadata = result.original_height, result.original_width, result.original_mode
    assert metadata == (3, 7, "RGB")
    expected = torch.tensor(
        ((0.0 - 0.485) / 0.229, (128 / 255 - 0.456) / 0.224, (1.0 - 0.406) / 0.225)
    )
    assert torch.allclose(result.image[:, 0, 0], expected, atol=1e-6, rtol=0)
    assert result.evaluation_mask.shape == (256, 256)
    assert result.evaluation_mask.dtype == torch.uint8
    assert result.evaluation_mask.device.type == "cpu"
    assert result.evaluation_mask.count_nonzero().item() == 0
    with pytest.raises(FrozenInstanceError):
        result.original_width = 1


def test_converts_grayscale_to_rgb_before_normalization(tmp_path: Path) -> None:
    image_path = tmp_path / "gray.png"
    _save(Image.new("L", (5, 2), 64), image_path)

    result = preprocess_image(image_path)

    expected = (torch.full((3,), 64 / 255) - _MEAN[:, 0, 0]) / _STD[:, 0, 0]
    assert torch.allclose(result.image[:, 100, 100], expected, atol=1e-6, rtol=0)
    metadata = result.original_height, result.original_width, result.original_mode
    assert metadata == (2, 5, "L")


def test_resizes_directly_with_bilinear_interpolation(tmp_path: Path) -> None:
    image_path = tmp_path / "wide.png"
    image = Image.new("RGB", (512, 128))
    image.putdata([_pattern_pixel(x, y) for y in range(128) for x in range(512)])
    source = F.pil_to_tensor(image).to(torch.float32).div(255)
    with F.resize(
        image, [256, 256], InterpolationMode.BILINEAR, antialias=True
    ) as reference_image:
        expected = F.pil_to_tensor(reference_image).to(torch.float32).div(255)
    without_antialias = F.resize(
        source, [256, 256], InterpolationMode.BILINEAR, antialias=False
    )
    _save(image, image_path)

    decoded = _decoded_values(preprocess_image(image_path).image)

    assert torch.allclose(decoded, expected, atol=1e-6, rtol=0)
    assert not torch.allclose(expected, without_antialias, atol=1e-3, rtol=0)
    assert (decoded[0, :, 32].mean() - decoded[0, :, 224].mean()).item() > 0.99
    assert (decoded[2, 224, :].mean() - decoded[2, 32, :].mean()).item() > 0.99


def test_resizes_mask_with_nearest_exact_and_binarizes(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    _save(Image.new("RGB", (3, 2)), image_path)
    values = (0, 2, 0, 4, 0, 8)
    mask = Image.new("L", (3, 2))
    mask.putdata(values)
    _save(mask, mask_path)

    result = preprocess_image(image_path, mask_path)
    source = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 3)
    expected = _resize_mask(source, InterpolationMode.NEAREST_EXACT)
    legacy = _resize_mask(source, InterpolationMode.NEAREST)

    assert torch.equal(result.evaluation_mask, expected)
    assert not torch.equal(expected, legacy)
    assert set(result.evaluation_mask.unique().tolist()) == {0, 1}


def test_keeps_image_region_and_mask_aligned(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    region = (2, 1, 6, 3)
    image = Image.new("RGB", (8, 4))
    image.paste((255, 255, 255), region)
    mask = Image.new("L", (8, 4))
    mask.paste(255, region)
    _save(image, image_path)
    _save(mask, mask_path)

    result = preprocess_image(image_path, mask_path)
    image_region = _decoded_values(result.image)[0].ge(0.5)
    image_points = image_region.nonzero()
    mask_points = result.evaluation_mask.nonzero()

    assert torch.equal(image_points.amin(dim=0), mask_points.amin(dim=0))
    assert torch.equal(image_points.amax(dim=0), mask_points.amax(dim=0))
    assert torch.equal(
        image_points.float().mean(dim=0), mask_points.float().mean(dim=0)
    )


def test_rejects_mismatched_source_dimensions(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    _save(Image.new("RGB", (5, 4)), image_path)
    _save(Image.new("L", (4, 5)), mask_path)

    with pytest.raises(ValueError, match="dimensions differ"):
        preprocess_image(image_path, mask_path)


@pytest.mark.parametrize("is_mask", [False, True], ids=["image", "mask"])
def test_rejects_invalid_image_or_mask_paths(tmp_path: Path, is_mask: bool) -> None:
    valid_image = tmp_path / "valid.png"
    _save(Image.new("RGB", (2, 2)), valid_image)
    description = "Mask" if is_mask else "Image"

    def call(path: Path) -> object:
        return (
            preprocess_image(valid_image, path) if is_mask else preprocess_image(path)
        )

    missing = tmp_path / "missing.png"
    with pytest.raises(FileNotFoundError, match=description):
        call(missing)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(IsADirectoryError, match=description):
        call(directory)
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        call(invalid)
