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


def _decoded(image: torch.Tensor) -> torch.Tensor:
    return image * _STD + _MEAN


def test_applies_the_fixed_rgb_profile_and_zero_mask(tmp_path: Path) -> None:
    rgb_path = tmp_path / "rgb.png"
    gray_path = tmp_path / "gray.png"
    _save(Image.new("RGB", (7, 3), (0, 128, 255)), rgb_path)
    _save(Image.new("L", (5, 2), 64), gray_path)

    rgb = preprocess_image(rgb_path)
    gray = preprocess_image(gray_path)

    assert PREPROCESSING_PROFILE == "inspectrt_resize256_v1"
    assert rgb.image.shape == gray.image.shape == (3, 256, 256)
    assert rgb.image.dtype == gray.image.dtype == torch.float32
    assert rgb.image.is_contiguous() and gray.image.is_contiguous()
    expected_rgb = torch.tensor(
        ((0.0 - 0.485) / 0.229, (128 / 255 - 0.456) / 0.224, (1.0 - 0.406) / 0.225)
    )
    torch.testing.assert_close(rgb.image[:, 0, 0], expected_rgb, atol=1e-6, rtol=0)
    expected_gray = (torch.full((3,), 64 / 255) - _MEAN[:, 0, 0]) / _STD[:, 0, 0]
    torch.testing.assert_close(
        gray.image[:, 100, 100], expected_gray, atol=1e-6, rtol=0
    )
    assert (rgb.original_height, rgb.original_width, rgb.original_mode) == (3, 7, "RGB")
    assert (gray.original_height, gray.original_width, gray.original_mode) == (
        2,
        5,
        "L",
    )
    assert rgb.evaluation_mask.shape == (256, 256)
    assert rgb.evaluation_mask.dtype == torch.uint8
    assert rgb.evaluation_mask.count_nonzero().item() == 0


def test_resizes_image_and_binary_mask_with_exact_aligned_geometry(
    tmp_path: Path,
) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    image = Image.new("RGB", (8, 4))
    image.paste((255, 255, 255), (2, 1, 6, 3))
    mask = Image.new("RGB", (8, 4))
    mask.paste((0, 2, 0), (2, 1, 6, 3))
    source_mask = F.pil_to_tensor(mask).to(torch.float32)
    with F.resize(
        image, [256, 256], interpolation=InterpolationMode.BILINEAR, antialias=True
    ) as resized_image:
        expected_image = F.pil_to_tensor(resized_image).to(torch.float32).div(255)
    expected_mask = (
        F.resize(
            source_mask,
            [256, 256],
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=False,
        )
        .ne(0)
        .any(dim=0)
        .to(torch.uint8)
    )
    _save(image, image_path)
    _save(mask, mask_path)

    result = preprocess_image(image_path, mask_path)

    torch.testing.assert_close(
        _decoded(result.image), expected_image, atol=1e-6, rtol=0
    )
    assert torch.equal(result.evaluation_mask, expected_mask)
    assert set(result.evaluation_mask.unique().tolist()) == {0, 1}
    image_points = _decoded(result.image)[0].ge(0.5).nonzero()
    mask_points = result.evaluation_mask.nonzero()
    assert torch.equal(image_points.amin(dim=0), mask_points.amin(dim=0))
    assert torch.equal(image_points.amax(dim=0), mask_points.amax(dim=0))

    tiny_image = tmp_path / "tiny-image.png"
    tiny_mask = tmp_path / "tiny-mask.png"
    values = (0, 2, 0, 4, 0, 8)
    source = torch.tensor(values, dtype=torch.float32).reshape(1, 2, 3)
    mask = Image.new("L", (3, 2))
    mask.putdata(values)
    _save(Image.new("RGB", (3, 2)), tiny_image)
    _save(mask, tiny_mask)
    nearest_exact = (
        F.resize(
            source,
            [256, 256],
            interpolation=InterpolationMode.NEAREST_EXACT,
            antialias=False,
        )
        .squeeze(0)
        .ne(0)
        .to(torch.uint8)
    )
    nearest = (
        F.resize(
            source,
            [256, 256],
            interpolation=InterpolationMode.NEAREST,
            antialias=False,
        )
        .squeeze(0)
        .ne(0)
        .to(torch.uint8)
    )
    assert torch.equal(
        preprocess_image(tiny_image, tiny_mask).evaluation_mask, nearest_exact
    )
    assert not torch.equal(nearest_exact, nearest)


def test_rejects_missing_undecodable_and_misaligned_inputs(tmp_path: Path) -> None:
    image_path = tmp_path / "image.png"
    mask_path = tmp_path / "mask.png"
    _save(Image.new("RGB", (5, 4)), image_path)
    _save(Image.new("L", (4, 5)), mask_path)
    with pytest.raises(ValueError, match="dimensions differ"):
        preprocess_image(image_path, mask_path)

    with pytest.raises(FileNotFoundError, match="Image"):
        preprocess_image(tmp_path / "missing.png")
    invalid = tmp_path / "invalid.png"
    invalid.write_bytes(b"not an image")
    with pytest.raises(UnidentifiedImageError):
        preprocess_image(invalid)
    directory = tmp_path / "directory"
    directory.mkdir()
    with pytest.raises(IsADirectoryError, match="Mask"):
        preprocess_image(image_path, directory)
