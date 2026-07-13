from dataclasses import FrozenInstanceError, astuple
from pathlib import Path

import pytest

from inspectrt.data import discover_mvtec_samples


def _touch(root: Path, relpath: str) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    path.touch()


def _make_layout(root: Path) -> Path:
    category_root = root / "bottle"
    (category_root / "train" / "good").mkdir(parents=True)
    (category_root / "test" / "good").mkdir(parents=True)
    return category_root


def test_discovers_canonical_samples_in_deterministic_order(tmp_path: Path) -> None:
    _make_layout(tmp_path)
    for relpath in (
        "bottle/train/good/010.png",
        "bottle/test/good/020.png",
        "bottle/test/crack/003.png",
        "bottle/ground_truth/crack/003_mask.png",
        "bottle/test/broken_large/011.png",
        "bottle/ground_truth/broken_large/011_mask.png",
        "bottle/train/good/002.png",
        "bottle/train/good/ignored.jpg",
        "bottle/test/good/ignored.PNG",
        "bottle/test/good/nested/999.png",
        "bottle/test/notes.txt",
    ):
        _touch(tmp_path, relpath)

    samples = discover_mvtec_samples(tmp_path, "bottle")

    expected_ids = [
        "mvtec_ad/bottle/test/broken_large/011.png",
        "mvtec_ad/bottle/test/crack/003.png",
        "mvtec_ad/bottle/test/good/020.png",
        "mvtec_ad/bottle/train/good/002.png",
        "mvtec_ad/bottle/train/good/010.png",
    ]
    assert [sample.sample_id for sample in samples] == expected_ids
    assert [sample.image_relpath for sample in samples] == [
        sample_id.removeprefix("mvtec_ad/") for sample_id in expected_ids
    ]
    assert [astuple(sample)[1:5] for sample in samples] == [
        ("bottle", "test", "broken_large", True),
        ("bottle", "test", "crack", True),
        ("bottle", "test", "good", False),
        ("bottle", "train", "good", False),
        ("bottle", "train", "good", False),
    ]
    assert [sample.mask_relpath for sample in samples] == [
        "bottle/ground_truth/broken_large/011_mask.png",
        "bottle/ground_truth/crack/003_mask.png",
        None,
        None,
        None,
    ]
    with pytest.raises(FrozenInstanceError):
        samples[0].category = "cable"


def test_pairs_anomalous_masks_by_image_stem(tmp_path: Path) -> None:
    _make_layout(tmp_path)
    for relpath in (
        "bottle/test/scratch/001.png",
        "bottle/test/scratch/002.png",
        "bottle/ground_truth/scratch/000_mask.png",
        "bottle/ground_truth/scratch/001_mask.png",
        "bottle/ground_truth/scratch/002_mask.png",
    ):
        _touch(tmp_path, relpath)

    samples = discover_mvtec_samples(tmp_path, "bottle")

    assert [(sample.image_relpath, sample.mask_relpath) for sample in samples] == [
        ("bottle/test/scratch/001.png", "bottle/ground_truth/scratch/001_mask.png"),
        ("bottle/test/scratch/002.png", "bottle/ground_truth/scratch/002_mask.png"),
    ]


def test_rejects_anomalous_image_without_exact_mask(tmp_path: Path) -> None:
    _make_layout(tmp_path)
    _touch(tmp_path, "bottle/test/scratch/001.png")
    _touch(tmp_path, "bottle/ground_truth/scratch/999_mask.png")

    with pytest.raises(FileNotFoundError, match="001_mask[.]png"):
        discover_mvtec_samples(tmp_path, "bottle")


@pytest.mark.parametrize("missing", ["train/good", "test"])
def test_rejects_incomplete_category_layout(tmp_path: Path, missing: str) -> None:
    missing_path = _make_layout(tmp_path) / missing
    if missing == "test":
        (missing_path / "good").rmdir()
    missing_path.rmdir()

    with pytest.raises(FileNotFoundError, match="does not exist"):
        discover_mvtec_samples(tmp_path, "bottle")


def test_rejects_missing_or_non_directory_roots(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset root"):
        discover_mvtec_samples(tmp_path / "missing", "bottle")
    dataset_file = tmp_path / "dataset.zip"
    dataset_file.touch()
    with pytest.raises(NotADirectoryError, match="Dataset root"):
        discover_mvtec_samples(dataset_file, "bottle")

    with pytest.raises(FileNotFoundError, match="Category"):
        discover_mvtec_samples(tmp_path, "bottle")
    (tmp_path / "bottle").touch()
    with pytest.raises(NotADirectoryError, match="Category"):
        discover_mvtec_samples(tmp_path, "bottle")


@pytest.mark.parametrize(
    "category", ["", ".", "..", "bottle/test", "../bottle", "bottle\\test"]
)
def test_rejects_invalid_category_paths(tmp_path: Path, category: str) -> None:
    with pytest.raises(ValueError, match="category"):
        discover_mvtec_samples(tmp_path, category)


@pytest.mark.parametrize("component", ["ground_truth", "ground_truth/crack"])
def test_rejects_known_component_with_wrong_type(
    tmp_path: Path, component: str
) -> None:
    category_root = _make_layout(tmp_path)
    (category_root / "test" / "crack").mkdir()
    component_path = category_root / component
    if component_path.is_dir():
        component_path.rmdir()
    component_path.parent.mkdir(parents=True, exist_ok=True)
    component_path.touch()

    with pytest.raises(NotADirectoryError, match="directory"):
        discover_mvtec_samples(tmp_path, "bottle")
