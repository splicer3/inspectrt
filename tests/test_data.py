from dataclasses import astuple
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


def test_discovers_ordered_samples_and_pairs_exact_masks(tmp_path: Path) -> None:
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


def test_rejects_missing_mask_or_required_partition(tmp_path: Path) -> None:
    for scenario in ("mask", "train", "test"):
        root = tmp_path / scenario
        category = _make_layout(root)
        if scenario == "mask":
            _touch(root, "bottle/test/scratch/001.png")
            match = "001_mask[.]png"
        else:
            missing = category / ("train/good" if scenario == "train" else "test")
            if scenario == "test":
                (missing / "good").rmdir()
            missing.rmdir()
            match = "does not exist"
        with pytest.raises(FileNotFoundError, match=match) as raised:
            discover_mvtec_samples(root, "bottle")
        assert raised.value, scenario


def test_rejects_untrusted_roots_categories_and_component_types(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="Dataset root"):
        discover_mvtec_samples(tmp_path / "missing", "bottle")
    dataset_file = tmp_path / "dataset.zip"
    dataset_file.touch()
    with pytest.raises(NotADirectoryError, match="Dataset root"):
        discover_mvtec_samples(dataset_file, "bottle")

    for category in ("", "..", "bottle/test", "bottle\\test"):
        with pytest.raises(ValueError, match="category"):
            discover_mvtec_samples(tmp_path, category)

    category_root = _make_layout(tmp_path / "wrong-type")
    (category_root / "test" / "crack").mkdir()
    ground_truth = category_root / "ground_truth"
    ground_truth.touch()
    with pytest.raises(NotADirectoryError, match="directory"):
        discover_mvtec_samples(tmp_path / "wrong-type", "bottle")
