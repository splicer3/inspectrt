"""MVTec AD sample discovery."""

from dataclasses import dataclass
from pathlib import Path, PurePosixPath


@dataclass(frozen=True, slots=True)
class MvtecSample:
    sample_id: str
    category: str
    split: str
    defect_type: str
    is_anomalous: bool
    image_relpath: str
    mask_relpath: str | None


def discover_mvtec_samples(dataset_root: Path, category: str) -> list[MvtecSample]:
    """Discover one MVTec AD category without reading image contents."""
    _validate_category(category)
    _require_directory(dataset_root, "Dataset root")

    category_root = dataset_root / category
    _require_directory(category_root, f"Category {category!r}")
    train_root = category_root / "train"
    _require_directory(train_root, "Training directory")
    train_good = train_root / "good"
    _require_directory(train_good, "Training good directory")
    test_root = category_root / "test"
    _require_directory(test_root, "Test directory")
    ground_truth = category_root / "ground_truth"
    if ground_truth.exists() or ground_truth.is_symlink():
        _require_directory(ground_truth, "Ground-truth directory")

    sources = [(train_good, "train", "good")]
    test_good = test_root / "good"
    if test_good.exists() or test_good.is_symlink():
        _require_directory(test_good, "Test good directory")
        sources.append((test_good, "test", "good"))
    for path in sorted(test_root.iterdir(), key=lambda item: item.name):
        if path.name == "good" or not path.is_dir():
            continue
        mask_directory = ground_truth / path.name
        if mask_directory.exists() or mask_directory.is_symlink():
            _require_directory(mask_directory, f"Mask directory for {path.name!r}")
        sources.append((path, "test", path.name))

    samples = []
    for image_directory, split, defect_type in sources:
        is_anomalous = split == "test" and defect_type != "good"
        for image_path in _png_files(image_directory):
            image_relpath = _posix(category, split, defect_type, image_path.name)
            mask_relpath = None
            if is_anomalous:
                mask_relpath = _mask_relpath(
                    category_root, category, defect_type, image_path
                )
            samples.append(
                MvtecSample(
                    sample_id=_posix("mvtec_ad", image_relpath),
                    category=category,
                    split=split,
                    defect_type=defect_type,
                    is_anomalous=is_anomalous,
                    image_relpath=image_relpath,
                    mask_relpath=mask_relpath,
                )
            )
    return sorted(samples, key=lambda sample: sample.sample_id)


def _mask_relpath(
    category_root: Path, category: str, defect_type: str, image_path: Path
) -> str:
    ground_truth = category_root / "ground_truth"
    mask_directory = ground_truth / defect_type
    mask_path = mask_directory / f"{image_path.stem}_mask.png"
    image_relpath = _posix(category, "test", defect_type, image_path.name)
    mask_relpath = _posix(category, "ground_truth", defect_type, mask_path.name)
    if not mask_path.exists():
        raise FileNotFoundError(
            f"Missing mask for anomalous image {image_relpath!r}: "
            f"expected {mask_relpath!r}"
        )
    if not mask_path.is_file():
        raise IsADirectoryError(f"Expected mask is not a file: {mask_relpath!r}")
    return mask_relpath


def _png_files(directory: Path) -> list[Path]:
    images = []
    for path in sorted(directory.iterdir(), key=lambda item: item.name):
        if path.suffix != ".png":
            continue
        if not path.is_file():
            raise IsADirectoryError(f"Expected PNG image is not a file: {path}")
        images.append(path)
    return images


def _validate_category(category: str) -> None:
    if not category or category in {".", ".."} or "/" in category or "\\" in category:
        raise ValueError(
            f"Invalid category {category!r}: expected a non-empty single path component"
        )


def _require_directory(path: Path, description: str) -> None:
    if not path.exists() and not path.is_symlink():
        raise FileNotFoundError(f"{description} does not exist: {path}")
    if not path.is_dir():
        raise NotADirectoryError(f"{description} is not a directory: {path}")


def _posix(*parts: str) -> str:
    return PurePosixPath(*parts).as_posix()
