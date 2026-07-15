from dataclasses import replace
import json
import math
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

from PIL import Image
import pytest
import torch
from torch import Tensor, nn

import inspectrt.artifacts as artifacts
import inspectrt.benchmark as benchmark_module
import inspectrt.cli as cli
from inspectrt.artifacts import BaselineRunMetadata, persist_baseline_run
from inspectrt.benchmark import benchmark_mvtec_category
from inspectrt.evaluation import evaluate_mvtec_category

_ROOT = Path(__file__).resolve().parents[1]
_PROFILE = _ROOT / "configs" / "baseline.toml"
_CHUNK_SIZE = 4096
_STAGES = {
    "image_decode",
    "canonical_image_preprocessing",
    "host_to_device_transfer",
    "frozen_feature_extraction",
    "exact_chunked_retrieval",
    "anomaly_map_reconstruction",
}
_EVALUATION_FILES = {
    "anomaly_maps.pt",
    "memory_bank.pt",
    "metrics.json",
    "predictions.jsonl",
    "retrieval.pt",
    "run.json",
    "samples.jsonl",
}


class _ControlledExtractor(nn.Module):
    def forward(self, images: Tensor) -> dict[str, Tensor]:
        signal = images[:, :1, ::8, ::8]
        return {"layer2": signal.expand(-1, 512, -1, -1).contiguous()}


def _save(root: Path, relpath: str, image: Image.Image) -> None:
    path = root / relpath
    path.parent.mkdir(parents=True, exist_ok=True)
    with image:
        image.save(path)


def _make_dataset(root: Path) -> None:
    category = root / "bottle"
    _save(category, "train/good/010.png", Image.new("RGB", (8, 5), (64,) * 3))
    _save(category, "train/good/002.png", Image.new("L", (7, 4), 0))
    _save(category, "test/good/001.png", Image.new("RGB", (6, 3), 0))
    _save(category, "test/crack/001.png", Image.new("L", (5, 7), 255))
    mask = Image.new("L", (5, 7), 0)
    mask.paste(255, (1, 2, 4, 6))
    _save(category, "ground_truth/crack/001_mask.png", mask)


def _timing(stage_value: float) -> dict[str, float]:
    return {name: stage_value for name in (*_STAGES, "end_to_end")}


@pytest.fixture
def cpu_benchmark(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    _make_dataset(tmp_path)
    calls: list[dict[str, float]] = []

    def measure(*args: object, **kwargs: object) -> dict[str, float]:
        index = len(calls)
        value = 1000.0 if index < 2 else float(index - 1)
        result = _timing(value)
        calls.append(result)
        return result

    clock = iter((0, 2_000_000, 10_000_000, 13_000_000))
    monkeypatch.setattr(benchmark_module, "_measure_batch1", measure)
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", lambda: next(clock))
    measured, benchmark = benchmark_mvtec_category(
        tmp_path,
        "bottle",
        _ControlledExtractor(),
        device="cpu",
        bank_chunk_size=_CHUNK_SIZE,
        warmup_count=2,
        repeat_count=3,
        model_and_weight_load_ms=1.0,
        run_id="benchmark-run",
        created_at_utc="2026-07-15T12:00:00Z",
    )
    ordinary = evaluate_mvtec_category(
        tmp_path,
        "bottle",
        _ControlledExtractor(),
        device="cpu",
        bank_chunk_size=_CHUNK_SIZE,
    )
    return SimpleNamespace(
        dataset_root=tmp_path,
        measured=measured,
        ordinary=ordinary,
        benchmark=benchmark,
        timing_calls=calls,
    )


def _metadata(run_id: str) -> BaselineRunMetadata:
    return BaselineRunMetadata(
        run_id=run_id,
        created_at_utc="2026-07-15T12:00:00Z",
        dataset_root="/private/mvtec-ad",
        requested_device="cpu",
        bank_chunk_size=_CHUNK_SIZE,
        git_commit="c" * 40,
        git_dirty=True,
        uv_lock_sha256="a" * 64,
        python_version="3.11.15",
        platform_description="Linux-test",
        dependency_versions={"torch": "2.13.0"},
        determinism_flags={"seed": 0, "deterministic": True},
        weight_enum="ResNet50_Weights.IMAGENET1K_V2",
        weight_source_url="https://example.invalid/resnet50.pth",
        weight_file_sha256="b" * 64,
    )


def _assert_builtin_json(value: object) -> None:
    if type(value) is dict:
        assert all(type(key) is str for key in value)
        for item in value.values():  # type: ignore[union-attr]
            _assert_builtin_json(item)
        return
    if type(value) is list:
        for item in value:  # type: ignore[union-attr]
            _assert_builtin_json(item)
        return
    assert value is None or type(value) in {str, int, float, bool}
    if type(value) is float:
        assert math.isfinite(value)


def _console(*arguments: str) -> subprocess.CompletedProcess[str]:
    executable = shutil.which("inspectrt")
    assert executable is not None
    return subprocess.run(
        (executable, *arguments),
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )


def test_statistics_use_linear_percentiles_and_finite_builtins() -> None:
    statistics = benchmark_module._statistics([1.0, 2.0, 3.0, 4.0])
    assert statistics == {
        "count": 4,
        "maximum": 4.0,
        "mean": 2.5,
        "minimum": 1.0,
        "p50": 2.5,
        "p95": pytest.approx(3.85),
    }
    assert all(type(value) in {int, float} for value in statistics.values())
    with pytest.raises(ValueError, match="empty"):
        benchmark_module._statistics([])
    with pytest.raises(ValueError, match="finite"):
        benchmark_module._statistics([float("nan")])


@pytest.mark.parametrize("value", [0, -1, True, 1.0, "1"])
def test_benchmark_counts_must_be_positive_integers(value: object) -> None:
    with pytest.raises(ValueError, match="positive integer"):
        benchmark_module._positive_count(value, "repeat_count")


def test_cpu_benchmark_preserves_scientific_outputs_and_records_schema(
    cpu_benchmark: SimpleNamespace,
) -> None:
    measured = cpu_benchmark.measured
    ordinary = cpu_benchmark.ordinary
    benchmark = cpu_benchmark.benchmark

    assert measured.samples == ordinary.samples
    assert measured.test_samples == ordinary.test_samples
    assert measured.metrics == ordinary.metrics
    for name in (
        "memory_bank",
        "test_labels",
        "pixel_masks",
        "patch_distances",
        "nearest_bank_indices",
        "image_scores",
        "anomaly_maps",
    ):
        assert torch.equal(getattr(measured, name), getattr(ordinary, name))

    assert benchmark.benchmark_sample_id == ("mvtec_ad/bottle/test/crack/001.png")
    assert [item.sample.sample_id for item in measured.test_samples][0] == (
        benchmark.benchmark_sample_id
    )
    workload = benchmark.workload
    assert {name: workload[name] for name in ("Q", "M", "D", "k")} == {
        "Q": 1024,
        "M": 2048,
        "D": 512,
        "k": 1,
    }
    assert workload["dtype"] == "float32"
    assert workload["bank_chunk_size"] == _CHUNK_SIZE
    assert workload["bank_bytes"] == 2048 * 512 * 4
    assert workload["training_sample_count"] == 2
    assert workload["test_sample_count"] == 2
    assert set(workload["tensor_layout"]) == {
        "anomaly_map",
        "image",
        "memory_bank",
        "patch_embeddings",
    }

    assert len(cpu_benchmark.timing_calls) == 5
    assert all(
        value == 1000.0
        for call in cpu_benchmark.timing_calls[:2]
        for value in call.values()
    )
    repeated = benchmark.results["repeated_stages"]
    assert set(repeated) == _STAGES
    for statistics in repeated.values():
        assert statistics["count"] == 3
        assert statistics["mean"] == 2.0
        assert statistics["p50"] == 2.0
        assert statistics["p95"] == pytest.approx(2.9)
    end_to_end = benchmark.results["synchronized_end_to_end"]
    assert end_to_end["count"] == 3
    assert end_to_end["mean"] == 2.0
    assert end_to_end is not repeated

    assert benchmark.results["one_off_ms"] == {
        "bank_transfer_and_device_setup": 3.0,
        "full_nominal_bank_build": 2.0,
        "model_and_weight_load": 1.0,
    }
    assert _STAGES <= set(benchmark.methodology["stage_inclusion_boundaries"])
    assert (
        "synchronized_end_to_end" in benchmark.methodology["stage_inclusion_boundaries"]
    )
    assert benchmark.methodology["warmup_samples_in_statistics"] is False
    assert benchmark.methodology["cpu_timing_method"] == (
        "time.perf_counter_ns wall clock"
    )
    assert benchmark.methodology["cuda_timing_method"] is None
    assert benchmark.environment == {
        "cuda_compute_capability": None,
        "cuda_device_name": None,
        "pytorch_cuda_runtime_version": None,
    }
    assert benchmark.results["device_memory"]["peak_allocated_bytes"] is None
    assert benchmark.results["device_memory"]["peak_reserved_bytes"] is None


def test_benchmark_json_is_canonical_finite_and_builtin(
    cpu_benchmark: SimpleNamespace,
) -> None:
    benchmark = cpu_benchmark.benchmark
    value = benchmark.to_json_value()
    _assert_builtin_json(value)
    encoded = benchmark.canonical_json()
    assert encoded == (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode("utf-8")
    assert json.loads(encoded) == value


def test_cuda_bank_transfer_uses_explicit_events_stream_and_sync(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:3")
    stream = object()
    log: list[tuple[object, ...]] = []
    events = []

    class FakeEvent:
        def __init__(self, *, enable_timing: bool = False) -> None:
            log.append(("event", enable_timing))
            assert enable_timing is True
            events.append(self)

        def record(self, requested_stream: object) -> None:
            log.append(("record", requested_stream))

        def elapsed_time(self, end: object) -> float:
            assert log[-1] == ("synchronize", device)
            assert end is events[1]
            log.append(("elapsed", end))
            return 4.25

    monkeypatch.setattr(benchmark_module.torch.cuda, "Event", FakeEvent)
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "synchronize",
        lambda requested: log.append(("synchronize", requested)),
    )

    def current_stream(requested: torch.device) -> object:
        log.append(("current_stream", requested))
        return stream

    sentinel = object()
    monkeypatch.setattr(benchmark_module.torch.cuda, "current_stream", current_stream)
    monkeypatch.setattr(
        benchmark_module,
        "_transfer_memory_bank",
        lambda bank, requested: sentinel,
    )

    transferred, milliseconds = benchmark_module._time_bank_transfer(
        object(),
        device,  # type: ignore[arg-type]
    )
    assert transferred is sentinel
    assert milliseconds == 4.25
    assert [item for item in log if item[0] == "synchronize"] == [
        ("synchronize", device),
        ("synchronize", device),
    ]
    assert ("current_stream", device) in log
    assert [item for item in log if item[0] == "record"] == [
        ("record", stream),
        ("record", stream),
    ]
    assert len(events) == 2


def test_cuda_batch1_uses_ordered_events_and_synchronized_end_to_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:3")
    stream = object()
    log: list[tuple[object, ...]] = []

    class DecodedPixels:
        def close(self) -> None:
            log.append(("close",))

    class PreparedImage:
        def unsqueeze(self, dimension: int) -> "PreparedImage":
            assert dimension == 0
            log.append(("unsqueeze",))
            return self

        def to(self, requested: torch.device) -> object:
            assert requested == device
            log.append(("transfer", requested))
            return object()

    class FakeEvent:
        def __init__(self, name: str, duration: float) -> None:
            self.name = name
            self.duration = duration

        def record(self, requested_stream: object) -> None:
            assert requested_stream is stream
            log.append(("record", self.name))

        def elapsed_time(self, end: "FakeEvent") -> float:
            assert sum(item[0] == "synchronize" for item in log) == 2
            log.append(("elapsed", self.name, end.name))
            return self.duration

    durations = (0.1, 0.2, 0.3, 0.4)
    events = tuple(
        (FakeEvent(f"{index}-start", duration), FakeEvent(f"{index}-end", 0.0))
        for index, duration in enumerate(durations)
    )
    decoded = SimpleNamespace(image=DecodedPixels())
    patches = (object(),)
    retrieval_bank = object()
    monkeypatch.setattr(benchmark_module, "decode_image", lambda path: decoded)
    monkeypatch.setattr(
        benchmark_module, "preprocess_decoded_image", lambda value: PreparedImage()
    )
    monkeypatch.setattr(
        benchmark_module,
        "extract_patch_embeddings",
        lambda extractor, images: patches,
    )

    def retrieve(
        queries: object, bank: object, *, bank_chunk_size: int
    ) -> tuple[Tensor, Tensor]:
        assert queries is patches[0]
        assert bank is retrieval_bank
        assert bank_chunk_size == 123
        return torch.zeros(1024), torch.arange(1024)

    monkeypatch.setattr(benchmark_module, "exact_top1_squared_l2", retrieve)
    monkeypatch.setattr(
        benchmark_module,
        "reconstruct_anomaly_maps",
        lambda distances: torch.zeros(1, 256, 256),
    )
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "synchronize",
        lambda requested: log.append(("synchronize", requested)),
    )
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "current_stream",
        lambda requested: log.append(("current_stream", requested)) or stream,
    )
    clock = iter((0, 1_000_000, 2_000_000, 3_000_000, 4_000_000, 10_000_000))
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", lambda: next(clock))

    result = benchmark_module._measure_cuda_batch1(
        Path("image.png"),
        object(),  # type: ignore[arg-type]
        retrieval_bank,  # type: ignore[arg-type]
        device,
        123,
        events,  # type: ignore[arg-type]
    )

    assert result == {
        "image_decode": 1.0,
        "canonical_image_preprocessing": 1.0,
        "host_to_device_transfer": 0.1,
        "frozen_feature_extraction": 0.2,
        "exact_chunked_retrieval": 0.3,
        "anomaly_map_reconstruction": 0.4,
        "end_to_end": 10.0,
    }
    assert [item for item in log if item[0] == "synchronize"] == [
        ("synchronize", device),
        ("synchronize", device),
    ]
    assert ("current_stream", device) in log
    assert [item[1] for item in log if item[0] == "record"] == [
        "0-start",
        "0-end",
        "1-start",
        "1-end",
        "2-start",
        "2-end",
        "3-start",
        "3-end",
    ]
    assert ("close",) in log


def test_artifact_inventory_is_seven_or_eight_and_failure_is_atomic(
    tmp_path: Path,
    cpu_benchmark: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    evaluation = cpu_benchmark.measured
    benchmark = cpu_benchmark.benchmark
    normal = persist_baseline_run(evaluation, tmp_path, _metadata("normal-run"))
    measured = persist_baseline_run(
        evaluation,
        tmp_path,
        _metadata("benchmark-run"),
        benchmark=benchmark,
    )
    assert {path.name for path in normal.iterdir()} == _EVALUATION_FILES
    assert {path.name for path in measured.iterdir()} == _EVALUATION_FILES | {
        "benchmark.json"
    }
    assert json.loads((normal / "run.json").read_bytes())["benchmark"] is None
    assert json.loads((measured / "run.json").read_bytes())["benchmark"] == {
        "artifact": "benchmark.json",
        "schema_version": 1,
        "timing_device": "cpu",
    }
    assert (measured / "benchmark.json").read_bytes() == benchmark.canonical_json()

    real_write_bytes = Path.write_bytes

    def fail_benchmark(path: Path, value: bytes) -> int:
        if path.name == "benchmark.json":
            raise OSError("late benchmark write failed")
        return real_write_bytes(path, value)

    monkeypatch.setattr(Path, "write_bytes", fail_benchmark)
    late_root = tmp_path / "late"
    with pytest.raises(OSError, match="late benchmark write failed"):
        persist_baseline_run(
            evaluation,
            late_root,
            _metadata("late-run"),
            benchmark=replace(benchmark, run_id="late-run"),
        )
    assert not (late_root / "runs" / "late-run").exists()
    assert list((late_root / "runs").iterdir()) == []


def test_benchmark_help_and_nonpositive_arguments_use_argparse_status_two() -> None:
    help_result = _console("benchmark", "--help")
    assert help_result.returncode == 0, help_result.stderr
    for option in (
        "--config",
        "--dataset-root",
        "--category",
        "--device",
        "--output-root",
        "--run-id",
        "--warmup-count",
        "--repeat-count",
    ):
        assert option in help_result.stdout
    for forbidden in ("--batch-size", "--precision", "--profiler", "--plot"):
        assert forbidden not in help_result.stdout

    base = (
        "benchmark",
        "--config",
        str(_PROFILE),
        "--dataset-root",
        "missing",
        "--category",
        "bottle",
        "--device",
        "cpu",
    )
    for option, value in (("--warmup-count", "0"), ("--repeat-count", "-1")):
        result = _console(*base, option, value)
        assert result.returncode == 2
        assert "positive integer" in result.stderr


def test_benchmark_cli_defaults_forwarding_and_success_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    parser_arguments = cli._argument_parser().parse_args(
        [
            "benchmark",
            "--config",
            str(_PROFILE),
            "--dataset-root",
            "dataset",
            "--category",
            "bottle",
            "--device",
            "cpu",
        ]
    )
    assert (parser_arguments.warmup_count, parser_arguments.repeat_count) == (5, 30)

    import inspectrt.features as features

    calls: dict[str, object] = {}

    class Extractor:
        def to(self, device: torch.device) -> "Extractor":
            calls["extractor_device"] = device
            return self

        def eval(self) -> "Extractor":
            calls["extractor_eval"] = True
            return self

    extractor = Extractor()
    evaluation = SimpleNamespace(
        category="bottle",
        metrics=SimpleNamespace(
            image_auroc=0.75,
            image_average_precision=0.5,
            pixel_auroc=0.625,
        ),
    )
    benchmark = SimpleNamespace(
        device="cpu",
        results={"synchronized_end_to_end": {"p50": 1.5, "p95": 2.5}},
    )
    metadata = SimpleNamespace(
        run_id="cli-benchmark",
        created_at_utc="2026-07-15T12:00:00Z",
    )

    monkeypatch.setattr(cli, "_configure_determinism", lambda *args: {"seed": 0})
    monkeypatch.setattr(cli, "_resolve_device", lambda *args: torch.device("cpu"))
    monkeypatch.setattr(cli, "_baseline_run_metadata", lambda *args: metadata)
    clock = iter((10_000_000, 11_500_000))
    monkeypatch.setattr(cli, "perf_counter_ns", lambda: next(clock))
    monkeypatch.setattr(
        features,
        "build_resnet50_layer2_extractor",
        lambda *, weights: extractor,
    )

    def run_benchmark(*args: object, **kwargs: object) -> tuple[object, object]:
        calls["benchmark"] = (args, kwargs)
        return evaluation, benchmark

    monkeypatch.setattr(benchmark_module, "benchmark_mvtec_category", run_benchmark)

    def persist(
        result: object,
        output_root: Path,
        run_metadata: object,
        *,
        benchmark: object,
    ) -> Path:
        calls["persist"] = (result, output_root, run_metadata, benchmark)
        return output_root / "runs" / "cli-benchmark"

    monkeypatch.setattr(artifacts, "persist_baseline_run", persist)
    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "outputs"
    status = cli.main(
        [
            "benchmark",
            "--config",
            str(_PROFILE),
            "--dataset-root",
            str(dataset_root),
            "--category",
            "bottle",
            "--device",
            "cpu",
            "--output-root",
            str(output_root),
            "--run-id",
            "cli-benchmark",
            "--warmup-count",
            "7",
            "--repeat-count",
            "9",
        ]
    )
    assert status == 0
    arguments, keywords = calls["benchmark"]
    assert arguments == (dataset_root, "bottle", extractor)
    assert keywords == {
        "device": torch.device("cpu"),
        "bank_chunk_size": 16384,
        "warmup_count": 7,
        "repeat_count": 9,
        "model_and_weight_load_ms": 1.5,
        "run_id": "cli-benchmark",
        "created_at_utc": "2026-07-15T12:00:00Z",
    }
    assert calls["persist"] == (evaluation, output_root, metadata, benchmark)
    assert calls["extractor_device"] == torch.device("cpu")
    assert calls["extractor_eval"] is True
    assert capsys.readouterr().out.splitlines() == [
        f"Run written to {output_root}/runs/cli-benchmark",
        "category=bottle",
        "image_auroc=0.75",
        "image_average_precision=0.5",
        "pixel_auroc=0.625",
        "synchronized_end_to_end_p50_ms=1.5",
        "synchronized_end_to_end_p95_ms=2.5",
        "timing_device=cpu",
    ]


def test_cli_import_remains_lightweight() -> None:
    code = (
        "import sys; import inspectrt.cli; "
        "assert 'torch' not in sys.modules; "
        "assert 'torchvision' not in sys.modules; "
        "assert 'numpy' not in sys.modules"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr
