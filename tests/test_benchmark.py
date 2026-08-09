from dataclasses import FrozenInstanceError, fields
import inspect
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

import inspectrt.benchmark as benchmark_module
import inspectrt.evaluation as evaluation_module
import inspectrt.features as features_module
import inspectrt.retrieval as retrieval_module
from inspectrt.benchmark import BaselineBenchmark, benchmark_mvtec_category

_STAGES = (
    "image_decode",
    "canonical_image_preprocessing",
    "host_to_device_transfer",
    "frozen_feature_extraction",
    "exact_chunked_retrieval",
    "anomaly_map_reconstruction",
)


def _measurement(values: list[int]) -> dict[str, object]:
    return {"raw_ns": values, "summary_ns": benchmark_module._summary_ns(values)}


def _environment(kind: str) -> tuple[str, dict[str, object]]:
    if kind == "cpu":
        return "cpu", {"kind": "cpu", "properties": {}}
    if kind == "cuda":
        return "cuda:2", {
            "kind": "cuda",
            "properties": {
                "available": True,
                "device_index": 2,
                "compute_capability": [8, 9],
                "device_name": "Synthetic CUDA Device",
                "pytorch_cuda_runtime_version": "13.0",
            },
        }
    return "mps", {
        "kind": "mps",
        "properties": {
            "available": True,
            "built": True,
            "pytorch_enable_mps_fallback": "unset",
        },
    }


def _memory(kind: str) -> dict[str, object]:
    if kind == "cpu":
        return {"kind": "cpu", "host_peak_memory": "not_measured"}
    if kind == "cuda":
        return {
            "kind": "cuda",
            "peak_allocated_bytes": 10,
            "peak_reserved_bytes": 20,
            "peak_window": "after_warmups_through_all_measured_passes",
        }
    return {
        "kind": "mps",
        "observations": [
            {
                "boundary": boundary,
                "current_allocated_bytes": index,
                "driver_allocated_bytes": index + 1,
            }
            for index, boundary in enumerate(
                ("after_setup", "after_warmups", "after_measured_passes")
            )
        ],
        "peak_memory": "not_available_in_selected_pytorch_api",
        "recommended_max_memory_bytes": 100,
    }


def _record_value(kind: str = "cpu") -> dict[str, object]:
    device, environment = _environment(kind)
    repeated = list(range(30))
    return {
        "schema_version": 2,
        "profile_id": "inspectrt_feature_memory_v1",
        "category": "bottle",
        "device": device,
        "benchmark_sample_id": ("mvtec_ad/bottle/test/broken_large/000.png"),
        "run_id": "timing-v2-run",
        "created_at_utc": "2026-08-09T12:00:00Z",
        "workload": benchmark_module._workload(),
        "methodology": benchmark_module._methodology(),
        "environment": environment,
        "results": {
            "one_off": {
                "model_and_weight_load": _measurement([1]),
                "full_nominal_bank_build": _measurement([2]),
                "bank_transfer_and_device_setup": _measurement([3]),
            },
            "repeated_stages": {
                name: _measurement(repeated.copy()) for name in _STAGES
            },
            "synchronized_end_to_end": _measurement(repeated.copy()),
            "memory_observations": _memory(kind),
        },
    }


def _record(kind: str = "cpu") -> BaselineBenchmark:
    return BaselineBenchmark(**_record_value(kind))  # type: ignore[arg-type]


def _nested(value: object, *path: str | int) -> object:
    for part in path:
        value = value[part]  # type: ignore[index]
    return value


def test_schema_two_record_has_the_exact_top_level_role() -> None:
    record = _record()
    assert record.schema_version == 2
    assert record.profile_id == "inspectrt_feature_memory_v1"
    assert tuple(field.name for field in fields(BaselineBenchmark)) == (
        "schema_version",
        "profile_id",
        "category",
        "device",
        "benchmark_sample_id",
        "run_id",
        "created_at_utc",
        "workload",
        "methodology",
        "environment",
        "results",
    )
    with pytest.raises(ValueError, match="schema_version must be 2"):
        BaselineBenchmark(**{**_record_value(), "schema_version": 1})  # type: ignore[arg-type]


def test_methodology_identity_and_boundary_literals_are_exact() -> None:
    methodology = _record().methodology
    assert methodology["methodology_id"] == "inspectrt_synchronized_wall_clock_v2"
    assert methodology["methodology_version"] == 2
    assert methodology["clock"] == "time.perf_counter_ns"
    assert methodology["timing_unit"] == "nanoseconds"
    assert methodology["warmup_count"] == 5
    assert methodology["repeat_count"] == 30
    assert methodology["warmup_samples_in_statistics"] is False
    assert methodology["percentile_method"] == (
        "linear interpolation at rank (n - 1) * q"
    )
    assert methodology["stage_measurement_pass"] == ("segmented_complete_pipeline")
    assert methodology["end_to_end_measurement_pass"] == (
        "separate_uninterrupted_complete_pipeline"
    )
    assert methodology["synchronization_policy"] == {
        "cpu": (
            "No accelerator synchronization; time.perf_counter_ns surrounds synchronous "
            "host work."
        ),
        "cuda": (
            "Call torch.cuda.synchronize(requested_device) immediately before each "
            "accelerator or complete-pass boundary outside the timer, then call it after "
            "submitted work inside the timer before the end timestamp."
        ),
        "mps": (
            "Call torch.mps.synchronize() immediately before each accelerator or "
            "complete-pass boundary outside the timer, then call it after submitted work "
            "inside the timer before the end timestamp."
        ),
    }


def test_raw_arrays_and_recomputed_summaries_are_valid_and_ordered() -> None:
    record = _record()
    repeated = record.results["repeated_stages"]
    assert set(repeated) == set(_STAGES)
    for component in repeated.values():
        assert component["raw_ns"] == tuple(range(30))
        assert component["summary_ns"] == benchmark_module._summary_ns(range(30))
        assert component["summary_ns"]["count"] == 30
    for component in record.results["one_off"].values():
        assert len(component["raw_ns"]) == 1
        assert component["summary_ns"]["count"] == 1


@pytest.mark.parametrize("raw", (-1, True, 9_223_372_036_854_775_808))
def test_raw_nanoseconds_reject_negative_bool_and_signed_64_overflow(
    raw: object,
) -> None:
    value = _record_value()
    component = _nested(value, "results", "repeated_stages", "image_decode")
    component["raw_ns"][0] = raw  # type: ignore[index]
    with pytest.raises((TypeError, ValueError)):
        BaselineBenchmark(**value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("field", "replacement"),
    (
        ("count", 29),
        ("minimum", 1),
        ("maximum", 28),
        ("mean", 14.0),
        ("p50", 14.0),
        ("p95", 27.0),
    ),
)
def test_supplied_summary_must_match_raw_observations(
    field: str, replacement: int | float
) -> None:
    value = _record_value()
    summary = _nested(
        value,
        "results",
        "repeated_stages",
        "image_decode",
        "summary_ns",
    )
    summary[field] = replacement  # type: ignore[index]
    with pytest.raises(ValueError, match=field):
        BaselineBenchmark(**value)  # type: ignore[arg-type]


def test_percentiles_cover_exact_and_interpolated_ranks() -> None:
    assert benchmark_module._summary_ns([20, 0, 10])["p50"] == 10.0
    summary = benchmark_module._summary_ns([30, 0, 20, 10])
    assert summary["p50"] == 15.0
    assert summary["p95"] == 28.499999999999996


@pytest.mark.parametrize(
    ("path", "raw"),
    (
        (("results", "repeated_stages", "image_decode"), list(range(29))),
        (("results", "synchronized_end_to_end"), list(range(31))),
        (("results", "one_off", "model_and_weight_load"), []),
        (("results", "one_off", "full_nominal_bank_build"), [1, 2]),
    ),
)
def test_raw_array_lengths_are_exact(path: tuple[str, ...], raw: list[int]) -> None:
    value = _record_value()
    component = _nested(value, *path)
    component["raw_ns"] = raw  # type: ignore[index]
    component["summary_ns"] = (  # type: ignore[index]
        benchmark_module._summary_ns(raw) if raw else {}
    )
    with pytest.raises(ValueError, match="exactly"):
        BaselineBenchmark(**value)  # type: ignore[arg-type]


def test_record_is_recursively_immutable_and_caller_isolated() -> None:
    value = _record_value("mps")
    record = BaselineBenchmark(**value)  # type: ignore[arg-type]
    value["workload"]["bank_shape"][0] = 1  # type: ignore[index]
    with pytest.raises(FrozenInstanceError):
        record.schema_version = 3  # type: ignore[misc]
    with pytest.raises(TypeError):
        record.results["one_off"] = {}  # type: ignore[index]
    with pytest.raises(TypeError):
        record.results["repeated_stages"]["image_decode"]["raw_ns"][0] = 1  # type: ignore[index]
    assert record.workload["bank_shape"] == (214_016, 512)


def test_cpu_synchronization_path_calls_no_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "synchronize",
        lambda *args: (_ for _ in ()).throw(AssertionError("CUDA synchronized")),
    )
    monkeypatch.setattr(
        benchmark_module.torch.mps,
        "synchronize",
        lambda: (_ for _ in ()).throw(AssertionError("MPS synchronized")),
    )
    benchmark_module._synchronize_backend(torch.device("cpu"))


def test_cuda_requires_an_available_explicit_index(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with pytest.raises(ValueError, match="explicit index"):
        benchmark_module._synchronize_backend(torch.device("cuda"))
    monkeypatch.setattr(benchmark_module.torch.cuda, "is_available", lambda: False)
    with pytest.raises(RuntimeError, match="unavailable"):
        benchmark_module._synchronize_backend(torch.device("cuda:2"))


def test_cuda_pre_sync_is_before_start_and_post_sync_is_before_end(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:2")
    log: list[object] = []
    monkeypatch.setattr(benchmark_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark_module.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "synchronize",
        lambda requested: log.append(("synchronize", requested)),
    )
    clock = iter((10, 25))

    def now() -> int:
        value = next(clock)
        log.append(("clock", value))
        return value

    monkeypatch.setattr(benchmark_module, "perf_counter_ns", now)
    result, duration = benchmark_module._time_backend_operation(
        device, lambda: log.append(("work",)) or "result"
    )
    assert result == "result"
    assert duration == 15
    assert log == [
        ("synchronize", device),
        ("clock", 10),
        ("work",),
        ("synchronize", device),
        ("clock", 25),
    ]


def test_cuda_synchronization_receives_the_requested_device(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    requested = torch.device("cuda:3")
    calls = []
    monkeypatch.setattr(benchmark_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark_module.torch.cuda, "device_count", lambda: 4)
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "synchronize",
        lambda device: calls.append(device),
    )
    benchmark_module._synchronize_backend(requested)
    assert calls == [requested]


def test_mps_requires_built_and_available_backend(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(benchmark_module.torch.backends.mps, "is_built", lambda: False)
    with pytest.raises(RuntimeError, match="not built"):
        benchmark_module._synchronize_backend(torch.device("mps"))
    monkeypatch.setattr(benchmark_module.torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(
        benchmark_module.torch.backends.mps, "is_available", lambda: False
    )
    with pytest.raises(RuntimeError, match="unavailable"):
        benchmark_module._synchronize_backend(torch.device("mps"))


def test_mps_accepts_absent_fallback_and_never_selects_cpu(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = []
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(benchmark_module.torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(
        benchmark_module.torch.backends.mps, "is_available", lambda: True
    )
    monkeypatch.setattr(
        benchmark_module.torch.mps, "synchronize", lambda: calls.append("mps")
    )
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "synchronize",
        lambda *args: calls.append("cuda"),
    )
    benchmark_module._synchronize_backend(torch.device("mps"))
    assert calls == ["mps"]


@pytest.mark.parametrize("fallback", ("1", "0", ""))
def test_mps_rejects_any_present_fallback_value(
    monkeypatch: pytest.MonkeyPatch, fallback: str
) -> None:
    monkeypatch.setenv("PYTORCH_ENABLE_MPS_FALLBACK", fallback)
    monkeypatch.setattr(benchmark_module.torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(
        benchmark_module.torch.backends.mps, "is_available", lambda: True
    )
    with pytest.raises(RuntimeError, match="must be absent"):
        benchmark_module._synchronize_backend(torch.device("mps"))


def test_mps_pre_and_post_synchronization_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log = []
    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(benchmark_module.torch.backends.mps, "is_built", lambda: True)
    monkeypatch.setattr(
        benchmark_module.torch.backends.mps, "is_available", lambda: True
    )
    monkeypatch.setattr(
        benchmark_module.torch.mps, "synchronize", lambda: log.append("sync")
    )
    clock = iter((2, 9))
    monkeypatch.setattr(
        benchmark_module,
        "perf_counter_ns",
        lambda: log.append("clock") or next(clock),
    )
    _, duration = benchmark_module._time_backend_operation(
        torch.device("mps"), lambda: log.append("work")
    )
    assert duration == 7
    assert log == ["sync", "clock", "work", "sync", "clock"]


def _mock_pipeline(monkeypatch: pytest.MonkeyPatch, log: list[str]) -> None:
    class Pixels:
        def close(self) -> None:
            log.append("close")

    class Image:
        def unsqueeze(self, dimension: int) -> "Batch":
            assert dimension == 0
            log.append("batch")
            return Batch()

    class Batch:
        def to(self, device: torch.device) -> str:
            log.append("transfer")
            return "images"

    monkeypatch.setattr(
        benchmark_module,
        "decode_image",
        lambda path: log.append("decode") or SimpleNamespace(image=Pixels()),
    )
    monkeypatch.setattr(
        benchmark_module,
        "preprocess_decoded_image",
        lambda decoded: log.append("preprocess") or Image(),
    )
    monkeypatch.setattr(
        benchmark_module,
        "extract_patch_embeddings",
        lambda extractor, images: log.append("feature") or [object()],
    )

    def retrieve(
        queries: object, bank: object, *, bank_chunk_size: int
    ) -> tuple[torch.Tensor, torch.Tensor]:
        log.append("retrieve")
        return torch.zeros(1024), torch.arange(1024)

    monkeypatch.setattr(benchmark_module, "exact_top1_squared_l2", retrieve)
    monkeypatch.setattr(
        benchmark_module,
        "reconstruct_anomaly_maps",
        lambda distances: log.append("map") or torch.zeros(1, 256, 256),
    )


def test_segmented_pass_executes_the_complete_pipeline_in_stage_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    _mock_pipeline(monkeypatch, log)
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", iter(range(12)).__next__)
    result = benchmark_module._measure_segmented_batch1(
        Path("image.png"), object(), object(), torch.device("cpu"), 16_384
    )
    assert result == (1, 1, 1, 1, 1, 1)
    assert log == [
        "decode",
        "preprocess",
        "batch",
        "transfer",
        "feature",
        "retrieve",
        "map",
        "close",
    ]


def test_uninterrupted_pass_has_only_outer_accelerator_synchronization(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    log: list[str] = []
    _mock_pipeline(monkeypatch, log)
    device = torch.device("cuda:2")
    monkeypatch.setattr(benchmark_module.torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(benchmark_module.torch.cuda, "device_count", lambda: 3)
    monkeypatch.setattr(
        benchmark_module.torch.cuda,
        "synchronize",
        lambda requested: log.append("sync"),
    )
    monkeypatch.setattr(benchmark_module, "perf_counter_ns", iter((10, 30)).__next__)
    duration = benchmark_module._measure_end_to_end_batch1(
        Path("image.png"), object(), object(), device, 16_384
    )
    assert duration == 20
    assert log == [
        "sync",
        "decode",
        "preprocess",
        "batch",
        "transfer",
        "feature",
        "retrieve",
        "map",
        "sync",
        "close",
    ]


def test_runner_discards_five_warmups_and_measures_thirty_separate_passes(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    nominal = tuple(SimpleNamespace(sample_id=f"train-{index}") for index in range(209))
    tests = (
        SimpleNamespace(
            sample_id="mvtec_ad/bottle/test/broken_large/000.png",
            image_relpath="bottle/test/broken_large/000.png",
        ),
        *(SimpleNamespace(sample_id=f"test-{index}") for index in range(82)),
    )
    bank = SimpleNamespace(shape=(214_016, 512))
    retrieval_bank = object()
    evaluation = object()
    monkeypatch.setattr(benchmark_module, "_synchronize_backend", lambda device: None)
    monkeypatch.setattr(
        benchmark_module,
        "_resolve_evaluation_device",
        lambda extractor, device: device,
    )
    monkeypatch.setattr(
        benchmark_module,
        "_discover_category_samples",
        lambda root, category: ((*nominal, *tests), nominal, tests),
    )
    monkeypatch.setattr(
        benchmark_module, "_build_nominal_memory_bank", lambda *args: bank
    )
    monkeypatch.setattr(
        benchmark_module, "_transfer_memory_bank", lambda *args: retrieval_bank
    )
    one_off = iter((20, 30))

    def time_operation(device: torch.device, operation: object) -> tuple[object, int]:
        return operation(), next(one_off)  # type: ignore[operator]

    monkeypatch.setattr(benchmark_module, "_time_backend_operation", time_operation)
    segmented_calls = []
    complete_calls = []

    def segmented(*args: object) -> tuple[int, ...]:
        ordinal = len(segmented_calls)
        segmented_calls.append(ordinal)
        base = 10_000 if ordinal < 5 else ordinal - 5
        return tuple(base + stage for stage in range(6))

    def complete(*args: object) -> int:
        ordinal = len(complete_calls)
        complete_calls.append(ordinal)
        return 20_000 if ordinal < 5 else 100 + ordinal - 5

    monkeypatch.setattr(benchmark_module, "_measure_segmented_batch1", segmented)
    monkeypatch.setattr(benchmark_module, "_measure_end_to_end_batch1", complete)
    monkeypatch.setattr(
        benchmark_module, "_score_and_finalize_category", lambda *args: evaluation
    )
    measured, record = benchmark_mvtec_category(
        Path("dataset"),
        "bottle",
        object(),  # type: ignore[arg-type]
        device="cpu",
        bank_chunk_size=16_384,
        warmup_count=5,
        repeat_count=30,
        model_and_weight_load_ns=10,
        run_id="run",
        created_at_utc="2026-08-09T12:00:00Z",
    )
    assert measured is evaluation
    assert segmented_calls == list(range(35))
    assert complete_calls == list(range(35))
    repeated = record.results["repeated_stages"]
    for stage, name in enumerate(_STAGES):
        assert repeated[name]["raw_ns"] == tuple(
            ordinal + stage for ordinal in range(30)
        )
        assert 10_000 + stage not in repeated[name]["raw_ns"]
    end_to_end = record.results["synchronized_end_to_end"]["raw_ns"]
    assert end_to_end == tuple(range(100, 130))
    assert 20_000 not in end_to_end
    assert end_to_end[0] != sum(repeated[name]["raw_ns"][0] for name in _STAGES)
    assert record.results["one_off"]["model_and_weight_load"]["raw_ns"] == (10,)


@pytest.mark.parametrize(
    ("warmups", "repeats", "message"),
    ((4, 30, "warmup_count"), (5, 29, "repeat_count")),
)
def test_runner_requires_fixed_warmup_and_repeat_counts(
    warmups: int, repeats: int, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        benchmark_mvtec_category(
            Path("dataset"),
            "bottle",
            object(),  # type: ignore[arg-type]
            device="cpu",
            bank_chunk_size=16_384,
            warmup_count=warmups,
            repeat_count=repeats,
            model_and_weight_load_ns=1,
            run_id="run",
            created_at_utc="2026-08-09T12:00:00Z",
        )


def test_model_load_requires_raw_integer_nanoseconds() -> None:
    with pytest.raises(TypeError, match="ordinary Python integers"):
        benchmark_mvtec_category(
            Path("dataset"),
            "bottle",
            object(),  # type: ignore[arg-type]
            device="cpu",
            bank_chunk_size=16_384,
            warmup_count=5,
            repeat_count=30,
            model_and_weight_load_ns=1.0,  # type: ignore[arg-type]
            run_id="run",
            created_at_utc="2026-08-09T12:00:00Z",
        )


def test_cpu_memory_representation_is_closed() -> None:
    assert benchmark_module._memory_observations(torch.device("cpu"), ()) == {
        "kind": "cpu",
        "host_peak_memory": "not_measured",
    }


def test_cuda_memory_representation_uses_the_measured_peak_window(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    device = torch.device("cuda:2")
    monkeypatch.setattr(
        benchmark_module.torch.cuda, "max_memory_allocated", lambda requested: 10
    )
    monkeypatch.setattr(
        benchmark_module.torch.cuda, "max_memory_reserved", lambda requested: 20
    )
    assert benchmark_module._memory_observations(device, ()) == {
        "kind": "cuda",
        "peak_allocated_bytes": 10,
        "peak_reserved_bytes": 20,
        "peak_window": "after_warmups_through_all_measured_passes",
    }


def test_mps_memory_uses_ordered_point_observations_not_peaks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    current = iter((1, 3, 5))
    driver = iter((2, 4, 6))
    monkeypatch.setattr(
        benchmark_module.torch.mps,
        "current_allocated_memory",
        lambda: next(current),
    )
    monkeypatch.setattr(
        benchmark_module.torch.mps,
        "driver_allocated_memory",
        lambda: next(driver),
    )
    monkeypatch.setattr(
        benchmark_module.torch.mps, "recommended_max_memory", lambda: 100
    )
    points = [
        benchmark_module._mps_memory_observation(boundary)
        for boundary in ("after_setup", "after_warmups", "after_measured_passes")
    ]
    memory = benchmark_module._memory_observations(torch.device("mps"), points)
    assert [point["boundary"] for point in memory["observations"]] == [
        "after_setup",
        "after_warmups",
        "after_measured_passes",
    ]
    assert memory["peak_memory"] == "not_available_in_selected_pytorch_api"
    assert all("peak" not in point for point in memory["observations"])


def test_mps_memory_rejects_reordered_or_peak_labeled_points() -> None:
    value = _record_value("mps")
    observations = _nested(value, "results", "memory_observations", "observations")
    observations.reverse()  # type: ignore[union-attr]
    with pytest.raises(ValueError, match="out of order"):
        BaselineBenchmark(**value)  # type: ignore[arg-type]
    value = _record_value("mps")
    point = _nested(value, "results", "memory_observations", "observations", 0)
    point["peak_allocated_bytes"] = 1  # type: ignore[index]
    with pytest.raises(ValueError, match="fields"):
        BaselineBenchmark(**value)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("kind", "device"),
    (("cpu", "mps"), ("mps", "cpu"), ("cuda", "cuda:1")),
)
def test_environment_and_requested_backend_must_match(kind: str, device: str) -> None:
    value = _record_value(kind)
    value["device"] = device
    with pytest.raises(ValueError, match="agree|index"):
        BaselineBenchmark(**value)  # type: ignore[arg-type]


def test_environment_union_rejects_unknown_fields() -> None:
    value = _record_value("cpu")
    value["environment"]["properties"]["available"] = True  # type: ignore[index]
    with pytest.raises(ValueError, match="fields"):
        BaselineBenchmark(**value)  # type: ignore[arg-type]


def test_schema_two_source_has_no_cuda_events_or_schema_one_writer_branch() -> None:
    source = inspect.getsource(benchmark_module)
    assert "torch.cuda.Event" not in source
    assert "current_stream" not in source
    assert "elapsed_time" not in source
    assert "schema_version == 1" not in source
    assert "schema_version=1" not in source


def test_existing_scientific_helpers_are_reused_without_reimplementation() -> None:
    assert (
        benchmark_module._build_nominal_memory_bank
        is evaluation_module._build_nominal_memory_bank
    )
    assert (
        benchmark_module._score_and_finalize_category
        is evaluation_module._score_and_finalize_category
    )
    assert (
        benchmark_module.extract_patch_embeddings
        is features_module.extract_patch_embeddings
    )
    assert (
        benchmark_module.exact_top1_squared_l2 is retrieval_module.exact_top1_squared_l2
    )
    assert (
        benchmark_module.reconstruct_anomaly_maps
        is retrieval_module.reconstruct_anomaly_maps
    )
