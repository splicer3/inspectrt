from datetime import datetime, timezone
import hashlib
from importlib import metadata
import json
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tomllib
from types import SimpleNamespace
from unittest.mock import Mock

import pytest

import inspectrt.cli as cli

_ROOT = Path(__file__).resolve().parents[1]
_PROFILE = _ROOT / "configs" / "baseline.toml"
_RETRIEVAL_FIXTURE = _ROOT / "tests" / "fixtures" / "retrieval_v1"
_TOP_LEVEL_KEYS = {
    "schema_version",
    "profile_id",
    "preprocessing_profile_id",
    "weights",
    "bank_chunk_size",
    "seed",
    "determinism",
}
_DETERMINISM_KEYS = {
    "use_deterministic_algorithms",
    "cudnn_benchmark",
    "allow_tf32",
    "cublas_workspace_config",
}


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


def _modified_profile(tmp_path: Path, old: str, new: str) -> Path:
    text = _PROFILE.read_text(encoding="utf-8")
    assert old in text
    path = tmp_path / "baseline.toml"
    path.write_text(text.replace(old, new, 1), encoding="utf-8")
    return path


def test_import_does_not_load_heavy_runtime_modules() -> None:
    code = (
        "import sys; import inspectrt.cli; "
        "assert 'torch' not in sys.modules; "
        "assert 'torchvision' not in sys.modules; "
        "assert 'numpy' not in sys.modules; "
        "assert 'inspectrt.portability' not in sys.modules; "
        "assert 'inspectrt.fixtures' not in sys.modules; "
        "assert 'inspectrt.retrieval' not in sys.modules"
    )
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_committed_profile_is_supported_and_contains_only_known_keys() -> None:
    config = cli.load_baseline_config(_PROFILE)
    assert config == cli.BaselineConfig(
        1,
        "inspectrt_feature_memory_v1",
        "inspectrt_resize256_v1",
        "IMAGENET1K_V2",
        16384,
        0,
        True,
        False,
        False,
        ":4096:8",
    )
    with _PROFILE.open("rb") as stream:
        raw = tomllib.load(stream)
    assert set(raw) == _TOP_LEVEL_KEYS
    assert set(raw["determinism"]) == _DETERMINISM_KEYS
    assert not (
        {"dataset_root", "category", "device", "output_root", "run_id"} & raw.keys()
    )


@pytest.mark.parametrize(
    ("old", "message"),
    [
        ("bank_chunk_size = 16384\n", "bank_chunk_size"),
        ("allow_tf32 = false\n", "allow_tf32"),
    ],
)
def test_missing_config_keys_fail_clearly(
    tmp_path: Path, old: str, message: str
) -> None:
    path = _modified_profile(tmp_path, old, "")
    with pytest.raises(ValueError, match=rf"missing.*{message}"):
        cli.load_baseline_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("[determinism]", "unknown = 1\n\n[determinism]", "profile.*unknown"),
        (
            'cublas_workspace_config = ":4096:8"',
            'cublas_workspace_config = ":4096:8"\nunknown = 1',
            "determinism.*unknown",
        ),
    ],
)
def test_unknown_config_keys_fail_clearly(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _modified_profile(tmp_path, old, new)
    with pytest.raises(ValueError, match=message):
        cli.load_baseline_config(path)


@pytest.mark.parametrize(
    ("old", "new", "message"),
    [
        ("schema_version = 1", "schema_version = 2", "schema_version"),
        (
            'profile_id = "inspectrt_feature_memory_v1"',
            'profile_id = "other"',
            "profile_id",
        ),
        (
            'preprocessing_profile_id = "inspectrt_resize256_v1"',
            'preprocessing_profile_id = "other"',
            "preprocessing_profile_id",
        ),
        ('weights = "IMAGENET1K_V2"', 'weights = "DEFAULT"', "weights"),
        ("bank_chunk_size = 16384", "bank_chunk_size = 0", "positive integer"),
        ("bank_chunk_size = 16384", "bank_chunk_size = true", "positive integer"),
        ("seed = 0", "seed = -1", "nonnegative integer"),
        ("seed = 0", "seed = true", "nonnegative integer"),
        (
            "use_deterministic_algorithms = true",
            "use_deterministic_algorithms = false",
            "use_deterministic_algorithms",
        ),
        ("cudnn_benchmark = false", "cudnn_benchmark = true", "cudnn_benchmark"),
        ("allow_tf32 = false", "allow_tf32 = true", "allow_tf32"),
        (
            'cublas_workspace_config = ":4096:8"',
            'cublas_workspace_config = ":16:8"',
            "cublas_workspace_config",
        ),
    ],
)
def test_unsupported_config_values_fail_clearly(
    tmp_path: Path, old: str, new: str, message: str
) -> None:
    path = _modified_profile(tmp_path, old, new)
    with pytest.raises(ValueError, match=message):
        cli.load_baseline_config(path)


@pytest.mark.parametrize(
    ("arguments", "expected"),
    [(("--help",), "evaluate"), (("evaluate", "--help"), "--dataset-root")],
)
def test_command_help_succeeds(arguments: tuple[str, ...], expected: str) -> None:
    result = _console(*arguments)
    assert result.returncode == 0, result.stderr
    assert expected in result.stdout
    if arguments[0] == "evaluate":
        for forbidden in (
            "--backbone",
            "--feature-layer",
            "--image-size",
            "--weights",
            "--bank-chunk-size",
            "--batch-size",
            "--seed",
        ):
            assert forbidden not in result.stdout


@pytest.mark.parametrize(
    "arguments",
    (
        ("--help",),
        ("portability", "--help"),
        ("portability", "compare", "--help"),
    ),
)
def test_portability_help_does_not_load_runtime_modules(
    arguments: tuple[str, ...],
) -> None:
    code = f"""
import sys
import inspectrt.cli as cli
try:
    cli.main({list(arguments)!r})
except SystemExit as error:
    assert error.code == 0
else:
    raise AssertionError("help did not exit")
for name in ("numpy", "torch", "torchvision", "inspectrt.portability"):
    assert name not in sys.modules, name
"""
    result = subprocess.run(
        (sys.executable, "-c", code),
        cwd=_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_portability_command_help_and_action_surface() -> None:
    root = _console("--help")
    group = _console("portability", "--help")
    compare = _console("portability", "compare", "--help")
    assert root.returncode == group.returncode == compare.returncode == 0
    assert "portability" in root.stdout
    assert "{compare}" in group.stdout
    for argument in (
        "--reference-run",
        "--candidate-run",
        "--environment-map",
        "--policy",
        "--output",
    ):
        assert argument in compare.stdout
    graph = _console("portability", "graph")
    assert graph.returncode == 2
    assert "invalid choice" in graph.stderr


def test_portability_compare_requires_all_four_inputs() -> None:
    values = {
        "--reference-run": "reference",
        "--candidate-run": "candidate",
        "--environment-map": "environment.json",
        "--output": "comparison",
    }
    for omitted in values:
        arguments = ["portability", "compare"]
        for name, value in values.items():
            if name != omitted:
                arguments.extend((name, value))
        result = _console(*arguments)
        assert result.returncode == 2
        assert omitted in result.stderr


def test_portability_compare_requires_at_least_one_candidate() -> None:
    result = _console(
        "portability",
        "compare",
        "--reference-run",
        "reference",
        "--environment-map",
        "environment.json",
        "--output",
        "comparison",
    )
    assert result.returncode == 2
    assert "--candidate-run" in result.stderr


def _portability_arguments(
    reference: Path,
    candidates: tuple[Path, ...],
    environment_map: Path,
    output: Path,
    policy: Path | None = None,
) -> list[str]:
    arguments = [
        "portability",
        "compare",
        "--reference-run",
        str(reference),
    ]
    for candidate in candidates:
        arguments.extend(("--candidate-run", str(candidate)))
    arguments.extend(("--environment-map", str(environment_map)))
    if policy is not None:
        arguments.extend(("--policy", str(policy)))
    arguments.extend(("--output", str(output)))
    return arguments


@pytest.fixture
def wired_portability(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> SimpleNamespace:
    import inspectrt.portability as portability

    private_root = tmp_path / "private-user" / "secret-host"
    reference = private_root / "reference-benchmark"
    environment_map = private_root / "environment-map.json"
    output = private_root / "comparison"
    records: dict[str, object] = {}

    def publish_portability_comparison(
        *,
        reference_run: Path,
        candidate_runs: tuple[Path, ...],
        environment_map_path: Path,
        output: Path,
        generator: object,
        policy_path: Path | None,
    ) -> tuple[SimpleNamespace, SimpleNamespace]:
        statuses = tuple(
            SimpleNamespace(
                environment_id=f"candidate-{index}",
                status="observed_unclassified",
            )
            for index, _ in enumerate(candidate_runs, 1)
        )
        comparison = SimpleNamespace(
            comparison_id="c" * 64,
            candidates=tuple(object() for _ in candidate_runs),
            policy=None
            if policy_path is None
            else SimpleNamespace(policy_id="reviewed"),
            scientific_results=statuses,
        )
        excluded = tuple(
            SimpleNamespace(environment_id=status.environment_id)
            for candidate, status in zip(candidate_runs, statuses, strict=True)
            if "evaluation" in candidate.name
        )
        performance = SimpleNamespace(
            included_runs=tuple(
                object() for _ in range(1 + len(candidate_runs) - len(excluded))
            ),
            excluded_candidates=excluded,
        )
        records.update(comparison=comparison, performance=performance)
        return comparison, performance

    publisher = Mock(side_effect=publish_portability_comparison)
    monkeypatch.setattr(
        portability,
        "publish_portability_comparison",
        publisher,
        raising=False,
    )
    monkeypatch.setattr(
        cli,
        "_repository_metadata",
        lambda cwd: (_ROOT, "a" * 40, True, "b" * 64),
    )

    def run(
        *candidates: Path,
        policy: Path | None = None,
        reference_run: Path = reference,
        environment_map_path: Path = environment_map,
        destination: Path = output,
    ) -> int:
        return cli.main(
            _portability_arguments(
                reference_run,
                tuple(candidates),
                environment_map_path,
                destination,
                policy,
            )
        )

    return SimpleNamespace(
        run=run,
        publisher=publisher,
        records=records,
        module=portability,
        private_root=private_root,
        reference=reference,
        environment_map=environment_map,
        output=output,
    )


@pytest.mark.parametrize("count", (1, 3))
def test_portability_routes_one_or_multiple_candidates_in_explicit_order(
    wired_portability: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    count: int,
) -> None:
    command = wired_portability
    candidates = tuple(
        command.private_root / f"candidate-{index}" for index in range(count)
    )
    assert command.run(*candidates) == 0
    call = command.publisher.call_args.kwargs
    assert call["reference_run"] == command.reference
    assert call["candidate_runs"] == candidates
    assert call["environment_map_path"] == command.environment_map
    assert call["output"] == command.output
    assert call["generator"] == command.module.ScientificGenerator("a" * 40, True)
    assert f"candidates={count}" in capsys.readouterr().out


@pytest.mark.parametrize("policy_mode", (False, True))
def test_portability_observation_and_policy_modes(
    wired_portability: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    policy_mode: bool,
) -> None:
    command = wired_portability
    candidate = command.private_root / "candidate"
    policy = command.private_root / "policy.json" if policy_mode else None
    assert command.run(candidate, policy=policy) == 0
    assert command.publisher.call_args.kwargs["policy_path"] == policy
    output = capsys.readouterr().out
    assert f"mode={'policy' if policy_mode else 'observation'}" in output
    assert "accepted" not in output


@pytest.mark.parametrize(
    "error",
    (
        ValueError("environment map JSON bytes are not canonical"),
        ValueError("policy JSON contains a duplicate key"),
        ValueError("reference run must be a benchmark bundle"),
        ValueError("reference benchmark data is required"),
        FileExistsError("output directory already exists"),
        OSError("late publication failure"),
        ValueError("comparison validation failed"),
    ),
)
def test_portability_failures_use_one_concise_error_boundary(
    wired_portability: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
    error: Exception,
) -> None:
    command = wired_portability
    command.publisher.side_effect = error
    assert command.run(command.private_root / "candidate") == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == f"inspectrt portability compare failed: {error}\n"


def test_portability_keeps_evaluation_candidate_scientific_and_excludes_timing(
    wired_portability: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = wired_portability
    assert command.run(command.private_root / "evaluation-candidate") == 0
    comparison = command.records["comparison"]
    performance = command.records["performance"]
    assert comparison.scientific_results[0].status == "observed_unclassified"
    assert len(performance.included_runs) == 1
    assert len(performance.excluded_candidates) == 1
    output = capsys.readouterr().out
    assert "performance_included=1" in output
    assert "performance_excluded=1" in output


def test_portability_success_prints_only_the_bounded_summary(
    wired_portability: SimpleNamespace,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = wired_portability
    candidate = command.private_root / "candidate"
    assert command.run(candidate) == 0
    output = capsys.readouterr().out
    assert output.splitlines() == [
        f"comparison_id={'c' * 64}",
        "candidates=1",
        "performance_included=2",
        "performance_excluded=0",
        "mode=observation",
        "status=published",
    ]
    assert str(command.private_root) not in output


def test_fixture_command_help_and_action_surface() -> None:
    root = _console("--help")
    group = _console("fixture", "--help")
    validate = _console("fixture", "validate", "--help")
    export = _console("fixture", "export", "--help")
    assert root.returncode == group.returncode == validate.returncode == 0
    assert export.returncode == 0
    assert "fixture" in root.stdout
    assert "{validate,export}" in group.stdout
    assert "--fixture" in validate.stdout
    assert "--device" in validate.stdout
    for argument in (
        "--config",
        "--run-dir",
        "--dataset-root",
        "--sample-id",
        "--device",
        "--output-root",
    ):
        assert argument in export.stdout
    for action in ("inspect", "generate"):
        result = _console("fixture", action)
        assert result.returncode == 2
        assert "invalid choice" in result.stderr


def test_fixture_export_requires_all_six_arguments() -> None:
    values = {
        "--config": str(_PROFILE),
        "--run-dir": "run",
        "--dataset-root": "dataset",
        "--sample-id": "sample",
        "--device": "cuda:0",
        "--output-root": "outputs",
    }
    for omitted in values:
        arguments = ["fixture", "export"]
        for name, value in values.items():
            if name != omitted:
                arguments.extend((name, value))
        result = _console(*arguments)
        assert result.returncode == 2
        assert omitted in result.stderr


def _export_arguments(*, run_directory: str = "run") -> list[str]:
    return [
        "fixture",
        "export",
        "--config",
        str(_PROFILE),
        "--run-dir",
        run_directory,
        "--dataset-root",
        "dataset",
        "--sample-id",
        "mvtec_ad/bottle/test/broken_large/000.png",
        "--device",
        "cuda:0",
        "--output-root",
        "outputs",
    ]


def test_controlled_fixture_export_returns_accepted_identity(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    def export(arguments: object, config: cli.BaselineConfig) -> int:
        assert config == cli.load_baseline_config(_PROFILE)
        print("fixture_id=bottle-broken-large-000-bc330b9070c5")
        print("fixture_class=real_application")
        print("source_run=accepted-run")
        print("Q=2")
        print("M=3")
        print("D=2")
        print("k=1")
        print(f"payload_sha256={'a' * 64}")
        print(f"fixture_digest={'b' * 64}")
        print("indices=exact")
        print("distances=exact")
        print("status=accepted")
        return 0

    monkeypatch.setattr(cli, "_export_fixture", export)
    assert cli.main(_export_arguments()) == 0
    output = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert output["fixture_id"] == "bottle-broken-large-000-bc330b9070c5"
    assert output["source_run"] == "accepted-run"
    assert (output["Q"], output["M"], output["D"], output["k"]) == (
        "2",
        "3",
        "2",
        "1",
    )
    assert output["indices"] == output["distances"] == "exact"
    assert output["status"] == "accepted"


def test_fixture_export_missing_source_is_concise(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_repository_metadata",
        lambda cwd: (
            _ROOT,
            _git(_ROOT, "rev-parse", "HEAD"),
            False,
            hashlib.sha256((_ROOT / "uv.lock").read_bytes()).hexdigest(),
        ),
    )
    assert cli.main(_export_arguments(run_directory=str(tmp_path / "missing"))) == 1
    captured = capsys.readouterr()
    assert "source run must be a real directory" in captured.err
    assert "Traceback" not in captured.err


@pytest.mark.parametrize(
    ("message", "expected"),
    (
        ("evaluation-only source run", "evaluation-only"),
        ("reference distance mismatch", "reference distance mismatch"),
        ("fixture directory already exists", "already exists"),
    ),
)
def test_fixture_export_expected_failures_return_one(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    message: str,
    expected: str,
) -> None:
    def fail(*args: object) -> int:
        raise ValueError(message)

    monkeypatch.setattr(cli, "_export_fixture", fail)
    assert cli.main(_export_arguments()) == 1
    captured = capsys.readouterr()
    assert expected in captured.err
    assert "Traceback" not in captured.err


def test_fixture_export_rejects_dirty_generator_before_runtime_work(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    monkeypatch.setattr(
        cli,
        "_repository_metadata",
        lambda cwd: (_ROOT, "f" * 40, True, "a" * 64),
    )
    assert cli.main(_export_arguments()) == 1
    assert "working tree must be clean" in capsys.readouterr().err


def test_fixture_validate_requires_both_arguments() -> None:
    for arguments in (
        ("fixture", "validate"),
        ("fixture", "validate", "--fixture", str(_RETRIEVAL_FIXTURE)),
        ("fixture", "validate", "--device", "cpu"),
    ):
        result = _console(*arguments)
        assert result.returncode == 2
        assert "required" in result.stderr


def test_committed_fixture_validates_on_cpu() -> None:
    result = _console(
        "fixture",
        "validate",
        "--fixture",
        str(_RETRIEVAL_FIXTURE),
        "--device",
        "cpu",
    )
    assert result.returncode == 0, result.stderr
    output = dict(line.split("=", 1) for line in result.stdout.splitlines())
    manifest_bytes = (_RETRIEVAL_FIXTURE / "manifest.json").read_bytes()
    payload = (_RETRIEVAL_FIXTURE / "tensors.bin").read_bytes()
    manifest = json.loads(manifest_bytes)
    assert output == {
        "fixture_id": "synthetic-correctness-v1",
        "fixture_class": "synthetic_correctness",
        "Q": "4",
        "M": "7",
        "D": "5",
        "k": "1",
        "chunk_size": "3",
        "payload_sha256": manifest["payload"]["sha256"],
        "fixture_digest": hashlib.sha256(manifest_bytes + payload).hexdigest(),
        "indices": "exact",
        "distances": "exact",
        "status": "accepted",
    }
    assert re.fullmatch(r"[0-9a-f]{64}", output["payload_sha256"])
    assert re.fullmatch(r"[0-9a-f]{64}", output["fixture_digest"])


def _mismatched_fixture(
    tmp_path: Path, *, index: bool = False, distance: bool = False
) -> Path:
    import numpy as np

    from inspectrt.fixtures import RetrievalFixture, load_retrieval_fixture
    from inspectrt.fixtures import write_retrieval_fixture

    loaded = load_retrieval_fixture(_RETRIEVAL_FIXTURE)
    expected_indices = loaded.expected_indices.copy()
    expected_distances = loaded.expected_squared_l2_distances.copy()
    if index:
        expected_indices[0] = np.int64(2)
    if distance:
        expected_distances[0] = np.float32(2)
    fixture = RetrievalFixture(
        loaded.metadata,
        loaded.queries,
        loaded.memory_bank,
        expected_distances,
        expected_indices,
    )
    directory = tmp_path / f"mismatch-{index}-{distance}"
    write_retrieval_fixture(fixture, directory)
    return directory


@pytest.mark.parametrize(
    ("kind", "message"),
    (("missing", "not found"), ("corrupt", "SHA-256 mismatch")),
)
def test_fixture_path_and_hash_failures_are_concise(
    tmp_path: Path, kind: str, message: str
) -> None:
    fixture = tmp_path / kind
    if kind == "corrupt":
        shutil.copytree(_RETRIEVAL_FIXTURE, fixture)
        payload = bytearray((fixture / "tensors.bin").read_bytes())
        payload[0] ^= 1
        (fixture / "tensors.bin").write_bytes(payload)
    result = _console(
        "fixture", "validate", "--fixture", str(fixture), "--device", "cpu"
    )
    assert result.returncode == 1
    assert message in result.stderr
    assert "Traceback" not in result.stderr


@pytest.mark.parametrize(
    ("index", "distance", "message"),
    ((True, False, "index mismatch"), (False, True, "distance mismatch")),
)
def test_fixture_reference_mismatch_fails(
    tmp_path: Path, index: bool, distance: bool, message: str
) -> None:
    fixture = _mismatched_fixture(tmp_path, index=index, distance=distance)
    result = _console(
        "fixture", "validate", "--fixture", str(fixture), "--device", "cpu"
    )
    assert result.returncode == 1
    assert message in result.stderr
    assert "status=accepted" not in result.stdout


def _real_cli_fixture(tmp_path: Path) -> tuple[Path, object]:
    from inspectrt.fixtures import (
        RealApplicationFixtureSource,
        RetrievalFixture,
        RetrievalFixtureMetadata,
        load_retrieval_fixture,
        real_fixture_id,
        write_retrieval_fixture,
    )

    loaded = load_retrieval_fixture(_RETRIEVAL_FIXTURE)
    source_commit = "a" * 40
    lock_digest = hashlib.sha256((_ROOT / "uv.lock").read_bytes()).hexdigest()
    source = RealApplicationFixtureSource(
        category="bottle",
        sample_id="mvtec_ad/bottle/test/crack/000.png",
        test_tensor_index=0,
        accepted_run_id="accepted-run",
        source_commit=source_commit,
        source_dirty=False,
        inventory_sha256="b" * 64,
        uv_lock_sha256=lock_digest,
        weight_enum="ResNet50_Weights.IMAGENET1K_V2",
        weight_file_sha256="d" * 64,
        baseline_profile="inspectrt_feature_memory_v1",
        configuration_sha256="e" * 64,
        preprocessing_identity="inspectrt_resize256_v1",
        feature_layer="layer2",
        source_image_sha256="f" * 64,
        python_version=cli.platform.python_version(),
        dependency_versions={
            name: metadata.version(name)
            for name in (
                "inspectrt",
                "numpy",
                "pillow",
                "scikit-learn",
                "torch",
                "torchvision",
            )
        },
        platform_description=cli.platform.platform(),
        requested_device="cuda:0",
        determinism={
            "allow_tf32": False,
            "cublas_workspace_config": ":4096:8",
            "cudnn_benchmark": False,
            "deterministic_algorithms_warn_only": False,
            "fp32_precision": "ieee",
            "numpy_seed": 0,
            "python_random_seed": 0,
            "torch_cpu_seed": 0,
            "torch_cuda_seed_all": 0,
            "use_deterministic_algorithms": True,
        },
        cuda_device_name="test GPU",
        cuda_compute_capability=(7, 5),
        pytorch_cuda_runtime_version="13.0",
        source_artifact_sha256={
            name: str(index) * 64
            for index, name in enumerate(
                (
                    "run.json",
                    "samples.jsonl",
                    "memory_bank.pt",
                    "retrieval.pt",
                    "benchmark.json",
                ),
                1,
            )
        },
    )
    fixture_metadata = RetrievalFixtureMetadata(
        real_fixture_id(source.category, source.sample_id, source_commit),
        "real_application",
        3,
        source,
        loaded.metadata.generator,
    )
    directory = tmp_path / "real"
    write_retrieval_fixture(
        RetrievalFixture(
            fixture_metadata,
            loaded.queries,
            loaded.memory_bank,
            loaded.expected_squared_l2_distances,
            loaded.expected_indices,
        ),
        directory,
    )
    return directory, source


def test_real_fixture_nonmatching_environment_is_structurally_valid(
    tmp_path: Path,
) -> None:
    directory, _ = _real_cli_fixture(tmp_path)
    result = _console(
        "fixture", "validate", "--fixture", str(directory), "--device", "cpu"
    )
    assert result.returncode == 0, result.stderr
    output = dict(line.split("=", 1) for line in result.stdout.splitlines())
    assert output["fixture_class"] == "real_application"
    assert output["status"] == "structurally_valid"
    assert output["reference_status"] == "unavailable"
    assert "requested_device" in output["environment_mismatches"]
    assert "status=accepted" not in result.stdout


def test_real_structural_validation_skips_reference_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    directory, _ = _real_cli_fixture(tmp_path)
    monkeypatch.setattr(
        cli,
        "_run_fixture_reference",
        lambda *args: (_ for _ in ()).throw(AssertionError("reference executed")),
    )
    assert (
        cli.main(
            [
                "fixture",
                "validate",
                "--fixture",
                str(directory),
                "--device",
                "cpu",
            ]
        )
        == 0
    )
    assert "reference_status=unavailable" in capsys.readouterr().out


def test_real_fixture_matching_environment_requires_exact_reference(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspectrt.fixtures as fixture_module

    directory, source = _real_cli_fixture(tmp_path)
    monkeypatch.setattr(
        cli,
        "_repository_metadata",
        lambda cwd: (_ROOT, "f" * 40, False, source.uv_lock_sha256),
    )
    device = SimpleNamespace(index=0)
    monkeypatch.setattr(cli, "_resolve_device", lambda requested, torch: device)
    monkeypatch.setattr(fixture_module, "cuda_environment_mismatches", lambda *args: [])
    monkeypatch.setattr(
        cli, "_configure_determinism", lambda *args: dict(source.determinism)
    )
    reference_calls: list[object] = []
    monkeypatch.setattr(
        cli,
        "_run_fixture_reference",
        lambda fixture, actual_device, torch: reference_calls.append(actual_device),
    )
    assert (
        cli.main(
            [
                "fixture",
                "validate",
                "--fixture",
                str(directory),
                "--device",
                "cuda:0",
            ]
        )
        == 0
    )
    output = dict(line.split("=", 1) for line in capsys.readouterr().out.splitlines())
    assert output["environment"] == "exact"
    assert output["indices"] == output["distances"] == "exact"
    assert output["status"] == "accepted"
    assert reference_calls == [device]


def test_fixture_invalid_device_fails_without_fallback() -> None:
    result = _console(
        "fixture",
        "validate",
        "--fixture",
        str(_RETRIEVAL_FIXTURE),
        "--device",
        "not-a-device",
    )
    assert result.returncode == 1
    assert "device" in result.stderr.lower()
    assert "status=accepted" not in result.stdout


def test_missing_runtime_arguments_use_argparse_status_two() -> None:
    result = _console("evaluate")
    assert result.returncode == 2
    assert "required" in result.stderr


def test_console_entry_point_is_registered() -> None:
    entries = [
        entry
        for entry in metadata.entry_points(group="console_scripts")
        if entry.name == "inspectrt"
    ]
    assert len(entries) == 1
    assert entries[0].value == "inspectrt.cli:main"


@pytest.fixture
def wired_command(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    import numpy as np
    import torch
    from torchvision.models import ResNet50_Weights

    import inspectrt.artifacts as artifacts
    import inspectrt.evaluation as evaluation_module
    import inspectrt.features as features

    output_root = tmp_path / "outputs"
    dataset_root = tmp_path / "mvtec_ad"
    cached = tmp_path / "hub" / "checkpoints" / "resnet50-11ad3fa6.pth"
    cached.parent.mkdir(parents=True)
    weight_bytes = b"exact cached official weight bytes"
    cached.write_bytes(weight_bytes)
    result = SimpleNamespace(
        category="bottle",
        metrics=SimpleNamespace(
            image_auroc=0.75,
            image_average_precision=0.5,
            pixel_auroc=0.625,
        ),
    )
    calls: dict[str, object] = {}
    extractor = object()

    def build(*, weights: object) -> object:
        calls["weights"] = weights
        return extractor

    def evaluate(*args: object, **kwargs: object) -> object:
        calls["evaluate"] = (args, kwargs)
        return result

    def persist(*args: object) -> Path:
        calls["persist"] = args
        return output_root / "runs" / args[2].run_id

    monkeypatch.setattr(features, "build_resnet50_layer2_extractor", build)
    monkeypatch.setattr(evaluation_module, "evaluate_mvtec_category", evaluate)
    monkeypatch.setattr(artifacts, "persist_baseline_run", persist)
    monkeypatch.setattr(torch.hub, "get_dir", lambda: str(tmp_path / "hub"))
    monkeypatch.setattr(
        cli,
        "_repository_metadata",
        lambda cwd: (tmp_path, "a" * 40, True, "b" * 64),
    )
    monkeypatch.setattr(
        cli, "_utc_now", lambda: datetime(2026, 7, 15, 14, 30, 12, 123456, timezone.utc)
    )
    queried: list[str] = []

    def version(name: str) -> str:
        queried.append(name)
        return f"version-{name}"

    monkeypatch.setattr(cli.importlib_metadata, "version", version)
    monkeypatch.setattr(cli.platform, "python_version", lambda: "3.11.test")
    monkeypatch.setattr(cli.platform, "platform", lambda: "Test-Platform")
    seeds: list[tuple[str, int]] = []
    monkeypatch.setattr(cli.random, "seed", lambda seed: seeds.append(("python", seed)))
    monkeypatch.setattr(np.random, "seed", lambda seed: seeds.append(("numpy", seed)))
    monkeypatch.setattr(
        torch, "manual_seed", lambda seed: seeds.append(("torch", seed))
    )
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(
        torch.cuda, "manual_seed_all", lambda seed: seeds.append(("cuda", seed))
    )
    deterministic: list[tuple[bool, bool]] = []
    monkeypatch.setattr(
        torch,
        "use_deterministic_algorithms",
        lambda enabled, *, warn_only: deterministic.append((enabled, warn_only)),
    )
    monkeypatch.setattr(torch, "are_deterministic_algorithms_enabled", lambda: True)
    monkeypatch.setattr(
        torch, "is_deterministic_algorithms_warn_only_enabled", lambda: False
    )
    monkeypatch.setattr(torch.backends.cudnn, "benchmark", True)
    monkeypatch.setattr(torch.backends, "fp32_precision", torch.backends.fp32_precision)
    monkeypatch.delenv("CUBLAS_WORKSPACE_CONFIG", raising=False)

    def run(*extra: str, device: str = "cpu") -> int:
        return cli.main(
            [
                "evaluate",
                "--config",
                str(_PROFILE),
                "--dataset-root",
                str(dataset_root),
                "--category",
                "bottle",
                "--device",
                device,
                "--output-root",
                str(output_root),
                *extra,
            ]
        )

    return SimpleNamespace(
        run=run,
        calls=calls,
        result=result,
        extractor=extractor,
        output_root=output_root,
        dataset_root=dataset_root,
        weight_bytes=weight_bytes,
        weights=ResNet50_Weights.IMAGENET1K_V2,
        queried=queried,
        seeds=seeds,
        deterministic=deterministic,
        evaluation_module=evaluation_module,
        artifacts=artifacts,
        torch=torch,
    )


def test_success_wires_complete_run_and_truthful_metadata(
    wired_command: SimpleNamespace, capsys: pytest.CaptureFixture[str]
) -> None:
    command = wired_command
    assert command.run("--run-id", "bottle-local-check") == 0
    evaluate_args, evaluate_kwargs = command.calls["evaluate"]
    assert evaluate_args == (
        command.dataset_root,
        "bottle",
        command.extractor,
    )
    assert evaluate_kwargs == {
        "device": command.torch.device("cpu"),
        "bank_chunk_size": 16384,
    }
    evaluation, output_root, run_metadata = command.calls["persist"]
    assert evaluation is command.result
    assert output_root == command.output_root
    assert run_metadata.run_id == "bottle-local-check"
    assert run_metadata.created_at_utc == "2026-07-15T14:30:12.123456Z"
    assert run_metadata.dataset_root == str(command.dataset_root.resolve())
    assert run_metadata.requested_device == "cpu"
    assert (run_metadata.git_commit, run_metadata.git_dirty) == ("a" * 40, True)
    assert run_metadata.uv_lock_sha256 == "b" * 64
    assert run_metadata.python_version == "3.11.test"
    assert run_metadata.platform_description == "Test-Platform"
    assert command.queried == [
        "inspectrt",
        "numpy",
        "pillow",
        "torch",
        "torchvision",
        "scikit-learn",
    ]
    assert dict(run_metadata.dependency_versions) == {
        name: f"version-{name}" for name in command.queried
    }
    assert command.calls["weights"] is command.weights
    assert run_metadata.weight_enum == "ResNet50_Weights.IMAGENET1K_V2"
    assert run_metadata.weight_source_url == command.weights.url
    assert (
        run_metadata.weight_file_sha256
        == hashlib.sha256(command.weight_bytes).hexdigest()
    )
    assert command.seeds == [
        ("python", 0),
        ("numpy", 0),
        ("torch", 0),
        ("cuda", 0),
    ]
    assert command.deterministic == [(True, False)]
    assert command.torch.backends.cudnn.benchmark is False
    assert command.torch.backends.fp32_precision == "ieee"
    assert dict(run_metadata.determinism_flags) == {
        "python_random_seed": 0,
        "numpy_seed": 0,
        "torch_cpu_seed": 0,
        "torch_cuda_seed_all": 0,
        "use_deterministic_algorithms": True,
        "deterministic_algorithms_warn_only": False,
        "cudnn_benchmark": False,
        "allow_tf32": False,
        "fp32_precision": "ieee",
        "cublas_workspace_config": ":4096:8",
    }
    assert not hasattr(run_metadata, "timing")
    assert not command.output_root.exists()
    assert not (command.output_root / "runs" / "benchmark.json").exists()
    assert capsys.readouterr().out.splitlines() == [
        f"Run written to {command.output_root}/runs/bottle-local-check",
        "category=bottle",
        "image_auroc=0.75",
        "image_average_precision=0.5",
        "pixel_auroc=0.625",
    ]


def test_omitted_run_id_generates_a_safe_deterministic_component(
    wired_command: SimpleNamespace,
) -> None:
    assert wired_command.run() == 0
    run_id = wired_command.calls["persist"][2].run_id
    assert run_id == "20260715T143012123456Z-bottle-aaaaaaa"
    assert re.fullmatch(r"[A-Za-z0-9._-]+", run_id)


@pytest.mark.parametrize(
    ("device", "message"),
    [
        ("cuda", "must include an explicit index"),
        ("cuda:0", "requested but unavailable"),
    ],
)
def test_cuda_request_never_falls_back_to_cpu(
    wired_command: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    device: str,
    message: str,
) -> None:
    monkeypatch.setattr(wired_command.torch.cuda, "is_available", lambda: False)
    assert wired_command.run(device=device) == 1
    assert message in capsys.readouterr().err
    assert "evaluate" not in wired_command.calls


def _git(cwd: Path, *arguments: str) -> str:
    return subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    ).stdout.strip()


def test_repository_commit_dirty_state_and_exact_lock_digest(tmp_path: Path) -> None:
    _git(tmp_path, "init", "--quiet")
    _git(tmp_path, "config", "user.email", "test@example.com")
    _git(tmp_path, "config", "user.name", "Test User")
    lock_bytes = b"exact lock bytes\x00\n"
    (tmp_path / "uv.lock").write_bytes(lock_bytes)
    (tmp_path / ".gitignore").write_text("outputs/\n_extra/\n", encoding="utf-8")
    tracked = tmp_path / "tracked.txt"
    tracked.write_text("clean\n", encoding="utf-8")
    _git(tmp_path, "add", ".")
    _git(tmp_path, "commit", "--quiet", "-m", "fixture")
    nested = tmp_path / "nested" / "directory"
    nested.mkdir(parents=True)
    (tmp_path / "outputs").mkdir()
    (tmp_path / "outputs" / "ignored").write_text("ignored", encoding="utf-8")
    (tmp_path / "_extra").mkdir()
    (tmp_path / "_extra" / "ignored").write_text("ignored", encoding="utf-8")

    root, commit, dirty, digest = cli._repository_metadata(nested)
    assert root == tmp_path.resolve()
    assert commit == _git(tmp_path, "rev-parse", "HEAD")
    assert dirty is False
    assert digest == hashlib.sha256(lock_bytes).hexdigest()

    tracked.write_text("dirty\n", encoding="utf-8")
    assert cli._repository_metadata(nested)[2] is True


def test_repository_metadata_fails_outside_git_and_without_lock(tmp_path: Path) -> None:
    with pytest.raises(RuntimeError, match="Git metadata command failed"):
        cli._repository_metadata(tmp_path)
    repository = tmp_path / "repository"
    repository.mkdir()
    _git(repository, "init", "--quiet")
    _git(repository, "config", "user.email", "test@example.com")
    _git(repository, "config", "user.name", "Test User")
    (repository / "tracked").write_text("data", encoding="utf-8")
    _git(repository, "add", ".")
    _git(repository, "commit", "--quiet", "-m", "fixture")
    with pytest.raises(FileNotFoundError, match="uv.lock not found"):
        cli._repository_metadata(repository)


def test_evaluation_failure_is_concise_and_does_not_persist(
    wired_command: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: object, **kwargs: object) -> None:
        raise ValueError("invalid dataset structure")

    monkeypatch.setattr(
        wired_command.evaluation_module, "evaluate_mvtec_category", fail
    )
    assert wired_command.run("--run-id", "failed") == 1
    captured = capsys.readouterr()
    assert "invalid dataset structure" in captured.err
    assert "Traceback" not in captured.err
    assert "persist" not in wired_command.calls
    assert not wired_command.output_root.exists()


def test_existing_run_id_is_reported_without_overwrite(
    wired_command: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def fail(*args: object) -> None:
        raise FileExistsError("Run directory already exists: existing")

    monkeypatch.setattr(wired_command.artifacts, "persist_baseline_run", fail)
    assert wired_command.run("--run-id", "existing") == 1
    captured = capsys.readouterr()
    assert "already exists" in captured.err
    assert "Traceback" not in captured.err
    assert not wired_command.output_root.exists()
