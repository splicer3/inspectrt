from datetime import datetime, timezone
import hashlib
from pathlib import Path
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


def _installed_source_error() -> None:
    raise RuntimeError(
        "this command requires an InspectRT source checkout with Git and "
        "uv.lock provenance"
    )


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


def test_baseline_profile_accepts_contract_and_rejects_invalid_classes(
    tmp_path: Path,
) -> None:
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
    bundled, bundled_digest = cli._runtime_baseline(None)
    assert bundled == config
    assert bundled_digest == hashlib.sha256(_PROFILE.read_bytes()).hexdigest()
    explicit = tmp_path / "explicit-baseline.toml"
    explicit.write_bytes(_PROFILE.read_bytes() + b"\n")
    explicit_config, explicit_digest = cli._runtime_baseline(explicit)
    assert explicit_config == config
    assert explicit_digest == hashlib.sha256(explicit.read_bytes()).hexdigest()
    with _PROFILE.open("rb") as stream:
        raw = tomllib.load(stream)
    assert set(raw) == _TOP_LEVEL_KEYS
    assert set(raw["determinism"]) == _DETERMINISM_KEYS
    assert not (
        {"dataset_root", "category", "device", "output_root", "run_id"} & raw.keys()
    )
    scenarios = (
        ("missing", "bank_chunk_size = 16384\n", "", "missing.*bank_chunk_size"),
        (
            "unknown",
            "[determinism]",
            "unknown = 1\n\n[determinism]",
            "profile.*unknown",
        ),
        (
            "wrong-type",
            "bank_chunk_size = 16384",
            "bank_chunk_size = true",
            "positive integer",
        ),
        (
            "unsupported",
            'weights = "IMAGENET1K_V2"',
            'weights = "DEFAULT"',
            "weights",
        ),
    )
    for scenario, old, new, message in scenarios:
        (tmp_path / scenario).mkdir()
        with pytest.raises((TypeError, ValueError), match=message) as raised:
            cli.load_baseline_config(_modified_profile(tmp_path / scenario, old, new))
        assert raised.value, scenario


def test_command_groups_expose_the_documented_parser_surface() -> None:
    root = _console("--help")
    evaluate = _console("evaluate", "--help")
    benchmark = _console("benchmark", "--help")
    group = _console("portability", "--help")
    compare = _console("portability", "compare", "--help")
    performance = _console("portability", "performance", "--help")
    assert root.returncode == group.returncode == compare.returncode == 0
    assert performance.returncode == 0
    assert "portability" in root.stdout
    assert "{compare,performance}" in group.stdout
    for argument in (
        "--reference-run",
        "--candidate-run",
        "--environment-map",
        "--policy",
        "--output",
    ):
        assert argument in compare.stdout
    for argument in (
        "--scientific",
        "--policy",
        "--environment-map",
        "--timing-run",
        "--output",
    ):
        assert argument in performance.stdout
    graph = _console("portability", "graph")
    assert graph.returncode == 2
    assert "invalid choice" in graph.stderr
    fixture_group = _console("fixture", "--help")
    fixture_validate = _console("fixture", "validate", "--help")
    fixture_export = _console("fixture", "export", "--help")
    assert fixture_group.returncode == fixture_validate.returncode == 0
    assert fixture_export.returncode == 0
    assert "{validate,export}" in fixture_group.stdout
    assert "[--fixture FIXTURE]" in fixture_validate.stdout
    assert "fixture directory (default: bundled synthetic-" in fixture_validate.stdout
    assert "correctness-v1)" in fixture_validate.stdout
    assert "[--config CONFIG]" in evaluate.stdout
    assert "[--config CONFIG]" in benchmark.stdout
    assert "baseline TOML profile (default: bundled" in evaluate.stdout
    assert "inspectrt_feature_memory_v1)" in evaluate.stdout
    assert all(
        argument in fixture_export.stdout
        for argument in (
            "--config",
            "--run-dir",
            "--dataset-root",
            "--sample-id",
            "--device",
            "--output-root",
        )
    )


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
        lambda: (_ROOT, "a" * 40, True, "b" * 64),
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


def test_portability_routes_candidates_in_explicit_order(
    wired_portability: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    command = wired_portability
    count = 3
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

    command.publisher.side_effect = ValueError("candidate contract mismatch")
    assert command.run(*candidates) == 1
    captured = capsys.readouterr()
    assert "candidate contract mismatch" in captured.err
    assert "Traceback" not in captured.err
    command.publisher.reset_mock()
    monkeypatch.setattr(cli, "_repository_metadata", _installed_source_error)
    assert command.run(*candidates) == 1
    assert "source checkout with Git and uv.lock provenance" in capsys.readouterr().err
    command.publisher.assert_not_called()


def test_portability_performance_routes_six_runs_and_has_one_error_boundary(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspectrt.portability as portability

    paths = tuple(tmp_path / f"run-{index}" for index in range(6))
    scientific_path = tmp_path / "scientific.json"
    policy_path = tmp_path / "policy.json"
    environment_path = tmp_path / "environment.json"
    output = tmp_path / "performance_v2.json"
    scientific = {"scientific": "identity"}
    policy = object()
    environment = object()
    bundles = tuple(object() for _ in paths)
    performance = {"performance_id": "d" * 64}
    payload = b'{"performance_id":"' + b"d" * 64 + b'"}\n'
    load_scientific = Mock(return_value=scientific)
    load_policy = Mock(return_value=policy)
    load_environment = Mock(return_value=environment)
    load_bundle = Mock(side_effect=bundles)
    build = Mock(return_value=performance)
    encode = Mock(return_value=payload)
    publish = Mock()
    monkeypatch.setattr(
        portability, "load_portability_scientific_identity", load_scientific
    )
    monkeypatch.setattr(portability, "load_portability_policy", load_policy)
    monkeypatch.setattr(
        portability, "load_portability_environment_map", load_environment
    )
    monkeypatch.setattr(portability, "load_timing_bundle", load_bundle)
    monkeypatch.setattr(portability, "build_portability_performance_v2", build)
    monkeypatch.setattr(portability, "encode_portability_performance_v2", encode)
    monkeypatch.setattr(portability, "publish_portability_performance_v2", publish)
    monkeypatch.setattr(
        cli,
        "_repository_metadata",
        lambda: (_ROOT, "a" * 40, True, "b" * 64),
    )
    arguments = [
        "portability",
        "performance",
        "--scientific",
        str(scientific_path),
        "--policy",
        str(policy_path),
        "--environment-map",
        str(environment_path),
    ]
    for path in paths:
        arguments.extend(("--timing-run", str(path)))
    arguments.extend(("--output", str(output)))

    assert cli.main(arguments) == 0
    assert [call.args[0] for call in load_bundle.call_args_list] == list(paths)
    build.assert_called_once_with(
        scientific,
        policy,
        environment,
        bundles,
        generator=portability.ScientificGenerator("a" * 40, True),
    )
    publish.assert_called_once_with(payload, output)
    assert capsys.readouterr().out.splitlines() == [
        f"performance_id={'d' * 64}",
        "runs=6",
        f"byte_count={len(payload)}",
        f"sha256={hashlib.sha256(payload).hexdigest()}",
        "status=published",
    ]

    inside_bundle = [*arguments[:-1], str(paths[0] / "performance_v2.json")]
    assert cli.main(inside_bundle) == 1
    assert capsys.readouterr().err == (
        "inspectrt portability performance failed: "
        "--output must be outside source timing bundles\n"
    )

    load_scientific.side_effect = ValueError("scientific identity mismatch")
    assert cli.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "inspectrt portability performance failed: scientific identity mismatch\n"
    )
    load_scientific.reset_mock()
    monkeypatch.setattr(cli, "_repository_metadata", _installed_source_error)
    assert cli.main(arguments) == 1
    assert "source checkout with Git and uv.lock provenance" in capsys.readouterr().err
    load_scientific.assert_not_called()


def test_onnx_help_action_arguments_and_import_isolation() -> None:
    root = _console("--help")
    group = _console("onnx", "--help")
    export = _console("onnx", "export", "--help")
    validate = _console("onnx", "validate", "--help")
    assert root.returncode == group.returncode == export.returncode == 0
    assert validate.returncode == 0
    assert "onnx" in root.stdout
    assert "{export,validate}" in group.stdout
    assert "--output-root" in export.stdout
    assert "--artifact" in validate.stdout
    for arguments, required in (
        (("onnx", "export"), "--output-root"),
        (("onnx", "validate"), "--artifact"),
    ):
        result = _console(*arguments)
        assert result.returncode == 2
        assert required in result.stderr
    invalid = _console("onnx", "run")
    assert invalid.returncode == 2
    assert "invalid choice" in invalid.stderr

    code = """
import sys
import inspectrt.cli as cli
for arguments in (
    ["onnx", "--help"],
    ["onnx", "export", "--help"],
    ["onnx", "validate", "--help"],
):
    try:
        cli.main(arguments)
    except SystemExit as error:
        assert error.code == 0
    else:
        raise AssertionError("help did not exit")
for name in (
    "numpy",
    "torch",
    "torchvision",
    "onnx",
    "onnxscript",
    "onnxruntime",
    "inspectrt.onnx_artifacts",
    "inspectrt.onnx_features",
):
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


def test_onnx_export_dirty_gate_routing_and_exact_output(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspectrt.onnx_artifacts as onnx_artifacts

    artifact = onnx_artifacts.LoadedOnnxFeatureArtifact(
        "resnet50-layer2-opset20-143b305b37a9",
        5_857_483,
        "a" * 64,
        "b" * 64,
        20,
    )
    destination = tmp_path / artifact.artifact_id

    def publish(*args: object, **kwargs: object) -> tuple[Path, object]:
        print("exporter progress")
        return destination, artifact

    publisher = Mock(side_effect=publish)
    monkeypatch.setattr(
        onnx_artifacts,
        "publish_onnx_feature_artifact",
        publisher,
    )
    monkeypatch.setattr(
        cli,
        "_repository_metadata",
        lambda: (_ROOT, "c" * 40, False, "d" * 64),
    )
    arguments = ["onnx", "export", "--output-root", str(tmp_path)]
    assert cli.main(arguments) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"artifact_id={artifact.artifact_id}",
        f"artifact_path={destination}",
        "model_bytes=5857483",
        f"model_sha256={'a' * 64}",
        f"artifact_digest={'b' * 64}",
        "status=published",
    ]
    assert captured.err == "exporter progress\n"
    publisher.assert_called_once_with(
        tmp_path,
        git_commit="c" * 40,
        git_dirty=False,
        uv_lock_sha256="d" * 64,
    )

    publisher.reset_mock()
    monkeypatch.setattr(
        cli,
        "_repository_metadata",
        lambda: (_ROOT, "c" * 40, True, "d" * 64),
    )
    assert cli.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "inspectrt onnx export failed: "
        "ONNX artifact source working tree must be clean\n"
    )
    publisher.assert_not_called()
    monkeypatch.setattr(cli, "_repository_metadata", _installed_source_error)
    assert cli.main(arguments) == 1
    assert "source checkout with Git and uv.lock provenance" in capsys.readouterr().err
    publisher.assert_not_called()
    _assert_onnx_validate_routing(tmp_path, monkeypatch, capsys)


def _assert_onnx_validate_routing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    import inspectrt.onnx_artifacts as onnx_artifacts

    artifact = onnx_artifacts.LoadedOnnxFeatureArtifact(
        "resnet50-layer2-opset20-143b305b37a9",
        5_857_483,
        "a" * 64,
        "b" * 64,
        20,
    )
    path = tmp_path / "artifact"

    def load(value: Path) -> object:
        print("checker progress")
        assert value == path
        return artifact

    loader = Mock(side_effect=load)
    monkeypatch.setattr(onnx_artifacts, "load_onnx_feature_artifact", loader)
    arguments = ["onnx", "validate", "--artifact", str(path)]
    assert cli.main(arguments) == 0
    captured = capsys.readouterr()
    assert captured.out.splitlines() == [
        f"artifact_id={artifact.artifact_id}",
        "model_bytes=5857483",
        f"model_sha256={'a' * 64}",
        f"artifact_digest={'b' * 64}",
        "opset=20",
        "status=valid",
    ]
    assert captured.err == "checker progress\n"

    loader.side_effect = RuntimeError(
        "install inspectrt[onnx] to use ONNX artifact commands"
    )
    assert cli.main(arguments) == 1
    captured = capsys.readouterr()
    assert captured.out == ""
    assert captured.err == (
        "inspectrt onnx validate failed: "
        "install inspectrt[onnx] to use ONNX artifact commands\n"
    )


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
    real_export = cli._export_fixture

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
    monkeypatch.setattr(
        cli,
        "_export_fixture",
        lambda *args: (_ for _ in ()).throw(ValueError("fixture contract mismatch")),
    )
    assert cli.main(_export_arguments()) == 1
    captured = capsys.readouterr()
    assert "fixture contract mismatch" in captured.err
    assert "Traceback" not in captured.err
    monkeypatch.setattr(cli, "_export_fixture", real_export)
    monkeypatch.setattr(cli, "_repository_metadata", _installed_source_error)
    assert cli.main(_export_arguments()) == 1
    assert "source checkout with Git and uv.lock provenance" in capsys.readouterr().err


def test_fixture_validate_accepts_exact_cpu_and_reports_device_errors() -> None:
    bundled = _console("fixture", "validate", "--device", "cpu")
    accepted = _console(
        "fixture",
        "validate",
        "--fixture",
        str(_RETRIEVAL_FIXTURE),
        "--device",
        "cpu",
    )
    assert bundled.returncode == 0, bundled.stderr
    assert accepted.returncode == 0, accepted.stderr
    assert bundled.stdout == accepted.stdout
    output = dict(line.split("=", 1) for line in accepted.stdout.splitlines())
    assert output["fixture_id"] == "synthetic-correctness-v1"
    assert (
        output["fixture_digest"]
        == "ec30a68439f52051028a56cbd5a1c560edc2bccc4e77e603fa2d3355a26a4e9e"
    )
    assert output["fixture_class"] == "synthetic_correctness"
    assert output["indices"] == output["distances"] == "exact"
    assert output["status"] == "accepted"

    rejected = _console(
        "fixture",
        "validate",
        "--fixture",
        str(_RETRIEVAL_FIXTURE),
        "--device",
        "cuda",
    )
    assert rejected.returncode == 1
    assert "explicit index" in rejected.stderr
    assert "Traceback" not in rejected.stderr


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
        lambda: (tmp_path, "a" * 40, True, "b" * 64),
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

    def run(*extra: str, device: str = "cpu", config_path: Path | None = None) -> int:
        arguments = ["evaluate"]
        if config_path is not None:
            arguments.extend(("--config", str(config_path)))
        arguments.extend(
            (
                "--dataset-root",
                str(dataset_root),
                "--category",
                "bottle",
                "--device",
                device,
                "--output-root",
                str(output_root),
                *extra,
            )
        )
        return cli.main(arguments)

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
    wired_command: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
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
    assert run_metadata.source_kind == "repository"
    assert run_metadata.distribution_name is None
    assert run_metadata.baseline_profile_sha256 is None
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
    command.calls.clear()
    assert command.run("--run-id", "explicit-profile", config_path=_PROFILE) == 0
    assert command.calls["persist"][2].run_id == "explicit-profile"
    capsys.readouterr()
    command.calls.clear()
    monkeypatch.setattr(
        command.evaluation_module,
        "evaluate_mvtec_category",
        lambda *args, **kwargs: (_ for _ in ()).throw(
            ValueError("invalid dataset structure")
        ),
    )
    assert command.run("--run-id", "failed") == 1
    captured = capsys.readouterr()
    assert "invalid dataset structure" in captured.err
    assert "Traceback" not in captured.err
    assert "persist" not in command.calls


def _benchmark_arguments(
    device: str, *extra: str, explicit_config: bool = True
) -> list[str]:
    arguments = ["benchmark"]
    if explicit_config:
        arguments.extend(("--config", str(_PROFILE)))
    arguments.extend(
        [
            "--dataset-root",
            "dataset",
            "--category",
            "bottle",
            "--device",
            device,
            "--output-root",
            "outputs",
            "--run-id",
            "timing-v2",
            *extra,
        ]
    )
    return arguments


def test_benchmark_routes_cpu_indexed_cuda_and_mps_with_exact_load_nanoseconds(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    import torch

    import inspectrt.artifacts as artifacts
    import inspectrt.benchmark as benchmark_module
    import inspectrt.features as features

    calls: list[tuple[str, object]] = []

    class Extractor:
        def to(self, device: object) -> object:
            calls.append(("placement", str(device)))
            return self

        def eval(self) -> object:
            calls.append(("evaluation_mode", True))
            return self

    evaluation = SimpleNamespace(
        category="bottle",
        metrics=SimpleNamespace(
            image_auroc=0.75,
            image_average_precision=0.5,
            pixel_auroc=0.625,
        ),
    )
    record = SimpleNamespace(
        device="cpu",
        results={
            "synchronized_end_to_end": {
                "summary_ns": {"p50": 2_500_000.0, "p95": 4_000_000.0}
            }
        },
    )

    monkeypatch.delenv("PYTORCH_ENABLE_MPS_FALLBACK", raising=False)
    monkeypatch.setattr(cli, "_configure_determinism", lambda *args: {})
    monkeypatch.setattr(
        cli, "_resolve_device", lambda requested, runtime: torch.device(requested)
    )
    monkeypatch.setattr(
        cli,
        "_baseline_run_identity",
        lambda arguments, profile_digest: {
            "run_id": arguments.run_id,
            "created_at_utc": "now",
        },
    )
    monkeypatch.setattr(
        cli,
        "_baseline_run_metadata",
        lambda arguments, config, device, determinism, weights, runtime, identity, *, weight_digest: (
            SimpleNamespace(run_id=identity["run_id"], created_at_utc="now")
        ),
    )
    monkeypatch.setattr(
        cli,
        "_cached_weight_sha256",
        lambda url, runtime: calls.append(("cached_weight", url)) or "f" * 64,
    )

    def build(**kwargs: object) -> Extractor:
        calls.append(("model_construction", kwargs["weights"]))
        return Extractor()

    monkeypatch.setattr(features, "build_resnet50_layer2_extractor", build)

    def time_operation(device: object, operation: object) -> tuple[object, int]:
        calls.append(("timing_device", str(device)))
        calls.append(("timer", "start"))
        result = operation()  # type: ignore[operator]
        calls.append(("timer", "end"))
        return result, 987_654_321

    monkeypatch.setattr(benchmark_module, "_time_backend_operation", time_operation)

    def benchmark(*args: object, **kwargs: object) -> tuple[object, object]:
        calls.append(("benchmark", kwargs.copy()))
        record.device = str(kwargs["device"])
        return evaluation, record

    monkeypatch.setattr(benchmark_module, "benchmark_mvtec_category", benchmark)
    monkeypatch.setattr(
        artifacts,
        "persist_baseline_run",
        lambda evaluation, output, metadata, *, benchmark: (
            output / "runs" / metadata.run_id
        ),
    )

    for device in ("cpu", "cuda:2", "mps"):
        calls.clear()
        assert (
            cli.main(_benchmark_arguments(device, explicit_config=device != "cpu")) == 0
        )
        benchmark_call = next(value for name, value in calls if name == "benchmark")
        assert benchmark_call["device"] == torch.device(device)
        assert benchmark_call["model_and_weight_load_ns"] == 987_654_321
        assert ("timing_device", device) in calls
        assert ("placement", device) in calls
        assert [name for name, _ in calls[1:6]] == [
            "timer",
            "cached_weight",
            "model_construction",
            "placement",
            "evaluation_mode",
        ]
        assert calls[6] == ("timer", "end")
        output = capsys.readouterr().out
        assert "synchronized_end_to_end_p50_ms=2.5" in output
        assert f"timing_device={device}" in output

    monkeypatch.setattr(
        cli,
        "_resolve_device",
        lambda *args: (_ for _ in ()).throw(RuntimeError("accelerator unavailable")),
    )
    assert cli.main(_benchmark_arguments("cuda:2")) == 1
    captured = capsys.readouterr()
    assert "accelerator unavailable" in captured.err
    assert "Traceback" not in captured.err
