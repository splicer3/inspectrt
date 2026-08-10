"""Command-line entry point for the InspectRT baseline."""

import argparse
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
from importlib import metadata as importlib_metadata
import os
from pathlib import Path
import platform
import random
import re
import subprocess
import sys
import tomllib
from typing import Any
from urllib.parse import urlsplit

_PROFILE_ID = "inspectrt_feature_memory_v1"
_PREPROCESSING_PROFILE_ID = "inspectrt_resize256_v1"
_WEIGHT_ID = "IMAGENET1K_V2"
_CUBLAS_WORKSPACE_CONFIG = ":4096:8"
_PROFILE_KEYS = {
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
_DISTRIBUTIONS = (
    "inspectrt",
    "numpy",
    "pillow",
    "torch",
    "torchvision",
    "scikit-learn",
)


@dataclass(frozen=True, slots=True)
class BaselineConfig:
    """Validated values in the single committed baseline profile."""

    schema_version: int
    profile_id: str
    preprocessing_profile_id: str
    weights: str
    bank_chunk_size: int
    seed: int
    use_deterministic_algorithms: bool
    cudnn_benchmark: bool
    allow_tf32: bool
    cublas_workspace_config: str


def load_baseline_config(path: Path) -> BaselineConfig:
    """Load and strictly validate the supported baseline TOML profile."""
    with path.open("rb") as stream:
        profile = tomllib.load(stream)
    _validate_keys(profile, _PROFILE_KEYS, "profile")
    determinism = profile["determinism"]
    if not isinstance(determinism, Mapping):
        raise ValueError("determinism must be a TOML table")
    _validate_keys(determinism, _DETERMINISM_KEYS, "determinism")

    _require_fixed(profile, "schema_version", 1)
    _require_fixed(profile, "profile_id", _PROFILE_ID)
    _require_fixed(profile, "preprocessing_profile_id", _PREPROCESSING_PROFILE_ID)
    _require_fixed(profile, "weights", _WEIGHT_ID)
    chunk_size = profile["bank_chunk_size"]
    if type(chunk_size) is not int or chunk_size <= 0:
        raise ValueError("bank_chunk_size must be a positive integer")
    seed = profile["seed"]
    if type(seed) is not int or seed < 0:
        raise ValueError("seed must be a nonnegative integer")
    _require_fixed(determinism, "use_deterministic_algorithms", True)
    _require_fixed(determinism, "cudnn_benchmark", False)
    _require_fixed(determinism, "allow_tf32", False)
    _require_fixed(determinism, "cublas_workspace_config", _CUBLAS_WORKSPACE_CONFIG)
    return BaselineConfig(
        profile["schema_version"],
        profile["profile_id"],
        profile["preprocessing_profile_id"],
        profile["weights"],
        chunk_size,
        seed,
        determinism["use_deterministic_algorithms"],
        determinism["cudnn_benchmark"],
        determinism["allow_tf32"],
        determinism["cublas_workspace_config"],
    )


def _validate_keys(table: Mapping[str, object], expected: set[str], name: str) -> None:
    actual = set(table)
    missing = sorted(expected - actual)
    if missing:
        raise ValueError(f"{name} is missing required keys: {', '.join(missing)}")
    unknown = sorted(actual - expected)
    if unknown:
        raise ValueError(f"{name} contains unknown keys: {', '.join(unknown)}")


def _require_fixed(table: Mapping[str, object], key: str, expected: object) -> None:
    value = table[key]
    if type(value) is not type(expected) or value != expected:
        raise ValueError(f"{key} must be {expected!r}")


def _argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="inspectrt")
    commands = parser.add_subparsers(dest="command", required=True)
    evaluate = commands.add_parser(
        "evaluate", help="evaluate and persist one MVTec AD category"
    )
    _add_runtime_arguments(evaluate)
    benchmark = commands.add_parser(
        "benchmark", help="measure and persist one MVTec AD category"
    )
    _add_runtime_arguments(benchmark)
    benchmark.add_argument("--warmup-count", type=_positive_integer, default=5)
    benchmark.add_argument("--repeat-count", type=_positive_integer, default=30)
    fixture = commands.add_parser(
        "fixture", help="export or validate retrieval fixtures"
    )
    fixture_commands = fixture.add_subparsers(dest="fixture_command", required=True)
    validate = fixture_commands.add_parser(
        "validate", help="validate a retrieval fixture"
    )
    validate.add_argument("--fixture", required=True, type=Path)
    validate.add_argument("--device", required=True)
    export = fixture_commands.add_parser(
        "export", help="export an accepted benchmark run as a retrieval fixture"
    )
    export.add_argument("--config", required=True, type=Path)
    export.add_argument("--run-dir", required=True, type=Path)
    export.add_argument("--dataset-root", required=True, type=Path)
    export.add_argument("--sample-id", required=True)
    export.add_argument("--device", required=True)
    export.add_argument("--output-root", required=True, type=Path)
    portability = commands.add_parser(
        "portability", help="compare portable run bundles"
    )
    portability_commands = portability.add_subparsers(
        dest="portability_command", required=True
    )
    compare = portability_commands.add_parser(
        "compare", help="publish scientific and descriptive performance records"
    )
    compare.add_argument("--reference-run", required=True, type=Path)
    compare.add_argument("--candidate-run", required=True, action="append", type=Path)
    compare.add_argument("--environment-map", required=True, type=Path)
    compare.add_argument("--policy", type=Path)
    compare.add_argument("--output", required=True, type=Path)
    performance = portability_commands.add_parser(
        "performance", help="aggregate reviewed synchronized timing bundles"
    )
    performance.add_argument("--scientific", required=True, type=Path)
    performance.add_argument("--policy", required=True, type=Path)
    performance.add_argument("--environment-map", required=True, type=Path)
    performance.add_argument("--timing-run", required=True, action="append", type=Path)
    performance.add_argument("--output", required=True, type=Path)
    return parser


def _add_runtime_arguments(command: argparse.ArgumentParser) -> None:
    command.add_argument("--config", required=True, type=Path)
    command.add_argument("--dataset-root", required=True, type=Path)
    command.add_argument("--category", required=True)
    command.add_argument("--device", required=True)
    command.add_argument("--output-root", type=Path, default=Path("outputs"))
    command.add_argument("--run-id")


def _positive_integer(value: str) -> int:
    try:
        parsed = int(value)
    except ValueError as error:
        raise argparse.ArgumentTypeError("must be a positive integer") from error
    if parsed <= 0:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return parsed


def main(argv: Sequence[str] | None = None) -> int:
    """Run the command-line interface."""
    arguments = _argument_parser().parse_args(argv)
    try:
        if arguments.command == "portability":
            return (
                _compare_portability(arguments)
                if arguments.portability_command == "compare"
                else _performance_portability(arguments)
            )
        if arguments.command == "fixture":
            if arguments.fixture_command == "validate":
                return _validate_fixture(arguments)
            config = load_baseline_config(arguments.config)
            os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
            return _export_fixture(arguments, config)
        if arguments.command == "benchmark":
            _validate_benchmark_arguments(arguments)
        config = load_baseline_config(arguments.config)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = config.cublas_workspace_config
        handler = _benchmark if arguments.command == "benchmark" else _evaluate
        return handler(arguments, config)
    except Exception as error:
        command = arguments.command
        if command == "fixture":
            command = f"{command} {arguments.fixture_command}"
        elif command == "portability":
            command = f"{command} {arguments.portability_command}"
        print(f"inspectrt {command} failed: {error}", file=sys.stderr)
        return 1


def _compare_portability(arguments: argparse.Namespace) -> int:
    from inspectrt.portability import (
        ScientificGenerator,
        publish_portability_comparison,
    )

    _, commit, dirty, _ = _repository_metadata(Path.cwd())
    comparison, performance = publish_portability_comparison(
        reference_run=arguments.reference_run,
        candidate_runs=tuple(arguments.candidate_run),
        environment_map_path=arguments.environment_map,
        output=arguments.output,
        generator=ScientificGenerator(commit, dirty),
        policy_path=arguments.policy,
    )
    print(f"comparison_id={comparison.comparison_id}")
    print(f"candidates={len(comparison.candidates)}")
    print(f"performance_included={len(performance.included_runs)}")
    print(f"performance_excluded={len(performance.excluded_candidates)}")
    print(f"mode={'policy' if comparison.policy is not None else 'observation'}")
    print("status=published")
    return 0


def _performance_portability(arguments: argparse.Namespace) -> int:
    from inspectrt.portability import (
        ScientificGenerator,
        build_portability_performance_v2,
        encode_portability_performance_v2,
        load_portability_environment_map,
        load_portability_policy,
        load_portability_scientific_identity,
        load_timing_bundle,
        publish_portability_performance_v2,
    )

    if len(arguments.timing_run) != 6:
        raise ValueError("--timing-run must occur exactly six times")
    resolved_output = arguments.output.resolve(strict=False)
    if any(
        resolved_output.is_relative_to(path.resolve(strict=False))
        for path in arguments.timing_run
    ):
        raise ValueError("--output must be outside source timing bundles")
    _, commit, dirty, _ = _repository_metadata(Path.cwd())
    scientific = load_portability_scientific_identity(arguments.scientific)
    policy = load_portability_policy(arguments.policy)
    environment_map = load_portability_environment_map(arguments.environment_map)
    bundles = tuple(load_timing_bundle(path) for path in arguments.timing_run)
    performance = build_portability_performance_v2(
        scientific,
        policy,
        environment_map,
        bundles,
        generator=ScientificGenerator(commit, dirty),
    )
    payload = encode_portability_performance_v2(performance)
    publish_portability_performance_v2(payload, arguments.output)
    print(f"performance_id={performance['performance_id']}")
    print("runs=6")
    print(f"byte_count={len(payload)}")
    print(f"sha256={hashlib.sha256(payload).hexdigest()}")
    print("status=published")
    return 0


def _validate_fixture(arguments: argparse.Namespace) -> int:
    fixture_directory = arguments.fixture
    if not fixture_directory.exists():
        raise FileNotFoundError(f"fixture directory not found: {fixture_directory}")
    if not fixture_directory.is_dir() or fixture_directory.is_symlink():
        raise ValueError(f"fixture path must be a real directory: {fixture_directory}")

    from inspectrt.fixtures import (
        RealApplicationFixtureSource,
        basic_environment_mismatches,
        cuda_environment_mismatches,
        load_retrieval_fixture,
    )

    fixture = load_retrieval_fixture(fixture_directory)
    if fixture.metadata.fixture_class == "real_application":
        source = fixture.metadata.source
        if not isinstance(source, RealApplicationFixtureSource):
            raise ValueError("real fixture source metadata is malformed")
        _, _, _, lock_digest = _repository_metadata(Path.cwd())
        mismatches = basic_environment_mismatches(
            source,
            requested_device=arguments.device,
            current_lock_sha256=lock_digest,
            python_version=platform.python_version(),
            dependency_versions={
                name: importlib_metadata.version(name) for name in _DISTRIBUTIONS
            },
            platform_description=platform.platform(),
        )
        if mismatches:
            _print_unavailable_fixture(fixture, mismatches)
            return 0
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = source.determinism[
            "cublas_workspace_config"
        ]
        import numpy as np
        import torch

        device = _resolve_device(arguments.device, torch)
        mismatches = cuda_environment_mismatches(source, device, torch)
        if mismatches:
            _print_unavailable_fixture(fixture, mismatches)
            return 0
        determinism = _configure_determinism(
            BaselineConfig(
                1,
                source.baseline_profile,
                source.preprocessing_identity,
                "IMAGENET1K_V2",
                fixture.metadata.reference_chunk_size,
                int(source.determinism["python_random_seed"]),
                True,
                False,
                False,
                str(source.determinism["cublas_workspace_config"]),
            ),
            np,
            torch,
        )
        if dict(determinism) != dict(source.determinism):
            raise ValueError("current determinism settings differ from the fixture")
        _run_fixture_reference(fixture, device, torch)
        _print_accepted_fixture(fixture, environment="exact")
        return 0

    import torch

    device = _resolve_device(arguments.device, torch)
    if str(device) != "cpu":
        raise ValueError("canonical synthetic fixture acceptance requires device cpu")
    _run_fixture_reference(fixture, device, torch)
    _print_accepted_fixture(fixture)
    return 0


def _run_fixture_reference(fixture: Any, device: Any, torch: Any) -> None:
    from inspectrt.retrieval import exact_top1_squared_l2

    distances, indices = exact_top1_squared_l2(
        torch.from_numpy(fixture.queries).to(device),
        torch.from_numpy(fixture.memory_bank).to(device),
        bank_chunk_size=fixture.metadata.reference_chunk_size,
    )
    expected_indices = torch.from_numpy(fixture.expected_indices)
    if not torch.equal(indices.cpu(), expected_indices):
        raise ValueError("expected index mismatch")
    expected_distances = torch.from_numpy(fixture.expected_squared_l2_distances)
    if not torch.equal(distances.cpu(), expected_distances):
        raise ValueError("expected distance mismatch")


def _print_accepted_fixture(fixture: Any, *, environment: str | None = None) -> None:
    workload = fixture.manifest["workload"]
    payload = fixture.manifest["payload"]
    print(f"fixture_id={fixture.metadata.fixture_id}")
    print(f"fixture_class={fixture.metadata.fixture_class}")
    for name in ("Q", "M", "D", "k"):
        print(f"{name}={workload[name]}")
    if fixture.metadata.fixture_class == "synthetic_correctness":
        print(f"chunk_size={fixture.metadata.reference_chunk_size}")
    print(f"payload_sha256={payload['sha256']}")
    print(f"fixture_digest={fixture.fixture_digest}")
    if environment is not None:
        print(f"environment={environment}")
    print("indices=exact")
    print("distances=exact")
    print("status=accepted")


def _print_unavailable_fixture(fixture: Any, mismatches: Sequence[str]) -> None:
    payload = fixture.manifest["payload"]
    print(f"fixture_id={fixture.metadata.fixture_id}")
    print(f"fixture_class={fixture.metadata.fixture_class}")
    print(f"payload_sha256={payload['sha256']}")
    print(f"fixture_digest={fixture.fixture_digest}")
    print("status=structurally_valid")
    print("reference_status=unavailable")
    print(f"environment_mismatches={','.join(mismatches)}")


def _export_fixture(arguments: argparse.Namespace, config: BaselineConfig) -> int:
    repository_root, generator_commit, generator_dirty, lock_digest = (
        _repository_metadata(Path.cwd())
    )
    if generator_dirty:
        raise ValueError("fixture generator working tree must be clean")

    import numpy as np
    import torch

    from inspectrt.fixtures import (
        prepare_accepted_run_fixture,
        publish_accepted_run_fixture,
        reconstruct_fixture_query,
    )

    source = prepare_accepted_run_fixture(
        run_directory=arguments.run_dir,
        dataset_root=arguments.dataset_root,
        sample_id=arguments.sample_id,
        config_path=arguments.config,
        repository_root=repository_root,
        generator_commit=generator_commit,
        generator_dirty=generator_dirty,
        current_lock_sha256=lock_digest,
        torch=torch,
    )
    determinism = _configure_determinism(config, np, torch)
    if dict(determinism) != dict(source.metadata.source.determinism):
        raise ValueError("current determinism settings differ from the accepted run")
    device = _resolve_device(arguments.device, torch)
    if str(device) != source.metadata.source.requested_device:
        raise ValueError("requested device must match the accepted run device cuda:0")

    from torchvision.models import ResNet50_Weights

    from inspectrt.features import (
        build_resnet50_layer2_extractor,
    )
    from inspectrt.retrieval import exact_top1_squared_l2

    weights = ResNet50_Weights.IMAGENET1K_V2
    if str(weights) != source.metadata.source.weight_enum:
        raise ValueError("resolved weight enum differs from the accepted run")
    extractor = build_resnet50_layer2_extractor(weights=weights).to(device).eval()
    if (
        _cached_weight_sha256(weights.url, torch)
        != source.metadata.source.weight_file_sha256
    ):
        raise ValueError("cached official weight changed during model construction")
    queries_device = reconstruct_fixture_query(source.image_path, extractor, device)
    bank_device = source.memory_bank.to(device)
    distances, indices = exact_top1_squared_l2(
        queries_device,
        bank_device,
        bank_chunk_size=config.bank_chunk_size,
    )
    queries = queries_device.cpu().contiguous()
    distances = distances.cpu().contiguous()
    indices = indices.cpu().contiguous()
    destination, loaded = publish_accepted_run_fixture(
        source, queries, distances, indices, arguments.output_root
    )
    workload = loaded.manifest["workload"]
    payload = loaded.manifest["payload"]
    print(f"fixture_id={loaded.metadata.fixture_id}")
    print(f"fixture_class={loaded.metadata.fixture_class}")
    print(f"fixture_path={destination}")
    print(f"source_run={source.metadata.source.accepted_run_id}")
    print(f"sample_id={source.metadata.source.sample_id}")
    for name in ("Q", "M", "D", "k"):
        print(f"{name}={workload[name]}")
    print(f"payload_bytes={payload['nbytes']}")
    print(f"payload_sha256={payload['sha256']}")
    print(f"fixture_digest={loaded.fixture_digest}")
    print("indices=exact")
    print("distances=exact")
    print("status=accepted")
    return 0


def _evaluate(arguments: argparse.Namespace, config: BaselineConfig) -> int:
    import numpy as np
    import torch

    determinism = _configure_determinism(config, np, torch)
    device = _resolve_device(arguments.device, torch)

    from torchvision.models import ResNet50_Weights

    from inspectrt.artifacts import persist_baseline_run
    from inspectrt.evaluation import evaluate_mvtec_category
    from inspectrt.features import build_resnet50_layer2_extractor

    identity = _baseline_run_identity(arguments)
    weights = ResNet50_Weights.IMAGENET1K_V2
    extractor = build_resnet50_layer2_extractor(weights=weights)
    weight_digest = _cached_weight_sha256(weights.url, torch)
    evaluation = evaluate_mvtec_category(
        arguments.dataset_root,
        arguments.category,
        extractor,
        device=device,
        bank_chunk_size=config.bank_chunk_size,
    )
    metadata = _baseline_run_metadata(
        arguments,
        config,
        device,
        determinism,
        weights,
        torch,
        identity,
        weight_digest=weight_digest,
    )
    run_directory = persist_baseline_run(evaluation, arguments.output_root, metadata)
    print(f"Run written to {run_directory}")
    print(f"category={evaluation.category}")
    print(f"image_auroc={evaluation.metrics.image_auroc}")
    print(f"image_average_precision={evaluation.metrics.image_average_precision}")
    print(f"pixel_auroc={evaluation.metrics.pixel_auroc}")
    return 0


def _benchmark(arguments: argparse.Namespace, config: BaselineConfig) -> int:
    import numpy as np
    import torch

    determinism = _configure_determinism(config, np, torch)
    device = _resolve_device(arguments.device, torch)

    from torchvision.models import ResNet50_Weights

    from inspectrt.artifacts import persist_baseline_run
    from inspectrt.benchmark import _time_backend_operation, benchmark_mvtec_category
    from inspectrt.features import build_resnet50_layer2_extractor

    identity = _baseline_run_identity(arguments)

    def load_model() -> tuple[Any, Any, str]:
        weights = ResNet50_Weights.IMAGENET1K_V2
        weight_digest = _cached_weight_sha256(weights.url, torch)
        extractor = build_resnet50_layer2_extractor(weights=weights)
        return weights, extractor.to(device).eval(), weight_digest

    (weights, extractor, weight_digest), model_load_ns = _time_backend_operation(
        device, load_model
    )

    metadata = _baseline_run_metadata(
        arguments,
        config,
        device,
        determinism,
        weights,
        torch,
        identity,
        weight_digest=weight_digest,
    )
    evaluation, benchmark = benchmark_mvtec_category(
        arguments.dataset_root,
        arguments.category,
        extractor,
        device=device,
        bank_chunk_size=config.bank_chunk_size,
        warmup_count=arguments.warmup_count,
        repeat_count=arguments.repeat_count,
        model_and_weight_load_ns=model_load_ns,
        run_id=metadata.run_id,
        created_at_utc=metadata.created_at_utc,
    )
    run_directory = persist_baseline_run(
        evaluation, arguments.output_root, metadata, benchmark=benchmark
    )
    end_to_end = benchmark.results["synchronized_end_to_end"]["summary_ns"]
    print(f"Run written to {run_directory}")
    print(f"category={evaluation.category}")
    print(f"image_auroc={evaluation.metrics.image_auroc}")
    print(f"image_average_precision={evaluation.metrics.image_average_precision}")
    print(f"pixel_auroc={evaluation.metrics.pixel_auroc}")
    print(f"synchronized_end_to_end_p50_ms={end_to_end['p50'] / 1_000_000}")
    print(f"synchronized_end_to_end_p95_ms={end_to_end['p95'] / 1_000_000}")
    print(f"timing_device={benchmark.device}")
    return 0


def _baseline_run_metadata(
    arguments: argparse.Namespace,
    config: BaselineConfig,
    device: Any,
    determinism: Mapping[str, str | int | bool | None],
    weights: Any,
    torch: Any,
    identity: Mapping[str, Any],
    *,
    weight_digest: str | None = None,
) -> Any:
    from inspectrt.artifacts import BaselineRunMetadata

    return BaselineRunMetadata(
        run_id=identity["run_id"],
        created_at_utc=identity["created_at_utc"],
        dataset_root=str(arguments.dataset_root.resolve()),
        requested_device=str(device),
        bank_chunk_size=config.bank_chunk_size,
        git_commit=identity["git_commit"],
        git_dirty=identity["git_dirty"],
        uv_lock_sha256=identity["uv_lock_sha256"],
        python_version=identity["python_version"],
        platform_description=identity["platform_description"],
        dependency_versions=identity["dependency_versions"],
        determinism_flags=determinism,
        weight_enum=str(weights),
        weight_source_url=weights.url,
        weight_file_sha256=(
            weight_digest
            if weight_digest is not None
            else _cached_weight_sha256(weights.url, torch)
        ),
    )


def _baseline_run_identity(arguments: argparse.Namespace) -> dict[str, Any]:
    now = _utc_now()
    _, commit, dirty, lock_digest = _repository_metadata(Path.cwd())
    run_id = arguments.run_id or _generated_run_id(now, arguments.category, commit)
    return {
        "created_at_utc": now.isoformat(timespec="microseconds").replace("+00:00", "Z"),
        "dependency_versions": {
            name: importlib_metadata.version(name) for name in _DISTRIBUTIONS
        },
        "git_commit": commit,
        "git_dirty": dirty,
        "platform_description": platform.platform(),
        "python_version": platform.python_version(),
        "run_id": run_id,
        "uv_lock_sha256": lock_digest,
    }


def _configure_determinism(
    config: BaselineConfig, np: Any, torch: Any
) -> dict[str, str | int | bool | None]:
    random.seed(config.seed)
    np.random.seed(config.seed)
    torch.manual_seed(config.seed)
    cuda_available = torch.cuda.is_available()
    if cuda_available:
        torch.cuda.manual_seed_all(config.seed)
    torch.use_deterministic_algorithms(
        config.use_deterministic_algorithms, warn_only=False
    )
    torch.backends.cudnn.benchmark = config.cudnn_benchmark
    torch.backends.fp32_precision = "tf32" if config.allow_tf32 else "ieee"
    return {
        "python_random_seed": config.seed,
        "numpy_seed": config.seed,
        "torch_cpu_seed": config.seed,
        "torch_cuda_seed_all": config.seed if cuda_available else None,
        "use_deterministic_algorithms": torch.are_deterministic_algorithms_enabled(),
        "deterministic_algorithms_warn_only": (
            torch.is_deterministic_algorithms_warn_only_enabled()
        ),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "allow_tf32": config.allow_tf32,
        "fp32_precision": torch.backends.fp32_precision,
        "cublas_workspace_config": os.environ["CUBLAS_WORKSPACE_CONFIG"],
    }


def _resolve_device(requested: str, torch: Any) -> Any:
    device = torch.device(requested)
    if device.type == "cuda" and device.index is None:
        raise ValueError("CUDA device must include an explicit index, such as cuda:0")
    if device.type == "cuda" and (
        not torch.cuda.is_available()
        or (device.index is not None and device.index >= torch.cuda.device_count())
    ):
        raise RuntimeError(f"CUDA device {device} requested but unavailable")
    if device.type == "mps" and (
        not torch.backends.mps.is_built() or not torch.backends.mps.is_available()
    ):
        raise RuntimeError("MPS device requested but unavailable")
    return device


def _validate_benchmark_arguments(arguments: argparse.Namespace) -> None:
    if (
        arguments.device not in {"cpu", "mps"}
        and re.fullmatch(r"cuda:\d+", arguments.device) is None
    ):
        if arguments.device == "cuda":
            raise ValueError("CUDA benchmark device must include an explicit index")
        raise ValueError("Benchmark device must be cpu, cuda:<index>, or mps")
    if arguments.warmup_count != 5:
        raise ValueError("benchmark --warmup-count must be 5")
    if arguments.repeat_count != 30:
        raise ValueError("benchmark --repeat-count must be 30")
    if arguments.device == "mps" and "PYTORCH_ENABLE_MPS_FALLBACK" in os.environ:
        raise RuntimeError(
            "MPS benchmark requires PYTORCH_ENABLE_MPS_FALLBACK to be absent"
        )


def _repository_metadata(cwd: Path) -> tuple[Path, str, bool, str]:
    root = Path(_git(cwd, "rev-parse", "--show-toplevel")).resolve()
    commit = _git(root, "rev-parse", "--verify", "HEAD")
    dirty = bool(_git(root, "status", "--porcelain=v1", "--untracked-files=normal"))
    lockfile = root / "uv.lock"
    if not lockfile.is_file():
        raise FileNotFoundError(f"uv.lock not found at repository root: {root}")
    return root, commit, dirty, _sha256(lockfile)


def _git(cwd: Path, *arguments: str) -> str:
    result = subprocess.run(
        ("git", *arguments),
        cwd=cwd,
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode:
        detail = result.stderr.strip() or result.stdout.strip() or "unknown error"
        raise RuntimeError(f"Git metadata command failed: {detail}")
    return result.stdout.strip()


def _sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _cached_weight_sha256(url: str, torch: Any) -> str:
    filename = Path(urlsplit(url).path).name
    if not filename:
        raise RuntimeError(f"Official weight URL has no filename: {url}")
    cached = Path(torch.hub.get_dir()) / "checkpoints" / filename
    if not cached.is_file():
        raise FileNotFoundError(f"Cached official weight file not found: {cached}")
    return _sha256(cached)


def _utc_now() -> datetime:
    return datetime.now(timezone.utc)


def _generated_run_id(now: datetime, category: str, commit: str) -> str:
    safe_category = re.sub(r"[^A-Za-z0-9._-]+", "-", category).strip("._-")
    safe_category = safe_category or "category"
    return f"{now:%Y%m%dT%H%M%S%fZ}-{safe_category}-{commit[:7]}"


if __name__ == "__main__":
    raise SystemExit(main())
