from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import re


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/inspectrt_onnx_feature_portability_v1"
SCIENTIFIC = EVIDENCE / "scientific.json"
SCIENTIFIC_BYTES = 8_705
SCIENTIFIC_SHA256 = "b07bbd05d7d6535e0d1088ce23b54e83b0a7754e0ffd921ced778e81d7c5430f"
ENVIRONMENT_ORDER = (
    "p53-linux-cpu",
    "m1pro-macos-cpu",
    "ryzen9700x-wsl2-cpu",
)
TOP_LEVEL_KEYS = {
    "artifact",
    "category",
    "environment_order",
    "environments",
    "evidence_bindings",
    "exact_structures",
    "graph_contract",
    "identities",
    "limitations",
    "milestone_id",
    "policy",
    "profile_id",
    "runtime_contract",
    "schema_id",
    "schema_version",
    "status",
}
OBSERVATIONS = {
    "p53-linux-cpu": {
        "maximum_absolute_errors": {
            "anomaly_maps": 0.00164794921875,
            "image_scores": 0.0010986328125,
            "layer2": 0.00031107664108276367,
            "memory_bank": 0.00010371208190917969,
            "patch_distances": 0.0018463134765625,
            "patch_embeddings": 0.00011277198791503906,
        },
        "metric_absolute_deltas": {
            "image_auroc": 0.0,
            "image_average_precision": 0.0,
            "pixel_auroc": 5.415049519896797e-10,
        },
        "nearest_index_mismatches": {
            "count": 2_504,
            "total": 84_992,
            "worst_per_test_sample": {"count": 47, "total": 1_024},
        },
    },
    "m1pro-macos-cpu": {
        "maximum_absolute_errors": {
            "anomaly_maps": 0.0020294189453125,
            "image_scores": 0.000946044921875,
            "layer2": 0.0003688335418701172,
            "memory_bank": 0.00010776519775390625,
            "patch_distances": 0.00225830078125,
            "patch_embeddings": 0.00010776519775390625,
        },
        "metric_absolute_deltas": {
            "image_auroc": 0.0,
            "image_average_precision": 0.0,
            "pixel_auroc": 1.3017835698292402e-9,
        },
        "nearest_index_mismatches": {
            "count": 2_956,
            "total": 84_992,
            "worst_per_test_sample": {"count": 49, "total": 1_024},
        },
    },
    "ryzen9700x-wsl2-cpu": {
        "maximum_absolute_errors": {
            "anomaly_maps": 0.001556396484375,
            "image_scores": 0.00146484375,
            "layer2": 0.00033849477767944336,
            "memory_bank": 0.00010466575622558594,
            "patch_distances": 0.00164794921875,
            "patch_embeddings": 0.00010854005813598633,
        },
        "metric_absolute_deltas": {
            "image_auroc": 0.0,
            "image_average_precision": 0.0,
            "pixel_auroc": 1.644684388679707e-10,
        },
        "nearest_index_mismatches": {
            "count": 2_466,
            "total": 84_992,
            "worst_per_test_sample": {"count": 41, "total": 1_024},
        },
    },
}
UNSUPPORTED_CLAIMS = [
    "universal_bitwise_equivalence",
    "all_category_parity",
    "arbitrary_shape_support",
    "dynamic_batch_support",
    "gpu_provider_parity",
    "core_ml_execution",
    "ort_cuda_execution",
    "tensorrt_support",
    "performance_improvement",
    "hard_real_time_behavior",
    "universal_ort_determinism",
]


def _object(pairs: list[tuple[str, object]]) -> dict[str, object]:
    value: dict[str, object] = {}
    for key, item in pairs:
        if key in value:
            raise ValueError(key)
        value[key] = item
    return value


def _reject_constant(value: str) -> object:
    raise ValueError(value)


def _load() -> tuple[bytes, dict[str, object]]:
    data = SCIENTIFIC.read_bytes()
    value = json.loads(
        data,
        object_pairs_hook=_object,
        parse_constant=_reject_constant,
    )
    assert isinstance(value, dict)
    return data, value


def _canonical(value: object) -> bytes:
    return (
        json.dumps(
            value,
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=False,
            allow_nan=False,
        )
        + "\n"
    ).encode()


def _numbers(value: object) -> list[float]:
    if isinstance(value, float):
        return [value]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numbers(item)]
    if isinstance(value, list):
        return [number for item in value for number in _numbers(item)]
    return []


def test_public_record_is_canonical_compact_and_frozen() -> None:
    data, value = _load()
    assert len(data) == SCIENTIFIC_BYTES < 49_152
    assert hashlib.sha256(data).hexdigest() == SCIENTIFIC_SHA256
    assert data == _canonical(value)
    assert data.endswith(b"\n") and not data.endswith(b"\n\n")
    assert all(math.isfinite(number) for number in _numbers(value))
    assert set(value) == TOP_LEVEL_KEYS
    assert {path.name for path in EVIDENCE.iterdir()} == {"scientific.json"}
    assert value["schema_id"] == "inspectrt_onnx_feature_portability_scientific_v1"
    assert value["milestone_id"] == "inspectrt_onnx_feature_portability_v1"
    assert tuple(value["environment_order"]) == ENVIRONMENT_ORDER
    assert value["identities"]["source"] == {
        "implementation_commit": "d99225474e4760becb1c46ce811a71c016c292e0",
        "root_lock_sha256": "d92724be7ede2442141cf898a67d12752e44d3bd5df6077dbd5ae97df325df42",
    }
    assert value["artifact"]["model"]["sha256"] == (
        "143b305b37a92e3f2c7dc4268c25baccdf3cfb01c5304f29068f422ff9d8146a"
    )
    assert value["graph_contract"]["pool"]["count_include_pad"] is True
    assert value["graph_contract"]["patch_order"] == "row_major_y_then_x"
    assert value["runtime_contract"]["requested_providers"] == ["CPUExecutionProvider"]
    assert set(value["policy"]) == {"limits", "nearest_index_policy", "v2"}
    assert value["policy"]["v2"]["holdout_policy_result"] == "PASS"
    environments = value["environments"]
    assert [environments[name]["policy_v2_role"] for name in ENVIRONMENT_ORDER] == [
        "calibration",
        "calibration",
        "holdout",
    ]
    assert all(
        environments[name]["ort_repeatability_exact"] for name in ENVIRONMENT_ORDER
    )


def test_policy_bounds_and_negative_claims_are_mechanical() -> None:
    _, value = _load()
    policy = value["policy"]
    limits = policy["limits"]
    nearest = policy["nearest_index_policy"]
    assert limits == {
        "floating_components": {
            "anomaly_maps": {"atol": 0.003, "rtol": 0},
            "image_scores": {"atol": 0.0015, "rtol": 0},
            "layer2": {"atol": 0.0004, "rtol": 0},
            "memory_bank": {"atol": 0.00011, "rtol": 0},
            "patch_distances": {"atol": 0.003, "rtol": 0},
            "patch_embeddings": {"atol": 0.0002, "rtol": 0},
        },
        "metric_absolute_deltas": {
            "image_auroc": 0,
            "image_average_precision": 0,
            "pixel_auroc": 3e-9,
        },
    }
    assert nearest["global_mismatch_fraction"] == {
        "denominator": 25,
        "inclusive": True,
        "numerator": 1,
    }
    assert nearest["per_test_sample_mismatch_fraction"] == {
        "denominator": 20,
        "inclusive": True,
        "numerator": 1,
    }
    assert nearest["patches_per_test_sample"] == 1_024
    assert nearest["total_nearest_indices"] == 84_992
    for environment in value["environments"].values():
        for name, observed in environment["maximum_absolute_errors"].items():
            limit = limits["floating_components"][name]
            assert limit["rtol"] == 0 and observed <= limit["atol"]
        for name, observed in environment["metric_absolute_deltas"].items():
            assert observed <= limits["metric_absolute_deltas"][name]
        mismatches = environment["nearest_index_mismatches"]
        global_limit = nearest["global_mismatch_fraction"]
        assert mismatches["count"] * global_limit["denominator"] <= (
            mismatches["total"] * global_limit["numerator"]
        )
        local = mismatches["worst_per_test_sample"]
        local_limit = nearest["per_test_sample_mismatch_fraction"]
        assert local["count"] * local_limit["denominator"] <= (
            local["total"] * local_limit["numerator"]
        )

    assert policy["v2"] == {
        "byte_count": 7_820,
        "calibration_environment_ids": ["p53-linux-cpu", "m1pro-macos-cpu"],
        "fourth_holdout_exists": False,
        "holdout_calibration_feedback": False,
        "holdout_environment_id": "ryzen9700x-wsl2-cpu",
        "holdout_independent": True,
        "holdout_measurement_result": "PASS",
        "holdout_policy_result": "PASS",
        "policy_id": "inspectrt-onnx-bottle-p53-m1-cpu-v2",
        "policy_unchanged": True,
        "policy_v3_exists": False,
        "sha256": "1d367dc55747b23d5941231a5f5d7c7434f32b0f71a55d0b09f6144321dbf6f3",
    }
    exact = value["exact_structures"]
    assert set(exact["agreements"]) == {
        "category",
        "complete_ordered_observations",
        "complete_ordered_sample_ids",
        "evaluation_masks",
        "original_image_metadata",
        "tensor_contiguity",
        "tensor_cpu_devices",
        "tensor_dtypes",
        "tensor_shapes",
        "test_labels",
        "test_ordered_observations",
        "test_ordered_sample_ids",
    }
    assert all(exact["agreements"].values())
    assert exact["all_floating_values_bitwise_equal"] is False
    assert exact["all_nearest_indices_equal"] is False
    assert nearest["exact_identity_required"] is False
    assert nearest["retrieval_contract"]["exact_computed_tie_behavior_unchanged"]
    assert nearest["retrieval_contract"]["epsilon_tie_rule"] is False
    assert value["limitations"]["unsupported_claims"] == UNSUPPORTED_CLAIMS
    assert value["limitations"]["scope"] == {
        "category_count": 1,
        "complete_sample_count": 292,
        "environment_ids": list(ENVIRONMENT_ORDER),
        "graph": "static_batch_1_float32_256x256",
        "onnxruntime_version": "1.28.0",
        "profile_id": "inspectrt_feature_memory_v1",
        "provider": "CPUExecutionProvider",
        "repeatability": "one_local_scope_per_environment",
        "weight_count": 1,
        "weight_enum": "ResNet50_Weights.IMAGENET1K_V2",
    }


def test_public_docs_use_the_real_surface_and_keep_private_data_out() -> None:
    readme = (ROOT / "README.md").read_text()
    document = (ROOT / "docs/onnx-portability.md").read_text()
    headings = (
        "## Scope",
        "## Graph boundary",
        "## Installation",
        "## Artifact export and validation",
        "## Direct ORT CPU consumer usage",
        "## Scientific method",
        "## Policy lineage",
        "## Reviewed results",
        "## Evidence identities",
        "## Evidence scope",
    )
    assert [document.index(heading) for heading in headings] == sorted(
        document.index(heading) for heading in headings
    )
    export = "uv run --extra onnx inspectrt onnx export \\\n  --output-root outputs"
    validate = (
        "uv run --extra onnx inspectrt onnx validate \\\n"
        "  --artifact \\\n"
        "  outputs/artifacts/inspectrt_onnx_feature_portability_v1/<artifact-id>"
    )
    for text in (readme, document):
        assert export in text and validate in text
        assert "docs/onnx-portability.md" in readme
        assert "evidence/inspectrt_onnx_feature_portability_v1/scientific.json" in text
    assert "uv sync --locked --extra onnx" in readme and document
    for token in (
        "from pathlib import Path",
        "preprocess_image",
        "OnnxRuntimeCpuFeatureConsumer.from_artifact",
        "prepared.image.unsqueeze(0)",
        "consumer.extract",
        "outputs.layer2",
        "outputs.patch_embeddings",
    ):
        assert token in document
    assert f"{SCIENTIFIC_BYTES:,} bytes" in document and SCIENTIFIC_SHA256 in document
    assert "manifest.json\nmodel.onnx" in document
    assert "pipeline uses for retrieval, scoring" in document
    for action in ("run", "evaluate", "benchmark"):
        assert f"inspectrt onnx {action}" not in readme + document
    documentation = (readme + "\n" + document).casefold()
    for claim in (
        "supports dynamic shapes",
        "dynamic shapes are supported",
        "dynamic batch is supported",
        "supports gpu",
        "gpu-provider parity is established",
        "supports core ml",
        "core ml is supported",
        "supports ort cuda",
        "ort cuda is supported",
        "supports tensorrt",
        "tensorrt is supported",
        "improves performance",
        "performance improvement is established",
        "faster than",
        "speedup",
        "higher throughput",
        "lower latency",
        "full-pipeline onnx",
        "onnx retrieval is supported",
        "model bytes are redistributed",
    ):
        assert claim not in documentation

    paths = (
        ROOT / "README.md",
        ROOT / "docs/onnx-portability.md",
        ROOT / "docs/portability.md",
        SCIENTIFIC,
    )
    public_text = "\n".join(path.read_text() for path in paths).casefold()
    assert chr(0xB7) not in public_text
    assert re.search(r"\b[\w.+-]+@[\w.-]+\.[a-z]{2,}\b", public_text) is None
    assert re.search(r"\b(?:\d{1,3}\.){3}\d{1,3}\b", public_text) is None
