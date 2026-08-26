from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import pytest

from scripts import render_portability_latency as renderer


ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "docs/evidence/inspectrt_cross_platform_evidence_v2"
SCIENTIFIC_PATH = EVIDENCE / "scientific.json"
PERFORMANCE_PATH = EVIDENCE / "performance.json"
SVG_PATH = EVIDENCE / "latency.svg"
POLICY_PATH = ROOT / "configs/portability_policy.json"
SCIENTIFIC_SHA256 = "81318cd81c0e5f23be953719c2bb03604c22c75bb8c1dd17c389786623d32b8a"
PERFORMANCE_SHA256 = "44057e5317b902341b1b359c0ff5a43f3900940115a206e1bd8ea2774adc85d9"
COMPARISON_ID = "1dec773f2d237598305a315145bec7bc40b9f94fbd326ed44f6330d3c9a11fe5"
POLICY_ID = "inspectrt-bottle-bc330b9-v1"
POLICY_SHA256 = "576717b70e53714eed8370619cc08c81517405728f767942298b0c8c415836a2"
REFERENCE_ID = "p53-linux-t1000-cuda-reference"
CANDIDATE_IDS = (
    "p53-linux-t1000-cuda-control",
    "p53-linux-cpu",
    "rtx4080-wsl2-cuda",
    "m1pro-macos-cpu",
    "m1pro-macos-mps",
)
TIMING_IDS = (REFERENCE_ID, *CANDIDATE_IDS)
AGGREGATION_COMMIT = "339b66848c5f47c3e03b0377df33315421fe96aa"
TIMING_HARNESS_COMMIT = "4f230679d52b5ed08e43230ebb1308cb85a33e57"
TIMING_HARNESS_LOCK = "4464c375e3bf0f9c575504b427a0e82aedc954ef3491807306b72c382ce07d5c"
MISMATCHES = {
    "p53-linux-t1000-cuda-control": 0,
    "p53-linux-cpu": 3569,
    "rtx4080-wsl2-cuda": 1826,
    "m1pro-macos-cpu": 2701,
    "m1pro-macos-mps": 3019,
}


@pytest.fixture(scope="module")
def evidence() -> tuple[bytes, bytes, dict[str, object], dict[str, object]]:
    scientific_bytes = SCIENTIFIC_PATH.read_bytes()
    performance_bytes = PERFORMANCE_PATH.read_bytes()
    return (
        scientific_bytes,
        performance_bytes,
        json.loads(scientific_bytes),
        json.loads(performance_bytes),
    )


def _canonical(value: object) -> bytes:
    return (json.dumps(value, sort_keys=True, separators=(",", ":")) + "\n").encode()


def _percentile(raw: list[int], quantile: float) -> float:
    ordered = sorted(raw)
    rank = (len(ordered) - 1) * quantile
    lower = math.floor(rank)
    upper = math.ceil(rank)
    return ordered[lower] + (ordered[upper] - ordered[lower]) * (rank - lower)


def test_public_json_has_exact_reviewed_byte_identity(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, performance_bytes, scientific, performance = evidence
    assert len(scientific_bytes) == 57_167
    assert hashlib.sha256(scientific_bytes).hexdigest() == SCIENTIFIC_SHA256
    assert len(performance_bytes) == 36_530
    assert hashlib.sha256(performance_bytes).hexdigest() == PERFORMANCE_SHA256
    assert scientific["comparison_id"] == COMPARISON_ID
    assert performance["scientific"]["comparison_id"] == COMPARISON_ID
    assert scientific["policy"] == {"policy_id": POLICY_ID, "sha256": POLICY_SHA256}
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == POLICY_SHA256
    assert scientific["reference"]["environment_id"] == REFERENCE_ID
    assert (
        tuple(item["environment_id"] for item in scientific["candidates"])
        == CANDIDATE_IDS
    )
    assert {path.name for path in EVIDENCE.iterdir()} == {
        "scientific.json",
        "performance.json",
        "latency.svg",
    }


def test_all_structural_gates_and_required_discrete_outputs_are_exact(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    _, _, scientific, _ = evidence
    assert all(
        len(value["gates"]) == 29 and all(value["gates"].values())
        for value in scientific["comparability"].values()
    )
    for result in scientific["scientific_results"].values():
        discrete = result["discrete_components"]
        for name in ("test_sample_ids", "test_labels", "evaluation_masks"):
            assert discrete[name]["exact"] is True
            assert discrete[name]["mismatch_count"] == 0
    policy = json.loads(POLICY_PATH.read_bytes())
    for environment_id, expected in MISMATCHES.items():
        indices = scientific["scientific_results"][environment_id][
            "discrete_components"
        ]["nearest_bank_indices"]
        assert indices["mismatch_count"] == expected
        assert indices["exact"] is (expected == 0)
    for result in scientific["scientific_results"].values():
        assert all(
            component["policy_violation_count"] == 0
            for component in result["floating_components"].values()
        )
        for metric in result["metrics"]:
            assert (
                metric["absolute_delta"]
                <= policy["metric_absolute_delta_limits"][metric["metric_name"]]
            )


def test_performance_matrix_and_timing_provenance_are_exact(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    _, _, _, performance = evidence
    assert performance["schema_version"] == 2
    assert performance["schema_id"] == "inspectrt_portability_performance_v2"
    assert performance["milestone_id"] == "inspectrt_portable_timing_v2"
    assert performance["status"] == "descriptive_only"
    assert tuple(performance["environment_order"]) == TIMING_IDS
    assert (
        tuple(item["environment"]["environment_id"] for item in performance["runs"])
        == TIMING_IDS
    )
    assert performance["runs"][3]["environment"]["execution_layer"] == "wsl2"
    assert performance["runs"][5]["environment"]["requested_device"] == "mps"
    assert performance["generator"] == {
        "dirty": False,
        "source_commit": AGGREGATION_COMMIT,
    }
    assert performance["timing_harness"]["source_commit"] == TIMING_HARNESS_COMMIT
    assert performance["timing_harness"]["uv_lock_sha256"] == TIMING_HARNESS_LOCK
    assert performance["timing_harness"]["dirty"] is False
    _assert_raw_arrays_and_summaries(evidence)


def _assert_raw_arrays_and_summaries(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    _, _, _, performance = evidence
    for run in performance["runs"]:
        measurements = run["measurements"]
        records = [
            *measurements["repeated_stages"].values(),
            measurements["synchronized_end_to_end"],
        ]
        for record in records:
            raw = record["raw_ns"]
            summary = record["summary_ns"]
            assert len(raw) == summary["count"] == 30
            assert summary["minimum"] == min(raw)
            assert summary["maximum"] == max(raw)
            assert summary["mean"] == sum(raw) / len(raw)
            assert summary["p50"] == _percentile(raw, 0.50)
            assert summary["p95"] == _percentile(raw, 0.95)


def test_renderer_rejects_broken_scientific_hash_binding(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, _, _, performance = evidence
    broken = copy.deepcopy(performance)
    broken["scientific"]["scientific_json"]["sha256"] = "0" * 64
    with pytest.raises(ValueError, match="scientific hash binding"):
        renderer.render_svg(scientific_bytes, _canonical(broken))


def test_graph_generation_is_deterministic_and_matches_tracked_svg(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, performance_bytes, _, _ = evidence
    first = renderer.render_svg(scientific_bytes, performance_bytes)
    assert first == renderer.render_svg(scientific_bytes, performance_bytes)
    assert first == SVG_PATH.read_bytes()
    assert first.endswith(b"\n") and not first.endswith(b"\n\n")
    root = ET.fromstring(first)
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.find(f"{namespace}title") is not None
    assert root.find(f"{namespace}desc") is not None
    assert root.findall(f".//{namespace}script") == []
    assert root.findall(f".//{namespace}image") == []
    for element in root.iter():
        for name, value in element.attrib.items():
            if name.endswith("href"):
                assert value.startswith("#")
            if "url(" in value:
                assert value.startswith("url(#")


def test_new_public_files_have_no_private_identifier_or_internal_path() -> None:
    for path in (SCIENTIFIC_PATH, PERFORMANCE_PATH, SVG_PATH):
        text = path.read_text()
        assert re.search(r"(?:^|\s)/(?:tmp|mnt|var|opt)/", text) is None
    rendered = "\n".join(
        (
            SCIENTIFIC_PATH.read_text(),
            PERFORMANCE_PATH.read_text(),
            SVG_PATH.read_text(),
            (ROOT / "README.md").read_text(),
            (ROOT / "docs/portability.md").read_text(),
        )
    ).casefold()
    for word in (
        "speedup",
        "ranking",
        "winner",
        "portable_everywhere",
        "confidence interval",
        "confidence-interval",
    ):
        assert word not in rendered
