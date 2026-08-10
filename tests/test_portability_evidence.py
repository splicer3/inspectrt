from __future__ import annotations

import copy
import hashlib
import json
import math
from pathlib import Path
import re
import subprocess
import sys
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
PRIVATE_IDENTIFIERS = (
    "Chonkpad",
    "WORKSTATIONPC",
    "workstation-wsl",
    "macbook-mps",
    "/home/",
    "Users/",
    "C:\\",
    "ssh",
    "_extra/",
    "rsync",
)


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


def test_public_json_exists_and_loads_with_standard_library(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, performance_bytes, scientific, performance = evidence
    assert scientific_bytes and performance_bytes
    assert isinstance(scientific, dict) and isinstance(performance, dict)


def test_public_json_has_exact_reviewed_byte_identity(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, performance_bytes, _, _ = evidence
    assert len(scientific_bytes) == 57_167
    assert hashlib.sha256(scientific_bytes).hexdigest() == SCIENTIFIC_SHA256
    assert len(performance_bytes) == 36_530
    assert hashlib.sha256(performance_bytes).hexdigest() == PERFORMANCE_SHA256


def test_comparison_policy_and_hash_binding_are_exact(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, _, scientific, performance = evidence
    assert scientific["comparison_id"] == COMPARISON_ID
    assert performance["scientific"]["comparison_id"] == COMPARISON_ID
    assert scientific["policy"] == {"policy_id": POLICY_ID, "sha256": POLICY_SHA256}
    assert performance["policy"] == scientific["policy"]
    assert performance["scientific"]["scientific_json"] == {
        "byte_count": 57_167,
        "schema_id": "inspectrt_portability_comparison_v1",
        "schema_version": 1,
        "sha256": hashlib.sha256(scientific_bytes).hexdigest(),
    }
    assert hashlib.sha256(POLICY_PATH.read_bytes()).hexdigest() == POLICY_SHA256


def test_reference_candidate_and_required_environment_order_is_exact(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    _, _, scientific, _ = evidence
    assert scientific["reference"]["environment_id"] == REFERENCE_ID
    assert tuple(item["environment_id"] for item in scientific["candidates"]) == (
        CANDIDATE_IDS
    )
    assert scientific["candidates"][2]["execution_layer"] == "wsl2"
    assert scientific["candidates"][3]["policy_role"] == "holdout"
    assert scientific["candidates"][4]["requested_device"] == "mps"


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


def test_nearest_index_mismatch_counts_are_the_reviewed_observations(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    _, _, scientific, _ = evidence
    for environment_id, expected in MISMATCHES.items():
        indices = scientific["scientific_results"][environment_id][
            "discrete_components"
        ]["nearest_bank_indices"]
        assert indices["mismatch_count"] == expected
        assert indices["exact"] is (expected == 0)


def test_every_floating_limit_and_metric_limit_is_satisfied(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    _, _, scientific, _ = evidence
    policy = json.loads(POLICY_PATH.read_bytes())
    metric_limits = policy["metric_absolute_delta_limits"]
    for result in scientific["scientific_results"].values():
        assert all(
            component["policy_violation_count"] == 0
            for component in result["floating_components"].values()
        )
        for metric in result["metrics"]:
            assert metric["absolute_delta"] <= metric_limits[metric["metric_name"]]


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


def test_raw_arrays_and_summaries_recompute_exactly(
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


def test_renderer_rejects_unexpected_environment_ordering(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, _, _, performance = evidence
    broken = copy.deepcopy(performance)
    broken["runs"][0], broken["runs"][1] = (
        broken["runs"][1],
        broken["runs"][0],
    )
    with pytest.raises(ValueError, match="environment ordering"):
        renderer.render_svg(scientific_bytes, _canonical(broken))


def test_renderer_rejects_missing_mps_timing(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, _, _, performance = evidence
    broken = copy.deepcopy(performance)
    broken["runs"][5]["environment"]["environment_id"] = "m1pro-macos-cpu"
    with pytest.raises(ValueError):
        renderer.render_svg(scientific_bytes, _canonical(broken))


def test_graph_generation_is_deterministic_and_matches_tracked_svg(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    scientific_bytes, performance_bytes, _, _ = evidence
    first = renderer.render_svg(scientific_bytes, performance_bytes)
    assert first == renderer.render_svg(scientific_bytes, performance_bytes)
    assert first == SVG_PATH.read_bytes()
    assert first.endswith(b"\n") and not first.endswith(b"\n\n")


def test_graph_rates_are_derived_from_persisted_counts(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    _, _, scientific, performance = evidence
    changed = copy.deepcopy(scientific)
    changed["scientific_results"]["p53-linux-cpu"]["discrete_components"][
        "nearest_bank_indices"
    ]["mismatch_rate"] = 0.99
    scientific_bytes = _canonical(changed)
    performance = copy.deepcopy(performance)
    performance["scientific"]["scientific_json"]["sha256"] = hashlib.sha256(
        scientific_bytes
    ).hexdigest()
    svg = renderer.render_svg(scientific_bytes, _canonical(performance))
    assert b">4.20%</text>" in svg


def test_graph_check_mode_accepts_only_the_tracked_bytes(tmp_path: Path) -> None:
    result = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/render_portability_latency.py"),
            "--scientific",
            str(SCIENTIFIC_PATH),
            "--performance",
            str(PERFORMANCE_PATH),
            "--check",
            str(SVG_PATH),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 0, result.stderr

    same_output = tmp_path / "same-output"
    result = subprocess.run(
        (
            sys.executable,
            str(ROOT / "scripts/render_portability_latency.py"),
            "--scientific",
            str(SCIENTIFIC_PATH),
            "--performance",
            str(PERFORMANCE_PATH),
            "--output",
            str(same_output),
            "--png-preview",
            str(same_output),
        ),
        check=False,
        capture_output=True,
        text=True,
    )
    assert result.returncode == 2
    assert "--output and --png-preview must differ" in result.stderr


def test_svg_is_accessible_standalone_xml_with_exact_dimensions() -> None:
    root = ET.fromstring(SVG_PATH.read_bytes())
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.tag == f"{namespace}svg"
    assert root.attrib["width"] == "1200" and root.attrib["height"] == "1500"
    assert root.attrib["viewBox"] == "0 0 864 1080"
    assert root.find(f"{namespace}title") is not None
    assert root.find(f"{namespace}desc") is not None


def test_svg_has_no_external_or_executable_resource() -> None:
    root = ET.fromstring(SVG_PATH.read_bytes())
    namespace = "{http://www.w3.org/2000/svg}"
    assert root.findall(f".//{namespace}script") == []
    assert root.findall(f".//{namespace}image") == []
    for element in root.iter():
        for name, value in element.attrib.items():
            if name.endswith("href"):
                assert value.startswith("#")
            if "url(" in value:
                assert value.startswith("url(#")
    rendered = SVG_PATH.read_text().casefold()
    assert "https://" not in rendered and "@font-face" not in rendered


def test_svg_uses_the_direct_two_panel_copy_and_mps_treatment() -> None:
    rendered = SVG_PATH.read_text()
    required = (
        "A. Stage latency",
        "p50 → p95",
        "Feature extraction",
        "Exact retrieval",
        "End to end",
        "T1000 · CUDA · reference",
        "T1000 · CUDA · repeat",
        "P53 · CPU",
        "RTX 4080 Super · CUDA · WSL 2",
        "M1 Pro · CPU",
        "M1 Pro · MPS",
        "Milliseconds (log scale)",
        "B. Top-1 index differences",
        "Different top-1 index (%)",
        "T1000 repeat",
        "P53 CPU",
        "RTX 4080 Super",
        "M1 Pro CPU",
        "M1 Pro MPS",
    )
    assert all(value in rendered for value in required)
    assert len(re.findall(r">[0-9]+\.[0-9]–[0-9]+\.[0-9] ms</text>", rendered)) == 6
    for value in (
        "InspectRT: one frozen inspection workload across CPU, CUDA and MPS",
        "MPS latency was not collected",
        "All floating outputs and metrics",
        "Share of 84,992",
        "scientific only",
        "scientific-only",
        "post-policy",
        "non-gating",
        "confidence interval",
        "confidence-interval",
        "evaluation only",
        "same-stack control",
        "Ubuntu 24.04.4",
        "macOS 26.5.2",
        "arm64",
    ):
        assert value not in rendered


def test_svg_candidate_percentages_match_exact_counts() -> None:
    rendered = SVG_PATH.read_text()
    expected = tuple(f"{count / 84_992 * 100:.2f}%" for count in MISMATCHES.values())
    assert expected == ("0.00%", "4.20%", "2.15%", "3.18%", "3.55%")
    assert all(f">{value}</text>" in rendered for value in expected)


def test_svg_metadata_binds_exact_evidence_hashes() -> None:
    root = ET.fromstring(SVG_PATH.read_bytes())
    metadata = root.find("{http://www.w3.org/2000/svg}metadata")
    assert metadata is not None
    value = json.loads(metadata.text or "")
    assert value["comparison_id"] == COMPARISON_ID
    assert value["scientific_sha256"] == SCIENTIFIC_SHA256
    assert value["performance_sha256"] == PERFORMANCE_SHA256


def test_new_public_files_have_no_private_identifier_or_internal_path() -> None:
    for path in (SCIENTIFIC_PATH, PERFORMANCE_PATH, SVG_PATH):
        text = path.read_text()
        assert not any(value in text for value in PRIVATE_IDENTIFIERS), path
        assert re.search(r"(?:^|\s)/(?:tmp|mnt|var|opt)/", text) is None


def test_public_documentation_has_no_private_identifier() -> None:
    text = "\n".join(
        (ROOT / path).read_text() for path in ("README.md", "docs/portability.md")
    )
    for value in PRIVATE_IDENTIFIERS:
        assert value not in text


def test_performance_json_has_no_comparative_or_universal_claim_field(
    evidence: tuple[bytes, bytes, dict[str, object], dict[str, object]],
) -> None:
    _, _, _, performance = evidence
    rendered = "\n".join(
        (
            json.dumps(performance),
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


def test_evidence_directory_has_only_the_three_public_artifacts() -> None:
    assert {path.name for path in EVIDENCE.iterdir()} == {
        "scientific.json",
        "performance.json",
        "latency.svg",
    }
