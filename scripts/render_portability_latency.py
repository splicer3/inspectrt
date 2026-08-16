"""Render the tracked cross-platform evidence as a deterministic SVG."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import math
from pathlib import Path
import re
from xml.sax.saxutils import escape

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


SCIENTIFIC_SCHEMA = "inspectrt_portability_comparison_v1"
PERFORMANCE_SCHEMA = "inspectrt_portability_performance_v2"
SCIENTIFIC_MILESTONE = "inspectrt_cross_platform_evidence_v1"
PERFORMANCE_MILESTONE = "inspectrt_portable_timing_v2"
TIMING_ENVIRONMENTS = (
    "p53-linux-t1000-cuda-reference",
    "p53-linux-t1000-cuda-control",
    "p53-linux-cpu",
    "rtx4080-wsl2-cuda",
    "m1pro-macos-cpu",
    "m1pro-macos-mps",
)
CANDIDATE_ENVIRONMENTS = (
    "p53-linux-t1000-cuda-control",
    "p53-linux-cpu",
    "rtx4080-wsl2-cuda",
    "m1pro-macos-cpu",
    "m1pro-macos-mps",
)
TIMING_LABELS = (
    "T1000 CUDA reference",
    "T1000 CUDA repeat",
    "Intel Core i7-9850H CPU",
    "RTX 4080 Super CUDA WSL 2",
    "M1 Pro CPU",
    "M1 Pro MPS",
)
CANDIDATE_LABELS = (
    "T1000 repeat",
    "Intel Core i7-9850H CPU",
    "RTX 4080 Super",
    "M1 Pro CPU",
    "M1 Pro MPS",
)
STAGES = (
    ("frozen_feature_extraction", "Feature extraction"),
    ("exact_chunked_retrieval", "Exact retrieval"),
    ("synchronized_end_to_end", "End to end"),
)
RENDERER_ID = "inspectrt_portability_latency"
RENDERER_VERSION = 5
SVG_WIDTH = 1200
SVG_HEIGHT = 1500


def _document(payload: bytes, name: str) -> dict[str, object]:
    try:
        value = json.loads(payload)
    except (UnicodeDecodeError, json.JSONDecodeError) as error:
        raise ValueError(f"{name} is not valid UTF-8 JSON") from error
    if not isinstance(value, dict):
        raise ValueError(f"{name} must contain a JSON object")
    return value


def _require(value: object, expected: object, name: str) -> None:
    if value != expected:
        raise ValueError(f"{name} is invalid")


def _number(value: object, name: str, *, allow_zero: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise ValueError(f"{name} must be numeric")
    number = float(value)
    if not math.isfinite(number) or number < 0 or (not allow_zero and number == 0):
        raise ValueError(f"{name} must be finite and non-negative")
    return number


def _stage_summary(run: dict[str, object], key: str) -> dict[str, object]:
    measurements = run["measurements"]
    assert isinstance(measurements, dict)
    if key == "synchronized_end_to_end":
        measurement = measurements[key]
    else:
        repeated = measurements["repeated_stages"]
        assert isinstance(repeated, dict)
        measurement = repeated[key]
    assert isinstance(measurement, dict)
    summary = measurement["summary_ns"]
    assert isinstance(summary, dict)
    return summary


def _validate(
    scientific: dict[str, object],
    performance: dict[str, object],
    scientific_sha256: str,
) -> None:
    _require(scientific.get("schema_version"), 1, "scientific schema version")
    _require(scientific.get("schema_id"), SCIENTIFIC_SCHEMA, "scientific schema ID")
    _require(
        scientific.get("milestone_id"),
        SCIENTIFIC_MILESTONE,
        "scientific milestone ID",
    )
    _require(performance.get("schema_version"), 2, "performance schema version")
    _require(performance.get("schema_id"), PERFORMANCE_SCHEMA, "performance schema ID")
    _require(
        performance.get("milestone_id"),
        PERFORMANCE_MILESTONE,
        "performance milestone ID",
    )
    performance_scientific = performance.get("scientific")
    if not isinstance(performance_scientific, dict):
        raise ValueError("performance scientific identity is invalid")
    _require(
        performance_scientific.get("comparison_id"),
        scientific.get("comparison_id"),
        "comparison ID binding",
    )
    scientific_json = performance_scientific.get("scientific_json")
    if not isinstance(scientific_json, dict):
        raise ValueError("performance scientific JSON identity is invalid")
    _require(
        scientific_json.get("sha256"),
        scientific_sha256,
        "scientific hash binding",
    )
    _require(performance.get("status"), "descriptive_only", "performance status")

    candidates = scientific.get("candidates")
    if not isinstance(candidates, list):
        raise ValueError("scientific candidates are invalid")
    _require(
        [item.get("environment_id") for item in candidates],
        list(CANDIDATE_ENVIRONMENTS),
        "scientific candidate ordering",
    )

    results = scientific.get("scientific_results")
    if not isinstance(results, dict) or set(results) != set(CANDIDATE_ENVIRONMENTS):
        raise ValueError("scientific results are invalid")
    query_count: int | None = None
    for environment_id in CANDIDATE_ENVIRONMENTS:
        result = results[environment_id]
        if not isinstance(result, dict):
            raise ValueError("scientific result is invalid")
        floating = result.get("floating_components")
        discrete = result.get("discrete_components")
        if not isinstance(floating, dict) or not isinstance(discrete, dict):
            raise ValueError("scientific result components are invalid")
        if any(
            not isinstance(component, dict)
            or component.get("policy_violation_count") != 0
            for component in floating.values()
        ):
            raise ValueError("floating outputs exceed the reviewed policy")
        indices = discrete.get("nearest_bank_indices")
        if not isinstance(indices, dict):
            raise ValueError("nearest-index result is invalid")
        count = indices.get("element_count")
        mismatches = indices.get("mismatch_count")
        if (
            isinstance(count, bool)
            or not isinstance(count, int)
            or count <= 0
            or isinstance(mismatches, bool)
            or not isinstance(mismatches, int)
            or not 0 <= mismatches <= count
        ):
            raise ValueError("nearest-index counts are invalid")
        query_count = count if query_count is None else query_count
        _require(count, query_count, "nearest-index query count")

    _require(
        performance.get("environment_order"),
        list(TIMING_ENVIRONMENTS),
        "performance environment ordering",
    )
    runs = performance.get("runs")
    if not isinstance(runs, list):
        raise ValueError("performance timing records are invalid")
    _require(
        [
            run.get("environment", {}).get("environment_id")
            if isinstance(run, dict) and isinstance(run.get("environment"), dict)
            else None
            for run in runs
        ],
        list(TIMING_ENVIRONMENTS),
        "performance environment ordering",
    )
    if runs[3].get("environment", {}).get("execution_layer") != "wsl2":
        raise ValueError("RTX execution layer must remain WSL 2")

    methodology = performance.get("timing_methodology")
    if not isinstance(methodology, dict):
        raise ValueError("timing methodology is invalid")
    for key, expected in {
        "timing_unit": "nanoseconds",
        "warmup_count": 5,
        "repeat_count": 30,
        "warmup_samples_in_statistics": False,
    }.items():
        _require(methodology.get(key), expected, f"timing methodology {key}")

    for run in runs:
        if not isinstance(run, dict):
            raise ValueError("performance timing record is invalid")
        for key, _ in STAGES:
            summary = _stage_summary(run, key)
            p50 = _number(summary.get("p50"), f"{key} p50")
            p95 = _number(summary.get("p95"), f"{key} p95")
            if p95 < p50:
                raise ValueError(f"{key} p95 must be greater than or equal to p50")


def _figure(
    scientific: dict[str, object], performance: dict[str, object]
) -> plt.Figure:
    with plt.rc_context(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 18,
            "axes.labelsize": 12,
            "axes.linewidth": 0.8,
            "xtick.labelsize": 11,
            "ytick.labelsize": 12,
            "legend.fontsize": 11,
            "svg.fonttype": "none",
            "svg.hashsalt": "inspectrt-portability-latency-v2",
        }
    ):
        figure = plt.figure(figsize=(12, 15), dpi=100)
        grid = figure.add_gridspec(
            2,
            1,
            left=0.31,
            right=0.94,
            top=0.87,
            bottom=0.10,
            hspace=0.48,
            height_ratios=(1.15, 0.85),
        )
        latency = figure.add_subplot(grid[0])
        differences = figure.add_subplot(grid[1])

        runs = performance["runs"]
        assert isinstance(runs, list)
        colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
        offsets = (-0.20, 0.0, 0.20)
        rows = list(range(len(runs)))
        for stage_index, ((key, label), offset) in enumerate(
            zip(STAGES, offsets, strict=True)
        ):
            color = colors[stage_index]
            for row, run in zip(rows, runs, strict=True):
                assert isinstance(run, dict)
                summary = _stage_summary(run, key)
                p50 = float(summary["p50"]) / 1_000_000
                p95 = float(summary["p95"]) / 1_000_000
                y = row + offset
                latency.plot(
                    [p50, p95],
                    [y, y],
                    color=color,
                    linewidth=2,
                    solid_capstyle="butt",
                    zorder=2,
                )
                latency.plot(
                    p50,
                    y,
                    marker="o",
                    markersize=6,
                    color=color,
                    linestyle="none",
                    label=label if row == 0 else None,
                    zorder=3,
                )
                latency.plot(
                    p95,
                    y,
                    marker="|",
                    markersize=10,
                    markeredgewidth=1.8,
                    color=color,
                    linestyle="none",
                    zorder=3,
                )

        for row, run in zip(rows, runs, strict=True):
            assert isinstance(run, dict)
            end = _stage_summary(run, "synchronized_end_to_end")
            p50 = float(end["p50"]) / 1_000_000
            p95 = float(end["p95"]) / 1_000_000
            label_x = p50 / 1.08 if p95 > 800 else p95 * 1.06
            latency.text(
                label_x,
                row + offsets[-1],
                f"{p50:.1f}–{p95:.1f} ms",
                va="center",
                ha="right" if p95 > 800 else "left",
                fontsize=10.5,
                clip_on=True,
            )

        latency.set_xscale("log")
        latency.set_xlim(1, 1800)
        latency.set_xticks((1, 3, 10, 30, 100, 300, 1000))
        latency.set_xticklabels(("1", "3", "10", "30", "100", "300", "1000"))
        latency.set_xlabel("Milliseconds (log scale)", labelpad=10)
        latency.set_ylim(-0.65, len(rows) - 0.35)
        latency.set_yticks(rows, TIMING_LABELS)
        latency.invert_yaxis()
        latency.grid(axis="x", which="major", linewidth=0.7, alpha=0.35)
        latency.spines[["top", "right"]].set_visible(False)
        latency.tick_params(axis="y", length=0, pad=10)
        latency.text(
            0,
            1.28,
            "A. Stage latency",
            transform=latency.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
        )
        latency.legend(
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            frameon=False,
            ncols=3,
            borderaxespad=0,
            handletextpad=0.5,
            columnspacing=1.4,
            title="p50 → p95",
        )

        results = scientific["scientific_results"]
        assert isinstance(results, dict)
        rates: list[float] = []
        for environment_id in CANDIDATE_ENVIRONMENTS:
            result = results[environment_id]
            assert isinstance(result, dict)
            discrete = result["discrete_components"]
            assert isinstance(discrete, dict)
            indices = discrete["nearest_bank_indices"]
            assert isinstance(indices, dict)
            count = int(indices["element_count"])
            mismatches = int(indices["mismatch_count"])
            rates.append(mismatches / count * 100)

        candidate_rows = list(range(len(CANDIDATE_ENVIRONMENTS)))
        differences.barh(candidate_rows, rates, height=0.52)
        for row, rate in zip(candidate_rows, rates, strict=True):
            differences.text(
                max(rate + 0.08, 0.08),
                row,
                f"{rate:.2f}%",
                va="center",
                fontsize=11.5,
            )
        differences.set_xlim(0, 5)
        differences.set_ylim(-0.65, len(candidate_rows) - 0.35)
        differences.set_yticks(candidate_rows, CANDIDATE_LABELS)
        differences.invert_yaxis()
        differences.set_xlabel("Different top-1 index (%)", labelpad=10)
        differences.grid(axis="x", linewidth=0.7, alpha=0.35)
        differences.set_axisbelow(True)
        differences.spines[["top", "right"]].set_visible(False)
        differences.tick_params(axis="y", length=0, pad=10)
        differences.text(
            0,
            1.16,
            "B. Top-1 index differences",
            transform=differences.transAxes,
            fontsize=18,
            fontweight="bold",
            va="bottom",
        )
        return figure


def _canonical_svg(figure: plt.Figure, metadata: str) -> bytes:
    buffer = io.BytesIO()
    with plt.rc_context(
        {
            "svg.fonttype": "none",
            "svg.hashsalt": "inspectrt-portability-latency-v2",
        }
    ):
        figure.savefig(
            buffer,
            format="svg",
            bbox_inches=None,
            facecolor="white",
            edgecolor="white",
            metadata={"Date": None, "Creator": None},
        )
    svg = buffer.getvalue().decode("utf-8")
    svg = re.sub(r"<!DOCTYPE svg[^>]*>\n", "", svg)
    svg = re.sub(r"\n <metadata>.*?</metadata>", "", svg, flags=re.DOTALL)
    svg = re.sub(
        r'width="[^"]+" height="[^"]+"',
        f'width="{SVG_WIDTH}" height="{SVG_HEIGHT}"',
        svg,
        count=1,
    )
    svg = svg.replace(
        "<svg ",
        '<svg role="img" aria-labelledby="title desc" ',
        1,
    )
    opening_end = svg.index(">", svg.index("<svg ")) + 1
    accessible = (
        '\n <title id="title">InspectRT cross-platform latency and nearest-neighbour '
        "index comparison</title>"
        '\n <desc id="desc">Panel A shows p50 to p95 pipeline stage latency for six '
        "environments. Panel B shows exact top-1 nearest-neighbour index "
        "differences for five completed candidates, including MPS.</desc>"
        f"\n <metadata>{escape(metadata)}</metadata>"
    )
    svg = svg[:opening_end] + accessible + svg[opening_end:]
    return (svg.rstrip() + "\n").encode()


def _render(
    scientific_bytes: bytes,
    performance_bytes: bytes,
    *,
    include_png: bool = False,
) -> tuple[bytes, bytes | None]:
    scientific_sha = hashlib.sha256(scientific_bytes).hexdigest()
    performance_sha = hashlib.sha256(performance_bytes).hexdigest()
    scientific = _document(scientific_bytes, "scientific evidence")
    performance = _document(performance_bytes, "performance evidence")
    _validate(scientific, performance, scientific_sha)
    metadata = json.dumps(
        {
            "comparison_id": scientific["comparison_id"],
            "performance_sha256": performance_sha,
            "renderer_id": RENDERER_ID,
            "renderer_version": RENDERER_VERSION,
            "schema_id": "inspectrt_portability_latency_graph_v1",
            "schema_version": 1,
            "scientific_sha256": scientific_sha,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    figure = _figure(scientific, performance)
    try:
        svg = _canonical_svg(figure, metadata)
        if not include_png:
            return svg, None
        buffer = io.BytesIO()
        figure.savefig(
            buffer,
            format="png",
            dpi=100,
            bbox_inches=None,
            facecolor="white",
            edgecolor="white",
            metadata={"Software": None},
        )
        return svg, buffer.getvalue()
    finally:
        plt.close(figure)


def render_svg(scientific_bytes: bytes, performance_bytes: bytes) -> bytes:
    """Validate both evidence records and return canonical SVG bytes."""
    return _render(scientific_bytes, performance_bytes)[0]


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--scientific", required=True, type=Path)
    parser.add_argument("--performance", required=True, type=Path)
    destination = parser.add_mutually_exclusive_group(required=True)
    destination.add_argument("--output", type=Path)
    destination.add_argument("--check", type=Path)
    parser.add_argument("--png-preview", type=Path)
    return parser


def main() -> int:
    parser = _parser()
    arguments = parser.parse_args()
    try:
        if arguments.check is not None and arguments.png_preview is not None:
            raise ValueError("--png-preview requires --output")
        svg, png = _render(
            arguments.scientific.read_bytes(),
            arguments.performance.read_bytes(),
            include_png=arguments.png_preview is not None,
        )
        if arguments.check is not None:
            if arguments.check.read_bytes() != svg:
                raise ValueError("tracked SVG does not match deterministic rendering")
            return 0
        output = arguments.output
        assert output is not None
        preview = arguments.png_preview
        if preview is not None and preview.resolve(strict=False) == output.resolve(
            strict=False
        ):
            raise ValueError("--output and --png-preview must differ")
        for path in (output, preview):
            if path is not None and path.exists():
                raise FileExistsError(f"output already exists: {path}")
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_bytes(svg)
        if preview is not None:
            assert png is not None
            preview.parent.mkdir(parents=True, exist_ok=True)
            preview.write_bytes(png)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
