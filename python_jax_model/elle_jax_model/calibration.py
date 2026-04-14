from __future__ import annotations

import json
import re
from itertools import product
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .benchmark_validation import (
    collect_elle_label_component_snapshots,
    evaluate_rasterized_grain_growth_benchmark,
    evaluate_static_grain_growth_benchmark,
)
from .elle_export import write_unode_elle
from .simulation import GrainGrowthConfig, load_elle_label_seed
from .topology import run_simulation_with_topology


BEST_KNOWN_FINE_FOAM_PARAMS = {
    "dt": 0.025,
    "mobility": 1.0,
    "gradient_penalty": 1.25,
    "interaction_strength": 1.5,
}

BEST_KNOWN_FINE_FOAM_PRESET = {
    **BEST_KNOWN_FINE_FOAM_PARAMS,
    "init_smoothing_steps": 0,
    "init_noise": 0.0,
}

BEST_KNOWN_FINE_FOAM_CALIBRATED_PRESET = dict(BEST_KNOWN_FINE_FOAM_PRESET)

BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET = {
    **BEST_KNOWN_FINE_FOAM_PRESET,
    "mesh_relax_steps": 1,
    "mesh_topology_steps": 1,
    "mesh_movement_model": "elle_surface",
    "mesh_surface_diagonal_trials": True,
    "mesh_use_elle_physical_units": True,
    "mesh_update_mode": "mesh_only",
    "mesh_random_seed": 0,
    "mesh_feedback_every": 1,
    "mesh_feedback_strength": 0.15,
    "mesh_transport_strength": 0.5,
    "mesh_kernel_every": 0,
    "mesh_kernel_strength": 0.0,
    "mesh_kernel_corrector": False,
    "mesh_feedback_boundary_width": 1,
}


def _format_value_tag(value: float) -> str:
    text = f"{float(value):.4f}".rstrip("0").rstrip(".")
    return text.replace("-", "m").replace(".", "p")


def _metric_nrmse(metric: dict[str, Any] | None) -> float:
    if metric is None:
        return float("inf")
    return float(metric.get("normalized_rmse", float("inf")))


def _score_candidate(static_report: dict[str, Any], raster_report: dict[str, Any]) -> dict[str, Any]:
    raster_comparison = raster_report.get("comparison", {})
    static_comparison = static_report.get("comparison", {})

    grain_count_nrmse = _metric_nrmse(raster_comparison.get("grain_count"))
    mean_area_nrmse = _metric_nrmse(raster_comparison.get("mean_grain_area"))
    radius_nrmse = _metric_nrmse(raster_comparison.get("mean_equivalent_radius"))
    static_area_nrmse = _metric_nrmse(static_comparison.get("mean_grain_area"))

    score = (
        0.50 * grain_count_nrmse
        + 0.35 * mean_area_nrmse
        + 0.10 * radius_nrmse
        + 0.05 * static_area_nrmse
    )
    return {
        "score": float(score),
        "components": {
            "rasterized_grain_count_nrmse": grain_count_nrmse,
            "rasterized_mean_grain_area_nrmse": mean_area_nrmse,
            "rasterized_mean_equivalent_radius_nrmse": radius_nrmse,
            "static_mean_grain_area_nrmse": static_area_nrmse,
        },
    }


def _reference_seed(reference_sequence: list[dict[str, Any]]) -> dict[str, Any]:
    if not reference_sequence:
        raise ValueError("reference sequence is empty")

    snapshots_with_steps = [snapshot for snapshot in reference_sequence if snapshot.get("step") is not None]
    if not snapshots_with_steps:
        raise ValueError("reference sequence has no step-numbered ELLE snapshots")
    return min(snapshots_with_steps, key=lambda snapshot: int(snapshot["step"]))


def _as_float_list(values: Iterable[float]) -> list[float]:
    return [float(value) for value in values]


def _step_filename(reference_path: str | Path, step: int) -> str:
    path = Path(reference_path)
    match = re.search(r"(\d+)(?=\.elle$)", path.name)
    if match is None:
        return f"candidate_step{int(step):03d}.elle"
    width = len(match.group(1))
    return f"{path.name[:match.start(1)]}{int(step):0{width}d}{path.suffix}"


def _matched_step_count(report: dict[str, Any]) -> int:
    comparison = report.get("comparison", {})
    for metric_name in ("grain_count", "mean_grain_area", "mean_equivalent_radius"):
        metric = comparison.get(metric_name)
        if metric is not None:
            return int(len(metric.get("matched_steps", ())))
    return 0


def _run_coverage(reference_steps: list[int], report: dict[str, Any]) -> dict[str, Any]:
    matched_steps = _matched_step_count(report)
    expected = int(len(reference_steps))
    coverage = float(matched_steps / expected) if expected > 0 else 0.0
    return {
        "matched_steps": matched_steps,
        "expected_steps": expected,
        "coverage_fraction": coverage,
        "complete": bool(expected > 0 and matched_steps == expected),
    }


def _coverage_adjusted_score(base_score: float, coverage_fraction: float, penalty_weight: float) -> float:
    missing_fraction = max(0.0, 1.0 - float(coverage_fraction))
    return float(base_score + penalty_weight * missing_fraction)


def _parse_run_name(run_name: str) -> dict[str, float | None]:
    pattern = re.compile(
        r"^dt(?P<dt>[^_]+)_m(?P<mobility>[^_]+)_gp(?P<gradient_penalty>[^_]+)_is(?P<interaction_strength>[^_]+)$"
    )
    match = pattern.match(run_name)
    if match is None:
        return {
            "dt": None,
            "mobility": None,
            "gradient_penalty": None,
            "interaction_strength": None,
        }

    def _decode(text: str) -> float:
        return float(text.replace("m", "-").replace("p", "."))

    return {
        "dt": _decode(match.group("dt")),
        "mobility": _decode(match.group("mobility")),
        "gradient_penalty": _decode(match.group("gradient_penalty")),
        "interaction_strength": _decode(match.group("interaction_strength")),
    }


def generate_elle_seeded_candidate_sequence(
    *,
    reference_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.elle",
    dt: float,
    mobility: float,
    gradient_penalty: float,
    interaction_strength: float,
    seed: int = 0,
    init_elle_attribute: str = "auto",
    init_smoothing_steps: int = 0,
    init_noise: float = 0.0,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    reference_sequence = collect_elle_label_component_snapshots(
        reference_dir,
        pattern=pattern,
        attribute=init_elle_attribute,
    )
    seed_snapshot = _reference_seed(reference_sequence)
    reference_steps = sorted(int(snapshot["step"]) for snapshot in reference_sequence if snapshot.get("step") is not None)
    seed_info = load_elle_label_seed(seed_snapshot["path"], attribute=init_elle_attribute)

    outdir = Path(output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    saved_paths: list[str] = []
    target_steps = set(reference_steps)
    expected_paths = [outdir / _step_filename(seed_snapshot["path"], int(step)) for step in reference_steps]

    if reuse_existing and expected_paths and all(path.exists() for path in expected_paths):
        return {
            "reference_seed_path": str(seed_snapshot["path"]),
            "reference_steps": [int(step) for step in reference_steps],
            "seed_attribute": str(seed_info["attribute"]),
            "grid_shape": [int(seed_info["grid_shape"][0]), int(seed_info["grid_shape"][1])],
            "num_grains": int(seed_info["num_labels"]),
            "candidate_dir": str(outdir),
            "saved_snapshots": [str(path) for path in expected_paths],
            "reused_existing": True,
        }

    config = GrainGrowthConfig(
        nx=int(seed_info["grid_shape"][0]),
        ny=int(seed_info["grid_shape"][1]),
        num_grains=int(seed_info["num_labels"]),
        dt=float(dt),
        mobility=float(mobility),
        gradient_penalty=float(gradient_penalty),
        interaction_strength=float(interaction_strength),
        seed=int(seed),
        init_mode="elle",
        init_elle_path=str(seed_snapshot["path"]),
        init_elle_attribute=str(seed_info["attribute"]),
        init_smoothing_steps=int(init_smoothing_steps),
        init_noise=float(init_noise),
    )

    def _save_snapshot(step: int, phi, topology_snapshot, _mesh_feedback_context=None) -> None:
        if int(step) not in target_steps:
            return
        outpath = outdir / _step_filename(seed_snapshot["path"], int(step))
        write_unode_elle(outpath, np.asarray(phi), step=int(step), tracked_topology=topology_snapshot)
        saved_paths.append(str(outpath))

    run_simulation_with_topology(
        config=config,
        steps=max(reference_steps),
        save_every=1,
        on_snapshot=_save_snapshot,
    )

    return {
        "reference_seed_path": str(seed_snapshot["path"]),
        "reference_steps": [int(step) for step in reference_steps],
        "seed_attribute": str(seed_info["attribute"]),
        "grid_shape": [int(seed_info["grid_shape"][0]), int(seed_info["grid_shape"][1])],
        "num_grains": int(seed_info["num_labels"]),
        "candidate_dir": str(outdir),
        "saved_snapshots": saved_paths,
        "reused_existing": False,
    }


def calibrate_fine_foam(
    *,
    reference_dir: str | Path,
    output_dir: str | Path,
    pattern: str = "*.elle",
    dt_grid: Iterable[float],
    mobility_grid: Iterable[float],
    gradient_penalty_grid: Iterable[float] = (1.0,),
    interaction_strength_grid: Iterable[float] = (2.0,),
    seed: int = 0,
    init_elle_attribute: str = "auto",
    init_smoothing_steps: int = 0,
    init_noise: float = 0.0,
    reuse_existing: bool = True,
) -> dict[str, Any]:
    dt_values = _as_float_list(dt_grid)
    mobility_values = _as_float_list(mobility_grid)
    gradient_values = _as_float_list(gradient_penalty_grid)
    interaction_values = _as_float_list(interaction_strength_grid)
    if not dt_values or not mobility_values or not gradient_values or not interaction_values:
        raise ValueError("all calibration grids must contain at least one value")

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)
    reference_sequence = collect_elle_label_component_snapshots(
        reference_dir,
        pattern=pattern,
        attribute=init_elle_attribute,
    )
    seed_snapshot = _reference_seed(reference_sequence)
    runs: list[dict[str, Any]] = []

    for dt, mobility, gradient_penalty, interaction_strength in product(
        dt_values,
        mobility_values,
        gradient_values,
        interaction_values,
    ):
        run_name = (
            f"dt{_format_value_tag(dt)}_"
            f"m{_format_value_tag(mobility)}_"
            f"gp{_format_value_tag(gradient_penalty)}_"
            f"is{_format_value_tag(interaction_strength)}"
        )
        candidate_dir = output_root / run_name
        generation = generate_elle_seeded_candidate_sequence(
            reference_dir=reference_dir,
            output_dir=candidate_dir,
            pattern=pattern,
            dt=dt,
            mobility=mobility,
            gradient_penalty=gradient_penalty,
            interaction_strength=interaction_strength,
            seed=seed,
            init_elle_attribute=init_elle_attribute,
            init_smoothing_steps=init_smoothing_steps,
            init_noise=init_noise,
            reuse_existing=reuse_existing,
        )
        static_report = evaluate_static_grain_growth_benchmark(
            reference_dir,
            candidate_dir=candidate_dir,
            pattern=pattern,
        )
        raster_report = evaluate_rasterized_grain_growth_benchmark(
            reference_dir,
            candidate_dir=candidate_dir,
            pattern=pattern,
            attribute=init_elle_attribute,
        )
        score = _score_candidate(static_report, raster_report)
        runs.append(
            {
                "run_name": run_name,
                "params": {
                    "dt": float(dt),
                    "mobility": float(mobility),
                    "gradient_penalty": float(gradient_penalty),
                    "interaction_strength": float(interaction_strength),
                    "seed": int(seed),
                    "init_smoothing_steps": int(init_smoothing_steps),
                    "init_noise": float(init_noise),
                },
                "candidate_dir": str(candidate_dir),
                "generation": generation,
                "score": score,
                "static_comparison": static_report.get("comparison", {}),
                "rasterized_comparison": raster_report.get("comparison", {}),
            }
        )

    runs.sort(key=lambda item: (float(item["score"]["score"]), item["run_name"]))
    best_run = runs[0] if runs else None
    return {
        "reference_dir": str(Path(reference_dir)),
        "pattern": pattern,
        "reference_seed_path": str(seed_snapshot["path"]),
        "reference_steps": [int(snapshot["step"]) for snapshot in reference_sequence if snapshot.get("step") is not None],
        "search_space": {
            "dt_grid": dt_values,
            "mobility_grid": mobility_values,
            "gradient_penalty_grid": gradient_values,
            "interaction_strength_grid": interaction_values,
        },
        "num_runs": int(len(runs)),
        "runs": runs,
        "best_run": best_run,
    }


def score_existing_calibration_runs(
    *,
    reference_dir: str | Path,
    candidate_root: str | Path,
    pattern: str = "*.elle",
    init_elle_attribute: str = "auto",
    require_complete: bool = False,
    coverage_penalty_weight: float = 0.25,
) -> dict[str, Any]:
    reference_sequence = collect_elle_label_component_snapshots(
        reference_dir,
        pattern=pattern,
        attribute=init_elle_attribute,
    )
    reference_steps = [int(snapshot["step"]) for snapshot in reference_sequence if snapshot.get("step") is not None]
    candidate_root_path = Path(candidate_root)
    runs: list[dict[str, Any]] = []

    for candidate_dir in sorted(path for path in candidate_root_path.iterdir() if path.is_dir()):
        static_report = evaluate_static_grain_growth_benchmark(
            reference_dir,
            candidate_dir=candidate_dir,
            pattern=pattern,
        )
        raster_report = evaluate_rasterized_grain_growth_benchmark(
            reference_dir,
            candidate_dir=candidate_dir,
            pattern=pattern,
            attribute=init_elle_attribute,
        )
        base_score = _score_candidate(static_report, raster_report)
        coverage = _run_coverage(reference_steps, raster_report)
        if require_complete and not coverage["complete"]:
            continue
        adjusted_score = _coverage_adjusted_score(
            base_score["score"],
            coverage["coverage_fraction"],
            coverage_penalty_weight,
        )
        runs.append(
            {
                "run_name": candidate_dir.name,
                "params": _parse_run_name(candidate_dir.name),
                "candidate_dir": str(candidate_dir),
                "coverage": coverage,
                "score": {
                    **base_score,
                    "coverage_adjusted_score": adjusted_score,
                    "coverage_penalty_weight": float(coverage_penalty_weight),
                },
                "static_comparison": static_report.get("comparison", {}),
                "rasterized_comparison": raster_report.get("comparison", {}),
            }
        )

    runs.sort(
        key=lambda item: (
            float(item["score"]["coverage_adjusted_score"]),
            -float(item["coverage"]["coverage_fraction"]),
            item["run_name"],
        )
    )
    best_run = runs[0] if runs else None
    return {
        "reference_dir": str(Path(reference_dir)),
        "candidate_root": str(candidate_root_path),
        "pattern": pattern,
        "reference_steps": reference_steps,
        "num_runs": int(len(runs)),
        "coverage_penalty_weight": float(coverage_penalty_weight),
        "require_complete": bool(require_complete),
        "runs": runs,
        "best_run": best_run,
    }


def write_calibration_report(path: str | Path, report: dict[str, Any]) -> Path:
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return outpath
