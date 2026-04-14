from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path
from typing import Any

import numpy as np

from .elle_phasefield import (
    EllePhaseFieldConfig,
    load_elle_phasefield_state,
    phasefield_statistics,
    run_elle_phasefield_simulation,
    save_elle_phasefield_artifacts,
)


def _rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.sqrt(np.mean(diff * diff)))


def _mae(a: np.ndarray, b: np.ndarray) -> float:
    diff = np.asarray(a, dtype=np.float64) - np.asarray(b, dtype=np.float64)
    return float(np.mean(np.abs(diff)))


def compare_elle_phasefield_states(
    reference_theta,
    reference_temperature,
    candidate_theta,
    candidate_temperature,
) -> dict[str, Any]:
    reference_theta_np = np.asarray(reference_theta, dtype=np.float32)
    reference_temperature_np = np.asarray(reference_temperature, dtype=np.float32)
    candidate_theta_np = np.asarray(candidate_theta, dtype=np.float32)
    candidate_temperature_np = np.asarray(candidate_temperature, dtype=np.float32)

    if reference_theta_np.shape != candidate_theta_np.shape:
        raise ValueError("theta grids do not match")
    if reference_temperature_np.shape != candidate_temperature_np.shape:
        raise ValueError("temperature grids do not match")
    if reference_theta_np.shape != reference_temperature_np.shape:
        raise ValueError("reference theta/temperature shapes do not match")

    reference_stats = phasefield_statistics(reference_theta_np, reference_temperature_np)
    candidate_stats = phasefield_statistics(candidate_theta_np, candidate_temperature_np)
    reference_solid = reference_theta_np >= 0.5
    candidate_solid = candidate_theta_np >= 0.5
    solid_union = np.logical_or(reference_solid, candidate_solid)
    solid_intersection = np.logical_and(reference_solid, candidate_solid)
    intersection_over_union = (
        float(solid_intersection.sum() / solid_union.sum()) if np.any(solid_union) else 1.0
    )

    return {
        "grid_shape": [int(reference_theta_np.shape[0]), int(reference_theta_np.shape[1])],
        "theta_rmse": _rmse(reference_theta_np, candidate_theta_np),
        "theta_mae": _mae(reference_theta_np, candidate_theta_np),
        "theta_max_abs": float(np.max(np.abs(reference_theta_np - candidate_theta_np))),
        "theta_mean_signed": float(np.mean(candidate_theta_np - reference_theta_np)),
        "theta_solid_mismatch_fraction": float(np.mean(reference_solid != candidate_solid)),
        "theta_solid_iou": intersection_over_union,
        "temperature_rmse": _rmse(reference_temperature_np, candidate_temperature_np),
        "temperature_mae": _mae(reference_temperature_np, candidate_temperature_np),
        "temperature_max_abs": float(
            np.max(np.abs(reference_temperature_np - candidate_temperature_np))
        ),
        "temperature_mean_signed": float(
            np.mean(candidate_temperature_np - reference_temperature_np)
        ),
        "solid_fraction_reference": float(reference_stats["solid_fraction"]),
        "solid_fraction_candidate": float(candidate_stats["solid_fraction"]),
        "solid_fraction_delta": float(
            candidate_stats["solid_fraction"] - reference_stats["solid_fraction"]
        ),
        "interface_fraction_reference": float(reference_stats["interface_fraction"]),
        "interface_fraction_candidate": float(candidate_stats["interface_fraction"]),
        "interface_fraction_delta": float(
            candidate_stats["interface_fraction"] - reference_stats["interface_fraction"]
        ),
        "mean_temperature_reference": float(reference_stats["mean_temperature"]),
        "mean_temperature_candidate": float(candidate_stats["mean_temperature"]),
        "mean_temperature_delta": float(
            candidate_stats["mean_temperature"] - reference_stats["mean_temperature"]
        ),
    }


def compare_elle_phasefield_files(
    reference_path: str | Path,
    candidate_path: str | Path,
) -> dict[str, Any]:
    reference_theta, reference_temperature, _ = load_elle_phasefield_state(reference_path)
    candidate_theta, candidate_temperature, _ = load_elle_phasefield_state(candidate_path)
    result = compare_elle_phasefield_states(
        reference_theta,
        reference_temperature,
        candidate_theta,
        candidate_temperature,
    )
    result["reference_path"] = str(Path(reference_path))
    result["candidate_path"] = str(Path(candidate_path))
    return result


def _extract_snapshot_step(path: str | Path) -> int | None:
    match = re.search(r"(\d+)$", Path(path).stem)
    return int(match.group(1)) if match else None


def collect_elle_phasefield_snapshots(directory: str | Path) -> dict[int, Path]:
    directory_path = Path(directory)
    snapshots: dict[int, Path] = {}
    for path in sorted(directory_path.glob("*.elle")):
        step = _extract_snapshot_step(path)
        if step is not None:
            snapshots[step] = path
    return snapshots


def compare_elle_phasefield_sequences(
    reference_dir: str | Path,
    candidate_dir: str | Path,
) -> dict[str, Any]:
    reference = collect_elle_phasefield_snapshots(reference_dir)
    candidate = collect_elle_phasefield_snapshots(candidate_dir)
    matched_steps = sorted(set(reference) & set(candidate))
    missing_in_candidate = sorted(set(reference) - set(candidate))
    missing_in_reference = sorted(set(candidate) - set(reference))

    per_step = []
    for step in matched_steps:
        report = compare_elle_phasefield_files(reference[step], candidate[step])
        report["step"] = int(step)
        per_step.append(report)

    theta_rmses = [entry["theta_rmse"] for entry in per_step]
    temperature_rmses = [entry["temperature_rmse"] for entry in per_step]
    theta_ious = [entry["theta_solid_iou"] for entry in per_step]

    return {
        "reference_dir": str(Path(reference_dir)),
        "candidate_dir": str(Path(candidate_dir)),
        "matched_steps": [int(step) for step in matched_steps],
        "missing_in_candidate": [int(step) for step in missing_in_candidate],
        "missing_in_reference": [int(step) for step in missing_in_reference],
        "per_step": per_step,
        "summary": {
            "num_matched_steps": int(len(per_step)),
            "theta_rmse_mean": float(np.mean(theta_rmses)) if theta_rmses else None,
            "theta_rmse_max": float(np.max(theta_rmses)) if theta_rmses else None,
            "temperature_rmse_mean": float(np.mean(temperature_rmses))
            if temperature_rmses
            else None,
            "temperature_rmse_max": float(np.max(temperature_rmses))
            if temperature_rmses
            else None,
            "theta_solid_iou_mean": float(np.mean(theta_ious)) if theta_ious else None,
            "theta_solid_iou_min": float(np.min(theta_ious)) if theta_ious else None,
        },
    }


def inspect_elle_phasefield_binary(binary_path: str | Path) -> dict[str, Any]:
    binary = Path(binary_path)
    if not binary.exists():
        return {
            "binary_path": str(binary),
            "exists": False,
            "ready": False,
            "missing_libraries": [],
        }

    try:
        completed = subprocess.run(
            ["ldd", str(binary)],
            check=False,
            capture_output=True,
            text=True,
        )
    except FileNotFoundError:
        return {
            "binary_path": str(binary),
            "exists": True,
            "ready": False,
            "missing_libraries": ["ldd unavailable"],
        }

    missing = []
    for line in completed.stdout.splitlines():
        if "=> not found" in line:
            missing.append(line.split("=>", 1)[0].strip())

    return {
        "binary_path": str(binary),
        "exists": True,
        "ready": len(missing) == 0 and completed.returncode == 0,
        "missing_libraries": missing,
    }


def run_python_elle_phasefield_sequence(
    input_elle: str | Path,
    outdir: str | Path,
    *,
    config: EllePhaseFieldConfig | None = None,
    steps: int,
    save_every: int,
) -> dict[str, Any]:
    theta_init, temperature_init, template = load_elle_phasefield_state(input_elle)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    if config is None:
        config = EllePhaseFieldConfig(nx=int(theta_init.shape[0]), ny=int(theta_init.shape[1]))
    else:
        config = EllePhaseFieldConfig(
            **{
                **config.__dict__,
                "nx": int(theta_init.shape[0]),
                "ny": int(theta_init.shape[1]),
            }
        )

    saved_steps: list[int] = []

    def _save_snapshot(step: int, theta, temperature) -> None:
        save_elle_phasefield_artifacts(
            outdir_path,
            step,
            theta,
            temperature,
            save_preview=True,
            save_elle=True,
            elle_template=template,
        )
        saved_steps.append(int(step))

    run_elle_phasefield_simulation(
        config=config,
        steps=steps,
        save_every=save_every,
        on_snapshot=_save_snapshot,
        initial_state=(theta_init, temperature_init),
    )
    return {
        "input_elle": str(Path(input_elle)),
        "outdir": str(outdir_path),
        "saved_steps": saved_steps,
        "snapshots": [str(path) for path in sorted(outdir_path.glob("*.elle"))],
    }


def run_original_elle_phasefield_sequence(
    binary_path: str | Path,
    input_elle: str | Path,
    outdir: str | Path,
    *,
    steps: int,
    save_every: int,
    user_data: list[float] | None = None,
) -> dict[str, Any]:
    binary_status = inspect_elle_phasefield_binary(binary_path)
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    report: dict[str, Any] = {
        "binary_status": binary_status,
        "input_elle": str(Path(input_elle)),
        "outdir": str(outdir_path),
    }
    if not binary_status["ready"]:
        report["ran"] = False
        report["command"] = None
        report["stdout"] = ""
        report["stderr"] = ""
        report["snapshots"] = []
        return report

    command = [
        str(Path(binary_path)),
        "-i",
        str(Path(input_elle).resolve()),
        "-s",
        str(int(steps)),
        "-f",
        str(int(save_every)),
        "-n",
    ]
    if user_data:
        command.append("-u")
        command.extend(str(value) for value in user_data)

    completed = subprocess.run(
        command,
        cwd=outdir_path,
        check=False,
        capture_output=True,
        text=True,
    )
    report["ran"] = completed.returncode == 0
    report["returncode"] = int(completed.returncode)
    report["command"] = command
    report["stdout"] = completed.stdout
    report["stderr"] = completed.stderr
    report["snapshots"] = [str(path) for path in sorted(outdir_path.glob("phasefield*.elle"))]
    return report


def write_phasefield_comparison_report(path: str | Path, report: dict[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        json.dump(report, handle, indent=2)
    return path
