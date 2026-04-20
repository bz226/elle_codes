from __future__ import annotations

import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from typing import Any
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portable_elle_viewer import build_viewer_payload

from run_simulation import resolve_mesh_preset, resolve_runtime_preset
from validate_benchmarks import (
    _discover_experiment_family_dirs,
    _discover_experiment_family_report_paths,
    _load_experiment_family_manifest,
    _load_experiment_family_report_manifest,
    _resolve_experiment_family_dirs,
    _resolve_experiment_family_report_paths,
)
import elle_jax_model.mesh as mesh_module
import elle_jax_model.nucleation as nucleation_module
import elle_jax_model.recovery as recovery_module
from elle_jax_model.artifacts import dominant_grain_map, save_snapshot_artifacts, snapshot_statistics
from elle_jax_model.benchmark_validation import (
    assess_benchmark_report_acceptance,
    assess_experiment_family_signature_alignment,
    assess_paper_signature_alignment,
    build_benchmark_validation_report,
    evaluate_experiment_family_benchmarks,
    evaluate_experiment_family_benchmark_reports,
    evaluate_experiment_family_suite,
    evaluate_release_dataset_benchmarks,
    evaluate_rasterized_grain_growth_benchmark,
    evaluate_static_grain_growth_benchmark,
    load_experiment_family_benchmark_report,
    summarize_elle_label_components,
    summarize_sequence_trends,
)
from elle_jax_model.calibration import (
    BEST_KNOWN_FINE_FOAM_CALIBRATED_PRESET,
    BEST_KNOWN_FINE_FOAM_PARAMS,
    BEST_KNOWN_FINE_FOAM_PRESET,
    BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET,
    calibrate_fine_foam,
    generate_elle_seeded_candidate_sequence,
    score_existing_calibration_runs,
)
from elle_jax_model.elle_export import extract_flynn_topology, write_unode_elle
from elle_jax_model.fft_bridge import (
    apply_legacy_fft_snapshot_to_mesh_state,
    build_legacy_elle2fft_bridge_payload,
    build_legacy_fft_bridge_payload,
    compare_applied_legacy_fft_snapshot_to_mesh_state,
    compare_legacy_elle2fft_bridge_payload,
    diagnose_legacy_elle2fft_header_sources,
    LegacyFFTImportOptions,
    load_legacy_elle2fft_bridge_payload,
    load_legacy_fft_snapshot,
    load_legacy_fft_snapshot_sequence,
    write_legacy_elle2fft_bridge_payload,
)
from elle_jax_model.gbm_faithful import (
    FAITHFUL_GBM_DEFAULTS,
    build_faithful_gbm_setup,
    run_faithful_gbm_simulation,
)
from elle_jax_model.legacy_reference import (
    build_legacy_reference_bundle,
    compare_legacy_reference_bundle,
    compare_legacy_reference_snapshot,
    compare_legacy_reference_swept_unode_transition,
    compare_legacy_reference_transition,
    extract_legacy_reference_snapshot,
    extract_legacy_reference_swept_unode_transition,
    extract_legacy_reference_transition,
    load_legacy_reference_bundle,
)
from elle_jax_model.legacy_statistics import (
    compare_snapshot_summary_to_legacy_statistics,
    compare_mesh_bookkeeping_to_legacy_old_stats,
    load_legacy_allout_statistics,
    load_legacy_last_stats_summary,
    load_legacy_old_stats_summary,
    load_legacy_statistics_summary,
    load_legacy_tmpstats_summary,
    summarize_current_mesh_bookkeeping,
)
from elle_jax_model.mechanics_replay import (
    run_faithful_mechanics_replay,
    validate_faithful_mechanics_outerstep_transition,
    validate_faithful_mechanics_transition,
)
from elle_jax_model.elle_phasefield import (
    EllePhaseFieldConfig,
    initialize_elle_phasefield,
    load_elle_phasefield_state,
    phasefield_statistics,
    run_elle_phasefield_simulation,
    save_elle_phasefield_artifacts,
    write_elle_phasefield_state,
)
from elle_jax_model.figure2_validation import (
    build_figure2_line_validation_report,
    summarize_elle_label_area_distribution,
    write_figure2_line_validation_html,
)
from elle_jax_model.faithful_config import load_faithful_seed
from elle_jax_model.faithful_runtime import (
    _apply_initial_raw_seed_topology_pass,
    _current_runtime_labels,
)
from elle_jax_model.elle_html_viewer import write_elle_html_viewer
from elle_jax_model.elle_visualize import _parse_elle_sections, render_elle_file
from elle_jax_model.phasefield_compare import (
    compare_elle_phasefield_sequences,
    compare_elle_phasefield_files,
    compare_elle_phasefield_states,
    inspect_elle_phasefield_binary,
    run_original_elle_phasefield_sequence,
)
from elle_jax_model.mesh import (
    _build_edge_boundary_energy_lookup,
    _build_edge_mobility_lookup,
    _apply_segment_mass_partition,
    _enforce_connected_label_ownership,
    _elle_surface_effective_dt,
    _elle_surface_velocity_from_force,
    _entry_partition_terms,
    _find_split_pair,
    _move_node_elle_surface,
    _partition_mass_node,
    _segment_swept_records,
    _surface_force_from_trial_energies,
    _trial_node_energy,
    MeshFeedbackConfig,
    MeshRelaxationConfig,
    assign_seed_unodes_from_mesh,
    apply_mesh_feedback,
    apply_mesh_transport,
    build_mesh_state,
    compute_mesh_motion_velocity,
    couple_mesh_to_order_parameters,
    incremental_seed_unode_reassignment,
    load_elle_mesh_seed,
    mesh_labels_to_order_parameters,
    rasterize_mesh_labels,
    relax_mesh_state,
    update_seed_node_fields,
    update_seed_unode_fields,
    update_seed_unode_sections,
)
from elle_jax_model.mobility import (
    arrhenius_boundary_mobility,
    boundary_segment_energy,
    boundary_segment_mobility,
    caxis_misorientation_degrees,
    load_phase_boundary_db,
    misorientation_mobility_reduction,
)
from elle_jax_model.nucleation import NucleationConfig, apply_nucleation_stage
from elle_jax_model.microstructure_validation import (
    collect_elle_microstructure_snapshots,
    compare_elle_microstructure_sequences,
    summarize_grain_survival_diagnostics,
    summarize_elle_microstructure,
    summarize_liu_suckale_datasets,
)
from elle_jax_model.paper_validation import (
    assess_current_rewrite_against_papers,
    summarize_liu_suckale_paper_from_text,
    summarize_llorens_structure_from_text,
)
from elle_jax_model.recovery import RecoveryConfig, apply_recovery_stage
from elle_jax_model.simulation import (
    GrainGrowthConfig,
    initialize_order_parameters,
    load_elle_label_seed,
    run_mesh_only_simulation,
    run_simulation,
)
from elle_jax_model.topology import TopologyTracker, run_simulation_with_topology


def _read_ppm_size(path: Path) -> tuple[int, int]:
    with path.open("rb") as handle:
        magic = handle.readline().strip()
        dimensions = handle.readline().strip().split()
        _maxval = handle.readline().strip()
    if magic != b"P6" or len(dimensions) != 2:
        raise ValueError(f"unexpected PPM header in {path}")
    return int(dimensions[0]), int(dimensions[1])


def _write_scattered_unode_example(path: Path) -> Path:
    path.write_text(
        "\n".join(
            [
                "UNODES",
                "1 0.10 0.10",
                "2 0.25 0.62",
                "3 0.58 0.31",
                "4 0.91 0.84",
                "U_ATTRIB_A",
                "Default 0.0",
                "1 0",
                "2 1",
                "3 2",
                "4 3",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return path


def _write_periodic_flynn_example(path: Path) -> Path:
    lines = [
        "OPTIONS",
        "CellBoundingBox 0.0 0.0",
        "1.0 0.0",
        "1.0 1.0",
        "0.0 1.0",
        "LOCATION",
        "1 0.95 0.40",
        "2 0.98 0.60",
        "3 0.02 0.60",
        "4 0.05 0.40",
        "FLYNNS",
        "1 4 1 2 3 4",
        "EULER_3",
        "1 30.0",
        "UNODES",
    ]
    unode_id = 1
    for iy in range(4):
        for ix in range(4):
            x_coord = 0.125 + 0.25 * ix
            y_coord = 0.125 + 0.25 * iy
            lines.append(f"{unode_id} {x_coord:.3f} {y_coord:.3f}")
            unode_id += 1
    lines.extend(
        [
            "U_ATTRIB_A",
            "Default 0.0",
            "1 1",
            "6 2",
            "11 3",
            "16 4",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_fabric_euler_example(path: Path) -> Path:
    lines = [
        "OPTIONS",
        "CellBoundingBox 0.0 0.0",
        "1.0 0.0",
        "1.0 1.0",
        "0.0 1.0",
        "LOCATION",
        "1 0.0 0.0",
        "2 1.0 0.0",
        "3 1.0 1.0",
        "4 0.0 1.0",
        "FLYNNS",
        "1 4 1 2 3 4",
        "UNODES",
        "1 0.25 0.25",
        "2 0.25 0.75",
        "3 0.75 0.25",
        "4 0.75 0.75",
        "U_ATTRIB_A",
        "Default 0.0",
        "1 1",
        "2 1",
        "3 1",
        "4 1",
        "U_EULER_3",
        "Default 0.0 0.0 0.0",
        "1 0.0 0.0 0.0",
        "2 0.0 0.0 0.0",
        "3 0.0 0.0 0.0",
        "4 0.0 0.0 0.0",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_aspect_ratio_example(path: Path) -> Path:
    lines = [
        "OPTIONS",
        "CellBoundingBox 0.0 0.0",
        "1.0 0.0",
        "1.0 1.0",
        "0.0 1.0",
        "LOCATION",
        "1 0.1 0.2",
        "2 0.9 0.2",
        "3 0.9 0.6",
        "4 0.1 0.6",
        "FLYNNS",
        "1 4 1 2 3 4",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_mechanics_scalar_example(
    path: Path,
    *,
    strain_values: tuple[float, float, float, float] = (2.0, 4.0, 6.0, 8.0),
    stress_values: tuple[float, float, float, float] = (3.0, 5.0, 7.0, 9.0),
    basal_values: tuple[float, float, float, float] = (1.0, 2.0, 3.0, 4.0),
    prismatic_values: tuple[float, float, float, float] = (2.0, 4.0, 6.0, 8.0),
) -> Path:
    lines = [
        "OPTIONS",
        "CellBoundingBox 0.0 0.0",
        "1.0 0.0",
        "1.0 1.0",
        "0.0 1.0",
        "LOCATION",
        "1 0.0 0.0",
        "2 1.0 0.0",
        "3 1.0 1.0",
        "4 0.0 1.0",
        "FLYNNS",
        "1 4 1 2 3 4",
        "UNODES",
        "1 0.25 0.25",
        "2 0.25 0.75",
        "3 0.75 0.25",
        "4 0.75 0.75",
        "U_ATTRIB_A",
        "Default 0.0",
        f"1 {strain_values[0]}",
        f"2 {strain_values[1]}",
        f"3 {strain_values[2]}",
        f"4 {strain_values[3]}",
        "U_ATTRIB_B",
        "Default 0.0",
        f"1 {stress_values[0]}",
        f"2 {stress_values[1]}",
        f"3 {stress_values[2]}",
        f"4 {stress_values[3]}",
        "U_ATTRIB_D",
        "Default 0.0",
        f"1 {basal_values[0]}",
        f"2 {basal_values[1]}",
        f"3 {basal_values[2]}",
        f"4 {basal_values[3]}",
        "U_ATTRIB_E",
        "Default 0.0",
        f"1 {prismatic_values[0]}",
        f"2 {prismatic_values[1]}",
        f"3 {prismatic_values[2]}",
        f"4 {prismatic_values[3]}",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_mechanics_sequence_examples(directory: Path) -> tuple[Path, Path]:
    step1 = _write_mechanics_scalar_example(directory / "mechanics_0001.elle")
    step2 = _write_mechanics_scalar_example(
        directory / "mechanics_0002.elle",
        strain_values=(4.0, 5.0, 6.0, 7.0),
        stress_values=(2.0, 4.0, 6.0, 8.0),
        basal_values=(1.0, 2.0, 3.0, 4.0),
        prismatic_values=(3.0, 4.0, 5.0, 6.0),
    )
    return step1, step2


def _write_mechanics_mesh_sidecar(
    elle_path: Path,
    *,
    cumulative_simple_shear: float,
    simple_shear_increment: float | None = None,
    direct_strain_axis: float | None = None,
    strain_axis_source: str | None = None,
    mean_normalized_strain_rate: float = 5.0,
    mean_normalized_stress: float = 6.0,
    mean_differential_stress: float | None = None,
    mean_basal_activity: float = 2.5,
    mean_prismatic_activity: float = 5.0,
    mean_pyramidal_activity: float | None = None,
) -> Path:
    step_match = re.search(r"(\d+)(?=\.elle$)", elle_path.name)
    if step_match is None:
        raise ValueError(f"expected numeric step suffix in {elle_path}")
    step_suffix = step_match.group(1)
    if mean_pyramidal_activity is None:
        mean_pyramidal_activity = 0.0
    total_activity = float(mean_basal_activity + mean_prismatic_activity + mean_pyramidal_activity)
    prismatic_fraction = float(mean_prismatic_activity / total_activity) if abs(total_activity) > 1.0e-12 else 0.0
    prismatic_to_basal = float(mean_prismatic_activity / mean_basal_activity) if abs(mean_basal_activity) > 1.0e-12 else float("nan")
    if simple_shear_increment is None:
        simple_shear_increment = float(cumulative_simple_shear)
    if direct_strain_axis is None:
        direct_strain_axis = float(cumulative_simple_shear)
    if strain_axis_source is None:
        strain_axis_source = "cumulative_simple_shear"
    payload = {
        "stats": {
            "mechanics_snapshot_cumulative_simple_shear": float(cumulative_simple_shear),
            "mechanics_snapshot_simple_shear_increment": float(simple_shear_increment),
            "mechanics_snapshot_simple_shear_offset": float(cumulative_simple_shear),
            "mechanics_snapshot_direct_strain_axis": float(direct_strain_axis),
            "mechanics_snapshot_strain_axis_source": str(strain_axis_source),
        },
        "mechanics_payload_summary": {
            "source": "legacy_fft_snapshot",
            "has_tex": 1,
            "mean_normalized_strain_rate": float(mean_normalized_strain_rate),
            "mean_normalized_stress": float(mean_normalized_stress),
            "mean_differential_stress": (
                None if mean_differential_stress is None else float(mean_differential_stress)
            ),
            "mean_basal_activity": float(mean_basal_activity),
            "mean_prismatic_activity": float(mean_prismatic_activity),
            "mean_pyramidal_activity": float(mean_pyramidal_activity),
            "mean_total_activity": float(total_activity),
            "mean_prismatic_fraction": float(prismatic_fraction),
            "prismatic_to_basal_ratio": float(prismatic_to_basal),
            "simple_shear_increment": float(simple_shear_increment),
            "simple_shear_offset": float(cumulative_simple_shear),
            "cumulative_simple_shear": float(cumulative_simple_shear),
            "direct_strain_axis": float(direct_strain_axis),
            "strain_axis_source": str(strain_axis_source),
        },
    }
    mesh_path = elle_path.with_name(f"mesh_{step_suffix}.json")
    mesh_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return mesh_path


def _write_legacy_allout_example(
    path: Path,
    *,
    rows: list[tuple[float, ...]],
) -> Path:
    header = (
        "SVM DVM diffStress stressFieldErr strainrateFieldErr "
        "basalact prismact pyramact s11 s22 s33 s23 s13 s12 "
        "d11 d22 d33 d23 d13 d12"
    )
    line_text = []
    for row in rows:
        if len(row) != 20:
            raise ValueError(f"expected 20 values per AllOutData row, got {len(row)}")
        line_text.append(
            header
            + " "
            + " ".join(f"{float(value):.6e}" for value in row)
        )
    path.write_text("\n".join(line_text) + "\n", encoding="utf-8")
    return path


def _make_experiment_family_static_report(
    *,
    mean_grain_area_end: float,
    mean_grain_area_relative_delta: float,
    mean_aspect_ratio_end: float,
    mean_differential_stress_end: float,
    peak_stress_strain: float,
    c_axis_p_delta: float,
    c_axis_vertical_delta: float,
    size_stronger_than_schmid: bool = True,
) -> dict[str, Any]:
    return {
        "reference_dir": "synthetic_family",
        "reference_trends": {
            "metrics": {
                "mean_grain_area": {
                    "delta": float(mean_grain_area_relative_delta),
                    "relative_delta": float(mean_grain_area_relative_delta),
                    "end": float(mean_grain_area_end),
                },
                "mean_aspect_ratio": {
                    "delta": -1.0,
                    "end": float(mean_aspect_ratio_end),
                },
                "fabric_c_axis_p_index": {
                    "delta": float(c_axis_p_delta),
                    "end": float(c_axis_p_delta),
                },
                "fabric_c_axis_vertical_fraction_15deg": {
                    "delta": float(c_axis_vertical_delta),
                    "end": float(c_axis_vertical_delta),
                },
                "mechanics_mean_differential_stress": {
                    "delta": -1.0,
                    "end": float(mean_differential_stress_end),
                },
            },
            "curves": {
                "mechanics_stress_strain": {
                    "steps": [1, 2, 3],
                    "strain": [0.0, float(peak_stress_strain), float(peak_stress_strain) + 0.1],
                    "stress": [1.0, 2.0, 1.5],
                }
            },
        },
        "reference_survival_diagnostics": {
            "summary": {
                "initial_size_correlation_stronger_than_schmid": bool(size_stronger_than_schmid),
            }
        },
    }


def _write_experiment_family_benchmark_report(path: Path, report: dict[str, Any]) -> Path:
    payload = {
        "static_grain_growth": report,
        "rasterized_grain_growth": {},
        "release_dataset_benchmarks": {},
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def _write_survival_sequence_examples(directory: Path) -> tuple[Path, Path]:
    step1 = directory / "survival_0001.elle"
    step2 = directory / "survival_0002.elle"
    common_prefix = [
        "OPTIONS",
        "CellBoundingBox 0.0 0.0",
        "1.0 0.0",
        "1.0 1.0",
        "0.0 1.0",
        "UNODES",
        "1 0.167 0.167",
        "2 0.500 0.167",
        "3 0.833 0.167",
        "4 0.167 0.500",
        "5 0.500 0.500",
        "6 0.833 0.500",
        "7 0.167 0.833",
        "8 0.500 0.833",
        "9 0.833 0.833",
    ]
    step1_lines = common_prefix + [
        "U_ATTRIB_C",
        "Default 1.0",
        "6 2.0",
        "9 2.0",
        "8 3.0",
        "U_EULER_3",
        "Default 0.0 0.0 0.0",
        "1 0.0 0.0 0.0",
        "2 0.0 0.0 0.0",
        "3 0.0 0.0 0.0",
        "4 0.0 0.0 0.0",
        "5 0.0 0.0 0.0",
        "6 90.0 45.0 0.0",
        "7 0.0 0.0 0.0",
        "8 90.0 20.0 0.0",
        "9 90.0 45.0 0.0",
        "",
    ]
    step2_lines = common_prefix + [
        "U_ATTRIB_C",
        "Default 1.0",
        "6 2.0",
        "9 2.0",
        "U_EULER_3",
        "Default 0.0 0.0 0.0",
        "1 0.0 0.0 0.0",
        "2 0.0 0.0 0.0",
        "3 0.0 0.0 0.0",
        "4 0.0 0.0 0.0",
        "5 0.0 0.0 0.0",
        "6 90.0 45.0 0.0",
        "7 0.0 0.0 0.0",
        "8 0.0 0.0 0.0",
        "9 90.0 45.0 0.0",
        "",
    ]
    step1.write_text("\n".join(step1_lines), encoding="utf-8")
    step2.write_text("\n".join(step2_lines), encoding="utf-8")
    return step1, step2


def _write_elle_mesh_seed_example(path: Path) -> Path:
    lines = [
        "OPTIONS",
        "SwitchDistance 0.05",
        "MaxNodeSeparation 0.11",
        "MinNodeSeparation 0.05",
        "Timestep 3.0",
        "SpeedUp 2.0",
        "UnitLength 0.5",
        "Temperature -10.0",
        "Pressure 8800000.0",
        "BoundaryWidth 0.1",
        "MassIncrement 0.02",
        "LOCATION",
        "10 0.00 0.00",
        "11 0.50 0.00",
        "12 0.50 1.00",
        "13 0.00 1.00",
        "14 1.00 0.00",
        "15 1.00 1.00",
        "N_ATTRIB_A",
        "Default 0.0",
        "10 1.5",
        "11 2.5",
        "12 3.5",
        "13 4.5",
        "14 5.5",
        "15 6.5",
        "FLYNNS",
        "10 4 10 11 12 13",
        "20 4 11 14 15 12",
        "F_ATTRIB_C",
        "Default 0.0",
        "10 100",
        "20 200",
        "UNODES",
        "1 0.25 0.25",
        "2 0.25 0.75",
        "3 0.75 0.25",
        "4 0.75 0.75",
        "U_ATTRIB_A",
        "Default 0.0",
        "1 1.0",
        "2 2.0",
        "3 3.0",
        "4 4.0",
        "U_ATTRIB_C",
        "Default 0.0",
        "1 100",
        "2 100",
        "3 200",
        "4 200",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")
    return path


def _write_legacy_fft_snapshot_example(
    root: Path,
    *,
    zero_based_ids: bool = False,
    value_offset: float = 0.0,
    temp_rows: tuple[tuple[float, float, float], ...] | None = None,
    unodexyz_rows: tuple[tuple[float, float, float], ...] | None = None,
) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    base_ids = [0, 1, 2, 3] if zero_based_ids else [1, 2, 3, 4]
    if temp_rows is None:
        temp_rows = (
            (1.0, 0.0, 0.0),
            (0.0, 1.0, 0.0),
            (0.0, 0.0, 1.0),
        )
    (root / "temp-FFT.out").write_text(
        "\n".join(
            f"{float(row[0]):g} {float(row[1]):g} {float(row[2]):g}" for row in temp_rows
        ),
        encoding="utf-8",
    )
    (root / "unodexyz.out").write_text(
        "\n".join(
            (
                f"{unode_id} {float(unodexyz_rows[index][0]):g} {float(unodexyz_rows[index][1]):g} {float(unodexyz_rows[index][2]):g}"
                if unodexyz_rows is not None
                else f"{unode_id} {0.1 * (index + 1) + value_offset:g} {0.2 * (index + 1) + value_offset:g} {0.3 * (index + 1) + value_offset:g}"
            )
            for index, unode_id in enumerate(base_ids)
        ),
        encoding="utf-8",
    )
    (root / "unodeang.out").write_text(
        "\n".join(
            f"{unode_id} {10 * (index + 1) + value_offset:g} {20 * (index + 1) + value_offset:g} {30 * (index + 1) + value_offset:g}"
            for index, unode_id in enumerate(base_ids)
        ),
        encoding="utf-8",
    )
    (root / "tex.out").write_text(
        "\n".join(
            f"{unode_id} 0 0 0 {4 + 10 * index + value_offset:g} {5 + 10 * index + value_offset:g} {6 + 10 * index + value_offset:g} {7 + 10 * index + value_offset:g} {8 + 10 * index + value_offset:g} {9 + 10 * index + value_offset:g} {10 + 10 * index + value_offset:g} {11 + 10 * index + value_offset:g}"
            for index, unode_id in enumerate(base_ids)
        ),
        encoding="utf-8",
    )
    return root


def _build_recovery_mesh_state(
    *,
    include_attr_f: bool = False,
    attr_f_value: float = 0.0,
    euler_values: tuple[tuple[float, float, float], ...] | None = None,
) -> dict[str, object]:
    field_values: dict[str, tuple[float, ...]] = {
        "U_DISLOCDEN": (10.0, 10.0, 10.0, 10.0),
    }
    field_order: list[str] = ["U_DISLOCDEN"]
    if include_attr_f:
        field_values["U_ATTRIB_F"] = tuple(float(attr_f_value) for _ in range(4))
        field_order.append("U_ATTRIB_F")
    if euler_values is None:
        euler_values = (
            (2.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
            (0.0, 0.0, 0.0),
        )
    return {
        "_runtime_seed_unodes": {
            "ids": (1, 2, 3, 4),
            "positions": (
                (0.25, 0.25),
                (0.25, 0.75),
                (0.75, 0.25),
                (0.75, 0.75),
            ),
            "grid_indices": (
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ),
            "grid_shape": (2, 2),
        },
        "_runtime_seed_unode_fields": {
            "field_order": tuple(field_order),
            "values": field_values,
        },
        "_runtime_seed_unode_sections": {
            "field_order": ("U_EULER_3",),
            "component_counts": {"U_EULER_3": 3},
            "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
            "values": {
                "U_EULER_3": tuple(tuple(float(component) for component in values) for values in euler_values)
            },
        },
        "_runtime_seed_flynn_sections": {
            "field_order": ("EULER_3", "DISLOCDEN"),
            "id_order": (10,),
            "defaults": {
                "EULER_3": (0.0, 0.0, 0.0),
                "DISLOCDEN": 0.0,
            },
            "values": {
                "EULER_3": ((0.0, 0.0, 0.0),),
                "DISLOCDEN": (10.0,),
            },
        },
        "flynns": [
            {
                "flynn_id": 10,
                "source_flynn_id": 10,
                "label": 0,
            }
        ],
        "stats": {},
    }


def _build_mechanics_phase_mesh_state() -> dict[str, object]:
    return {
        "nodes": [
            {"x": 0.0, "y": 0.0},
            {"x": 0.5, "y": 0.0},
            {"x": 0.5, "y": 1.0},
            {"x": 0.0, "y": 1.0},
            {"x": 1.0, "y": 0.0},
            {"x": 1.0, "y": 1.0},
        ],
        "flynns": [
            {
                "flynn_id": 10,
                "source_flynn_id": 10,
                "label": 0,
                "node_ids": [0, 1, 2, 3],
            },
            {
                "flynn_id": 20,
                "source_flynn_id": 20,
                "label": 1,
                "node_ids": [1, 4, 5, 2],
            },
        ],
        "_runtime_seed_unodes": {
            "ids": (1, 2, 3, 4),
            "positions": (
                (0.25, 0.25),
                (0.25, 0.75),
                (0.75, 0.25),
                (0.75, 0.75),
            ),
            "grid_indices": (
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ),
            "grid_shape": (2, 2),
        },
        "_runtime_seed_unode_fields": {
            "field_order": ("U_DISLOCDEN",),
            "values": {
                "U_DISLOCDEN": (10.0, 10.0, 10.0, 10.0),
            },
        },
        "_runtime_seed_unode_sections": {
            "field_order": ("U_EULER_3",),
            "component_counts": {"U_EULER_3": 3},
            "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
            "values": {
                "U_EULER_3": (
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                    (0.0, 0.0, 0.0),
                ),
            },
        },
        "_runtime_seed_flynn_sections": {
            "field_order": ("VISCOSITY",),
            "id_order": (10, 20),
            "defaults": {
                "VISCOSITY": 0.0,
            },
            "values": {
                "VISCOSITY": (1.0, 2.0),
            },
        },
        "stats": {},
    }


class SimulationTests(unittest.TestCase):
    def test_elle_phasefield_initialization_creates_center_seed(self) -> None:
        cfg = EllePhaseFieldConfig(nx=20, ny=16, initial_radius_sq=9.0)
        theta, temperature = initialize_elle_phasefield(cfg)

        theta_np = np.asarray(theta)
        temperature_np = np.asarray(temperature)
        self.assertEqual(theta_np.shape, (20, 16))
        self.assertEqual(temperature_np.shape, (20, 16))
        self.assertGreater(theta_np[10, 8], 0.5)
        self.assertEqual(float(temperature_np.mean()), 0.0)

    def test_run_elle_phasefield_simulation_shapes_and_artifacts(self) -> None:
        cfg = EllePhaseFieldConfig(nx=32, ny=32, dt=1.0e-4, spatial_step=0.03)
        theta, temperature, snapshots = run_elle_phasefield_simulation(cfg, steps=6, save_every=3)

        theta_np = np.asarray(theta)
        temperature_np = np.asarray(temperature)
        self.assertEqual(theta_np.shape, (32, 32))
        self.assertEqual(temperature_np.shape, (32, 32))
        self.assertEqual(len(snapshots), 2)
        self.assertGreaterEqual(float(theta_np.min()), 0.0)
        self.assertLessEqual(float(theta_np.max()), 1.0)
        self.assertGreater(float(np.abs(temperature_np).max()), 0.0)

        with tempfile.TemporaryDirectory() as tmpdir:
            written = save_elle_phasefield_artifacts(tmpdir, 6, theta_np, temperature_np)
            self.assertTrue(written["theta"].exists())
            self.assertTrue(written["temperature"].exists())
            self.assertTrue(written["stats"].exists())
            stats = json.loads(written["stats"].read_text(encoding="utf-8"))
            self.assertEqual(stats["step"], 6)
            self.assertGreaterEqual(stats["solid_fraction"], 0.0)
            self.assertLessEqual(stats["solid_fraction"], 1.0)

    def test_elle_phasefield_anisotropy_changes_solution(self) -> None:
        isotropic_cfg = EllePhaseFieldConfig(nx=28, ny=28, delta=0.0, dt=1.0e-4)
        anisotropic_cfg = EllePhaseFieldConfig(nx=28, ny=28, delta=0.04, aniso=6.0, dt=1.0e-4)

        theta_iso, temp_iso, _ = run_elle_phasefield_simulation(isotropic_cfg, steps=8, save_every=8)
        theta_aniso, temp_aniso, _ = run_elle_phasefield_simulation(anisotropic_cfg, steps=8, save_every=8)

        self.assertGreater(float(np.abs(np.asarray(theta_iso) - np.asarray(theta_aniso)).max()), 1e-6)
        stats = phasefield_statistics(theta_aniso, temp_aniso, step=8)
        self.assertEqual(stats["step"], 8)
        self.assertIn("interface_fraction", stats)

    def test_elle_phasefield_round_trips_elle_unode_sections(self) -> None:
        theta = np.zeros((4, 3), dtype=np.float32)
        temperature = np.zeros((4, 3), dtype=np.float32)
        theta[1, 1] = 1.0
        theta[2, 1] = 0.5
        temperature[1, 1] = -0.25
        temperature[2, 1] = 0.75

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "phasefield_roundtrip.elle"
            write_elle_phasefield_state(outpath, theta, temperature, step=3)
            text = outpath.read_text(encoding="utf-8")
            self.assertIn("UNODES\n", text)
            self.assertIn("U_CONC_A\n", text)
            self.assertIn("U_ATTRIB_A\n", text)

            loaded_theta, loaded_temperature, template = load_elle_phasefield_state(outpath)

        np.testing.assert_allclose(loaded_theta, theta)
        np.testing.assert_allclose(loaded_temperature, temperature)
        self.assertEqual(len(template.unodes), 12)

    def test_elle_phasefield_loads_original_example_file(self) -> None:
        source = PROJECT_ROOT.parent / "processes" / "phasefield" / "inphase.elle"
        theta, temperature, template = load_elle_phasefield_state(source)

        self.assertEqual(theta.shape, (300, 300))
        self.assertEqual(temperature.shape, (300, 300))
        self.assertEqual(len(template.unodes), 90000)
        self.assertGreater(float(theta.max()), 0.9)
        self.assertGreater(float(np.abs(temperature).max()), 0.1)

    def test_render_elle_file_renders_original_phasefield_example(self) -> None:
        source = PROJECT_ROOT.parent / "processes" / "phasefield" / "inphase.elle"
        showelle_in = PROJECT_ROOT.parent / "processes" / "phasefield" / "showelle.in"

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "inphase_preview.ppm"
            result = render_elle_file(source, outpath=outpath, showelle_in=showelle_in)

            self.assertTrue(outpath.exists())
            self.assertEqual(result["attribute"], "U_CONC_A")
            self.assertEqual(result["grid_shape"], [300, 300])
            self.assertTrue(result["overlay_boundaries"])

    def test_render_elle_file_supports_scaling_legend_and_labels(self) -> None:
        source = PROJECT_ROOT.parent / "processes" / "phasefield" / "inphase.elle"
        showelle_in = PROJECT_ROOT.parent / "processes" / "phasefield" / "showelle.in"

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "inphase_zoomed.ppm"
            result = render_elle_file(
                source,
                outpath=outpath,
                showelle_in=showelle_in,
                scale=2,
                legend=True,
                label_flynns=True,
            )

            self.assertTrue(outpath.exists())
            self.assertEqual(result["scale"], 2)
            self.assertTrue(result["legend"])
            self.assertTrue(result["flynn_labels"])
            self.assertEqual(result["image_shape"][0], 600)
            self.assertGreater(result["image_shape"][1], 600)
            width, height = _read_ppm_size(outpath)
            self.assertEqual((height, width), tuple(result["image_shape"]))

    def test_write_elle_html_viewer_builds_interactive_html(self) -> None:
        source = PROJECT_ROOT.parent / "processes" / "phasefield" / "inphase.elle"
        showelle_in = PROJECT_ROOT.parent / "processes" / "phasefield" / "showelle.in"

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = Path(tmpdir) / "inphase_viewer.html"
            result = write_elle_html_viewer(
                source,
                outpath=outpath,
                showelle_in=showelle_in,
                scale=2,
                legend=True,
                label_flynns=True,
            )
            text = outpath.read_text(encoding="utf-8")
            data_path = Path(result["data_outpath"])
            data_text = data_path.read_text(encoding="utf-8")

            self.assertTrue(outpath.exists())
            self.assertTrue(data_path.exists())
            self.assertEqual(result["attribute"], "U_CONC_A")
            self.assertEqual(result["scale"], 2)
            self.assertTrue(result["legend"])
            self.assertTrue(result["flynn_labels"])
            self.assertIn("viewerCanvas", text)
            self.assertIn("paletteSelect", text)
            self.assertIn("hoverStatus", text)
            self.assertIn(f'{data_path.name}?v=', text)
            self.assertIn('"attribute":"U_CONC_A"', data_text)
            self.assertLess(outpath.stat().st_size, data_path.stat().st_size)

    def test_render_elle_file_renders_grain_unodes_export(self) -> None:
        cfg = GrainGrowthConfig(nx=8, ny=6, num_grains=3, seed=4, init_mode="voronoi")
        phi = np.asarray(initialize_order_parameters(cfg))

        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = write_unode_elle(Path(tmpdir) / "snapshot.elle", phi, step=4)
            outpath = Path(tmpdir) / "snapshot_preview.ppm"
            result = render_elle_file(elle_path, outpath=outpath)

            self.assertTrue(outpath.exists())
            self.assertEqual(result["attribute"], "U_ATTRIB_A")
            self.assertEqual(result["palette"], "labels")
            self.assertEqual(result["grid_shape"], [8, 6])

    def test_render_elle_file_scales_label_preview(self) -> None:
        cfg = GrainGrowthConfig(nx=8, ny=6, num_grains=3, seed=4, init_mode="voronoi")
        phi = np.asarray(initialize_order_parameters(cfg))

        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = write_unode_elle(Path(tmpdir) / "snapshot.elle", phi, step=4)
            outpath = Path(tmpdir) / "snapshot_preview_zoom.ppm"
            result = render_elle_file(elle_path, outpath=outpath, scale=3, label_flynns=True)

            self.assertTrue(outpath.exists())
            self.assertEqual(result["scale"], 3)
            self.assertTrue(result["flynn_labels"])
            self.assertFalse(result["legend"])
            self.assertEqual(result["image_shape"], [18, 24])

    def test_write_elle_html_viewer_handles_grain_export(self) -> None:
        cfg = GrainGrowthConfig(nx=8, ny=6, num_grains=3, seed=4, init_mode="voronoi")
        phi = np.asarray(initialize_order_parameters(cfg))

        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = write_unode_elle(Path(tmpdir) / "snapshot.elle", phi, step=4)
            outpath = Path(tmpdir) / "snapshot_viewer.html"
            result = write_elle_html_viewer(elle_path, outpath=outpath, scale=3, label_flynns=True)
            text = outpath.read_text(encoding="utf-8")
            data_path = Path(result["data_outpath"])
            data_text = data_path.read_text(encoding="utf-8")

            self.assertTrue(outpath.exists())
            self.assertTrue(data_path.exists())
            self.assertEqual(result["attribute"], "U_ATTRIB_A")
            self.assertEqual(result["palette"], "labels")
            self.assertEqual(result["grid_shape"], [8, 6])
            self.assertIn(data_path.name, text)
            self.assertIn('"attribute":"U_ATTRIB_A"', data_text)
            self.assertIn('"palette":"labels"', data_text)

    def test_render_elle_file_handles_scattered_unodes_compactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_scattered_unode_example(Path(tmpdir) / "scattered.elle")
            outpath = Path(tmpdir) / "scattered_preview.ppm"
            result = render_elle_file(elle_path, outpath=outpath)

            self.assertTrue(outpath.exists())
            self.assertLessEqual(result["grid_shape"][0] * result["grid_shape"][1], 16)
            self.assertEqual(result["palette"], "labels")

    def test_write_elle_html_viewer_handles_scattered_unodes_compactly(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_scattered_unode_example(Path(tmpdir) / "scattered.elle")
            outpath = Path(tmpdir) / "scattered_viewer.html"
            result = write_elle_html_viewer(elle_path, outpath=outpath, label_flynns=True)
            data_path = Path(result["data_outpath"])

            self.assertTrue(outpath.exists())
            self.assertTrue(data_path.exists())
            self.assertLessEqual(result["grid_shape"][0] * result["grid_shape"][1], 16)
            self.assertLess(data_path.stat().st_size, 10_000)

    def test_build_viewer_payload_splits_periodic_flynn_boundaries(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_periodic_flynn_example(Path(tmpdir) / "periodic.elle")
            payload = build_viewer_payload(elle_path, label_flynns=True)

        self.assertEqual(payload["grid_shape"], [4, 4])
        self.assertEqual(payload["num_flynns"], 1)
        flynn = payload["flynns"][0]
        self.assertIn("paths", flynn)
        self.assertIn("label", flynn)
        self.assertGreaterEqual(len(flynn["paths"]), 2)
        for path in flynn["paths"]:
            self.assertEqual(len(path), 2)
            dx = abs(path[1][0] - path[0][0])
            dy = abs(path[1][1] - path[0][1])
            self.assertLessEqual(dx, 1)
            self.assertLessEqual(dy, 1)

    def test_summarize_elle_microstructure_reports_grain_and_orientation_stats(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_periodic_flynn_example(Path(tmpdir) / "periodic_0001.elle")
            summary = summarize_elle_microstructure(elle_path)

        self.assertEqual(summary["step"], 1)
        self.assertEqual(summary["grain_count"], 1)
        self.assertGreater(summary["mean_grain_area"], 0.0)
        self.assertIn("orientation", summary)
        self.assertAlmostEqual(summary["orientation"]["axial_mean_deg"], 30.0, places=4)

    def test_summarize_elle_microstructure_reports_fabric_tensor_indices(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_fabric_euler_example(Path(tmpdir) / "fabric_0001.elle")
            summary = summarize_elle_microstructure(elle_path)

        self.assertIn("fabric", summary)
        self.assertEqual(summary["fabric"]["source"], "U_EULER_3")
        np.testing.assert_allclose(
            np.asarray(summary["fabric"]["c_axis"]["eigenvalues"], dtype=np.float64),
            [1.0, 0.0, 0.0],
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertAlmostEqual(float(summary["fabric"]["c_axis"]["P_index"]), 1.0, places=12)
        self.assertAlmostEqual(float(summary["fabric"]["c_axis"]["G_index"]), 0.0, places=12)
        self.assertAlmostEqual(float(summary["fabric"]["c_axis"]["R_index"]), 0.0, places=12)
        self.assertAlmostEqual(
            float(summary["fabric"]["c_axis"]["pole_figure"]["fraction_within_15deg"]),
            1.0,
            places=12,
        )
        self.assertAlmostEqual(
            float(summary["fabric"]["c_axis"]["pole_figure"]["mean_colatitude_deg"]),
            0.0,
            places=12,
        )
        np.testing.assert_allclose(
            np.asarray(summary["fabric"]["c_axis"]["principal_direction"], dtype=np.float64),
            [0.0, 0.0, 1.0],
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_summarize_elle_microstructure_reports_aspect_ratio_distribution(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_aspect_ratio_example(Path(tmpdir) / "aspect_0001.elle")
            summary = summarize_elle_microstructure(elle_path)

        self.assertAlmostEqual(float(summary["mean_aspect_ratio"]), 2.0, places=12)
        self.assertAlmostEqual(float(summary["median_aspect_ratio"]), 2.0, places=12)
        self.assertAlmostEqual(sum(float(value) for value in summary["aspect_ratio_histogram"]), 1.0, places=12)
        self.assertAlmostEqual(float(summary["second_moment_grain_size"]), 0.0, places=12)

    def test_summarize_elle_microstructure_reports_mechanics_summary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_mechanics_scalar_example(Path(tmpdir) / "mechanics_0001.elle")
            summary = summarize_elle_microstructure(elle_path)

        self.assertIn("mechanics", summary)
        self.assertAlmostEqual(float(summary["mechanics"]["mean_normalized_strain_rate"]), 5.0, places=12)
        self.assertAlmostEqual(float(summary["mechanics"]["mean_normalized_stress"]), 6.0, places=12)
        self.assertAlmostEqual(float(summary["mechanics"]["mean_basal_activity"]), 2.5, places=12)
        self.assertAlmostEqual(float(summary["mechanics"]["mean_prismatic_activity"]), 5.0, places=12)
        self.assertAlmostEqual(float(summary["mechanics"]["mean_prismatic_fraction"]), 2.0 / 3.0, places=12)
        self.assertAlmostEqual(float(summary["mechanics"]["prismatic_to_basal_ratio"]), 2.0, places=12)

    def test_summarize_elle_microstructure_reads_adjacent_mechanics_payload_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_mechanics_scalar_example(Path(tmpdir) / "mechanics_0001.elle")
            _write_mechanics_mesh_sidecar(
                elle_path,
                cumulative_simple_shear=0.25,
                mean_normalized_stress=6.5,
                mean_differential_stress=9.5,
                mean_pyramidal_activity=1.5,
            )
            summary = summarize_elle_microstructure(elle_path)

        self.assertIn("mechanics_payload", summary)
        self.assertEqual(summary["mechanics"]["strain_axis_source"], "cumulative_simple_shear")
        self.assertAlmostEqual(float(summary["mechanics"]["direct_strain_axis"]), 0.25, places=12)
        self.assertAlmostEqual(float(summary["mechanics"]["mean_normalized_stress"]), 6.5, places=12)
        self.assertAlmostEqual(float(summary["mechanics"]["mean_differential_stress"]), 9.5, places=12)
        self.assertAlmostEqual(float(summary["mechanics"]["mean_pyramidal_activity"]), 1.5, places=12)

    def test_summarize_elle_microstructure_reads_vertical_shortening_direct_strain_axis_from_sidecar(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_mechanics_scalar_example(Path(tmpdir) / "mechanics_0001.elle")
            _write_mechanics_mesh_sidecar(
                elle_path,
                cumulative_simple_shear=0.0,
                direct_strain_axis=12.5,
                strain_axis_source="vertical_shortening_pct",
            )
            summary = summarize_elle_microstructure(elle_path)

        self.assertEqual(summary["mechanics"]["strain_axis_source"], "vertical_shortening_pct")
        self.assertAlmostEqual(float(summary["mechanics"]["direct_strain_axis"]), 12.5, places=12)

    def test_collect_elle_microstructure_snapshots_overlays_legacy_allout_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _write_mechanics_sequence_examples(directory)
            _write_legacy_allout_example(
                directory / "AllOutData.txt",
                rows=[
                    (
                        2.4e-2,
                        1.15e-9,
                        1.38e-2,
                        3.55e-6,
                        2.01e-6,
                        0.64,
                        0.20,
                        0.16,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                        12.0,
                    ),
                    (
                        2.5e-2,
                        1.25e-9,
                        1.48e-2,
                        3.65e-6,
                        2.11e-6,
                        0.60,
                        0.22,
                        0.18,
                        1.5,
                        2.5,
                        3.5,
                        4.5,
                        5.5,
                        6.5,
                        7.5,
                        8.5,
                        9.5,
                        10.5,
                        11.5,
                        12.5,
                    ),
                ],
            )
            snapshots = collect_elle_microstructure_snapshots(directory, pattern="mechanics_*.elle")

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(snapshots[0]["mechanics"]["statistics_source"], "AllOutData.txt")
        self.assertEqual(int(snapshots[0]["mechanics"]["statistics_row_index"]), 0)
        self.assertAlmostEqual(float(snapshots[0]["mechanics"]["mean_von_mises_stress"]), 2.4e-2, places=12)
        self.assertAlmostEqual(float(snapshots[0]["mechanics"]["mean_von_mises_strain_rate"]), 1.15e-9, places=18)
        self.assertAlmostEqual(float(snapshots[0]["mechanics"]["mean_differential_stress"]), 1.38e-2, places=12)
        self.assertAlmostEqual(float(snapshots[0]["mechanics"]["mean_basal_activity"]), 0.64, places=12)
        self.assertAlmostEqual(float(snapshots[0]["mechanics"]["mean_prismatic_activity"]), 0.20, places=12)
        self.assertAlmostEqual(float(snapshots[0]["mechanics"]["mean_pyramidal_activity"]), 0.16, places=12)
        self.assertAlmostEqual(float(snapshots[1]["mechanics"]["mean_prismatic_fraction"]), 0.22, places=12)
        self.assertAlmostEqual(float(snapshots[1]["mechanics"]["prismatic_to_basal_ratio"]), 0.22 / 0.60, places=12)
        np.testing.assert_allclose(
            np.asarray(snapshots[0]["mechanics"]["stress_tensor"], dtype=np.float64),
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(snapshots[1]["mechanics"]["strain_rate_tensor"], dtype=np.float64),
            np.asarray([7.5, 8.5, 9.5, 10.5, 11.5, 12.5], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_load_legacy_allout_statistics_parses_shipped_row_contract(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            allout_path = _write_legacy_allout_example(
                Path(tmpdir) / "AllOutData.txt",
                rows=[
                    (
                        2.4e-2,
                        1.15e-9,
                        1.38e-2,
                        3.55e-6,
                        2.01e-6,
                        0.64,
                        0.20,
                        0.16,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                        12.0,
                    ),
                ],
            )
            rows = load_legacy_allout_statistics(allout_path)

        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertAlmostEqual(float(row.mean_von_mises_stress), 2.4e-2, places=12)
        self.assertAlmostEqual(float(row.mean_von_mises_strain_rate), 1.15e-9, places=18)
        self.assertAlmostEqual(float(row.mean_differential_stress), 1.38e-2, places=12)
        self.assertAlmostEqual(float(row.mean_pyramidal_activity), 0.16, places=12)
        self.assertAlmostEqual(float(row.mean_prismatic_fraction), 0.20, places=12)
        self.assertAlmostEqual(float(row.prismatic_to_basal_ratio), 0.20 / 0.64, places=12)
        np.testing.assert_allclose(
            np.asarray(row.stress_tensor, dtype=np.float64),
            np.asarray([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_load_legacy_tmpstats_summary_parses_shipped_fixture(self) -> None:
        path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "tmpstats.dat"
        )
        summary = load_legacy_tmpstats_summary(path)

        self.assertEqual(summary.max_subgrain_count, 0)
        self.assertEqual(summary.total_grain_number, 26)
        self.assertAlmostEqual(float(summary.average_grain_size), 0.038462, places=6)
        self.assertAlmostEqual(float(summary.second_moment_grain_size), 0.760010, places=6)
        self.assertAlmostEqual(float(summary.total_boundary_length), 12.325700, places=6)
        self.assertAlmostEqual(float(summary.ratio), 1.139464, places=6)
        self.assertAlmostEqual(float(summary.max_ang), 78.0, places=6)
        self.assertAlmostEqual(float(summary.min_ang), 178.0, places=6)
        self.assertAlmostEqual(float(summary.accuracy), 10.0, places=6)
        self.assertAlmostEqual(float(summary.min_max_orientation_bins["<=5"]), 2.573460, places=6)
        self.assertAlmostEqual(float(summary.max_orientation_bins["<=180"]), 12.219727, places=6)

    def test_load_legacy_old_stats_summary_parses_shipped_fixture(self) -> None:
        path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "old.stats"
        )
        summary = load_legacy_old_stats_summary(path)

        self.assertEqual(summary.max_subgrain_count, 49)
        self.assertEqual(summary.total_grain_number, 26)
        self.assertEqual(len(summary.flynn_rows), 49)
        self.assertEqual(summary.mapped_flynn_count, 26)
        self.assertEqual(summary.orphan_flynn_count, 23)
        self.assertEqual(summary.split_flynn_count, 0)
        self.assertEqual(summary.active_cycle_flynn_count, 26)
        self.assertAlmostEqual(float(summary.average_grain_size), 0.038462, places=6)
        self.assertAlmostEqual(float(summary.second_moment_grain_size), 0.760010, places=6)
        self.assertAlmostEqual(float(summary.total_boundary_length), 12.325700, places=6)
        self.assertEqual(summary.flynn_rows[0].mineral, "Quartz")
        self.assertEqual(summary.flynn_rows[0].flynn_number, 0)
        self.assertEqual(summary.flynn_rows[0].grain_number, 0)
        self.assertEqual(summary.flynn_rows[-1].mineral, "Fsp")
        self.assertEqual(summary.flynn_rows[-1].flynn_number, 25)
        self.assertEqual(summary.flynn_rows[-1].grain_number, 19)

    def test_load_legacy_last_stats_summary_parses_shipped_fixture(self) -> None:
        path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "last.stats"
        )
        summary = load_legacy_last_stats_summary(path)

        self.assertEqual(summary.max_subgrain_count, 0)
        self.assertEqual(summary.total_grain_number, 26)
        self.assertEqual(summary.active_flynn_count, 26)
        self.assertEqual(summary.active_cycle_flynn_count, 26)
        self.assertAlmostEqual(float(summary.average_grain_size), 0.038462, places=6)
        self.assertAlmostEqual(float(summary.second_moment_grain_size), 0.760010, places=6)
        self.assertEqual(summary.flynn_rows[0].mineral, "Quartz")
        self.assertEqual(summary.flynn_rows[0].flynn_number, 0)
        self.assertEqual(summary.flynn_rows[-1].mineral, "Fsp")
        self.assertEqual(summary.flynn_rows[-1].flynn_number, 25)

    def test_load_legacy_statistics_summary_autodetects_summary_kind(self) -> None:
        tmpstats_path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "tmpstats.dat"
        )
        old_stats_path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "old.stats"
        )
        last_stats_path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "last.stats"
        )

        self.assertEqual(type(load_legacy_statistics_summary(tmpstats_path)).__name__, "LegacyTmpStatsSummary")
        self.assertEqual(type(load_legacy_statistics_summary(old_stats_path)).__name__, "LegacyOldStatsSummary")
        self.assertEqual(type(load_legacy_statistics_summary(last_stats_path)).__name__, "LegacyLastStatsSummary")

    def test_compare_snapshot_summary_to_legacy_statistics_reports_shared_metrics(self) -> None:
        snapshot_summary = {
            "grain_count": 20,
            "mean_grain_area": 0.05,
            "second_moment_grain_size": 0.70,
        }
        legacy_path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "tmpstats.dat"
        )

        comparison = compare_snapshot_summary_to_legacy_statistics(snapshot_summary, legacy_path).to_dict()

        self.assertEqual(comparison["legacy_statistics_kind"], "tmpstats.dat")
        self.assertEqual(int(comparison["current_grain_count"]), 20)
        self.assertEqual(int(comparison["legacy_total_grain_number"]), 26)
        self.assertAlmostEqual(float(comparison["current_mean_grain_area"]), 0.05, places=12)
        self.assertAlmostEqual(float(comparison["legacy_average_grain_size"]), 0.038462, places=6)
        self.assertEqual(int(comparison["grain_count_delta"]), -6)
        self.assertFalse(bool(comparison["grain_count_match"]))
        self.assertFalse(bool(comparison["mean_grain_area_match"]))

    def test_summarize_current_mesh_bookkeeping_counts_saved_fields(self) -> None:
        mesh_state = {
            "flynns": [
                {
                    "flynn_id": 10,
                    "source_flynn_id": 10,
                    "retained_identity": True,
                    "parents": [],
                },
                {
                    "flynn_id": 11,
                    "source_flynn_id": 11,
                    "retained_identity": True,
                    "parents": [],
                },
                {
                    "flynn_id": 12,
                    "source_flynn_id": 11,
                    "retained_identity": False,
                    "parents": [11, 12],
                },
                {
                    "flynn_id": 13,
                    "source_flynn_id": -1,
                    "retained_identity": False,
                    "parents": [11],
                },
            ],
            "stats": {
                "mesh_split_flynns": 1,
                "mesh_merged_flynns": 2,
                "num_flynns": 4,
            },
        }

        summary = summarize_current_mesh_bookkeeping(mesh_state)

        self.assertEqual(summary.total_flynn_count, 4)
        self.assertEqual(summary.retained_flynn_count, 2)
        self.assertEqual(summary.nonretained_flynn_count, 2)
        self.assertEqual(summary.source_mapped_flynn_count, 3)
        self.assertEqual(summary.source_orphan_flynn_count, 1)
        self.assertEqual(summary.unique_source_flynn_count, 2)
        self.assertEqual(summary.multi_parent_flynn_count, 1)
        self.assertEqual(summary.mesh_split_flynn_count, 1)
        self.assertEqual(summary.mesh_merged_flynn_count, 2)
        self.assertEqual(summary.mesh_stats_num_flynns, 4)

    def test_compare_mesh_bookkeeping_to_legacy_old_stats_reports_count_deltas(self) -> None:
        mesh_state = {
            "flynns": [
                {
                    "flynn_id": 10,
                    "source_flynn_id": 10,
                    "retained_identity": True,
                    "parents": [],
                },
                {
                    "flynn_id": 11,
                    "source_flynn_id": 11,
                    "retained_identity": True,
                    "parents": [],
                },
                {
                    "flynn_id": 12,
                    "source_flynn_id": 11,
                    "retained_identity": False,
                    "parents": [11, 12],
                },
                {
                    "flynn_id": 13,
                    "source_flynn_id": -1,
                    "retained_identity": False,
                    "parents": [11],
                },
            ],
            "stats": {
                "mesh_split_flynns": 1,
                "mesh_merged_flynns": 0,
                "num_flynns": 4,
            },
        }
        legacy_path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "old.stats"
        )

        comparison = compare_mesh_bookkeeping_to_legacy_old_stats(mesh_state, legacy_path)
        payload = comparison.to_dict()

        self.assertEqual(payload["legacy_total_flynn_count"], 49)
        self.assertEqual(payload["current_total_flynn_count"], 4)
        self.assertEqual(payload["legacy_mapped_flynn_count"], 26)
        self.assertEqual(payload["current_source_mapped_flynn_count"], 3)
        self.assertEqual(payload["legacy_orphan_flynn_count"], 23)
        self.assertEqual(payload["current_source_orphan_flynn_count"], 1)
        self.assertEqual(payload["legacy_total_grain_number"], 26)
        self.assertEqual(payload["current_unique_source_flynn_count"], 2)
        self.assertEqual(payload["current_multi_parent_flynn_count"], 1)
        self.assertEqual(payload["current_nonretained_flynn_count"], 2)
        self.assertFalse(payload["total_flynn_count_match"])
        self.assertFalse(payload["mapped_flynn_count_match"])
        self.assertFalse(payload["orphan_flynn_count_match"])
        self.assertFalse(payload["total_grain_number_match"])
        self.assertFalse(payload["split_count_match_via_multi_parent"])
        self.assertFalse(payload["split_count_match_via_mesh_stats"])
        self.assertEqual(payload["count_deltas"]["total_flynn_count"], -45)
        self.assertEqual(payload["count_deltas"]["mapped_flynn_count"], -23)
        self.assertEqual(payload["count_deltas"]["orphan_flynn_count"], -22)
        self.assertEqual(payload["count_deltas"]["total_grain_number"], -24)

    def test_collect_elle_microstructure_snapshots_aligns_legacy_allout_after_step0(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            directory = Path(tmpdir)
            _write_aspect_ratio_example(directory / "periodic_0000.elle")
            _write_mechanics_scalar_example(directory / "mechanics_0001.elle")
            _write_mechanics_scalar_example(directory / "mechanics_0002.elle")
            _write_legacy_allout_example(
                directory / "AllOutData.txt",
                rows=[
                    (
                        2.4e-2,
                        1.15e-9,
                        1.38e-2,
                        3.55e-6,
                        2.01e-6,
                        0.64,
                        0.20,
                        0.16,
                        1.0,
                        2.0,
                        3.0,
                        4.0,
                        5.0,
                        6.0,
                        7.0,
                        8.0,
                        9.0,
                        10.0,
                        11.0,
                        12.0,
                    ),
                    (
                        2.5e-2,
                        1.25e-9,
                        1.48e-2,
                        3.65e-6,
                        2.11e-6,
                        0.60,
                        0.22,
                        0.18,
                        1.5,
                        2.5,
                        3.5,
                        4.5,
                        5.5,
                        6.5,
                        7.5,
                        8.5,
                        9.5,
                        10.5,
                        11.5,
                        12.5,
                    ),
                ],
            )
            snapshots = collect_elle_microstructure_snapshots(directory, pattern="*.elle")

        self.assertEqual([snapshot["step"] for snapshot in snapshots], [0, 1, 2])
        self.assertNotIn("mechanics", snapshots[0])
        self.assertAlmostEqual(float(snapshots[1]["mechanics"]["mean_differential_stress"]), 1.38e-2, places=12)
        self.assertAlmostEqual(float(snapshots[2]["mechanics"]["mean_differential_stress"]), 1.48e-2, places=12)

    def test_compare_elle_microstructure_sequences_matches_steps(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_0001.elle")
            _write_periodic_flynn_example(Path(candidate_dir) / "periodic_0001.elle")
            report = compare_elle_microstructure_sequences(reference_dir, candidate_dir)

        self.assertEqual(report["summary"]["num_matched_steps"], 1)
        self.assertEqual(report["summary"]["grain_count_abs_diff_mean"], 0.0)
        self.assertEqual(report["summary"]["mean_grain_area_abs_diff_mean"], 0.0)
        self.assertEqual(report["summary"]["mean_aspect_ratio_abs_diff_mean"], 0.0)

    def test_compare_elle_microstructure_sequences_tracks_c_axis_pole_figure(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_fabric_euler_example(Path(reference_dir) / "fabric_0001.elle")
            _write_fabric_euler_example(Path(candidate_dir) / "fabric_0001.elle")
            report = compare_elle_microstructure_sequences(reference_dir, candidate_dir)

        self.assertEqual(report["summary"]["fabric_c_axis_pole_figure_l1_mean"], 0.0)
        self.assertEqual(report["summary"]["fabric_c_axis_vertical_fraction_abs_diff_mean"], 0.0)

    def test_summarize_lui_suckale_datasets_reads_repo_examples(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"
        report = summarize_liu_suckale_datasets(data_dir)

        self.assertEqual(report["num_datasets"], 8)
        grain_entries = [entry for entry in report["datasets"] if entry.get("variable") == "grain_kde"]
        euler_entries = [entry for entry in report["datasets"] if str(entry.get("variable", "")).startswith("euler_")]
        self.assertEqual(len(grain_entries), 2)
        self.assertEqual(len(euler_entries), 6)

    def test_summarize_llorens_structure_from_text_detects_dual_layer_model(self) -> None:
        text = (
            "The data structure of the models consists of two basic layers: "
            "(1) flynns defined by boundary nodes (bnodes) and "
            "(2) unodes that provide a higher resolution of physical properties within grains. "
            "The FFT approach provides the stress and velocity fields. "
            "These data are used to simulate intracrystalline recovery and GBM driven by the "
            "reduction of surface energy and stored strain energy."
        )

        summary = summarize_llorens_structure_from_text(text, title="Test Llorens")

        self.assertTrue(summary["dual_layer_model"]["present"])
        self.assertTrue(summary["fft_elle_coupling"]["present"])
        self.assertTrue(summary["recrystallization_drivers"]["present"])

    def test_summarize_liu_suckale_paper_from_text_extracts_targets(self) -> None:
        text = (
            "We validate the microscale model against two benchmark laboratory experiments. "
            "The first set of laboratory experiments are static recrystallizations of ice at high temperature "
            "by Fan et al. Grain size data were collected at seven time points. "
            "To validate the ability of our microscale model to capture fabric evolution, we compare our "
            "simulations with experiments of synthetic ice in simple shear by Qi et al. The angle is defined "
            "as the sweep angle. "
            "Our dataset contains 11,932 training samples, 1492 validation samples, and 1492 testing samples. "
            "Fig. 5 shows the comparison between the FNO predictions and the simulation results for all three Euler angles. "
            "We then apply the coupled framework to a one-dimensional vertical ice column. "
            "We compare modeled grain size at five ice-core sites: GISP2, GRIP, EGRIP, EDML, and Siple Dome."
        )

        summary = summarize_liu_suckale_paper_from_text(text, title="Test Liu")

        self.assertEqual(summary["microscale_benchmarks"][0]["status"], "detected")
        self.assertEqual(summary["microscale_benchmarks"][1]["status"], "detected")
        self.assertEqual(summary["surrogate_benchmarks"][0]["sample_counts"]["training"], 11932)
        self.assertEqual(summary["surrogate_benchmarks"][0]["sample_counts"]["validation"], 1492)
        self.assertEqual(summary["surrogate_benchmarks"][0]["sample_counts"]["testing"], 1492)
        self.assertEqual(summary["macro_benchmarks"][1]["sites"], ["GISP2", "GRIP", "EGRIP", "EDML", "Siple Dome"])

    def test_assess_current_rewrite_against_papers_reports_known_gaps(self) -> None:
        report = assess_current_rewrite_against_papers()

        llorens_targets = {entry["target"]: entry["status"] for entry in report["llorens_alignment"]}
        liu_targets = {entry["target"]: entry["status"] for entry in report["liu_suckale_alignment"]}

        self.assertEqual(llorens_targets["ELLE dual-layer flynns/bnodes/unodes structure"], "implemented")
        self.assertEqual(llorens_targets["FFT deformation fields coupled directly into ELLE recrystallization"], "missing")
        self.assertEqual(liu_targets["Vertical-column multiscale coupling"], "missing")

    def test_summarize_sequence_trends_detects_coarsening(self) -> None:
        sequence = [
            {"step": 1, "grain_count": 10, "mean_grain_area": 0.10, "second_moment_grain_size": 0.40, "mean_equivalent_radius": 0.20, "mean_shape_factor": 0.80},
            {"step": 2, "grain_count": 9, "mean_grain_area": 0.11, "second_moment_grain_size": 0.35, "mean_equivalent_radius": 0.21, "mean_shape_factor": 0.79},
            {"step": 3, "grain_count": 8, "mean_grain_area": 0.13, "second_moment_grain_size": 0.30, "mean_equivalent_radius": 0.23, "mean_shape_factor": 0.78},
        ]

        summary = summarize_sequence_trends(sequence)

        self.assertTrue(summary["flags"]["coarsening_present"])
        self.assertGreater(summary["metrics"]["mean_grain_area"]["delta"], 0.0)
        self.assertLess(summary["metrics"]["grain_count"]["delta"], 0.0)
        self.assertLess(summary["metrics"]["second_moment_grain_size"]["delta"], 0.0)

    def test_summarize_sequence_trends_tracks_fabric_indices(self) -> None:
        sequence = [
            {
                "step": 1,
                "grain_count": 10,
                "mean_grain_area": 0.10,
                "mean_equivalent_radius": 0.20,
                "mean_shape_factor": 0.80,
                "mean_aspect_ratio": 1.40,
                "fabric": {
                    "c_axis": {
                        "eigenvalues": [0.60, 0.30, 0.10],
                        "P_index": 0.30,
                        "G_index": 0.40,
                        "R_index": 0.30,
                        "pole_figure": {"fraction_within_15deg": 0.20, "mean_colatitude_deg": 35.0},
                    },
                    "a_axis": {"eigenvalues": [0.50, 0.30, 0.20], "P_index": 0.20, "G_index": 0.20, "R_index": 0.60},
                    "prism_normal": {"eigenvalues": [0.45, 0.35, 0.20], "P_index": 0.10, "G_index": 0.30, "R_index": 0.60},
                },
            },
            {
                "step": 2,
                "grain_count": 9,
                "mean_grain_area": 0.11,
                "mean_equivalent_radius": 0.21,
                "mean_shape_factor": 0.79,
                "mean_aspect_ratio": 1.20,
                "fabric": {
                    "c_axis": {
                        "eigenvalues": [0.70, 0.20, 0.10],
                        "P_index": 0.50,
                        "G_index": 0.20,
                        "R_index": 0.30,
                        "pole_figure": {"fraction_within_15deg": 0.55, "mean_colatitude_deg": 12.0},
                    },
                    "a_axis": {"eigenvalues": [0.55, 0.25, 0.20], "P_index": 0.30, "G_index": 0.10, "R_index": 0.60},
                    "prism_normal": {"eigenvalues": [0.40, 0.35, 0.25], "P_index": 0.05, "G_index": 0.20, "R_index": 0.75},
                },
            },
        ]

        summary = summarize_sequence_trends(sequence)

        self.assertIn("fabric_c_axis_largest_eigenvalue", summary["metrics"])
        self.assertIn("fabric_c_axis_p_index", summary["metrics"])
        self.assertIn("fabric_c_axis_vertical_fraction_15deg", summary["metrics"])
        self.assertIn("fabric_a_axis_p_index", summary["metrics"])
        self.assertGreater(summary["metrics"]["fabric_c_axis_p_index"]["delta"], 0.0)
        self.assertGreater(summary["metrics"]["fabric_c_axis_vertical_fraction_15deg"]["delta"], 0.0)

    def test_summarize_sequence_trends_tracks_mechanics_metrics(self) -> None:
        sequence = [
            {
                "step": 1,
                "grain_count": 10,
                "mean_grain_area": 0.10,
                "mean_equivalent_radius": 0.20,
                "mean_shape_factor": 0.80,
                "mean_aspect_ratio": 1.30,
                "mechanics": {
                    "mean_normalized_strain_rate": 0.5,
                    "mean_normalized_stress": 1.5,
                    "mean_differential_stress": 4.0,
                    "mean_basal_activity": 2.0,
                    "mean_prismatic_activity": 1.0,
                    "mean_pyramidal_activity": 0.2,
                    "mean_prismatic_fraction": 0.3333333333333333,
                    "prismatic_to_basal_ratio": 0.5,
                },
            },
            {
                "step": 2,
                "grain_count": 9,
                "mean_grain_area": 0.11,
                "mean_equivalent_radius": 0.21,
                "mean_shape_factor": 0.79,
                "mean_aspect_ratio": 1.20,
                "mechanics": {
                    "mean_normalized_strain_rate": 0.6,
                    "mean_normalized_stress": 1.2,
                    "mean_differential_stress": 3.0,
                    "mean_basal_activity": 1.8,
                    "mean_prismatic_activity": 1.4,
                    "mean_pyramidal_activity": 0.4,
                    "mean_prismatic_fraction": 0.4375,
                    "prismatic_to_basal_ratio": 0.7777777777777778,
                },
            },
        ]

        summary = summarize_sequence_trends(sequence)

        self.assertIn("mechanics_mean_normalized_stress", summary["metrics"])
        self.assertIn("mechanics_mean_differential_stress", summary["metrics"])
        self.assertIn("mechanics_mean_prismatic_activity", summary["metrics"])
        self.assertIn("mechanics_mean_pyramidal_activity", summary["metrics"])
        self.assertIn("mechanics_prismatic_to_basal_ratio", summary["metrics"])
        self.assertIn("mechanics_cumulative_normalized_strain", summary["metrics"])
        self.assertIn("mechanics_stress_strain", summary["curves"])
        self.assertIn("mechanics_activity_strain", summary["curves"])
        self.assertLess(summary["metrics"]["mechanics_mean_differential_stress"]["delta"], 0.0)
        self.assertGreater(summary["metrics"]["mechanics_mean_pyramidal_activity"]["delta"], 0.0)
        self.assertGreater(summary["metrics"]["mechanics_prismatic_to_basal_ratio"]["delta"], 0.0)
        self.assertEqual(
            summary["curves"]["mechanics_stress_strain"]["stress_source"],
            "mean_differential_stress",
        )
        np.testing.assert_allclose(
            np.asarray(summary["curves"]["mechanics_stress_strain"]["strain"], dtype=np.float64),
            np.asarray([0.0, 0.55], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(summary["curves"]["mechanics_stress_strain"]["stress"], dtype=np.float64),
            np.asarray([4.0, 3.0], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(summary["curves"]["mechanics_activity_strain"]["prismatic_activity"], dtype=np.float64),
            np.asarray([1.0, 1.4], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(summary["curves"]["mechanics_activity_strain"]["pyramidal_activity"], dtype=np.float64),
            np.asarray([0.2, 0.4], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_summarize_sequence_trends_uses_von_mises_statistics_when_direct_axis_missing(self) -> None:
        sequence = [
            {
                "step": 1,
                "grain_count": 10,
                "mean_grain_area": 0.10,
                "mean_equivalent_radius": 0.20,
                "mean_shape_factor": 0.80,
                "mean_aspect_ratio": 1.30,
                "mechanics": {
                    "mean_von_mises_strain_rate": 1.0,
                    "mean_von_mises_stress": 3.0,
                    "mean_differential_stress": 4.0,
                    "mean_basal_activity": 0.60,
                    "mean_prismatic_activity": 0.20,
                    "mean_pyramidal_activity": 0.20,
                    "mean_total_activity": 1.0,
                    "mean_prismatic_fraction": 0.20,
                    "prismatic_to_basal_ratio": 1.0 / 3.0,
                },
            },
            {
                "step": 2,
                "grain_count": 9,
                "mean_grain_area": 0.11,
                "mean_equivalent_radius": 0.21,
                "mean_shape_factor": 0.79,
                "mean_aspect_ratio": 1.20,
                "mechanics": {
                    "mean_von_mises_strain_rate": 2.0,
                    "mean_von_mises_stress": 2.5,
                    "mean_differential_stress": 3.5,
                    "mean_basal_activity": 0.55,
                    "mean_prismatic_activity": 0.25,
                    "mean_pyramidal_activity": 0.20,
                    "mean_total_activity": 1.0,
                    "mean_prismatic_fraction": 0.25,
                    "prismatic_to_basal_ratio": 0.25 / 0.55,
                },
            },
        ]

        summary = summarize_sequence_trends(sequence)

        self.assertIn("mechanics_mean_von_mises_stress", summary["metrics"])
        self.assertIn("mechanics_mean_von_mises_strain_rate", summary["metrics"])
        self.assertEqual(
            summary["curves"]["mechanics_stress_strain"]["strain_source"],
            "integrated_von_mises_strain_rate",
        )
        self.assertEqual(
            summary["curves"]["mechanics_stress_strain"]["stress_source"],
            "mean_differential_stress",
        )
        np.testing.assert_allclose(
            np.asarray(summary["curves"]["mechanics_stress_strain"]["strain"], dtype=np.float64),
            np.asarray([0.0, 1.5], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(summary["curves"]["mechanics_activity_strain"]["pyramidal_activity"], dtype=np.float64),
            np.asarray([0.2, 0.2], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_evaluate_static_grain_growth_benchmark_reports_reference_coarsening(self) -> None:
        reference_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "elle" / "example" / "results"

        report = evaluate_static_grain_growth_benchmark(reference_dir, pattern="fine_foam_step*.elle")

        self.assertEqual(report["reference_trends"]["num_snapshots"], 10)
        self.assertTrue(report["reference_trends"]["flags"]["coarsening_present"])
        self.assertGreater(report["reference_trends"]["metrics"]["mean_grain_area"]["delta"], 0.0)

    def test_evaluate_static_grain_growth_benchmark_compares_fabric_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_fabric_euler_example(Path(reference_dir) / "fabric_0001.elle")
            _write_fabric_euler_example(Path(candidate_dir) / "fabric_0001.elle")
            report = evaluate_static_grain_growth_benchmark(reference_dir, candidate_dir=candidate_dir)

        self.assertIn("fabric_c_axis_largest_eigenvalue", report["comparison"])
        self.assertIn("fabric_c_axis_p_index", report["comparison"])
        self.assertIn("fabric_c_axis_vertical_fraction_15deg", report["comparison"])
        self.assertIn("second_moment_grain_size", report["comparison"])
        self.assertEqual(report["comparison"]["fabric_c_axis_largest_eigenvalue"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["fabric_c_axis_p_index"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["fabric_c_axis_vertical_fraction_15deg"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["second_moment_grain_size"]["rmse"], 0.0)

    def test_evaluate_static_grain_growth_benchmark_compares_mechanics_metrics(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_mechanics_scalar_example(Path(reference_dir) / "mechanics_0001.elle")
            _write_mechanics_scalar_example(Path(candidate_dir) / "mechanics_0001.elle")
            report = evaluate_static_grain_growth_benchmark(reference_dir, candidate_dir=candidate_dir)

        self.assertIn("mechanics_mean_normalized_stress", report["comparison"])
        self.assertIn("mechanics_mean_prismatic_activity", report["comparison"])
        self.assertIn("mechanics_prismatic_to_basal_ratio", report["comparison"])
        self.assertEqual(report["comparison"]["mechanics_mean_normalized_stress"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_mean_prismatic_activity"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_prismatic_to_basal_ratio"]["rmse"], 0.0)

    def test_evaluate_static_grain_growth_benchmark_emits_mechanics_curve_comparisons(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_mechanics_sequence_examples(Path(reference_dir))
            _write_mechanics_sequence_examples(Path(candidate_dir))
            report = evaluate_static_grain_growth_benchmark(
                reference_dir,
                candidate_dir=candidate_dir,
                pattern="mechanics_*.elle",
            )

        self.assertIn("mechanics_stress_strain_curve", report["comparison"])
        self.assertIn("mechanics_activity_strain_curve", report["comparison"])
        self.assertEqual(report["comparison"]["mechanics_stress_strain_curve"]["stress"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_stress_strain_curve"]["strain"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_activity_strain_curve"]["prismatic_activity"]["rmse"], 0.0)
        check_names = {entry["name"] for entry in report["paper_signature_assessment"]["checks"]}
        self.assertIn("stress_strain_curve_alignment", check_names)
        self.assertIn("activity_curve_alignment", check_names)

    def test_evaluate_static_grain_growth_benchmark_prefers_direct_mechanics_strain_axis_from_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            ref_step1, ref_step2 = _write_mechanics_sequence_examples(Path(reference_dir))
            cand_step1, cand_step2 = _write_mechanics_sequence_examples(Path(candidate_dir))
            _write_mechanics_mesh_sidecar(ref_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(ref_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)
            _write_mechanics_mesh_sidecar(cand_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(cand_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)

            report = evaluate_static_grain_growth_benchmark(
                reference_dir,
                candidate_dir=candidate_dir,
                pattern="mechanics_*.elle",
            )

        np.testing.assert_allclose(
            np.asarray(report["reference_trends"]["curves"]["mechanics_stress_strain"]["strain"], dtype=np.float64),
            np.asarray([0.20, 0.55], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertEqual(
            report["reference_trends"]["curves"]["mechanics_stress_strain"]["strain_source"],
            "cumulative_simple_shear",
        )
        self.assertEqual(report["comparison"]["mechanics_stress_strain_curve"]["strain"]["rmse"], 0.0)

    def test_evaluate_static_grain_growth_benchmark_prefers_vertical_shortening_from_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            ref_step1, ref_step2 = _write_mechanics_sequence_examples(Path(reference_dir))
            cand_step1, cand_step2 = _write_mechanics_sequence_examples(Path(candidate_dir))
            _write_mechanics_mesh_sidecar(
                ref_step1,
                cumulative_simple_shear=0.0,
                direct_strain_axis=5.0,
                strain_axis_source="vertical_shortening_pct",
            )
            _write_mechanics_mesh_sidecar(
                ref_step2,
                cumulative_simple_shear=0.0,
                direct_strain_axis=12.0,
                strain_axis_source="vertical_shortening_pct",
            )
            _write_mechanics_mesh_sidecar(
                cand_step1,
                cumulative_simple_shear=0.0,
                direct_strain_axis=5.0,
                strain_axis_source="vertical_shortening_pct",
            )
            _write_mechanics_mesh_sidecar(
                cand_step2,
                cumulative_simple_shear=0.0,
                direct_strain_axis=12.0,
                strain_axis_source="vertical_shortening_pct",
            )

            report = evaluate_static_grain_growth_benchmark(
                reference_dir,
                candidate_dir=candidate_dir,
                pattern="mechanics_*.elle",
            )

        np.testing.assert_allclose(
            np.asarray(report["reference_trends"]["curves"]["mechanics_stress_strain"]["strain"], dtype=np.float64),
            np.asarray([5.0, 12.0], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertEqual(
            report["reference_trends"]["curves"]["mechanics_stress_strain"]["strain_source"],
            "vertical_shortening_pct",
        )
        self.assertEqual(report["comparison"]["mechanics_stress_strain_curve"]["strain"]["rmse"], 0.0)

    def test_evaluate_static_grain_growth_benchmark_reads_legacy_allout_statistics(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_aspect_ratio_example(Path(reference_dir) / "periodic_0001.elle")
            _write_aspect_ratio_example(Path(reference_dir) / "periodic_0002.elle")
            _write_aspect_ratio_example(Path(candidate_dir) / "periodic_0001.elle")
            _write_aspect_ratio_example(Path(candidate_dir) / "periodic_0002.elle")
            rows = [
                (
                    2.4e-2,
                    1.0,
                    1.38e-2,
                    3.55e-6,
                    2.01e-6,
                    0.64,
                    0.20,
                    0.16,
                    1.0,
                    2.0,
                    3.0,
                    4.0,
                    5.0,
                    6.0,
                    7.0,
                    8.0,
                    9.0,
                    10.0,
                    11.0,
                    12.0,
                ),
                (
                    2.5e-2,
                    2.0,
                    1.48e-2,
                    3.65e-6,
                    2.11e-6,
                    0.60,
                    0.22,
                    0.18,
                    1.5,
                    2.5,
                    3.5,
                    4.5,
                    5.5,
                    6.5,
                    7.5,
                    8.5,
                    9.5,
                    10.5,
                    11.5,
                    12.5,
                ),
            ]
            _write_legacy_allout_example(Path(reference_dir) / "AllOutData.txt", rows=rows)
            _write_legacy_allout_example(Path(candidate_dir) / "AllOutData.txt", rows=rows)

            report = evaluate_static_grain_growth_benchmark(
                reference_dir,
                candidate_dir=candidate_dir,
                pattern="periodic_*.elle",
            )

        self.assertAlmostEqual(
            float(report["reference_sequence"][0]["mechanics"]["mean_pyramidal_activity"]),
            0.16,
            places=12,
        )
        self.assertEqual(report["comparison"]["mechanics_mean_von_mises_stress"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_mean_von_mises_strain_rate"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_mean_differential_stress"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_stress_field_error"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_strain_rate_field_error"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_mean_pyramidal_activity"]["rmse"], 0.0)
        self.assertEqual(
            report["reference_trends"]["curves"]["mechanics_stress_strain"]["strain_source"],
            "integrated_von_mises_strain_rate",
        )
        self.assertEqual(report["comparison"]["mechanics_stress_strain_curve"]["stress"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_activity_strain_curve"]["pyramidal_activity"]["rmse"], 0.0)

    def test_evaluate_static_grain_growth_benchmark_prefers_differential_stress_from_sidecars(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            ref_step1, ref_step2 = _write_mechanics_sequence_examples(Path(reference_dir))
            cand_step1, cand_step2 = _write_mechanics_sequence_examples(Path(candidate_dir))
            _write_mechanics_mesh_sidecar(
                ref_step1,
                cumulative_simple_shear=0.10,
                mean_normalized_stress=6.0,
                mean_differential_stress=12.0,
                mean_pyramidal_activity=1.0,
            )
            _write_mechanics_mesh_sidecar(
                ref_step2,
                cumulative_simple_shear=0.30,
                simple_shear_increment=0.20,
                mean_normalized_stress=5.0,
                mean_differential_stress=10.0,
                mean_pyramidal_activity=2.0,
            )
            _write_mechanics_mesh_sidecar(
                cand_step1,
                cumulative_simple_shear=0.10,
                mean_normalized_stress=6.0,
                mean_differential_stress=12.0,
                mean_pyramidal_activity=1.0,
            )
            _write_mechanics_mesh_sidecar(
                cand_step2,
                cumulative_simple_shear=0.30,
                simple_shear_increment=0.20,
                mean_normalized_stress=5.0,
                mean_differential_stress=10.0,
                mean_pyramidal_activity=2.0,
            )

            report = evaluate_static_grain_growth_benchmark(
                reference_dir,
                candidate_dir=candidate_dir,
                pattern="mechanics_*.elle",
            )

        self.assertIn("mechanics_mean_differential_stress", report["comparison"])
        self.assertIn("mechanics_mean_pyramidal_activity", report["comparison"])
        self.assertEqual(report["comparison"]["mechanics_mean_differential_stress"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_mean_pyramidal_activity"]["rmse"], 0.0)
        self.assertEqual(
            report["reference_trends"]["curves"]["mechanics_stress_strain"]["stress_source"],
            "mean_differential_stress",
        )
        np.testing.assert_allclose(
            np.asarray(report["reference_trends"]["curves"]["mechanics_stress_strain"]["stress"], dtype=np.float64),
            np.asarray([12.0, 10.0], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(report["reference_trends"]["curves"]["mechanics_activity_strain"]["pyramidal_activity"], dtype=np.float64),
            np.asarray([1.0, 2.0], dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertEqual(report["comparison"]["mechanics_stress_strain_curve"]["stress"]["rmse"], 0.0)
        self.assertEqual(report["comparison"]["mechanics_activity_strain_curve"]["pyramidal_activity"]["rmse"], 0.0)

    def test_assess_paper_signature_alignment_reports_directional_passes(self) -> None:
        reference_trends = {
            "metrics": {
                "mean_grain_area": {"delta": 0.20},
                "grain_count": {"delta": -5.0},
                "mean_aspect_ratio": {"delta": -0.10},
                "fabric_c_axis_p_index": {"delta": 0.08},
                "fabric_c_axis_vertical_fraction_15deg": {"delta": 0.05},
                "fabric_c_axis_mean_colatitude_deg": {"delta": -3.0},
                "mechanics_mean_normalized_stress": {"delta": -0.20},
                "mechanics_mean_prismatic_activity": {"delta": 0.15},
                "mechanics_prismatic_to_basal_ratio": {"delta": 0.10},
            }
        }
        candidate_trends = {
            "metrics": {
                "mean_grain_area": {"delta": 0.18},
                "grain_count": {"delta": -4.0},
                "mean_aspect_ratio": {"delta": -0.08},
                "fabric_c_axis_p_index": {"delta": 0.07},
                "fabric_c_axis_vertical_fraction_15deg": {"delta": 0.03},
                "fabric_c_axis_mean_colatitude_deg": {"delta": -2.0},
                "mechanics_mean_normalized_stress": {"delta": -0.15},
                "mechanics_mean_prismatic_activity": {"delta": 0.12},
                "mechanics_prismatic_to_basal_ratio": {"delta": 0.07},
            }
        }
        comparison = {
            "mean_grain_area": {"normalized_rmse": 0.05},
            "grain_count": {"normalized_rmse": 0.08},
            "mean_aspect_ratio": {"normalized_rmse": 0.10},
            "fabric_c_axis_p_index": {"normalized_rmse": 0.04},
            "fabric_c_axis_vertical_fraction_15deg": {"normalized_rmse": 0.06},
            "fabric_c_axis_mean_colatitude_deg": {"normalized_rmse": 0.02},
            "mechanics_mean_normalized_stress": {"normalized_rmse": 0.09},
            "mechanics_mean_prismatic_activity": {"normalized_rmse": 0.11},
            "mechanics_prismatic_to_basal_ratio": {"normalized_rmse": 0.12},
        }
        reference_survival = {"summary": {"initial_size_correlation_stronger_than_schmid": True}}
        candidate_survival = {"summary": {"initial_size_correlation_stronger_than_schmid": True}}

        assessment = assess_paper_signature_alignment(
            reference_trends=reference_trends,
            candidate_trends=candidate_trends,
            comparison=comparison,
            reference_survival=reference_survival,
            candidate_survival=candidate_survival,
        )

        self.assertEqual(assessment["applicable_checks"], 10)
        self.assertEqual(assessment["passed_checks"], 10)
        self.assertAlmostEqual(float(assessment["pass_fraction"]), 1.0, places=12)

    def test_assess_paper_signature_alignment_falls_back_to_von_mises_stress(self) -> None:
        assessment = assess_paper_signature_alignment(
            reference_trends={
                "metrics": {
                    "mean_grain_area": {"delta": 0.20},
                    "grain_count": {"delta": -5.0},
                    "mean_aspect_ratio": {"delta": -0.10},
                    "fabric_c_axis_p_index": {"delta": 0.08},
                    "fabric_c_axis_vertical_fraction_15deg": {"delta": 0.05},
                    "fabric_c_axis_mean_colatitude_deg": {"delta": -3.0},
                    "mechanics_mean_von_mises_stress": {"delta": -0.20},
                    "mechanics_mean_prismatic_activity": {"delta": 0.15},
                    "mechanics_prismatic_to_basal_ratio": {"delta": 0.10},
                }
            },
            candidate_trends={
                "metrics": {
                    "mean_grain_area": {"delta": 0.18},
                    "grain_count": {"delta": -4.0},
                    "mean_aspect_ratio": {"delta": -0.08},
                    "fabric_c_axis_p_index": {"delta": 0.07},
                    "fabric_c_axis_vertical_fraction_15deg": {"delta": 0.03},
                    "fabric_c_axis_mean_colatitude_deg": {"delta": -2.0},
                    "mechanics_mean_von_mises_stress": {"delta": -0.15},
                    "mechanics_mean_prismatic_activity": {"delta": 0.12},
                    "mechanics_prismatic_to_basal_ratio": {"delta": 0.07},
                }
            },
            comparison={
                "mean_grain_area": {"normalized_rmse": 0.05},
                "grain_count": {"normalized_rmse": 0.08},
                "mean_aspect_ratio": {"normalized_rmse": 0.10},
                "fabric_c_axis_p_index": {"normalized_rmse": 0.04},
                "fabric_c_axis_vertical_fraction_15deg": {"normalized_rmse": 0.06},
                "fabric_c_axis_mean_colatitude_deg": {"normalized_rmse": 0.02},
                "mechanics_mean_von_mises_stress": {"normalized_rmse": 0.09},
                "mechanics_mean_prismatic_activity": {"normalized_rmse": 0.11},
                "mechanics_prismatic_to_basal_ratio": {"normalized_rmse": 0.12},
            },
        )

        by_name = {entry["name"]: entry for entry in assessment["checks"]}
        self.assertEqual(by_name["mean_stress_trend"]["status"], "pass")
        self.assertEqual(by_name["mean_stress_trend"]["metric"], "mechanics_mean_von_mises_stress")

    def test_assess_paper_signature_alignment_can_include_curve_checks(self) -> None:
        assessment = assess_paper_signature_alignment(
            reference_trends={"metrics": {}},
            candidate_trends={"metrics": {}},
            comparison={
                "mechanics_stress_strain_curve": {
                    "strain": {"normalized_rmse": 0.10},
                    "stress": {"normalized_rmse": 0.20},
                },
                "mechanics_activity_strain_curve": {
                    "basal_activity": {"normalized_rmse": 0.10},
                    "prismatic_activity": {"normalized_rmse": 0.15},
                    "pyramidal_activity": {"normalized_rmse": 0.25},
                    "prismatic_fraction": {"normalized_rmse": 0.20},
                },
            },
        )

        statuses = {entry["name"]: entry["status"] for entry in assessment["checks"]}
        self.assertEqual(statuses["stress_strain_curve_alignment"], "pass")
        self.assertEqual(statuses["activity_curve_alignment"], "pass")

    def test_evaluate_static_grain_growth_benchmark_emits_paper_signature_assessment(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_fabric_euler_example(Path(reference_dir) / "fabric_0001.elle")
            _write_fabric_euler_example(Path(candidate_dir) / "fabric_0001.elle")
            report = evaluate_static_grain_growth_benchmark(reference_dir, candidate_dir=candidate_dir)

        self.assertIn("paper_signature_assessment", report)
        self.assertIn("checks", report["paper_signature_assessment"])

    def test_summarize_grain_survival_diagnostics_tracks_initial_size_and_schmid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            _write_survival_sequence_examples(Path(tmpdir))
            report = summarize_grain_survival_diagnostics(tmpdir, pattern="survival_*.elle")

        self.assertEqual(report["summary"]["num_initial_grains"], 3)
        self.assertEqual(report["summary"]["num_survived_to_final"], 2)
        self.assertAlmostEqual(float(report["summary"]["survival_fraction"]), 2.0 / 3.0, places=12)
        self.assertGreater(float(report["summary"]["initial_size_survival_correlation"]), 0.0)
        self.assertTrue(np.isfinite(float(report["summary"]["initial_basal_schmid_survival_correlation"])))
        self.assertTrue(report["summary"]["initial_size_correlation_stronger_than_schmid"])

    def test_evaluate_static_grain_growth_benchmark_emits_survival_diagnostics(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_survival_sequence_examples(Path(reference_dir))
            _write_survival_sequence_examples(Path(candidate_dir))
            report = evaluate_static_grain_growth_benchmark(
                reference_dir,
                candidate_dir=candidate_dir,
                pattern="survival_*.elle",
            )

        self.assertIn("reference_survival_diagnostics", report)
        self.assertIn("candidate_survival_diagnostics", report)
        self.assertIn("survival_comparison", report)
        self.assertAlmostEqual(
            float(report["candidate_survival_diagnostics"]["summary"]["survival_fraction"]),
            2.0 / 3.0,
            places=12,
        )
        self.assertAlmostEqual(float(report["survival_comparison"]["survival_fraction_diff"]), 0.0, places=12)
        check_names = {entry["name"] for entry in report["paper_signature_assessment"]["checks"]}
        self.assertIn("size_stronger_than_schmid_survival", check_names)

    def test_summarize_elle_label_components_reads_fine_foam_grid_components(self) -> None:
        source = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )

        summary = summarize_elle_label_components(source)

        self.assertEqual(summary["attribute"], "U_ATTRIB_C")
        self.assertEqual(summary["grain_count"], 182)
        self.assertEqual(summary["source_label_count"], 136)

    def test_evaluate_rasterized_grain_growth_benchmark_reports_reference_trend(self) -> None:
        reference_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "elle" / "example" / "results"

        report = evaluate_rasterized_grain_growth_benchmark(reference_dir, pattern="fine_foam_step*.elle")

        self.assertEqual(report["reference_trends"]["num_snapshots"], 10)
        self.assertEqual(report["reference_sequence"][0]["grain_count"], 182)
        self.assertEqual(report["reference_sequence"][-1]["grain_count"], 172)
        self.assertTrue(report["reference_trends"]["flags"]["coarsening_present"])

    def test_summarize_elle_label_area_distribution_reads_component_areas(self) -> None:
        source = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )

        summary = summarize_elle_label_area_distribution(source)

        self.assertEqual(summary["attribute"], "U_ATTRIB_C")
        self.assertEqual(summary["grain_count"], 182)
        self.assertEqual(len(summary["grain_area_fractions"]), 182)
        self.assertGreater(summary["mean_grain_area"], 0.0)

    def test_build_figure2_line_validation_report_matches_identical_sequence(self) -> None:
        reference_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "elle" / "example" / "results"

        report = build_figure2_line_validation_report(
            reference_dir=reference_dir,
            candidate_dir=reference_dir,
            pattern="fine_foam_step00[1-3].elle",
            kde_points=32,
        )

        line = report["figure2_like_validation"]["mean_grain_area_line"]
        distributions = report["figure2_like_validation"]["grain_area_distributions"]

        self.assertEqual(line["steps"], [1, 2, 3])
        self.assertEqual(line["rmse"], 0.0)
        self.assertEqual(len(distributions), 3)
        self.assertEqual(len(distributions[0]["grain_area_grid"]), 32)
        self.assertIn("grain_area_histogram_bin_edges", distributions[0])
        self.assertIn("histogram_density", distributions[0]["reference"])
        self.assertEqual(
            distributions[0]["reference"]["kde"],
            distributions[0]["candidate"]["kde"],
        )

    def test_write_figure2_line_validation_html_writes_svg_report(self) -> None:
        report = {
            "reference_dir": "reference",
            "candidate_dir": "candidate",
            "figure2_like_validation": {
                "mean_grain_area_line": {
                    "steps": [1, 2, 3],
                    "reference_mean_grain_area": [0.10, 0.11, 0.13],
                    "reference_std_grain_area": [0.01, 0.01, 0.02],
                    "candidate_mean_grain_area": [0.09, 0.12, 0.14],
                    "candidate_std_grain_area": [0.01, 0.015, 0.02],
                    "rmse": 0.01,
                    "normalized_rmse": 0.05,
                },
                "grain_area_distributions": [
                    {
                        "step": 1,
                        "grain_area_grid": [0.0, 0.5, 1.0],
                        "grain_area_histogram_bin_edges": [0.0, 0.5, 1.0],
                        "reference": {
                            "grain_count": 3,
                            "histogram_density": [0.5, 0.2],
                            "kde": [0.0, 1.0, 0.0],
                        },
                        "candidate": {
                            "grain_count": 3,
                            "histogram_density": [0.4, 0.3],
                            "kde": [0.2, 0.8, 0.1],
                        },
                    },
                    {
                        "step": 2,
                        "grain_area_grid": [0.0, 0.5, 1.0],
                        "grain_area_histogram_bin_edges": [0.0, 0.5, 1.0],
                        "reference": {
                            "grain_count": 4,
                            "histogram_density": [0.45, 0.25],
                            "kde": [0.0, 0.9, 0.0],
                        },
                        "candidate": {
                            "grain_count": 4,
                            "histogram_density": [0.35, 0.3],
                            "kde": [0.1, 0.85, 0.05],
                        },
                    },
                ],
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = write_figure2_line_validation_html(Path(tmpdir) / "figure2.html", report)
            text = outpath.read_text(encoding="utf-8")

        self.assertIn("Figure 2 style grain-area validation", text)
        self.assertIn("<svg", text)
        self.assertIn("Reference mean", text)
        self.assertIn("Figure 2(a)-Style Grain-Area Distribution By Step", text)
        self.assertIn("Figure 2(b)-Style Grain-Area KDE Comparison By Step", text)
        self.assertIn("Figure 2(c)-Style Mean Grain Area Over Time", text)
        self.assertIn("step 1", text)

    def test_generate_elle_seeded_candidate_sequence_writes_matched_steps(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as output_dir:
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_step001.elle")
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_step002.elle")

            report = generate_elle_seeded_candidate_sequence(
                reference_dir=reference_dir,
                output_dir=output_dir,
                pattern="periodic_step*.elle",
                dt=0.01,
                mobility=0.5,
                gradient_penalty=1.0,
                interaction_strength=2.0,
                init_smoothing_steps=0,
                init_noise=0.0,
            )

            self.assertEqual(report["seed_attribute"], "U_ATTRIB_A")
            self.assertEqual(report["reference_steps"], [1, 2])
            self.assertEqual(len(report["saved_snapshots"]), 2)
            self.assertTrue(all(Path(path).exists() for path in report["saved_snapshots"]))
            self.assertFalse(report["reused_existing"])

    def test_generate_elle_seeded_candidate_sequence_reuses_existing_outputs(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as output_dir:
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_step001.elle")
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_step002.elle")

            first = generate_elle_seeded_candidate_sequence(
                reference_dir=reference_dir,
                output_dir=output_dir,
                pattern="periodic_step*.elle",
                dt=0.01,
                mobility=0.5,
                gradient_penalty=1.0,
                interaction_strength=2.0,
                init_smoothing_steps=0,
                init_noise=0.0,
            )
            second = generate_elle_seeded_candidate_sequence(
                reference_dir=reference_dir,
                output_dir=output_dir,
                pattern="periodic_step*.elle",
                dt=0.01,
                mobility=0.5,
                gradient_penalty=1.0,
                interaction_strength=2.0,
                init_smoothing_steps=0,
                init_noise=0.0,
            )

            self.assertFalse(first["reused_existing"])
            self.assertTrue(second["reused_existing"])
            self.assertEqual(first["saved_snapshots"], second["saved_snapshots"])

    def test_calibrate_fine_foam_ranks_small_search_space(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as output_dir:
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_step001.elle")
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_step002.elle")

            report = calibrate_fine_foam(
                reference_dir=reference_dir,
                output_dir=output_dir,
                pattern="periodic_step*.elle",
                dt_grid=[0.01],
                mobility_grid=[0.5, 1.0],
                gradient_penalty_grid=[1.0],
                interaction_strength_grid=[2.0],
                init_smoothing_steps=0,
                init_noise=0.0,
            )

            self.assertEqual(report["num_runs"], 2)
            self.assertIsNotNone(report["best_run"])
            self.assertEqual(len(report["runs"]), 2)
            self.assertTrue(Path(report["best_run"]["candidate_dir"]).exists())
            self.assertIn("rasterized_grain_count_nrmse", report["best_run"]["score"]["components"])

    def test_score_existing_calibration_runs_penalizes_incomplete_coverage(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_root:
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_step001.elle")
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_step002.elle")

            complete_dir = Path(candidate_root) / "dt0p01_m0p5_gp1_is2"
            partial_dir = Path(candidate_root) / "dt0p03_m1_gp1_is2"
            generate_elle_seeded_candidate_sequence(
                reference_dir=reference_dir,
                output_dir=complete_dir,
                pattern="periodic_step*.elle",
                dt=0.01,
                mobility=0.5,
                gradient_penalty=1.0,
                interaction_strength=2.0,
                init_smoothing_steps=0,
                init_noise=0.0,
            )
            partial_report = generate_elle_seeded_candidate_sequence(
                reference_dir=reference_dir,
                output_dir=partial_dir,
                pattern="periodic_step*.elle",
                dt=0.03,
                mobility=1.0,
                gradient_penalty=1.0,
                interaction_strength=2.0,
                init_smoothing_steps=0,
                init_noise=0.0,
            )
            Path(partial_report["saved_snapshots"][-1]).unlink()

            report = score_existing_calibration_runs(
                reference_dir=reference_dir,
                candidate_root=candidate_root,
                pattern="periodic_step*.elle",
                coverage_penalty_weight=0.25,
            )

            self.assertEqual(report["num_runs"], 2)
            self.assertTrue(report["runs"][0]["coverage"]["complete"])
            self.assertFalse(report["runs"][1]["coverage"]["complete"])
            self.assertGreater(
                report["runs"][1]["score"]["coverage_adjusted_score"],
                report["runs"][1]["score"]["score"],
            )

    def test_resolve_runtime_preset_uses_best_known_fine_foam_params(self) -> None:
        params = resolve_runtime_preset(
            "fine-foam-best",
            dt=0.05,
            mobility=1.0,
            gradient_penalty=1.0,
            interaction_strength=2.0,
            init_smoothing_steps=2,
            init_noise=0.02,
        )

        self.assertEqual(params, BEST_KNOWN_FINE_FOAM_PRESET)

    def test_resolve_runtime_preset_supports_calibrated_alias(self) -> None:
        params = resolve_runtime_preset(
            "fine-foam-calibrated",
            dt=0.05,
            mobility=1.0,
            gradient_penalty=1.0,
            interaction_strength=2.0,
            init_smoothing_steps=2,
            init_noise=0.02,
        )

        self.assertEqual(params, BEST_KNOWN_FINE_FOAM_CALIBRATED_PRESET)

    def test_resolve_runtime_preset_supports_truthful_mesh_physics(self) -> None:
        params = resolve_runtime_preset(
            "fine-foam-truthful-mesh",
            dt=0.05,
            mobility=1.0,
            gradient_penalty=1.0,
            interaction_strength=2.0,
            init_smoothing_steps=2,
            init_noise=0.02,
        )

        self.assertEqual(
            params,
            {
                "dt": BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET["dt"],
                "mobility": BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET["mobility"],
                "gradient_penalty": BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET["gradient_penalty"],
                "interaction_strength": BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET["interaction_strength"],
                "init_smoothing_steps": BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET["init_smoothing_steps"],
                "init_noise": BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET["init_noise"],
            },
        )

    def test_resolve_runtime_preset_supports_general_faithful_defaults(self) -> None:
        params = resolve_runtime_preset(
            "gbm-faithful-default",
            dt=0.04,
            mobility=0.8,
            gradient_penalty=1.1,
            interaction_strength=1.9,
            init_smoothing_steps=3,
            init_noise=0.07,
        )

        self.assertEqual(
            params,
            {
                "dt": 0.04,
                "mobility": 0.8,
                "gradient_penalty": 1.1,
                "interaction_strength": 1.9,
                "init_smoothing_steps": 0,
                "init_noise": 0.0,
            },
        )

    def test_resolve_mesh_preset_preserves_manual_values_for_old_branch(self) -> None:
        params = resolve_mesh_preset(
            "fine-foam-calibrated",
            mesh_relax_steps=0,
            mesh_topology_steps=0,
            mesh_movement_model="legacy",
            mesh_surface_diagonal_trials=False,
            mesh_use_elle_physical_units=False,
            mesh_update_mode="blend",
            mesh_random_seed=4,
            mesh_feedback_every=3,
            mesh_feedback_strength=0.2,
            mesh_transport_strength=0.8,
            mesh_kernel_every=2,
            mesh_kernel_strength=1.5,
            mesh_kernel_corrector=True,
            mesh_feedback_boundary_width=2,
        )

        self.assertEqual(
            params,
            {
                "mesh_relax_steps": 0,
                "mesh_topology_steps": 0,
                "mesh_movement_model": "legacy",
                "mesh_surface_diagonal_trials": False,
                "mesh_use_elle_physical_units": False,
                "mesh_update_mode": "blend",
                "mesh_random_seed": 4,
                "mesh_feedback_every": 3,
                "mesh_feedback_strength": 0.2,
                "mesh_transport_strength": 0.8,
                "mesh_kernel_every": 2,
                "mesh_kernel_strength": 1.5,
                "mesh_kernel_corrector": True,
                "mesh_feedback_boundary_width": 2,
            },
        )

    def test_resolve_mesh_preset_uses_truthful_mesh_values(self) -> None:
        params = resolve_mesh_preset(
            "fine-foam-truthful-mesh",
            mesh_relax_steps=0,
            mesh_topology_steps=0,
            mesh_movement_model="legacy",
            mesh_surface_diagonal_trials=False,
            mesh_use_elle_physical_units=False,
            mesh_update_mode="blend",
            mesh_random_seed=4,
            mesh_feedback_every=0,
            mesh_feedback_strength=0.0,
            mesh_transport_strength=0.0,
            mesh_kernel_every=2,
            mesh_kernel_strength=1.5,
            mesh_kernel_corrector=True,
            mesh_feedback_boundary_width=2,
        )

        self.assertEqual(params, {key: BEST_KNOWN_FINE_FOAM_TRUTHFUL_MESH_PRESET[key] for key in params})

    def test_resolve_mesh_preset_uses_general_faithful_defaults(self) -> None:
        params = resolve_mesh_preset(
            "gbm-faithful-default",
            mesh_relax_steps=9,
            mesh_topology_steps=8,
            mesh_movement_model="legacy",
            mesh_surface_diagonal_trials=False,
            mesh_use_elle_physical_units=False,
            mesh_update_mode="blend",
            mesh_random_seed=4,
            mesh_feedback_every=7,
            mesh_feedback_strength=0.6,
            mesh_transport_strength=0.7,
            mesh_kernel_every=2,
            mesh_kernel_strength=1.5,
            mesh_kernel_corrector=True,
            mesh_feedback_boundary_width=6,
        )

        self.assertEqual(
            params,
            {
                "mesh_relax_steps": int(FAITHFUL_GBM_DEFAULTS["motion_passes"]),
                "mesh_topology_steps": int(FAITHFUL_GBM_DEFAULTS["topology_passes"]),
                "mesh_movement_model": str(FAITHFUL_GBM_DEFAULTS["movement_model"]),
                "mesh_surface_diagonal_trials": bool(FAITHFUL_GBM_DEFAULTS["use_diagonal_trials"]),
                "mesh_use_elle_physical_units": bool(FAITHFUL_GBM_DEFAULTS["use_elle_physical_units"]),
                "mesh_update_mode": "mesh_only",
                "mesh_random_seed": int(FAITHFUL_GBM_DEFAULTS["random_seed"]),
                "mesh_feedback_every": int(FAITHFUL_GBM_DEFAULTS["stage_interval"]),
                "mesh_feedback_strength": 0.0,
                "mesh_transport_strength": 0.0,
                "mesh_kernel_every": 0,
                "mesh_kernel_strength": 0.0,
                "mesh_kernel_corrector": False,
                "mesh_feedback_boundary_width": int(FAITHFUL_GBM_DEFAULTS["raster_boundary_band"]),
            },
        )

    def test_build_faithful_gbm_setup_uses_truthful_mesh_defaults(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")

            setup = build_faithful_gbm_setup(elle_path)

        self.assertEqual(setup.seed_info.attribute, "U_ATTRIB_C")
        self.assertEqual(setup.config.init_elle_attribute, "U_ATTRIB_C")
        self.assertEqual(setup.mesh_feedback.update_mode, "mesh_only")
        self.assertEqual(
            setup.mesh_feedback.relax_config.movement_model,
            FAITHFUL_GBM_DEFAULTS["movement_model"],
        )
        self.assertEqual(setup.mesh_feedback.relax_config.steps, FAITHFUL_GBM_DEFAULTS["motion_passes"])
        self.assertEqual(
            setup.mesh_feedback.relax_config.topology_steps,
            FAITHFUL_GBM_DEFAULTS["topology_passes"],
        )
        self.assertEqual(setup.mesh_feedback.every, FAITHFUL_GBM_DEFAULTS["stage_interval"])
        self.assertEqual(setup.subloops_per_snapshot, FAITHFUL_GBM_DEFAULTS["subloops_per_snapshot"])
        self.assertEqual(setup.gbm_steps_per_subloop, FAITHFUL_GBM_DEFAULTS["gbm_steps_per_subloop"])
        self.assertEqual(setup.mesh_feedback.boundary_width, FAITHFUL_GBM_DEFAULTS["raster_boundary_band"])
        self.assertTrue(bool(setup.mesh_feedback.relax_config.use_diagonal_trials))
        self.assertTrue(bool(setup.mesh_feedback.relax_config.use_elle_physical_units))
        self.assertEqual(setup.mesh_feedback.strength, 0.0)
        self.assertEqual(setup.mesh_feedback.transport_strength, 0.0)
        self.assertEqual(setup.config.nx, 2)
        self.assertEqual(setup.config.ny, 2)
        self.assertEqual(setup.config.num_grains, 2)
        self.assertEqual(setup.mesh_feedback.relax_config.temperature_c, -10.0)
        np.testing.assert_array_equal(setup.config.seed_data.label_field, np.array([[0, 0], [1, 1]], dtype=np.int32))

    def test_load_faithful_seed_derives_labels_from_flynn_polygons(self) -> None:
        elle_path = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "fine_foam.elle"
        )

        seed = load_faithful_seed(elle_path)

        self.assertEqual(seed.attribute, "derived_from_flynns")
        self.assertEqual(seed.grid_shape, (128, 128))
        self.assertEqual(seed.label_field.shape, (128, 128))
        self.assertEqual(seed.unode_field_order, ())
        self.assertGreater(seed.num_labels, 0)
        self.assertEqual(seed.num_labels, len(seed.source_labels))
        self.assertEqual(seed.num_labels, 134)

    def test_build_faithful_gbm_setup_accepts_raw_fine_foam_launch_seed(self) -> None:
        elle_path = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "fine_foam.elle"
        )

        setup = build_faithful_gbm_setup(elle_path)

        self.assertEqual(setup.seed_info.attribute, "derived_from_flynns")
        self.assertEqual(setup.config.init_elle_attribute, "derived_from_flynns")
        self.assertEqual(setup.config.nx, 128)
        self.assertEqual(setup.config.ny, 128)
        self.assertEqual(setup.config.num_grains, setup.seed_info.num_labels)
        self.assertEqual(setup.mesh_seed["stats"]["mesh_seed_source"], "elle")
        self.assertEqual(setup.seed_info.num_labels, 134)
        self.assertEqual(setup.mesh_seed["stats"]["num_flynns"], 134)
        self.assertEqual(setup.mesh_seed["stats"]["mesh_seed_extended_source_labels"], 0)
        self.assertIn("_runtime_seed_unode_fields", setup.mesh_seed)
        self.assertIn("_runtime_seed_unode_sections", setup.mesh_seed)
        runtime_fields = setup.mesh_seed["_runtime_seed_unode_fields"]
        runtime_sections = setup.mesh_seed["_runtime_seed_unode_sections"]
        flynn_sections = setup.mesh_seed["_runtime_seed_flynn_sections"]
        self.assertEqual(runtime_fields["label_attribute"], "U_ATTRIB_C")
        self.assertIn("U_ATTRIB_C", runtime_fields["values"])
        self.assertIn("U_ATTRIB_A", runtime_fields["values"])
        self.assertIn("U_ATTRIB_B", runtime_fields["values"])
        self.assertIn("U_ATTRIB_D", runtime_fields["values"])
        self.assertIn("U_ATTRIB_E", runtime_fields["values"])
        self.assertIn("U_DISLOCDEN", runtime_fields["values"])
        self.assertIn("U_EULER_3", runtime_sections["values"])
        self.assertEqual(runtime_sections["component_counts"]["U_EULER_3"], 3)
        self.assertIn("F_ATTRIB_C", flynn_sections["values"])

    def test_initial_raw_seed_topology_pass_applies_for_polygon_seed(self) -> None:
        elle_path = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "fine_foam.elle"
        )
        setup = build_faithful_gbm_setup(elle_path)
        captured: dict[str, object] = {}

        def _fake_relax(mesh_state: dict[str, object], config: object) -> dict[str, object]:
            captured["config"] = config
            updated = copy.deepcopy(mesh_state)
            updated.setdefault("stats", {})
            updated["stats"]["num_flynns"] = 136
            return updated

        with patch("elle_jax_model.faithful_runtime.relax_mesh_state", side_effect=_fake_relax):
            updated = _apply_initial_raw_seed_topology_pass(
                setup.config,
                copy.deepcopy(setup.mesh_seed),
                setup.mesh_feedback,
                has_mechanics_snapshot=False,
            )

        self.assertIn("config", captured)
        config = captured["config"]
        self.assertEqual(config.steps, 0)
        self.assertGreaterEqual(config.topology_steps, 1)
        self.assertEqual(updated["stats"]["num_flynns"], 136)
        self.assertEqual(updated["stats"]["workflow_initial_topocheck_applied"], 1)

    def test_initial_raw_seed_topology_pass_skips_processed_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            setup = build_faithful_gbm_setup(elle_path)

        with patch("elle_jax_model.faithful_runtime.relax_mesh_state") as relax_mock:
            updated = _apply_initial_raw_seed_topology_pass(
                setup.config,
                copy.deepcopy(setup.mesh_seed),
                setup.mesh_feedback,
                has_mechanics_snapshot=False,
            )

        relax_mock.assert_not_called()
        self.assertEqual(updated["stats"]["num_flynns"], setup.mesh_seed["stats"]["num_flynns"])

    def test_current_runtime_labels_prefers_mesh_state_over_phi(self) -> None:
        phi = np.zeros((2, 2, 2), dtype=np.float32)
        phi[0, :, :] = 1.0
        runtime_mesh_state = {
            "_runtime_seed_unodes": {
                "ids": (1, 2, 3, 4),
                "positions": (
                    (0.25, 0.75),
                    (0.25, 0.25),
                    (0.75, 0.75),
                    (0.75, 0.25),
                ),
                "grid_indices": ((0, 0), (0, 1), (1, 0), (1, 1)),
                "grid_shape": (2, 2),
            },
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 10, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 11, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [2, 2]},
            "events": [],
        }

        labels = _current_runtime_labels(phi, runtime_mesh_state)

        np.testing.assert_array_equal(labels, np.array([[0, 0], [1, 1]], dtype=np.int32))

    def test_current_runtime_labels_distinguishes_current_flynns_with_shared_source_label(self) -> None:
        phi = np.zeros((2, 2, 2), dtype=np.float32)
        phi[0, :, :] = 1.0
        runtime_mesh_state = {
            "_runtime_seed_unodes": {
                "ids": (1, 2, 3, 4),
                "positions": (
                    (0.25, 0.75),
                    (0.25, 0.25),
                    (0.75, 0.75),
                    (0.75, 0.25),
                ),
                "grid_indices": ((0, 0), (0, 1), (1, 0), (1, 1)),
                "grid_shape": (2, 2),
            },
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 10, "label": 7, "source_label": 7, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 11, "label": 7, "source_label": 7, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [2, 2]},
            "events": [],
        }

        labels = _current_runtime_labels(phi, runtime_mesh_state)

        np.testing.assert_array_equal(labels, np.array([[0, 0], [1, 1]], dtype=np.int32))

    def test_build_faithful_gbm_setup_inherits_seed_elle_options_by_default(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")

            setup = build_faithful_gbm_setup(elle_path)

        self.assertEqual(setup.seed_info.elle_options.get("Temperature"), -10.0)
        self.assertEqual(setup.mesh_feedback.relax_config.temperature_c, -10.0)
        self.assertEqual(setup.mesh_seed["_runtime_elle_options"]["scalar_values"]["Temperature"], -10.0)
        self.assertEqual(setup.mesh_seed["_runtime_elle_options"]["scalar_values"]["Pressure"], 8800000.0)

    def test_build_faithful_gbm_setup_applies_runtime_temperature_override_to_export_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")

            setup = build_faithful_gbm_setup(elle_path, temperature_c=-5.0)

        self.assertEqual(setup.mesh_feedback.relax_config.temperature_c, -5.0)
        self.assertEqual(setup.mesh_seed["_runtime_elle_options"]["scalar_values"]["Temperature"], -5.0)

    def test_build_faithful_gbm_setup_derives_raster_boundary_band_from_elle_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            text = elle_path.read_text(encoding="utf-8")
            text = text.replace(
                "BoundaryWidth 0.1\nMassIncrement 0.02",
                "BoundaryWidth 0.4\nMassIncrement 0.02",
            )
            text = text.replace("UnitLength 0.5", "UnitLength 0.2")
            elle_path.write_text(text, encoding="utf-8")

            setup = build_faithful_gbm_setup(elle_path)

        self.assertEqual(setup.mesh_feedback.boundary_width, 4)

    def test_build_faithful_gbm_setup_infers_flynn_labels_when_ids_do_not_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = Path(tmpdir) / "seed.elle"
            elle_path.write_text(
                "\n".join(
                    [
                        "LOCATION",
                        "10 0.0 0.0",
                        "11 0.5 0.0",
                        "12 0.5 1.0",
                        "13 0.0 1.0",
                        "20 1.0 0.0",
                        "21 1.0 1.0",
                        "FLYNNS",
                        "10 4 10 11 12 13",
                        "20 4 11 20 21 12",
                        "UNODES",
                        "0 0.25 0.75",
                        "1 0.25 0.25",
                        "2 0.75 0.75",
                        "3 0.75 0.25",
                        "U_ATTRIB_A",
                        "Default 0.0",
                        "0 1",
                        "1 1",
                        "2 2",
                        "3 2",
                        "",
                    ]
                ),
                encoding="utf-8",
            )

            setup = build_faithful_gbm_setup(elle_path, init_elle_attribute="U_ATTRIB_A")

        self.assertEqual(setup.seed_info.source_labels, (1, 2))
        self.assertEqual(setup.mesh_seed["stats"]["num_flynns"], 2)
        self.assertEqual([flynn["flynn_id"] for flynn in setup.mesh_seed["flynns"]], [10, 20])
        self.assertEqual([flynn["label"] for flynn in setup.mesh_seed["flynns"]], [0, 1])

    def test_build_faithful_gbm_setup_supports_stage_named_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")

            setup = build_faithful_gbm_setup(
                elle_path,
                movement_model="legacy",
                motion_passes=2,
                topology_passes=3,
                stage_interval=4,
                subloops_per_snapshot=5,
                gbm_steps_per_subloop=2,
                raster_boundary_band=5,
                use_diagonal_trials=False,
                use_elle_physical_units=False,
            )

        self.assertEqual(setup.mesh_feedback.relax_config.movement_model, "legacy")
        self.assertEqual(setup.mesh_feedback.relax_config.steps, 2)
        self.assertEqual(setup.mesh_feedback.relax_config.topology_steps, 3)
        self.assertEqual(setup.mesh_feedback.every, 4)
        self.assertEqual(setup.subloops_per_snapshot, 5)
        self.assertEqual(setup.gbm_steps_per_subloop, 2)
        self.assertEqual(setup.mesh_feedback.boundary_width, 5)
        self.assertFalse(bool(setup.mesh_feedback.relax_config.use_diagonal_trials))
        self.assertFalse(bool(setup.mesh_feedback.relax_config.use_elle_physical_units))

    def test_run_faithful_gbm_simulation_exposes_numpy_mesh_only_backend(self) -> None:
        contexts = []

        def record_snapshot(step, phi, topology_snapshot, mesh_feedback_context) -> None:
            contexts.append(mesh_feedback_context)

        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            final_state, snapshots, topology_history = run_faithful_gbm_simulation(
                init_elle_path=elle_path,
                steps=1,
                save_every=1,
                on_snapshot=record_snapshot,
                mesh_relax_steps=0,
                mesh_topology_steps=0,
            )

        self.assertEqual(len(snapshots), 1)
        self.assertTrue(topology_history)
        self.assertEqual(contexts[0]["solver_backend"], "numpy_mesh_only")
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_seed_source"], "elle")
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_solver_backend"], "numpy_mesh_only")
        self.assertEqual(np.asarray(final_state).shape, (2, 2, 2))

    def test_run_faithful_gbm_simulation_is_deterministic_for_same_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            default_setup = build_faithful_gbm_setup(elle_path)

            default_state, _, _ = run_faithful_gbm_simulation(
                steps=1,
                save_every=1,
                setup=default_setup,
            )
            repeated_state, _, _ = run_faithful_gbm_simulation(
                steps=1,
                save_every=1,
                setup=default_setup,
            )

        np.testing.assert_allclose(np.asarray(default_state), np.asarray(repeated_state))

    def test_run_faithful_gbm_simulation_can_emit_initial_snapshot(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            observed_steps: list[int] = []
            final_state, snapshots, topology_history = run_faithful_gbm_simulation(
                init_elle_path=elle_path,
                steps=1,
                save_every=1,
                include_initial_snapshot=True,
                mesh_relax_steps=0,
                mesh_topology_steps=0,
                on_snapshot=lambda step, *_args: observed_steps.append(int(step)),
            )

        self.assertEqual(observed_steps, [0, 1])
        self.assertEqual(len(snapshots), 2)
        self.assertEqual(np.asarray(snapshots[0]).shape, (2, 2, 2))
        self.assertEqual(np.asarray(final_state).shape, (2, 2, 2))
        self.assertTrue(any(snapshot["step"] == 0 for snapshot in topology_history))

    def test_load_legacy_fft_snapshot_reads_bridge_snapshot_files(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "temp-FFT.out").write_text(
                "\n".join(
                    [
                        "1 0 0",
                        "0 2 0",
                        "0 0 3",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "unodexyz.out").write_text(
                "\n".join(
                    [
                        "0 0.1 0.2 0.3",
                        "1 0.4 0.5 0.6",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "unodeang.out").write_text(
                "\n".join(
                    [
                        "0 10 20 30",
                        "1 40 50 60",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "tex.out").write_text(
                "\n".join(
                    [
                        "0 1 2 3 4 5 6 7 8 9 10 11",
                        "1 2 3 4 14 15 16 17 18 19 20 21",
                    ]
                ),
                encoding="utf-8",
            )

            snapshot = load_legacy_fft_snapshot(root)

        np.testing.assert_allclose(
            snapshot.temp_matrix,
            np.asarray([[1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]], dtype=np.float64),
        )
        np.testing.assert_array_equal(snapshot.unode_ids, np.asarray([0, 1], dtype=np.int32))
        np.testing.assert_allclose(
            snapshot.unode_strain_xyz,
            np.asarray([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]], dtype=np.float64),
        )
        np.testing.assert_array_equal(snapshot.euler_ids, np.asarray([0, 1], dtype=np.int32))
        np.testing.assert_allclose(
            snapshot.unode_euler_deg,
            np.asarray([[10.0, 20.0, 30.0], [40.0, 50.0, 60.0]], dtype=np.float64),
        )
        np.testing.assert_allclose(snapshot.normalized_strain_rate, np.asarray([4.0, 14.0]))
        np.testing.assert_allclose(snapshot.normalized_stress, np.asarray([5.0, 15.0]))
        np.testing.assert_allclose(snapshot.basal_activity, np.asarray([6.0, 16.0]))
        np.testing.assert_allclose(snapshot.prismatic_activity, np.asarray([7.0, 17.0]))
        np.testing.assert_allclose(snapshot.geometrical_dislocation_density, np.asarray([8.0, 18.0]))
        np.testing.assert_allclose(snapshot.statistical_dislocation_density, np.asarray([9.0, 19.0]))
        np.testing.assert_array_equal(snapshot.fourier_point_ids, np.asarray([10, 20], dtype=np.int32))
        np.testing.assert_array_equal(snapshot.fft_grain_numbers, np.asarray([11, 21], dtype=np.int32))

    def test_build_legacy_fft_bridge_payload_maps_named_legacy_channels(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(_write_legacy_fft_snapshot_example(Path(tmpdir)))

        payload = build_legacy_fft_bridge_payload(np.asarray([1, 2, 3, 4], dtype=np.int32), snapshot)

        self.assertEqual(payload.alignment_mode, "exact_ids")
        self.assertEqual(payload.euler_alignment_mode, "exact_ids")
        np.testing.assert_array_equal(payload.ordered_unode_ids, np.asarray([1, 2, 3, 4], dtype=np.int32))
        np.testing.assert_allclose(payload.cell_lengths, np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(payload.cell_strain_triplet, np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(payload.cell_shear_triplet, np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
        np.testing.assert_allclose(
            payload.unode_strain_xyz,
            np.asarray(
                [
                    [0.1, 0.2, 0.3],
                    [0.2, 0.4, 0.6],
                    [0.3, 0.6, 0.9],
                    [0.4, 0.8, 1.2],
                ],
                dtype=np.float64,
            ),
        )
        np.testing.assert_allclose(
            payload.unode_euler_deg,
            np.asarray(
                [
                    [10.0, 20.0, 30.0],
                    [20.0, 40.0, 60.0],
                    [30.0, 60.0, 90.0],
                    [40.0, 80.0, 120.0],
                ],
                dtype=np.float64,
            ),
        )
        np.testing.assert_allclose(payload.normalized_strain_rate, np.asarray([4.0, 14.0, 24.0, 34.0], dtype=np.float64))
        np.testing.assert_allclose(payload.normalized_stress, np.asarray([5.0, 15.0, 25.0, 35.0], dtype=np.float64))
        np.testing.assert_allclose(payload.basal_activity, np.asarray([6.0, 16.0, 26.0, 36.0], dtype=np.float64))
        np.testing.assert_allclose(payload.prismatic_activity, np.asarray([7.0, 17.0, 27.0, 37.0], dtype=np.float64))
        np.testing.assert_allclose(
            payload.geometrical_dislocation_density_increment,
            np.asarray([8.0, 18.0, 28.0, 38.0], dtype=np.float64),
        )
        np.testing.assert_allclose(
            payload.statistical_dislocation_density,
            np.asarray([9.0, 19.0, 29.0, 39.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(payload.fourier_point_ids, np.asarray([10, 20, 30, 40], dtype=np.int32))
        np.testing.assert_array_equal(payload.fft_grain_numbers, np.asarray([11, 21, 31, 41], dtype=np.int32))

    def test_build_legacy_fft_bridge_payload_aligns_zero_based_snapshot_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(Path(tmpdir), zero_based_ids=True)
            )

        payload = build_legacy_fft_bridge_payload(np.asarray([1, 2, 3, 4], dtype=np.int32), snapshot)

        self.assertEqual(payload.alignment_mode, "snapshot_zero_based")
        self.assertEqual(payload.euler_alignment_mode, "snapshot_zero_based")
        np.testing.assert_allclose(payload.normalized_strain_rate, np.asarray([4.0, 14.0, 24.0, 34.0], dtype=np.float64))

    def test_load_legacy_fft_snapshot_falls_back_to_temp_out_and_allows_missing_tex(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            (root / "temp.out").write_text(
                "\n".join(
                    [
                        "5 6 7",
                        "8 9 10",
                        "11 12 13",
                    ]
                ),
                encoding="utf-8",
            )
            (root / "unodexyz.out").write_text("0 1 2 3\n", encoding="utf-8")
            (root / "unodeang.out").write_text("0 4 5 6\n", encoding="utf-8")

            snapshot = load_legacy_fft_snapshot(root)

        np.testing.assert_allclose(
            snapshot.temp_matrix,
            np.asarray([[5.0, 6.0, 7.0], [8.0, 9.0, 10.0], [11.0, 12.0, 13.0]], dtype=np.float64),
        )
        self.assertTrue(snapshot.paths.temp_matrix_path.endswith("temp.out"))
        self.assertIsNone(snapshot.tex_columns)
        self.assertIsNone(snapshot.normalized_strain_rate)

    def test_load_legacy_fft_snapshot_sequence_reads_sorted_snapshot_subdirectories(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            _write_legacy_fft_snapshot_example(root / "step002", value_offset=100.0)
            _write_legacy_fft_snapshot_example(root / "step001", value_offset=0.0)

            snapshots = load_legacy_fft_snapshot_sequence(root)

        self.assertEqual(len(snapshots), 2)
        self.assertTrue(snapshots[0].paths.temp_matrix_path.endswith("step001/temp-FFT.out"))
        np.testing.assert_allclose(snapshots[0].normalized_strain_rate, [4.0, 14.0, 24.0, 34.0])
        np.testing.assert_allclose(snapshots[1].normalized_strain_rate, [104.0, 114.0, 124.0, 134.0])

    def test_load_legacy_elle2fft_bridge_payload_reads_shipped_step0_contract(self) -> None:
        source_dir = (
            PROJECT_ROOT.parent / "processes" / "fft" / "example" / "step0"
        )

        payload = load_legacy_elle2fft_bridge_payload(source_dir)

        self.assertEqual(payload.grain_count, 16)
        self.assertEqual(len(payload.grain_rows), 16)
        self.assertEqual(len(payload.point_rows), 256 * 256)
        self.assertEqual(payload.grain_rows[0][-1], 3)
        self.assertEqual(payload.grain_rows[-1][-1], 53)
        self.assertEqual(payload.point_rows[0][3:], (1, 1, 1, 9, 1))
        self.assertEqual(payload.point_rows[-1][3:], (256, 256, 1, 9, 1))
        np.testing.assert_allclose(
            np.asarray(payload.temp_rows, dtype=np.float64),
            np.asarray(
                [
                    [1.0, 1.0, 1.0],
                    [0.0, 0.0, 0.0],
                    [0.0, 0.0, 0.0],
                ],
                dtype=np.float64,
            ),
        )

    def test_build_legacy_elle2fft_bridge_payload_matches_shipped_step0_point_and_temp_blocks(self) -> None:
        source_dir = (
            PROJECT_ROOT.parent / "processes" / "fft" / "example" / "step0"
        )
        setup = build_faithful_gbm_setup(source_dir / "inifft001.elle")

        generated = build_legacy_elle2fft_bridge_payload(setup.mesh_seed)
        reference = load_legacy_elle2fft_bridge_payload(source_dir)

        self.assertEqual(generated.grain_count, reference.grain_count)
        self.assertEqual(
            tuple(int(row[-1]) for row in generated.grain_rows),
            tuple(int(row[-1]) for row in reference.grain_rows),
        )
        np.testing.assert_allclose(
            np.asarray(generated.point_rows, dtype=np.float64),
            np.asarray(reference.point_rows, dtype=np.float64),
            atol=5.0e-3,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(generated.temp_rows, dtype=np.float64),
            np.asarray(reference.temp_rows, dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_build_legacy_elle2fft_bridge_payload_supports_explicit_phase_attribute(self) -> None:
        mesh_state = _build_mechanics_phase_mesh_state()
        mesh_state["_runtime_seed_flynn_sections"] = {
            "field_order": ("VISCOSITY", "DISLOCDEN"),
            "id_order": (10, 20),
            "defaults": {
                "VISCOSITY": 0.0,
                "DISLOCDEN": 0.0,
            },
            "values": {
                "VISCOSITY": (1.0, 2.0),
                "DISLOCDEN": (3.0, 4.0),
            },
        }

        viscosity_payload = build_legacy_elle2fft_bridge_payload(
            mesh_state,
            phase_attribute="VISCOSITY",
        )
        dislocden_payload = build_legacy_elle2fft_bridge_payload(
            mesh_state,
            phase_attribute="DISLOCDEN",
        )

        self.assertEqual(
            tuple(int(row[-1]) for row in viscosity_payload.point_rows),
            (1, 2, 1, 2),
        )
        self.assertEqual(
            tuple(int(row[-1]) for row in dislocden_payload.point_rows),
            (3, 4, 3, 4),
        )

    def test_build_legacy_elle2fft_bridge_payload_auto_falls_back_to_dislocden_phase_source(self) -> None:
        mesh_state = _build_mechanics_phase_mesh_state()
        mesh_state["_runtime_seed_flynn_sections"] = {
            "field_order": ("DISLOCDEN",),
            "id_order": (10, 20),
            "defaults": {
                "DISLOCDEN": 0.0,
            },
            "values": {
                "DISLOCDEN": (7.0, 8.0),
            },
        }

        payload = build_legacy_elle2fft_bridge_payload(mesh_state)

        self.assertEqual(
            tuple(int(row[-1]) for row in payload.point_rows),
            (7, 8, 7, 8),
        )

    def test_build_legacy_elle2fft_bridge_payload_can_exclude_grain_headers(self) -> None:
        mesh_state = _build_mechanics_phase_mesh_state()

        payload = build_legacy_elle2fft_bridge_payload(
            mesh_state,
            include_grain_headers=False,
        )

        self.assertEqual(payload.grain_count, 0)
        self.assertEqual(payload.grain_rows, ())
        self.assertEqual(
            tuple(int(row[6]) for row in payload.point_rows),
            (0, 0, 0, 0),
        )

    def test_write_legacy_elle2fft_bridge_payload_roundtrips_payload(self) -> None:
        source_dir = (
            PROJECT_ROOT.parent / "processes" / "fft" / "example" / "step0"
        )
        setup = build_faithful_gbm_setup(source_dir / "inifft001.elle")
        payload = build_legacy_elle2fft_bridge_payload(setup.mesh_seed)

        with tempfile.TemporaryDirectory() as tmpdir:
            make_path, temp_path = write_legacy_elle2fft_bridge_payload(tmpdir, payload)
            roundtrip = load_legacy_elle2fft_bridge_payload(tmpdir)

        self.assertTrue(str(make_path).endswith("make.out"))
        self.assertTrue(str(temp_path).endswith("temp.out"))
        self.assertEqual(roundtrip.grain_count, payload.grain_count)
        np.testing.assert_allclose(
            np.asarray(roundtrip.grain_rows, dtype=np.float64),
            np.asarray(payload.grain_rows, dtype=np.float64),
            atol=5.0e-3,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(roundtrip.point_rows, dtype=np.float64),
            np.asarray(payload.point_rows, dtype=np.float64),
            atol=5.0e-3,
            rtol=0.0,
        )
        np.testing.assert_allclose(
            np.asarray(roundtrip.temp_rows, dtype=np.float64),
            np.asarray(payload.temp_rows, dtype=np.float64),
            atol=1.0e-12,
            rtol=0.0,
        )

    def test_compare_legacy_elle2fft_bridge_payload_exposes_remaining_grain_header_gap(self) -> None:
        source_dir = (
            PROJECT_ROOT.parent / "processes" / "fft" / "example" / "step0"
        )
        setup = build_faithful_gbm_setup(source_dir / "inifft001.elle")
        candidate = build_legacy_elle2fft_bridge_payload(setup.mesh_seed)
        reference = load_legacy_elle2fft_bridge_payload(source_dir)

        report = compare_legacy_elle2fft_bridge_payload(candidate, reference)

        self.assertEqual(report["grain_count_match"], 1)
        self.assertEqual(report["grain_rows_shape_match"], 1)
        self.assertEqual(report["point_rows_shape_match"], 1)
        self.assertEqual(report["temp_rows_shape_match"], 1)
        self.assertEqual(report["grain_header_flynn_id_match_count"], 16)
        self.assertLess(float(report["point_euler_rmse"]), 5.0e-3)
        self.assertEqual(report["point_contract_match"], 1)
        self.assertAlmostEqual(report["temp_rows_numeric_rmse"], 0.0, places=12)
        self.assertEqual(report["temp_contract_match"], 1)
        self.assertGreater(float(report["grain_header_euler_rmse"]), 1.0)
        self.assertEqual(report["grain_header_contract_match"], 0)
        self.assertEqual(report["bridge_contract_match_excluding_grain_headers"], 1)
        self.assertEqual(report["bridge_contract_match_full"], 0)
        self.assertEqual(report["bridge_header_only_mismatch"], 1)

    def test_diagnose_legacy_elle2fft_header_sources_reports_best_known_candidate(self) -> None:
        source_dir = (
            PROJECT_ROOT.parent / "processes" / "fft" / "example" / "step0"
        )
        setup = build_faithful_gbm_setup(source_dir / "inifft001.elle")
        reference = load_legacy_elle2fft_bridge_payload(source_dir)

        diagnostics = diagnose_legacy_elle2fft_header_sources(setup.mesh_seed, reference)

        self.assertIn("candidate_rmse", diagnostics)
        self.assertIn("best_candidate", diagnostics)
        self.assertIn("best_candidate_rmse", diagnostics)
        self.assertIn("flynn_euler_3", diagnostics["candidate_rmse"])
        self.assertIn("mean_unode_euler_3", diagnostics["candidate_rmse"])
        self.assertIn("median_unode_euler_3", diagnostics["candidate_rmse"])
        self.assertIn("first_unode_euler_3", diagnostics["candidate_rmse"])
        self.assertIn(
            diagnostics["best_candidate"],
            {"mean_unode_euler_3", "median_unode_euler_3", "first_unode_euler_3"},
        )
        self.assertLessEqual(
            float(diagnostics["candidate_rmse"]["mean_unode_euler_3"]),
            float(diagnostics["candidate_rmse"]["flynn_euler_3"]),
        )
        self.assertGreater(float(diagnostics["best_candidate_rmse"]), 1.0)

    def test_apply_legacy_fft_snapshot_to_mesh_state_updates_runtime_mechanics_fields(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(_write_legacy_fft_snapshot_example(Path(tmpdir)))

        mesh_state = _build_recovery_mesh_state(include_attr_f=False)
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(mesh_state, snapshot)

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        updated_sections = updated_mesh["_runtime_seed_unode_sections"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_sections["U_EULER_3"], dtype=np.float64),
            np.asarray(
                [
                    [10.0, 20.0, 30.0],
                    [20.0, 40.0, 60.0],
                    [30.0, 60.0, 90.0],
                    [40.0, 80.0, 120.0],
                ],
                dtype=np.float64,
            ),
        )
        np.testing.assert_allclose(np.asarray(updated_fields["U_ATTRIB_A"], dtype=np.float64), [4.0, 14.0, 24.0, 34.0])
        np.testing.assert_allclose(np.asarray(updated_fields["U_ATTRIB_B"], dtype=np.float64), [5.0, 15.0, 25.0, 35.0])
        np.testing.assert_allclose(np.asarray(updated_fields["U_ATTRIB_D"], dtype=np.float64), [6.0, 16.0, 26.0, 36.0])
        np.testing.assert_allclose(np.asarray(updated_fields["U_ATTRIB_E"], dtype=np.float64), [7.0, 17.0, 27.0, 37.0])
        np.testing.assert_allclose(np.asarray(updated_fields["U_DISLOCDEN"], dtype=np.float64), [18.0, 28.0, 38.0, 48.0])
        self.assertEqual(mechanics_stats["mechanics_applied"], 1)
        self.assertEqual(mechanics_stats["alignment_mode"], "exact_ids")
        self.assertEqual(mechanics_stats["updated_unodes"], 4)
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_applied"], 1)
        runtime_snapshot = updated_mesh["_runtime_mechanics_snapshot"]
        np.testing.assert_allclose(runtime_snapshot["cell_lengths"], np.asarray([1.0, 0.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(runtime_snapshot["cell_strain_triplet"], np.asarray([0.0, 1.0, 0.0], dtype=np.float64))
        np.testing.assert_allclose(runtime_snapshot["cell_shear_triplet"], np.asarray([0.0, 0.0, 1.0], dtype=np.float64))
        np.testing.assert_allclose(
            runtime_snapshot["geometrical_dislocation_density_increment"],
            np.asarray([8.0, 18.0, 28.0, 38.0], dtype=np.float64),
        )
        np.testing.assert_array_equal(runtime_snapshot["fourier_point_ids"], np.asarray([10, 20, 30, 40], dtype=np.int32))
        np.testing.assert_array_equal(runtime_snapshot["fft_grain_numbers"], np.asarray([11, 21, 31, 41], dtype=np.int32))
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_has_cell_reset_payload"], 1)
        np.testing.assert_allclose(
            np.asarray(updated_mesh["_runtime_seed_unodes"]["positions"], dtype=np.float64),
            np.asarray(
                [
                    [0.1, 0.2],
                    [0.2, 0.4],
                    [0.3, 0.6],
                    [0.4, 0.8],
                ],
                dtype=np.float64,
            ),
        )
        self.assertEqual(mechanics_stats["position_update_mode"], "pure_shear_xy")
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_updated_unode_positions"], 4)

    def test_apply_legacy_fft_snapshot_to_mesh_state_applies_cell_reset_and_simple_shear_x_only(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    temp_rows=(
                        (1.0, 0.0, 0.0),
                        (0.1, 0.0, 0.0),
                        (0.05, 0.0, 0.0),
                    ),
                )
            )

        mesh_state = _build_recovery_mesh_state(include_attr_f=False)
        mesh_state["_runtime_elle_options"] = {
            "scalar_values": {
                "Temperature": -10.0,
            },
            "cell_bounding_box": [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            "simple_shear_offset": 0.0,
            "cumulative_simple_shear": 0.0,
        }
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(mesh_state, snapshot)

        np.testing.assert_allclose(
            np.asarray(updated_mesh["_runtime_seed_unodes"]["positions"], dtype=np.float64),
            np.asarray(
                [
                    [0.1, 0.25],
                    [0.2, 0.75],
                    [0.3, 0.25],
                    [0.4, 0.75],
                ],
                dtype=np.float64,
            ),
        )
        updated_options = updated_mesh["_runtime_elle_options"]
        self.assertEqual(mechanics_stats["position_update_mode"], "simple_shear_x_only")
        self.assertEqual(mechanics_stats["cell_reset_applied"], 1)
        self.assertAlmostEqual(updated_options["simple_shear_offset"], 0.05)
        self.assertAlmostEqual(updated_options["cumulative_simple_shear"], 0.05)
        self.assertEqual(
            updated_options["cell_bounding_box"],
            [
                [0.0, 0.0],
                [1.1, 0.0],
                [1.1500000000000001, 1.0],
                [0.05, 1.0],
            ],
        )
        self.assertAlmostEqual(
            float(updated_mesh["mechanics_payload_summary"]["direct_strain_axis"]),
            0.05,
            places=12,
        )
        self.assertEqual(
            updated_mesh["mechanics_payload_summary"]["strain_axis_source"],
            "cumulative_simple_shear",
        )

    def test_apply_legacy_fft_snapshot_to_mesh_state_tracks_vertical_shortening_direct_strain_axis(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    temp_rows=(
                        (1.0, 0.0, 0.0),
                        (0.0, -0.1, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                )
            )

        mesh_state = _build_recovery_mesh_state(include_attr_f=False)
        mesh_state["_runtime_elle_options"] = {
            "scalar_values": {
                "Temperature": -10.0,
            },
            "cell_bounding_box": [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
            ],
            "simple_shear_offset": 0.0,
            "cumulative_simple_shear": 0.0,
        }
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(mesh_state, snapshot)

        self.assertEqual(mechanics_stats["cell_reset_applied"], 1)
        self.assertAlmostEqual(
            float(updated_mesh["mechanics_payload_summary"]["direct_strain_axis"]),
            10.0,
            places=12,
        )
        self.assertEqual(
            updated_mesh["mechanics_payload_summary"]["strain_axis_source"],
            "vertical_shortening_pct",
        )
        self.assertAlmostEqual(
            float(updated_mesh["stats"]["mechanics_snapshot_direct_strain_axis"]),
            10.0,
            places=12,
        )
        self.assertEqual(
            updated_mesh["stats"]["mechanics_snapshot_strain_axis_source"],
            "vertical_shortening_pct",
        )

    def test_apply_legacy_fft_snapshot_to_mesh_state_updates_boundary_node_positions_from_mechanics_displacement(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    temp_rows=(
                        (1.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    unodexyz_rows=(
                        (0.35, 0.25, 0.0),
                        (0.35, 0.75, 0.0),
                        (0.85, 0.25, 0.0),
                        (0.85, 0.75, 0.0),
                    ),
                )
            )

        mesh_state = _build_mechanics_phase_mesh_state()
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(mesh_state, snapshot)

        np.testing.assert_allclose(
            np.asarray([[node["x"], node["y"]] for node in updated_mesh["nodes"]], dtype=np.float64),
            np.asarray(
                [
                    [0.1, 0.0],
                    [0.6, 0.0],
                    [0.6, 0.0],
                    [0.1, 0.0],
                    [0.1, 0.0],
                    [0.1, 0.0],
                ],
                dtype=np.float64,
            ),
            atol=1.0e-12,
            rtol=0.0,
        )
        self.assertEqual(mechanics_stats["updated_node_positions"], 6)
        self.assertEqual(mechanics_stats["node_position_update_mode"], "legacy_bnode_strain_weighted")
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_updated_node_positions"], 6)
        self.assertEqual(
            updated_mesh["stats"]["mechanics_snapshot_node_position_update_mode"],
            "legacy_bnode_strain_weighted",
        )

    def test_apply_legacy_fft_snapshot_to_mesh_state_aligns_zero_based_snapshot_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(Path(tmpdir), zero_based_ids=True)
            )

        mesh_state = _build_recovery_mesh_state(include_attr_f=False)
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(mesh_state, snapshot)

        self.assertEqual(mechanics_stats["alignment_mode"], "snapshot_zero_based")
        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(np.asarray(updated_fields["U_ATTRIB_A"], dtype=np.float64), [4.0, 14.0, 24.0, 34.0])

    def test_apply_legacy_fft_snapshot_to_mesh_state_can_skip_density_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(_write_legacy_fft_snapshot_example(Path(tmpdir)))

        mesh_state = _build_recovery_mesh_state(include_attr_f=False)
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(
            mesh_state,
            snapshot,
            import_options=LegacyFFTImportOptions(import_dislocation_densities=False),
        )

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_fields["U_DISLOCDEN"], dtype=np.float64),
            [10.0, 10.0, 10.0, 10.0],
        )
        self.assertEqual(mechanics_stats["import_dislocation_densities"], 0)
        self.assertEqual(mechanics_stats["density_imported_unodes"], 0)
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_import_dislocation_densities"], 0)

    def test_apply_legacy_fft_snapshot_to_mesh_state_excludes_phase_from_density_import(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.25, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        mesh_state = _build_mechanics_phase_mesh_state()
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(
            mesh_state,
            snapshot,
            import_options=LegacyFFTImportOptions(
                import_dislocation_densities=True,
                exclude_phase_id=2,
            ),
        )

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_fields["U_DISLOCDEN"], dtype=np.float64),
            [18.0, 28.0, 0.0, 0.0],
        )
        self.assertEqual(mechanics_stats["exclude_phase_id"], 2)
        self.assertEqual(mechanics_stats["density_imported_unodes"], 4)
        self.assertEqual(mechanics_stats["density_excluded_unodes"], 0)
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_exclude_phase_id"], 2)
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_density_excluded_unodes"], 0)

    def test_compare_applied_legacy_fft_snapshot_to_mesh_state_reports_full_contract_match(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(_write_legacy_fft_snapshot_example(Path(tmpdir)))

        before_mesh = _build_recovery_mesh_state(include_attr_f=False)
        after_mesh = copy.deepcopy(before_mesh)
        after_mesh, _mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(after_mesh, snapshot)

        report = compare_applied_legacy_fft_snapshot_to_mesh_state(
            before_mesh,
            after_mesh,
            snapshot,
        )

        self.assertEqual(report["euler_contract_match"], 1)
        self.assertEqual(report["position_contract_match"], 1)
        self.assertEqual(report["cell_reset_contract_match"], 1)
        self.assertEqual(report["tex_contract_match"], 1)
        self.assertEqual(report["density_contract_match"], 1)
        self.assertEqual(report["tracer_contract_match"], 1)
        self.assertEqual(report["runtime_snapshot_contract_match"], 1)
        self.assertEqual(report["mechanics_import_contract_match"], 1)

    def test_compare_applied_legacy_fft_snapshot_to_mesh_state_tracks_boundary_node_position_update(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    temp_rows=(
                        (1.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                        (0.0, 0.0, 0.0),
                    ),
                    unodexyz_rows=(
                        (0.35, 0.25, 0.0),
                        (0.35, 0.75, 0.0),
                        (0.85, 0.25, 0.0),
                        (0.85, 0.75, 0.0),
                    ),
                )
            )

        before_mesh = _build_mechanics_phase_mesh_state()
        after_mesh = copy.deepcopy(before_mesh)
        after_mesh, _mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(after_mesh, snapshot)

        report = compare_applied_legacy_fft_snapshot_to_mesh_state(
            before_mesh,
            after_mesh,
            snapshot,
        )

        self.assertEqual(report["node_position_xy_rmse"], 0.0)
        self.assertEqual(report["updated_node_positions_expected"], 6)
        self.assertEqual(report["updated_node_positions_actual"], 6)
        self.assertEqual(
            report["node_position_update_mode_expected"],
            "legacy_bnode_strain_weighted",
        )
        self.assertEqual(
            report["node_position_update_mode_actual"],
            "legacy_bnode_strain_weighted",
        )
        self.assertEqual(report["position_contract_match"], 1)
        self.assertEqual(report["mechanics_import_contract_match"], 1)

    def test_compare_applied_legacy_fft_snapshot_to_mesh_state_handles_density_exclusion(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.25, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        before_mesh = _build_mechanics_phase_mesh_state()
        after_mesh = copy.deepcopy(before_mesh)
        import_options = LegacyFFTImportOptions(
            import_dislocation_densities=True,
            exclude_phase_id=2,
        )
        after_mesh, _mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(
            after_mesh,
            snapshot,
            import_options=import_options,
        )

        report = compare_applied_legacy_fft_snapshot_to_mesh_state(
            before_mesh,
            after_mesh,
            snapshot,
            import_options=import_options,
        )

        self.assertEqual(report["density_imported_unodes_expected"], 4)
        self.assertEqual(report["density_excluded_unodes_expected"], 0)
        self.assertEqual(report["density_contract_match"], 1)
        self.assertEqual(report["mechanics_import_contract_match"], 1)

    def test_apply_legacy_fft_snapshot_to_mesh_state_updates_label_attribute_from_host_flynn(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.75, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        mesh_state = _build_mechanics_phase_mesh_state()
        mesh_state["_runtime_seed_unode_fields"] = {
            "label_attribute": "U_ATTRIB_C",
            "field_order": ("U_ATTRIB_C", "U_DISLOCDEN"),
            "source_labels": (100, 200),
            "values": {
                "U_ATTRIB_C": (100.0, 100.0, 200.0, 200.0),
                "U_DISLOCDEN": (10.0, 10.0, 10.0, 10.0),
            },
        }

        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(mesh_state, snapshot)

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_fields["U_ATTRIB_C"], dtype=np.float64),
            [100.0, 100.0, 200.0, 200.0],
        )
        self.assertEqual(mechanics_stats["label_update_applied"], 1)
        self.assertEqual(mechanics_stats["label_changed_unodes"], 0)
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_label_assignment_mode"], "mesh_host_flynn")

    def test_apply_legacy_fft_snapshot_to_mesh_state_initializes_legacy_tracer_fields_when_missing(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.75, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        mesh_state = _build_mechanics_phase_mesh_state()
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(mesh_state, snapshot)

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_fields["U_ATTRIB_C"], dtype=np.float64),
            [10.0, 10.0, 10.0, 10.0],
        )
        flynn_sections = updated_mesh["_runtime_seed_flynn_sections"]
        self.assertIn("F_ATTRIB_C", flynn_sections["values"])
        self.assertEqual(
            tuple(float(entry[0]) for entry in flynn_sections["values"]["F_ATTRIB_C"]),
            (10.0, 20.0),
        )
        self.assertEqual(mechanics_stats["unode_tracer_initialized"], 1)
        self.assertEqual(mechanics_stats["flynn_tracer_initialized"], 1)
        self.assertEqual(mechanics_stats["tracer_assignment_mode"], "mesh_host_flynn_ids")
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_unode_tracer_initialized"], 1)
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_flynn_tracer_initialized"], 1)

    def test_compare_applied_legacy_fft_snapshot_to_mesh_state_tracks_tracer_initialization(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.75, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        before_mesh = _build_mechanics_phase_mesh_state()
        after_mesh = copy.deepcopy(before_mesh)
        after_mesh, _mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(after_mesh, snapshot)

        report = compare_applied_legacy_fft_snapshot_to_mesh_state(
            before_mesh,
            after_mesh,
            snapshot,
        )

        self.assertEqual(report["unode_tracer_initialized_expected"], 1)
        self.assertEqual(report["unode_tracer_initialized_actual"], 1)
        self.assertEqual(report["flynn_tracer_initialized_expected"], 1)
        self.assertEqual(report["flynn_tracer_initialized_actual"], 1)
        self.assertEqual(report["tracer_assignment_mode_expected"], "mesh_host_flynn_ids")
        self.assertEqual(report["tracer_assignment_mode_actual"], "mesh_host_flynn_ids")
        self.assertEqual(report["unode_tracer_rmse"], 0.0)
        self.assertEqual(report["flynn_tracer_rmse"], 0.0)
        self.assertEqual(report["tracer_contract_match"], 1)
        self.assertEqual(report["mechanics_import_contract_match"], 1)

    def test_compare_applied_legacy_fft_snapshot_to_mesh_state_tracks_label_assignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.75, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        before_mesh = _build_mechanics_phase_mesh_state()
        before_mesh["_runtime_seed_unode_fields"] = {
            "label_attribute": "U_ATTRIB_C",
            "field_order": ("U_ATTRIB_C", "U_DISLOCDEN"),
            "source_labels": (100, 200),
            "values": {
                "U_ATTRIB_C": (100.0, 100.0, 200.0, 200.0),
                "U_DISLOCDEN": (10.0, 10.0, 10.0, 10.0),
            },
        }
        after_mesh = copy.deepcopy(before_mesh)
        after_mesh, _mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(after_mesh, snapshot)

        report = compare_applied_legacy_fft_snapshot_to_mesh_state(
            before_mesh,
            after_mesh,
            snapshot,
        )

        self.assertEqual(report["label_update_applied_expected"], 1)
        self.assertEqual(report["label_update_applied_actual"], 1)
        self.assertEqual(report["label_changed_unodes_expected"], 0)
        self.assertEqual(report["label_changed_unodes_actual"], 0)
        self.assertEqual(report["label_assignment_mode_expected"], "mesh_host_flynn")
        self.assertEqual(report["label_assignment_mode_actual"], "mesh_host_flynn")
        self.assertEqual(report["label_contract_match"], 1)
        self.assertEqual(report["mechanics_import_contract_match"], 1)

    def test_apply_legacy_fft_snapshot_to_mesh_state_reassigns_swept_unode_euler_and_density(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.75, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        mesh_state = _build_mechanics_phase_mesh_state()
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(mesh_state, snapshot)

        updated_sections = updated_mesh["_runtime_seed_unode_sections"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_sections["U_EULER_3"][0], dtype=np.float64),
            [10.0, 20.0, 30.0],
        )
        np.testing.assert_allclose(
            np.asarray(updated_mesh["_runtime_seed_unode_fields"]["values"]["U_DISLOCDEN"], dtype=np.float64),
            [18.0, 28.0, 0.0, 0.0],
        )
        self.assertEqual(mechanics_stats["swept_unodes"], 2)
        self.assertEqual(mechanics_stats["swept_reassignment_applied"], 1)
        self.assertEqual(mechanics_stats["swept_reassignment_mode"], "fs_check_unodes")
        self.assertEqual(mechanics_stats["updated_orientation_unodes"], 2)
        self.assertEqual(mechanics_stats["density_reset_unodes"], 2)
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_swept_unodes"], 2)
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_density_reset_unodes"], 2)

    def test_compare_applied_legacy_fft_snapshot_to_mesh_state_tracks_swept_reassignment(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.75, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        before_mesh = _build_mechanics_phase_mesh_state()
        after_mesh = copy.deepcopy(before_mesh)
        after_mesh, _mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(after_mesh, snapshot)

        report = compare_applied_legacy_fft_snapshot_to_mesh_state(
            before_mesh,
            after_mesh,
            snapshot,
        )

        self.assertEqual(report["swept_unodes_expected"], 2)
        self.assertEqual(report["swept_unodes_actual"], 2)
        self.assertEqual(report["swept_reassignment_applied_expected"], 1)
        self.assertEqual(report["swept_reassignment_applied_actual"], 1)
        self.assertEqual(report["swept_reassignment_mode_expected"], "fs_check_unodes")
        self.assertEqual(report["swept_reassignment_mode_actual"], "fs_check_unodes")
        self.assertEqual(report["updated_orientation_unodes_expected"], 2)
        self.assertEqual(report["updated_orientation_unodes_actual"], 2)
        self.assertEqual(report["density_reset_unodes_expected"], 2)
        self.assertEqual(report["density_reset_unodes_actual"], 2)
        self.assertEqual(report["swept_reassignment_contract_match"], 1)
        self.assertEqual(report["mechanics_import_contract_match"], 1)

    def test_apply_legacy_fft_snapshot_to_mesh_state_can_overwrite_density_field(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(_write_legacy_fft_snapshot_example(Path(tmpdir)))

        mesh_state = _build_recovery_mesh_state(include_attr_f=False)
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(
            mesh_state,
            snapshot,
            import_options=LegacyFFTImportOptions(
                import_dislocation_densities=True,
                density_update_mode="overwrite",
            ),
        )

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_fields["U_DISLOCDEN"], dtype=np.float64),
            [8.0, 18.0, 28.0, 38.0],
        )
        self.assertEqual(mechanics_stats["density_update_mode"], "overwrite")
        self.assertEqual(updated_mesh["stats"]["mechanics_snapshot_density_update_mode"], "overwrite")

    def test_compare_applied_legacy_fft_snapshot_to_mesh_state_tracks_density_update_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(_write_legacy_fft_snapshot_example(Path(tmpdir)))

        before_mesh = _build_recovery_mesh_state(include_attr_f=False)
        after_mesh = copy.deepcopy(before_mesh)
        import_options = LegacyFFTImportOptions(
            import_dislocation_densities=True,
            density_update_mode="overwrite",
        )
        after_mesh, _mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(
            after_mesh,
            snapshot,
            import_options=import_options,
        )

        report = compare_applied_legacy_fft_snapshot_to_mesh_state(
            before_mesh,
            after_mesh,
            snapshot,
            import_options=import_options,
        )

        self.assertEqual(report["density_update_mode_expected"], "overwrite")
        self.assertEqual(report["density_update_mode_actual"], "overwrite")
        self.assertEqual(report["density_contract_match"], 1)
        self.assertEqual(report["mechanics_import_contract_match"], 1)

    def test_apply_legacy_fft_snapshot_to_mesh_state_supports_check_error_host_repair_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.25, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        mesh_state = _build_mechanics_phase_mesh_state()
        mesh_state["_runtime_seed_unode_fields"] = {
            "field_order": ("U_ATTRIB_C", "U_DISLOCDEN"),
            "values": {
                "U_ATTRIB_C": (20.0, 10.0, 20.0, 20.0),
                "U_DISLOCDEN": (10.0, 10.0, 10.0, 10.0),
            },
        }
        updated_mesh, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(
            mesh_state,
            snapshot,
            import_options=LegacyFFTImportOptions(host_repair_mode="check_error"),
        )

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        updated_sections = updated_mesh["_runtime_seed_unode_sections"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_fields["U_ATTRIB_C"], dtype=np.float64),
            [10.0, 10.0, 10.0, 10.0],
        )
        np.testing.assert_allclose(
            np.asarray(updated_sections["U_EULER_3"][0], dtype=np.float64),
            [20.0, 40.0, 60.0],
        )
        np.testing.assert_allclose(
            np.asarray(updated_sections["U_EULER_3"][2], dtype=np.float64),
            [20.0, 40.0, 60.0],
        )
        np.testing.assert_allclose(
            np.asarray(updated_fields["U_DISLOCDEN"], dtype=np.float64),
            [18.0, 28.0, 38.0, 48.0],
        )
        self.assertEqual(mechanics_stats["host_repair_mode"], "check_error")
        self.assertEqual(mechanics_stats["swept_reassignment_mode"], "check_error")
        self.assertEqual(mechanics_stats["swept_unodes"], 3)
        self.assertEqual(mechanics_stats["updated_orientation_unodes"], 3)
        self.assertEqual(mechanics_stats["density_reset_unodes"], 0)

    def test_compare_applied_legacy_fft_snapshot_to_mesh_state_tracks_check_error_host_repair_mode(
        self,
    ) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            snapshot = load_legacy_fft_snapshot(
                _write_legacy_fft_snapshot_example(
                    Path(tmpdir),
                    unodexyz_rows=(
                        (0.25, 0.25, 0.0),
                        (0.25, 0.75, 0.0),
                        (0.75, 0.25, 0.0),
                        (0.75, 0.75, 0.0),
                    ),
                )
            )

        before_mesh = _build_mechanics_phase_mesh_state()
        before_mesh["_runtime_seed_unode_fields"] = {
            "field_order": ("U_ATTRIB_C", "U_DISLOCDEN"),
            "values": {
                "U_ATTRIB_C": (20.0, 10.0, 20.0, 20.0),
                "U_DISLOCDEN": (10.0, 10.0, 10.0, 10.0),
            },
        }
        after_mesh = copy.deepcopy(before_mesh)
        import_options = LegacyFFTImportOptions(host_repair_mode="check_error")
        after_mesh, _mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(
            after_mesh,
            snapshot,
            import_options=import_options,
        )

        report = compare_applied_legacy_fft_snapshot_to_mesh_state(
            before_mesh,
            after_mesh,
            snapshot,
            import_options=import_options,
        )

        self.assertEqual(report["host_repair_mode_expected"], "check_error")
        self.assertEqual(report["host_repair_mode_actual"], "check_error")
        self.assertEqual(report["swept_reassignment_mode_expected"], "check_error")
        self.assertEqual(report["swept_reassignment_mode_actual"], "check_error")
        self.assertEqual(report["swept_unodes_expected"], 3)
        self.assertEqual(report["swept_unodes_actual"], 3)
        self.assertEqual(report["density_reset_unodes_expected"], 0)
        self.assertEqual(report["density_reset_unodes_actual"], 0)
        self.assertEqual(report["swept_reassignment_contract_match"], 1)
        self.assertEqual(report["mechanics_import_contract_match"], 1)

    def test_run_faithful_gbm_simulation_can_apply_mechanics_snapshot_before_outer_step(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_snapshot")
            setup = build_faithful_gbm_setup(
                elle_path,
                mechanics_snapshot_dir=root / "fft_snapshot",
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=1,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=0,
                mechanics_exclude_phase_id=0,
            )
            observed_contexts: list[dict[str, object]] = []

            run_faithful_gbm_simulation(
                setup=setup,
                steps=1,
                save_every=1,
                on_snapshot=lambda _step, _phi, _topology=None, context=None: observed_contexts.append(context),
            )

        self.assertEqual(len(observed_contexts), 1)
        context = observed_contexts[0]
        self.assertEqual(context["mechanics_snapshot_enabled"], 1)
        self.assertIsNotNone(context["mechanics_stats"])
        self.assertEqual(context["mechanics_stats"]["mechanics_applied"], 1)
        mesh_state = context["mesh_state"]
        self.assertEqual(mesh_state["stats"]["workflow_mechanics_applied_total"], 1)
        self.assertEqual(mesh_state["stats"]["workflow_mechanics_steps_per_snapshot"], 1)
        updated_fields = mesh_state["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(np.asarray(updated_fields["U_ATTRIB_A"], dtype=np.float64), [4.0, 14.0, 24.0, 34.0])

    def test_run_faithful_gbm_simulation_threads_mechanics_density_import_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_snapshot")
            setup = build_faithful_gbm_setup(
                elle_path,
                mechanics_snapshot_dir=root / "fft_snapshot",
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=1,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=0,
                mechanics_import_dislocation_densities=False,
            )
            observed_contexts: list[dict[str, object]] = []

            run_faithful_gbm_simulation(
                setup=setup,
                steps=1,
                save_every=1,
                on_snapshot=lambda _step, _phi, _topology=None, context=None: observed_contexts.append(context),
            )

        context = observed_contexts[0]
        self.assertEqual(context["mechanics_stats"]["import_dislocation_densities"], 0)
        updated_fields = context["mesh_state"]["_runtime_seed_unode_fields"]["values"]
        self.assertNotIn("U_DISLOCDEN", updated_fields)

    def test_run_faithful_gbm_simulation_threads_mechanics_density_update_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_snapshot")
            setup = build_faithful_gbm_setup(
                elle_path,
                mechanics_snapshot_dir=root / "fft_snapshot",
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=1,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=0,
                mechanics_density_update_mode="overwrite",
            )
            observed_contexts: list[dict[str, object]] = []

            run_faithful_gbm_simulation(
                setup=setup,
                steps=1,
                save_every=1,
                on_snapshot=lambda _step, _phi, _topology=None, context=None: observed_contexts.append(context),
            )

        context = observed_contexts[0]
        self.assertEqual(context["mechanics_stats"]["density_update_mode"], "overwrite")
        self.assertEqual(context["mechanics_stats"]["swept_reassignment_mode"], "fs_check_unodes")
        self.assertEqual(context["mechanics_stats"]["density_reset_unodes"], 2)
        updated_fields = context["mesh_state"]["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(
            np.asarray(updated_fields["U_DISLOCDEN"], dtype=np.float64),
            [8.0, 18.0, 0.0, 0.0],
        )

    def test_run_faithful_gbm_simulation_threads_mechanics_host_repair_mode(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_snapshot")
            setup = build_faithful_gbm_setup(
                elle_path,
                mechanics_snapshot_dir=root / "fft_snapshot",
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=1,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=0,
                mechanics_host_repair_mode="check_error",
            )
            observed_contexts: list[dict[str, object]] = []

            run_faithful_gbm_simulation(
                setup=setup,
                steps=1,
                save_every=1,
                on_snapshot=lambda _step, _phi, _topology=None, context=None: observed_contexts.append(context),
            )

        context = observed_contexts[0]
        self.assertEqual(context["mechanics_stats"]["host_repair_mode"], "check_error")
        self.assertEqual(
            context["mesh_state"]["stats"]["mechanics_snapshot_host_repair_mode"],
            "check_error",
        )

    def test_run_faithful_gbm_simulation_can_apply_mechanics_snapshot_sequence_across_outer_steps(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_sequence" / "step001", value_offset=0.0)
            _write_legacy_fft_snapshot_example(root / "fft_sequence" / "step002", value_offset=100.0)
            setup = build_faithful_gbm_setup(
                elle_path,
                mechanics_snapshot_dir=root / "fft_sequence",
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=1,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=0,
            )
            observed_contexts: list[dict[str, object]] = []

            run_faithful_gbm_simulation(
                setup=setup,
                steps=2,
                save_every=1,
                on_snapshot=lambda _step, _phi, _topology=None, context=None: observed_contexts.append(context),
            )

        self.assertEqual(len(setup.mechanics_snapshots), 2)
        self.assertEqual(len(observed_contexts), 2)
        first_context = observed_contexts[0]
        second_context = observed_contexts[1]
        self.assertEqual(first_context["mechanics_stats"]["snapshot_index"], 1)
        self.assertEqual(second_context["mechanics_stats"]["snapshot_index"], 2)
        self.assertEqual(second_context["mechanics_snapshot_count"], 2)
        first_fields = first_context["mesh_state"]["_runtime_seed_unode_fields"]["values"]
        second_fields = second_context["mesh_state"]["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(
            np.asarray(first_fields["U_ATTRIB_A"], dtype=np.float64),
            [4.0, 14.0, 24.0, 34.0],
        )
        np.testing.assert_allclose(
            np.asarray(second_fields["U_ATTRIB_A"], dtype=np.float64),
            [104.0, 114.0, 124.0, 134.0],
        )

    def test_run_faithful_gbm_simulation_supports_mechanics_only_snapshot_sequence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_sequence" / "step001", value_offset=0.0)
            _write_legacy_fft_snapshot_example(root / "fft_sequence" / "step002", value_offset=100.0)
            setup = build_faithful_gbm_setup(
                elle_path,
                mechanics_snapshot_dir=root / "fft_sequence",
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=0,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=0,
            )
            observed_contexts: list[dict[str, object]] = []

            run_faithful_gbm_simulation(
                setup=setup,
                steps=2,
                save_every=1,
                on_snapshot=lambda _step, _phi, _topology=None, context=None: observed_contexts.append(context),
            )

        self.assertEqual(len(observed_contexts), 2)
        self.assertEqual(observed_contexts[0]["stage_kind"], "mechanics")
        self.assertEqual(observed_contexts[1]["stage_kind"], "mechanics")
        self.assertEqual(observed_contexts[1]["stages_per_snapshot"], 1)
        self.assertEqual(observed_contexts[1]["mechanics_snapshot_index"], 2)
        second_fields = observed_contexts[1]["mesh_state"]["_runtime_seed_unode_fields"]["values"]
        np.testing.assert_allclose(
            np.asarray(second_fields["U_ATTRIB_A"], dtype=np.float64),
            [104.0, 114.0, 124.0, 134.0],
        )

    def test_run_faithful_mechanics_replay_writes_step0_and_step1_elle_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_sequence" / "step001", value_offset=0.0)
            report = run_faithful_mechanics_replay(
                elle_path,
                root / "fft_sequence",
                outdir=root / "replay",
            )

        self.assertEqual(report["steps"], 1)
        self.assertEqual(report["mechanics_snapshot_count"], 1)
        self.assertIn("0", report["checkpoint_paths"])
        self.assertIn("1", report["checkpoint_paths"])
        self.assertTrue(report["checkpoint_paths"]["0"]["elle"].endswith("grain_unodes_00000.elle"))
        self.assertTrue(report["checkpoint_paths"]["1"]["elle"].endswith("grain_unodes_00001.elle"))
        self.assertTrue(report["topology_history_path"].endswith("topology_history.json"))

    def test_validate_faithful_mechanics_transition_matches_reference_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_sequence" / "step001", value_offset=0.0)
            reference_replay = run_faithful_mechanics_replay(
                elle_path,
                root / "fft_sequence",
                outdir=root / "reference_replay",
            )
            report = validate_faithful_mechanics_transition(
                elle_path,
                root / "fft_sequence",
                reference_replay["checkpoint_paths"]["0"]["elle"],
                reference_replay["checkpoint_paths"]["1"]["elle"],
                outdir=root / "candidate_replay",
            )

        self.assertTrue(report["matches"])
        self.assertEqual(report["mismatched_field_transitions"], [])
        self.assertEqual(report["missing_field_transitions"], [])
        self.assertEqual(report["unexpected_field_transitions"], [])

    def test_validate_faithful_mechanics_outerstep_transition_matches_reference_pair(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            _write_legacy_fft_snapshot_example(root / "fft_sequence" / "step001", value_offset=0.0)

            setup = build_faithful_gbm_setup(
                elle_path,
                mechanics_snapshot_dir=root / "fft_sequence",
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=0,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=1,
                recovery_trial_rotation_deg=0.1,
                recovery_rotation_mobility_length=50.0,
            )
            reference_paths: dict[str, dict[str, str]] = {}

            def _save_reference(step: int, phi, topology_snapshot, mesh_feedback_context=None) -> None:
                mesh_state = None if mesh_feedback_context is None else mesh_feedback_context.get("mesh_state")
                written = save_snapshot_artifacts(
                    root / "reference_outerstep",
                    int(step),
                    phi,
                    save_preview=False,
                    save_elle=True,
                    tracked_topology=topology_snapshot,
                    save_topology=True,
                    mesh_state=mesh_state,
                    save_mesh=True,
                )
                reference_paths[str(int(step))] = {
                    str(name): str(path) for name, path in written.items()
                }

            run_faithful_gbm_simulation(
                setup=setup,
                steps=1,
                save_every=1,
                on_snapshot=_save_reference,
                include_initial_snapshot=True,
            )

            report = validate_faithful_mechanics_outerstep_transition(
                elle_path,
                root / "fft_sequence",
                reference_paths["0"]["elle"],
                reference_paths["1"]["elle"],
                outdir=root / "candidate_outerstep",
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=0,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=1,
                recovery_trial_rotation_deg=0.1,
                recovery_rotation_mobility_length=50.0,
            )

        self.assertTrue(report["matches"])
        self.assertEqual(report["mismatched_field_transitions"], [])
        self.assertEqual(report["missing_field_transitions"], [])
        self.assertEqual(report["unexpected_field_transitions"], [])

    def test_run_faithful_gbm_simulation_rejects_empty_outer_step_without_mechanics(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            elle_path = _write_elle_mesh_seed_example(root / "seed.elle")
            setup = build_faithful_gbm_setup(
                elle_path,
                motion_passes=0,
                topology_passes=0,
                subloops_per_snapshot=1,
                gbm_steps_per_subloop=0,
                nucleation_steps_per_subloop=0,
                recovery_steps_per_subloop=0,
            )

            with self.assertRaisesRegex(ValueError, "at least one GBM/nucleation/recovery stage per subloop or a mechanics snapshot"):
                run_faithful_gbm_simulation(
                    setup=setup,
                    steps=1,
                    save_every=1,
                )

    def test_apply_recovery_stage_adds_u_attrib_f_without_reducing_density_on_first_stage(self) -> None:
        mesh_state = _build_recovery_mesh_state(include_attr_f=False)
        labels = np.zeros((2, 2), dtype=np.int32)

        updated_mesh, recovery_stats = apply_recovery_stage(
            mesh_state,
            labels,
            RecoveryConfig(high_angle_boundary_deg=5.0, trial_rotation_deg=0.1, rotation_mobility_length=50.0),
            recovery_stage_index=0,
        )

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        self.assertIn("U_ATTRIB_F", updated_fields)
        self.assertGreater(float(updated_fields["U_ATTRIB_F"][0]), 0.0)
        self.assertEqual(tuple(updated_fields["U_DISLOCDEN"]), (10.0, 10.0, 10.0, 10.0))
        self.assertEqual(recovery_stats["density_reduced_unodes"], 0)
        self.assertGreaterEqual(recovery_stats["rotated_unodes"], 1)

    def test_symmetry_aware_recovery_misorientation_collapses_hex_equivalent_rotation(self) -> None:
        symmetry_operators = recovery_module._resolve_recovery_symmetry_operators(RecoveryConfig())
        misorientation = recovery_module._symmetry_misorientation_deg(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            np.asarray([[60.0, 0.0, 300.0]], dtype=np.float64),
            symmetry_operators,
        )

        self.assertLess(float(misorientation[0]), 1.0e-3)

    def test_legacy_recovery_trial_rotation_matches_old_trial_matrix_for_identity_orientation(self) -> None:
        rotated = recovery_module._apply_legacy_recovery_trial_rotation(
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
            trial_index=0,
            theta_deg=0.1,
        )
        expected_euler = recovery_module._euler_deg_from_orientation_matrices(
            recovery_module._legacy_recovery_trial_matrices(0, 0.1)[None, ...],
            np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
        )
        symmetry_operators = recovery_module._resolve_recovery_symmetry_operators(RecoveryConfig())
        misorientation = recovery_module._symmetry_misorientation_deg(
            rotated,
            expected_euler,
            symmetry_operators,
        )

        self.assertLess(float(misorientation[0]), 1.0e-6)

    def test_recovery_rotation_cap_matches_legacy_limits(self) -> None:
        capped = recovery_module._recovery_rotation_cap_deg(
            np.asarray([0.03, 4.5], dtype=np.float64),
            theta_deg=0.1,
        )

        np.testing.assert_allclose(capped, np.asarray([0.03, 2.0], dtype=np.float64))

    def test_legacy_recovery_site_response_matches_expected_tiny_fixture_pattern(self) -> None:
        mesh_state = _build_recovery_mesh_state(include_attr_f=True, attr_f_value=2.0)
        labels = np.zeros((2, 2), dtype=np.int32)
        config = RecoveryConfig(high_angle_boundary_deg=5.0, trial_rotation_deg=0.1, rotation_mobility_length=50.0)

        grid_indices = np.asarray(mesh_state["_runtime_seed_unodes"]["grid_indices"], dtype=np.int32)
        sample_labels = labels[grid_indices[:, 0], grid_indices[:, 1]]
        neighbour_matrix, neighbour_mask, _counts = recovery_module._recovery_neighbour_context(
            mesh_state,
            grid_indices,
            sample_labels,
            (2, 2),
        )
        euler_values = np.asarray(
            mesh_state["_runtime_seed_unode_sections"]["values"]["U_EULER_3"],
            dtype=np.float64,
        )
        density_values = np.asarray(
            mesh_state["_runtime_seed_unode_fields"]["values"]["U_DISLOCDEN"],
            dtype=np.float64,
        )
        attr_f_values = np.asarray(
            mesh_state["_runtime_seed_unode_fields"]["values"]["U_ATTRIB_F"],
            dtype=np.float64,
        )
        symmetry_operators = recovery_module._resolve_recovery_symmetry_operators(config)

        response = recovery_module._legacy_recovery_site_response(
            euler_values[0],
            euler_values[neighbour_matrix[0][neighbour_mask[0]]],
            config,
            recovery_stage_index=1,
            previous_attr_f=float(attr_f_values[0]),
            current_density=float(density_values[0]),
            symmetry_operators=symmetry_operators,
        )

        self.assertEqual(response["accepted_trials"], [3])
        self.assertTrue(response["trial_history"][3]["accepted"])
        self.assertLess(
            float(response["trial_history"][3]["trial_energy"]),
            float(response["trial_history"][2]["trial_energy"]),
        )
        np.testing.assert_allclose(
            np.asarray(response["final_euler_deg"], dtype=np.float64),
            np.asarray([0.0, 0.0, -1.9219682942568618], dtype=np.float64),
            atol=1.0e-9,
        )
        self.assertAlmostEqual(float(response["final_avg_misorientation_deg"]), 1.9219682942568619)
        self.assertAlmostEqual(float(response["final_density"]), 9.609841471284309)

    def test_apply_recovery_stage_matches_sequential_legacy_site_responses(self) -> None:
        mesh_state = _build_recovery_mesh_state(include_attr_f=True, attr_f_value=2.0)
        labels = np.zeros((2, 2), dtype=np.int32)
        config = RecoveryConfig(high_angle_boundary_deg=5.0, trial_rotation_deg=0.1, rotation_mobility_length=50.0)

        grid_indices = np.asarray(mesh_state["_runtime_seed_unodes"]["grid_indices"], dtype=np.int32)
        sample_labels = labels[grid_indices[:, 0], grid_indices[:, 1]]
        neighbour_matrix, neighbour_mask, _counts = recovery_module._recovery_neighbour_context(
            mesh_state,
            grid_indices,
            sample_labels,
            (2, 2),
        )
        expected_euler = np.asarray(
            mesh_state["_runtime_seed_unode_sections"]["values"]["U_EULER_3"],
            dtype=np.float64,
        ).copy()
        expected_density = np.asarray(
            mesh_state["_runtime_seed_unode_fields"]["values"]["U_DISLOCDEN"],
            dtype=np.float64,
        ).copy()
        previous_attr_f = np.asarray(
            mesh_state["_runtime_seed_unode_fields"]["values"]["U_ATTRIB_F"],
            dtype=np.float64,
        ).copy()
        expected_attr_f = previous_attr_f.copy()
        expected_rotated = 0
        symmetry_operators = recovery_module._resolve_recovery_symmetry_operators(config)

        for index in range(expected_euler.shape[0]):
            response = recovery_module._legacy_recovery_site_response(
                expected_euler[index],
                expected_euler[neighbour_matrix[index][neighbour_mask[index]]],
                config,
                recovery_stage_index=1,
                previous_attr_f=float(previous_attr_f[index]),
                current_density=float(expected_density[index]),
                symmetry_operators=symmetry_operators,
            )
            expected_euler[index] = np.asarray(response["final_euler_deg"], dtype=np.float64)
            expected_attr_f[index] = float(response["final_avg_misorientation_deg"])
            expected_density[index] = float(response["final_density"])
            expected_rotated += int(bool(response["accepted_trials"]))

        updated_mesh, recovery_stats = apply_recovery_stage(
            _build_recovery_mesh_state(include_attr_f=True, attr_f_value=2.0),
            labels,
            config,
            recovery_stage_index=1,
        )

        updated_sections = np.asarray(
            updated_mesh["_runtime_seed_unode_sections"]["values"]["U_EULER_3"],
            dtype=np.float64,
        )
        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        updated_attr_f = np.asarray(updated_fields["U_ATTRIB_F"], dtype=np.float64)
        updated_density = np.asarray(updated_fields["U_DISLOCDEN"], dtype=np.float64)

        np.testing.assert_allclose(updated_sections, expected_euler, atol=1.0e-9)
        np.testing.assert_allclose(updated_attr_f, expected_attr_f, atol=1.0e-9)
        np.testing.assert_allclose(updated_density, expected_density, atol=1.0e-9)
        self.assertEqual(recovery_stats["rotated_unodes"], expected_rotated)

    def test_recovery_grid_neighbours_keep_only_six_nearest_same_label_points(self) -> None:
        grid_indices = np.asarray(
            [
                (ix, iy)
                for ix in range(3)
                for iy in range(3)
            ],
            dtype=np.int32,
        )
        label_values = np.zeros((9,), dtype=np.int32)

        neighbours = recovery_module._grid_neighbour_indices(
            grid_indices,
            label_values,
            4,
            (3, 3),
        )

        self.assertEqual(len(neighbours), 6)
        direct_neighbours = {1, 3, 5, 7}
        self.assertTrue(direct_neighbours.issubset(set(int(value) for value in neighbours)))

    def test_apply_recovery_stage_treats_hex_symmetry_equivalent_orientation_as_low_misorientation(self) -> None:
        mesh_state = _build_recovery_mesh_state(
            include_attr_f=False,
            euler_values=(
                (60.0, 0.0, 300.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
        labels = np.zeros((2, 2), dtype=np.int32)

        updated_mesh, recovery_stats = apply_recovery_stage(
            mesh_state,
            labels,
            RecoveryConfig(high_angle_boundary_deg=5.0, trial_rotation_deg=0.1, rotation_mobility_length=50.0),
            recovery_stage_index=0,
        )

        updated_fields = updated_mesh["_runtime_seed_unode_fields"]["values"]
        updated_attr_f = np.asarray(updated_fields["U_ATTRIB_F"], dtype=np.float64)
        self.assertTrue(np.all(updated_attr_f < 1.0e-3))
        self.assertEqual(recovery_stats["rotated_unodes"], 0)

    def test_apply_recovery_stage_reduces_u_dislocden_after_first_stage(self) -> None:
        mesh_state = _build_recovery_mesh_state(include_attr_f=True, attr_f_value=2.0)
        labels = np.zeros((2, 2), dtype=np.int32)

        updated_mesh, recovery_stats = apply_recovery_stage(
            mesh_state,
            labels,
            RecoveryConfig(high_angle_boundary_deg=5.0, trial_rotation_deg=0.1, rotation_mobility_length=50.0),
            recovery_stage_index=1,
        )

        updated_density = np.asarray(updated_mesh["_runtime_seed_unode_fields"]["values"]["U_DISLOCDEN"])
        self.assertTrue(np.any(updated_density < 10.0))
        self.assertGreaterEqual(recovery_stats["density_reduced_unodes"], 1)

    def test_apply_recovery_stage_updates_flynn_level_orientation_and_density_sections(self) -> None:
        mesh_state = _build_recovery_mesh_state(include_attr_f=True, attr_f_value=2.0)
        labels = np.zeros((2, 2), dtype=np.int32)

        updated_mesh, _recovery_stats = apply_recovery_stage(
            mesh_state,
            labels,
            RecoveryConfig(high_angle_boundary_deg=5.0, trial_rotation_deg=0.1, rotation_mobility_length=50.0),
            recovery_stage_index=1,
        )

        flynn_values = updated_mesh["_runtime_seed_flynn_sections"]["values"]
        updated_flynn_euler = np.asarray(flynn_values["EULER_3"], dtype=np.float64)
        updated_flynn_density = np.asarray(flynn_values["DISLOCDEN"], dtype=np.float64)
        self.assertEqual(updated_flynn_euler.shape, (1, 3))
        self.assertGreater(float(np.linalg.norm(updated_flynn_euler[0])), 0.0)
        self.assertLess(float(updated_flynn_density[0]), 10.0)

    def test_apply_nucleation_stage_promotes_large_secondary_subgrain_cluster(self) -> None:
        current_labels = np.zeros((4, 4), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        for ix in range(4):
            for iy in range(4):
                if ix >= 2 and iy >= 2:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(0.0 for _ in range(16)),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "_runtime_seed_flynn_sections": {
                "field_order": ("F_ATTRIB_C",),
                "id_order": (11,),
                "defaults": {"F_ATTRIB_C": (7.0,)},
                "component_counts": {"F_ATTRIB_C": 1},
                "values": {"F_ATTRIB_C": ((7.0,),)},
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 1.0, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 1.0, "y": 1.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11},
            ],
            "stats": {},
        }

        updated_labels, updated_mesh, stats = apply_nucleation_stage(
            mesh_state,
            current_labels,
            NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4),
        )

        self.assertEqual(stats["nucleated_clusters"], 1)
        self.assertEqual(stats["nucleated_unodes"], 4)
        self.assertEqual(stats["new_labels_added"], 1)
        self.assertEqual(int(np.max(updated_labels)), 1)
        self.assertEqual(int(np.count_nonzero(updated_labels == 1)), 4)
        self.assertNotIn("_runtime_label_overrides", updated_mesh)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_mesh_rebuilt"]), 1)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_incremental_label_handoff"]), 1)
        self.assertEqual(int(updated_mesh["_runtime_incremental_label_remap_stages"]), 1)
        self.assertEqual(len(updated_mesh["flynns"]), 2)
        self.assertTrue(any(not bool(flynn.get("retained_identity", True)) for flynn in updated_mesh["flynns"]))
        updated_label_values = np.asarray(
            updated_mesh["_runtime_seed_unode_fields"]["values"]["U_ATTRIB_C"],
            dtype=np.float64,
        ).reshape(4, 4)
        expected_source_labels = np.where(updated_labels == 1, 8.0, 7.0)
        np.testing.assert_array_equal(updated_label_values, expected_source_labels)
        self.assertEqual(updated_mesh["_runtime_seed_unode_fields"]["source_labels"], (7, 8))
        current_flynn_sections = updated_mesh["_runtime_current_flynn_sections"]
        self.assertEqual(tuple(current_flynn_sections["id_order"]), tuple(
            int(flynn["flynn_id"]) for flynn in updated_mesh["flynns"]
        ))
        flynn_label_values = tuple(
            float(entry[0]) for entry in current_flynn_sections["values"]["F_ATTRIB_C"]
        )
        self.assertIn(7.0, flynn_label_values)
        self.assertIn(8.0, flynn_label_values)

    def test_apply_nucleation_stage_sets_incremental_handoff_to_gbm_steps_per_subloop(self) -> None:
        current_labels = np.zeros((4, 4), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        for ix in range(4):
            for iy in range(4):
                if ix >= 2 and iy >= 2:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(0.0 for _ in range(16)),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 1.0, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 1.0, "y": 1.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11},
            ],
            "stats": {
                "workflow_gbm_steps_per_subloop": 2,
            },
        }

        _updated_labels, updated_mesh, _stats = apply_nucleation_stage(
            mesh_state,
            current_labels,
            NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4),
        )

        self.assertEqual(int(updated_mesh["_runtime_incremental_label_remap_stages"]), 2)

    def test_rebuild_mesh_state_from_nucleated_labels_preserves_existing_compact_labels(self) -> None:
        current_labels = np.zeros((4, 4), dtype=np.int32)
        updated_labels = np.zeros((4, 4), dtype=np.int32)
        updated_labels[0, 0] = 1
        updated_labels[3, 3] = 1
        mesh_state = build_mesh_state(current_labels)
        mesh_state["_runtime_seed_unodes"] = {
            "ids": tuple(range(16)),
            "positions": tuple(
                (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                for ix in range(4)
                for iy in range(4)
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            "grid_shape": (4, 4),
        }
        mesh_state["_runtime_seed_flynn_sections"] = {
            "field_order": ("F_ATTRIB_C",),
            "id_order": tuple(int(flynn["flynn_id"]) for flynn in mesh_state["flynns"]),
            "defaults": {"F_ATTRIB_C": (7.0,)},
            "component_counts": {"F_ATTRIB_C": 1},
            "values": {"F_ATTRIB_C": tuple((7.0,) for _ in mesh_state["flynns"])},
        }
        for flynn in mesh_state["flynns"]:
            flynn["source_flynn_id"] = int(flynn["flynn_id"])

        rebuilt = nucleation_module._rebuild_mesh_state_from_nucleated_labels(
            mesh_state,
            updated_labels,
            (7, 8),
        )

        rebuilt_labels = [int(flynn["label"]) for flynn in rebuilt["flynns"]]
        self.assertEqual(sorted(rebuilt_labels), [0, 1, 1])
        self.assertEqual(tuple(rebuilt["_runtime_rebuilt_source_labels"]), (7, 8))
        self.assertEqual(
            np.asarray(rebuilt["_runtime_relabelled_labels"], dtype=np.int32).tolist(),
            updated_labels.tolist(),
        )
        self.assertEqual(
            len(tuple(rebuilt["_runtime_current_flynn_sections"]["id_order"])),
            len(rebuilt["flynns"]),
        )

    def test_label_boundary_seed_mask_from_mesh_uses_boundary_node_proximity(self) -> None:
        mesh_state = build_mesh_state(np.zeros((4, 4), dtype=np.int32))
        label_mask = np.ones((4, 4), dtype=bool)
        seed_position_grid = np.array(
            [
                [[0.05, 0.95], [0.05, 0.65], [0.05, 0.35], [0.05, 0.05]],
                [[0.35, 0.95], [0.35, 0.65], [0.35, 0.35], [0.35, 0.05]],
                [[0.65, 0.95], [0.65, 0.65], [0.65, 0.35], [0.65, 0.05]],
                [[0.95, 0.95], [0.95, 0.65], [0.95, 0.35], [0.95, 0.05]],
            ],
            dtype=np.float64,
        )

        boundary_seed_mask = nucleation_module._label_boundary_seed_mask_from_mesh(
            mesh_state,
            0,
            label_mask,
            seed_position_grid,
            max_distance=0.1,
        )

        self.assertTrue(bool(boundary_seed_mask[0, 0]))
        self.assertTrue(bool(boundary_seed_mask[3, 0]))
        self.assertFalse(bool(boundary_seed_mask[1, 0]))
        self.assertFalse(bool(boundary_seed_mask[2, 0]))

    def test_apply_nucleation_stage_skips_already_nucleated_source_flynn(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        euler_values: list[tuple[float, float, float]] = []
        for ix in range(4):
            for iy in range(4):
                if ix < 2:
                    euler_values.append((0.0, 0.0, 0.0))
                elif iy >= 2:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7, 8),
                "values": {
                    "U_ATTRIB_C": tuple(float(value) for value in current_labels.ravel()),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "_runtime_seed_flynn_sections": {
                "field_order": ("F_ATTRIB_C",),
                "id_order": (11, 12),
                "defaults": {"F_ATTRIB_C": (7.0,)},
                "component_counts": {"F_ATTRIB_C": 1},
                "values": {"F_ATTRIB_C": ((7.0,), (8.0,))},
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 0.5, "y": 0.0, "neighbors": [0, 2], "flynns": [11, 12]},
                {"node_id": 2, "x": 0.5, "y": 1.0, "neighbors": [1, 3], "flynns": [11, 12]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 4, "x": 1.0, "y": 0.0, "neighbors": [1, 5], "flynns": [12]},
                {"node_id": 5, "x": 1.0, "y": 1.0, "neighbors": [4, 2], "flynns": [12]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11, "retained_identity": True},
                {"flynn_id": 12, "label": 1, "node_ids": [1, 4, 5, 2], "source_flynn_id": 12, "retained_identity": False},
            ],
            "_runtime_nucleated_source_flynns": (11,),
            "stats": {},
        }

        updated_labels, updated_mesh, stats = apply_nucleation_stage(
            mesh_state,
            current_labels,
            NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=2),
        )

        self.assertEqual(stats["nucleated_clusters"], 0)
        self.assertGreaterEqual(stats["skipped_repeated_sources"], 1)
        np.testing.assert_array_equal(updated_labels, current_labels)
        self.assertIs(updated_mesh, mesh_state)

    def test_apply_nucleation_stage_rejects_pathological_mesh_rebuild(self) -> None:
        current_labels = np.zeros((4, 4), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        for ix in range(4):
            for iy in range(4):
                if ix >= 2 and iy >= 2:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(0.0 for _ in range(16)),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 1.0, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 1.0, "y": 1.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11, "retained_identity": True},
            ],
            "stats": {},
        }

        fake_rebuilt = {
            "nodes": [],
            "flynns": [{"flynn_id": i, "label": i, "node_ids": [0, 1, 2]} for i in range(10)],
            "stats": {},
            "events": [],
        }

        with patch("elle_jax_model.nucleation._rebuild_mesh_state_from_nucleated_labels", return_value=fake_rebuilt):
            updated_labels, updated_mesh, stats = apply_nucleation_stage(
                mesh_state,
                current_labels,
                NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4, max_mesh_rebuild_growth_factor=2.0),
            )

        self.assertEqual(stats["nucleated_clusters"], 1)
        self.assertIn("_runtime_label_overrides", updated_mesh)
        self.assertEqual(updated_mesh["_runtime_fallback_override_labels"], (1,))
        self.assertEqual(int(updated_mesh["stats"]["nucleation_mesh_rebuilt"]), 0)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_mesh_rebuild_rejected"]), 1)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_rebuild_candidate_flynns"]), 10)
        self.assertTrue(np.any(np.asarray(updated_mesh["_runtime_label_overrides"], dtype=np.int32) >= 0))
        self.assertEqual(updated_mesh["_runtime_seed_unode_fields"]["source_labels"], (7, 8))

    def test_apply_nucleation_stage_retries_mesh_rebuild_after_connectivity_cleanup(self) -> None:
        current_labels = np.zeros((4, 4), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        for ix in range(4):
            for iy in range(4):
                if ix >= 2 and iy >= 2:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(0.0 for _ in range(16)),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 1.0, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 1.0, "y": 1.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11, "retained_identity": True},
            ],
            "stats": {},
        }

        fake_rebuilt_bad = {
            "nodes": [],
            "flynns": [{"flynn_id": i, "label": i, "node_ids": [0, 1, 2]} for i in range(10)],
            "stats": {},
            "events": [],
        }
        fake_rebuilt_good = {
            "nodes": [],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2], "source_flynn_id": 11},
                {"flynn_id": 12, "label": 1, "node_ids": [2, 3, 0], "source_flynn_id": 11},
            ],
            "stats": {},
            "events": [],
        }
        cleaned_labels = current_labels.copy()
        cleaned_labels[2:, 2:] = 1

        with patch(
            "elle_jax_model.nucleation._rebuild_mesh_state_from_nucleated_labels",
            side_effect=[fake_rebuilt_bad, fake_rebuilt_good],
        ), patch(
            "elle_jax_model.nucleation._enforce_connected_label_ownership",
            return_value=(cleaned_labels, {"connectivity_reassigned_unodes": 3, "connectivity_merged_components": 1}),
        ):
            updated_labels, updated_mesh, stats = apply_nucleation_stage(
                mesh_state,
                current_labels,
                NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4, max_mesh_rebuild_growth_factor=2.0),
            )

        self.assertEqual(stats["nucleated_clusters"], 1)
        np.testing.assert_array_equal(updated_labels, cleaned_labels)
        self.assertNotIn("_runtime_label_overrides", updated_mesh)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_mesh_rebuild_retried"]), 1)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_pre_rebuild_connectivity_reassigned_unodes"]), 3)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_pre_rebuild_connectivity_merged_components"]), 1)
        updated_label_values = np.asarray(
            updated_mesh["_runtime_seed_unode_fields"]["values"]["U_ATTRIB_C"],
            dtype=np.float64,
        ).reshape(4, 4)
        expected_source_labels = np.where(cleaned_labels == 1, 8.0, 7.0)
        np.testing.assert_array_equal(updated_label_values, expected_source_labels)

    def test_apply_nucleation_stage_rejects_retry_that_still_violates_fragment_guard(self) -> None:
        current_labels = np.zeros((4, 4), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        for ix in range(4):
            for iy in range(4):
                if ix >= 2 and iy >= 2:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(0.0 for _ in range(16)),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 1.0, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 1.0, "y": 1.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11, "retained_identity": True},
            ],
            "stats": {},
        }

        fake_rebuilt_bad = {
            "nodes": [],
            "flynns": [{"flynn_id": i, "label": i, "node_ids": [0, 1, 2]} for i in range(10)],
            "stats": {},
            "events": [],
        }
        cleaned_labels = current_labels.copy()
        cleaned_labels[2:, 2:] = 1

        with patch(
            "elle_jax_model.nucleation._rebuild_mesh_state_from_nucleated_labels",
            side_effect=[fake_rebuilt_bad, fake_rebuilt_bad],
        ), patch(
            "elle_jax_model.nucleation._enforce_connected_label_ownership",
            return_value=(cleaned_labels, {"connectivity_reassigned_unodes": 3, "connectivity_merged_components": 1}),
        ):
            updated_labels, updated_mesh, stats = apply_nucleation_stage(
                mesh_state,
                current_labels,
                NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4, max_mesh_rebuild_growth_factor=100.0),
            )

        self.assertEqual(stats["nucleated_clusters"], 1)
        self.assertIn("_runtime_label_overrides", updated_mesh)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_mesh_rebuild_rejected"]), 1)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_mesh_rebuild_retried"]), 1)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_rebuild_candidate_flynns"]), 10)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_rebuild_retry_candidate_flynns"]), 10)
        self.assertEqual(
            int(updated_mesh["stats"]["nucleation_rebuild_max_allowed_fragmented_flynns"]),
            5,
        )
        self.assertTrue(np.any(np.asarray(updated_mesh["_runtime_label_overrides"], dtype=np.int32) >= 0))

    def test_apply_nucleation_stage_skips_interior_secondary_cluster(self) -> None:
        current_labels = np.zeros((6, 6), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        for ix in range(6):
            for iy in range(6):
                if 2 <= ix <= 3 and 2 <= iy <= 3:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(36)),
                "positions": tuple(
                    ((ix + 0.5) / 6.0, 1.0 - ((iy + 0.5) / 6.0))
                    for ix in range(6)
                    for iy in range(6)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(6) for iy in range(6)),
                "grid_shape": (6, 6),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(7.0 for _ in range(36)),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 1.0, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 1.0, "y": 1.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11},
            ],
            "stats": {},
        }

        updated_labels, updated_mesh, stats = apply_nucleation_stage(
            mesh_state,
            current_labels,
            NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4),
        )

        np.testing.assert_array_equal(updated_labels, current_labels)
        self.assertEqual(stats["nucleated_clusters"], 0)
        self.assertGreaterEqual(stats["skipped_interior_clusters"], 1)
        self.assertIs(updated_mesh, mesh_state)

    def test_apply_nucleation_stage_skips_small_parent_flynn(self) -> None:
        current_labels = np.zeros((4, 4), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        for ix in range(4):
            for iy in range(4):
                if ix >= 2 and iy >= 2:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(
                    (0.00125 + 0.0025 * ix, 0.00875 - 0.0025 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(7.0 for _ in range(16)),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 0.01, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 0.01, "y": 0.01, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 0.01, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11},
            ],
            "stats": {},
        }

        updated_labels, updated_mesh, stats = apply_nucleation_stage(
            mesh_state,
            current_labels,
            NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4, parent_area_crit=5.0e-4),
        )

        np.testing.assert_array_equal(updated_labels, current_labels)
        self.assertEqual(stats["nucleated_clusters"], 0)
        self.assertGreaterEqual(stats["skipped_small_parent_flynns"], 1)
        self.assertIs(updated_mesh, mesh_state)

    def test_apply_nucleation_stage_uses_boundary_local_critical_seed_patch(self) -> None:
        current_labels = np.zeros((6, 6), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        attr_f_values: list[float] = []
        for ix in range(6):
            for iy in range(6):
                if ix < 4 and iy < 4:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))
                attr_f_values.append(10.0 if (ix, iy) == (0, 0) else 0.0)

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(36)),
                "positions": tuple(
                    ((ix + 0.5) / 6.0, 1.0 - ((iy + 0.5) / 6.0))
                    for ix in range(6)
                    for iy in range(6)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(6) for iy in range(6)),
                "grid_shape": (6, 6),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(7.0 for _ in range(36)),
                    "U_ATTRIB_F": tuple(attr_f_values),
                    "U_DISLOCDEN": tuple(1.0 for _ in range(36)),
                },
            },
            "_runtime_seed_unode_sections": {
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 1.0, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 1.0, "y": 1.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11},
            ],
            "stats": {},
        }

        updated_labels, updated_mesh, stats = apply_nucleation_stage(
            mesh_state,
            current_labels,
            NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4),
        )

        promoted_mask = updated_labels == int(np.max(updated_labels))
        expected_mask = np.zeros_like(current_labels, dtype=bool)
        expected_mask[0:3, 0:3] = True
        np.testing.assert_array_equal(promoted_mask, expected_mask)
        self.assertEqual(stats["nucleated_clusters"], 1)
        self.assertGreaterEqual(stats["trimmed_cluster_unodes"], 1)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_trimmed_cluster_unodes"]), int(stats["trimmed_cluster_unodes"]))

    def test_apply_nucleation_stage_ignores_bootstrap_dislocden_for_single_seed_gate(self) -> None:
        current_labels = np.zeros((6, 6), dtype=np.int32)
        euler_values: list[tuple[float, float, float]] = []
        attr_f_values: list[float] = []
        for ix in range(6):
            for iy in range(6):
                if ix < 4 and iy < 4:
                    euler_values.append((25.0, 0.0, 0.0))
                else:
                    euler_values.append((0.0, 0.0, 0.0))
                attr_f_values.append(10.0 if (ix, iy) == (0, 0) else 0.0)

        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(36)),
                "positions": tuple(
                    ((ix + 0.5) / 6.0, 1.0 - ((iy + 0.5) / 6.0))
                    for ix in range(6)
                    for iy in range(6)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(6) for iy in range(6)),
                "grid_shape": (6, 6),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (7,),
                "values": {
                    "U_ATTRIB_C": tuple(7.0 for _ in range(36)),
                    "U_ATTRIB_F": tuple(attr_f_values),
                    "U_DISLOCDEN": tuple(0.0 for _ in range(36)),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_EULER_3",),
                "values": {
                    "U_EULER_3": tuple(euler_values),
                },
            },
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 1, "x": 1.0, "y": 0.0, "neighbors": [0, 2], "flynns": [11]},
                {"node_id": 2, "x": 1.0, "y": 1.0, "neighbors": [1, 3], "flynns": [11]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "neighbors": [0, 2], "flynns": [11]},
            ],
            "flynns": [
                {"flynn_id": 11, "label": 0, "node_ids": [0, 1, 2, 3], "source_flynn_id": 11},
            ],
            "stats": {},
        }

        with patch(
            "elle_jax_model.nucleation._label_boundary_seed_mask_from_mesh",
            return_value=np.ones_like(current_labels, dtype=bool),
        ):
            updated_labels, updated_mesh, stats = apply_nucleation_stage(
                mesh_state,
                current_labels,
                NucleationConfig(high_angle_boundary_deg=5.0, min_cluster_unodes=4),
            )

        promoted_mask = updated_labels == int(np.max(updated_labels))
        self.assertEqual(stats["nucleated_clusters"], 1)
        self.assertEqual(int(np.count_nonzero(promoted_mask)), 16)
        self.assertEqual(int(updated_mesh["stats"]["nucleation_trimmed_cluster_unodes"]), 0)

    def test_enforce_connected_label_ownership_reassigns_smaller_fragment(self) -> None:
        labels = np.array(
            [
                [1, 0, 0],
                [0, 0, 0],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        stabilized, stats = _enforce_connected_label_ownership(
            labels,
            reference_labels=np.zeros_like(labels, dtype=np.int32),
            protected_labels=(1,),
        )

        self.assertEqual(int(stats["connectivity_reassigned_unodes"]), 1)
        self.assertEqual(int(stats["connectivity_merged_components"]), 1)
        self.assertEqual(int(np.count_nonzero(stabilized == 1)), 1)
        self.assertEqual(int(np.max(stabilized)), 1)

    def test_enforce_connected_label_ownership_keeps_large_secondary_component(self) -> None:
        labels = np.array(
            [
                [1, 1, 0, 0],
                [1, 1, 0, 0],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        stabilized, stats = _enforce_connected_label_ownership(
            labels,
            reference_labels=np.zeros_like(labels, dtype=np.int32),
            merge_max_component_size=3,
        )

        np.testing.assert_array_equal(stabilized, labels)
        self.assertEqual(int(stats["connectivity_reassigned_unodes"]), 0)
        self.assertEqual(int(stats["connectivity_merged_components"]), 0)

    def test_stabilize_label_identities_preserves_reference_ids_under_permutation(self) -> None:
        reference = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [2, 2, 3, 3],
                [2, 2, 3, 3],
            ],
            dtype=np.int32,
        )
        candidate = np.array(
            [
                [2, 2, 3, 3],
                [2, 2, 3, 3],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )

        stabilized, stats = mesh_module._stabilize_label_identities(reference, candidate)

        np.testing.assert_array_equal(stabilized, reference)
        self.assertEqual(int(stats["identity_stabilized_labels"]), 4)
        self.assertEqual(int(stats["identity_stabilized_pixels"]), 16)

    def test_stabilize_label_identities_keeps_new_split_label_unique(self) -> None:
        reference = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        candidate = np.array(
            [
                [2, 2, 1, 1],
                [2, 2, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )

        stabilized, stats = mesh_module._stabilize_label_identities(reference, candidate)

        self.assertEqual(set(np.unique(stabilized).tolist()), {0, 1, 2})
        np.testing.assert_array_equal(stabilized[:2, :2], np.full((2, 2), 2, dtype=np.int32))
        self.assertEqual(int(stats["identity_stabilized_labels"]), 0)
        self.assertEqual(int(stats["identity_stabilized_pixels"]), 0)

    def test_run_faithful_gbm_simulation_can_bundle_multiple_gbm_stages_per_snapshot(self) -> None:
        contexts: list[dict[str, object]] = []
        stage_calls: list[int] = []

        def record_snapshot(_step, _phi, _topology_snapshot, mesh_feedback_context) -> None:
            contexts.append(dict(mesh_feedback_context))

        def fake_couple(phi, mesh_feedback, tracked_topology=None, base_mesh_state=None):
            del tracked_topology
            stage_calls.append(len(stage_calls) + 1)
            mesh_state = copy.deepcopy(
                base_mesh_state if base_mesh_state is not None else mesh_feedback.initial_mesh_state
            )
            if mesh_state is None:
                mesh_state = {"stats": {}}
            mesh_state.setdefault("stats", {})
            mesh_state["stats"]["fake_stage_call"] = len(stage_calls)
            return np.asarray(phi, dtype=np.float32), mesh_state, {"fake_stage_call": len(stage_calls)}

        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            with patch("elle_jax_model.faithful_runtime.couple_mesh_to_order_parameters", side_effect=fake_couple):
                _final_state, snapshots, _topology_history = run_faithful_gbm_simulation(
                    init_elle_path=elle_path,
                    steps=2,
                    save_every=1,
                    mesh_relax_steps=0,
                    mesh_topology_steps=0,
                    subloops_per_snapshot=2,
                    gbm_steps_per_subloop=3,
                    on_snapshot=record_snapshot,
                )

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(len(stage_calls), 12)
        self.assertEqual(contexts[0]["outer_step"], 1)
        self.assertEqual(contexts[0]["stage_index"], 6)
        self.assertEqual(contexts[0]["stages_per_snapshot"], 6)
        self.assertEqual(contexts[0]["subloops_per_snapshot"], 2)
        self.assertEqual(contexts[0]["gbm_steps_per_subloop"], 3)
        self.assertEqual(contexts[1]["outer_step"], 2)
        self.assertEqual(contexts[1]["stage_index"], 12)
        self.assertEqual(contexts[1]["mesh_state"]["stats"]["workflow_stages_per_snapshot"], 6)

    def test_run_faithful_gbm_simulation_can_include_recovery_stages_per_subloop(self) -> None:
        contexts: list[dict[str, object]] = []
        gbm_calls: list[int] = []
        recovery_calls: list[int] = []

        def record_snapshot(_step, _phi, _topology_snapshot, mesh_feedback_context) -> None:
            contexts.append(dict(mesh_feedback_context))

        def fake_couple(phi, mesh_feedback, tracked_topology=None, base_mesh_state=None):
            del tracked_topology
            gbm_calls.append(len(gbm_calls) + 1)
            mesh_state = copy.deepcopy(
                base_mesh_state if base_mesh_state is not None else mesh_feedback.initial_mesh_state
            )
            if mesh_state is None:
                mesh_state = {"stats": {}}
            mesh_state.setdefault("stats", {})
            return np.asarray(phi, dtype=np.float32), mesh_state, {"fake_stage_call": len(gbm_calls)}

        def fake_recovery(mesh_state, current_labels, config, *, recovery_stage_index=0):
            del current_labels, config
            recovery_calls.append(int(recovery_stage_index))
            mesh_copy = copy.deepcopy(mesh_state)
            mesh_copy.setdefault("stats", {})
            mesh_copy["stats"]["fake_recovery_stage_index"] = int(recovery_stage_index)
            return mesh_copy, {
                "recovery_stage_index": int(recovery_stage_index),
                "recovery_applied": 1,
                "rotated_unodes": int(recovery_stage_index + 1),
                "density_reduced_unodes": int(recovery_stage_index),
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            with patch("elle_jax_model.faithful_runtime.couple_mesh_to_order_parameters", side_effect=fake_couple):
                with patch("elle_jax_model.faithful_runtime.apply_recovery_stage", side_effect=fake_recovery):
                    _final_state, snapshots, _topology_history = run_faithful_gbm_simulation(
                        init_elle_path=elle_path,
                        steps=1,
                        save_every=1,
                        mesh_relax_steps=0,
                        mesh_topology_steps=0,
                        subloops_per_snapshot=2,
                        gbm_steps_per_subloop=3,
                        recovery_steps_per_subloop=2,
                        on_snapshot=record_snapshot,
                    )

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(len(gbm_calls), 6)
        self.assertEqual(recovery_calls, [0, 1, 0, 1])
        self.assertEqual(contexts[0]["stage_kind"], "recovery")
        self.assertEqual(contexts[0]["stage_index"], 10)
        self.assertEqual(contexts[0]["stages_per_snapshot"], 10)
        self.assertEqual(contexts[0]["recovery_steps_per_subloop"], 2)
        self.assertEqual(
            contexts[0]["recovery_cumulative_stats"]["recovery_applied_stages_total"],
            4,
        )
        self.assertEqual(
            contexts[0]["recovery_cumulative_stats"]["recovery_rotated_unodes_total"],
            6,
        )
        self.assertEqual(
            contexts[0]["recovery_cumulative_stats"]["recovery_density_reduced_unodes_total"],
            2,
        )
        self.assertEqual(
            contexts[0]["mesh_state"]["stats"]["workflow_recovery_rotated_unodes_total"],
            6,
        )

    def test_run_faithful_gbm_simulation_can_include_nucleation_stages_per_subloop(self) -> None:
        contexts: list[dict[str, object]] = []
        gbm_calls: list[int] = []
        nucleation_calls: list[int] = []

        def record_snapshot(_step, _phi, _topology_snapshot, mesh_feedback_context) -> None:
            contexts.append(dict(mesh_feedback_context))

        def fake_couple(phi, mesh_feedback, tracked_topology=None, base_mesh_state=None):
            del tracked_topology
            gbm_calls.append(len(gbm_calls) + 1)
            mesh_state = copy.deepcopy(
                base_mesh_state if base_mesh_state is not None else mesh_feedback.initial_mesh_state
            )
            if mesh_state is None:
                mesh_state = {"stats": {}}
            mesh_state.setdefault("stats", {})
            return np.asarray(phi, dtype=np.float32), mesh_state, {"fake_stage_call": len(gbm_calls)}

        def fake_nucleation(mesh_state, current_labels, config, *, nucleation_stage_index=0):
            del current_labels, config
            nucleation_calls.append(int(nucleation_stage_index))
            mesh_copy = copy.deepcopy(mesh_state)
            mesh_copy.setdefault("stats", {})
            mesh_copy["stats"]["fake_nucleation_stage_index"] = int(nucleation_stage_index)
            labels = np.zeros((2, 2), dtype=np.int32)
            return labels, mesh_copy, {
                "nucleation_stage_index": int(nucleation_stage_index),
                "nucleation_applied": 1,
            }

        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            with patch("elle_jax_model.faithful_runtime.couple_mesh_to_order_parameters", side_effect=fake_couple):
                with patch("elle_jax_model.faithful_runtime.apply_nucleation_stage", side_effect=fake_nucleation):
                    _final_state, snapshots, _topology_history = run_faithful_gbm_simulation(
                        init_elle_path=elle_path,
                        steps=1,
                        save_every=1,
                        mesh_relax_steps=0,
                        mesh_topology_steps=0,
                        subloops_per_snapshot=2,
                        nucleation_steps_per_subloop=2,
                        gbm_steps_per_subloop=1,
                        on_snapshot=record_snapshot,
                    )

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(len(gbm_calls), 2)
        self.assertEqual(nucleation_calls, [0, 1, 0, 1])
        self.assertEqual(contexts[0]["stage_kind"], "gbm")
        self.assertEqual(contexts[0]["stage_index"], 6)
        self.assertEqual(contexts[0]["stages_per_snapshot"], 6)
        self.assertEqual(contexts[0]["nucleation_steps_per_subloop"], 2)
        self.assertEqual(contexts[0]["gbm_steps_per_subloop"], 1)

    def test_resolve_runtime_preset_preserves_manual_params_when_none(self) -> None:
        params = resolve_runtime_preset(
            "none",
            dt=0.04,
            mobility=0.8,
            gradient_penalty=1.1,
            interaction_strength=1.9,
            init_smoothing_steps=1,
            init_noise=0.03,
        )

        self.assertEqual(
            params,
            {
                "dt": 0.04,
                "mobility": 0.8,
                "gradient_penalty": 1.1,
                "interaction_strength": 1.9,
                "init_smoothing_steps": 1,
                "init_noise": 0.03,
            },
        )

    def test_evaluate_release_dataset_benchmarks_detects_hotter_case_more_active(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"

        report = evaluate_release_dataset_benchmarks(data_dir)

        self.assertTrue(report["flags"]["grain_hotter_case_more_active"])
        self.assertTrue(report["flags"]["all_euler_hotter_cases_more_active"])
        self.assertIn("paper_signature_assessment", report)
        self.assertEqual(report["paper_signature_assessment"]["applicable_checks"], 5)
        self.assertEqual(report["paper_signature_assessment"]["passed_checks"], 5)

    def test_assess_experiment_family_signature_alignment_tracks_cross_family_signatures(self) -> None:
        family_reports = {
            "0": _make_experiment_family_static_report(
                mean_grain_area_end=1.0,
                mean_grain_area_relative_delta=0.05,
                mean_aspect_ratio_end=3.2,
                mean_differential_stress_end=12.0,
                peak_stress_strain=0.80,
                c_axis_p_delta=0.10,
                c_axis_vertical_delta=0.05,
            ),
            "1": _make_experiment_family_static_report(
                mean_grain_area_end=1.2,
                mean_grain_area_relative_delta=0.25,
                mean_aspect_ratio_end=2.8,
                mean_differential_stress_end=10.0,
                peak_stress_strain=0.65,
                c_axis_p_delta=0.15,
                c_axis_vertical_delta=0.08,
            ),
            "10": _make_experiment_family_static_report(
                mean_grain_area_end=1.5,
                mean_grain_area_relative_delta=0.50,
                mean_aspect_ratio_end=2.1,
                mean_differential_stress_end=7.0,
                peak_stress_strain=0.45,
                c_axis_p_delta=0.20,
                c_axis_vertical_delta=0.12,
            ),
            "25": _make_experiment_family_static_report(
                mean_grain_area_end=1.9,
                mean_grain_area_relative_delta=0.85,
                mean_aspect_ratio_end=1.5,
                mean_differential_stress_end=5.0,
                peak_stress_strain=0.30,
                c_axis_p_delta=0.25,
                c_axis_vertical_delta=0.16,
            ),
        }

        assessment = assess_experiment_family_signature_alignment(family_reports)

        self.assertEqual(assessment["family_order"], ["0", "1", "10", "25"])
        self.assertEqual(assessment["applicable_checks"], 8)
        self.assertEqual(assessment["passed_checks"], 8)
        by_name = {entry["name"]: entry for entry in assessment["checks"]}
        self.assertEqual(by_name["family0_mean_area_nearly_constant"]["status"], "pass")
        self.assertEqual(by_name["higher_drx_mean_area_end_increasing"]["status"], "pass")
        self.assertEqual(by_name["higher_drx_aspect_ratio_end_decreasing"]["status"], "pass")
        self.assertEqual(by_name["higher_drx_end_stress_decreasing"]["status"], "pass")
        self.assertEqual(by_name["higher_drx_peak_stress_strain_decreasing"]["status"], "pass")
        self.assertEqual(by_name["all_families_c_axis_strengthening"]["status"], "pass")
        self.assertEqual(by_name["all_families_c_axis_vertical_clustering"]["status"], "pass")
        self.assertEqual(by_name["all_families_size_stronger_than_schmid_survival"]["status"], "pass")

    def test_evaluate_experiment_family_benchmarks_orders_families_and_emits_assessment(self) -> None:
        synthetic_reports = [
            _make_experiment_family_static_report(
                mean_grain_area_end=1.0,
                mean_grain_area_relative_delta=0.05,
                mean_aspect_ratio_end=3.2,
                mean_differential_stress_end=12.0,
                peak_stress_strain=0.80,
                c_axis_p_delta=0.10,
                c_axis_vertical_delta=0.05,
            ),
            _make_experiment_family_static_report(
                mean_grain_area_end=1.2,
                mean_grain_area_relative_delta=0.25,
                mean_aspect_ratio_end=2.8,
                mean_differential_stress_end=10.0,
                peak_stress_strain=0.65,
                c_axis_p_delta=0.15,
                c_axis_vertical_delta=0.08,
            ),
            _make_experiment_family_static_report(
                mean_grain_area_end=1.5,
                mean_grain_area_relative_delta=0.50,
                mean_aspect_ratio_end=2.1,
                mean_differential_stress_end=7.0,
                peak_stress_strain=0.45,
                c_axis_p_delta=0.20,
                c_axis_vertical_delta=0.12,
            ),
            _make_experiment_family_static_report(
                mean_grain_area_end=1.9,
                mean_grain_area_relative_delta=0.85,
                mean_aspect_ratio_end=1.5,
                mean_differential_stress_end=5.0,
                peak_stress_strain=0.30,
                c_axis_p_delta=0.25,
                c_axis_vertical_delta=0.16,
            ),
        ]

        with patch(
            "elle_jax_model.benchmark_validation.evaluate_static_grain_growth_benchmark",
            side_effect=synthetic_reports,
        ):
            report = evaluate_experiment_family_benchmarks(
                {
                    "10": Path("/tmp/family10"),
                    "0": Path("/tmp/family0"),
                    "25": Path("/tmp/family25"),
                    "1": Path("/tmp/family1"),
                }
            )

        self.assertEqual(report["family_order"], ["0", "1", "10", "25"])
        self.assertEqual(report["paper_signature_assessment"]["passed_checks"], 8)
        self.assertEqual(report["benchmarks"]["end_mean_grain_area"]["25"], 1.9)
        self.assertEqual(report["benchmarks"]["peak_stress_strain"]["10"], 0.45)

    def test_load_experiment_family_benchmark_report_extracts_static_section(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            path = _write_experiment_family_benchmark_report(
                Path(tmpdir) / "family_10.json",
                _make_experiment_family_static_report(
                    mean_grain_area_end=1.5,
                    mean_grain_area_relative_delta=0.50,
                    mean_aspect_ratio_end=2.1,
                    mean_differential_stress_end=7.0,
                    peak_stress_strain=0.45,
                    c_axis_p_delta=0.20,
                    c_axis_vertical_delta=0.12,
                ),
            )

            loaded = load_experiment_family_benchmark_report(path)

        self.assertEqual(loaded["reference_dir"], "synthetic_family")
        self.assertEqual(
            loaded["reference_trends"]["metrics"]["mean_grain_area"]["end"],
            1.5,
        )
        self.assertTrue(str(path).endswith(loaded["source_report_path"]))

    def test_evaluate_experiment_family_benchmark_reports_reads_json_sources(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            paths = {
                "0": _write_experiment_family_benchmark_report(
                    Path(tmpdir) / "family_0.json",
                    _make_experiment_family_static_report(
                        mean_grain_area_end=1.0,
                        mean_grain_area_relative_delta=0.05,
                        mean_aspect_ratio_end=3.2,
                        mean_differential_stress_end=12.0,
                        peak_stress_strain=0.80,
                        c_axis_p_delta=0.10,
                        c_axis_vertical_delta=0.05,
                    ),
                ),
                "1": _write_experiment_family_benchmark_report(
                    Path(tmpdir) / "family_1.json",
                    _make_experiment_family_static_report(
                        mean_grain_area_end=1.2,
                        mean_grain_area_relative_delta=0.25,
                        mean_aspect_ratio_end=2.8,
                        mean_differential_stress_end=10.0,
                        peak_stress_strain=0.65,
                        c_axis_p_delta=0.15,
                        c_axis_vertical_delta=0.08,
                    ),
                ),
                "10": _write_experiment_family_benchmark_report(
                    Path(tmpdir) / "family_10.json",
                    _make_experiment_family_static_report(
                        mean_grain_area_end=1.5,
                        mean_grain_area_relative_delta=0.50,
                        mean_aspect_ratio_end=2.1,
                        mean_differential_stress_end=7.0,
                        peak_stress_strain=0.45,
                        c_axis_p_delta=0.20,
                        c_axis_vertical_delta=0.12,
                    ),
                ),
                "25": _write_experiment_family_benchmark_report(
                    Path(tmpdir) / "family_25.json",
                    _make_experiment_family_static_report(
                        mean_grain_area_end=1.9,
                        mean_grain_area_relative_delta=0.85,
                        mean_aspect_ratio_end=1.5,
                        mean_differential_stress_end=5.0,
                        peak_stress_strain=0.30,
                        c_axis_p_delta=0.25,
                        c_axis_vertical_delta=0.16,
                    ),
                ),
            }

            report = evaluate_experiment_family_benchmark_reports(paths)

        self.assertEqual(report["family_order"], ["0", "1", "10", "25"])
        self.assertEqual(report["paper_signature_assessment"]["passed_checks"], 8)
        self.assertEqual(
            report["families"]["10"]["source_report_path"],
            str(paths["10"]),
        )

    def test_evaluate_experiment_family_suite_prefers_report_input_over_directory(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            report_path = _write_experiment_family_benchmark_report(
                Path(tmpdir) / "family_10.json",
                _make_experiment_family_static_report(
                    mean_grain_area_end=9.9,
                    mean_grain_area_relative_delta=0.50,
                    mean_aspect_ratio_end=2.1,
                    mean_differential_stress_end=7.0,
                    peak_stress_strain=0.45,
                    c_axis_p_delta=0.20,
                    c_axis_vertical_delta=0.12,
                ),
            )
            synthetic_dir_report = _make_experiment_family_static_report(
                mean_grain_area_end=1.5,
                mean_grain_area_relative_delta=0.50,
                mean_aspect_ratio_end=2.1,
                mean_differential_stress_end=7.0,
                peak_stress_strain=0.45,
                c_axis_p_delta=0.20,
                c_axis_vertical_delta=0.12,
            )

            with patch(
                "elle_jax_model.benchmark_validation.evaluate_static_grain_growth_benchmark",
                return_value=synthetic_dir_report,
            ):
                report = evaluate_experiment_family_suite(
                    experiment_family_dirs={"10": Path("/tmp/family10")},
                    experiment_family_reports={"10": report_path},
                )

        self.assertEqual(report["benchmarks"]["end_mean_grain_area"]["10"], 9.9)
        self.assertEqual(
            report["families"]["10"]["source_report_path"],
            str(report_path),
        )

    def test_build_benchmark_validation_report_emits_aggregate_acceptance(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_mechanics_sequence_examples(Path(reference_dir))
            _write_mechanics_sequence_examples(Path(candidate_dir))
            ref_step1 = Path(reference_dir) / "mechanics_0001.elle"
            ref_step2 = Path(reference_dir) / "mechanics_0002.elle"
            cand_step1 = Path(candidate_dir) / "mechanics_0001.elle"
            cand_step2 = Path(candidate_dir) / "mechanics_0002.elle"
            _write_mechanics_mesh_sidecar(ref_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(ref_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)
            _write_mechanics_mesh_sidecar(cand_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(cand_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)

            report = build_benchmark_validation_report(
                reference_dir=reference_dir,
                candidate_dir=candidate_dir,
                data_dir=data_dir,
                pattern="mechanics_*.elle",
            )

        self.assertIn("benchmark_acceptance", report)
        section_names = {entry["section"] for entry in report["benchmark_acceptance"]["sections"]}
        self.assertIn("static_grain_growth", section_names)
        self.assertIn("release_dataset_benchmarks", section_names)
        self.assertGreaterEqual(int(report["benchmark_acceptance"]["applicable_checks"]), 5)
        self.assertGreaterEqual(int(report["benchmark_acceptance"]["passed_checks"]), 5)

    def test_build_benchmark_validation_report_includes_experiment_family_acceptance(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"
        family_assessment = {
            "family_order": ["0", "1", "10", "25"],
            "checks": [],
            "applicable_checks": 7,
            "passed_checks": 7,
            "pass_fraction": 1.0,
        }
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_mechanics_sequence_examples(Path(reference_dir))
            _write_mechanics_sequence_examples(Path(candidate_dir))
            ref_step1 = Path(reference_dir) / "mechanics_0001.elle"
            ref_step2 = Path(reference_dir) / "mechanics_0002.elle"
            cand_step1 = Path(candidate_dir) / "mechanics_0001.elle"
            cand_step2 = Path(candidate_dir) / "mechanics_0002.elle"
            _write_mechanics_mesh_sidecar(ref_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(ref_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)
            _write_mechanics_mesh_sidecar(cand_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(cand_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)

            with patch(
                "elle_jax_model.benchmark_validation.evaluate_experiment_family_suite",
                return_value={"paper_signature_assessment": family_assessment},
            ):
                report = build_benchmark_validation_report(
                    reference_dir=reference_dir,
                    candidate_dir=candidate_dir,
                    data_dir=data_dir,
                    experiment_family_dirs={"0": Path("/tmp/f0"), "1": Path("/tmp/f1")},
                    pattern="mechanics_*.elle",
                )

        self.assertIn("experiment_family_benchmarks", report)
        aggregate = assess_benchmark_report_acceptance(report)
        section_names = {entry["section"] for entry in aggregate["sections"]}
        self.assertIn("experiment_family_benchmarks", section_names)
        self.assertGreaterEqual(int(aggregate["applicable_checks"]), 7)
        self.assertGreaterEqual(int(aggregate["passed_checks"]), 7)

    def test_build_benchmark_validation_report_includes_experiment_family_reports(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"
        family_assessment = {
            "family_order": ["0", "1", "10", "25"],
            "checks": [],
            "applicable_checks": 6,
            "passed_checks": 6,
            "pass_fraction": 1.0,
        }
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_mechanics_sequence_examples(Path(reference_dir))
            _write_mechanics_sequence_examples(Path(candidate_dir))
            ref_step1 = Path(reference_dir) / "mechanics_0001.elle"
            ref_step2 = Path(reference_dir) / "mechanics_0002.elle"
            cand_step1 = Path(candidate_dir) / "mechanics_0001.elle"
            cand_step2 = Path(candidate_dir) / "mechanics_0002.elle"
            _write_mechanics_mesh_sidecar(ref_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(ref_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)
            _write_mechanics_mesh_sidecar(cand_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(cand_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)

            with patch(
                "elle_jax_model.benchmark_validation.evaluate_experiment_family_suite",
                return_value={"paper_signature_assessment": family_assessment},
            ):
                report = build_benchmark_validation_report(
                    reference_dir=reference_dir,
                    candidate_dir=candidate_dir,
                    data_dir=data_dir,
                    experiment_family_reports={"0": Path("/tmp/f0.json"), "1": Path("/tmp/f1.json")},
                    pattern="mechanics_*.elle",
                )

        self.assertIn("experiment_family_benchmarks", report)
        aggregate = assess_benchmark_report_acceptance(report)
        section_names = {entry["section"] for entry in aggregate["sections"]}
        self.assertIn("experiment_family_benchmarks", section_names)
        self.assertGreaterEqual(int(aggregate["applicable_checks"]), 6)
        self.assertGreaterEqual(int(aggregate["passed_checks"]), 6)

    def test_build_benchmark_validation_report_includes_legacy_old_stats_bookkeeping(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"
        legacy_old_stats_path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "old.stats"
        )
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_mechanics_sequence_examples(Path(reference_dir))
            _write_mechanics_sequence_examples(Path(candidate_dir))
            ref_step1 = Path(reference_dir) / "mechanics_0001.elle"
            ref_step2 = Path(reference_dir) / "mechanics_0002.elle"
            cand_step1 = Path(candidate_dir) / "mechanics_0001.elle"
            cand_step2 = Path(candidate_dir) / "mechanics_0002.elle"
            _write_mechanics_mesh_sidecar(ref_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(ref_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)
            _write_mechanics_mesh_sidecar(cand_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(cand_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)
            candidate_mesh_json = Path(candidate_dir) / "mesh_00001.json"
            candidate_mesh_json.write_text(
                json.dumps(
                    {
                        "flynns": [
                            {
                                "flynn_id": 10,
                                "source_flynn_id": 10,
                                "retained_identity": True,
                                "parents": [],
                            },
                            {
                                "flynn_id": 11,
                                "source_flynn_id": 11,
                                "retained_identity": False,
                                "parents": [10, 11],
                            },
                            {
                                "flynn_id": 12,
                                "source_flynn_id": -1,
                                "retained_identity": False,
                                "parents": [11],
                            },
                        ],
                        "stats": {
                            "mesh_split_flynns": 1,
                            "mesh_merged_flynns": 0,
                            "num_flynns": 3,
                        },
                    }
                ),
                encoding="utf-8",
            )

            report = build_benchmark_validation_report(
                reference_dir=reference_dir,
                candidate_dir=candidate_dir,
                candidate_mesh_json_path=candidate_mesh_json,
                data_dir=data_dir,
                legacy_old_stats_path=legacy_old_stats_path,
                pattern="mechanics_*.elle",
            )

        self.assertIn("legacy_old_stats_bookkeeping", report)
        bookkeeping = report["legacy_old_stats_bookkeeping"]
        self.assertEqual(int(bookkeeping["legacy_total_flynn_count"]), 49)
        self.assertEqual(int(bookkeeping["current_total_flynn_count"]), 3)
        self.assertEqual(int(bookkeeping["current_source_mapped_flynn_count"]), 2)
        self.assertEqual(int(bookkeeping["current_source_orphan_flynn_count"]), 1)
        self.assertFalse(bool(bookkeeping["total_flynn_count_match"]))

    def test_build_benchmark_validation_report_requires_both_old_stats_inputs(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"
        with tempfile.TemporaryDirectory() as reference_dir:
            _write_mechanics_sequence_examples(Path(reference_dir))
            with self.assertRaises(ValueError):
                build_benchmark_validation_report(
                    reference_dir=reference_dir,
                    data_dir=data_dir,
                    candidate_mesh_json_path=Path("/tmp/mesh_00001.json"),
                    pattern="mechanics_*.elle",
                )

    def test_build_benchmark_validation_report_includes_legacy_final_statistics(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"
        legacy_stats_path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "tmpstats.dat"
        )
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_mechanics_sequence_examples(Path(reference_dir))
            _write_mechanics_sequence_examples(Path(candidate_dir))
            ref_step1 = Path(reference_dir) / "mechanics_0001.elle"
            ref_step2 = Path(reference_dir) / "mechanics_0002.elle"
            cand_step1 = Path(candidate_dir) / "mechanics_0001.elle"
            cand_step2 = Path(candidate_dir) / "mechanics_0002.elle"
            _write_mechanics_mesh_sidecar(ref_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(ref_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)
            _write_mechanics_mesh_sidecar(cand_step1, cumulative_simple_shear=0.20)
            _write_mechanics_mesh_sidecar(cand_step2, cumulative_simple_shear=0.55, simple_shear_increment=0.35)

            report = build_benchmark_validation_report(
                reference_dir=reference_dir,
                candidate_dir=candidate_dir,
                candidate_legacy_statistics_path=legacy_stats_path,
                data_dir=data_dir,
                pattern="mechanics_*.elle",
            )

        self.assertIn("legacy_final_statistics", report)
        comparison = report["legacy_final_statistics"]["comparison"]
        self.assertEqual(comparison["legacy_statistics_kind"], "tmpstats.dat")
        self.assertIn("grain_count_delta", comparison)
        self.assertIn("mean_grain_area_delta", comparison)
        self.assertIn("second_moment_grain_size_delta", comparison)

    def test_build_benchmark_validation_report_requires_candidate_dir_for_legacy_final_statistics(self) -> None:
        data_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "data"
        legacy_stats_path = (
            PROJECT_ROOT.parent
            / "processes"
            / "statistics"
            / "tmpstats.dat"
        )
        with tempfile.TemporaryDirectory() as reference_dir:
            _write_mechanics_sequence_examples(Path(reference_dir))
            with self.assertRaises(ValueError):
                build_benchmark_validation_report(
                    reference_dir=reference_dir,
                    data_dir=data_dir,
                    candidate_legacy_statistics_path=legacy_stats_path,
                    pattern="mechanics_*.elle",
                )

    def test_discover_experiment_family_dirs_prefers_direct_numeric_subdirs(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("0", "1", "10", "25", "family_10_alt", "notes"):
                (root / name).mkdir()

            discovered = _discover_experiment_family_dirs(root)

        self.assertEqual(set(discovered), {"0", "1", "10", "25"})
        self.assertEqual(discovered["10"], root / "10")
        self.assertEqual(discovered["25"], root / "25")

    def test_load_experiment_family_manifest_reads_json_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "families.json"
            manifest.write_text(json.dumps({"0": "/tmp/f0", "10": "/tmp/f10"}), encoding="utf-8")

            loaded = _load_experiment_family_manifest(manifest)

        self.assertEqual(loaded["0"], Path("/tmp/f0"))
        self.assertEqual(loaded["10"], Path("/tmp/f10"))

    def test_resolve_experiment_family_dirs_applies_root_manifest_and_explicit_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            root.mkdir()
            (root / "0").mkdir()
            (root / "family_10_run").mkdir()
            manifest = Path(tmpdir) / "families.json"
            manifest.write_text(
                json.dumps({"1": "/tmp/manifest_f1", "10": "/tmp/manifest_f10"}),
                encoding="utf-8",
            )

            resolved = _resolve_experiment_family_dirs(
                explicit_values=["25=/tmp/explicit_f25", "10=/tmp/explicit_f10"],
                manifest_path=manifest,
                root=root,
            )

        self.assertEqual(resolved["0"], root / "0")
        self.assertEqual(resolved["1"], Path("/tmp/manifest_f1"))
        self.assertEqual(resolved["10"], Path("/tmp/explicit_f10"))
        self.assertEqual(resolved["25"], Path("/tmp/explicit_f25"))

    def test_discover_experiment_family_report_paths_prefers_direct_numeric_json(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            for name in ("0.json", "1.json", "10.json", "25.json", "family_10_alt.json", "notes.json"):
                (root / name).write_text("{}", encoding="utf-8")

            discovered = _discover_experiment_family_report_paths(root)

        self.assertEqual(set(discovered), {"0", "1", "10", "25"})
        self.assertEqual(discovered["10"], root / "10.json")
        self.assertEqual(discovered["25"], root / "25.json")

    def test_load_experiment_family_report_manifest_reads_json_mapping(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            manifest = Path(tmpdir) / "family_reports.json"
            manifest.write_text(json.dumps({"0": "/tmp/f0.json", "10": "/tmp/f10.json"}), encoding="utf-8")

            loaded = _load_experiment_family_report_manifest(manifest)

        self.assertEqual(loaded["0"], Path("/tmp/f0.json"))
        self.assertEqual(loaded["10"], Path("/tmp/f10.json"))

    def test_resolve_experiment_family_report_paths_applies_root_manifest_and_explicit_precedence(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            root = Path(tmpdir) / "root"
            root.mkdir()
            (root / "0.json").write_text("{}", encoding="utf-8")
            (root / "family_10_report.json").write_text("{}", encoding="utf-8")
            manifest = Path(tmpdir) / "family_reports.json"
            manifest.write_text(
                json.dumps({"1": "/tmp/manifest_f1.json", "10": "/tmp/manifest_f10.json"}),
                encoding="utf-8",
            )

            resolved = _resolve_experiment_family_report_paths(
                explicit_values=["25=/tmp/explicit_f25.json", "10=/tmp/explicit_f10.json"],
                manifest_path=manifest,
                root=root,
            )

        self.assertEqual(resolved["0"], root / "0.json")
        self.assertEqual(resolved["1"], Path("/tmp/manifest_f1.json"))
        self.assertEqual(resolved["10"], Path("/tmp/explicit_f10.json"))
        self.assertEqual(resolved["25"], Path("/tmp/explicit_f25.json"))

    def test_compare_elle_phasefield_states_reports_zero_for_identical_fields(self) -> None:
        theta = np.zeros((5, 4), dtype=np.float32)
        temperature = np.zeros((5, 4), dtype=np.float32)
        theta[2, 1] = 0.75
        temperature[2, 1] = -0.5

        report = compare_elle_phasefield_states(theta, temperature, theta, temperature)

        self.assertEqual(report["theta_rmse"], 0.0)
        self.assertEqual(report["temperature_rmse"], 0.0)
        self.assertEqual(report["theta_solid_iou"], 1.0)

    def test_compare_elle_phasefield_files_detects_changes(self) -> None:
        theta_a = np.zeros((4, 4), dtype=np.float32)
        temp_a = np.zeros((4, 4), dtype=np.float32)
        theta_b = theta_a.copy()
        temp_b = temp_a.copy()
        theta_b[1, 1] = 1.0
        temp_b[1, 1] = 0.25

        with tempfile.TemporaryDirectory() as tmpdir:
            ref = Path(tmpdir) / "ref.elle"
            cand = Path(tmpdir) / "cand.elle"
            write_elle_phasefield_state(ref, theta_a, temp_a)
            write_elle_phasefield_state(cand, theta_b, temp_b)
            report = compare_elle_phasefield_files(ref, cand)

        self.assertGreater(report["theta_rmse"], 0.0)
        self.assertGreater(report["temperature_rmse"], 0.0)
        self.assertLess(report["theta_solid_iou"], 1.0)

    def test_compare_elle_phasefield_sequences_matches_steps(self) -> None:
        theta = np.zeros((4, 4), dtype=np.float32)
        temperature = np.zeros((4, 4), dtype=np.float32)
        theta_next = theta.copy()
        theta_next[1, 1] = 1.0

        with tempfile.TemporaryDirectory() as tmpdir:
            ref_dir = Path(tmpdir) / "ref"
            cand_dir = Path(tmpdir) / "cand"
            ref_dir.mkdir()
            cand_dir.mkdir()
            write_elle_phasefield_state(ref_dir / "phasefield001.elle", theta, temperature)
            write_elle_phasefield_state(ref_dir / "phasefield002.elle", theta_next, temperature)
            write_elle_phasefield_state(cand_dir / "phasefield_state_00001.elle", theta, temperature)
            write_elle_phasefield_state(cand_dir / "phasefield_state_00002.elle", theta, temperature)
            report = compare_elle_phasefield_sequences(ref_dir, cand_dir)

        self.assertEqual(report["matched_steps"], [1, 2])
        self.assertEqual(report["missing_in_candidate"], [])
        self.assertEqual(report["missing_in_reference"], [])
        self.assertEqual(report["summary"]["num_matched_steps"], 2)
        self.assertGreater(report["summary"]["theta_rmse_max"], 0.0)

    def test_inspect_elle_phasefield_binary_reports_missing_file(self) -> None:
        report = inspect_elle_phasefield_binary(PROJECT_ROOT / "missing_elle_phasefield")

        self.assertFalse(report["exists"])
        self.assertFalse(report["ready"])
        self.assertEqual(report["missing_libraries"], [])

    def test_run_original_elle_phasefield_sequence_reports_unavailable_binary(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            outdir = Path(tmpdir) / "original"
            report = run_original_elle_phasefield_sequence(
                PROJECT_ROOT / "missing_elle_phasefield",
                PROJECT_ROOT.parent / "processes" / "phasefield" / "inphase.elle",
                outdir,
                steps=1,
                save_every=1,
            )

        self.assertFalse(report["ran"])
        self.assertEqual(report["snapshots"], [])
        self.assertFalse(report["binary_status"]["ready"])

    def test_initialization_is_normalized(self) -> None:
        cfg = GrainGrowthConfig(nx=16, ny=12, num_grains=4, seed=123)
        phi = np.asarray(initialize_order_parameters(cfg))
        sums = phi.sum(axis=0)

        self.assertEqual(phi.shape, (4, 16, 12))
        self.assertLess(float(((sums - 1.0) ** 2).mean()), 1e-10)

    def test_load_elle_label_seed_reads_structured_grid(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_periodic_flynn_example(Path(tmpdir) / "seed.elle")
            seed = load_elle_label_seed(elle_path)

        self.assertEqual(seed["grid_shape"], (4, 4))
        self.assertEqual(seed["num_labels"], 5)
        label_field = np.asarray(seed["label_field"])
        self.assertEqual(label_field.shape, (4, 4))
        self.assertEqual(len(seed["unode_positions"]), 16)
        self.assertEqual(len(seed["unode_grid_indices"]), 16)

    def test_load_elle_label_seed_auto_detects_fine_foam_grain_attribute(self) -> None:
        source = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )
        seed = load_elle_label_seed(source)

        self.assertEqual(seed["attribute"], "U_ATTRIB_C")
        self.assertGreaterEqual(seed["num_labels"], 130)
        self.assertEqual(tuple(seed["grid_shape"]), (128, 128))

    def test_load_elle_label_seed_derives_labels_from_raw_fine_foam_launch_seed(self) -> None:
        source = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "fine_foam.elle"
        )

        seed = load_elle_label_seed(source)

        self.assertEqual(seed["attribute"], "derived_from_flynns")
        self.assertEqual(tuple(seed["grid_shape"]), (128, 128))
        self.assertEqual(np.asarray(seed["label_field"]).shape, (128, 128))
        self.assertGreater(int(seed["num_labels"]), 0)
        self.assertEqual(len(seed["source_labels"]), int(seed["num_labels"]))
        self.assertEqual(tuple(seed["unode_field_order"]), ())

    def test_load_elle_label_seed_auto_prefers_u_attrib_c_when_integer_like(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = Path(tmpdir) / "auto_prefers_c.elle"
            elle_path.write_text(
                "\n".join(
                    [
                        "OPTIONS",
                        "LOCATION",
                        "1 0.0 0.0",
                        "2 1.0 0.0",
                        "3 1.0 1.0",
                        "4 0.0 1.0",
                        "FLYNNS",
                        "1 4 1 2 3 4",
                        "UNODES",
                        "1 0.25 0.25",
                        "2 0.25 0.75",
                        "3 0.75 0.25",
                        "4 0.75 0.75",
                        "U_ATTRIB_B",
                        "Default 0.0",
                        "1 10",
                        "2 20",
                        "U_ATTRIB_C",
                        "Default 0.0",
                        "1 100",
                        "2 100",
                        "3 200",
                        "4 200",
                        "",
                    ]
                ),
                encoding="utf-8",
            )
            seed = load_elle_label_seed(elle_path)

        self.assertEqual(seed["attribute"], "U_ATTRIB_C")

    def test_load_elle_mesh_seed_maps_original_flynns_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "mesh_seed.elle")
            label_seed = load_elle_label_seed(elle_path, attribute="U_ATTRIB_C")
            mesh_state, relax_overrides = load_elle_mesh_seed(elle_path, label_seed)

        self.assertEqual(label_seed["attribute"], "U_ATTRIB_C")
        self.assertEqual(mesh_state["stats"]["num_flynns"], 2)
        self.assertEqual(mesh_state["stats"]["mesh_seed_source"], "elle")
        self.assertAlmostEqual(relax_overrides["switch_distance"], 0.05)
        self.assertAlmostEqual(relax_overrides["min_node_separation_factor"], 1.0)
        self.assertAlmostEqual(relax_overrides["max_node_separation_factor"], 2.2)
        self.assertAlmostEqual(relax_overrides["time_step"], 3.0)
        self.assertAlmostEqual(relax_overrides["speed_up"], 2.0)
        self.assertAlmostEqual(mesh_state["stats"]["elle_option_temperature"], -10.0)
        self.assertAlmostEqual(mesh_state["stats"]["elle_option_pressure"], 8800000.0)
        self.assertAlmostEqual(mesh_state["stats"]["elle_option_unitlength"], 0.5)
        self.assertAlmostEqual(mesh_state["stats"]["elle_option_massincrement"], 0.02)
        self.assertEqual(
            sorted(flynn["label"] for flynn in mesh_state["flynns"]),
            [0, 1],
        )
        self.assertIn("_runtime_seed_unodes", mesh_state)
        self.assertEqual(mesh_state["_runtime_seed_unodes"]["grid_shape"], (2, 2))
        self.assertIn("_runtime_seed_unode_fields", mesh_state)
        self.assertEqual(mesh_state["_runtime_seed_unode_fields"]["label_attribute"], "U_ATTRIB_C")
        self.assertIn("U_ATTRIB_A", mesh_state["_runtime_seed_unode_fields"]["values"])
        self.assertGreater(mesh_state["_runtime_seed_unode_fields"]["roi"], 0.0)
        self.assertIn("_runtime_seed_unode_sections", mesh_state)
        self.assertIn("U_ATTRIB_A", mesh_state["_runtime_seed_unode_sections"]["values"])
        self.assertIn("_runtime_seed_node_fields", mesh_state)
        self.assertIn("N_ATTRIB_A", mesh_state["_runtime_seed_node_fields"]["values"])
        self.assertEqual(len(mesh_state["_runtime_seed_node_fields"]["values"]["N_ATTRIB_A"]), 6)
        self.assertIn("_runtime_seed_node_sections", mesh_state)
        self.assertIn("_runtime_seed_flynn_sections", mesh_state)
        self.assertIn("_runtime_elle_options", mesh_state)
        self.assertEqual(mesh_state["_runtime_elle_options"]["scalar_values"]["Temperature"], -10.0)

    def test_update_seed_unode_fields_skips_swept_geometry_when_only_generic_scalar_fields_exist(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "mesh_seed.elle")
            label_seed = load_elle_label_seed(elle_path, attribute="U_ATTRIB_C")
            mesh_state, _ = load_elle_mesh_seed(elle_path, label_seed)

        current_labels = np.asarray(label_seed["label_field"], dtype=np.int32)
        target_labels = current_labels.copy()
        target_labels[0, 0] = current_labels[0, 1]

        with patch("elle_jax_model.mesh._compute_swept_seed_unode_mask", side_effect=AssertionError("should not compute swept mask")):
            with patch("elle_jax_model.mesh._segment_swept_records", side_effect=AssertionError("should not build segment records")):
                updated_fields, stats, mass_ledgers = update_seed_unode_fields(
                    current_labels,
                    target_labels,
                    mesh_state,
                    mesh_state,
                    mesh_state["_runtime_seed_unodes"],
                    mesh_state["_runtime_seed_unode_fields"],
                    node_fields=mesh_state["_runtime_seed_node_fields"],
                )

        self.assertIn("U_ATTRIB_A", updated_fields)
        self.assertIn("U_ATTRIB_C", updated_fields)
        self.assertEqual(stats["mass_partitioned_fields"], 0)
        self.assertEqual(stats["scalar_swept_unodes"], 0)
        self.assertEqual(mass_ledgers, {})

    def test_initialize_order_parameters_supports_elle_seed(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_periodic_flynn_example(Path(tmpdir) / "seed.elle")
            seed = load_elle_label_seed(elle_path)
            cfg = GrainGrowthConfig(
                nx=seed["grid_shape"][0],
                ny=seed["grid_shape"][1],
                num_grains=seed["num_labels"],
                init_mode="elle",
                init_elle_path=str(elle_path),
                init_smoothing_steps=0,
                init_noise=0.0,
            )
            phi = np.asarray(initialize_order_parameters(cfg))

        self.assertEqual(phi.shape, (5, 4, 4))
        self.assertTrue(np.allclose(phi.sum(axis=0), 1.0, atol=1e-6))

    def test_voronoi_initialization_uses_all_grains(self) -> None:
        cfg = GrainGrowthConfig(
            nx=24,
            ny=18,
            num_grains=5,
            seed=7,
            init_mode="voronoi",
            init_smoothing_steps=1,
        )
        phi = np.asarray(initialize_order_parameters(cfg))
        labels = dominant_grain_map(phi)
        counts = np.bincount(labels.ravel(), minlength=cfg.num_grains)

        self.assertEqual(phi.shape, (5, 24, 18))
        self.assertTrue(np.all(counts > 0))

    def test_run_simulation_shapes(self) -> None:
        cfg = GrainGrowthConfig(
            nx=12,
            ny=8,
            num_grains=3,
            seed=1,
            init_mode="voronoi",
        )
        final_state, snapshots = run_simulation(cfg, steps=10, save_every=5)

        self.assertEqual(tuple(final_state.shape), (3, 12, 8))
        self.assertEqual(len(snapshots), 2)

    def test_artifact_export_writes_preview_and_stats(self) -> None:
        cfg = GrainGrowthConfig(nx=10, ny=6, num_grains=4, seed=2, init_mode="voronoi")
        phi = np.asarray(initialize_order_parameters(cfg))
        topology_snapshot = TopologyTracker().update(phi, step=5)

        with tempfile.TemporaryDirectory() as tmpdir:
            written = save_snapshot_artifacts(
                tmpdir,
                5,
                phi,
                save_elle=True,
                tracked_topology=topology_snapshot,
                save_topology=True,
            )
            stats_path = written["grain_stats"]

            self.assertTrue(written["order_parameter"].exists())
            self.assertTrue(written["grain_ids"].exists())
            self.assertTrue(written["boundary_mask"].exists())
            self.assertTrue(written["grain_preview"].exists())
            self.assertTrue(written["elle"].exists())
            self.assertTrue(written["topology"].exists())
            self.assertTrue(stats_path.exists())

            with stats_path.open("r", encoding="utf-8") as handle:
                stats = json.load(handle)

            self.assertEqual(stats["step"], 5)
            self.assertEqual(stats["num_grains"], 4)
            self.assertEqual(len(stats["grains"]), 4)
            self.assertGreaterEqual(stats["boundary_fraction"], 0.0)
            self.assertLessEqual(stats["boundary_fraction"], 1.0)

    def test_unode_elle_export_contains_expected_sections(self) -> None:
        cfg = GrainGrowthConfig(nx=8, ny=5, num_grains=3, seed=4, init_mode="voronoi")
        phi = np.asarray(initialize_order_parameters(cfg))
        tracked = TopologyTracker().update(phi, step=9)
        mesh_state = relax_mesh_state(
            build_mesh_state(phi, tracked_topology=tracked),
            MeshRelaxationConfig(steps=1, random_seed=0),
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = write_unode_elle(
                Path(tmpdir) / "snapshot.elle",
                phi,
                step=9,
                tracked_topology=tracked,
                mesh_state=mesh_state,
            )
            text = outpath.read_text(encoding="utf-8")

            self.assertIn("OPTIONS\n", text)
            self.assertIn("FLYNNS\n", text)
            self.assertIn("LOCATION\n", text)
            self.assertIn("UNODES\n", text)
            self.assertIn("F_ATTRIB_A\n", text)
            self.assertIn("U_ATTRIB_A\n", text)
            self.assertIn("U_ATTRIB_B\n", text)
            self.assertIn("# step=9", text)
            self.assertIn("# tracked_topology=1", text)
            self.assertIn("# mesh_relaxed=1", text)

    def test_write_unode_elle_preserves_seed_unode_fields_for_truthful_branch(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "mesh_seed.elle")
            label_seed = load_elle_label_seed(elle_path, attribute="U_ATTRIB_C")
            mesh_state, _ = load_elle_mesh_seed(elle_path, label_seed)
            labels = np.asarray(label_seed["label_field"], dtype=np.int32)
            phi = mesh_labels_to_order_parameters(labels, int(label_seed["num_labels"]))

            outpath = write_unode_elle(
                Path(tmpdir) / "truthful_snapshot.elle",
                phi,
                step=1,
                mesh_state=mesh_state,
            )
            text = outpath.read_text(encoding="utf-8")

        self.assertIn("UNODES\n", text)
        self.assertIn("N_ATTRIB_A\n", text)
        self.assertIn("U_ATTRIB_C\n", text)
        self.assertIn("U_ATTRIB_A\n", text)
        self.assertNotIn("U_ATTRIB_B\nDefault 0.00000000e+00\n0 ", text)
        self.assertIn("0 1.50000000e+00", text)
        self.assertIn("1 0.2500000000 0.2500000000", text)
        self.assertIn("1 1.00000000e+00", text)
        self.assertIn("1 1.00000000e+02", text)
        self.assertIn("Timestep 3.00000000e+00", text)
        self.assertIn("UnitLength 5.00000000e-01", text)
        self.assertIn("Temperature -1.00000000e+01", text)
        self.assertIn("Pressure 8.80000000e+06", text)
        self.assertIn("BoundaryWidth 1.00000000e-01", text)
        self.assertIn("MassIncrement 2.00000000e-02", text)

    def test_write_unode_elle_uses_runtime_temperature_override_in_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "mesh_seed.elle")
            setup = build_faithful_gbm_setup(elle_path, temperature_c=-5.0)
            labels = np.asarray(setup.seed_info.label_field, dtype=np.int32)
            phi = mesh_labels_to_order_parameters(labels, int(setup.seed_info.num_labels))

            outpath = write_unode_elle(
                Path(tmpdir) / "truthful_runtime_options.elle",
                phi,
                step=0,
                mesh_state=setup.mesh_seed,
            )
            exported = outpath.read_text(encoding="utf-8")

        self.assertIn("Temperature -5.00000000e+00", exported)
        self.assertIn("Timestep 3.00000000e+00", exported)
        self.assertIn("Pressure 8.80000000e+06", exported)

    def test_inifft001_seed_roundtrips_key_elle_options_without_override(self) -> None:
        elle_path = (
            Path("/home/bz1229682991/research/Elle/newcode/elle")
            / "processes"
            / "fft"
            / "example"
            / "step0"
            / "inifft001.elle"
        )
        setup = build_faithful_gbm_setup(elle_path)
        labels = np.asarray(setup.seed_info.label_field, dtype=np.int32)
        phi = mesh_labels_to_order_parameters(labels, int(setup.seed_info.num_labels))

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = write_unode_elle(
                Path(tmpdir) / "roundtrip.elle",
                phi,
                step=0,
                mesh_state=setup.mesh_seed,
            )
            exported = outpath.read_text(encoding="utf-8")

        self.assertIn("SwitchDistance 4.00000000e-03", exported)
        self.assertIn("Timestep 1.00000000e+03", exported)
        self.assertIn("UnitLength 1.00000000e-02", exported)
        self.assertIn("Temperature -1.00000000e+01", exported)
        self.assertIn("Pressure 1.00000000e+00", exported)
        self.assertIn("BoundaryWidth 1.00000000e-09", exported)
        self.assertIn("MassIncrement 0.00000000e+00", exported)

    def test_write_unode_elle_preserves_defaults_and_multicomponent_seed_sections(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "mesh_seed.elle")
            text = elle_path.read_text(encoding="utf-8")
            text = text.replace(
                "U_ATTRIB_A\nDefault 0.0\n1 1.0\n2 2.0\n3 3.0\n4 4.0\nU_ATTRIB_C",
                "U_ATTRIB_A\nDefault 3.0\n1 4.0\nU_EULER_3\nDefault 0.0 0.0 0.0\n1 10.0 20.0 30.0\n2 40.0 50.0 60.0\nU_ATTRIB_C",
            )
            elle_path.write_text(text, encoding="utf-8")
            label_seed = load_elle_label_seed(elle_path, attribute="U_ATTRIB_C")
            mesh_state, _ = load_elle_mesh_seed(elle_path, label_seed)
            labels = np.asarray(label_seed["label_field"], dtype=np.int32)
            phi = mesh_labels_to_order_parameters(labels, int(label_seed["num_labels"]))

            outpath = write_unode_elle(
                Path(tmpdir) / "truthful_multicomponent.elle",
                phi,
                step=0,
                mesh_state=mesh_state,
            )
            exported = outpath.read_text(encoding="utf-8")

        self.assertIn("U_ATTRIB_A\nDefault 3.00000000e+00\n", exported)
        self.assertIn("\n1 4.00000000e+00\n", exported)
        self.assertNotIn("\n2 3.00000000e+00\n", exported)
        self.assertIn("U_EULER_3\nDefault 0.00000000e+00 0.00000000e+00 0.00000000e+00\n", exported)
        self.assertIn("\n1 1.00000000e+01 2.00000000e+01 3.00000000e+01\n", exported)
        self.assertIn("\n2 4.00000000e+01 5.00000000e+01 6.00000000e+01\n", exported)

    def test_write_unode_elle_prefers_runtime_label_field_over_stale_section_values(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "mesh_seed.elle")
            label_seed = load_elle_label_seed(elle_path, attribute="U_ATTRIB_C")
            mesh_state, _ = load_elle_mesh_seed(elle_path, label_seed)
            labels = np.asarray(label_seed["label_field"], dtype=np.int32)
            phi = mesh_labels_to_order_parameters(labels, int(label_seed["num_labels"]))

            runtime_fields = dict(mesh_state["_runtime_seed_unode_fields"]["values"])
            runtime_fields["U_ATTRIB_C"] = tuple(900.0 + float(index) for index in range(len(runtime_fields["U_ATTRIB_C"])))
            mesh_state["_runtime_seed_unode_fields"] = {
                **mesh_state["_runtime_seed_unode_fields"],
                "values": runtime_fields,
            }

            runtime_sections = dict(mesh_state["_runtime_seed_unode_sections"]["values"])
            stale_values = tuple((111.0,) for _ in range(len(runtime_fields["U_ATTRIB_C"])))
            runtime_sections["U_ATTRIB_C"] = stale_values
            mesh_state["_runtime_seed_unode_sections"] = {
                **mesh_state["_runtime_seed_unode_sections"],
                "values": runtime_sections,
            }

            outpath = write_unode_elle(
                Path(tmpdir) / "truthful_label_override.elle",
                phi,
                step=0,
                mesh_state=mesh_state,
            )
            exported = outpath.read_text(encoding="utf-8")

        self.assertIn("U_ATTRIB_C\n", exported)
        self.assertIn("1 9.00000000e+02", exported)
        self.assertIn("2 9.01000000e+02", exported)
        self.assertNotIn("1 1.11000000e+02", exported)

    def test_write_unode_elle_dense_exports_fallback_unode_labels(self) -> None:
        labels = np.asarray([[0, 1], [1, 1]], dtype=np.int32)
        phi = mesh_labels_to_order_parameters(labels, 2)
        mesh_state = {
            "nodes": (
                {"node_id": 0, "x": 0.0, "y": 0.0},
                {"node_id": 1, "x": 1.0, "y": 0.0},
                {"node_id": 2, "x": 1.0, "y": 1.0},
            ),
            "flynns": (
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2]},
            ),
            "stats": {
                "num_flynns": 1,
                "holes_skipped": 0,
                "export_dense_unodes": 1,
            },
            "_runtime_seed_unodes": {
                "ids": (10, 11, 12),
                "positions": ((0.25, 0.75), (0.75, 0.75), (0.75, 0.25)),
                "grid_indices": ((0, 0), (1, 0), (1, 1)),
                "grid_shape": (2, 2),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "field_order": ("U_ATTRIB_C", "U_ATTRIB_A"),
                "source_labels": (100, 200),
                "values": {
                    "U_ATTRIB_C": (100.0, 200.0, 200.0),
                    "U_ATTRIB_A": (1.0, 2.0, 3.0),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "id_order": (10, 11, 12),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": ((100.0,), (200.0,), (200.0,)),
                },
            },
        }

        with tempfile.TemporaryDirectory() as tmpdir:
            outpath = write_unode_elle(
                Path(tmpdir) / "fallback_dense.elle",
                phi,
                step=1,
                mesh_state=mesh_state,
            )
            exported_seed = load_elle_label_seed(outpath, attribute="U_ATTRIB_C")
            exported_text = outpath.read_text(encoding="utf-8")

        np.testing.assert_array_equal(np.asarray(exported_seed["label_field"], dtype=np.int32), labels)
        self.assertEqual(len(exported_seed["unode_ids"]), 4)
        self.assertIn("0 0.2500000000 0.2500000000", exported_text)
        self.assertIn("3 0.7500000000 0.7500000000", exported_text)
        self.assertIn("U_ATTRIB_A\n", exported_text)
        self.assertIn("0 1.00000000e+00\n", exported_text)
        self.assertIn("3 3.00000000e+00\n", exported_text)

    def test_extract_flynn_topology_returns_shared_nodes(self) -> None:
        labels = np.array(
            [
                [0, 0],
                [1, 1],
            ],
            dtype=np.int32,
        )
        nodes, flynns, topology_stats = extract_flynn_topology(labels)

        self.assertEqual(len(flynns), 2)
        self.assertGreater(len(nodes), 0)
        self.assertEqual({flynn["label"] for flynn in flynns}, {0, 1})
        self.assertEqual(topology_stats["holes_skipped"], 0)
        shared_nodes = set(flynns[0]["node_ids"]) & set(flynns[1]["node_ids"])
        self.assertTrue(shared_nodes)

    def test_rasterize_mesh_labels_round_trips_simple_partition(self) -> None:
        labels = np.array(
            [
                [0, 0, 1],
                [0, 1, 1],
                [2, 2, 1],
            ],
            dtype=np.int32,
        )
        mesh_state = build_mesh_state(labels)
        rasterized = rasterize_mesh_labels(mesh_state, labels.shape, fallback_labels=labels)

        np.testing.assert_array_equal(rasterized, labels)

    def test_apply_mesh_feedback_moves_phi_toward_mesh_labels(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
                [0, 1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [0, 1, 1],
                [0, 1, 1],
                [0, 1, 1],
            ],
            dtype=np.int32,
        )
        phi = np.moveaxis(np.eye(2, dtype=np.float32)[current_labels], -1, 0)
        mesh_state = build_mesh_state(target_labels)

        phi_feedback, feedback_stats = apply_mesh_feedback(
            phi,
            mesh_state,
            strength=0.5,
            boundary_width=0,
        )

        self.assertGreater(feedback_stats["changed_pixels"], 0)
        self.assertGreater(phi_feedback[1, 0, 1], phi[1, 0, 1])
        self.assertAlmostEqual(float(phi_feedback[:, 0, 1].sum()), 1.0, places=6)

    def test_compute_mesh_motion_velocity_can_skip_dense_velocity_field(self) -> None:
        labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
        phi = mesh_labels_to_order_parameters(labels, num_grains=2)
        feedback = MeshFeedbackConfig(
            every=1,
            update_mode="mesh_only",
            relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
        )

        base_mesh_state, motion_mesh_state, velocity_field, transport_mask, stats = compute_mesh_motion_velocity(
            phi,
            feedback,
            compute_velocity_field=False,
        )

        self.assertEqual(np.asarray(velocity_field).shape, (2, 2, 2))
        self.assertFalse(np.any(np.asarray(velocity_field)))
        self.assertFalse(np.any(np.asarray(transport_mask)))
        self.assertTrue(bool(stats["velocity_field_skipped"]))
        self.assertEqual(base_mesh_state["stats"]["num_flynns"], motion_mesh_state["stats"]["num_flynns"])

    def test_mesh_feedback_preserves_inherited_relaxation_settings(self) -> None:
        labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
        phi = mesh_labels_to_order_parameters(labels, num_grains=2)
        feedback = MeshFeedbackConfig(
            every=1,
            strength=0.0,
            transport_strength=0.0,
            update_mode="mesh_only",
            relax_config=MeshRelaxationConfig(
                steps=1,
                topology_steps=1,
                time_step=864000.0,
                speed_up=1.0,
                switch_distance=0.005,
                min_angle_degrees=20.0,
                movement_model="elle_surface",
                use_diagonal_trials=True,
                use_elle_physical_units=True,
                boundary_energy=1.0,
                random_seed=0,
                min_node_separation_factor=1.0,
                max_node_separation_factor=2.2,
                temperature_c=-23.7,
                phase_db_path="/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/phase_db.txt",
            ),
        )
        seen_configs: list[MeshRelaxationConfig] = []
        original_relax = relax_mesh_state

        def capture_relax(mesh_state, config):
            seen_configs.append(config)
            return original_relax(mesh_state, config)

        with patch("elle_jax_model.mesh.relax_mesh_state", side_effect=capture_relax):
            _phi_feedback, _mesh_state, _feedback_stats = couple_mesh_to_order_parameters(
                phi,
                feedback,
            )

        self.assertGreaterEqual(len(seen_configs), 2)
        motion_config = seen_configs[0]
        topology_config = seen_configs[1]
        self.assertEqual(motion_config.temperature_c, -23.7)
        self.assertEqual(topology_config.temperature_c, -23.7)
        self.assertTrue(bool(motion_config.use_elle_physical_units))
        self.assertTrue(bool(topology_config.use_elle_physical_units))
        self.assertTrue(bool(motion_config.use_diagonal_trials))
        self.assertTrue(bool(topology_config.use_diagonal_trials))
        self.assertEqual(
            motion_config.phase_db_path,
            "/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/phase_db.txt",
        )
        self.assertEqual(
            topology_config.phase_db_path,
            "/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/phase_db.txt",
        )

    def test_relax_mesh_state_reuses_static_edge_context_without_topology_change(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.75, "y": 0.50},
                {"x": 0.60, "y": 0.70},
                {"x": 0.60, "y": 0.30},
            ],
            "flynns": [
                {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 2]},
                {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [1, 0, 3]},
            ],
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY", "EULER_3"),
                "id_order": (10, 20),
                "defaults": {
                    "VISCOSITY": (1.0,),
                    "EULER_3": (0.0, 0.0, 0.0),
                },
                "component_counts": {
                    "VISCOSITY": 1,
                    "EULER_3": 3,
                },
                "values": {
                    "VISCOSITY": ((1.0,), (2.0,)),
                    "EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
            "_runtime_seed_unodes": {
                "ids": (0, 1),
                "positions": ((0.62, 0.55), (0.62, 0.45)),
                "grid_indices": ((0, 0), (1, 0)),
                "grid_shape": (2, 1),
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_EULER_3",),
                "id_order": (0, 1),
                "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
                "component_counts": {"U_EULER_3": 3},
                "values": {
                    "U_EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
            "_runtime_elle_options": {
                "cell_bounding_box": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            },
            "stats": {"grid_shape": [256, 256]},
            "events": [],
        }

        original_move = mesh_module._move_node_elle_surface
        original_maintain = mesh_module._maintain_mesh_locally
        original_phase_lookup = mesh_module._build_flynn_phase_lookup
        original_boundary_lookup = mesh_module._build_edge_boundary_energy_lookup
        original_mobility_lookup = mesh_module._build_edge_mobility_lookup

        move_calls = 0
        phase_lookup_calls = 0
        boundary_lookup_calls = 0
        mobility_lookup_calls = 0

        def single_move(*args, **kwargs):
            nonlocal move_calls
            move_calls += 1
            if move_calls == 1:
                return np.array([1.0e-3, 0.0], dtype=np.float64)
            return np.zeros(2, dtype=np.float64)

        def no_change_maintain(*args, **kwargs):
            nodes, flynns = args[0], args[1]
            return (
                nodes,
                flynns,
                [],
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

        def count_phase_lookup(*args, **kwargs):
            nonlocal phase_lookup_calls
            phase_lookup_calls += 1
            return original_phase_lookup(*args, **kwargs)

        def count_boundary_lookup(*args, **kwargs):
            nonlocal boundary_lookup_calls
            boundary_lookup_calls += 1
            return original_boundary_lookup(*args, **kwargs)

        def count_mobility_lookup(*args, **kwargs):
            nonlocal mobility_lookup_calls
            mobility_lookup_calls += 1
            return original_mobility_lookup(*args, **kwargs)

        with patch.object(mesh_module, "_move_node_elle_surface", side_effect=single_move), \
             patch.object(mesh_module, "_maintain_mesh_locally", side_effect=no_change_maintain), \
             patch.object(mesh_module, "_build_flynn_phase_lookup", side_effect=count_phase_lookup), \
             patch.object(mesh_module, "_build_edge_boundary_energy_lookup", side_effect=count_boundary_lookup), \
             patch.object(mesh_module, "_build_edge_mobility_lookup", side_effect=count_mobility_lookup):
            relaxed = relax_mesh_state(
                mesh_state,
                MeshRelaxationConfig(
                    steps=1,
                    topology_steps=1,
                    movement_model="elle_surface",
                    use_diagonal_trials=True,
                    use_elle_physical_units=True,
                    switch_distance=0.05,
                    speed_up=1.0,
                ),
            )

        self.assertEqual(phase_lookup_calls, 1)
        self.assertEqual(boundary_lookup_calls, 1)
        self.assertGreaterEqual(mobility_lookup_calls, 2)
        self.assertEqual(relaxed["stats"]["num_flynns"], 2)

    def test_apply_mesh_transport_moves_interface(self) -> None:
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        phi = np.moveaxis(np.eye(2, dtype=np.float32)[labels], -1, 0)
        base_mesh = build_mesh_state(labels)
        moved_mesh = {
            "nodes": [dict(node) for node in base_mesh["nodes"]],
            "flynns": [
                {key: ([int(entry) for entry in value] if key == "node_ids" else value) for key, value in flynn.items()}
                for flynn in base_mesh["flynns"]
            ],
            "stats": dict(base_mesh["stats"]),
            "events": list(base_mesh.get("events", [])),
        }
        for node in moved_mesh["nodes"]:
            if node["node_id"] in {1, 2}:
                node["y"] = 0.60

        phi_transport, transport_stats = apply_mesh_transport(
            phi,
            base_mesh,
            moved_mesh,
            strength=1.0,
            boundary_width=1,
        )

        self.assertGreater(transport_stats["transport_pixels"], 0)
        self.assertGreater(phi_transport[1, 1, 1], phi[1, 1, 1])
        self.assertAlmostEqual(float(phi_transport[:, 1, 1].sum()), 1.0, places=6)

    def test_topology_tracker_keeps_stable_ids_for_unchanged_state(self) -> None:
        labels = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        tracker = TopologyTracker()
        first = tracker.update(labels, step=1)
        second = tracker.update(labels, step=2)

        self.assertEqual(
            [flynn["flynn_id"] for flynn in first["flynns"]],
            [flynn["flynn_id"] for flynn in second["flynns"]],
        )
        self.assertFalse(second["events"]["births"])
        self.assertFalse(second["events"]["deaths"])

    def test_topology_tracker_can_reuse_previous_snapshot_without_recomputing(self) -> None:
        labels = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        tracker = TopologyTracker()
        first = tracker.update(labels, step=1)
        reused = tracker.reuse_previous(step=2)

        self.assertEqual(reused["step"], 2)
        self.assertEqual(
            [flynn["flynn_id"] for flynn in first["flynns"]],
            [flynn["flynn_id"] for flynn in reused["flynns"]],
        )
        self.assertEqual(first["events"], reused["events"])
        self.assertEqual(len(tracker.history), 2)

    def test_topology_tracker_reports_split_events(self) -> None:
        tracker = TopologyTracker()
        tracker.update(
            np.array(
                [
                    [0, 0, 0],
                    [0, 0, 0],
                    [0, 0, 0],
                ],
                dtype=np.int32,
            ),
            step=1,
        )
        split_snapshot = tracker.update(
            np.array(
                [
                    [0, 1, 0],
                    [0, 1, 0],
                    [0, 1, 0],
                ],
                dtype=np.int32,
            ),
            step=2,
        )

        self.assertTrue(split_snapshot["events"]["splits"])
        split_event = split_snapshot["events"]["splits"][0]
        self.assertGreaterEqual(len(split_event["children"]), 2)
        self.assertTrue(any(not flynn["retained_identity"] for flynn in split_snapshot["flynns"]))

    def test_build_mesh_state_and_relaxation_preserve_structure(self) -> None:
        labels = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        tracked = TopologyTracker().update(labels, step=1)
        mesh_state = build_mesh_state(labels, tracked_topology=tracked)
        relaxed = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(steps=2, random_seed=3),
        )

        self.assertEqual(mesh_state["stats"]["num_flynns"], 2)
        self.assertEqual(relaxed["stats"]["num_flynns"], 2)
        self.assertEqual(len(mesh_state["nodes"]), len(relaxed["nodes"]))
        self.assertIn("mesh_relaxation_steps", relaxed["stats"])
        self.assertTrue(any(node["junction_type"] in {"double", "triple"} for node in relaxed["nodes"]))

    def test_relax_mesh_state_supports_elle_surface_motion_model(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.75, "y": 0.50},
                {"x": 0.60, "y": 0.70},
                {"x": 0.60, "y": 0.30},
            ],
            "flynns": [
                {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 2]},
                {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [1, 0, 3]},
            ],
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY",),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,)},
                "component_counts": {"VISCOSITY": 1},
                "values": {
                    "VISCOSITY": ((1.0,), (2.0,)),
                },
            },
            "stats": {"grid_shape": [256, 256]},
            "events": [],
        }

        relaxed = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=1,
                topology_steps=0,
                movement_model="elle_surface",
                use_diagonal_trials=True,
                use_elle_physical_units=True,
                switch_distance=0.05,
                speed_up=1.0,
            ),
        )

        moved = np.hypot(
            relaxed["nodes"][0]["x"] - mesh_state["nodes"][0]["x"],
            relaxed["nodes"][0]["y"] - mesh_state["nodes"][0]["y"],
        )
        self.assertGreater(moved, 0.0)
        self.assertEqual(relaxed["stats"]["mesh_movement_model"], "elle_surface")
        self.assertEqual(relaxed["stats"]["mesh_surface_diagonal_trials"], 1)
        self.assertEqual(relaxed["stats"]["mesh_surface_use_elle_physical_units"], 1)
        self.assertEqual(relaxed["stats"]["mesh_unit_length"], 1.0)
        self.assertEqual(relaxed["stats"]["mesh_boundary_energy"], 1.0)

    def test_surface_force_from_trial_energies_supports_diagonal_trials(self) -> None:
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.18, 0.42],
                [0.81, 0.56],
                [0.47, 0.84],
            ],
            dtype=np.float64,
        )
        ordered_neighbors = [1, 2, 3]
        switch_distance = 0.05

        force_cardinal = _surface_force_from_trial_energies(
            0,
            node_xy,
            ordered_neighbors,
            nodes,
            switch_distance=switch_distance,
            boundary_energy=1.0,
            use_diagonal_trials=False,
        )
        force_diagonal = _surface_force_from_trial_energies(
            0,
            node_xy,
            ordered_neighbors,
            nodes,
            switch_distance=switch_distance,
            boundary_energy=1.0,
            use_diagonal_trials=True,
        )

        plus_x = np.array([(node_xy[0] + switch_distance) % 1.0, node_xy[1]], dtype=np.float64)
        minus_x = np.array([(node_xy[0] - switch_distance) % 1.0, node_xy[1]], dtype=np.float64)
        plus_y = np.array([node_xy[0], (node_xy[1] + switch_distance) % 1.0], dtype=np.float64)
        minus_y = np.array([node_xy[0], (node_xy[1] - switch_distance) % 1.0], dtype=np.float64)
        diagonal_distance = switch_distance / np.sqrt(2.0)
        plus_plus = np.array(
            [(node_xy[0] + diagonal_distance) % 1.0, (node_xy[1] + diagonal_distance) % 1.0],
            dtype=np.float64,
        )
        minus_minus = np.array(
            [(node_xy[0] - diagonal_distance) % 1.0, (node_xy[1] - diagonal_distance) % 1.0],
            dtype=np.float64,
        )
        plus_minus = np.array(
            [(node_xy[0] + diagonal_distance) % 1.0, (node_xy[1] - diagonal_distance) % 1.0],
            dtype=np.float64,
        )
        minus_plus = np.array(
            [(node_xy[0] - diagonal_distance) % 1.0, (node_xy[1] + diagonal_distance) % 1.0],
            dtype=np.float64,
        )

        surface_delta_x = _trial_node_energy(plus_x, ordered_neighbors, nodes, boundary_energy=1.0) - _trial_node_energy(
            minus_x,
            ordered_neighbors,
            nodes,
            boundary_energy=1.0,
        )
        surface_delta_y = _trial_node_energy(plus_y, ordered_neighbors, nodes, boundary_energy=1.0) - _trial_node_energy(
            minus_y,
            ordered_neighbors,
            nodes,
            boundary_energy=1.0,
        )
        diagonal_delta = (
            _trial_node_energy(plus_plus, ordered_neighbors, nodes, boundary_energy=1.0)
            - _trial_node_energy(minus_minus, ordered_neighbors, nodes, boundary_energy=1.0)
            + _trial_node_energy(plus_minus, ordered_neighbors, nodes, boundary_energy=1.0)
            - _trial_node_energy(minus_plus, ordered_neighbors, nodes, boundary_energy=1.0)
        ) / np.sqrt(2.0)
        expected_force = np.array(
            [
                -((surface_delta_x + diagonal_delta) / 2.0) / (2.0 * switch_distance),
                -((surface_delta_y + diagonal_delta) / 2.0) / (2.0 * switch_distance),
            ],
            dtype=np.float64,
        )

        self.assertFalse(np.allclose(force_cardinal, force_diagonal))
        self.assertTrue(np.allclose(force_diagonal, expected_force))

    def test_elle_surface_effective_dt_resets_speedup_before_clamp(self) -> None:
        dt = _elle_surface_effective_dt(
            velocity_length=0.004,
            switch_distance=0.05,
            time_step=10.0,
            speed_up=2.0,
        )

        self.assertAlmostEqual(dt, 10.0)

    def test_elle_surface_velocity_from_force_uses_elle_denominator(self) -> None:
        force = np.array([3.0, 4.0], dtype=np.float64)
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.75, 0.50],
                [0.50, 0.80],
            ],
            dtype=np.float64,
        )
        ordered_neighbors = [1, 2]
        velocity = _elle_surface_velocity_from_force(
            force,
            node_xy,
            ordered_neighbors,
            nodes,
            unit_length=0.03,
            segment_mobilities=[2.0, 0.5],
        )

        force_unit = force / np.hypot(*force)
        cosalpha1 = max(abs(np.dot(np.array([0.0, -0.25]), force_unit) / 0.25), 0.01745)
        cosalpha2 = max(abs(np.dot(np.array([0.3, 0.0]), force_unit) / 0.3), 0.01745)
        aa = (((0.25 * 0.03) * (cosalpha1 ** 2)) / 2.0) + (((0.3 * 0.03) * (cosalpha2 ** 2)) / 0.5)
        expected = ((2.0 * force_unit * float(np.hypot(*force))) / aa) / 0.03

        self.assertTrue(np.allclose(velocity, expected))

    def test_elle_surface_velocity_from_force_scales_with_force_magnitude(self) -> None:
        force_a = np.array([3.0, 4.0], dtype=np.float64)
        force_b = 10.0 * force_a
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.75, 0.50],
                [0.50, 0.80],
            ],
            dtype=np.float64,
        )
        ordered_neighbors = [1, 2]

        velocity_a = _elle_surface_velocity_from_force(
            force_a,
            node_xy,
            ordered_neighbors,
            nodes,
            unit_length=0.03,
            segment_mobilities=[2.0, 0.5],
        )
        velocity_b = _elle_surface_velocity_from_force(
            force_b,
            node_xy,
            ordered_neighbors,
            nodes,
            unit_length=0.03,
            segment_mobilities=[2.0, 0.5],
        )

        self.assertTrue(np.allclose(velocity_b, 10.0 * velocity_a))

    def test_load_phase_boundary_db_reads_original_pair_data(self) -> None:
        phase_db = load_phase_boundary_db()

        self.assertEqual(phase_db.first_phase, 1)
        self.assertEqual(phase_db.no_phases, 2)
        self.assertAlmostEqual(phase_db.pairs[(1, 2)].mobility, 0.0032)
        self.assertAlmostEqual(phase_db.pairs[(1, 2)].boundary_energy, 0.52)
        self.assertAlmostEqual(phase_db.pairs[(1, 2)].activation_energy, 51.1e3)
        self.assertAlmostEqual(phase_db.cluster_multiplier_a, 0.001)
        self.assertAlmostEqual(phase_db.cluster_multiplier_d, 2.0)

    def test_boundary_segment_mobility_matches_old_arrhenius_formula(self) -> None:
        phase_db = load_phase_boundary_db()
        mobility = boundary_segment_mobility(
            phase_db,
            1,
            2,
            temperature_c=25.0,
        )
        expected = arrhenius_boundary_mobility(0.0032, 51.1e3, temperature_c=25.0)
        self.assertAlmostEqual(mobility, expected)

    def test_boundary_segment_energy_matches_phase_pair_data(self) -> None:
        phase_db = load_phase_boundary_db()
        self.assertAlmostEqual(boundary_segment_energy(phase_db, 1, 2), 0.52)

    def test_build_edge_mobility_lookup_uses_phase_pairs_and_misorientation(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.75, "y": 0.50},
                {"x": 0.60, "y": 0.70},
                {"x": 0.60, "y": 0.30},
            ],
            "flynns": [
                {
                    "flynn_id": 10,
                    "source_flynn_id": 10,
                    "label": 0,
                    "node_ids": [0, 1, 2],
                },
                {
                    "flynn_id": 20,
                    "source_flynn_id": 20,
                    "label": 1,
                    "node_ids": [1, 0, 3],
                },
            ],
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY", "EULER_3"),
                "id_order": (10, 20),
                "defaults": {
                    "VISCOSITY": (1.0,),
                    "EULER_3": (0.0, 0.0, 0.0),
                },
                "component_counts": {
                    "VISCOSITY": 1,
                    "EULER_3": 3,
                },
                "values": {
                    "VISCOSITY": ((1.0,), (2.0,)),
                    "EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
        }
        edge_map = {
            (0, 1): [(0, 0), (1, 0)],
        }

        lookup = _build_edge_mobility_lookup(
            mesh_state,
            mesh_state["flynns"],
            edge_map,
            phase_db_path=None,
            temperature_c=25.0,
        )

        phase_db = load_phase_boundary_db()
        misorientation = caxis_misorientation_degrees((0.0, 0.0, 0.0), (0.0, 5.0, 0.0))
        expected = boundary_segment_mobility(
            phase_db,
            1,
            2,
            temperature_c=25.0,
            misorientation_degrees=misorientation,
        )
        self.assertAlmostEqual(lookup[(0, 1)], expected)
        self.assertLess(lookup[(0, 1)], boundary_segment_mobility(phase_db, 1, 2, temperature_c=25.0))

    def test_build_edge_boundary_energy_lookup_uses_phase_pairs(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.75, "y": 0.50},
                {"x": 0.60, "y": 0.70},
                {"x": 0.60, "y": 0.30},
            ],
            "flynns": [
                {
                    "flynn_id": 10,
                    "source_flynn_id": 10,
                    "label": 0,
                    "node_ids": [0, 1, 2],
                },
                {
                    "flynn_id": 20,
                    "source_flynn_id": 20,
                    "label": 1,
                    "node_ids": [1, 0, 3],
                },
            ],
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY",),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,)},
                "component_counts": {"VISCOSITY": 1},
                "values": {
                    "VISCOSITY": ((1.0,), (2.0,)),
                },
            },
        }
        edge_map = {
            (0, 1): [(0, 0), (1, 0)],
        }

        lookup = _build_edge_boundary_energy_lookup(
            mesh_state,
            mesh_state["flynns"],
            edge_map,
            phase_db_path=None,
            default_boundary_energy=1.0,
        )

        self.assertAlmostEqual(lookup[(0, 1)], 0.52)

    def test_build_edge_mobility_lookup_prefers_unode_euler3_midpoint_sampling(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.75, "y": 0.50},
                {"x": 0.60, "y": 0.70},
                {"x": 0.60, "y": 0.30},
            ],
            "flynns": [
                {
                    "flynn_id": 10,
                    "source_flynn_id": 10,
                    "label": 0,
                    "node_ids": [0, 1, 2],
                },
                {
                    "flynn_id": 20,
                    "source_flynn_id": 20,
                    "label": 1,
                    "node_ids": [1, 0, 3],
                },
            ],
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY", "EULER_3"),
                "id_order": (10, 20),
                "defaults": {
                    "VISCOSITY": (1.0,),
                    "EULER_3": (0.0, 0.0, 0.0),
                },
                "component_counts": {
                    "VISCOSITY": 1,
                    "EULER_3": 3,
                },
                "values": {
                    "VISCOSITY": ((1.0,), (2.0,)),
                    "EULER_3": ((0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
                },
            },
            "_runtime_seed_unodes": {
                "ids": (0, 1),
                "positions": ((0.62, 0.55), (0.62, 0.45)),
                "grid_indices": ((0, 0), (1, 0)),
                "grid_shape": (2, 1),
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_EULER_3",),
                "id_order": (0, 1),
                "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
                "component_counts": {"U_EULER_3": 3},
                "values": {
                    "U_EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
            "_runtime_elle_options": {
                "cell_bounding_box": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            },
        }
        edge_map = {
            (0, 1): [(0, 0), (1, 0)],
        }

        lookup = _build_edge_mobility_lookup(
            mesh_state,
            mesh_state["flynns"],
            edge_map,
            phase_db_path=None,
            temperature_c=25.0,
        )

        phase_db = load_phase_boundary_db()
        expected = boundary_segment_mobility(
            phase_db,
            1,
            2,
            temperature_c=25.0,
            misorientation_degrees=caxis_misorientation_degrees((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
        )
        self.assertAlmostEqual(lookup[(0, 1)], expected)
        self.assertGreater(lookup[(0, 1)], 0.0)

    def test_build_edge_mobility_lookup_uses_live_nodes_when_provided(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.10, "y": 0.10},
                {"x": 0.20, "y": 0.10},
            ],
            "flynns": [
                {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 0]},
                {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [1, 0, 1]},
            ],
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY", "EULER_3"),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,), "EULER_3": (0.0, 0.0, 0.0)},
                "component_counts": {"VISCOSITY": 1, "EULER_3": 3},
                "values": {
                    "VISCOSITY": ((1.0,), (2.0,)),
                    "EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
        }
        edge_map = {
            (0, 1): [(0, 0), (1, 0)],
        }
        live_nodes = np.array(
            [
                [0.60, 0.60],
                [0.80, 0.60],
            ],
            dtype=np.float64,
        )

        def midpoint_sensitive_euler_lookup(_mesh_state, _flynns, *, nodes=None):
            self.assertIsNotNone(nodes)
            np.testing.assert_allclose(np.asarray(nodes, dtype=np.float64), live_nodes)
            return {
                "search_roi": 1.0,
                "all_positions": np.asarray([[0.70, 0.60], [0.70, 0.60]], dtype=np.float64),
                "all_eulers": np.asarray([[0.0, 0.0, 0.0], [0.0, 5.0, 0.0]], dtype=np.float64),
                "by_flynn": {
                    10: {
                        "positions": np.asarray([[0.70, 0.60]], dtype=np.float64),
                        "eulers": np.asarray([[0.0, 0.0, 0.0]], dtype=np.float64),
                    },
                    20: {
                        "positions": np.asarray([[0.70, 0.60]], dtype=np.float64),
                        "eulers": np.asarray([[0.0, 5.0, 0.0]], dtype=np.float64),
                    },
                },
            }

        with patch.object(mesh_module, "_build_flynn_unode_euler_lookup", side_effect=midpoint_sensitive_euler_lookup):
            lookup = _build_edge_mobility_lookup(
                mesh_state,
                mesh_state["flynns"],
                edge_map,
                phase_db_path=None,
                temperature_c=25.0,
                nodes=live_nodes,
            )

        self.assertGreater(lookup[(0, 1)], 0.0)

    def test_seed_projection_cache_is_reused_across_mobility_and_stored_energy_setup(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.75, "y": 0.50},
                {"x": 0.60, "y": 0.70},
                {"x": 0.60, "y": 0.30},
            ],
            "flynns": [
                {
                    "flynn_id": 10,
                    "source_flynn_id": 10,
                    "label": 0,
                    "node_ids": [0, 1, 2],
                },
                {
                    "flynn_id": 20,
                    "source_flynn_id": 20,
                    "label": 1,
                    "node_ids": [1, 0, 3],
                },
            ],
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY", "EULER_3"),
                "id_order": (10, 20),
                "defaults": {
                    "VISCOSITY": (1.0,),
                    "EULER_3": (0.0, 0.0, 0.0),
                },
                "component_counts": {
                    "VISCOSITY": 1,
                    "EULER_3": 3,
                },
                "values": {
                    "VISCOSITY": ((1.0,), (2.0,)),
                    "EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
            "_runtime_seed_unodes": {
                "ids": (0, 1),
                "positions": ((0.62, 0.55), (0.62, 0.45)),
                "grid_indices": ((0, 0), (1, 0)),
                "grid_shape": (2, 1),
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_EULER_3",),
                "id_order": (0, 1),
                "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
                "component_counts": {"U_EULER_3": 3},
                "values": {
                    "U_EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
            "_runtime_seed_unode_fields": {
                "values": {"U_DISLOCDEN": (1.0, 2.0)},
            },
            "_runtime_elle_options": {
                "cell_bounding_box": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            },
            "stats": {"grid_shape": [2, 1]},
        }
        flynns = mesh_state["flynns"]
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.75, 0.50],
                [0.60, 0.70],
                [0.60, 0.30],
            ],
            dtype=np.float64,
        )
        edge_map = {(0, 1): [(0, 0), (1, 0)]}

        original_assign = mesh_module.assign_seed_unodes_from_mesh
        call_count = 0

        def counting_assign(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            return original_assign(*args, **kwargs)

        with patch("elle_jax_model.mesh.assign_seed_unodes_from_mesh", side_effect=counting_assign):
            _lookup = _build_edge_mobility_lookup(
                mesh_state,
                flynns,
                edge_map,
                phase_db_path=None,
                temperature_c=25.0,
            )
            _stored = mesh_module._build_stored_energy_context(
                mesh_state,
                nodes,
                flynns,
            )

        self.assertEqual(call_count, 1)
        self.assertIn("_runtime_seed_projection_cache", mesh_state)

    def test_edge_lookup_helpers_can_reuse_shared_phase_context(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.75, "y": 0.50},
                {"x": 0.60, "y": 0.70},
                {"x": 0.60, "y": 0.30},
            ],
            "flynns": [
                {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 2]},
                {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [1, 0, 3]},
            ],
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY", "EULER_3"),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,), "EULER_3": (0.0, 0.0, 0.0)},
                "component_counts": {"VISCOSITY": 1, "EULER_3": 3},
                "values": {
                    "VISCOSITY": ((1.0,), (2.0,)),
                    "EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
            "_runtime_seed_unodes": {
                "ids": (0, 1),
                "positions": ((0.62, 0.55), (0.62, 0.45)),
                "grid_indices": ((0, 0), (1, 0)),
                "grid_shape": (2, 1),
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_EULER_3",),
                "id_order": (0, 1),
                "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
                "component_counts": {"U_EULER_3": 3},
                "values": {
                    "U_EULER_3": ((0.0, 0.0, 0.0), (0.0, 5.0, 0.0)),
                },
            },
            "_runtime_elle_options": {
                "cell_bounding_box": [
                    [0.0, 0.0],
                    [1.0, 0.0],
                    [1.0, 1.0],
                    [0.0, 1.0],
                ],
            },
        }
        flynns = mesh_state["flynns"]
        edge_map = {(0, 1): [(0, 0), (1, 0)]}
        shared_phase_db = SimpleNamespace(first_phase=1)
        shared_phase_lookup = {10: 1, 20: 2}

        with patch.object(mesh_module, "load_phase_boundary_db", side_effect=AssertionError("should not reload phase db")), \
             patch.object(mesh_module, "_build_flynn_phase_lookup", side_effect=AssertionError("should not rebuild phase lookup")), \
             patch.object(mesh_module, "_build_flynn_euler_lookup", return_value={10: (0.0, 0.0, 0.0), 20: (0.0, 5.0, 0.0)}), \
             patch.object(mesh_module, "_build_flynn_unode_euler_lookup", return_value=None), \
             patch.object(mesh_module, "boundary_segment_mobility", return_value=1.25) as mobility_mock, \
             patch.object(mesh_module, "boundary_segment_energy", return_value=0.75) as energy_mock:
            mobility_lookup = mesh_module._build_edge_mobility_lookup(
                mesh_state,
                flynns,
                edge_map,
                phase_db_path=None,
                temperature_c=25.0,
                phase_db=shared_phase_db,
                phase_lookup=shared_phase_lookup,
            )
            energy_lookup = mesh_module._build_edge_boundary_energy_lookup(
                mesh_state,
                flynns,
                edge_map,
                phase_db_path=None,
                default_boundary_energy=1.0,
                phase_db=shared_phase_db,
                phase_lookup=shared_phase_lookup,
            )

        self.assertEqual(mobility_lookup[(0, 1)], 1.25)
        self.assertEqual(energy_lookup[(0, 1)], 0.75)
        mobility_mock.assert_called_once()
        energy_mock.assert_called_once()

    def test_misorientation_mobility_reduction_matches_holm_style_cutoff(self) -> None:
        self.assertAlmostEqual(misorientation_mobility_reduction(25.0), 1.0)
        self.assertLess(misorientation_mobility_reduction(5.0), 1.0)

    def test_elle_surface_mobility_changes_speed_not_direction(self) -> None:
        force = np.array([3.0, 4.0], dtype=np.float64)
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.75, 0.50],
                [0.50, 0.80],
            ],
            dtype=np.float64,
        )
        ordered_neighbors = [1, 2]
        phase_db = load_phase_boundary_db()

        hot_mobility = boundary_segment_mobility(phase_db, 1, 2, temperature_c=25.0)
        cold_mobility = boundary_segment_mobility(phase_db, 1, 2, temperature_c=-10.0)
        high_angle_mobility = boundary_segment_mobility(
            phase_db,
            1,
            2,
            temperature_c=25.0,
            misorientation_degrees=25.0,
        )
        low_angle_mobility = boundary_segment_mobility(
            phase_db,
            1,
            2,
            temperature_c=25.0,
            misorientation_degrees=5.0,
        )

        hot_velocity = _elle_surface_velocity_from_force(
            force,
            node_xy,
            ordered_neighbors,
            nodes,
            unit_length=0.03,
            segment_mobilities=[hot_mobility, hot_mobility],
        )
        cold_velocity = _elle_surface_velocity_from_force(
            force,
            node_xy,
            ordered_neighbors,
            nodes,
            unit_length=0.03,
            segment_mobilities=[cold_mobility, cold_mobility],
        )
        high_angle_velocity = _elle_surface_velocity_from_force(
            force,
            node_xy,
            ordered_neighbors,
            nodes,
            unit_length=0.03,
            segment_mobilities=[high_angle_mobility, high_angle_mobility],
        )
        low_angle_velocity = _elle_surface_velocity_from_force(
            force,
            node_xy,
            ordered_neighbors,
            nodes,
            unit_length=0.03,
            segment_mobilities=[low_angle_mobility, low_angle_mobility],
        )

        hot_direction = hot_velocity / np.hypot(*hot_velocity)
        cold_direction = cold_velocity / np.hypot(*cold_velocity)
        high_angle_direction = high_angle_velocity / np.hypot(*high_angle_velocity)
        low_angle_direction = low_angle_velocity / np.hypot(*low_angle_velocity)

        self.assertTrue(np.allclose(hot_direction, cold_direction))
        self.assertTrue(np.allclose(high_angle_direction, low_angle_direction))
        self.assertGreater(float(np.hypot(*hot_velocity)), float(np.hypot(*cold_velocity)))
        self.assertGreater(float(np.hypot(*high_angle_velocity)), float(np.hypot(*low_angle_velocity)))

    def test_surface_force_from_trial_energies_uses_segment_boundary_energies(self) -> None:
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.80, 0.55],
                [0.35, 0.80],
            ],
            dtype=np.float64,
        )
        ordered_neighbors = [1, 2]
        switch_distance = 0.05
        segment_boundary_energies = [2.0, 0.5]

        force = _surface_force_from_trial_energies(
            0,
            node_xy,
            ordered_neighbors,
            nodes,
            switch_distance=switch_distance,
            boundary_energy=1.0,
            segment_boundary_energies=segment_boundary_energies,
            use_diagonal_trials=False,
        )
        scalar_force = _surface_force_from_trial_energies(
            0,
            node_xy,
            ordered_neighbors,
            nodes,
            switch_distance=switch_distance,
            boundary_energy=1.0,
            use_diagonal_trials=False,
        )
        plus_x = node_xy + np.array([switch_distance, 0.0], dtype=np.float64)
        minus_x = node_xy + np.array([-switch_distance, 0.0], dtype=np.float64)
        plus_y = node_xy + np.array([0.0, switch_distance], dtype=np.float64)
        minus_y = node_xy + np.array([0.0, -switch_distance], dtype=np.float64)
        expected = np.array(
            [
                -(
                    _trial_node_energy(
                        plus_x,
                        ordered_neighbors,
                        nodes,
                        boundary_energy=1.0,
                        segment_boundary_energies=segment_boundary_energies,
                    )
                    - _trial_node_energy(
                        minus_x,
                        ordered_neighbors,
                        nodes,
                        boundary_energy=1.0,
                        segment_boundary_energies=segment_boundary_energies,
                    )
                )
                / (2.0 * switch_distance),
                -(
                    _trial_node_energy(
                        plus_y,
                        ordered_neighbors,
                        nodes,
                        boundary_energy=1.0,
                        segment_boundary_energies=segment_boundary_energies,
                    )
                    - _trial_node_energy(
                        minus_y,
                        ordered_neighbors,
                        nodes,
                        boundary_energy=1.0,
                        segment_boundary_energies=segment_boundary_energies,
                    )
                )
                / (2.0 * switch_distance),
            ],
            dtype=np.float64,
        )

        self.assertTrue(np.allclose(force, expected))
        self.assertFalse(np.allclose(force, scalar_force))

    def test_move_node_elle_surface_matches_getmovedir_axis_gate(self) -> None:
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.30, 0.40],
                [0.30, 0.60],
            ],
            dtype=np.float64,
        )
        ordered_neighbors = [1, 2]

        force = _surface_force_from_trial_energies(
            0,
            node_xy,
            ordered_neighbors,
            nodes,
            switch_distance=0.05,
            boundary_energy=1.0,
            use_diagonal_trials=False,
        )
        increment = _move_node_elle_surface(
            0,
            node_xy,
            ordered_neighbors,
            nodes,
            switch_distance=0.05,
            speed_up=1.0,
            boundary_energy=1.0,
            use_diagonal_trials=False,
        )

        self.assertGreater(abs(float(force[0])), 1.0e-6)
        self.assertAlmostEqual(float(force[1]), 0.0, places=12)
        self.assertTrue(np.allclose(increment, np.zeros(2, dtype=np.float64)))

    def test_surface_force_from_trial_energies_can_include_stored_energy_term(self) -> None:
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.50, 0.80],
                [0.50, 0.20],
                [0.90, 0.80],
                [0.90, 0.20],
                [0.10, 0.20],
                [0.10, 0.80],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 3, 4, 2]},
            {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [0, 2, 5, 6, 1]},
        ]
        seed_positions = []
        seed_grid_indices = []
        density_values = []
        for ix in range(4):
            for iy in range(4):
                x = (ix + 0.5) / 4.0
                y = 1.0 - (iy + 0.5) / 4.0
                seed_positions.append((x, y))
                seed_grid_indices.append((ix, iy))
                density_values.append(10.0 if x > 0.5 else 1.0)
        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(seed_positions),
                "grid_indices": tuple(seed_grid_indices),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {
                "values": {
                    "U_DISLOCDEN": tuple(density_values),
                },
            },
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY",),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,)},
                "component_counts": {"VISCOSITY": 1},
                "values": {
                    "VISCOSITY": ((1.0,), (1.0,)),
                },
            },
            "stats": {"grid_shape": [4, 4]},
        }
        stored_energy_context = mesh_module._build_stored_energy_context(mesh_state, nodes, flynns)

        force = _surface_force_from_trial_energies(
            0,
            node_xy,
            [1, 2],
            nodes,
            flynns=flynns,
            switch_distance=0.05,
            boundary_energy=0.0,
            use_diagonal_trials=False,
            stored_energy_context=stored_energy_context,
        )

        self.assertGreater(float(force[0]), 0.0)

    def test_surface_force_from_trial_energies_equal_density_collapses_to_curvature_only(self) -> None:
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.30, 0.40],
                [0.30, 0.60],
                [0.90, 0.60],
                [0.90, 0.40],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 2, 4, 3]},
            {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [0, 3, 4, 2, 1]},
        ]
        seed_positions = []
        seed_grid_indices = []
        density_values = []
        for ix in range(4):
            for iy in range(4):
                x = (ix + 0.5) / 4.0
                y = 1.0 - (iy + 0.5) / 4.0
                seed_positions.append((x, y))
                seed_grid_indices.append((ix, iy))
                density_values.append(5.0)
        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(seed_positions),
                "grid_indices": tuple(seed_grid_indices),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {"values": {"U_DISLOCDEN": tuple(density_values)}},
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY",),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,)},
                "component_counts": {"VISCOSITY": 1},
                "values": {"VISCOSITY": ((1.0,), (1.0,))},
            },
            "stats": {"grid_shape": [4, 4]},
        }
        stored_energy_context = mesh_module._build_stored_energy_context(mesh_state, nodes, flynns)

        curvature_force = _surface_force_from_trial_energies(
            0,
            node_xy,
            [1, 2],
            nodes,
            switch_distance=0.05,
            boundary_energy=1.0,
            use_diagonal_trials=False,
        )
        mixed_force = _surface_force_from_trial_energies(
            0,
            node_xy,
            [1, 2],
            nodes,
            flynns=flynns,
            switch_distance=0.05,
            boundary_energy=1.0,
            use_diagonal_trials=False,
            stored_energy_context=stored_energy_context,
        )

        np.testing.assert_allclose(mixed_force, curvature_force, atol=1.0e-9)

    def test_surface_force_from_trial_energies_combines_curvature_and_stored_energy_terms(self) -> None:
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.30, 0.40],
                [0.30, 0.60],
                [0.90, 0.60],
                [0.90, 0.40],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 2, 4, 3]},
            {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [0, 3, 4, 2, 1]},
        ]
        seed_positions = []
        seed_grid_indices = []
        density_values = []
        for ix in range(4):
            for iy in range(4):
                x = (ix + 0.5) / 4.0
                y = 1.0 - (iy + 0.5) / 4.0
                seed_positions.append((x, y))
                seed_grid_indices.append((ix, iy))
                density_values.append(10.0 if x > 0.5 else 1.0)
        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(seed_positions),
                "grid_indices": tuple(seed_grid_indices),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {"values": {"U_DISLOCDEN": tuple(density_values)}},
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY",),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,)},
                "component_counts": {"VISCOSITY": 1},
                "values": {"VISCOSITY": ((1.0,), (1.0,))},
            },
            "stats": {"grid_shape": [4, 4]},
        }
        stored_energy_context = mesh_module._build_stored_energy_context(mesh_state, nodes, flynns)

        curvature_force = _surface_force_from_trial_energies(
            0,
            node_xy,
            [1, 2],
            nodes,
            switch_distance=0.05,
            boundary_energy=1.0,
            use_diagonal_trials=False,
        )
        stored_force = _surface_force_from_trial_energies(
            0,
            node_xy,
            [1, 2],
            nodes,
            flynns=flynns,
            switch_distance=0.05,
            boundary_energy=0.0,
            use_diagonal_trials=False,
            stored_energy_context=stored_energy_context,
        )
        mixed_force = _surface_force_from_trial_energies(
            0,
            node_xy,
            [1, 2],
            nodes,
            flynns=flynns,
            switch_distance=0.05,
            boundary_energy=1.0,
            use_diagonal_trials=False,
            stored_energy_context=stored_energy_context,
        )

        np.testing.assert_allclose(mixed_force, curvature_force + stored_force, atol=1.0e-9)

    def test_roi_weighted_flynn_density_prefers_target_flynn_support_before_global_fallback(self) -> None:
        flynn_density_index = {
            20: {
                "positions": np.array([[0.55, 0.50], [0.75, 0.50]], dtype=np.float64),
                "densities": np.array([1.0, 5.0], dtype=np.float64),
                "mean_density": 3.0,
            },
        }

        density = mesh_module._roi_weighted_flynn_density(
            np.array([0.50, 0.50], dtype=np.float64),
            20,
            flynn_density_index,
            roi=0.40,
            dummy_density=50.0,
            all_positions=np.array([[0.55, 0.50], [0.75, 0.50], [0.52, 0.52]], dtype=np.float64),
            all_densities=np.array([1.0, 5.0, 20.0], dtype=np.float64),
        )

        self.assertLess(float(density), 20.0)
        self.assertGreater(float(density), 1.0)

    def test_stored_energy_density_roi_matches_legacy_fs_getroi_formula(self) -> None:
        seed_unodes = {
            "positions": (
                (0.25, 0.75),
                (0.25, 0.25),
                (0.75, 0.75),
                (0.75, 0.25),
            ),
        }
        runtime_elle_options = {
            "cell_bounding_box": [
                [0.0, 0.0],
                [2.0, 0.0],
                [2.0, 1.0],
                [0.0, 1.0],
            ]
        }

        roi = mesh_module._stored_energy_density_roi(
            seed_unodes,
            runtime_elle_options=runtime_elle_options,
        )

        self.assertAlmostEqual(float(roi), 3.0 / np.sqrt(np.pi), places=8)

    def test_legacy_trial_stored_energy_uses_incident_max_energy_when_trial_hits_no_neighboring_flynn(self) -> None:
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.50, 0.80],
                [0.50, 0.20],
                [0.90, 0.80],
                [0.90, 0.20],
                [0.10, 0.20],
                [0.10, 0.80],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 3, 4, 2]},
            {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [0, 2, 5, 6, 1]},
        ]
        seed_positions = ((0.55, 0.50),)
        seed_grid_indices = ((0, 0),)
        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": (0,),
                "positions": seed_positions,
                "grid_indices": seed_grid_indices,
                "grid_shape": (1, 1),
            },
            "_runtime_seed_unode_fields": {"values": {"U_DISLOCDEN": (2.0,)}},
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY",),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,)},
                "component_counts": {"VISCOSITY": 1},
                "values": {"VISCOSITY": ((1.0,), (2.0,))},
            },
            "stats": {"grid_shape": [1, 1]},
        }
        stored_energy_context = mesh_module._build_stored_energy_context(mesh_state, nodes, flynns)
        stored_energy_context["phase_db"] = SimpleNamespace(
            first_phase=1,
            phase_properties={
                1: SimpleNamespace(stored_energy=2.0, disscale=1.0, disbondscale=1.0),
                2: SimpleNamespace(stored_energy=5.0, disscale=1.0, disbondscale=1.0),
            },
        )
        stored_energy_context["phase_lookup"] = {10: 1, 20: 2}

        energy = mesh_module._legacy_trial_stored_energy(
            0,
            node_xy,
            np.array([0.80, 0.50], dtype=np.float64),
            np.array([[0.50, 0.80], [0.50, 0.20]], dtype=np.float64),
            nodes,
            [10, 20],
            flynns,
            stored_energy_context,
        )

        self.assertGreater(float(energy), 0.0)

    def test_legacy_trial_stored_energy_uses_disscale_only_as_gate_for_same_phase(self) -> None:
        node_xy = np.array([0.50, 0.50], dtype=np.float64)
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.30, 0.40],
                [0.30, 0.60],
                [0.90, 0.60],
                [0.90, 0.40],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 2, 4, 3]},
            {"flynn_id": 20, "source_flynn_id": 20, "label": 1, "node_ids": [0, 3, 4, 2, 1]},
        ]
        seed_positions = []
        seed_grid_indices = []
        density_values = []
        for ix in range(4):
            for iy in range(4):
                x = (ix + 0.5) / 4.0
                y = 1.0 - (iy + 0.5) / 4.0
                seed_positions.append((x, y))
                seed_grid_indices.append((ix, iy))
                density_values.append(10.0 if x > 0.5 else 1.0)
        mesh_state = {
            "_runtime_seed_unodes": {
                "ids": tuple(range(16)),
                "positions": tuple(seed_positions),
                "grid_indices": tuple(seed_grid_indices),
                "grid_shape": (4, 4),
            },
            "_runtime_seed_unode_fields": {"values": {"U_DISLOCDEN": tuple(density_values)}},
            "_runtime_seed_flynn_sections": {
                "field_order": ("VISCOSITY",),
                "id_order": (10, 20),
                "defaults": {"VISCOSITY": (1.0,)},
                "component_counts": {"VISCOSITY": 1},
                "values": {"VISCOSITY": ((1.0,), (1.0,))},
            },
            "stats": {"grid_shape": [4, 4]},
        }
        stored_energy_context = mesh_module._build_stored_energy_context(mesh_state, nodes, flynns)
        stored_energy_context["phase_lookup"] = {10: 1, 20: 1}

        stored_energy_context["phase_db"] = SimpleNamespace(
            first_phase=1,
            phase_properties={
                1: SimpleNamespace(stored_energy=2.0, disscale=1.0, disbondscale=0.25),
            },
        )
        full_energy = mesh_module._legacy_trial_stored_energy(
            0,
            node_xy,
            np.array([0.55, 0.50], dtype=np.float64),
            np.array([[0.30, 0.40], [0.30, 0.60]], dtype=np.float64),
            nodes,
            [10, 20],
            flynns,
            stored_energy_context,
        )

        stored_energy_context["phase_db"] = SimpleNamespace(
            first_phase=1,
            phase_properties={
                1: SimpleNamespace(stored_energy=2.0, disscale=0.25, disbondscale=0.25),
            },
        )
        gated_energy = mesh_module._legacy_trial_stored_energy(
            0,
            node_xy,
            np.array([0.55, 0.50], dtype=np.float64),
            np.array([[0.30, 0.40], [0.30, 0.60]], dtype=np.float64),
            nodes,
            [10, 20],
            flynns,
            stored_energy_context,
        )

        stored_energy_context["phase_db"] = SimpleNamespace(
            first_phase=1,
            phase_properties={
                1: SimpleNamespace(stored_energy=2.0, disscale=0.0, disbondscale=0.25),
            },
        )
        blocked_energy = mesh_module._legacy_trial_stored_energy(
            0,
            node_xy,
            np.array([0.55, 0.50], dtype=np.float64),
            np.array([[0.30, 0.40], [0.30, 0.60]], dtype=np.float64),
            nodes,
            [10, 20],
            flynns,
            stored_energy_context,
        )

        self.assertGreater(float(full_energy), 0.0)
        self.assertAlmostEqual(float(gated_energy), float(full_energy), places=8)
        self.assertEqual(float(blocked_energy), 0.0)

    def test_roi_weighted_label_density_prefers_same_label_support_before_fallback(self) -> None:
        label_density_index = {
            7: {
                "positions": np.array([[0.55, 0.50], [0.75, 0.50]], dtype=np.float64),
                "densities": np.array([1.0, 5.0], dtype=np.float64),
                "mean_density": 3.0,
            },
        }

        density = mesh_module._roi_weighted_label_density(
            np.array([0.50, 0.50], dtype=np.float64),
            7,
            label_density_index,
            roi=0.40,
            dummy_density=50.0,
            all_positions=np.array([[0.55, 0.50], [0.75, 0.50], [0.52, 0.52]], dtype=np.float64),
            all_densities=np.array([1.0, 5.0, 20.0], dtype=np.float64),
        )

        self.assertGreater(density, 1.0)
        self.assertLess(density, 5.0)
        self.assertLess(density, 20.0)

    def test_roi_weighted_label_density_uses_dummy_density_when_same_flynn_has_no_roi_support(self) -> None:
        density = mesh_module._roi_weighted_label_density(
            np.array([0.50, 0.50], dtype=np.float64),
            7,
            {
                7: {
                    "positions": np.array([[0.90, 0.90]], dtype=np.float64),
                    "densities": np.array([3.0], dtype=np.float64),
                    "mean_density": 3.0,
                },
            },
            roi=0.10,
            dummy_density=50.0,
            all_positions=np.array([[0.52, 0.50]], dtype=np.float64),
            all_densities=np.array([20.0], dtype=np.float64),
        )

        self.assertEqual(density, 50.0)

    def test_roi_weighted_label_density_falls_back_to_all_unodes_when_target_flynn_has_none(self) -> None:
        density = mesh_module._roi_weighted_label_density(
            np.array([0.50, 0.50], dtype=np.float64),
            7,
            {
                7: {
                    "positions": np.zeros((0, 2), dtype=np.float64),
                    "densities": np.zeros((0,), dtype=np.float64),
                    "mean_density": 0.0,
                },
            },
            roi=0.10,
            dummy_density=50.0,
            all_positions=np.array([[0.52, 0.50], [0.55, 0.50]], dtype=np.float64),
            all_densities=np.array([20.0, 10.0], dtype=np.float64),
        )

        self.assertGreater(density, 10.0)
        self.assertLess(density, 20.0)

    def test_fallback_incident_stored_energy_uses_dummy_density_and_max_phase_energy(self) -> None:
        phase_db = SimpleNamespace(
            first_phase=1,
            phase_properties={
                1: SimpleNamespace(stored_energy=2.0),
                2: SimpleNamespace(stored_energy=5.0),
            },
        )

        energy = mesh_module._fallback_incident_stored_energy(
            0.25,
            [10, 20],
            {10: 1, 20: 2},
            phase_db,
            dummy_density=8.0,
        )

        self.assertAlmostEqual(energy, 10.0, places=12)

    def test_trial_cluster_area_energy_uses_target_cluster_area(self) -> None:
        nodes = np.array(
            [
                [0.10, 0.10],
                [0.30, 0.10],
                [0.10, 0.30],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 10, "source_flynn_id": 10, "label": 0, "node_ids": [0, 1, 2]},
        ]
        cluster_energy = mesh_module._trial_cluster_area_energy(
            0,
            np.array([0.15, 0.10], dtype=np.float64),
            nodes,
            flynns,
            [10],
            {
                "by_flynn": {
                    10: {
                        "phase_id": 2,
                        "current_area": 0.02,
                        "target_area": 0.02,
                    },
                },
                "multiplier_a": 2.0,
                "multiplier_d": 2.0,
            },
        )

        self.assertAlmostEqual(cluster_energy, 0.125, places=12)

    def test_trial_swept_area_unions_same_side_triangles(self) -> None:
        area = mesh_module._trial_swept_area(
            np.array([0.50, 0.50], dtype=np.float64),
            np.array([0.60, 0.50], dtype=np.float64),
            np.array(
                [
                    [0.525, 0.60],
                    [0.575, 0.60],
                ],
                dtype=np.float64,
            ),
        )

        self.assertAlmostEqual(area, 0.0075, places=12)

    def test_relax_mesh_state_elle_surface_runs_local_topology_after_each_moved_node(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.20, "y": 0.45},
                {"x": 0.78, "y": 0.57},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2]},
            ],
            "stats": {"grid_shape": [256, 256]},
            "events": [],
        }

        with patch.object(
            mesh_module,
            "_move_node_elle_surface",
            return_value=np.array([0.01, 0.0], dtype=np.float64),
        ) as move_mock, patch.object(
            mesh_module,
            "_maintain_mesh_locally",
            side_effect=lambda nodes, flynns, **kwargs: (
                nodes,
                flynns,
                [],
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        ) as local_maintain_mock, patch.object(
            mesh_module,
            "_maintain_mesh_once",
            side_effect=lambda nodes, flynns, **kwargs: (
                nodes,
                flynns,
                [],
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        ) as maintain_mock:
            relax_mesh_state(
                mesh_state,
                MeshRelaxationConfig(
                    steps=1,
                    topology_steps=1,
                    movement_model="elle_surface",
                    switch_distance=0.05,
                ),
            )

        self.assertEqual(move_mock.call_count, 3)
        self.assertEqual(local_maintain_mock.call_count, 3)
        self.assertEqual(maintain_mock.call_count, 1)

    def test_rebuild_remaining_node_order_preserves_pending_nodes_and_appends_new_nodes(self) -> None:
        previous_positions = np.array(
            [
                [0.10, 0.10],
                [0.30, 0.10],
                [0.50, 0.10],
            ],
            dtype=np.float64,
        )
        current_positions = np.array(
            [
                [0.10, 0.10],
                [0.30, 0.10],
                [0.50, 0.10],
                [0.70, 0.10],
            ],
            dtype=np.float64,
        )

        rebuilt = mesh_module._rebuild_remaining_node_order(
            previous_positions,
            current_positions,
            processed_old_ids=[2],
            remaining_old_ids=[0, 1],
            rng=np.random.default_rng(0),
        )

        self.assertEqual(rebuilt, [0, 1, 3])

    def test_relax_mesh_state_rebuilds_remaining_node_order_when_topology_changes_indices(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.20, "y": 0.45},
                {"x": 0.78, "y": 0.57},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2]},
            ],
            "stats": {"grid_shape": [256, 256]},
            "events": [],
        }

        call_counter = {"count": 0}

        def _maintain_with_insert(nodes, flynns, **kwargs):
            call_counter["count"] += 1
            if call_counter["count"] == 1:
                updated_nodes = np.vstack(
                    [
                        nodes,
                        np.array([[0.35, 0.48]], dtype=np.float64),
                    ]
                )
                updated_flynns = copy.deepcopy(flynns)
                updated_flynns[0]["node_ids"] = [0, 1, 3, 2]
                return (
                    updated_nodes,
                    updated_flynns,
                    [],
                    0,
                    0,
                    0,
                    0,
                    1,
                    0,
                    0,
                    0,
                    0,
                )
            return (
                nodes,
                flynns,
                [],
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            )

        with patch.object(
            mesh_module,
            "_move_node_elle_surface",
            return_value=np.array([0.01, 0.0], dtype=np.float64),
        ), patch.object(
            mesh_module,
            "_maintain_mesh_locally",
            side_effect=_maintain_with_insert,
        ), patch.object(
            mesh_module,
            "_maintain_mesh_once",
            side_effect=lambda nodes, flynns, **kwargs: (
                nodes,
                flynns,
                [],
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
                0,
            ),
        ), patch.object(
            mesh_module,
            "_rebuild_remaining_node_order",
            wraps=mesh_module._rebuild_remaining_node_order,
        ) as rebuild_mock:
            relax_mesh_state(
                mesh_state,
                MeshRelaxationConfig(
                    steps=1,
                    topology_steps=1,
                    movement_model="elle_surface",
                    switch_distance=0.05,
                ),
            )

        self.assertGreaterEqual(rebuild_mock.call_count, 1)

    def test_couple_mesh_to_order_parameters_supports_mesh_only_translation(self) -> None:
        current_labels = np.array(
            [
                [0, 0],
                [1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [0, 1],
                [0, 1],
            ],
            dtype=np.int32,
        )
        phi = mesh_labels_to_order_parameters(current_labels, 2)
        target_mesh_state = build_mesh_state(target_labels)

        phi_feedback, mesh_state, feedback_stats = couple_mesh_to_order_parameters(
            phi,
            MeshFeedbackConfig(
                every=1,
                strength=0.0,
                transport_strength=0.0,
                update_mode="mesh_only",
                relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
            ),
            base_mesh_state=target_mesh_state,
        )

        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), target_labels)
        self.assertEqual(mesh_state["stats"]["mesh_update_mode"], "mesh_only")
        self.assertGreater(feedback_stats["changed_pixels"], 0)

    def test_couple_mesh_to_order_parameters_mesh_only_uses_full_seed_label_remap(self) -> None:
        current_labels = np.zeros((4, 4), dtype=np.int32)
        target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        phi = mesh_labels_to_order_parameters(current_labels, 2)
        target_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
            "_runtime_seed_unodes": {
                "grid_shape": (4, 4),
                "positions": (
                    (0.125, 0.875),
                    (0.125, 0.625),
                    (0.125, 0.375),
                    (0.125, 0.125),
                    (0.375, 0.875),
                    (0.375, 0.625),
                    (0.375, 0.375),
                    (0.375, 0.125),
                    (0.625, 0.875),
                    (0.625, 0.625),
                    (0.625, 0.375),
                    (0.625, 0.125),
                    (0.875, 0.875),
                    (0.875, 0.625),
                    (0.875, 0.375),
                    (0.875, 0.125),
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (100, 200),
                "field_order": ("U_ATTRIB_C",),
                "values": {
                    "U_ATTRIB_C": tuple(0.0 for _ in range(16)),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": tuple((0.0,) for _ in range(16)),
                },
            },
        }

        phi_feedback, mesh_state, feedback_stats = couple_mesh_to_order_parameters(
            phi,
            MeshFeedbackConfig(
                every=1,
                strength=0.0,
                transport_strength=0.0,
                update_mode="mesh_only",
                relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
            ),
            base_mesh_state=target_mesh_state,
        )

        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), target_labels)
        self.assertEqual(feedback_stats["assigned_unodes"], 16)
        self.assertEqual(feedback_stats["unassigned_unodes"], 0)
        self.assertEqual(feedback_stats["changed_unodes"], 8)
        self.assertEqual(feedback_stats["label_remap_mode"], "full_seed_remap")
        self.assertEqual(mesh_state["stats"]["mesh_label_remap_mode"], "full_seed_remap")
        remapped = np.asarray(mesh_state["_runtime_seed_unode_fields"]["values"]["U_ATTRIB_C"], dtype=np.float64)
        self.assertTrue(np.allclose(remapped[:8], 100.0))
        self.assertTrue(np.allclose(remapped[8:], 200.0))

    def test_couple_mesh_to_order_parameters_extends_source_labels_for_new_identity_stable_label(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [2, 2, 1, 1],
                [2, 2, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        phi = mesh_labels_to_order_parameters(current_labels, 3)
        target_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 2, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
            "_runtime_seed_unodes": {
                "grid_shape": (4, 4),
                "positions": (
                    (0.125, 0.875),
                    (0.125, 0.625),
                    (0.125, 0.375),
                    (0.125, 0.125),
                    (0.375, 0.875),
                    (0.375, 0.625),
                    (0.375, 0.375),
                    (0.375, 0.125),
                    (0.625, 0.875),
                    (0.625, 0.625),
                    (0.625, 0.375),
                    (0.625, 0.125),
                    (0.875, 0.875),
                    (0.875, 0.625),
                    (0.875, 0.375),
                    (0.875, 0.125),
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (100, 200),
                "field_order": ("U_ATTRIB_C",),
                "values": {
                    "U_ATTRIB_C": tuple(float(100 if value == 0 else 200) for value in current_labels.ravel()),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": tuple((float(100 if value == 0 else 200),) for value in current_labels.ravel()),
                },
            },
        }

        with patch.object(
            mesh_module,
            "assign_seed_unodes_from_mesh",
            return_value=(
                target_labels,
                {
                    "assigned_unodes": 16,
                    "unassigned_unodes": 0,
                    "changed_unodes": int(np.count_nonzero(target_labels != current_labels)),
                },
            ),
        ):
            phi_feedback, mesh_state, _feedback_stats = couple_mesh_to_order_parameters(
                phi,
                MeshFeedbackConfig(
                    every=1,
                    strength=0.0,
                    transport_strength=0.0,
                    update_mode="mesh_only",
                    relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
                ),
                base_mesh_state=target_mesh_state,
            )

        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), target_labels)
        self.assertEqual(
            tuple(mesh_state["_runtime_seed_unode_fields"]["source_labels"]),
            (100, 200, 201),
        )
        remapped = np.asarray(mesh_state["_runtime_seed_unode_fields"]["values"]["U_ATTRIB_C"], dtype=np.float64)
        self.assertIn(201.0, remapped.tolist())

    def test_couple_mesh_to_order_parameters_uses_incremental_remap_after_nucleation_rebuild(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        full_target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        incremental_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        phi = mesh_labels_to_order_parameters(current_labels, 2)
        base_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
            "_runtime_incremental_label_remap_stages": 1,
        }
        motion_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
            "_runtime_seed_unodes": {
                "grid_shape": (4, 4),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (100, 200),
                "field_order": ("U_ATTRIB_C",),
                "values": {
                    "U_ATTRIB_C": tuple(
                        float(100 if value == 0 else 200) for value in current_labels.ravel()
                    ),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": tuple(
                        (float(100 if value == 0 else 200),) for value in current_labels.ravel()
                    ),
                },
            },
        }

        with (
            patch(
                "elle_jax_model.mesh.compute_mesh_motion_velocity",
                return_value=(
                    base_mesh_state,
                    motion_mesh_state,
                    None,
                    None,
                    {
                        "transport_pixels": 0,
                        "max_displacement": 0.0,
                        "mean_displacement": 0.0,
                    },
                ),
            ),
            patch(
                "elle_jax_model.mesh.assign_seed_unodes_from_mesh",
                return_value=(full_target_labels, {"assigned_unodes": 16, "unassigned_unodes": 0}),
            ),
            patch(
                "elle_jax_model.mesh.incremental_seed_unode_reassignment",
                return_value=(incremental_labels, {"swept_unodes": 4, "changed_unodes": 1}),
            ) as incremental_mock,
            patch(
                "elle_jax_model.mesh.update_seed_unode_fields",
                return_value=(
                    dict(motion_mesh_state["_runtime_seed_unode_fields"]["values"]),
                    {
                        "updated_scalar_unodes": 0,
                        "scalar_swept_unodes": 0,
                        "mass_conserved_fields": 0,
                        "mass_partitioned_fields": 0,
                        "scalar_mass_residual": 0.0,
                    },
                    {},
                ),
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_sections",
                return_value=(
                    motion_mesh_state["_runtime_seed_unode_sections"],
                    {
                        "updated_orientation_unodes": 0,
                        "fallback_orientation_unodes": 0,
                        "old_value_fallback_orientation_unodes": 0,
                    },
                ),
            ),
        ):
            phi_feedback, mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                MeshFeedbackConfig(
                    every=1,
                    strength=0.0,
                    transport_strength=0.0,
                    update_mode="mesh_only",
                    relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
                ),
                base_mesh_state=base_mesh_state,
            )

        incremental_mock.assert_called_once()
        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), incremental_labels)
        self.assertEqual(feedback_stats["label_remap_mode"], "incremental_post_nucleation_rebuild")
        self.assertEqual(feedback_stats["incremental_swept_unodes"], 4)
        self.assertEqual(feedback_stats["incremental_changed_unodes"], 1)
        self.assertNotIn("_runtime_incremental_label_remap_stages", mesh_state)

    def test_incremental_post_nucleation_remap_falls_back_to_full_remap_when_more_fragmented(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        full_target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        incremental_labels = np.array(
            [
                [1, 0, 1, 0],
                [0, 0, 0, 0],
                [1, 0, 1, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        phi = mesh_labels_to_order_parameters(current_labels, 2)
        base_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
            "_runtime_incremental_label_remap_stages": 1,
        }
        motion_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
            "_runtime_seed_unodes": {
                "grid_shape": (4, 4),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (100, 200),
                "field_order": ("U_ATTRIB_C",),
                "values": {
                    "U_ATTRIB_C": tuple(100.0 for _ in current_labels.ravel()),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": tuple((100.0,) for _ in current_labels.ravel()),
                },
            },
        }

        with (
            patch(
                "elle_jax_model.mesh.compute_mesh_motion_velocity",
                return_value=(
                    base_mesh_state,
                    motion_mesh_state,
                    None,
                    None,
                    {
                        "transport_pixels": 0,
                        "max_displacement": 0.0,
                        "mean_displacement": 0.0,
                    },
                ),
            ),
            patch(
                "elle_jax_model.mesh.assign_seed_unodes_from_mesh",
                return_value=(full_target_labels, {"assigned_unodes": 16, "unassigned_unodes": 0}),
            ),
            patch(
                "elle_jax_model.mesh.incremental_seed_unode_reassignment",
                return_value=(incremental_labels, {"swept_unodes": 8, "changed_unodes": 6}),
            ) as incremental_mock,
            patch(
                "elle_jax_model.mesh.update_seed_unode_fields",
                return_value=(
                    dict(motion_mesh_state["_runtime_seed_unode_fields"]["values"]),
                    {
                        "updated_scalar_unodes": 0,
                        "scalar_swept_unodes": 0,
                        "mass_conserved_fields": 0,
                        "mass_partitioned_fields": 0,
                        "scalar_mass_residual": 0.0,
                    },
                    {},
                ),
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_sections",
                return_value=(
                    motion_mesh_state["_runtime_seed_unode_sections"],
                    {
                        "updated_orientation_unodes": 0,
                        "fallback_orientation_unodes": 0,
                        "old_value_fallback_orientation_unodes": 0,
                    },
                ),
            ),
        ):
            phi_feedback, mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                MeshFeedbackConfig(
                    every=1,
                    strength=0.0,
                    transport_strength=0.0,
                    update_mode="mesh_only",
                    relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
                ),
                base_mesh_state=base_mesh_state,
            )

        incremental_mock.assert_called_once()
        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), full_target_labels)
        self.assertEqual(feedback_stats["label_remap_mode"], "full_seed_remap_post_nucleation_fallback")
        self.assertEqual(feedback_stats["incremental_fragmentation_fallback"], 1)
        self.assertGreater(
            feedback_stats["incremental_component_count"],
            feedback_stats["full_remap_component_count"],
        )
        self.assertNotIn("_runtime_incremental_label_remap_stages", mesh_state)

    def test_incremental_post_nucleation_full_fallback_applies_connectivity_cleanup(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
            ],
            dtype=np.int32,
        )
        full_target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        cleaned_full_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        incremental_labels = np.array(
            [
                [0, 1, 0, 1],
                [0, 0, 0, 0],
                [0, 1, 0, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        phi = mesh_labels_to_order_parameters(current_labels, 2)
        base_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
            "_runtime_incremental_label_remap_stages": 1,
        }
        motion_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4], "nucleation_fragment_merge_max_size": 10},
            "events": [],
            "_runtime_seed_unodes": {
                "grid_shape": (4, 4),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (100, 200),
                "field_order": ("U_ATTRIB_C",),
                "values": {
                    "U_ATTRIB_C": tuple(100.0 for _ in current_labels.ravel()),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": tuple((100.0,) for _ in current_labels.ravel()),
                },
            },
        }

        with (
            patch(
                "elle_jax_model.mesh.compute_mesh_motion_velocity",
                return_value=(
                    base_mesh_state,
                    motion_mesh_state,
                    None,
                    None,
                    {
                        "transport_pixels": 0,
                        "max_displacement": 0.0,
                        "mean_displacement": 0.0,
                    },
                ),
            ),
            patch(
                "elle_jax_model.mesh.assign_seed_unodes_from_mesh",
                return_value=(full_target_labels, {"assigned_unodes": 16, "unassigned_unodes": 0}),
            ),
            patch(
                "elle_jax_model.mesh.incremental_seed_unode_reassignment",
                return_value=(incremental_labels, {"swept_unodes": 12, "changed_unodes": 8}),
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_fields",
                return_value=(
                    dict(motion_mesh_state["_runtime_seed_unode_fields"]["values"]),
                    {
                        "updated_scalar_unodes": 0,
                        "scalar_swept_unodes": 0,
                        "mass_conserved_fields": 0,
                        "mass_partitioned_fields": 0,
                        "scalar_mass_residual": 0.0,
                    },
                    {},
                ),
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_sections",
                return_value=(
                    motion_mesh_state["_runtime_seed_unode_sections"],
                    {
                        "updated_orientation_unodes": 0,
                        "fallback_orientation_unodes": 0,
                        "old_value_fallback_orientation_unodes": 0,
                    },
                ),
            ),
        ):
            phi_feedback, _mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                MeshFeedbackConfig(
                    every=1,
                    strength=0.0,
                    transport_strength=0.0,
                    update_mode="mesh_only",
                    relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
                ),
                base_mesh_state=base_mesh_state,
            )

        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), cleaned_full_labels)
        self.assertEqual(feedback_stats["label_remap_mode"], "full_seed_remap_post_nucleation_fallback")
        self.assertEqual(feedback_stats["incremental_fragmentation_fallback"], 1)
        self.assertGreater(feedback_stats["connectivity_reassigned_unodes"], 0)
        self.assertGreater(feedback_stats["connectivity_merged_components"], 0)

    def test_topology_static_mesh_feedback_uses_incremental_host_repair(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        full_target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 1, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        incremental_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        phi = mesh_labels_to_order_parameters(current_labels, 2)
        base_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        motion_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {
                "grid_shape": [4, 4],
                "mesh_inserted_nodes": 0,
                "mesh_removed_nodes": 0,
                "mesh_switched_triples": 0,
                "mesh_merged_flynns": 0,
                "mesh_split_flynns": 0,
                "mesh_deleted_small_flynns": 0,
            },
            "events": [],
            "_runtime_seed_unodes": {
                "grid_shape": (4, 4),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (100, 200),
                "field_order": ("U_ATTRIB_C",),
                "values": {
                    "U_ATTRIB_C": tuple(
                        float(100 if value == 0 else 200) for value in current_labels.ravel()
                    ),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": tuple(
                        (float(100 if value == 0 else 200),) for value in current_labels.ravel()
                    ),
                },
            },
        }

        with (
            patch(
                "elle_jax_model.mesh.compute_mesh_motion_velocity",
                return_value=(
                    base_mesh_state,
                    motion_mesh_state,
                    None,
                    None,
                    {
                        "transport_pixels": 0,
                        "max_displacement": 0.0,
                        "mean_displacement": 0.0,
                    },
                ),
            ),
            patch(
                "elle_jax_model.mesh.assign_seed_unodes_from_mesh",
                return_value=(full_target_labels, {"assigned_unodes": 16, "unassigned_unodes": 0}),
            ),
            patch(
                "elle_jax_model.mesh.incremental_seed_unode_reassignment",
                return_value=(incremental_labels, {"swept_unodes": 1, "changed_unodes": 0}),
            ) as incremental_mock,
            patch(
                "elle_jax_model.mesh.update_seed_unode_fields",
                return_value=(
                    dict(motion_mesh_state["_runtime_seed_unode_fields"]["values"]),
                    {
                        "updated_scalar_unodes": 0,
                        "scalar_swept_unodes": 0,
                        "mass_conserved_fields": 0,
                        "mass_partitioned_fields": 0,
                        "scalar_mass_residual": 0.0,
                    },
                    {},
                ),
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_sections",
                return_value=(
                    motion_mesh_state["_runtime_seed_unode_sections"],
                    {
                        "updated_orientation_unodes": 0,
                        "fallback_orientation_unodes": 0,
                        "old_value_fallback_orientation_unodes": 0,
                    },
                ),
            ),
        ):
            phi_feedback, _mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                MeshFeedbackConfig(
                    every=1,
                    strength=0.0,
                    transport_strength=0.0,
                    update_mode="mesh_only",
                    relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
                ),
                base_mesh_state=base_mesh_state,
            )

        incremental_mock.assert_called_once()
        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), incremental_labels)
        self.assertEqual(feedback_stats["label_remap_mode"], "incremental_host_repair")
        self.assertEqual(feedback_stats["incremental_swept_unodes"], 1)
        self.assertEqual(feedback_stats["incremental_changed_unodes"], 0)

    def test_topology_static_override_feedback_prefers_incremental_host_repair(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        full_target_labels = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        incremental_labels = np.array(current_labels, copy=True)
        override_labels = np.full(current_labels.shape, -1, dtype=np.int32)
        override_labels[1, 1] = 1
        expected_labels = np.array(current_labels, copy=True)
        expected_labels[1, 1] = 1
        phi = mesh_labels_to_order_parameters(current_labels, 2)
        base_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        motion_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {
                "grid_shape": [4, 4],
                "mesh_inserted_nodes": 0,
                "mesh_removed_nodes": 0,
                "mesh_switched_triples": 0,
                "mesh_merged_flynns": 0,
                "mesh_split_flynns": 0,
                "mesh_deleted_small_flynns": 0,
                "nucleation_fragment_merge_max_size": 0,
            },
            "events": [],
            "_runtime_label_overrides": override_labels,
            "_runtime_seed_unodes": {
                "grid_shape": (4, 4),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (100, 200),
                "field_order": ("U_ATTRIB_C",),
                "values": {
                    "U_ATTRIB_C": tuple(
                        float(100 if value == 0 else 200) for value in current_labels.ravel()
                    ),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": tuple(
                        (float(100 if value == 0 else 200),) for value in current_labels.ravel()
                    ),
                },
            },
        }

        def passthrough_connectivity(labels, **_kwargs):
            return np.asarray(labels, dtype=np.int32), {
                "connectivity_reassigned_unodes": 0,
                "connectivity_merged_components": 0,
            }

        with (
            patch(
                "elle_jax_model.mesh.compute_mesh_motion_velocity",
                return_value=(
                    base_mesh_state,
                    motion_mesh_state,
                    None,
                    None,
                    {
                        "transport_pixels": 0,
                        "max_displacement": 0.0,
                        "mean_displacement": 0.0,
                    },
                ),
            ),
            patch(
                "elle_jax_model.mesh.assign_seed_unodes_from_mesh",
                return_value=(full_target_labels, {"assigned_unodes": 16, "unassigned_unodes": 0}),
            ),
            patch(
                "elle_jax_model.mesh.incremental_seed_unode_reassignment",
                return_value=(incremental_labels, {"swept_unodes": 1, "changed_unodes": 0}),
            ) as incremental_mock,
            patch(
                "elle_jax_model.mesh._enforce_connected_label_ownership",
                side_effect=passthrough_connectivity,
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_fields",
                return_value=(
                    dict(motion_mesh_state["_runtime_seed_unode_fields"]["values"]),
                    {
                        "updated_scalar_unodes": 0,
                        "scalar_swept_unodes": 0,
                        "mass_conserved_fields": 0,
                        "mass_partitioned_fields": 0,
                        "scalar_mass_residual": 0.0,
                    },
                    {},
                ),
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_sections",
                return_value=(
                    motion_mesh_state["_runtime_seed_unode_sections"],
                    {
                        "updated_orientation_unodes": 0,
                        "fallback_orientation_unodes": 0,
                        "old_value_fallback_orientation_unodes": 0,
                    },
                ),
            ),
        ):
            phi_feedback, _mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                MeshFeedbackConfig(
                    every=1,
                    strength=0.0,
                    transport_strength=0.0,
                    update_mode="mesh_only",
                    relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
                ),
                base_mesh_state=base_mesh_state,
            )

        incremental_mock.assert_called_once()
        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), expected_labels)
        self.assertEqual(feedback_stats["label_remap_mode"], "incremental_host_repair_overrides")
        self.assertEqual(feedback_stats["incremental_swept_unodes"], 1)

    def test_topology_static_rejected_rebuild_prefers_incremental_host_repair(self) -> None:
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        full_target_labels = np.array(
            [
                [1, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        incremental_labels = np.array(current_labels, copy=True)
        phi = mesh_labels_to_order_parameters(current_labels, 2)
        base_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        motion_mesh_state = {
            "nodes": [],
            "flynns": [],
            "stats": {
                "grid_shape": [4, 4],
                "mesh_inserted_nodes": 0,
                "mesh_removed_nodes": 0,
                "mesh_switched_triples": 0,
                "mesh_merged_flynns": 0,
                "mesh_split_flynns": 0,
                "mesh_deleted_small_flynns": 0,
                "nucleation_mesh_rebuild_rejected": 1,
                "nucleation_fragment_merge_max_size": 0,
            },
            "events": [],
            "_runtime_seed_unodes": {
                "grid_shape": (4, 4),
                "positions": tuple(
                    (0.125 + 0.25 * ix, 0.875 - 0.25 * iy)
                    for ix in range(4)
                    for iy in range(4)
                ),
                "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            },
            "_runtime_seed_unode_fields": {
                "label_attribute": "U_ATTRIB_C",
                "source_labels": (100, 200),
                "field_order": ("U_ATTRIB_C",),
                "values": {
                    "U_ATTRIB_C": tuple(
                        float(100 if value == 0 else 200) for value in current_labels.ravel()
                    ),
                },
            },
            "_runtime_seed_unode_sections": {
                "field_order": ("U_ATTRIB_C",),
                "defaults": {"U_ATTRIB_C": (0.0,)},
                "component_counts": {"U_ATTRIB_C": 1},
                "values": {
                    "U_ATTRIB_C": tuple(
                        (float(100 if value == 0 else 200),) for value in current_labels.ravel()
                    ),
                },
            },
        }

        def passthrough_connectivity(labels, **_kwargs):
            return np.asarray(labels, dtype=np.int32), {
                "connectivity_reassigned_unodes": 0,
                "connectivity_merged_components": 0,
            }

        with (
            patch(
                "elle_jax_model.mesh.compute_mesh_motion_velocity",
                return_value=(
                    base_mesh_state,
                    motion_mesh_state,
                    None,
                    None,
                    {
                        "transport_pixels": 0,
                        "max_displacement": 0.0,
                        "mean_displacement": 0.0,
                    },
                ),
            ),
            patch(
                "elle_jax_model.mesh.assign_seed_unodes_from_mesh",
                return_value=(full_target_labels, {"assigned_unodes": 16, "unassigned_unodes": 0}),
            ),
            patch(
                "elle_jax_model.mesh.incremental_seed_unode_reassignment",
                return_value=(incremental_labels, {"swept_unodes": 1, "changed_unodes": 0}),
            ) as incremental_mock,
            patch(
                "elle_jax_model.mesh._enforce_connected_label_ownership",
                side_effect=passthrough_connectivity,
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_fields",
                return_value=(
                    dict(motion_mesh_state["_runtime_seed_unode_fields"]["values"]),
                    {
                        "updated_scalar_unodes": 0,
                        "scalar_swept_unodes": 0,
                        "mass_conserved_fields": 0,
                        "mass_partitioned_fields": 0,
                        "scalar_mass_residual": 0.0,
                    },
                    {},
                ),
            ),
            patch(
                "elle_jax_model.mesh.update_seed_unode_sections",
                return_value=(
                    motion_mesh_state["_runtime_seed_unode_sections"],
                    {
                        "updated_orientation_unodes": 0,
                        "fallback_orientation_unodes": 0,
                        "old_value_fallback_orientation_unodes": 0,
                    },
                ),
            ),
        ):
            phi_feedback, _mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                MeshFeedbackConfig(
                    every=1,
                    strength=0.0,
                    transport_strength=0.0,
                    update_mode="mesh_only",
                    relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
                ),
                base_mesh_state=base_mesh_state,
            )

        incremental_mock.assert_called_once()
        np.testing.assert_array_equal(dominant_grain_map(phi_feedback), incremental_labels)
        self.assertEqual(
            feedback_stats["label_remap_mode"],
            "incremental_host_repair_rejected_rebuild",
        )
        self.assertEqual(feedback_stats["incremental_swept_unodes"], 1)

    def test_assign_seed_unodes_from_mesh_respects_original_unode_positions(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (4, 4),
            "positions": (
                (0.125, 0.875),
                (0.125, 0.625),
                (0.125, 0.375),
                (0.125, 0.125),
                (0.375, 0.875),
                (0.375, 0.625),
                (0.375, 0.375),
                (0.375, 0.125),
                (0.625, 0.875),
                (0.625, 0.625),
                (0.625, 0.375),
                (0.625, 0.125),
                (0.875, 0.875),
                (0.875, 0.625),
                (0.875, 0.375),
                (0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
        }

        labels, stats = assign_seed_unodes_from_mesh(mesh_state, seed_unodes)

        np.testing.assert_array_equal(
            labels,
            np.array(
                [
                    [0, 0, 0, 0],
                    [0, 0, 0, 0],
                    [1, 1, 1, 1],
                    [1, 1, 1, 1],
                ],
                dtype=np.int32,
            ),
        )
        self.assertEqual(stats["assigned_unodes"], 16)
        self.assertEqual(stats["unassigned_unodes"], 0)

    def test_assign_seed_unodes_from_mesh_fills_polygon_gaps_from_rasterized_mesh(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (4, 4),
            "positions": (
                (0.125, 0.875),
                (0.125, 0.625),
                (0.125, 0.375),
                (0.125, 0.125),
                (0.375, 0.875),
                (0.375, 0.625),
                (0.375, 0.375),
                (0.375, 0.125),
                (0.625, 0.875),
                (0.625, 0.625),
                (0.625, 0.375),
                (0.625, 0.125),
                (0.875, 0.875),
                (0.875, 0.625),
                (0.875, 0.375),
                (0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
        }
        fallback_labels = np.full((4, 4), 9, dtype=np.int32)
        rasterized_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )

        original_mark = mesh_module._mark_seed_points_in_polygon
        state = {"call_index": 0}

        def sparse_mark(points, polygon_points, active_mask=None):
            result = original_mark(points, polygon_points, active_mask=active_mask)
            assigned_indices = np.flatnonzero(result)
            if state["call_index"] == 0 and assigned_indices.size:
                result[assigned_indices[-1]] = False
            state["call_index"] += 1
            return result

        with (
            patch("elle_jax_model.mesh._mark_seed_points_in_polygon", side_effect=sparse_mark),
            patch("elle_jax_model.mesh.rasterize_mesh_labels", return_value=rasterized_labels),
        ):
            labels, stats = assign_seed_unodes_from_mesh(
                mesh_state,
                seed_unodes,
                fallback_labels=fallback_labels,
            )

        np.testing.assert_array_equal(labels, rasterized_labels)
        self.assertEqual(stats["assigned_unodes"], 16)
        self.assertEqual(stats["unassigned_unodes"], 0)
        self.assertGreater(stats["raster_filled_unassigned"], 0)

    def test_assign_seed_unodes_from_mesh_prefers_smallest_matching_flynn_per_point(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 0.5},
                {"x": 0.0, "y": 0.5},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [4, 5, 6, 7]},
            ],
            "stats": {"grid_shape": [1, 1]},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (1, 1),
            "positions": ((0.25, 0.25),),
            "grid_indices": ((0, 0),),
        }

        with patch(
            "elle_jax_model.mesh._mark_seed_points_in_polygon",
            side_effect=[
                np.array([True], dtype=bool),
                np.array([True], dtype=bool),
            ],
        ):
            labels, stats = assign_seed_unodes_from_mesh(mesh_state, seed_unodes)

        np.testing.assert_array_equal(labels, np.array([[1]], dtype=np.int32))
        self.assertEqual(stats["assigned_unodes"], 1)
        self.assertEqual(stats["unassigned_unodes"], 0)
        self.assertEqual(stats["raster_filled_unassigned"], 0)

    def test_assign_seed_unodes_from_mesh_keeps_current_host_before_reassignment(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 0.5},
                {"x": 0.0, "y": 0.5},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [4, 5, 6, 7]},
            ],
            "stats": {"grid_shape": [1, 1]},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (1, 1),
            "positions": ((0.25, 0.25),),
            "grid_indices": ((0, 0),),
        }
        fallback_labels = np.array([[0]], dtype=np.int32)

        labels, stats = assign_seed_unodes_from_mesh(
            mesh_state,
            seed_unodes,
            fallback_labels=fallback_labels,
        )

        np.testing.assert_array_equal(labels, np.array([[0]], dtype=np.int32))
        self.assertEqual(stats["assigned_unodes"], 1)
        self.assertEqual(stats["unassigned_unodes"], 0)
        self.assertEqual(stats["raster_filled_unassigned"], 0)

    def test_incremental_seed_unode_reassignment_updates_only_swept_region(self) -> None:
        base_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        moved_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.75, "y": 0.0},
                {"x": 0.75, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (4, 4),
            "positions": (
                (0.125, 0.875),(0.125, 0.625),(0.125, 0.375),(0.125, 0.125),
                (0.375, 0.875),(0.375, 0.625),(0.375, 0.375),(0.375, 0.125),
                (0.625, 0.875),(0.625, 0.625),(0.625, 0.375),(0.625, 0.125),
                (0.875, 0.875),(0.875, 0.625),(0.875, 0.375),(0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
        }
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )

        updated_labels, stats = incremental_seed_unode_reassignment(
            current_labels,
            target_labels,
            base_mesh_state,
            moved_mesh_state,
            seed_unodes,
        )

        np.testing.assert_array_equal(updated_labels, target_labels)
        self.assertEqual(stats["swept_unodes"], 4)
        self.assertEqual(stats["changed_unodes"], 4)

    def test_update_seed_unode_fields_uses_local_weighting(self) -> None:
        base_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        moved_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.75, "y": 0.0},
                {"x": 0.75, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (4, 4),
            "positions": (
                (0.125, 0.875),(0.125, 0.625),(0.125, 0.375),(0.125, 0.125),
                (0.375, 0.875),(0.375, 0.625),(0.375, 0.375),(0.375, 0.125),
                (0.625, 0.875),(0.625, 0.625),(0.625, 0.375),(0.625, 0.125),
                (0.875, 0.875),(0.875, 0.625),(0.875, 0.375),(0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
        }
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        seed_fields = {
            "label_attribute": "U_ATTRIB_C",
            "source_labels": (100, 200),
            "roi": 0.7,
            "values": {
                "U_ATTRIB_C": (100.0,) * 8 + (200.0,) * 8,
                "U_ATTRIB_A": (
                    1.0, 1.0, 1.0, 1.0,
                    2.0, 2.0, 2.0, 2.0,
                    10.0, 20.0, 30.0, 40.0,
                    50.0, 60.0, 70.0, 80.0,
                ),
            },
        }

        updated_fields, stats, _mass_ledgers = update_seed_unode_fields(
            current_labels,
            target_labels,
            base_mesh_state,
            moved_mesh_state,
            seed_unodes,
            seed_fields,
        )

        updated_attr = np.asarray(updated_fields["U_ATTRIB_A"], dtype=np.float64).reshape(4, 4)
        self.assertEqual(stats["scalar_swept_unodes"], 4)
        self.assertGreater(stats["updated_scalar_unodes"], 0)
        self.assertTrue(np.allclose(np.asarray(updated_fields["U_ATTRIB_C"], dtype=np.float64)[:12], 100.0))
        self.assertGreater(float(updated_attr[2, 0]), 1.5)
        self.assertLess(float(updated_attr[2, 0]), 2.1)
        self.assertGreater(float(updated_attr[2, 1]), 1.5)
        self.assertLess(float(updated_attr[2, 1]), 2.1)

    def test_update_seed_unode_fields_conserves_mass_for_u_conc(self) -> None:
        base_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        moved_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.75, "y": 0.0},
                {"x": 0.75, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (4, 4),
            "positions": (
                (0.125, 0.875),(0.125, 0.625),(0.125, 0.375),(0.125, 0.125),
                (0.375, 0.875),(0.375, 0.625),(0.375, 0.375),(0.375, 0.125),
                (0.625, 0.875),(0.625, 0.625),(0.625, 0.375),(0.625, 0.125),
                (0.875, 0.875),(0.875, 0.625),(0.875, 0.375),(0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
        }
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        conc_values = (
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            10.0, 20.0, 30.0, 40.0,
            50.0, 60.0, 70.0, 80.0,
        )
        seed_fields = {
            "label_attribute": "U_ATTRIB_C",
            "source_labels": (100, 200),
            "roi": 0.7,
            "unode_area": 1.0 / 16.0,
            "values": {
                "U_ATTRIB_C": (100.0,) * 8 + (200.0,) * 8,
                "U_CONC_A": conc_values,
            },
        }

        updated_fields, stats, _mass_ledgers = update_seed_unode_fields(
            current_labels,
            target_labels,
            base_mesh_state,
            moved_mesh_state,
            seed_unodes,
            seed_fields,
        )

        before_mass = float(np.sum(np.asarray(conc_values, dtype=np.float64)) * (1.0 / 16.0))
        after_mass = float(np.sum(np.asarray(updated_fields["U_CONC_A"], dtype=np.float64)) * (1.0 / 16.0))
        self.assertEqual(stats["mass_conserved_fields"], 1)
        self.assertEqual(stats["mass_partitioned_fields"], 1)
        self.assertAlmostEqual(stats["scalar_mass_residual"], 0.0, places=8)
        self.assertAlmostEqual(before_mass, after_mass, places=8)

    def test_update_seed_unode_fields_resets_swept_u_dislocden(self) -> None:
        base_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        moved_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.75, "y": 0.0},
                {"x": 0.75, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4]},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (4, 4),
            "positions": (
                (0.125, 0.875),(0.125, 0.625),(0.125, 0.375),(0.125, 0.125),
                (0.375, 0.875),(0.375, 0.625),(0.375, 0.375),(0.375, 0.125),
                (0.625, 0.875),(0.625, 0.625),(0.625, 0.375),(0.625, 0.125),
                (0.875, 0.875),(0.875, 0.625),(0.875, 0.375),(0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
        }
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        disloc_values = (
            1.0, 2.0, 3.0, 4.0,
            5.0, 6.0, 7.0, 8.0,
            9.0, 10.0, 11.0, 12.0,
            13.0, 14.0, 15.0, 16.0,
        )
        seed_fields = {
            "label_attribute": "U_ATTRIB_C",
            "source_labels": (100, 200),
            "roi": 0.7,
            "values": {
                "U_ATTRIB_C": (100.0,) * 8 + (200.0,) * 8,
                "U_DISLOCDEN": disloc_values,
            },
        }

        updated_fields, stats, _mass_ledgers = update_seed_unode_fields(
            current_labels,
            target_labels,
            base_mesh_state,
            moved_mesh_state,
            seed_unodes,
            seed_fields,
        )

        updated_disloc = np.asarray(updated_fields["U_DISLOCDEN"], dtype=np.float64).reshape(4, 4)
        before_disloc = np.asarray(disloc_values, dtype=np.float64).reshape(4, 4)

        np.testing.assert_allclose(updated_disloc[2, :], 0.0)
        np.testing.assert_allclose(updated_disloc[[0, 1, 3], :], before_disloc[[0, 1, 3], :])
        self.assertEqual(stats["scalar_swept_unodes"], 4)
        self.assertGreaterEqual(stats["updated_scalar_unodes"], 4)

    def test_update_seed_unode_sections_uses_nearest_same_label_donor_for_u_euler_3(self) -> None:
        seed_unodes = {
            "grid_shape": (4, 4),
            "positions": (
                (0.125, 0.875),(0.125, 0.625),(0.125, 0.375),(0.125, 0.125),
                (0.375, 0.875),(0.375, 0.625),(0.375, 0.375),(0.375, 0.125),
                (0.625, 0.875),(0.625, 0.625),(0.625, 0.375),(0.625, 0.125),
                (0.875, 0.875),(0.875, 0.625),(0.875, 0.375),(0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
        }
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        euler_values = (
            (10.0, 100.0, 1000.0), (11.0, 101.0, 1001.0), (12.0, 102.0, 1002.0), (13.0, 103.0, 1003.0),
            (20.0, 200.0, 2000.0), (21.0, 201.0, 2001.0), (22.0, 202.0, 2002.0), (23.0, 203.0, 2003.0),
            (30.0, 300.0, 3000.0), (31.0, 301.0, 3001.0), (32.0, 302.0, 3002.0), (33.0, 303.0, 3003.0),
            (40.0, 400.0, 4000.0), (41.0, 401.0, 4001.0), (42.0, 402.0, 4002.0), (43.0, 403.0, 4003.0),
        )
        seed_sections = {
            "field_order": ("U_EULER_3",),
            "id_order": tuple(range(16)),
            "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
            "component_counts": {"U_EULER_3": 3},
            "values": {"U_EULER_3": euler_values},
        }

        updated_sections, stats = update_seed_unode_sections(
            current_labels,
            target_labels,
            seed_unodes,
            seed_sections,
        )

        updated = np.asarray(updated_sections["values"]["U_EULER_3"], dtype=np.float64).reshape(4, 4, 3)
        original = np.asarray(euler_values, dtype=np.float64).reshape(4, 4, 3)

        np.testing.assert_allclose(updated[2, :, :], original[1, :, :])
        np.testing.assert_allclose(updated[[0, 1, 3], :, :], original[[0, 1, 3], :, :])
        self.assertEqual(stats["updated_orientation_unodes"], 4)
        self.assertEqual(stats["fallback_orientation_unodes"], 0)

    def test_update_seed_unode_sections_uses_distance_weighted_fallback_when_no_donor_is_in_roi(self) -> None:
        seed_unodes = {
            "grid_shape": (16, 16),
            "positions": tuple(
                ((ix + 0.5) / 16.0, 1.0 - ((iy + 0.5) / 16.0))
                for ix in range(16)
                for iy in range(16)
            ),
            "grid_indices": tuple((ix, iy) for ix in range(16) for iy in range(16)),
        }
        current_labels = np.zeros((16, 16), dtype=np.int32)
        current_labels[:2, :2] = 1
        target_labels = current_labels.copy()
        target_labels[7:9, 7:9] = 1
        euler_values = tuple(
            (float(index), float(index + 1000.0), float(index + 2000.0))
            for index in range(16 * 16)
        )
        seed_sections = {
            "field_order": ("U_EULER_3",),
            "id_order": tuple(range(16 * 16)),
            "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
            "component_counts": {"U_EULER_3": 3},
            "values": {"U_EULER_3": euler_values},
        }

        updated_sections, stats = update_seed_unode_sections(
            current_labels,
            target_labels,
            seed_unodes,
            seed_sections,
        )

        original = np.asarray(euler_values, dtype=np.float64)
        updated = np.asarray(updated_sections["values"]["U_EULER_3"], dtype=np.float64)
        sample_points = np.asarray(seed_unodes["positions"], dtype=np.float64)
        sample_grid_indices = np.asarray(seed_unodes["grid_indices"], dtype=np.int32)
        sample_target = np.asarray(target_labels, dtype=np.int32)[
            sample_grid_indices[:, 0], sample_grid_indices[:, 1]
        ]
        sample_current = np.asarray(current_labels, dtype=np.int32)[
            sample_grid_indices[:, 0], sample_grid_indices[:, 1]
        ]
        changed_mask = sample_current != sample_target
        point_index = 7 * 16 + 7
        donor_mask = (sample_target == 1) & (~changed_mask)
        donor_indices = np.flatnonzero(donor_mask)
        deltas = sample_points[donor_indices] - sample_points[point_index]
        deltas = (deltas + 0.5) % 1.0 - 0.5
        distances = np.sqrt(np.sum(deltas * deltas, axis=1))
        expected = np.sum(original[donor_indices] * distances[:, None], axis=0) / np.sum(distances)

        np.testing.assert_allclose(updated[point_index], expected)
        self.assertEqual(stats["fallback_orientation_unodes"], 4)
        self.assertEqual(stats["old_value_fallback_orientation_unodes"], 0)
        self.assertEqual(stats["updated_orientation_unodes"], 4)

    def test_update_seed_unode_sections_keeps_old_orientation_when_no_valid_donor_exists(self) -> None:
        seed_unodes = {
            "grid_shape": (16, 16),
            "positions": (
                *(( (ix + 0.5) / 16.0, 1.0 - ((iy + 0.5) / 16.0) ) for ix in range(16) for iy in range(16)),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(16) for iy in range(16)),
        }
        current_labels = np.zeros((16, 16), dtype=np.int32)
        target_labels = current_labels.copy()
        target_labels[7:9, 7:9] = 1
        euler_values = tuple(
            (float(index), float(index + 100.0), float(index + 1000.0))
            for index in range(16 * 16)
        )
        seed_sections = {
            "field_order": ("U_EULER_3",),
            "id_order": tuple(range(16 * 16)),
            "defaults": {"U_EULER_3": (0.0, 0.0, 0.0)},
            "component_counts": {"U_EULER_3": 3},
            "values": {"U_EULER_3": euler_values},
        }

        updated_sections, stats = update_seed_unode_sections(
            current_labels,
            target_labels,
            seed_unodes,
            seed_sections,
        )

        original = np.asarray(euler_values, dtype=np.float64)
        updated = np.asarray(updated_sections["values"]["U_EULER_3"], dtype=np.float64)
        changed_indices = [7 * 16 + 7, 7 * 16 + 8, 8 * 16 + 7, 8 * 16 + 8]
        np.testing.assert_allclose(updated[changed_indices], original[changed_indices])
        self.assertEqual(stats["fallback_orientation_unodes"], 0)
        self.assertEqual(stats["old_value_fallback_orientation_unodes"], 4)
        self.assertEqual(stats["updated_orientation_unodes"], 0)

    def test_update_seed_unode_fields_shares_mass_partition_with_node_fields(self) -> None:
        base_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.5, "y": 0.0},
                {"x": 0.5, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4], "elle_option_boundarywidth": 0.1, "elle_option_unitlength": 1.0},
            "events": [],
        }
        moved_mesh_state = {
            "nodes": [
                {"x": 0.0, "y": 0.0},
                {"x": 0.75, "y": 0.0},
                {"x": 0.75, "y": 1.0},
                {"x": 0.0, "y": 1.0},
                {"x": 1.0, "y": 0.0},
                {"x": 1.0, "y": 1.0},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"grid_shape": [4, 4], "elle_option_boundarywidth": 0.1, "elle_option_unitlength": 1.0},
            "events": [],
        }
        seed_unodes = {
            "grid_shape": (4, 4),
            "positions": (
                (0.125, 0.875),(0.125, 0.625),(0.125, 0.375),(0.125, 0.125),
                (0.375, 0.875),(0.375, 0.625),(0.375, 0.375),(0.375, 0.125),
                (0.625, 0.875),(0.625, 0.625),(0.625, 0.375),(0.625, 0.125),
                (0.875, 0.875),(0.875, 0.625),(0.875, 0.375),(0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
        }
        current_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        target_labels = np.array(
            [
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [0, 0, 0, 0],
                [1, 1, 1, 1],
            ],
            dtype=np.int32,
        )
        conc_values = (
            1.0, 1.0, 1.0, 1.0,
            2.0, 2.0, 2.0, 2.0,
            10.0, 20.0, 30.0, 40.0,
            50.0, 60.0, 70.0, 80.0,
        )
        seed_fields = {
            "label_attribute": "U_ATTRIB_C",
            "source_labels": (100, 200),
            "roi": 0.7,
            "unode_area": 1.0 / 16.0,
            "values": {
                "U_ATTRIB_C": (100.0,) * 8 + (200.0,) * 8,
                "U_CONC_A": conc_values,
            },
        }
        node_fields = {
            "field_order": ("N_CONC_A",),
            "positions": tuple((float(node["x"]), float(node["y"])) for node in base_mesh_state["nodes"]),
            "values": {"N_CONC_A": (2.0, 2.0, 2.0, 2.0, 2.0, 2.0)},
            "roi": 0.7,
        }

        updated_fields, stats, mass_ledgers = update_seed_unode_fields(
            current_labels,
            target_labels,
            base_mesh_state,
            moved_mesh_state,
            seed_unodes,
            seed_fields,
            node_fields=node_fields,
        )

        ledger = mass_ledgers["U_CONC_A"]
        total_put_mass = sum(float(entry["put_mass"]) for entry in ledger.values())
        total_swept_mass = sum(float(entry["swept_mass"]) for entry in ledger.values())
        before_mass = float(np.sum(np.asarray(conc_values, dtype=np.float64)) * (1.0 / 16.0))
        after_mass = float(np.sum(np.asarray(updated_fields["U_CONC_A"], dtype=np.float64)) * (1.0 / 16.0))

        self.assertEqual(stats["mass_partitioned_fields"], 1)
        self.assertIn("U_CONC_A", mass_ledgers)
        self.assertGreater(total_put_mass, 0.0)
        self.assertGreater(total_swept_mass, 0.0)
        self.assertAlmostEqual(after_mass, before_mass - total_swept_mass + total_put_mass, places=8)
        self.assertTrue(
            all(
                abs(
                    float(node_entry["mass_chge_s"])
                    + float(node_entry["mass_chge_b"])
                    + float(node_entry["mass_chge_e"])
                )
                <= 1.0e-8
                for node_entry in ledger.values()
            )
        )
        self.assertTrue(
            all(
                any(bool(entry["partition_active"]) for entry in node_entry["entries"])
                for node_entry in ledger.values()
            )
        )
        self.assertTrue(
            all(str(node_entry["partition_mode"]) == "elle_partitionmass" for node_entry in ledger.values())
        )

        mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.0, "y": 0.0, "flynns": [0], "neighbors": [1, 3]},
                {"node_id": 1, "x": 0.75, "y": 0.0, "flynns": [0, 1], "neighbors": [0, 2, 4]},
                {"node_id": 2, "x": 0.75, "y": 1.0, "flynns": [0, 1], "neighbors": [1, 3, 5]},
                {"node_id": 3, "x": 0.0, "y": 1.0, "flynns": [0], "neighbors": [0, 2]},
                {"node_id": 4, "x": 1.0, "y": 0.0, "flynns": [1], "neighbors": [1, 5]},
                {"node_id": 5, "x": 1.0, "y": 1.0, "flynns": [1], "neighbors": [2, 4]},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [1, 4, 5, 2]},
            ],
            "stats": {"elle_option_boundarywidth": 0.1, "elle_option_unitlength": 1.0},
            "events": [],
        }
        updated_nodes, node_stats = update_seed_node_fields(
            mesh_state,
            seed_unodes,
            target_labels,
            node_fields,
            updated_fields,
            mass_ledgers=mass_ledgers,
        )

        self.assertEqual(node_stats["partitioned_node_fields"], 1)
        node_values = np.asarray(updated_nodes["N_CONC_A"], dtype=np.float64)
        self.assertAlmostEqual(float(node_values[1]), float(ledger[1]["concentration"]), places=8)
        self.assertAlmostEqual(float(node_values[2]), float(ledger[2]["concentration"]), places=8)

    def test_apply_segment_mass_partition_prefers_segments_with_more_swept_mass(self) -> None:
        sample_points = np.array(
            [
                (0.20, 0.50),
                (0.80, 0.50),
                (0.30, 0.50),
                (0.70, 0.50),
            ],
            dtype=np.float64,
        )
        source_values = np.array([10.0, 1.0, 0.0, 0.0], dtype=np.float64)
        updated_values = np.zeros(4, dtype=np.float64)
        transfer_records = [
            {
                "area": 1.0,
                "mask": np.array([True, False, True, False], dtype=bool),
                "origin": np.array([0.20, 0.50], dtype=np.float64),
                "tip": np.array([0.30, 0.50], dtype=np.float64),
            },
            {
                "area": 1.0,
                "mask": np.array([False, True, False, True], dtype=bool),
                "origin": np.array([0.80, 0.50], dtype=np.float64),
                "tip": np.array([0.70, 0.50], dtype=np.float64),
            },
        ]
        changed_mask = np.array([False, False, True, True], dtype=bool)
        swept_mask = np.array([True, True, False, False], dtype=bool)

        adjusted, residual = _apply_segment_mass_partition(
            source_values,
            updated_values,
            transfer_records,
            changed_mask=changed_mask,
            swept_mask=swept_mask,
            sample_points=sample_points,
            unode_area=0.25,
            roi=0.2,
        )

        self.assertAlmostEqual(float(residual), 0.0, places=9)
        self.assertAlmostEqual(float(adjusted[2]), float(adjusted[3]), places=8)
        self.assertAlmostEqual(
            float(np.sum(adjusted) * 0.25),
            float(np.sum(source_values) * 0.25),
            places=9,
        )

    def test_apply_segment_mass_partition_is_order_independent_for_overlapping_sweeps(self) -> None:
        sample_points = np.array(
            [
                (0.50, 0.50),
                (0.25, 0.50),
                (0.75, 0.50),
            ],
            dtype=np.float64,
        )
        source_values = np.array([8.0, 0.0, 0.0], dtype=np.float64)
        updated_values = np.array([8.0, 0.0, 0.0], dtype=np.float64)
        record_a = {
            "area": 1.0,
            "mask": np.array([True, True, False], dtype=bool),
            "origin": np.array([0.50, 0.50], dtype=np.float64),
            "tip": np.array([0.25, 0.50], dtype=np.float64),
        }
        record_b = {
            "area": 1.0,
            "mask": np.array([True, False, True], dtype=bool),
            "origin": np.array([0.50, 0.50], dtype=np.float64),
            "tip": np.array([0.75, 0.50], dtype=np.float64),
        }
        changed_mask = np.array([False, True, True], dtype=bool)
        swept_mask = np.array([True, False, False], dtype=bool)

        adjusted_ab, residual_ab = _apply_segment_mass_partition(
            source_values,
            updated_values,
            [record_a, record_b],
            changed_mask=changed_mask,
            swept_mask=swept_mask,
            sample_points=sample_points,
            unode_area=1.0 / 3.0,
            roi=0.3,
        )
        adjusted_ba, residual_ba = _apply_segment_mass_partition(
            source_values,
            updated_values,
            [record_b, record_a],
            changed_mask=changed_mask,
            swept_mask=swept_mask,
            sample_points=sample_points,
            unode_area=1.0 / 3.0,
            roi=0.3,
        )

        np.testing.assert_allclose(adjusted_ab, adjusted_ba, atol=1.0e-9)
        self.assertAlmostEqual(float(residual_ab), float(residual_ba), places=9)

    def test_apply_segment_mass_partition_uses_enrich_mask_for_put_mass(self) -> None:
        sample_points = np.array(
            [
                (0.20, 0.50),
                (0.70, 0.50),
                (0.80, 0.50),
            ],
            dtype=np.float64,
        )
        source_values = np.array([6.0, 0.0, 0.0], dtype=np.float64)
        updated_values = np.zeros(3, dtype=np.float64)
        transfer_records = [
            {
                "area": 1.0,
                "mask": np.array([True, True, True], dtype=bool),
                "sweep_mask": np.array([True, False, False], dtype=bool),
                "enrich_mask": np.array([False, True, False], dtype=bool),
                "reassigned_mask": np.array([False, False, False], dtype=bool),
                "origin": np.array([0.20, 0.50], dtype=np.float64),
                "tip": np.array([0.70, 0.50], dtype=np.float64),
            },
        ]
        changed_mask = np.array([False, True, True], dtype=bool)
        swept_mask = np.array([True, False, False], dtype=bool)

        adjusted, residual = _apply_segment_mass_partition(
            source_values,
            updated_values,
            transfer_records,
            changed_mask=changed_mask,
            swept_mask=swept_mask,
            sample_points=sample_points,
            unode_area=1.0 / 3.0,
            roi=0.4,
        )

        self.assertAlmostEqual(float(residual), 0.0, places=9)
        self.assertGreater(float(adjusted[1]), 0.0)
        self.assertAlmostEqual(float(adjusted[2]), 0.0, places=9)

    def test_apply_segment_mass_partition_uses_enrich_support_for_put_mass(self) -> None:
        sample_points = np.array(
            [
                (0.20, 0.50),
                (0.68, 0.50),
                (0.82, 0.50),
            ],
            dtype=np.float64,
        )
        source_values = np.array([6.0, 0.0, 0.0], dtype=np.float64)
        updated_values = np.zeros(3, dtype=np.float64)
        transfer_records = [
            {
                "area": 1.0,
                "mask": np.array([True, True, True], dtype=bool),
                "sweep_mask": np.array([True, False, False], dtype=bool),
                "enrich_mask": np.array([False, True, True], dtype=bool),
                "reassigned_mask": np.array([False, False, False], dtype=bool),
                "origin": np.array([0.20, 0.50], dtype=np.float64),
                "tip": np.array([0.68, 0.50], dtype=np.float64),
            },
        ]
        changed_mask = np.array([False, True, False], dtype=bool)
        swept_mask = np.array([True, False, False], dtype=bool)

        adjusted, residual = _apply_segment_mass_partition(
            source_values,
            updated_values,
            transfer_records,
            changed_mask=changed_mask,
            swept_mask=swept_mask,
            sample_points=sample_points,
            unode_area=1.0 / 3.0,
            roi=0.4,
        )

        self.assertAlmostEqual(float(residual), 0.0, places=9)
        self.assertGreater(float(adjusted[1]), 0.0)
        self.assertGreater(float(adjusted[2]), 0.0)

    def test_apply_segment_mass_partition_prefers_stronger_enrich_weights(self) -> None:
        sample_points = np.array(
            [
                (0.20, 0.50),
                (0.80, 0.50),
                (0.68, 0.50),
                (0.95, 0.50),
            ],
            dtype=np.float64,
        )
        source_values = np.array([6.0, 6.0, 0.0, 0.0], dtype=np.float64)
        updated_values = np.zeros(4, dtype=np.float64)
        transfer_records = [
            {
                "node_id": 0,
                "area": 1.0,
                "mask": np.array([True, False, True, False], dtype=bool),
                "sweep_mask": np.array([True, False, False, False], dtype=bool),
                "enrich_mask": np.array([False, False, True, False], dtype=bool),
                "reassigned_mask": np.array([False, False, False, False], dtype=bool),
                "origin": np.array([0.20, 0.50], dtype=np.float64),
                "tip": np.array([0.68, 0.50], dtype=np.float64),
            },
            {
                "node_id": 0,
                "area": 1.0,
                "mask": np.array([False, True, False, True], dtype=bool),
                "sweep_mask": np.array([False, True, False, False], dtype=bool),
                "enrich_mask": np.array([False, False, False, True], dtype=bool),
                "reassigned_mask": np.array([False, False, False, False], dtype=bool),
                "origin": np.array([0.80, 0.50], dtype=np.float64),
                "tip": np.array([0.70, 0.50], dtype=np.float64),
            },
        ]
        changed_mask = np.array([False, False, True, True], dtype=bool)
        swept_mask = np.array([True, True, False, False], dtype=bool)

        adjusted, residual = _apply_segment_mass_partition(
            source_values,
            updated_values,
            transfer_records,
            changed_mask=changed_mask,
            swept_mask=swept_mask,
            sample_points=sample_points,
            unode_area=0.25,
            roi=0.35,
        )

        self.assertAlmostEqual(float(residual), 0.0, places=9)
        self.assertGreater(float(adjusted[2]), float(adjusted[3]))

    def test_apply_segment_mass_partition_tracks_boundary_mass_per_segment(self) -> None:
        sample_points = np.array(
            [
                (0.20, 0.50),
                (0.80, 0.50),
                (0.35, 0.50),
                (0.85, 0.50),
            ],
            dtype=np.float64,
        )
        source_values = np.array([6.0, 6.0, 0.0, 0.0], dtype=np.float64)
        updated_values = np.zeros(4, dtype=np.float64)
        transfer_records = [
            {
                "node_id": 0,
                "neighbor_id": 1,
                "area": 1.0,
                "mask": np.array([True, False, True, False], dtype=bool),
                "sweep_mask": np.array([True, False, False, False], dtype=bool),
                "enrich_mask": np.array([False, False, True, False], dtype=bool),
                "reassigned_mask": np.array([False, False, False, False], dtype=bool),
                "origin": np.array([0.20, 0.50], dtype=np.float64),
                "tip": np.array([0.35, 0.50], dtype=np.float64),
                "old_length": 1.0,
                "new_length": 0.5,
            },
            {
                "node_id": 0,
                "neighbor_id": 2,
                "area": 1.0,
                "mask": np.array([False, True, False, True], dtype=bool),
                "sweep_mask": np.array([False, True, False, False], dtype=bool),
                "enrich_mask": np.array([False, False, False, True], dtype=bool),
                "reassigned_mask": np.array([False, False, False, False], dtype=bool),
                "origin": np.array([0.80, 0.50], dtype=np.float64),
                "tip": np.array([0.85, 0.50], dtype=np.float64),
                "old_length": 1.0,
                "new_length": 1.5,
            },
        ]
        changed_mask = np.array([False, False, True, True], dtype=bool)
        swept_mask = np.array([True, True, False, False], dtype=bool)

        adjusted, residual, ledger = _apply_segment_mass_partition(
            source_values,
            updated_values,
            transfer_records,
            changed_mask=changed_mask,
            swept_mask=swept_mask,
            sample_points=sample_points,
            unode_area=0.25,
            roi=0.3,
            node_values=np.array([2.0], dtype=np.float64),
            boundary_scale=0.5,
            return_ledger=True,
        )

        self.assertAlmostEqual(float(residual), 0.0, places=9)
        self.assertIn(0, ledger)
        entries = sorted(ledger[0]["entries"], key=lambda entry: entry["neighbor_id"])
        self.assertEqual(len(entries), 2)
        self.assertGreater(float(entries[1]["gb_area_f"]), float(entries[0]["gb_area_f"]))
        self.assertGreater(float(entries[1]["gb_mass_f"]), float(entries[0]["gb_mass_f"]))
        self.assertTrue(bool(entries[0]["partition_active"]))
        self.assertTrue(bool(entries[1]["partition_active"]))
        self.assertGreater(float(entries[0]["conc_b"]), 0.0)
        self.assertGreater(float(entries[0]["conc_s"]), 0.0)
        self.assertGreater(float(entries[0]["conc_b_f"]), 0.0)
        self.assertAlmostEqual(float(ledger[0]["total_source_mass"]), 4.0, places=9)
        self.assertAlmostEqual(float(ledger[0]["total_capacity"]), 2.5, places=9)
        self.assertEqual(str(ledger[0]["partition_mode"]), "elle_partitionmass")
        self.assertAlmostEqual(float(ledger[0]["concentration"]), 1.7777777777777777, places=9)
        self.assertAlmostEqual(float(ledger[0]["partitionmass_put_mass"]), 3.5555555555555554, places=9)
        self.assertAlmostEqual(float(ledger[0]["partitionmass_gb_mass_f"]), 0.4444444444444444, places=9)
        self.assertAlmostEqual(float(ledger[0]["mass_chge_s"]), -1.767676767676768, places=9)
        self.assertAlmostEqual(float(ledger[0]["mass_chge_b"]), -0.23232323232323238, places=9)
        self.assertAlmostEqual(float(ledger[0]["mass_chge_e"]), 2.0, places=9)
        self.assertAlmostEqual(
            float(ledger[0]["mass_chge_s"] + ledger[0]["mass_chge_b"] + ledger[0]["mass_chge_e"]),
            0.0,
            places=9,
        )
        self.assertAlmostEqual(float(entries[0]["mass_chge_b"]), -0.2777777777777778, places=9)
        self.assertAlmostEqual(float(entries[1]["mass_chge_b"]), 0.045454545454545414, places=9)
        self.assertAlmostEqual(float(entries[0]["mass_chge_e"]), 1.0, places=9)
        self.assertAlmostEqual(float(entries[1]["mass_chge_e"]), 1.0000000000000002, places=9)
        self.assertAlmostEqual(float(entries[0]["partitionmass_put_mass"]), 1.7777777777777777, places=9)
        self.assertAlmostEqual(float(entries[1]["partitionmass_put_mass"]), 1.7777777777777777, places=9)
        self.assertAlmostEqual(
            float(entries[0]["mass_chge_s"] + entries[0]["mass_chge_b"] + entries[0]["mass_chge_e"]),
            0.0,
            places=9,
        )
        self.assertAlmostEqual(
            float(entries[1]["mass_chge_s"] + entries[1]["mass_chge_b"] + entries[1]["mass_chge_e"]),
            0.0,
            places=9,
        )
        self.assertAlmostEqual(float(np.sum(adjusted) * 0.25), 3.5555555555555554, places=9)

    def test_entry_partition_terms_uses_explicit_secondary_weight_channel(self) -> None:
        terms = _entry_partition_terms(
            {
                "area": 1.0,
                "gb_area_f": 0.5,
                "gb_mass_i": 1.0,
                "swept_mass_0": 4.0,
                "swept_mass_1": 3.0,
                "enrich_mass_0": 6.0,
                "enrich_mass_1": 1.5,
                "sweep_weight_total_0": 4.0,
                "sweep_weight_total_1": 1.0,
                "sweep_weight_total_2": 2.0,
                "enrich_weight_total_0": 3.0,
                "enrich_weight_total_1": 0.5,
                "enrich_weight_total_2": 1.0,
            },
            unode_area=0.5,
        )

        self.assertAlmostEqual(float(terms["conc_s"]), 2.0, places=8)
        self.assertAlmostEqual(float(terms["conc_s1"]), 6.0, places=8)
        self.assertAlmostEqual(float(terms["conc_e"]), 4.0, places=8)
        self.assertAlmostEqual(float(terms["conc_e1"]), 6.0, places=8)
        self.assertAlmostEqual(float(terms["swept_area_frac"]), 2.0 / 3.0, places=8)
        self.assertAlmostEqual(float(terms["enrich_area_frac"]), 1.0 / 3.0, places=8)
        self.assertTrue(bool(terms["partition_active"]))

    def test_partition_mass_node_matches_original_formula(self) -> None:
        partition = _partition_mass_node(
            [
                {
                    "neighbor_id": 1,
                    "area": 1.0,
                    "gb_area_i": 0.25,
                    "gb_area_f": 0.125,
                    "gb_mass_i": 0.5,
                    "swept_mass": 0.75,
                },
                {
                    "neighbor_id": 2,
                    "area": 2.0,
                    "gb_area_i": 0.25,
                    "gb_area_f": 0.375,
                    "gb_mass_i": 0.5,
                    "swept_mass": 1.25,
                },
            ]
        )

        total_mass = 3.0
        total_boundary_area = 0.25
        expected_gb_mass_f = total_mass * total_boundary_area / (3.0 + total_boundary_area)
        expected_put_mass = total_mass - expected_gb_mass_f
        self.assertAlmostEqual(float(partition["gb_mass_f"]), expected_gb_mass_f, places=8)
        self.assertAlmostEqual(float(partition["total_put_mass"]), expected_put_mass, places=8)
        self.assertAlmostEqual(float(partition["entry_put_mass"][0]), expected_put_mass / 3.0, places=8)
        self.assertAlmostEqual(float(partition["entry_put_mass"][1]), expected_put_mass * 2.0 / 3.0, places=8)
        self.assertAlmostEqual(float(partition["concentration"]), expected_gb_mass_f / total_boundary_area, places=8)

    def test_segment_swept_records_subdivide_triangle_area_into_increments(self) -> None:
        base_mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.20, "y": 0.20},
                {"node_id": 1, "x": 0.20, "y": 0.40},
                {"node_id": 2, "x": 0.50, "y": 0.20},
            ],
            "flynns": [
                {"flynn_id": 10, "label": 10, "node_ids": [0, 1, 2]},
            ],
        }
        moved_mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.30, "y": 0.20},
                {"node_id": 1, "x": 0.20, "y": 0.40},
                {"node_id": 2, "x": 0.50, "y": 0.20},
            ],
            "flynns": [
                {"flynn_id": 10, "label": 10, "node_ids": [0, 1, 2]},
            ],
        }
        sample_points = np.array(
            [
                (0.22, 0.24),
                (0.24, 0.28),
                (0.26, 0.32),
                (0.80, 0.80),
            ],
            dtype=np.float64,
        )
        sample_current = np.array([0, 0, 1, 1], dtype=np.int32)
        sample_target = np.array([1, 1, 1, 1], dtype=np.int32)

        records = _segment_swept_records(
            base_mesh_state,
            moved_mesh_state,
            sample_points,
            sample_current=sample_current,
            sample_target=sample_target,
        )
        edge_records = [
            record
            for record in records
            if int(record["node_id"]) == 0 and int(record["neighbor_id"]) == 1
        ]

        self.assertEqual(len(edge_records), 20)
        self.assertAlmostEqual(
            float(sum(float(record["area"]) for record in edge_records)),
            0.01,
            places=8,
        )
        combined_mask = np.zeros(sample_points.shape[0], dtype=bool)
        unique_masks: set[tuple[bool, ...]] = set()
        for record in edge_records:
            record_mask = np.asarray(record["mask"], dtype=bool)
            combined_mask = combined_mask | record_mask
            unique_masks.add(tuple(bool(value) for value in record_mask.tolist()))
        self.assertTrue(np.array_equal(combined_mask, np.array([True, True, False, False], dtype=bool)))
        self.assertGreater(len(unique_masks), 1)
        self.assertTrue(any(np.asarray(record["reassigned_mask"], dtype=bool).any() for record in edge_records))
        self.assertGreater(float(edge_records[0]["area"]), float(edge_records[-1]["area"]))
        self.assertAlmostEqual(float(edge_records[0]["origin"][0]), 0.20, places=8)
        self.assertAlmostEqual(float(edge_records[-1]["origin"][0]), 0.20, places=8)
        self.assertAlmostEqual(float(edge_records[0]["tip"][0]), 0.2975, places=8)
        self.assertAlmostEqual(float(edge_records[-1]["tip"][0]), 0.2025, places=8)

    def test_update_seed_node_fields_uses_matching_unode_field_values(self) -> None:
        seed_unodes = {
            "positions": (
                (0.20, 0.50),
                (0.30, 0.50),
                (0.70, 0.50),
                (0.80, 0.50),
            ),
            "grid_indices": (
                (0, 0),
                (0, 1),
                (1, 0),
                (1, 1),
            ),
            "grid_shape": (2, 2),
        }
        target_labels = np.array([[0, 0], [1, 1]], dtype=np.int32)
        mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.25, "y": 0.50, "flynns": [10]},
                {"node_id": 1, "x": 0.75, "y": 0.50, "flynns": [20]},
            ],
            "flynns": [
                {"flynn_id": 10, "label": 0, "node_ids": [0, 1, 1]},
                {"flynn_id": 20, "label": 1, "node_ids": [0, 1, 1]},
            ],
            "stats": {},
            "events": [],
        }
        node_fields = {
            "field_order": ("N_ATTRIB_A",),
            "positions": ((0.20, 0.50), (0.80, 0.50)),
            "values": {"N_ATTRIB_A": (0.0, 0.0)},
            "roi": 0.15,
        }
        unode_fields = {"U_ATTRIB_A": (1.0, 1.0, 5.0, 5.0)}

        updated, stats = update_seed_node_fields(
            mesh_state,
            seed_unodes,
            target_labels,
            node_fields,
            unode_fields,
        )

        values = np.asarray(updated["N_ATTRIB_A"], dtype=np.float64)
        self.assertEqual(values.shape, (2,))
        self.assertAlmostEqual(float(values[0]), 1.0, places=5)
        self.assertAlmostEqual(float(values[1]), 5.0, places=5)
        self.assertGreater(stats["updated_node_values"], 0)

    def test_update_seed_node_fields_partitions_mass_for_n_conc(self) -> None:
        seed_unodes = {
            "positions": (
                (0.125, 0.875), (0.125, 0.625), (0.125, 0.375), (0.125, 0.125),
                (0.375, 0.875), (0.375, 0.625), (0.375, 0.375), (0.375, 0.125),
                (0.625, 0.875), (0.625, 0.625), (0.625, 0.375), (0.625, 0.125),
                (0.875, 0.875), (0.875, 0.625), (0.875, 0.375), (0.875, 0.125),
            ),
            "grid_indices": tuple((ix, iy) for ix in range(4) for iy in range(4)),
            "grid_shape": (4, 4),
        }
        target_labels = np.zeros((4, 4), dtype=np.int32)
        mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.40, "y": 0.20, "flynns": [10, 20], "neighbors": [1]},
                {"node_id": 1, "x": 0.20, "y": 0.80, "flynns": [10, 20], "neighbors": [0]},
            ],
            "flynns": [
                {"flynn_id": 10, "label": 0, "node_ids": [0, 1, 1]},
                {"flynn_id": 20, "label": 0, "node_ids": [0, 1, 1]},
            ],
            "stats": {
                "elle_option_boundarywidth": 0.1,
                "elle_option_unitlength": 1.0,
            },
            "events": [],
        }
        node_fields = {
            "field_order": ("N_CONC_A",),
            "positions": ((0.20, 0.20), (0.20, 0.80)),
            "values": {"N_CONC_A": (1.0, 1.0)},
            "roi": 0.2,
        }
        unode_fields = {
            "U_CONC_A": (
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 5.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
                0.0, 0.0, 0.0, 0.0,
            )
        }

        updated, stats = update_seed_node_fields(
            mesh_state,
            seed_unodes,
            target_labels,
            node_fields,
            unode_fields,
        )

        values = np.asarray(updated["N_CONC_A"], dtype=np.float64)
        self.assertEqual(values.shape, (2,))
        self.assertGreater(float(values[0]), 0.0)
        self.assertNotAlmostEqual(float(values[0]), 1.0, places=5)
        self.assertGreater(float(values[1]), 0.0)
        self.assertNotAlmostEqual(float(values[1]), 1.0, places=5)
        self.assertEqual(stats["partitioned_node_fields"], 1)

    def test_relax_mesh_state_resolves_coincident_nodes_in_topocheck_sequence(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.10, "y": 0.10},
                {"x": 0.10, "y": 0.10},
                {"x": 0.90, "y": 0.10},
                {"x": 0.90, "y": 0.90},
                {"x": 0.10, "y": 0.90},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3, 4]},
            ],
            "stats": {"grid_shape": [64, 64]},
            "events": [],
        }

        relaxed = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.05,
                min_node_separation_factor=0.0,
                max_node_separation_factor=100.0,
            ),
        )

        self.assertGreater(relaxed["stats"]["mesh_coincident_nodes_resolved"], 0)
        self.assertTrue(any(event["type"] == "resolve_coincident_nodes" for event in relaxed["events"]))

    def test_relax_mesh_state_deletes_small_flynn_in_topocheck_sequence(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.05, "y": 0.05},
                {"x": 0.15, "y": 0.20},
                {"x": 0.15, "y": 0.80},
                {"x": 0.05, "y": 0.95},
                {"x": 0.95, "y": 0.95},
                {"x": 0.95, "y": 0.05},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 5, 4, 3]},
                {"flynn_id": 1, "label": 1, "node_ids": [0, 1, 2, 3]},
            ],
            "stats": {"grid_shape": [128, 128]},
            "events": [],
        }

        relaxed = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.005,
                min_node_separation_factor=0.0,
                max_node_separation_factor=100.0,
            ),
        )

        self.assertEqual(relaxed["stats"]["num_flynns"], 1)
        self.assertGreater(relaxed["stats"]["mesh_deleted_small_flynns"], 0)
        self.assertTrue(any(event["type"] == "delete_small_flynn" for event in relaxed["events"]))

    def test_delete_single_nodes_local_removes_single_node(self) -> None:
        nodes = np.array(
            [
                [0.10, 0.10],
                [0.20, 0.10],
                [0.25, 0.20],
                [0.15, 0.25],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 1, "label": 1, "node_ids": [0, 1, 2, 3, 1]},
        ]

        updated_nodes, updated_flynns, events, deleted = mesh_module._delete_single_nodes_local(
            nodes,
            flynns,
            focus_nodes={0},
        )

        self.assertEqual(deleted, 1)
        self.assertEqual(len(updated_nodes), 3)
        self.assertEqual(len(updated_flynns), 1)
        self.assertEqual(updated_flynns[0]["node_ids"], [0, 1, 2])
        self.assertEqual(events[0]["type"], "delete_single")

    def test_check_double_node_local_deletes_short_gap_node(self) -> None:
        nodes = np.array(
            [
                [0.00, 0.00],
                [0.01, 0.00],
                [0.50, 0.00],
                [0.50, 0.50],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
        ]

        updated_nodes, updated_flynns, events, inserted, removed = mesh_module._check_double_node_local(
            nodes,
            flynns,
            node_id=1,
            switch_distance=0.1,
            min_node_separation=0.1,
            max_node_separation=100.0,
        )

        self.assertEqual(inserted, 0)
        self.assertEqual(removed, 1)
        self.assertEqual(len(updated_nodes), 3)
        self.assertEqual(updated_flynns[0]["node_ids"], [0, 1, 2])
        self.assertEqual(events[0]["type"], "delete_double")
        self.assertEqual(events[0]["reason"], "gap_too_small")

    def test_check_triple_node_local_respects_phase_boundary_switch_gate(self) -> None:
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.52, 0.50],
                [0.46, 0.58],
                [0.46, 0.42],
                [0.58, 0.58],
                [0.58, 0.42],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [2, 0, 1, 4]},
            {"flynn_id": 1, "label": 1, "node_ids": [2, 0, 3]},
            {"flynn_id": 2, "label": 2, "node_ids": [3, 0, 1, 5]},
            {"flynn_id": 3, "label": 3, "node_ids": [4, 1, 5]},
        ]

        blocked_nodes, blocked_flynns, blocked_events, _, _, blocked_switched, blocked_rejected = mesh_module._check_triple_node_local(
            nodes.copy(),
            copy.deepcopy(flynns),
            node_id=0,
            switch_distance=0.05,
            min_node_separation=0.01,
            max_node_separation=100.0,
            phase_lookup={0: 1, 1: 1, 2: 2, 3: 2},
        )

        self.assertEqual(blocked_switched, 0)
        self.assertGreaterEqual(blocked_rejected, 1)
        self.assertEqual(len(blocked_events), 1)
        self.assertEqual(blocked_events[0]["type"], "reject_triple_switch")
        self.assertEqual(blocked_events[0]["reason"], "phase_boundary_switch_forbidden")
        self.assertEqual(blocked_flynns[0]["node_ids"], [2, 0, 1, 4])

        switched_nodes, switched_flynns, switched_events, _, _, switched_count, switched_rejected = mesh_module._check_triple_node_local(
            nodes.copy(),
            copy.deepcopy(flynns),
            node_id=0,
            switch_distance=0.05,
            min_node_separation=0.01,
            max_node_separation=100.0,
            phase_lookup={0: 1, 1: 1, 2: 1, 3: 1},
        )

        self.assertEqual(switched_rejected, 0)
        self.assertEqual(switched_count, 1)
        self.assertEqual(switched_events[0]["reason"], "check_triple_switch_gap")
        self.assertNotEqual(switched_flynns[0]["node_ids"], [2, 0, 1, 4])

    def test_check_triple_node_local_rejects_same_flynn_neighborhood_switch(self) -> None:
        nodes = np.array(
            [
                [0.45, 0.50],
                [0.55, 0.50],
                [0.40, 0.60],
                [0.40, 0.40],
                [0.60, 0.60],
                [0.60, 0.40],
                [0.50, 0.30],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 10, "label": 10, "node_ids": [2, 0, 1, 4]},
            {"flynn_id": 11, "label": 11, "node_ids": [3, 0, 1, 5]},
            {"flynn_id": 12, "label": 12, "node_ids": [2, 0, 3, 6, 5, 1, 4]},
        ]

        updated_nodes, updated_flynns, events, _, _, switched_count, rejected_count = mesh_module._check_triple_node_local(
            nodes.copy(),
            copy.deepcopy(flynns),
            node_id=0,
            switch_distance=0.2,
            min_node_separation=0.01,
            max_node_separation=100.0,
            phase_lookup={10: 1, 11: 1, 12: 1},
        )

        self.assertEqual(switched_count, 0)
        self.assertGreaterEqual(rejected_count, 1)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "reject_triple_switch")
        self.assertEqual(events[0]["reason"], "same_flynn_neighborhood")
        self.assertEqual(updated_flynns[0]["node_ids"], [2, 0, 1, 4])
        np.testing.assert_allclose(updated_nodes, nodes, atol=1.0e-12)

    def test_nearest_short_triple_candidate_prefers_nearest_triple_neighbor(self) -> None:
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.52, 0.50],
                [0.54, 0.50],
                [0.50, 0.60],
                [0.60, 0.60],
                [0.60, 0.40],
                [0.64, 0.60],
                [0.64, 0.40],
                [0.50, 0.70],
            ],
            dtype=np.float64,
        )

        candidate = mesh_module._nearest_short_triple_candidate(
            nodes,
            [],
            0,
            0.05,
            node_neighbors={
                0: {1, 2, 3},
                1: {0, 4, 5},
                2: {0, 6, 7},
                3: {0, 8},
                4: {1},
                5: {1},
                6: {2},
                7: {2},
                8: {3},
            },
            edge_map={
                (0, 1): [(0, 0), (1, 0)],
                (0, 2): [(0, 1), (2, 0)],
                (0, 3): [(0, 2), (3, 0)],
            },
        )

        self.assertIsNotNone(candidate)
        self.assertEqual(candidate["node_id"], 0)
        self.assertEqual(candidate["target_neighbor"], 1)
        self.assertEqual(candidate["edge"], (0, 1))
        self.assertAlmostEqual(candidate["edge_length"], 0.02)

    def test_legacy_triple_switch_context_derives_old_style_neighbors_and_flynns(self) -> None:
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [2, 0, 1, 4]},
            {"flynn_id": 1, "label": 1, "node_ids": [2, 0, 3]},
            {"flynn_id": 2, "label": 2, "node_ids": [3, 0, 1, 5]},
            {"flynn_id": 3, "label": 3, "node_ids": [4, 1, 5]},
        ]

        context, reason = mesh_module._legacy_triple_switch_context(
            copy.deepcopy(flynns),
            0,
            1,
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(context)
        self.assertEqual(context["shared_indices"], [0, 2])
        self.assertEqual(context["exclusive_a_index"], 1)
        self.assertEqual(context["exclusive_b_index"], 3)
        self.assertEqual(
            context["legacy_switch_neighbors"],
            {"node3": 2, "node4": 3, "node5": 4, "node6": 5},
        )
        self.assertEqual(
            context["legacy_switch_flynn_indices"],
            {"full_id_0": 1, "full_id_1": 0, "full_id_2": 3, "full_id_3": 2},
        )
        self.assertEqual(
            context["legacy_switch_flynn_ids"],
            {"full_id_0": 1, "full_id_1": 0, "full_id_2": 3, "full_id_3": 2},
        )

    def test_legacy_rewrite_triple_switch_polygons_rewrites_old_style_four_flynn_context(self) -> None:
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [2, 0, 1, 4]},
            {"flynn_id": 1, "label": 1, "node_ids": [2, 0, 3]},
            {"flynn_id": 2, "label": 2, "node_ids": [3, 0, 1, 5]},
            {"flynn_id": 3, "label": 3, "node_ids": [4, 1, 5]},
        ]

        context, reason = mesh_module._legacy_triple_switch_context(
            copy.deepcopy(flynns),
            0,
            1,
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(context)
        rewritten, new_neighbors, rewrite_reason = mesh_module._legacy_rewrite_triple_switch_polygons(
            copy.deepcopy(flynns),
            0,
            1,
            legacy_context=context,
            variant_name="switch_triple_a0b1",
        )

        self.assertIsNone(rewrite_reason)
        self.assertEqual(new_neighbors, {"node_a": 4, "node_b": 3})
        self.assertEqual(rewritten[0], [2, 0, 4])
        self.assertEqual(rewritten[1], [2, 0, 1, 3])
        self.assertEqual(rewritten[2], [3, 1, 5])
        self.assertEqual(rewritten[3], [4, 0, 1, 5])

    def test_legacy_cleanup_two_sided_after_triple_switch_merges_local_two_sided_flynn(self) -> None:
        nodes = np.array(
            [
                [0.50, 0.70],
                [0.50, 0.30],
                [0.46, 0.50],
                [0.52, 0.50],
                [0.40, 0.85],
                [0.40, 0.15],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [4, 0, 2, 1, 5]},
            {"flynn_id": 1, "label": 1, "node_ids": [4, 5, 1, 3, 0]},
            {"flynn_id": 2, "label": 2, "node_ids": [2, 0, 3, 1]},
        ]

        updated_nodes, updated_flynns, events, merged = mesh_module._legacy_cleanup_two_sided_after_triple_switch(
            nodes.copy(),
            copy.deepcopy(flynns),
            0,
            1,
            candidate_flynn_ids={0, 1, 2},
        )

        self.assertEqual(merged, 1)
        self.assertEqual(len(updated_flynns), 2)
        self.assertEqual(len(events), 1)
        self.assertEqual(events[0]["type"], "merge_small_two_sided_flynn")
        self.assertEqual(events[0]["removed_flynn"], 2)
        self.assertEqual(events[0]["reason"], "legacy_post_switch_two_sided")
        self.assertEqual(events[0]["switched_nodes"], [0, 1])
        self.assertEqual(events[0]["triple_nodes"], [0, 1])
        np.testing.assert_allclose(updated_nodes, nodes, atol=1.0e-12)

    def test_legacy_validate_triple_switch_result_returns_role_specific_small_flynn_reason(self) -> None:
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.52, 0.50],
                [0.46, 0.58],
                [0.46, 0.42],
                [0.58, 0.58],
                [0.58, 0.42],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [2, 0, 1, 4]},
            {"flynn_id": 1, "label": 1, "node_ids": [2, 0, 3]},
            {"flynn_id": 2, "label": 2, "node_ids": [3, 0, 1, 5]},
            {"flynn_id": 3, "label": 3, "node_ids": [4, 1, 5]},
        ]

        context, reason = mesh_module._legacy_triple_switch_context(
            copy.deepcopy(flynns),
            0,
            1,
        )

        self.assertIsNone(reason)
        self.assertIsNotNone(context)
        valid, invalid_reason, affected_indices = mesh_module._legacy_validate_triple_switch_result(
            copy.deepcopy(flynns),
            nodes.copy(),
            legacy_context=context,
            cleanup_performed=False,
            min_area=1.0,
            geometry_cache={},
        )

        self.assertFalse(valid)
        self.assertEqual(invalid_reason, "full_id_0_small_flynn")
        self.assertEqual(affected_indices, {0, 1, 2, 3})

    def test_faithful_crossings_check_local_blocks_two_sided_flynn_overtake(self) -> None:
        old_nodes = np.array(
            [
                [0.40, 0.40],
                [0.70, 0.40],
                [0.70, 0.70],
                [0.40, 0.70],
            ],
            dtype=np.float64,
        )
        moved_nodes = old_nodes.copy()
        moved_nodes[1] = np.array([0.45, 0.75], dtype=np.float64)
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
        ]

        updated_nodes, _updated_flynns, events, coincident, blocked = mesh_module._faithful_crossings_check_local(
            moved_nodes,
            copy.deepcopy(flynns),
            moved_node_id=1,
            focus_nodes={0, 1, 2},
            switch_distance=0.05,
            previous_position=old_nodes[1],
        )

        self.assertEqual(coincident, 0)
        self.assertEqual(blocked, 1)
        np.testing.assert_allclose(updated_nodes[1], old_nodes[1], atol=1.0e-12)
        self.assertEqual(events[0]["type"], "faithful_topology_stage")
        self.assertEqual(events[0]["stage"], "crossings_check")
        self.assertEqual(events[1]["type"], "forbid_crossing")
        self.assertEqual(events[1]["reason"], "two_sided_flynn_edge_crossing")

    def test_faithful_crossings_check_local_blocks_impossible_triple_switch(self) -> None:
        old_nodes = np.array(
            [
                [0.25, 0.50],
                [0.55, 0.50],
                [0.40, 0.60],
                [0.40, 0.40],
                [0.60, 0.60],
                [0.60, 0.40],
                [0.50, 0.30],
            ],
            dtype=np.float64,
        )
        moved_nodes = old_nodes.copy()
        moved_nodes[0] = np.array([0.45, 0.50], dtype=np.float64)
        flynns = [
            {"flynn_id": 10, "label": 10, "node_ids": [2, 0, 1, 4]},
            {"flynn_id": 11, "label": 11, "node_ids": [3, 0, 1, 5]},
            {"flynn_id": 12, "label": 12, "node_ids": [2, 0, 3, 6, 5, 1, 4]},
        ]

        updated_nodes, _updated_flynns, events, coincident, blocked = mesh_module._faithful_crossings_check_local(
            moved_nodes,
            copy.deepcopy(flynns),
            moved_node_id=0,
            focus_nodes={0, 1, 2, 3, 4, 5, 6},
            switch_distance=0.2,
            previous_position=old_nodes[0],
        )

        self.assertEqual(coincident, 0)
        self.assertEqual(blocked, 1)
        np.testing.assert_allclose(updated_nodes[0], old_nodes[0], atol=1.0e-12)
        self.assertEqual(events[0]["type"], "faithful_topology_stage")
        self.assertEqual(events[0]["stage"], "crossings_check")
        self.assertEqual(events[1]["type"], "forbid_crossing")
        self.assertEqual(events[1]["reason"], "triple_switch_impossible")
        self.assertEqual(events[1]["neighbor_id"], 1)

    def test_maintain_mesh_locally_emits_faithful_topology_stage_order(self) -> None:
        nodes = np.array(
            [
                [0.00, 0.00],
                [0.01, 0.00],
                [0.50, 0.00],
                [0.50, 0.50],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
        ]

        (
            _updated_nodes,
            _updated_flynns,
            events,
            _switched_edges,
            _rejected_switches,
            _merged_flynns,
            _split_flynns,
            _inserted_nodes,
            _removed_nodes,
            _coincident_nodes_resolved,
            _deleted_small_flynns,
            _widened_angles,
        ) = mesh_module._maintain_mesh_locally(
            nodes,
            copy.deepcopy(flynns),
            moved_node_id=1,
            grid_shape=(64, 64),
            switch_distance=0.1,
            min_angle_degrees=20.0,
            min_node_separation=0.1,
            max_node_separation=100.0,
        )

        stage_events = [
            event["stage"]
            for event in events
            if event.get("type") == "faithful_topology_stage"
        ]
        collapsed_stage_events: list[str] = []
        for stage in stage_events:
            if not collapsed_stage_events or collapsed_stage_events[-1] != stage:
                collapsed_stage_events.append(str(stage))

        self.assertEqual(
            collapsed_stage_events,
            [
                "crossings_check",
                "delete_single_j",
                "check_double_j",
                "update_topology_state",
            ],
        )

    def test_maintain_mesh_locally_skips_focus_refresh_when_local_check_is_noop(self) -> None:
        nodes = np.array(
            [
                [0.00, 0.00],
                [0.50, 0.00],
                [0.50, 0.50],
                [0.00, 0.50],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
        ]

        original_refresh = mesh_module._refresh_local_focus_nodes
        refresh_calls: list[set[int]] = []

        def capture_refresh(moved_node_id, focus_nodes, flynns, *, node_neighbors=None):
            refresh_calls.append({int(node_id) for node_id in focus_nodes})
            return original_refresh(
                moved_node_id,
                focus_nodes,
                flynns,
                node_neighbors=node_neighbors,
            )

        with patch.object(mesh_module, "_refresh_local_focus_nodes", side_effect=capture_refresh):
            mesh_module._maintain_mesh_locally(
                nodes,
                copy.deepcopy(flynns),
                moved_node_id=1,
                grid_shape=(64, 64),
                switch_distance=0.1,
                min_angle_degrees=20.0,
                min_node_separation=0.1,
                max_node_separation=100.0,
            )

        self.assertEqual(len(refresh_calls), 1)
        self.assertEqual(refresh_calls[0], {0, 1, 2})

    def test_relax_mesh_state_widens_acute_angle_in_topocheck_sequence(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.50, "y": 0.50},
                {"x": 0.60, "y": 0.50},
                {"x": 0.59, "y": 0.52},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2]},
            ],
            "stats": {"grid_shape": [1000, 1000]},
            "events": [],
        }

        def node_angle(nodes: list[dict[str, float]]) -> float:
            point = np.array([nodes[0]["x"], nodes[0]["y"]], dtype=np.float64)
            first = np.array([nodes[1]["x"], nodes[1]["y"]], dtype=np.float64) - point
            second = np.array([nodes[2]["x"], nodes[2]["y"]], dtype=np.float64) - point
            cosine = float(np.dot(first, second) / (np.linalg.norm(first) * np.linalg.norm(second)))
            return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))))

        angle_before = node_angle(mesh_state["nodes"])
        relaxed = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.05,
                min_node_separation_factor=0.0,
                max_node_separation_factor=100.0,
            ),
        )
        angle_after = node_angle(relaxed["nodes"])

        self.assertLess(angle_before, 20.0)
        self.assertGreater(angle_after, angle_before)
        self.assertGreater(relaxed["stats"]["mesh_widened_angles"], 0)
        self.assertTrue(any(event["type"] == "widen_angle" for event in relaxed["events"]))

    def test_relax_mesh_state_splits_flynn_when_neck_is_too_narrow(self) -> None:
        mesh_state = {
            "nodes": [
                {"x": 0.45, "y": 0.60},
                {"x": 0.25, "y": 0.82},
                {"x": 0.15, "y": 0.72},
                {"x": 0.15, "y": 0.28},
                {"x": 0.25, "y": 0.18},
                {"x": 0.45, "y": 0.40},
                {"x": 0.75, "y": 0.18},
                {"x": 0.85, "y": 0.28},
                {"x": 0.85, "y": 0.72},
                {"x": 0.75, "y": 0.82},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 7, "node_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]},
            ],
            "stats": {"grid_shape": [1000, 1000]},
            "events": [],
        }

        relaxed = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.25,
                min_angle_degrees=0.0,
                min_node_separation_factor=0.0,
                max_node_separation_factor=100.0,
            ),
        )

        self.assertEqual(relaxed["stats"]["mesh_split_flynns"], 1)
        self.assertEqual(relaxed["stats"]["num_flynns"], 2)
        self.assertEqual(len(relaxed["flynns"]), 2)
        self.assertEqual(len(relaxed["nodes"]), 10)
        split_events = [event for event in relaxed["events"] if event["type"] == "split_flynn"]
        self.assertEqual(len(split_events), 1)
        self.assertEqual(split_events[0]["source_flynn"], 0)
        self.assertEqual(split_events[0]["label"], 7)
        self.assertEqual(sorted(flynn["label"] for flynn in relaxed["flynns"]), [7, 7])

    def test_find_split_pair_prefers_closest_non_neighbour_double_nodes(self) -> None:
        node_ids = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
        nodes = np.array(
            [
                [0.45, 0.60],
                [0.25, 0.82],
                [0.15, 0.72],
                [0.15, 0.28],
                [0.25, 0.18],
                [0.45, 0.40],
                [0.75, 0.18],
                [0.85, 0.28],
                [0.85, 0.72],
                [0.75, 0.82],
            ],
            dtype=np.float64,
        )
        node_neighbors = {
            0: {1, 9},
            1: {0, 2},
            2: {1, 3},
            3: {2, 4},
            4: {3, 5},
            5: {4, 6},
            6: {5, 7},
            7: {6, 8},
            8: {7, 9},
            9: {8, 0},
        }

        pair = _find_split_pair(
            node_ids,
            nodes,
            node_neighbors,
            switch_distance=0.25,
        )

        self.assertIsNotNone(pair)
        assert pair is not None
        self.assertEqual({pair[0], pair[1]}, {0, 5})
        self.assertLess(pair[2], 0.25)

    def test_run_simulation_supports_mesh_feedback(self) -> None:
        cfg = GrainGrowthConfig(
            nx=12,
            ny=8,
            num_grains=3,
            seed=5,
            init_mode="voronoi",
        )
        final_state, snapshots = run_simulation(
            cfg,
            steps=4,
            save_every=2,
            mesh_feedback=MeshFeedbackConfig(
                every=2,
                strength=0.25,
                boundary_width=1,
                relax_config=MeshRelaxationConfig(
                    steps=0,
                    topology_steps=1,
                    min_node_separation_factor=0.0,
                ),
            ),
        )

        final_np = np.asarray(final_state)
        self.assertEqual(final_np.shape, (3, 12, 8))
        self.assertEqual(len(snapshots), 2)
        self.assertLess(float(np.abs(final_np.sum(axis=0) - 1.0).max()), 1e-5)

    def test_run_simulation_supports_elle_seeded_runtime_mesh_feedback(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            label_seed = load_elle_label_seed(elle_path)
            mesh_seed, relax_overrides = load_elle_mesh_seed(elle_path, label_seed)
            cfg = GrainGrowthConfig(
                nx=label_seed["grid_shape"][0],
                ny=label_seed["grid_shape"][1],
                num_grains=label_seed["num_labels"],
                seed=3,
                init_mode="elle",
                init_elle_path=str(elle_path),
                init_elle_attribute=str(label_seed["attribute"]),
                init_smoothing_steps=0,
                init_noise=0.0,
            )
            contexts = []

            def record_snapshot(step, phi, mesh_feedback_context) -> None:
                contexts.append(mesh_feedback_context)

            final_state, snapshots = run_simulation(
                cfg,
                steps=2,
                save_every=2,
                on_snapshot=record_snapshot,
                mesh_feedback=MeshFeedbackConfig(
                    every=1,
                    strength=0.2,
                    transport_strength=0.0,
                    update_mode="blend",
                    boundary_width=1,
                    initial_mesh_state=mesh_seed,
                    relax_config=MeshRelaxationConfig(
                        steps=1,
                        topology_steps=1,
                        speed_up=relax_overrides["speed_up"],
                        switch_distance=relax_overrides["switch_distance"],
                        movement_model="elle_surface",
                        min_node_separation_factor=relax_overrides["min_node_separation_factor"],
                        max_node_separation_factor=relax_overrides["max_node_separation_factor"],
                    ),
                ),
            )

        final_np = np.asarray(final_state)
        self.assertEqual(final_np.shape, (2, 2, 2))
        self.assertEqual(len(snapshots), 1)
        self.assertEqual(len(contexts), 1)
        self.assertIsNotNone(contexts[0])
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_seed_source"], "elle")
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["elle_option_switchdistance"], 0.05)
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_movement_model"], "elle_surface")
        self.assertLess(float(np.abs(final_np.sum(axis=0) - 1.0).max()), 1e-5)

    def test_run_simulation_supports_mesh_only_runtime_translation(self) -> None:
        cfg = GrainGrowthConfig(
            nx=4,
            ny=4,
            num_grains=2,
            seed=1,
            init_mode="voronoi",
        )
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        mesh_seed = build_mesh_state(labels)
        contexts = []

        def record_snapshot(step, phi, mesh_feedback_context) -> None:
            contexts.append(mesh_feedback_context)

        final_state, snapshots = run_simulation(
            cfg,
            steps=1,
            save_every=1,
            on_snapshot=record_snapshot,
            mesh_feedback=MeshFeedbackConfig(
                every=1,
                strength=0.0,
                transport_strength=0.0,
                update_mode="mesh_only",
                initial_mesh_state=mesh_seed,
                relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
            ),
        )

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(len(contexts), 1)
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_update_mode"], "mesh_only")
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_solver_backend"], "numpy_mesh_only")
        self.assertEqual(contexts[0]["feedback_stats"]["unassigned_unodes"], 0)
        self.assertEqual(contexts[0]["solver_backend"], "numpy_mesh_only")
        self.assertEqual(np.asarray(final_state).shape, (2, 4, 4))

    def test_run_mesh_only_simulation_exposes_numpy_backend(self) -> None:
        cfg = GrainGrowthConfig(
            nx=4,
            ny=4,
            num_grains=2,
            seed=1,
            init_mode="voronoi",
        )
        labels = np.array(
            [
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
                [0, 0, 1, 1],
            ],
            dtype=np.int32,
        )
        mesh_seed = build_mesh_state(labels)
        contexts = []

        def record_snapshot(step, phi, mesh_feedback_context) -> None:
            contexts.append(mesh_feedback_context)

        final_state, snapshots = run_mesh_only_simulation(
            cfg,
            steps=1,
            save_every=1,
            on_snapshot=record_snapshot,
            mesh_feedback=MeshFeedbackConfig(
                every=1,
                strength=0.0,
                transport_strength=0.0,
                update_mode="mesh_only",
                initial_mesh_state=mesh_seed,
                relax_config=MeshRelaxationConfig(steps=0, topology_steps=0),
            ),
        )

        self.assertEqual(len(snapshots), 1)
        self.assertEqual(contexts[0]["solver_backend"], "numpy_mesh_only")
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_solver_backend"], "numpy_mesh_only")
        self.assertEqual(np.asarray(final_state).shape, (2, 4, 4))

    def test_run_simulation_supports_mesh_kernel_coupling(self) -> None:
        cfg = GrainGrowthConfig(
            nx=12,
            ny=8,
            num_grains=3,
            seed=9,
            init_mode="voronoi",
        )
        baseline_state, _ = run_simulation(cfg, steps=4, save_every=4)
        contexts = []

        def record_snapshot(step, phi, mesh_feedback_context) -> None:
            contexts.append(mesh_feedback_context)

        coupled_state, _ = run_simulation(
            cfg,
            steps=4,
            save_every=4,
            on_snapshot=record_snapshot,
            mesh_feedback=MeshFeedbackConfig(
                every=0,
                strength=0.0,
                transport_strength=0.0,
                kernel_advection_every=1,
                kernel_advection_strength=4.0,
                kernel_predictor_corrector=True,
                boundary_width=1,
                relax_config=MeshRelaxationConfig(
                    steps=1,
                    topology_steps=0,
                    min_node_separation_factor=0.0,
                ),
            ),
        )

        baseline_np = np.asarray(baseline_state)
        coupled_np = np.asarray(coupled_state)
        self.assertEqual(coupled_np.shape, (3, 12, 8))
        self.assertLess(float(np.abs(coupled_np.sum(axis=0) - 1.0).max()), 1e-5)
        self.assertGreater(float(np.abs(coupled_np - baseline_np).max()), 1e-4)
        self.assertEqual(len(contexts), 1)
        self.assertIsNotNone(contexts[0])
        self.assertIn("mesh_state", contexts[0])
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_kernel_advection_every"], 1)
        self.assertEqual(contexts[0]["mesh_state"]["stats"]["mesh_kernel_predictor_corrector"], 1)
        self.assertGreater(contexts[0]["mesh_state"]["stats"]["mesh_kernel_transport_pixels"], 0)

    def test_run_simulation_with_topology_reuses_runtime_tracked_ids(self) -> None:
        cfg = GrainGrowthConfig(
            nx=12,
            ny=8,
            num_grains=3,
            seed=11,
            init_mode="voronoi",
        )
        seen = []

        def record_snapshot(step, phi, topology_snapshot, mesh_feedback_context) -> None:
            seen.append((topology_snapshot, mesh_feedback_context))

        _, snapshots, history = run_simulation_with_topology(
            cfg,
            steps=4,
            save_every=2,
            on_snapshot=record_snapshot,
            mesh_feedback=MeshFeedbackConfig(
                every=0,
                strength=0.0,
                transport_strength=0.0,
                kernel_advection_every=1,
                kernel_advection_strength=2.0,
                kernel_predictor_corrector=True,
                boundary_width=1,
                relax_config=MeshRelaxationConfig(
                    steps=1,
                    topology_steps=0,
                    min_node_separation_factor=0.0,
                ),
            ),
        )

        self.assertEqual(len(snapshots), 2)
        self.assertEqual(len(history), 2)
        self.assertEqual(len(seen), 2)
        topology_snapshot, mesh_feedback_context = seen[-1]
        self.assertIsNotNone(mesh_feedback_context)
        self.assertEqual(
            {flynn["flynn_id"] for flynn in topology_snapshot["flynns"]},
            {flynn["flynn_id"] for flynn in mesh_feedback_context["mesh_state"]["flynns"]},
        )
        self.assertEqual(mesh_feedback_context["mesh_state"]["stats"]["mesh_runtime_topology_tracked"], 1)

    def test_mesh_topology_maintenance_splits_long_edges(self) -> None:
        labels = np.array(
            [
                [0, 0, 1],
                [0, 0, 1],
            ],
            dtype=np.int32,
        )
        tracked = TopologyTracker().update(labels, step=1)
        mesh_state = build_mesh_state(labels, tracked_topology=tracked)
        maintained = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.1,
                min_node_separation_factor=0.0,
            ),
        )

        self.assertGreater(maintained["stats"]["mesh_inserted_nodes"], 0)
        self.assertGreater(maintained["stats"]["mesh_event_count"], 0)
        self.assertEqual(maintained["stats"]["mesh_topology_steps"], 1)

        with tempfile.TemporaryDirectory() as tmpdir:
            phi = np.zeros((2, labels.shape[0], labels.shape[1]), dtype=np.float32)
            for grain_id in range(2):
                phi[grain_id] = (labels == grain_id).astype(np.float32)
            outpath = write_unode_elle(
                Path(tmpdir) / "topology_only.elle",
                phi,
                step=1,
                tracked_topology=tracked,
                mesh_state=maintained,
            )
            text = outpath.read_text(encoding="utf-8")

        self.assertIn("# mesh_topology_maintained=1", text)
        self.assertIn("# mesh_events switched=", text)

    def test_mesh_topology_maintenance_switches_short_triple_edge(self) -> None:
        mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.50, "y": 0.50, "degree": 3, "junction_type": "triple", "neighbors": [1, 2, 3], "flynns": [0, 1, 2]},
                {"node_id": 1, "x": 0.52, "y": 0.50, "degree": 3, "junction_type": "triple", "neighbors": [0, 4, 5], "flynns": [0, 2, 3]},
                {"node_id": 2, "x": 0.46, "y": 0.58, "degree": 2, "junction_type": "double", "neighbors": [0, 4], "flynns": [0, 1]},
                {"node_id": 3, "x": 0.46, "y": 0.42, "degree": 2, "junction_type": "double", "neighbors": [0, 5], "flynns": [1, 2]},
                {"node_id": 4, "x": 0.58, "y": 0.58, "degree": 2, "junction_type": "double", "neighbors": [1, 2], "flynns": [0, 3]},
                {"node_id": 5, "x": 0.58, "y": 0.42, "degree": 2, "junction_type": "double", "neighbors": [1, 3], "flynns": [2, 3]},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [2, 0, 1, 4]},
                {"flynn_id": 1, "label": 1, "node_ids": [2, 0, 3]},
                {"flynn_id": 2, "label": 2, "node_ids": [3, 0, 1, 5]},
                {"flynn_id": 3, "label": 3, "node_ids": [4, 1, 5]},
            ],
            "stats": {
                "grid_shape": [1000, 1000],
                "num_nodes": 6,
                "num_flynns": 4,
                "double_junctions": 4,
                "triple_junctions": 2,
                "topology_components": 4,
                "holes_skipped": 0,
            },
            "events": [],
        }

        maintained = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.05,
                min_node_separation_factor=0.0,
                max_node_separation_factor=100.0,
            ),
        )

        self.assertEqual(maintained["stats"]["mesh_switched_triples"], 1)
        self.assertEqual(maintained["events"][0]["type"], "switch_triple_a0b1")
        self.assertEqual(
            maintained["events"][0]["legacy_switch_neighbors"],
            {"node3": 2, "node4": 3, "node5": 4, "node6": 5},
        )
        self.assertEqual(
            maintained["events"][0]["legacy_switch_flynns"],
            {"full_id_0": 1, "full_id_1": 0, "full_id_2": 3, "full_id_3": 2},
        )
        self.assertIn("legacy_switch_targets", maintained["events"][0])
        self.assertEqual(maintained["events"][0]["post_switch_merged_flynns"], 0)
        self.assertEqual(maintained["events"][0]["post_switch_cleanup_events"], [])
        self.assertEqual(maintained["events"][0]["post_switch_inserted_nodes"], 0)
        self.assertEqual(maintained["events"][0]["post_switch_removed_nodes"], 0)
        self.assertEqual(maintained["events"][0]["post_switch_switched_edges"], 0)
        self.assertEqual(maintained["events"][0]["post_switch_rejected_switches"], 0)
        self.assertEqual(
            [event["stage"] for event in maintained["events"][0]["post_switch_topology_events"]],
            [
                "post_switch_topology_check",
                "delete_single_j",
                "check_triple_j",
                "post_switch_update_topology_state",
            ],
        )
        self.assertAlmostEqual(maintained["events"][0]["legacy_switch_targets"]["node_a"][0], 0.5166666666666667)
        self.assertAlmostEqual(maintained["events"][0]["legacy_switch_targets"]["node_a"][1], 0.5533333333333333)
        self.assertAlmostEqual(maintained["events"][0]["legacy_switch_targets"]["node_b"][0], 0.5166666666666667)
        self.assertAlmostEqual(maintained["events"][0]["legacy_switch_targets"]["node_b"][1], 0.44666666666666666)
        node_zero = next(node for node in maintained["nodes"] if node["node_id"] == 0)
        node_one = next(node for node in maintained["nodes"] if node["node_id"] == 1)
        self.assertEqual(node_zero["neighbors"], [1, 2, 4])
        self.assertEqual(node_one["neighbors"], [0, 3, 5])
        self.assertAlmostEqual(node_zero["x"], 0.5166666666666667)
        self.assertAlmostEqual(node_zero["y"], 0.5533333333333333)
        self.assertAlmostEqual(node_one["x"], 0.5166666666666667)
        self.assertAlmostEqual(node_one["y"], 0.44666666666666666)

    def test_switch_triple_edge_keeps_legacy_flynn_ids_stable_after_cleanup_index_shift(self) -> None:
        nodes = np.array(
            [
                [0.50, 0.50],
                [0.52, 0.50],
                [0.46, 0.58],
                [0.46, 0.42],
                [0.58, 0.58],
                [0.58, 0.42],
            ],
            dtype=np.float64,
        )
        flynns = [
            {"flynn_id": 0, "label": 0, "node_ids": [2, 0, 1, 4]},
            {"flynn_id": 1, "label": 1, "node_ids": [2, 0, 3]},
            {"flynn_id": 2, "label": 2, "node_ids": [3, 0, 1, 5]},
            {"flynn_id": 3, "label": 3, "node_ids": [4, 1, 5]},
        ]

        def fake_cleanup(
            cleanup_nodes: np.ndarray,
            cleanup_flynns: list[dict[str, object]],
            _node_a: int,
            _node_b: int,
            *,
            candidate_flynn_ids: set[int],
        ) -> tuple[np.ndarray, list[dict[str, object]], list[dict[str, object]], int]:
            self.assertEqual(candidate_flynn_ids, {0, 1, 2, 3})
            del cleanup_flynns[2]
            return cleanup_nodes, cleanup_flynns, [{"type": "merge_small_two_sided_flynn", "removed_flynn": 2}], 1

        with patch.object(mesh_module, "_legacy_cleanup_two_sided_after_triple_switch", side_effect=fake_cleanup):
            switched, event, reason = mesh_module._switch_triple_edge(
                nodes.copy(),
                copy.deepcopy(flynns),
                0,
                1,
                0.05,
                node_neighbors=mesh_module._node_neighbors(copy.deepcopy(flynns)),
                edge_map=mesh_module._edge_map(copy.deepcopy(flynns)),
                geometry_cache={},
            )

        self.assertTrue(switched)
        self.assertIsNone(reason)
        self.assertIsNotNone(event)
        self.assertEqual(event["shared_flynns"], [0, 2])
        self.assertEqual(event["exclusive_flynns"], [1, 3])
        self.assertEqual(
            event["legacy_switch_flynns"],
            {"full_id_0": 1, "full_id_1": 0, "full_id_2": 3, "full_id_3": 2},
        )
        self.assertEqual(event["post_switch_merged_flynns"], 1)
        self.assertEqual(
            [entry["stage"] for entry in event["post_switch_topology_events"]],
            [
                "post_switch_topology_check",
                "delete_single_j",
                "check_triple_j",
                "post_switch_update_topology_state",
            ],
        )

    def test_mesh_topology_rejects_switch_that_creates_tiny_flynn(self) -> None:
        mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.50, "y": 0.50, "degree": 3, "junction_type": "triple", "neighbors": [1, 2, 3], "flynns": [0, 1, 2]},
                {"node_id": 1, "x": 0.52, "y": 0.50, "degree": 3, "junction_type": "triple", "neighbors": [0, 4, 5], "flynns": [0, 2, 3]},
                {"node_id": 2, "x": 0.46, "y": 0.58, "degree": 2, "junction_type": "double", "neighbors": [0, 4], "flynns": [0, 1]},
                {"node_id": 3, "x": 0.46, "y": 0.42, "degree": 2, "junction_type": "double", "neighbors": [0, 5], "flynns": [1, 2]},
                {"node_id": 4, "x": 0.58, "y": 0.58, "degree": 2, "junction_type": "double", "neighbors": [1, 2], "flynns": [0, 3]},
                {"node_id": 5, "x": 0.58, "y": 0.42, "degree": 2, "junction_type": "double", "neighbors": [1, 3], "flynns": [2, 3]},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [2, 0, 1, 4]},
                {"flynn_id": 1, "label": 1, "node_ids": [2, 0, 3]},
                {"flynn_id": 2, "label": 2, "node_ids": [3, 0, 1, 5]},
                {"flynn_id": 3, "label": 3, "node_ids": [4, 1, 5]},
            ],
            "stats": {
                "grid_shape": [1000, 1000],
                "num_nodes": 6,
                "num_flynns": 4,
                "double_junctions": 4,
                "triple_junctions": 2,
                "topology_components": 4,
                "holes_skipped": 0,
            },
            "events": [],
        }

        maintained = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.10,
                min_node_separation_factor=0.0,
                max_node_separation_factor=100.0,
            ),
        )

        self.assertEqual(maintained["stats"]["mesh_switched_triples"], 0)
        self.assertGreaterEqual(maintained["stats"]["mesh_rejected_switches"], 1)
        self.assertTrue(maintained["events"])
        self.assertTrue(all(event["type"] == "reject_triple_switch" for event in maintained["events"]))
        node_zero = next(node for node in maintained["nodes"] if node["node_id"] == 0)
        node_one = next(node for node in maintained["nodes"] if node["node_id"] == 1)
        self.assertEqual(node_zero["neighbors"], [1, 2, 3])
        self.assertEqual(node_one["neighbors"], [0, 4, 5])

    def test_mesh_topology_attempts_shared_nearest_triple_edge_once(self) -> None:
        mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.50, "y": 0.50, "degree": 3, "junction_type": "triple", "neighbors": [1, 2, 3], "flynns": [0, 1, 2]},
                {"node_id": 1, "x": 0.52, "y": 0.50, "degree": 3, "junction_type": "triple", "neighbors": [0, 4, 5], "flynns": [0, 2, 3]},
                {"node_id": 2, "x": 0.46, "y": 0.58, "degree": 2, "junction_type": "double", "neighbors": [0, 4], "flynns": [0, 1]},
                {"node_id": 3, "x": 0.46, "y": 0.42, "degree": 2, "junction_type": "double", "neighbors": [0, 5], "flynns": [1, 2]},
                {"node_id": 4, "x": 0.58, "y": 0.58, "degree": 2, "junction_type": "double", "neighbors": [1, 2], "flynns": [0, 3]},
                {"node_id": 5, "x": 0.58, "y": 0.42, "degree": 2, "junction_type": "double", "neighbors": [1, 3], "flynns": [2, 3]},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [2, 0, 1, 4]},
                {"flynn_id": 1, "label": 1, "node_ids": [2, 0, 3]},
                {"flynn_id": 2, "label": 2, "node_ids": [3, 0, 1, 5]},
                {"flynn_id": 3, "label": 3, "node_ids": [4, 1, 5]},
            ],
            "stats": {
                "grid_shape": [1000, 1000],
                "num_nodes": 6,
                "num_flynns": 4,
                "double_junctions": 4,
                "triple_junctions": 2,
                "topology_components": 4,
                "holes_skipped": 0,
            },
            "events": [],
        }

        attempted_edges: list[tuple[int, int]] = []

        def fake_candidate(
            _nodes: np.ndarray,
            _flynns: list[dict[str, object]],
            node_id: int,
            _switch_distance: float,
            *,
            node_neighbors: dict[int, set[int]] | None = None,
            edge_map: dict[tuple[int, int], list[tuple[int, int]]] | None = None,
        ) -> dict[str, object] | None:
            if int(node_id) not in (0, 1):
                return None
            return {
                "node_id": int(node_id),
                "target_neighbor": 1 if int(node_id) == 0 else 0,
                "edge": (0, 1),
                "edge_length": 0.02,
            }

        def fake_switch(
            _nodes: np.ndarray,
            _flynns: list[dict[str, object]],
            node_a: int,
            node_b: int,
            _switch_distance: float,
            *,
            node_neighbors: dict[int, set[int]] | None = None,
            edge_map: dict[tuple[int, int], list[tuple[int, int]]] | None = None,
            geometry_cache: dict[int, dict[str, object]] | None = None,
            min_node_separation: float | None = None,
            max_node_separation: float | None = None,
            phase_lookup: dict[int, int] | None = None,
            run_post_switch_topology_check: bool = True,
        ) -> tuple[bool, dict[str, object] | None, str | None]:
            attempted_edges.append((int(node_a), int(node_b)))
            return False, None, "small_flynn"

        with patch.object(mesh_module, "_nearest_short_triple_candidate", side_effect=fake_candidate):
            with patch.object(mesh_module, "_switch_triple_edge", side_effect=fake_switch):
                maintained = relax_mesh_state(
                    mesh_state,
                    MeshRelaxationConfig(
                        steps=0,
                        topology_steps=1,
                        switch_distance=0.10,
                        min_node_separation_factor=0.0,
                        max_node_separation_factor=100.0,
                    ),
                )

        self.assertEqual(attempted_edges, [(0, 1)])
        self.assertEqual(maintained["stats"]["mesh_switched_triples"], 0)
        self.assertEqual(maintained["stats"]["mesh_rejected_switches"], 1)
        self.assertEqual(len(maintained["events"]), 1)
        self.assertEqual(maintained["events"][0]["type"], "reject_triple_switch")
        self.assertEqual(maintained["events"][0]["reason"], "small_flynn")

    def test_mesh_topology_merges_tiny_two_sided_flynn(self) -> None:
        mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.50, "y": 0.70, "degree": 3, "junction_type": "triple", "neighbors": [2, 3, 4], "flynns": [0, 1, 2]},
                {"node_id": 1, "x": 0.50, "y": 0.30, "degree": 3, "junction_type": "triple", "neighbors": [2, 3, 5], "flynns": [0, 1, 2]},
                {"node_id": 2, "x": 0.46, "y": 0.50, "degree": 2, "junction_type": "double", "neighbors": [0, 1], "flynns": [0, 2]},
                {"node_id": 3, "x": 0.52, "y": 0.50, "degree": 2, "junction_type": "double", "neighbors": [0, 1], "flynns": [1, 2]},
                {"node_id": 4, "x": 0.40, "y": 0.85, "degree": 2, "junction_type": "double", "neighbors": [0, 5], "flynns": [0, 1]},
                {"node_id": 5, "x": 0.40, "y": 0.15, "degree": 2, "junction_type": "double", "neighbors": [1, 4], "flynns": [0, 1]},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [4, 0, 2, 1, 5]},
                {"flynn_id": 1, "label": 1, "node_ids": [4, 5, 1, 3, 0]},
                {"flynn_id": 2, "label": 2, "node_ids": [2, 0, 3, 1]},
            ],
            "stats": {
                "grid_shape": [10, 10],
                "num_nodes": 6,
                "num_flynns": 3,
                "double_junctions": 4,
                "triple_junctions": 2,
                "topology_components": 3,
                "holes_skipped": 0,
            },
            "events": [],
        }

        maintained = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.25,
                min_angle_degrees=0.0,
                min_node_separation_factor=0.68,
                max_node_separation_factor=100.0,
            ),
        )

        self.assertEqual(maintained["stats"]["mesh_merged_flynns"], 1)
        self.assertEqual(maintained["stats"]["num_flynns"], 2)
        self.assertEqual(len(maintained["flynns"]), 2)
        self.assertEqual(len(maintained["nodes"]), 5)
        merge_events = [event for event in maintained["events"] if event["type"] == "merge_small_two_sided_flynn"]
        self.assertEqual(len(merge_events), 1)
        self.assertEqual(merge_events[0]["removed_flynn"], 2)
        self.assertEqual(merge_events[0]["kept_flynn"], 0)

    def test_mesh_topology_maintenance_collapses_short_double_nodes(self) -> None:
        mesh_state = {
            "nodes": [
                {"node_id": 0, "x": 0.00, "y": 0.00, "degree": 2, "junction_type": "double", "neighbors": [1, 3], "flynns": [0]},
                {"node_id": 1, "x": 0.01, "y": 0.00, "degree": 2, "junction_type": "double", "neighbors": [0, 2], "flynns": [0]},
                {"node_id": 2, "x": 0.50, "y": 0.00, "degree": 2, "junction_type": "double", "neighbors": [1, 3], "flynns": [0]},
                {"node_id": 3, "x": 0.50, "y": 0.50, "degree": 2, "junction_type": "double", "neighbors": [0, 2], "flynns": [0]},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2, 3]},
            ],
            "stats": {
                "grid_shape": [10, 10],
                "num_nodes": 4,
                "num_flynns": 1,
                "double_junctions": 4,
                "triple_junctions": 0,
                "topology_components": 1,
                "holes_skipped": 0,
            },
            "events": [],
        }

        maintained = relax_mesh_state(
            mesh_state,
            MeshRelaxationConfig(
                steps=0,
                topology_steps=1,
                switch_distance=0.1,
                max_node_separation_factor=100.0,
            ),
        )

        self.assertGreater(maintained["stats"]["mesh_removed_nodes"], 0)
        self.assertLess(len(maintained["nodes"]), len(mesh_state["nodes"]))

    def test_extract_legacy_reference_snapshot_reads_fft_example(self) -> None:
        example = (
            PROJECT_ROOT.parent
            / "processes"
            / "fft"
            / "example"
            / "step0"
            / "inifft001.elle"
        )

        snapshot = extract_legacy_reference_snapshot(example)

        self.assertEqual(snapshot["checkpoint_name"], "inifft001")
        self.assertEqual(snapshot["mesh"]["num_flynns"], 16)
        self.assertEqual(snapshot["mesh"]["num_nodes"], 1136)
        self.assertEqual(snapshot["label_summary"]["attribute"], "U_ATTRIB_A")
        self.assertEqual(snapshot["label_summary"]["grain_count"], 16)
        self.assertEqual(snapshot["label_summary"]["grid_shape"], [256, 256])
        self.assertIn("U_EULER_3", snapshot["field_summaries"])
        self.assertIn("U_DISLOCDEN", snapshot["field_summaries"])
        self.assertEqual(snapshot["skipped_sections"], [])

    def test_parse_elle_sections_handles_composite_headers_in_fine_foam(self) -> None:
        example = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )

        sections = _parse_elle_sections(example)

        self.assertIn("U_DISLOCDEN", sections)
        self.assertIn("U_FINITE_STRAIN", sections)
        self.assertNotIn(
            "U_FINITE_STRAIN START_S_X START_S_Y PREV_S_X PREV_S_Y CURR_S_X CURR_S_Y",
            sections["U_DISLOCDEN"],
        )

    def test_extract_legacy_reference_snapshot_reads_fine_foam_dislocden(self) -> None:
        example = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )

        snapshot = extract_legacy_reference_snapshot(example)

        self.assertIn("U_EULER_3", snapshot["field_summaries"])
        self.assertIn("U_DISLOCDEN", snapshot["field_summaries"])

    def test_legacy_reference_fixture_matches_fft_example_snapshot(self) -> None:
        fixture = (
            PROJECT_ROOT
            / "legacy_reference"
            / "testdata"
            / "fft_example_step0_reference.json"
        )
        example = (
            PROJECT_ROOT.parent
            / "processes"
            / "fft"
            / "example"
            / "step0"
            / "inifft001.elle"
        )

        bundle = load_legacy_reference_bundle(fixture)
        rebuilt = build_legacy_reference_bundle({"step0": example}, source_name="fft_example_step0")

        self.assertEqual(bundle["source_name"], "fft_example_step0")
        self.assertEqual(bundle["checkpoint_order"], ["step0"])
        self.assertEqual(bundle["checkpoints"]["step0"]["mesh"], rebuilt["checkpoints"]["step0"]["mesh"])
        self.assertEqual(
            bundle["checkpoints"]["step0"]["label_summary"],
            rebuilt["checkpoints"]["step0"]["label_summary"],
        )
        self.assertEqual(
            bundle["checkpoints"]["step0"]["field_summaries"],
            rebuilt["checkpoints"]["step0"]["field_summaries"],
        )

    def test_compare_legacy_reference_snapshot_matches_fft_example_fixture(self) -> None:
        fixture = (
            PROJECT_ROOT
            / "legacy_reference"
            / "testdata"
            / "fft_example_step0_reference.json"
        )
        example = (
            PROJECT_ROOT.parent
            / "processes"
            / "fft"
            / "example"
            / "step0"
            / "inifft001.elle"
        )
        bundle = load_legacy_reference_bundle(fixture)

        report = compare_legacy_reference_snapshot(example, bundle["checkpoints"]["step0"])

        self.assertTrue(report["matches"])
        self.assertEqual(report["mismatched_field_summaries"], [])
        self.assertEqual(report["missing_field_summaries"], [])
        self.assertEqual(report["unexpected_field_summaries"], [])
        self.assertTrue(all(entry["matches"] for entry in report["mesh"].values()))
        self.assertTrue(all(entry["matches"] for entry in report["label_summary"].values()))

    def test_compare_legacy_reference_snapshot_detects_changed_field(self) -> None:
        fixture = (
            PROJECT_ROOT
            / "legacy_reference"
            / "testdata"
            / "fft_example_step0_reference.json"
        )
        example = (
            PROJECT_ROOT.parent
            / "processes"
            / "fft"
            / "example"
            / "step0"
            / "inifft001.elle"
        )
        bundle = load_legacy_reference_bundle(fixture)

        with tempfile.TemporaryDirectory() as tmpdir:
            candidate = Path(tmpdir) / "changed.elle"
            text = example.read_text(encoding="utf-8")
            section_start = text.index("\nU_ATTRIB_A\n")
            prefix = text[:section_start]
            suffix = text[section_start:]
            suffix = suffix.replace(
                "\n1 9.00000000e+00\n",
                "\n1 9.90000000e+01\n",
                1,
            )
            candidate.write_text(prefix + suffix, encoding="utf-8")

            report = compare_legacy_reference_snapshot(candidate, bundle["checkpoints"]["step0"])

        self.assertFalse(report["matches"])
        self.assertIn("U_ATTRIB_A", report["mismatched_field_summaries"])
        self.assertFalse(report["field_summaries"]["U_ATTRIB_A"]["value_hash"]["matches"])

    def test_compare_legacy_reference_bundle_matches_fft_example_fixture(self) -> None:
        fixture = (
            PROJECT_ROOT
            / "legacy_reference"
            / "testdata"
            / "fft_example_step0_reference.json"
        )
        example = (
            PROJECT_ROOT.parent
            / "processes"
            / "fft"
            / "example"
            / "step0"
            / "inifft001.elle"
        )
        bundle = load_legacy_reference_bundle(fixture)

        report = compare_legacy_reference_bundle(bundle, {"step0": example})

        self.assertTrue(report["matches"])
        self.assertEqual(report["missing_checkpoints"], [])
        self.assertEqual(report["unexpected_checkpoints"], [])
        self.assertTrue(report["checkpoints"]["step0"]["matches"])

    def test_extract_legacy_reference_transition_tracks_label_and_field_deltas(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            before = _write_elle_mesh_seed_example(Path(tmpdir) / "before.elle")
            after = Path(tmpdir) / "after.elle"
            text = before.read_text(encoding="utf-8")
            text = text.replace("\n2 2.0\n", "\n2 9.0\n", 1)
            text = text.replace("\n2 100\n", "\n2 200\n", 1)
            after.write_text(text, encoding="utf-8")

            transition = extract_legacy_reference_transition(before, after, checkpoint_name="seed_step")

        self.assertEqual(transition["checkpoint_name"], "seed_step")
        self.assertIn("U_ATTRIB_A", transition["field_transitions"])
        self.assertIn("U_ATTRIB_C", transition["field_transitions"])
        self.assertTrue(transition["label_transition"]["available"])
        self.assertEqual(transition["label_transition"]["changed_pixels"], 1)
        self.assertEqual(transition["field_transitions"]["U_ATTRIB_A"]["changed_rows"], 1)
        self.assertEqual(transition["field_transitions"]["U_ATTRIB_C"]["changed_rows"], 1)

    def test_compare_legacy_reference_transition_matches_identical_transition(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            before = _write_elle_mesh_seed_example(Path(tmpdir) / "before.elle")
            after = Path(tmpdir) / "after.elle"
            text = before.read_text(encoding="utf-8")
            text = text.replace("\n3 3.0\n", "\n3 7.0\n", 1)
            after.write_text(text, encoding="utf-8")

            reference = extract_legacy_reference_transition(before, after, checkpoint_name="field_only")
            report = compare_legacy_reference_transition(before, after, reference)

        self.assertTrue(report["matches"])
        self.assertEqual(report["mismatched_field_transitions"], [])
        self.assertEqual(report["missing_field_transitions"], [])
        self.assertEqual(report["unexpected_field_transitions"], [])

    def test_compare_legacy_reference_transition_detects_changed_candidate(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            before = _write_elle_mesh_seed_example(Path(tmpdir) / "before.elle")
            reference_after = Path(tmpdir) / "reference_after.elle"
            candidate_after = Path(tmpdir) / "candidate_after.elle"

            base_text = before.read_text(encoding="utf-8")
            reference_after.write_text(
                base_text.replace("\n4 4.0\n", "\n4 8.0\n", 1),
                encoding="utf-8",
            )
            candidate_after.write_text(
                base_text.replace("\n4 4.0\n", "\n4 12.0\n", 1),
                encoding="utf-8",
            )

            reference = extract_legacy_reference_transition(
                before,
                reference_after,
                checkpoint_name="mismatch_case",
            )
            report = compare_legacy_reference_transition(before, candidate_after, reference)

        self.assertFalse(report["matches"])
        self.assertIn("U_ATTRIB_A", report["mismatched_field_transitions"])
        self.assertFalse(report["field_transitions"]["U_ATTRIB_A"]["delta_hash"]["matches"])

    def test_fine_foam_outerstep_transition_fixture_matches_reference_pair(self) -> None:
        fixture = (
            PROJECT_ROOT
            / "legacy_reference"
            / "testdata"
            / "fine_foam_outerstep_001_to_002_transition.json"
        )
        before = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )
        after = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step002.elle"
        )

        reference_transition = json.loads(fixture.read_text(encoding="utf-8"))
        report = compare_legacy_reference_transition(before, after, reference_transition)

        self.assertTrue(report["matches"])
        self.assertEqual(report["mismatched_field_transitions"], [])
        self.assertEqual(report["missing_field_transitions"], [])
        self.assertEqual(report["unexpected_field_transitions"], [])

    def test_extract_legacy_reference_swept_unode_transition_tracks_real_outerstep(self) -> None:
        before = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )
        after = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step002.elle"
        )

        transition = extract_legacy_reference_swept_unode_transition(
            before,
            after,
            checkpoint_name="fine_foam_swept",
            field_names=("U_EULER_3", "U_DISLOCDEN"),
        )

        self.assertEqual(transition["checkpoint_name"], "fine_foam_swept")
        self.assertTrue(transition["swept_unodes"]["available"])
        self.assertGreater(transition["swept_unodes"]["swept_rows"], 0)
        self.assertIn("U_EULER_3", transition["field_transitions"])
        self.assertIn("U_DISLOCDEN", transition["field_transitions"])
        self.assertTrue(transition["field_transitions"]["U_EULER_3"]["shape_matches"])
        self.assertTrue(transition["field_transitions"]["U_DISLOCDEN"]["shape_matches"])

    def test_compare_legacy_reference_swept_unode_transition_matches_reference_pair(self) -> None:
        before = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )
        after = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step002.elle"
        )

        reference = extract_legacy_reference_swept_unode_transition(
            before,
            after,
            checkpoint_name="fine_foam_swept",
            field_names=("U_EULER_3", "U_DISLOCDEN"),
        )
        report = compare_legacy_reference_swept_unode_transition(before, after, reference)

        self.assertTrue(report["matches"])
        self.assertEqual(report["mismatched_field_transitions"], [])
        self.assertEqual(report["missing_field_transitions"], [])
        self.assertEqual(report["unexpected_field_transitions"], [])

    def test_compare_legacy_reference_swept_unode_transition_detects_changed_candidate(self) -> None:
        before = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step001.elle"
        )
        after = (
            PROJECT_ROOT.parent.parent.parent
            / "TwoWayIceModel_Release"
            / "elle"
            / "example"
            / "results"
            / "fine_foam_step002.elle"
        )

        reference = extract_legacy_reference_swept_unode_transition(
            before,
            after,
            checkpoint_name="fine_foam_swept",
            field_names=("U_EULER_3", "U_DISLOCDEN"),
        )
        report = compare_legacy_reference_swept_unode_transition(before, before, reference)

        self.assertFalse(report["matches"])
        self.assertIn("U_EULER_3", report["mismatched_field_transitions"])
        self.assertIn("U_DISLOCDEN", report["mismatched_field_transitions"])

    def test_snapshot_statistics_tracks_active_grains(self) -> None:
        cfg = GrainGrowthConfig(nx=14, ny=10, num_grains=4, seed=11, init_mode="voronoi")
        phi = np.asarray(initialize_order_parameters(cfg))
        stats = snapshot_statistics(phi, step=3)

        self.assertEqual(stats["step"], 3)
        self.assertEqual(stats["active_grains"], 4)
        self.assertEqual(stats["grid_shape"], [14, 10])


if __name__ == "__main__":
    unittest.main()
