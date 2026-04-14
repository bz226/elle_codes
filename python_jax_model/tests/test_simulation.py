from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from portable_elle_viewer import build_viewer_payload

from run_simulation import resolve_mesh_preset, resolve_runtime_preset
from elle_jax_model.artifacts import dominant_grain_map, save_snapshot_artifacts, snapshot_statistics
from elle_jax_model.benchmark_validation import (
    evaluate_release_dataset_benchmarks,
    evaluate_rasterized_grain_growth_benchmark,
    evaluate_static_grain_growth_benchmark,
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
from elle_jax_model.gbm_faithful import (
    FAITHFUL_GBM_DEFAULTS,
    build_faithful_gbm_setup,
    run_faithful_gbm_simulation,
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
from elle_jax_model.elle_html_viewer import write_elle_html_viewer
from elle_jax_model.elle_visualize import render_elle_file
from elle_jax_model.phasefield_compare import (
    compare_elle_phasefield_sequences,
    compare_elle_phasefield_files,
    compare_elle_phasefield_states,
    inspect_elle_phasefield_binary,
    run_original_elle_phasefield_sequence,
)
from elle_jax_model.mesh import (
    _apply_segment_mass_partition,
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
)
from elle_jax_model.microstructure_validation import (
    compare_elle_microstructure_sequences,
    summarize_elle_microstructure,
    summarize_liu_suckale_datasets,
)
from elle_jax_model.paper_validation import (
    assess_current_rewrite_against_papers,
    summarize_liu_suckale_paper_from_text,
    summarize_llorens_structure_from_text,
)
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


def _write_elle_mesh_seed_example(path: Path) -> Path:
    lines = [
        "OPTIONS",
        "SwitchDistance 0.05",
        "MaxNodeSeparation 0.11",
        "MinNodeSeparation 0.05",
        "SpeedUp 2.0",
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

    def test_compare_elle_microstructure_sequences_matches_steps(self) -> None:
        with tempfile.TemporaryDirectory() as reference_dir, tempfile.TemporaryDirectory() as candidate_dir:
            _write_periodic_flynn_example(Path(reference_dir) / "periodic_0001.elle")
            _write_periodic_flynn_example(Path(candidate_dir) / "periodic_0001.elle")
            report = compare_elle_microstructure_sequences(reference_dir, candidate_dir)

        self.assertEqual(report["summary"]["num_matched_steps"], 1)
        self.assertEqual(report["summary"]["grain_count_abs_diff_mean"], 0.0)
        self.assertEqual(report["summary"]["mean_grain_area_abs_diff_mean"], 0.0)

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
            {"step": 1, "grain_count": 10, "mean_grain_area": 0.10, "mean_equivalent_radius": 0.20, "mean_shape_factor": 0.80},
            {"step": 2, "grain_count": 9, "mean_grain_area": 0.11, "mean_equivalent_radius": 0.21, "mean_shape_factor": 0.79},
            {"step": 3, "grain_count": 8, "mean_grain_area": 0.13, "mean_equivalent_radius": 0.23, "mean_shape_factor": 0.78},
        ]

        summary = summarize_sequence_trends(sequence)

        self.assertTrue(summary["flags"]["coarsening_present"])
        self.assertGreater(summary["metrics"]["mean_grain_area"]["delta"], 0.0)
        self.assertLess(summary["metrics"]["grain_count"]["delta"], 0.0)

    def test_evaluate_static_grain_growth_benchmark_reports_reference_coarsening(self) -> None:
        reference_dir = PROJECT_ROOT.parent.parent.parent / "TwoWayIceModel_Release" / "elle" / "example" / "results"

        report = evaluate_static_grain_growth_benchmark(reference_dir, pattern="fine_foam_step*.elle")

        self.assertEqual(report["reference_trends"]["num_snapshots"], 10)
        self.assertTrue(report["reference_trends"]["flags"]["coarsening_present"])
        self.assertGreater(report["reference_trends"]["metrics"]["mean_grain_area"]["delta"], 0.0)

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

        self.assertEqual(setup.seed_info["attribute"], "U_ATTRIB_C")
        self.assertEqual(setup.config.init_mode, "elle")
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
        self.assertEqual(setup.mesh_feedback.boundary_width, FAITHFUL_GBM_DEFAULTS["raster_boundary_band"])
        self.assertTrue(bool(setup.mesh_feedback.relax_config.use_diagonal_trials))
        self.assertTrue(bool(setup.mesh_feedback.relax_config.use_elle_physical_units))
        self.assertEqual(setup.mesh_feedback.strength, 0.0)
        self.assertEqual(setup.mesh_feedback.transport_strength, 0.0)
        self.assertEqual(setup.config.dt, 0.05)
        self.assertEqual(setup.config.mobility, 1.0)
        self.assertEqual(setup.config.gradient_penalty, 1.0)
        self.assertEqual(setup.config.interaction_strength, 2.0)
        self.assertEqual(setup.config.init_smoothing_steps, 0)
        self.assertEqual(setup.config.init_noise, 0.0)

    def test_build_faithful_gbm_setup_derives_raster_boundary_band_from_elle_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            text = elle_path.read_text(encoding="utf-8")
            text = text.replace(
                "SpeedUp 2.0",
                "SpeedUp 2.0\nBoundaryWidth 0.4\nUnitLength 0.2",
            )
            elle_path.write_text(text, encoding="utf-8")

            setup = build_faithful_gbm_setup(elle_path)

        self.assertEqual(setup.mesh_feedback.boundary_width, 4)

    def test_build_faithful_gbm_setup_supports_stage_named_overrides(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")

            setup = build_faithful_gbm_setup(
                elle_path,
                movement_model="legacy",
                motion_passes=2,
                topology_passes=3,
                stage_interval=4,
                raster_boundary_band=5,
                use_diagonal_trials=False,
                use_elle_physical_units=False,
            )

        self.assertEqual(setup.mesh_feedback.relax_config.movement_model, "legacy")
        self.assertEqual(setup.mesh_feedback.relax_config.steps, 2)
        self.assertEqual(setup.mesh_feedback.relax_config.topology_steps, 3)
        self.assertEqual(setup.mesh_feedback.every, 4)
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

    def test_run_faithful_gbm_simulation_mesh_only_ignores_legacy_phasefield_coefficients(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "seed.elle")
            default_setup = build_faithful_gbm_setup(elle_path)
            legacy_override_setup = build_faithful_gbm_setup(
                elle_path,
                dt=0.2,
                mobility=3.5,
                gradient_penalty=4.0,
                interaction_strength=0.25,
                init_smoothing_steps=5,
                init_noise=0.2,
            )

            default_state, _, _ = run_faithful_gbm_simulation(
                steps=1,
                save_every=1,
                setup=default_setup,
            )
            override_state, _, _ = run_faithful_gbm_simulation(
                steps=1,
                save_every=1,
                setup=legacy_override_setup,
            )

        np.testing.assert_allclose(np.asarray(default_state), np.asarray(override_state))

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

    def test_load_elle_mesh_seed_maps_original_flynns_and_options(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            elle_path = _write_elle_mesh_seed_example(Path(tmpdir) / "mesh_seed.elle")
            label_seed = load_elle_label_seed(elle_path)
            mesh_state, relax_overrides = load_elle_mesh_seed(elle_path, label_seed)

        self.assertEqual(label_seed["attribute"], "U_ATTRIB_C")
        self.assertEqual(mesh_state["stats"]["num_flynns"], 2)
        self.assertEqual(mesh_state["stats"]["mesh_seed_source"], "elle")
        self.assertAlmostEqual(relax_overrides["switch_distance"], 0.05)
        self.assertAlmostEqual(relax_overrides["min_node_separation_factor"], 1.0)
        self.assertAlmostEqual(relax_overrides["max_node_separation_factor"], 2.2)
        self.assertAlmostEqual(relax_overrides["speed_up"], 2.0)
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
        self.assertIn("_runtime_seed_node_fields", mesh_state)
        self.assertIn("N_ATTRIB_A", mesh_state["_runtime_seed_node_fields"]["values"])
        self.assertEqual(len(mesh_state["_runtime_seed_node_fields"]["values"]["N_ATTRIB_A"]), 6)

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
            label_seed = load_elle_label_seed(elle_path)
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
                {"x": 0.20, "y": 0.45},
                {"x": 0.78, "y": 0.57},
            ],
            "flynns": [
                {"flynn_id": 0, "label": 0, "node_ids": [0, 1, 2]},
            ],
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
            node_xy,
            ordered_neighbors,
            nodes,
            switch_distance=switch_distance,
            boundary_energy=1.0,
            use_diagonal_trials=False,
        )
        force_diagonal = _surface_force_from_trial_energies(
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
            velocity_length=0.04,
            switch_distance=0.05,
            speed_up=2.0,
        )

        self.assertAlmostEqual(dt, 1.0)

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
            node_xy,
            ordered_neighbors,
            nodes,
            switch_distance=0.05,
            boundary_energy=1.0,
            use_diagonal_trials=False,
        )
        increment = _move_node_elle_surface(
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
        node_zero = next(node for node in maintained["nodes"] if node["node_id"] == 0)
        node_one = next(node for node in maintained["nodes"] if node["node_id"] == 1)
        self.assertEqual(node_zero["neighbors"], [1, 2, 4])
        self.assertEqual(node_one["neighbors"], [0, 3, 5])

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
        self.assertEqual(maintained["stats"]["mesh_rejected_switches"], 1)
        self.assertFalse(maintained["events"])
        node_zero = next(node for node in maintained["nodes"] if node["node_id"] == 0)
        node_one = next(node for node in maintained["nodes"] if node["node_id"] == 1)
        self.assertEqual(node_zero["neighbors"], [1, 2, 3])
        self.assertEqual(node_one["neighbors"], [0, 4, 5])

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

    def test_snapshot_statistics_tracks_active_grains(self) -> None:
        cfg = GrainGrowthConfig(nx=14, ny=10, num_grains=4, seed=11, init_mode="voronoi")
        phi = np.asarray(initialize_order_parameters(cfg))
        stats = snapshot_statistics(phi, step=3)

        self.assertEqual(stats["step"], 3)
        self.assertEqual(stats["active_grains"], 4)
        self.assertEqual(stats["grid_shape"], [14, 10])


if __name__ == "__main__":
    unittest.main()
