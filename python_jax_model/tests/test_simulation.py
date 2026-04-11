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

from elle_jax_model.artifacts import dominant_grain_map, save_snapshot_artifacts, snapshot_statistics
from elle_jax_model.elle_export import extract_flynn_topology, write_unode_elle
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
    MeshFeedbackConfig,
    MeshRelaxationConfig,
    apply_mesh_feedback,
    apply_mesh_transport,
    build_mesh_state,
    rasterize_mesh_labels,
    relax_mesh_state,
)
from elle_jax_model.simulation import GrainGrowthConfig, initialize_order_parameters, run_simulation
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
                "grid_shape": [10, 10],
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
                "grid_shape": [10, 10],
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
