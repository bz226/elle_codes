from __future__ import annotations

"""CLI for the faithful ELLE GBM translation path.

This is the supported public entrypoint for solver-parity work.
Prototype phase-field runners remain archive-only for fidelity decisions.
"""

import argparse
from pathlib import Path

from elle_jax_model.artifacts import save_snapshot_artifacts
from elle_jax_model.gbm_faithful import (
    FAITHFUL_GBM_DEFAULTS,
    build_faithful_gbm_setup,
    run_faithful_gbm_simulation,
)
from elle_jax_model.topology import write_topology_history


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the dedicated faithful NumPy-first ELLE GBM translation path"
    )
    parser.add_argument("--init-elle", type=Path, required=True, help="Original ELLE file to seed from")
    parser.add_argument("--init-elle-attribute", default="auto")
    parser.add_argument("--steps", type=int, default=10)
    parser.add_argument("--save-every", type=int, default=1)
    parser.add_argument(
        "--include-step0",
        action="store_true",
        help="Also emit a step-0 faithful seed snapshot before any GBM stage",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--movement-model",
        choices=("legacy", "elle_surface"),
        default=FAITHFUL_GBM_DEFAULTS["movement_model"],
    )
    parser.add_argument(
        "--motion-passes",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["motion_passes"],
        help="Number of ELLE-style boundary-motion passes per saved stage",
    )
    parser.add_argument(
        "--topology-passes",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["topology_passes"],
        help="Number of topology-cleanup passes per saved stage",
    )
    parser.add_argument(
        "--stage-interval",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["stage_interval"],
        help="How often to apply one faithful GBM stage in the outer loop",
    )
    parser.add_argument(
        "--subloops-per-snapshot",
        "--total-subloops",
        dest="subloops_per_snapshot",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["subloops_per_snapshot"],
        help="Original-style subloop count to execute before saving one outer-step snapshot",
    )
    parser.add_argument(
        "--gbm-steps-per-subloop",
        "--gbm-steps",
        dest="gbm_steps_per_subloop",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["gbm_steps_per_subloop"],
        help="Original-style GBM stage count to execute inside each subloop before saving a snapshot",
    )
    parser.add_argument(
        "--nucleation-steps-per-subloop",
        "--nuc-steps",
        dest="nucleation_steps_per_subloop",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["nucleation_steps_per_subloop"],
        help="Original-style nucleation stage count to execute inside each subloop before GBM",
    )
    parser.add_argument(
        "--nucleation-hagb-deg",
        type=float,
        default=FAITHFUL_GBM_DEFAULTS["nucleation_hagb_deg"],
        help="High-angle grain-boundary cutoff used by the faithful nucleation stage",
    )
    parser.add_argument(
        "--nucleation-min-cluster-unodes",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["nucleation_min_cluster_unodes"],
        help="Minimum secondary subgrain size that can nucleate into a new grain",
    )
    parser.add_argument(
        "--nucleation-parent-area-crit",
        type=float,
        default=FAITHFUL_GBM_DEFAULTS["nucleation_parent_area_crit"],
        help="Original-style minimum parent flynn area required before a new grain can nucleate",
    )
    parser.add_argument(
        "--recovery-steps-per-subloop",
        "--recovery-steps",
        dest="recovery_steps_per_subloop",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["recovery_steps_per_subloop"],
        help="Original-style recovery stage count to execute inside each subloop after GBM",
    )
    parser.add_argument(
        "--recovery-hagb-deg",
        type=float,
        default=FAITHFUL_GBM_DEFAULTS["recovery_hagb_deg"],
        help="High-angle grain-boundary cutoff used by the faithful recovery stage",
    )
    parser.add_argument(
        "--recovery-trial-rotation-deg",
        type=float,
        default=FAITHFUL_GBM_DEFAULTS["recovery_trial_rotation_deg"],
        help="Trial rotation increment in degrees for the faithful recovery stage",
    )
    parser.add_argument(
        "--recovery-rotation-mobility-length",
        type=float,
        default=FAITHFUL_GBM_DEFAULTS["recovery_rotation_mobility_length"],
        help="Original-style recovery rotation-mobility-length factor",
    )
    parser.add_argument(
        "--raster-boundary-band",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["raster_boundary_band"],
        help="Integer raster-band support used only in the rewritten grid ownership path",
    )
    parser.add_argument(
        "--temperature-c",
        type=float,
        default=None,
        help="ELLE temperature in Celsius used by the faithful Arrhenius boundary-mobility law; defaults to the seed ELLE Temperature option",
    )
    parser.add_argument(
        "--phase-db",
        type=Path,
        help="Optional original ELLE phase_db.txt override for faithful phase-pair mobility data",
    )
    parser.add_argument(
        "--mechanics-snapshot-dir",
        type=Path,
        help="Optional directory containing one legacy FFT bridge snapshot or a sequence of snapshot subdirectories with temp-FFT.out, unodexyz.out, unodeang.out, and optional tex.out",
    )
    parser.add_argument(
        "--no-mechanics-import-dd",
        action="store_true",
        help="Mirror FS_fft2elle ImportDDs=0 by skipping tex.out dislocation-density increments during the mechanics stage",
    )
    parser.add_argument(
        "--mechanics-exclude-phase-id",
        type=int,
        default=FAITHFUL_GBM_DEFAULTS["mechanics_exclude_phase_id"],
        help="Mirror FS_fft2elle ExcludePhaseID by zeroing imported DD in the selected phase during the mechanics stage",
    )
    parser.add_argument(
        "--mechanics-density-update-mode",
        choices=("increment", "overwrite"),
        default=FAITHFUL_GBM_DEFAULTS["mechanics_density_update_mode"],
        help="Choose whether tex.out DD values increment the current U_DISLOCDEN field or overwrite it, mirroring different legacy fft2elle branches",
    )
    parser.add_argument(
        "--mechanics-host-repair-mode",
        choices=("fs_check_unodes", "check_error"),
        default=FAITHFUL_GBM_DEFAULTS["mechanics_host_repair_mode"],
        help="Choose whether post-import host repair follows the later FS_CheckUnodes path or the older check_error path",
    )
    parser.add_argument(
        "--mechanics-only",
        action="store_true",
        help="Replay only legacy mechanics snapshots before each saved outer step, disabling nucleation, GBM, and recovery subloops",
    )
    parser.add_argument(
        "--no-diagonal-trials",
        action="store_true",
        help="Disable ELLE-style diagonal trial positions in the faithful mover",
    )
    parser.add_argument(
        "--no-elle-physical-units",
        action="store_true",
        help="Disable UnitLength-aware physical scaling in the faithful mover",
    )
    parser.add_argument("--mesh-relax-steps", dest="motion_passes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mesh-topology-steps", dest="topology_passes", type=int, help=argparse.SUPPRESS)
    parser.add_argument("--mesh-movement-model", dest="movement_model", help=argparse.SUPPRESS)
    parser.add_argument("--mesh-feedback-every", dest="stage_interval", type=int, help=argparse.SUPPRESS)
    parser.add_argument(
        "--mesh-feedback-boundary-width",
        dest="raster_boundary_band",
        type=int,
        help=argparse.SUPPRESS,
    )
    parser.add_argument("--save-elle", action="store_true")
    parser.add_argument("--track-topology", action="store_true")
    parser.add_argument("--no-preview", action="store_true")
    parser.add_argument(
        "--outdir",
        type=Path,
        default=Path("python_jax_model/validation/gbm_faithful_output"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.mechanics_only and args.mechanics_snapshot_dir is None:
        raise SystemExit("--mechanics-only requires --mechanics-snapshot-dir")
    if args.mechanics_only:
        args.nucleation_steps_per_subloop = 0
        args.gbm_steps_per_subloop = 0
        args.recovery_steps_per_subloop = 0
    args.outdir.mkdir(parents=True, exist_ok=True)

    setup = build_faithful_gbm_setup(
        args.init_elle,
        init_elle_attribute=args.init_elle_attribute,
        seed=int(args.seed),
        movement_model=str(args.movement_model),
        motion_passes=int(args.motion_passes),
        topology_passes=int(args.topology_passes),
        stage_interval=int(args.stage_interval),
        subloops_per_snapshot=int(args.subloops_per_snapshot),
        gbm_steps_per_subloop=int(args.gbm_steps_per_subloop),
        nucleation_steps_per_subloop=int(args.nucleation_steps_per_subloop),
        nucleation_hagb_deg=float(args.nucleation_hagb_deg),
        nucleation_min_cluster_unodes=int(args.nucleation_min_cluster_unodes),
        nucleation_parent_area_crit=float(args.nucleation_parent_area_crit),
        recovery_steps_per_subloop=int(args.recovery_steps_per_subloop),
        recovery_hagb_deg=float(args.recovery_hagb_deg),
        recovery_trial_rotation_deg=float(args.recovery_trial_rotation_deg),
        recovery_rotation_mobility_length=float(args.recovery_rotation_mobility_length),
        raster_boundary_band=int(args.raster_boundary_band),
        temperature_c=None if args.temperature_c is None else float(args.temperature_c),
        phase_db_path=None if args.phase_db is None else str(args.phase_db),
        mechanics_snapshot_dir=None if args.mechanics_snapshot_dir is None else str(args.mechanics_snapshot_dir),
        mechanics_import_dislocation_densities=not bool(args.no_mechanics_import_dd),
        mechanics_exclude_phase_id=int(args.mechanics_exclude_phase_id),
        mechanics_density_update_mode=str(args.mechanics_density_update_mode),
        mechanics_host_repair_mode=str(args.mechanics_host_repair_mode),
        use_diagonal_trials=not bool(args.no_diagonal_trials),
        use_elle_physical_units=not bool(args.no_elle_physical_units),
    )

    print(
        "running faithful GBM translation: "
        f"attribute={setup.seed_info.attribute} "
        f"grid={setup.seed_info.grid_shape[0]}x{setup.seed_info.grid_shape[1]} "
        f"labels={setup.seed_info.num_labels} "
        f"movement_model={setup.mesh_feedback.relax_config.movement_model} "
        f"motion_passes={setup.mesh_feedback.relax_config.steps} "
        f"topology_passes={setup.mesh_feedback.relax_config.topology_steps} "
        f"stage_interval={setup.mesh_feedback.every} "
        f"subloops_per_snapshot={setup.subloops_per_snapshot} "
        f"gbm_steps_per_subloop={setup.gbm_steps_per_subloop} "
        f"nucleation_steps_per_subloop={setup.nucleation_steps_per_subloop} "
        f"nucleation_parent_area_crit={setup.nucleation_config.parent_area_crit} "
        f"recovery_steps_per_subloop={setup.recovery_steps_per_subloop} "
        f"stages_per_snapshot={setup.subloops_per_snapshot * (setup.nucleation_steps_per_subloop + setup.gbm_steps_per_subloop + setup.recovery_steps_per_subloop) + int(bool(setup.mechanics_snapshots))} "
        f"raster_boundary_band={setup.mesh_feedback.boundary_width} "
        f"temperature_c={setup.mesh_feedback.relax_config.temperature_c} "
        f"mechanics_snapshots={len(setup.mechanics_snapshots)} "
        f"mechanics_import_dd={int(setup.mechanics_import_options.import_dislocation_densities)} "
        f"mechanics_exclude_phase_id={setup.mechanics_import_options.exclude_phase_id} "
        f"mechanics_density_update_mode={setup.mechanics_import_options.density_update_mode} "
        f"mechanics_host_repair_mode={setup.mechanics_import_options.host_repair_mode} "
        f"mechanics_only={int(bool(args.mechanics_only))} "
        "update_mode=mesh_only"
    )

    save_topology = bool(args.track_topology or args.save_elle)

    def save_snapshot(step: int, phi, topology_snapshot, mesh_feedback_context=None) -> None:
        mesh_state = None if mesh_feedback_context is None else mesh_feedback_context.get("mesh_state")
        written = save_snapshot_artifacts(
            args.outdir,
            step,
            phi,
            save_preview=not args.no_preview,
            save_elle=args.save_elle,
            tracked_topology=topology_snapshot,
            save_topology=save_topology,
            mesh_state=mesh_state,
            save_mesh=True,
        )
        names = ", ".join(path.name for path in written.values())
        print(f"saved step {step:05d}: {names}")

    _, snapshots, topology_history = run_faithful_gbm_simulation(
        steps=int(args.steps),
        save_every=int(args.save_every),
        on_snapshot=save_snapshot,
        setup=setup,
        include_initial_snapshot=bool(args.include_step0),
    )

    if save_topology:
        history_path = args.outdir / "topology_history.json"
        write_topology_history(history_path, topology_history)
        print(f"saved topology history: {history_path.name}")

    print(f"done: wrote {len(snapshots)} snapshots with solver_backend=numpy_mesh_only")


if __name__ == "__main__":
    main()
