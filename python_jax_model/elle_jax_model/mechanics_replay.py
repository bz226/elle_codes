from __future__ import annotations

from pathlib import Path
from typing import Any

from .artifacts import save_snapshot_artifacts
from .gbm_faithful import build_faithful_gbm_setup, run_faithful_gbm_simulation
from .legacy_reference import (
    compare_legacy_reference_transition,
    extract_legacy_reference_transition,
)
from .topology import write_topology_history


def run_faithful_mechanics_replay(
    init_elle_path: str | Path,
    mechanics_snapshot_dir: str | Path,
    *,
    outdir: str | Path,
    init_elle_attribute: str = "auto",
    mechanics_density_update_mode: str = "increment",
    mechanics_host_repair_mode: str = "fs_check_unodes",
    steps: int | None = None,
    save_preview: bool = False,
    save_topology: bool = True,
    save_mesh: bool = True,
    save_elle: bool = True,
) -> dict[str, Any]:
    """Run a mechanics-only faithful replay from one snapshot or a snapshot sequence."""

    setup = build_faithful_gbm_setup(
        init_elle_path,
        init_elle_attribute=init_elle_attribute,
        mechanics_snapshot_dir=mechanics_snapshot_dir,
        motion_passes=0,
        topology_passes=0,
        subloops_per_snapshot=1,
        gbm_steps_per_subloop=0,
        nucleation_steps_per_subloop=0,
        recovery_steps_per_subloop=0,
        mechanics_density_update_mode=mechanics_density_update_mode,
        mechanics_host_repair_mode=mechanics_host_repair_mode,
    )
    if not setup.mechanics_snapshots:
        raise ValueError("mechanics replay requires at least one mechanics snapshot")

    replay_steps = int(steps) if steps is not None else int(len(setup.mechanics_snapshots))
    if replay_steps < 1:
        raise ValueError("mechanics replay requires steps >= 1")

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: dict[str, dict[str, str]] = {}

    def _save_snapshot(step: int, phi, topology_snapshot, mesh_feedback_context=None) -> None:
        mesh_state = None if mesh_feedback_context is None else mesh_feedback_context.get("mesh_state")
        written = save_snapshot_artifacts(
            outdir_path,
            int(step),
            phi,
            save_preview=bool(save_preview),
            save_elle=bool(save_elle),
            tracked_topology=topology_snapshot,
            save_topology=bool(save_topology),
            mesh_state=mesh_state,
            save_mesh=bool(save_mesh),
        )
        checkpoint_paths[str(int(step))] = {
            str(name): str(path) for name, path in written.items()
        }

    _, snapshots, topology_history = run_faithful_gbm_simulation(
        setup=setup,
        steps=int(replay_steps),
        save_every=1,
        on_snapshot=_save_snapshot,
        include_initial_snapshot=True,
    )

    topology_history_path: str | None = None
    if save_topology:
        history_path = write_topology_history(outdir_path / "topology_history.json", topology_history)
        topology_history_path = str(history_path)

    return {
        "output_dir": str(outdir_path),
        "steps": int(replay_steps),
        "written_snapshots": int(len(snapshots)),
        "mechanics_snapshot_count": int(len(setup.mechanics_snapshots)),
        "checkpoint_paths": checkpoint_paths,
        "topology_history_path": topology_history_path,
    }


def validate_faithful_mechanics_transition(
    init_elle_path: str | Path,
    mechanics_snapshot_dir: str | Path,
    reference_before_elle_path: str | Path,
    reference_after_elle_path: str | Path,
    *,
    outdir: str | Path,
    init_elle_attribute: str = "auto",
    mechanics_density_update_mode: str = "increment",
    mechanics_host_repair_mode: str = "fs_check_unodes",
    label_attribute: str = "auto",
    checkpoint_name: str | None = None,
    field_names: list[str] | tuple[str, ...] | None = None,
) -> dict[str, Any]:
    """Run a one-step mechanics-only replay and compare it against a legacy transition."""

    replay = run_faithful_mechanics_replay(
        init_elle_path,
        mechanics_snapshot_dir,
        outdir=outdir,
        init_elle_attribute=init_elle_attribute,
        mechanics_density_update_mode=mechanics_density_update_mode,
        mechanics_host_repair_mode=mechanics_host_repair_mode,
        steps=1,
        save_preview=False,
        save_topology=True,
        save_mesh=True,
        save_elle=True,
    )
    checkpoint_paths = replay["checkpoint_paths"]
    try:
        candidate_before = checkpoint_paths["0"]["elle"]
        candidate_after = checkpoint_paths["1"]["elle"]
    except KeyError as exc:
        raise ValueError(
            "mechanics transition validation requires step-0 and step-1 ELLE artifacts"
        ) from exc

    reference_transition = extract_legacy_reference_transition(
        reference_before_elle_path,
        reference_after_elle_path,
        checkpoint_name=checkpoint_name,
        label_attribute=label_attribute,
        field_names=field_names,
    )
    report = compare_legacy_reference_transition(
        candidate_before,
        candidate_after,
        reference_transition,
        label_attribute=label_attribute,
    )
    return {
        **report,
        "replay": replay,
        "reference_before_path": str(Path(reference_before_elle_path)),
        "reference_after_path": str(Path(reference_after_elle_path)),
    }


def validate_faithful_mechanics_outerstep_transition(
    init_elle_path: str | Path,
    mechanics_snapshot_dir: str | Path,
    reference_before_elle_path: str | Path,
    reference_after_elle_path: str | Path,
    *,
    outdir: str | Path,
    init_elle_attribute: str = "auto",
    label_attribute: str = "auto",
    checkpoint_name: str | None = None,
    field_names: list[str] | tuple[str, ...] | None = None,
    save_preview: bool = False,
    save_topology: bool = True,
    save_mesh: bool = True,
    save_elle: bool = True,
    **setup_kwargs: Any,
) -> dict[str, Any]:
    """Run one frozen-mechanics outer step with faithful GBM/recovery and compare the ELLE transition."""

    setup = build_faithful_gbm_setup(
        init_elle_path,
        init_elle_attribute=init_elle_attribute,
        mechanics_snapshot_dir=mechanics_snapshot_dir,
        **setup_kwargs,
    )
    if not setup.mechanics_snapshots:
        raise ValueError("mechanics outer-step validation requires at least one mechanics snapshot")

    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)
    checkpoint_paths: dict[str, dict[str, str]] = {}

    def _save_snapshot(step: int, phi, topology_snapshot, mesh_feedback_context=None) -> None:
        mesh_state = None if mesh_feedback_context is None else mesh_feedback_context.get("mesh_state")
        written = save_snapshot_artifacts(
            outdir_path,
            int(step),
            phi,
            save_preview=bool(save_preview),
            save_elle=bool(save_elle),
            tracked_topology=topology_snapshot,
            save_topology=bool(save_topology),
            mesh_state=mesh_state,
            save_mesh=bool(save_mesh),
        )
        checkpoint_paths[str(int(step))] = {
            str(name): str(path) for name, path in written.items()
        }

    _, snapshots, topology_history = run_faithful_gbm_simulation(
        setup=setup,
        steps=1,
        save_every=1,
        on_snapshot=_save_snapshot,
        include_initial_snapshot=True,
    )

    topology_history_path: str | None = None
    if save_topology:
        history_path = write_topology_history(outdir_path / "topology_history.json", topology_history)
        topology_history_path = str(history_path)

    try:
        candidate_before = checkpoint_paths["0"]["elle"]
        candidate_after = checkpoint_paths["1"]["elle"]
    except KeyError as exc:
        raise ValueError(
            "mechanics outer-step validation requires step-0 and step-1 ELLE artifacts"
        ) from exc

    reference_transition = extract_legacy_reference_transition(
        reference_before_elle_path,
        reference_after_elle_path,
        checkpoint_name=checkpoint_name,
        label_attribute=label_attribute,
        field_names=field_names,
    )
    report = compare_legacy_reference_transition(
        candidate_before,
        candidate_after,
        reference_transition,
        label_attribute=label_attribute,
    )
    return {
        **report,
        "replay": {
            "output_dir": str(outdir_path),
            "steps": 1,
            "written_snapshots": int(len(snapshots)),
            "mechanics_snapshot_count": int(len(setup.mechanics_snapshots)),
            "checkpoint_paths": checkpoint_paths,
            "topology_history_path": topology_history_path,
        },
        "reference_before_path": str(Path(reference_before_elle_path)),
        "reference_after_path": str(Path(reference_after_elle_path)),
    }
