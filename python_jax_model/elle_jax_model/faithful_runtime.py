from __future__ import annotations

import copy
from typing import Any, Callable

import numpy as np

from .artifacts import dominant_grain_map
from .faithful_config import FaithfulSolverConfig
from .fft_bridge import FFTMechanicsSnapshot, apply_legacy_fft_snapshot_to_mesh_state
from .mesh import MeshFeedbackConfig, couple_mesh_to_order_parameters, mesh_labels_to_order_parameters
from .nucleation import NucleationConfig, apply_nucleation_stage
from .recovery import RecoveryConfig, apply_recovery_stage
from .topology import TopologyTracker, run_simulation_with_topology


def _select_mechanics_snapshot_for_outer_step(
    mechanics_snapshot: FFTMechanicsSnapshot | None,
    mechanics_snapshots: tuple[FFTMechanicsSnapshot, ...] | None,
    outer_step: int,
) -> tuple[FFTMechanicsSnapshot | None, int, int]:
    if mechanics_snapshots:
        snapshot_count = int(len(mechanics_snapshots))
        snapshot_index = min(max(int(outer_step) - 1, 0), snapshot_count - 1)
        return mechanics_snapshots[snapshot_index], int(snapshot_index + 1), snapshot_count
    if mechanics_snapshot is not None:
        return mechanics_snapshot, 1, 1
    return None, 0, 0


def initialize_faithful_order_parameters(config: FaithfulSolverConfig) -> np.ndarray:
    labels = np.asarray(config.seed_data.label_field, dtype=np.int32)
    nx, ny = labels.shape
    if (int(config.nx), int(config.ny)) != (nx, ny):
        raise ValueError(
            f"faithful config grid {(config.nx, config.ny)} does not match ELLE grid {(nx, ny)} "
            f"from {config.init_elle_path}"
        )
    if int(config.num_grains) < int(config.seed_data.num_labels):
        raise ValueError(
            f"faithful num_grains={config.num_grains} is smaller than ELLE label count "
            f"{config.seed_data.num_labels} from {config.init_elle_path}"
        )

    grain_ids = np.arange(int(config.num_grains), dtype=np.int32)[:, None, None]
    return (labels[None, :, :] == grain_ids).astype(np.float32)


def run_faithful_simulation(
    config: FaithfulSolverConfig,
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object, dict[str, Any] | None], None] | None = None,
    mesh_feedback: MeshFeedbackConfig | None = None,
    mechanics_snapshot: FFTMechanicsSnapshot | None = None,
    mechanics_snapshots: tuple[FFTMechanicsSnapshot, ...] | None = None,
    include_initial_snapshot: bool = False,
    subloops_per_snapshot: int = 1,
    gbm_steps_per_subloop: int = 1,
    nucleation_steps_per_subloop: int = 0,
    nucleation_config: NucleationConfig | None = None,
    recovery_steps_per_subloop: int = 0,
    recovery_config: RecoveryConfig | None = None,
) -> tuple[object, list[object]]:
    if mesh_feedback is None or str(mesh_feedback.update_mode) != "mesh_only":
        raise ValueError("faithful simulation requires mesh_feedback.update_mode='mesh_only'")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if int(mesh_feedback.every) < 1:
        raise ValueError("faithful simulation requires mesh_feedback.every >= 1")
    if int(subloops_per_snapshot) < 1:
        raise ValueError("faithful simulation requires subloops_per_snapshot >= 1")
    if int(gbm_steps_per_subloop) < 0:
        raise ValueError("faithful simulation requires gbm_steps_per_subloop >= 0")
    if int(nucleation_steps_per_subloop) < 0:
        raise ValueError("faithful simulation requires nucleation_steps_per_subloop >= 0")
    if int(recovery_steps_per_subloop) < 0:
        raise ValueError("faithful simulation requires recovery_steps_per_subloop >= 0")
    if nucleation_config is None:
        nucleation_config = NucleationConfig()
    if recovery_config is None:
        recovery_config = RecoveryConfig()
    has_mechanics_snapshot = bool(mechanics_snapshots) or mechanics_snapshot is not None
    subloop_stage_count = (
        int(nucleation_steps_per_subloop)
        + int(gbm_steps_per_subloop)
        + int(recovery_steps_per_subloop)
    )
    if subloop_stage_count < 1 and not has_mechanics_snapshot:
        raise ValueError(
            "faithful simulation requires at least one GBM/nucleation/recovery stage per subloop or a mechanics snapshot"
        )

    phi = initialize_faithful_order_parameters(config)
    snapshots: list[object] = []
    runtime_topology_tracker = TopologyTracker()
    runtime_topology_snapshot = runtime_topology_tracker.update(phi, step=0)
    runtime_mesh_state = (
        copy.deepcopy(mesh_feedback.initial_mesh_state)
        if mesh_feedback.initial_mesh_state is not None
        else None
    )

    stages_per_snapshot = int(subloops_per_snapshot) * int(subloop_stage_count) + (
        1 if has_mechanics_snapshot else 0
    )

    if include_initial_snapshot:
        initial_context: dict[str, Any] = {
            "solver_backend": "numpy_mesh_only",
            "outer_step": 0,
            "stage_index": 0,
            "stages_per_snapshot": stages_per_snapshot,
            "subloops_per_snapshot": int(subloops_per_snapshot),
            "mechanics_snapshot_enabled": int(has_mechanics_snapshot),
            "mechanics_snapshot_count": int(len(mechanics_snapshots or ())) if mechanics_snapshots else int(mechanics_snapshot is not None),
            "gbm_steps_per_subloop": int(gbm_steps_per_subloop),
            "nucleation_steps_per_subloop": int(nucleation_steps_per_subloop),
            "recovery_steps_per_subloop": int(recovery_steps_per_subloop),
        }
        if runtime_mesh_state is not None:
            initial_mesh_state = copy.deepcopy(runtime_mesh_state)
            initial_mesh_state["stats"]["mesh_solver_backend"] = "numpy_mesh_only"
            initial_mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
            initial_mesh_state["stats"]["workflow_subloops_per_snapshot"] = int(subloops_per_snapshot)
            initial_mesh_state["stats"]["workflow_mechanics_steps_per_snapshot"] = int(has_mechanics_snapshot)
            initial_mesh_state["stats"]["workflow_mechanics_snapshot_count"] = int(
                len(mechanics_snapshots or ())
            ) if mechanics_snapshots else int(mechanics_snapshot is not None)
            initial_mesh_state["stats"]["workflow_gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
            initial_mesh_state["stats"]["workflow_nucleation_steps_per_subloop"] = int(
                nucleation_steps_per_subloop
            )
            initial_mesh_state["stats"]["workflow_recovery_steps_per_subloop"] = int(
                recovery_steps_per_subloop
            )
            initial_mesh_state["stats"]["workflow_stages_per_snapshot"] = stages_per_snapshot
            initial_context["mesh_state"] = initial_mesh_state
        initial_context["tracked_topology_snapshot"] = runtime_topology_snapshot
        initial_snapshot = np.asarray(phi, dtype=np.float32).copy()
        snapshots.append(initial_snapshot)
        if on_snapshot is not None:
            on_snapshot(0, initial_snapshot, initial_context)

    stage_index = 0
    for outer_step in range(1, steps + 1):
        mesh_feedback_context: dict[str, Any] | None = None
        recovery_applied_stages_total = 0
        recovery_rotated_unodes_total = 0
        recovery_density_reduced_unodes_total = 0
        mechanics_applied_total = 0
        current_mechanics_snapshot, mechanics_snapshot_index, mechanics_snapshot_count = (
            _select_mechanics_snapshot_for_outer_step(
                mechanics_snapshot,
                mechanics_snapshots,
                outer_step,
            )
        )
        if current_mechanics_snapshot is not None:
            stage_index += 1
            mechanics_mesh_state = (
                runtime_mesh_state
                if runtime_mesh_state is not None
                else copy.deepcopy(mesh_feedback.initial_mesh_state)
            )
            if mechanics_mesh_state is None:
                mechanics_mesh_state = {"stats": {}}
            mechanics_mesh_state, mechanics_stats = apply_legacy_fft_snapshot_to_mesh_state(
                mechanics_mesh_state,
                current_mechanics_snapshot,
            )
            mechanics_stats = {
                **mechanics_stats,
                "snapshot_index": int(mechanics_snapshot_index),
                "snapshot_count": int(mechanics_snapshot_count),
            }
            mechanics_applied_total += int(mechanics_stats.get("mechanics_applied", 0))
            mechanics_mesh_state.setdefault("stats", {})
            mechanics_mesh_state["stats"]["mesh_solver_backend"] = "numpy_mesh_only"
            mechanics_mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
            mechanics_mesh_state["stats"]["workflow_subloops_per_snapshot"] = int(subloops_per_snapshot)
            mechanics_mesh_state["stats"]["workflow_mechanics_steps_per_snapshot"] = int(1)
            mechanics_mesh_state["stats"]["workflow_mechanics_snapshot_count"] = int(
                mechanics_snapshot_count
            )
            mechanics_mesh_state["stats"]["workflow_mechanics_snapshot_index"] = int(
                mechanics_snapshot_index
            )
            mechanics_mesh_state["stats"]["workflow_gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
            mechanics_mesh_state["stats"]["workflow_nucleation_steps_per_subloop"] = int(
                nucleation_steps_per_subloop
            )
            mechanics_mesh_state["stats"]["workflow_recovery_steps_per_subloop"] = int(
                recovery_steps_per_subloop
            )
            mechanics_mesh_state["stats"]["workflow_stages_per_snapshot"] = stages_per_snapshot
            mechanics_mesh_state["stats"]["workflow_mechanics_applied_total"] = int(
                mechanics_applied_total
            )
            runtime_mesh_state = mechanics_mesh_state
            mesh_feedback_context = {
                "mesh_state": mechanics_mesh_state,
                "mechanics_stats": mechanics_stats,
                "solver_backend": "numpy_mesh_only",
            }
            runtime_topology_snapshot = runtime_topology_tracker.reuse_previous(stage_index)
            mesh_feedback_context["tracked_topology_snapshot"] = runtime_topology_snapshot
            mesh_feedback_context["outer_step"] = int(outer_step)
            mesh_feedback_context["stage_index"] = int(stage_index)
            mesh_feedback_context["stages_per_snapshot"] = stages_per_snapshot
            mesh_feedback_context["subloops_per_snapshot"] = int(subloops_per_snapshot)
            mesh_feedback_context["mechanics_snapshot_enabled"] = 1
            mesh_feedback_context["mechanics_snapshot_index"] = int(mechanics_snapshot_index)
            mesh_feedback_context["mechanics_snapshot_count"] = int(mechanics_snapshot_count)
            mesh_feedback_context["gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
            mesh_feedback_context["nucleation_steps_per_subloop"] = int(
                nucleation_steps_per_subloop
            )
            mesh_feedback_context["recovery_steps_per_subloop"] = int(recovery_steps_per_subloop)
            mesh_feedback_context["stage_kind"] = "mechanics"
        for subloop_index in range(1, int(subloops_per_snapshot) + 1):
            for nucleation_index in range(1, int(nucleation_steps_per_subloop) + 1):
                stage_index += 1
                nucleation_labels = dominant_grain_map(np.asarray(phi, dtype=np.float32))
                nucleation_mesh_state = (
                    runtime_mesh_state
                    if runtime_mesh_state is not None
                    else copy.deepcopy(mesh_feedback.initial_mesh_state)
                )
                if nucleation_mesh_state is None:
                    nucleation_mesh_state = {"stats": {}}
                nucleation_mesh_state.setdefault("stats", {})
                nucleation_mesh_state["stats"]["workflow_subloops_per_snapshot"] = int(subloops_per_snapshot)
                nucleation_mesh_state["stats"]["workflow_mechanics_steps_per_snapshot"] = int(
                    has_mechanics_snapshot
                )
                nucleation_mesh_state["stats"]["workflow_mechanics_snapshot_count"] = int(
                    mechanics_snapshot_count
                )
                nucleation_mesh_state["stats"]["workflow_mechanics_snapshot_index"] = int(
                    mechanics_snapshot_index
                )
                nucleation_mesh_state["stats"]["workflow_gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
                nucleation_mesh_state["stats"]["workflow_nucleation_steps_per_subloop"] = int(
                    nucleation_steps_per_subloop
                )
                nucleation_mesh_state["stats"]["workflow_recovery_steps_per_subloop"] = int(
                    recovery_steps_per_subloop
                )
                nucleation_mesh_state["stats"]["workflow_stages_per_snapshot"] = stages_per_snapshot
                nucleation_labels, nucleation_mesh_state, nucleation_stats = apply_nucleation_stage(
                    nucleation_mesh_state,
                    nucleation_labels,
                    nucleation_config,
                    nucleation_stage_index=int(nucleation_index - 1),
                )
                phi = mesh_labels_to_order_parameters(
                    nucleation_labels,
                    max(int(np.asarray(phi).shape[0]), int(np.max(nucleation_labels)) + 1),
                )
                nucleation_mesh_state.setdefault("stats", {})
                nucleation_mesh_state["stats"]["mesh_solver_backend"] = "numpy_mesh_only"
                nucleation_mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
                nucleation_mesh_state["stats"]["workflow_subloops_per_snapshot"] = int(subloops_per_snapshot)
                nucleation_mesh_state["stats"]["workflow_gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
                nucleation_mesh_state["stats"]["workflow_nucleation_steps_per_subloop"] = int(
                    nucleation_steps_per_subloop
                )
                nucleation_mesh_state["stats"]["workflow_recovery_steps_per_subloop"] = int(
                    recovery_steps_per_subloop
                )
                nucleation_mesh_state["stats"]["workflow_stages_per_snapshot"] = stages_per_snapshot
                nucleation_mesh_state["stats"]["workflow_mechanics_applied_total"] = int(
                    mechanics_applied_total
                )
                runtime_mesh_state = nucleation_mesh_state
                mesh_feedback_context = {
                    "mesh_state": nucleation_mesh_state,
                    "nucleation_stats": nucleation_stats,
                    "mechanics_stats": mesh_feedback_context.get("mechanics_stats") if isinstance(mesh_feedback_context, dict) else None,
                    "solver_backend": "numpy_mesh_only",
                }
                runtime_topology_snapshot = runtime_topology_tracker.reuse_previous(stage_index)
                mesh_feedback_context["tracked_topology_snapshot"] = runtime_topology_snapshot
                mesh_feedback_context["outer_step"] = int(outer_step)
                mesh_feedback_context["subloop_index"] = int(subloop_index)
                mesh_feedback_context["nucleation_stage_in_subloop"] = int(nucleation_index)
                mesh_feedback_context["stage_index"] = int(stage_index)
                mesh_feedback_context["stages_per_snapshot"] = stages_per_snapshot
                mesh_feedback_context["subloops_per_snapshot"] = int(subloops_per_snapshot)
                mesh_feedback_context["mechanics_snapshot_enabled"] = int(has_mechanics_snapshot)
                mesh_feedback_context["mechanics_snapshot_index"] = int(mechanics_snapshot_index)
                mesh_feedback_context["mechanics_snapshot_count"] = int(mechanics_snapshot_count)
                mesh_feedback_context["gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
                mesh_feedback_context["nucleation_steps_per_subloop"] = int(
                    nucleation_steps_per_subloop
                )
                mesh_feedback_context["recovery_steps_per_subloop"] = int(recovery_steps_per_subloop)
                mesh_feedback_context["stage_kind"] = "nucleation"

            for gbm_index in range(1, int(gbm_steps_per_subloop) + 1):
                stage_index += 1
                if stage_index % int(mesh_feedback.every) == 0:
                    phi_feedback, mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                        phi,
                        mesh_feedback,
                        tracked_topology=runtime_topology_snapshot,
                        base_mesh_state=runtime_mesh_state,
                    )
                    phi = np.asarray(phi_feedback, dtype=np.float32)
                    if runtime_mesh_state is not None:
                        runtime_mesh_state = copy.deepcopy(mesh_state)
                    mesh_state["stats"]["mesh_solver_backend"] = "numpy_mesh_only"
                    mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
                    mesh_state["stats"]["workflow_subloops_per_snapshot"] = int(subloops_per_snapshot)
                    mesh_state["stats"]["workflow_mechanics_steps_per_snapshot"] = int(
                        has_mechanics_snapshot
                    )
                    mesh_state["stats"]["workflow_mechanics_snapshot_count"] = int(
                        mechanics_snapshot_count
                    )
                    mesh_state["stats"]["workflow_mechanics_snapshot_index"] = int(
                        mechanics_snapshot_index
                    )
                    mesh_state["stats"]["workflow_gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
                    mesh_state["stats"]["workflow_nucleation_steps_per_subloop"] = int(
                        nucleation_steps_per_subloop
                    )
                    mesh_state["stats"]["workflow_stages_per_snapshot"] = stages_per_snapshot
                    mesh_state["stats"]["workflow_mechanics_applied_total"] = int(
                        mechanics_applied_total
                    )
                    mesh_feedback_context = {
                        "mesh_state": mesh_state,
                        "feedback_stats": feedback_stats,
                        "mechanics_stats": mesh_feedback_context.get("mechanics_stats") if isinstance(mesh_feedback_context, dict) else None,
                        "solver_backend": "numpy_mesh_only",
                    }

                runtime_topology_snapshot = runtime_topology_tracker.update(phi, stage_index)
                if mesh_feedback_context is None:
                    mesh_feedback_context = {"solver_backend": "numpy_mesh_only"}
                mesh_feedback_context["tracked_topology_snapshot"] = runtime_topology_snapshot
                mesh_feedback_context["outer_step"] = int(outer_step)
                mesh_feedback_context["subloop_index"] = int(subloop_index)
                mesh_feedback_context["gbm_stage_in_subloop"] = int(gbm_index)
                mesh_feedback_context["stage_index"] = int(stage_index)
                mesh_feedback_context["stages_per_snapshot"] = stages_per_snapshot
                mesh_feedback_context["subloops_per_snapshot"] = int(subloops_per_snapshot)
                mesh_feedback_context["mechanics_snapshot_enabled"] = int(has_mechanics_snapshot)
                mesh_feedback_context["mechanics_snapshot_index"] = int(mechanics_snapshot_index)
                mesh_feedback_context["mechanics_snapshot_count"] = int(mechanics_snapshot_count)
                mesh_feedback_context["gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
                mesh_feedback_context["nucleation_steps_per_subloop"] = int(
                    nucleation_steps_per_subloop
                )
                mesh_feedback_context["recovery_steps_per_subloop"] = int(recovery_steps_per_subloop)
                mesh_feedback_context["stage_kind"] = "gbm"

            recovery_labels = None
            if int(recovery_steps_per_subloop) > 0:
                recovery_labels = dominant_grain_map(np.asarray(phi, dtype=np.float32))
            for recovery_index in range(1, int(recovery_steps_per_subloop) + 1):
                stage_index += 1
                recovery_mesh_state = (
                    runtime_mesh_state
                    if runtime_mesh_state is not None
                    else copy.deepcopy(mesh_feedback.initial_mesh_state)
                )
                if recovery_mesh_state is None:
                    recovery_mesh_state = {"stats": {}}
                recovery_labels = dominant_grain_map(np.asarray(phi, dtype=np.float32))
                recovery_mesh_state, recovery_stats = apply_recovery_stage(
                    recovery_mesh_state,
                    recovery_labels,
                    recovery_config,
                    recovery_stage_index=int(recovery_index - 1),
                )
                recovery_applied_stages_total += int(recovery_stats.get("recovery_applied", 0))
                recovery_rotated_unodes_total += int(recovery_stats.get("rotated_unodes", 0))
                recovery_density_reduced_unodes_total += int(
                    recovery_stats.get("density_reduced_unodes", 0)
                )
                recovery_mesh_state.setdefault("stats", {})
                recovery_mesh_state["stats"]["mesh_solver_backend"] = "numpy_mesh_only"
                recovery_mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
                recovery_mesh_state["stats"]["workflow_subloops_per_snapshot"] = int(subloops_per_snapshot)
                recovery_mesh_state["stats"]["workflow_mechanics_steps_per_snapshot"] = int(
                    has_mechanics_snapshot
                )
                recovery_mesh_state["stats"]["workflow_mechanics_snapshot_count"] = int(
                    mechanics_snapshot_count
                )
                recovery_mesh_state["stats"]["workflow_mechanics_snapshot_index"] = int(
                    mechanics_snapshot_index
                )
                recovery_mesh_state["stats"]["workflow_gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
                recovery_mesh_state["stats"]["workflow_nucleation_steps_per_subloop"] = int(
                    nucleation_steps_per_subloop
                )
                recovery_mesh_state["stats"]["workflow_recovery_steps_per_subloop"] = int(
                    recovery_steps_per_subloop
                )
                recovery_mesh_state["stats"]["workflow_stages_per_snapshot"] = stages_per_snapshot
                recovery_mesh_state["stats"]["workflow_mechanics_applied_total"] = int(
                    mechanics_applied_total
                )
                recovery_mesh_state["stats"]["workflow_recovery_applied_stages_total"] = int(
                    recovery_applied_stages_total
                )
                recovery_mesh_state["stats"]["workflow_recovery_rotated_unodes_total"] = int(
                    recovery_rotated_unodes_total
                )
                recovery_mesh_state["stats"]["workflow_recovery_density_reduced_unodes_total"] = int(
                    recovery_density_reduced_unodes_total
                )
                runtime_mesh_state = recovery_mesh_state
                mesh_feedback_context = {
                    "mesh_state": recovery_mesh_state,
                    "mechanics_stats": mesh_feedback_context.get("mechanics_stats") if isinstance(mesh_feedback_context, dict) else None,
                    "recovery_stats": recovery_stats,
                    "recovery_cumulative_stats": {
                        "recovery_applied_stages_total": int(recovery_applied_stages_total),
                        "recovery_rotated_unodes_total": int(recovery_rotated_unodes_total),
                        "recovery_density_reduced_unodes_total": int(
                            recovery_density_reduced_unodes_total
                        ),
                    },
                    "solver_backend": "numpy_mesh_only",
                }
                runtime_topology_snapshot = runtime_topology_tracker.reuse_previous(stage_index)
                mesh_feedback_context["tracked_topology_snapshot"] = runtime_topology_snapshot
                mesh_feedback_context["outer_step"] = int(outer_step)
                mesh_feedback_context["subloop_index"] = int(subloop_index)
                mesh_feedback_context["recovery_stage_in_subloop"] = int(recovery_index)
                mesh_feedback_context["stage_index"] = int(stage_index)
                mesh_feedback_context["stages_per_snapshot"] = stages_per_snapshot
                mesh_feedback_context["subloops_per_snapshot"] = int(subloops_per_snapshot)
                mesh_feedback_context["mechanics_snapshot_enabled"] = int(has_mechanics_snapshot)
                mesh_feedback_context["mechanics_snapshot_index"] = int(mechanics_snapshot_index)
                mesh_feedback_context["mechanics_snapshot_count"] = int(mechanics_snapshot_count)
                mesh_feedback_context["gbm_steps_per_subloop"] = int(gbm_steps_per_subloop)
                mesh_feedback_context["nucleation_steps_per_subloop"] = int(
                    nucleation_steps_per_subloop
                )
                mesh_feedback_context["recovery_steps_per_subloop"] = int(recovery_steps_per_subloop)
                mesh_feedback_context["stage_kind"] = "recovery"

        if outer_step % save_every == 0 or outer_step == steps:
            snapshot = np.asarray(phi, dtype=np.float32).copy()
            snapshots.append(snapshot)
            if on_snapshot is not None:
                on_snapshot(outer_step, snapshot, mesh_feedback_context)

    return np.asarray(phi, dtype=np.float32), snapshots


def run_faithful_simulation_with_topology(
    config: FaithfulSolverConfig,
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object, dict[str, Any], dict[str, Any] | None], None] | None = None,
    mesh_feedback: MeshFeedbackConfig | None = None,
    include_initial_snapshot: bool = False,
    subloops_per_snapshot: int = 1,
    gbm_steps_per_subloop: int = 1,
    nucleation_steps_per_subloop: int = 0,
    recovery_steps_per_subloop: int = 0,
    mechanics_snapshot: FFTMechanicsSnapshot | None = None,
    mechanics_snapshots: tuple[FFTMechanicsSnapshot, ...] | None = None,
    nucleation_config: NucleationConfig | None = None,
    recovery_config: RecoveryConfig | None = None,
) -> tuple[object, list[object], list[dict[str, Any]]]:
    def _runner(**kwargs):
        return run_faithful_simulation(
            mechanics_snapshot=mechanics_snapshot,
            mechanics_snapshots=mechanics_snapshots,
            include_initial_snapshot=include_initial_snapshot,
            subloops_per_snapshot=int(subloops_per_snapshot),
            gbm_steps_per_subloop=int(gbm_steps_per_subloop),
            nucleation_steps_per_subloop=int(nucleation_steps_per_subloop),
            nucleation_config=nucleation_config,
            recovery_steps_per_subloop=int(recovery_steps_per_subloop),
            recovery_config=recovery_config,
            **kwargs,
        )

    return run_simulation_with_topology(
        config=config,
        steps=steps,
        save_every=save_every,
        on_snapshot=on_snapshot,
        mesh_feedback=mesh_feedback,
        runner=_runner,
    )
