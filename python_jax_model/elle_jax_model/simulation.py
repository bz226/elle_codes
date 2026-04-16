from __future__ import annotations

import copy
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, List, Tuple

import numpy as np

try:
    import jax
    import jax.numpy as jnp

    JAX_AVAILABLE = True
except ModuleNotFoundError:  # fallback for environments without jax
    import numpy as jnp

    JAX_AVAILABLE = False

    class _RandomCompat:
        @staticmethod
        def PRNGKey(seed: int) -> int:
            return seed

        @staticmethod
        def split(key: int, num: int = 2):
            return tuple(key + index + 1 for index in range(num))

        @staticmethod
        def uniform(key: int, shape: tuple[int, ...], minval: float, maxval: float):
            rng = jnp.random.default_rng(key)
            return rng.uniform(minval, maxval, size=shape)

    class _JaxCompat:
        random = _RandomCompat()

        @staticmethod
        def jit(fn):
            return fn

    jax = _JaxCompat()  # type: ignore

from .mesh import MeshFeedbackConfig, compute_mesh_motion_velocity, couple_mesh_to_order_parameters


@dataclass(frozen=True)
class GrainGrowthConfig:
    """Configuration for a simple multiphase grain-growth simulation."""

    nx: int = 128
    ny: int = 128
    num_grains: int = 12
    dt: float = 0.05
    mobility: float = 1.0
    gradient_penalty: float = 1.0
    interaction_strength: float = 2.0
    seed: int = 0
    init_mode: str = "random"
    init_elle_path: str | None = None
    init_elle_attribute: str = "auto"
    init_smoothing_steps: int = 2
    init_noise: float = 0.02


def _normalize_fields(phi):
    total = jnp.sum(phi, axis=0, keepdims=True) + 1e-12
    return phi / total


def _initialize_random(config: GrainGrowthConfig):
    key = jax.random.PRNGKey(config.seed)
    raw = jax.random.uniform(
        key,
        shape=(config.num_grains, config.nx, config.ny),
        minval=0.0,
        maxval=1.0,
    )
    return _normalize_fields(raw)


def _initialize_voronoi(config: GrainGrowthConfig):
    seed_key, noise_key = jax.random.split(jax.random.PRNGKey(config.seed), 2)
    seeds = jax.random.uniform(
        seed_key,
        shape=(config.num_grains, 2),
        minval=0.0,
        maxval=1.0,
    )

    x_coords = jnp.linspace(0.0, 1.0, config.nx, endpoint=False)[:, None]
    y_coords = jnp.linspace(0.0, 1.0, config.ny, endpoint=False)[None, :]

    dx = jnp.abs(x_coords[None, :, :] - seeds[:, 0][:, None, None])
    dy = jnp.abs(y_coords[None, :, :] - seeds[:, 1][:, None, None])
    dx = jnp.minimum(dx, 1.0 - dx)
    dy = jnp.minimum(dy, 1.0 - dy)
    distances = dx * dx + dy * dy

    labels = jnp.argmin(distances, axis=0)
    grain_ids = jnp.arange(config.num_grains)[:, None, None]
    phi = (labels[None, :, :] == grain_ids).astype(distances.dtype)

    if config.init_noise > 0.0:
        phi = phi + jax.random.uniform(
            noise_key,
            shape=phi.shape,
            minval=0.0,
            maxval=config.init_noise,
        )

    phi = _normalize_fields(phi)
    for _ in range(max(config.init_smoothing_steps, 0)):
        neighbor_average = 0.25 * (
            jnp.roll(phi, +1, axis=1)
            + jnp.roll(phi, -1, axis=1)
            + jnp.roll(phi, +1, axis=2)
            + jnp.roll(phi, -1, axis=2)
        )
        phi = _normalize_fields(0.5 * phi + 0.5 * neighbor_average)
    return phi


def load_elle_label_seed(
    elle_path: str | Path,
    *,
    attribute: str = "auto",
) -> dict[str, object]:
    from .elle_visualize import (
        _field_from_sparse_unodes,
        _parse_elle_sections,
        _parse_flynns,
        _parse_sparse_values,
        _parse_unodes,
    )

    path = Path(elle_path)
    sections = _parse_elle_sections(path)
    if "UNODES" not in sections:
        raise ValueError(f"ELLE file has no UNODES section: {path}")

    unodes = _parse_unodes(sections["UNODES"])
    requested_attribute = str(attribute)
    if requested_attribute.lower() == "auto":
        flynn_count = len(_parse_flynns(sections.get("FLYNNS", ())))
        attr_candidates = [
            name for name in sections if name.startswith("U_ATTRIB_") or name.startswith("U_CONC_")
        ]
        if not attr_candidates:
            raise ValueError(f"ELLE file has no unode attribute sections to seed from: {path}")

        scored_candidates: list[tuple[tuple[float, float, str], str]] = []
        integer_like_candidates: set[str] = set()
        for name in attr_candidates:
            field_np, _, _ = _field_from_sparse_unodes(unodes, sections[name])
            array = np.asarray(field_np, dtype=np.float64)
            rounded = np.rint(array)
            integer_residual = float(np.max(np.abs(array - rounded))) if array.size else float("inf")
            unique_labels = sorted(int(value) for value in np.unique(rounded))
            unique_count = len(unique_labels)
            if unique_count <= 1:
                continue
            if integer_residual <= 5.1e-1:
                integer_like_candidates.add(str(name))
                closeness = abs(unique_count - flynn_count) if flynn_count > 0 else float(unique_count)
                scored_candidates.append(((closeness, integer_residual, float(unique_count), name), name))

        if "U_ATTRIB_C" in integer_like_candidates:
            requested_attribute = "U_ATTRIB_C"
        elif scored_candidates:
            scored_candidates.sort(key=lambda item: item[0])
            requested_attribute = scored_candidates[0][1]
        else:
            raise ValueError(f"could not auto-detect an integer-like ELLE grain-label attribute in {path}")
    elif requested_attribute not in sections:
        raise ValueError(f"ELLE file has no {requested_attribute} section: {path}")

    field, grid_shape, layout = _field_from_sparse_unodes(unodes, sections[requested_attribute])
    label_field = np.rint(np.asarray(field, dtype=np.float64)).astype(np.int32)
    unique_labels = sorted(int(label) for label in np.unique(label_field))
    label_to_index = {label: index for index, label in enumerate(unique_labels)}
    compact = np.vectorize(label_to_index.get, otypes=[np.int32])(label_field)
    id_lookup = layout["id_lookup"]
    ordered_unodes = sorted((int(unode_id), float(x_coord), float(y_coord)) for unode_id, x_coord, y_coord in unodes)
    unode_field_values: dict[str, tuple[float, ...]] = {}
    field_order: list[str] = []
    for section_name in sections:
        if not section_name.startswith("U_"):
            continue
        if section_name == "UNODES":
            continue
        try:
            default_value, sparse_values = _parse_sparse_values(sections[section_name])
        except Exception:
            continue
        unode_field_values[section_name] = tuple(
            float(sparse_values.get(int(unode_id), default_value))
            for unode_id, _, _ in ordered_unodes
        )
        field_order.append(section_name)
    return {
        "path": str(path),
        "attribute": requested_attribute,
        "label_field": compact,
        "source_labels": unique_labels,
        "grid_shape": tuple(int(value) for value in grid_shape),
        "num_labels": int(len(unique_labels)),
        "unode_ids": tuple(int(unode_id) for unode_id, _, _ in ordered_unodes),
        "unode_positions": tuple((float(x_coord), float(y_coord)) for _, x_coord, y_coord in ordered_unodes),
        "unode_grid_indices": tuple(
            (
                int(id_lookup[int(unode_id)][0]),
                int(id_lookup[int(unode_id)][1]),
            )
            for unode_id, _, _ in ordered_unodes
        ),
        "unode_field_values": unode_field_values,
        "unode_field_order": tuple(field_order),
    }


def _initialize_from_elle(config: GrainGrowthConfig):
    if not config.init_elle_path:
        raise ValueError("init_elle_path is required when init_mode='elle'")

    seed = load_elle_label_seed(config.init_elle_path, attribute=config.init_elle_attribute)
    label_field = np.asarray(seed["label_field"], dtype=np.int32)
    nx, ny = label_field.shape
    if int(config.nx) != nx or int(config.ny) != ny:
        raise ValueError(
            f"config grid {(config.nx, config.ny)} does not match ELLE grid {(nx, ny)} "
            f"from {config.init_elle_path}"
        )
    num_labels = int(seed["num_labels"])
    if int(config.num_grains) < num_labels:
        raise ValueError(
            f"config.num_grains={config.num_grains} is smaller than ELLE label count {num_labels} "
            f"from {config.init_elle_path}"
        )

    grain_ids = np.arange(int(config.num_grains), dtype=np.int32)[:, None, None]
    phi = (label_field[None, :, :] == grain_ids).astype(np.float32)

    if config.init_noise > 0.0:
        noise_key = jax.random.PRNGKey(config.seed)
        phi = phi + jax.random.uniform(
            noise_key,
            shape=phi.shape,
            minval=0.0,
            maxval=config.init_noise,
        )

    phi = _normalize_fields(jnp.asarray(phi))
    for _ in range(max(config.init_smoothing_steps, 0)):
        neighbor_average = 0.25 * (
            jnp.roll(phi, +1, axis=1)
            + jnp.roll(phi, -1, axis=1)
            + jnp.roll(phi, +1, axis=2)
            + jnp.roll(phi, -1, axis=2)
        )
        phi = _normalize_fields(0.5 * phi + 0.5 * neighbor_average)
    return phi


def initialize_order_parameters(config: GrainGrowthConfig):
    """Create normalized random order parameters shaped (num_grains, nx, ny)."""

    if config.num_grains < 1:
        raise ValueError("num_grains must be >= 1")
    if config.nx < 1 or config.ny < 1:
        raise ValueError("nx and ny must be >= 1")

    init_mode = config.init_mode.lower()
    if init_mode == "random":
        return _initialize_random(config)
    if init_mode == "voronoi":
        return _initialize_voronoi(config)
    if init_mode == "elle":
        return _initialize_from_elle(config)
    raise ValueError(f"unsupported init_mode: {config.init_mode}")


def _laplacian_periodic(field):
    return (
        jnp.roll(field, +1, axis=1)
        + jnp.roll(field, -1, axis=1)
        + jnp.roll(field, +1, axis=2)
        + jnp.roll(field, -1, axis=2)
        - 4.0 * field
    )


def _mesh_advection_periodic(phi, mesh_velocity, config: GrainGrowthConfig):
    dx = 1.0 / float(config.nx)
    dy = 1.0 / float(config.ny)
    grad_x = (jnp.roll(phi, -1, axis=1) - jnp.roll(phi, +1, axis=1)) / (2.0 * dx)
    grad_y = (jnp.roll(phi, +1, axis=2) - jnp.roll(phi, -1, axis=2)) / (2.0 * dy)
    velocity_x = mesh_velocity[:, :, 0][None, :, :]
    velocity_y = mesh_velocity[:, :, 1][None, :, :]
    return velocity_x * grad_x + velocity_y * grad_y


def _step(phi, config: GrainGrowthConfig, kernel_velocity, kernel_strength=0.0):
    lap = _laplacian_periodic(phi)

    sum_phi_sq = jnp.sum(phi * phi, axis=0, keepdims=True)
    dfdphi = phi * (phi * phi - 1.0) + config.interaction_strength * phi * (
        sum_phi_sq - phi * phi
    )

    dphi_dt = config.mobility * (config.gradient_penalty * lap - dfdphi)
    dphi_dt = dphi_dt - kernel_strength * _mesh_advection_periodic(phi, kernel_velocity, config)
    phi_next = phi + config.dt * dphi_dt
    phi_next = jnp.clip(phi_next, 0.0, 1.0)

    norm = jnp.sum(phi_next, axis=0, keepdims=True) + 1e-12
    return phi_next / norm


def run_mesh_only_simulation(
    config: GrainGrowthConfig,
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object, dict | None], None] | None = None,
    mesh_feedback: MeshFeedbackConfig | None = None,
) -> Tuple[object, List[object]]:
    if mesh_feedback is None or str(mesh_feedback.update_mode) != "mesh_only":
        raise ValueError("mesh-only simulation requires mesh_feedback.update_mode='mesh_only'")
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if int(mesh_feedback.every) < 1:
        raise ValueError("mesh-only simulation requires mesh_feedback.every >= 1")

    phi = np.asarray(initialize_order_parameters(config), dtype=np.float32)
    snapshots: List[object] = []
    runtime_topology_tracker = None
    runtime_topology_snapshot = None
    runtime_mesh_state = None
    stateful_mesh_seeded = mesh_feedback.initial_mesh_state is not None

    from .topology import TopologyTracker

    runtime_topology_tracker = TopologyTracker()
    runtime_topology_snapshot = runtime_topology_tracker.update(phi, step=0)
    if stateful_mesh_seeded:
        runtime_mesh_state = copy.deepcopy(mesh_feedback.initial_mesh_state)

    for step in range(1, steps + 1):
        mesh_feedback_context = None
        if step % int(mesh_feedback.every) == 0:
            phi_feedback, mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                mesh_feedback,
                tracked_topology=runtime_topology_snapshot,
                base_mesh_state=runtime_mesh_state if stateful_mesh_seeded else None,
            )
            phi = np.asarray(phi_feedback, dtype=np.float32)
            if stateful_mesh_seeded:
                runtime_mesh_state = copy.deepcopy(mesh_state)
            mesh_state["stats"]["mesh_solver_backend"] = "numpy_mesh_only"
            if runtime_topology_snapshot is not None:
                mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
            mesh_feedback_context = {
                "mesh_state": mesh_state,
                "feedback_stats": feedback_stats,
                "solver_backend": "numpy_mesh_only",
            }

        if runtime_topology_tracker is not None:
            runtime_topology_snapshot = runtime_topology_tracker.update(phi, step)
            if mesh_feedback_context is None:
                mesh_feedback_context = {"solver_backend": "numpy_mesh_only"}
            mesh_feedback_context["tracked_topology_snapshot"] = runtime_topology_snapshot

        if step % save_every == 0 or step == steps:
            snapshot = np.asarray(phi, dtype=np.float32).copy()
            snapshots.append(snapshot)
            if on_snapshot is not None:
                on_snapshot(step, snapshot, mesh_feedback_context)

    return np.asarray(phi, dtype=np.float32), snapshots


def run_simulation(
    config: GrainGrowthConfig,
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object, dict | None], None] | None = None,
    mesh_feedback: MeshFeedbackConfig | None = None,
) -> Tuple[object, List[object]]:
    """Run simulation and optionally report snapshots."""

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")
    if mesh_feedback is not None and str(mesh_feedback.update_mode) == "mesh_only":
        return run_mesh_only_simulation(
            config=config,
            steps=steps,
            save_every=save_every,
            on_snapshot=on_snapshot,
            mesh_feedback=mesh_feedback,
        )

    phi = initialize_order_parameters(config)
    snapshots: List[object] = []
    # Capture config in the closure so JAX only traces array inputs.
    step_fn = jax.jit(
        lambda current_phi, kernel_velocity, kernel_strength: _step(
            current_phi,
            config,
            kernel_velocity,
            kernel_strength,
        )
    )
    mesh_only_mode = (
        mesh_feedback is not None
        and int(mesh_feedback.every) > 0
        and str(mesh_feedback.update_mode) == "mesh_only"
    )
    coupling_enabled = (
        mesh_feedback is not None
        and int(mesh_feedback.every) > 0
        and (
            mesh_only_mode
            or
            float(mesh_feedback.strength) > 0.0
            or float(mesh_feedback.transport_strength) > 0.0
        )
    )
    kernel_enabled = (
        not mesh_only_mode
        and
        mesh_feedback is not None
        and int(mesh_feedback.kernel_advection_every) > 0
        and float(mesh_feedback.kernel_advection_strength) > 0.0
    )
    runtime_topology_enabled = mesh_feedback is not None and (coupling_enabled or kernel_enabled)
    kernel_predictor_corrector = bool(
        kernel_enabled
        and mesh_feedback is not None
        and bool(mesh_feedback.kernel_predictor_corrector)
    )
    phi_dtype = jnp.asarray(phi).dtype
    kernel_velocity = jnp.zeros((config.nx, config.ny, 2), dtype=phi_dtype)
    kernel_strength = jnp.asarray(0.0, dtype=phi_dtype)
    kernel_transport_stats = {
        "transport_pixels": 0,
        "max_displacement": 0.0,
        "mean_displacement": 0.0,
    }
    kernel_mesh_state = None
    runtime_topology_tracker = None
    runtime_topology_snapshot = None
    runtime_mesh_state = None
    stateful_mesh_seeded = bool(mesh_feedback is not None and mesh_feedback.initial_mesh_state is not None)
    if runtime_topology_enabled:
        from .topology import TopologyTracker

        runtime_topology_tracker = TopologyTracker()
        runtime_topology_snapshot = runtime_topology_tracker.update(phi, step=0)
    if stateful_mesh_seeded:
        runtime_mesh_state = copy.deepcopy(mesh_feedback.initial_mesh_state)

    for step in range(1, steps + 1):
        if kernel_enabled and (step == 1 or (step - 1) % int(mesh_feedback.kernel_advection_every) == 0):
            _, kernel_mesh_state, velocity_field, _, kernel_transport_stats = compute_mesh_motion_velocity(
                phi,
                mesh_feedback,
                tracked_topology=runtime_topology_snapshot,
                base_mesh_state=runtime_mesh_state if stateful_mesh_seeded else None,
            )
            kernel_velocity = jnp.asarray(velocity_field, dtype=phi_dtype)
            kernel_strength = jnp.asarray(float(mesh_feedback.kernel_advection_strength), dtype=phi_dtype)
        elif not kernel_enabled:
            kernel_velocity = jnp.zeros((config.nx, config.ny, 2), dtype=phi_dtype)
            kernel_strength = jnp.asarray(0.0, dtype=phi_dtype)
            kernel_transport_stats = {
                "transport_pixels": 0,
                "max_displacement": 0.0,
                "mean_displacement": 0.0,
            }
            kernel_mesh_state = None

        if kernel_predictor_corrector:
            phi_trial = step_fn(phi, kernel_velocity, kernel_strength)
            (
                _,
                predictor_mesh_state,
                predictor_velocity_field,
                _,
                predictor_transport_stats,
            ) = compute_mesh_motion_velocity(
                phi_trial,
                mesh_feedback,
                tracked_topology=runtime_topology_snapshot,
                base_mesh_state=runtime_mesh_state if stateful_mesh_seeded else None,
            )
            corrected_velocity = 0.5 * (
                np.asarray(kernel_velocity, dtype=np.float32) + predictor_velocity_field
            )
            kernel_velocity = jnp.asarray(corrected_velocity, dtype=phi_dtype)
            kernel_mesh_state = predictor_mesh_state
            kernel_transport_stats = {
                "transport_pixels": int(predictor_transport_stats["transport_pixels"]),
                "max_displacement": max(
                    float(kernel_transport_stats["max_displacement"]),
                    float(predictor_transport_stats["max_displacement"]),
                ),
                "mean_displacement": 0.5
                * (
                    float(kernel_transport_stats["mean_displacement"])
                    + float(predictor_transport_stats["mean_displacement"])
                ),
            }

        mesh_feedback_context = None
        if mesh_only_mode and step % int(mesh_feedback.every) == 0:
            phi_feedback, mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                mesh_feedback,
                tracked_topology=runtime_topology_snapshot,
                base_mesh_state=runtime_mesh_state if stateful_mesh_seeded else None,
            )
            phi = jnp.asarray(phi_feedback, dtype=np.asarray(phi).dtype)
            if stateful_mesh_seeded:
                runtime_mesh_state = copy.deepcopy(mesh_state)
            if runtime_topology_snapshot is not None:
                mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
            mesh_feedback_context = {
                "mesh_state": mesh_state,
                "feedback_stats": feedback_stats,
            }
        else:
            phi = step_fn(phi, kernel_velocity, kernel_strength)

        if (not mesh_only_mode) and coupling_enabled and step % int(mesh_feedback.every) == 0:
            phi_feedback, mesh_state, feedback_stats = couple_mesh_to_order_parameters(
                phi,
                mesh_feedback,
                tracked_topology=runtime_topology_snapshot,
                base_mesh_state=runtime_mesh_state if stateful_mesh_seeded else None,
            )
            phi = jnp.asarray(phi_feedback, dtype=np.asarray(phi).dtype)
            if stateful_mesh_seeded:
                runtime_mesh_state = copy.deepcopy(mesh_state)
            if kernel_enabled:
                mesh_state["stats"]["mesh_kernel_advection_every"] = int(mesh_feedback.kernel_advection_every)
                mesh_state["stats"]["mesh_kernel_advection_strength"] = float(mesh_feedback.kernel_advection_strength)
                mesh_state["stats"]["mesh_kernel_predictor_corrector"] = int(kernel_predictor_corrector)
                mesh_state["stats"]["mesh_kernel_transport_pixels"] = int(
                    kernel_transport_stats["transport_pixels"]
                )
                mesh_state["stats"]["mesh_kernel_transport_max_displacement"] = float(
                    kernel_transport_stats["max_displacement"]
                )
                mesh_state["stats"]["mesh_kernel_transport_mean_displacement"] = float(
                    kernel_transport_stats["mean_displacement"]
                )
            if runtime_topology_snapshot is not None:
                mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
            mesh_feedback_context = {
                "mesh_state": mesh_state,
                "feedback_stats": feedback_stats,
            }
        elif kernel_enabled and kernel_mesh_state is not None:
            if stateful_mesh_seeded:
                runtime_mesh_state = copy.deepcopy(kernel_mesh_state)
            kernel_mesh_state["stats"]["mesh_kernel_advection_every"] = int(
                mesh_feedback.kernel_advection_every
            )
            kernel_mesh_state["stats"]["mesh_kernel_advection_strength"] = float(
                mesh_feedback.kernel_advection_strength
            )
            kernel_mesh_state["stats"]["mesh_kernel_predictor_corrector"] = int(
                kernel_predictor_corrector
            )
            kernel_mesh_state["stats"]["mesh_kernel_transport_pixels"] = int(
                kernel_transport_stats["transport_pixels"]
            )
            kernel_mesh_state["stats"]["mesh_kernel_transport_max_displacement"] = float(
                kernel_transport_stats["max_displacement"]
            )
            kernel_mesh_state["stats"]["mesh_kernel_transport_mean_displacement"] = float(
                kernel_transport_stats["mean_displacement"]
            )
            if runtime_topology_snapshot is not None:
                kernel_mesh_state["stats"]["mesh_runtime_topology_tracked"] = 1
            mesh_feedback_context = {
                "mesh_state": kernel_mesh_state,
                "feedback_stats": {
                    "changed_pixels": 0,
                    "feedback_pixels": 0,
                    "strength": 0.0,
                    "transport_pixels": int(kernel_transport_stats["transport_pixels"]),
                    "transport_max_displacement": float(kernel_transport_stats["max_displacement"]),
                    "transport_mean_displacement": float(kernel_transport_stats["mean_displacement"]),
                },
            }
        if runtime_topology_tracker is not None:
            runtime_topology_snapshot = runtime_topology_tracker.update(phi, step)
            if mesh_feedback_context is None:
                mesh_feedback_context = {}
            mesh_feedback_context["tracked_topology_snapshot"] = runtime_topology_snapshot
        if step % save_every == 0 or step == steps:
            snapshots.append(phi)
            if on_snapshot is not None:
                on_snapshot(step, phi, mesh_feedback_context)

    return phi, snapshots
