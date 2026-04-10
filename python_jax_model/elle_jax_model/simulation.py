from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Tuple

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
        def uniform(key: int, shape: tuple[int, ...], minval: float, maxval: float):
            rng = jnp.random.default_rng(key)
            return rng.uniform(minval, maxval, size=shape)

    class _JaxCompat:
        random = _RandomCompat()

        @staticmethod
        def jit(fn):
            return fn

    jax = _JaxCompat()  # type: ignore


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


def initialize_order_parameters(config: GrainGrowthConfig):
    """Create normalized random order parameters shaped (num_grains, nx, ny)."""

    key = jax.random.PRNGKey(config.seed)
    raw = jax.random.uniform(
        key,
        shape=(config.num_grains, config.nx, config.ny),
        minval=0.0,
        maxval=1.0,
    )
    total = jnp.sum(raw, axis=0, keepdims=True) + 1e-12
    return raw / total


def _laplacian_periodic(field):
    return (
        jnp.roll(field, +1, axis=1)
        + jnp.roll(field, -1, axis=1)
        + jnp.roll(field, +1, axis=2)
        + jnp.roll(field, -1, axis=2)
        - 4.0 * field
    )


def _step(phi, config: GrainGrowthConfig):
    lap = _laplacian_periodic(phi)

    sum_phi_sq = jnp.sum(phi * phi, axis=0, keepdims=True)
    dfdphi = phi * (phi * phi - 1.0) + config.interaction_strength * phi * (
        sum_phi_sq - phi * phi
    )

    dphi_dt = config.mobility * (config.gradient_penalty * lap - dfdphi)
    phi_next = phi + config.dt * dphi_dt
    phi_next = jnp.clip(phi_next, 0.0, 1.0)

    norm = jnp.sum(phi_next, axis=0, keepdims=True) + 1e-12
    return phi_next / norm


def run_simulation(
    config: GrainGrowthConfig,
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object], None] | None = None,
) -> Tuple[object, List[object]]:
    """Run simulation and optionally report snapshots."""

    if steps < 1:
        raise ValueError("steps must be >= 1")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    phi = initialize_order_parameters(config)
    snapshots: List[object] = []
    step_fn = jax.jit(_step)

    for step in range(1, steps + 1):
        phi = step_fn(phi, config)
        if step % save_every == 0 or step == steps:
            snapshots.append(phi)
            if on_snapshot is not None:
                on_snapshot(step, phi)

    return phi, snapshots
