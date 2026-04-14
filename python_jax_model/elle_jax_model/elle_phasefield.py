from __future__ import annotations

import json
import re
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

    class _JaxCompat:
        @staticmethod
        def jit(fn):
            return fn

    jax = _JaxCompat()  # type: ignore

from .artifacts import write_ppm


@dataclass(frozen=True)
class EllePhaseFieldTemplate:
    """Parsed ELLE template content for round-tripping unode phasefield files."""

    prefix_lines: tuple[str, ...]
    section_order: tuple[str, ...]
    sections: dict[str, tuple[str, ...]]
    unodes: tuple[tuple[int, float, float], ...]


@dataclass(frozen=True)
class EllePhaseFieldConfig:
    """Direct port parameters for ELLE's single-order-parameter phasefield process."""

    nx: int = 300
    ny: int = 300
    latent_heat: float = 1.8
    tau: float = 0.0003
    eps: float = 0.01
    delta: float = 0.02
    angle0: float = 1.57
    aniso: float = 6.0
    alpha: float = 0.9
    gamma: float = 10.0
    teq: float = 1.0
    spatial_step: float = 0.03
    dt: float = 2.0e-4
    initial_radius_sq: float = 10.0
    initial_theta_inside: float = 1.0
    initial_theta_outside: float = 0.0
    initial_temperature: float = 0.0


def _as_numpy_2d(field) -> np.ndarray:
    field_np = np.asarray(field, dtype=np.float32)
    if field_np.ndim != 2:
        raise ValueError("expected a 2D field")
    return field_np


def _section_lines(lines: tuple[str, ...]) -> list[str]:
    return [line.rstrip("\n") for line in lines]


def _parse_elle_sections(path: str | Path) -> EllePhaseFieldTemplate:
    path = Path(path)
    prefix_lines: list[str] = []
    section_order: list[str] = []
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    header_pattern = re.compile(r"[A-Z][A-Z0-9_]*")

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if header_pattern.fullmatch(stripped):
            current_section = stripped
            section_order.append(stripped)
            sections.setdefault(stripped, [])
            continue

        if current_section is None:
            prefix_lines.append(raw_line)
        else:
            sections[current_section].append(raw_line)

    unodes = []
    for line in sections.get("UNODES", []):
        stripped = line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) < 3:
            continue
        unodes.append((int(parts[0]), float(parts[1]), float(parts[2])))

    return EllePhaseFieldTemplate(
        prefix_lines=tuple(prefix_lines),
        section_order=tuple(section_order),
        sections={name: tuple(lines) for name, lines in sections.items()},
        unodes=tuple(unodes),
    )


def _sorted_unique_coords(unodes: tuple[tuple[int, float, float], ...]) -> tuple[list[float], list[float]]:
    x_values = sorted({round(float(x_coord), 10) for _, x_coord, _ in unodes})
    y_values = sorted({round(float(y_coord), 10) for _, _, y_coord in unodes})
    return x_values, y_values


def _parse_sparse_unode_section(lines: tuple[str, ...]) -> tuple[float, dict[int, float]]:
    default = 0.0
    values: dict[int, float] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if parts[0] == "Default" and len(parts) > 1:
            default = float(parts[1])
        elif len(parts) >= 2:
            values[int(parts[0])] = float(parts[1])
    return default, values


def load_elle_phasefield_state(
    path: str | Path,
) -> tuple[np.ndarray, np.ndarray, EllePhaseFieldTemplate]:
    template = _parse_elle_sections(path)
    if not template.unodes:
        raise ValueError("ELLE file does not contain a UNODES section")

    x_values, y_values = _sorted_unique_coords(template.unodes)
    if not x_values or not y_values:
        raise ValueError("ELLE UNODES section is empty")

    nx, ny = len(x_values), len(y_values)
    x_lookup = {round(value, 10): index for index, value in enumerate(x_values)}
    y_lookup = {round(value, 10): index for index, value in enumerate(y_values)}
    id_to_index: dict[int, tuple[int, int]] = {}
    for unode_id, x_coord, y_coord in template.unodes:
        id_to_index[int(unode_id)] = (
            x_lookup[round(float(x_coord), 10)],
            y_lookup[round(float(y_coord), 10)],
        )

    theta_default, theta_values = _parse_sparse_unode_section(template.sections.get("U_CONC_A", ()))
    temperature_default, temperature_values = _parse_sparse_unode_section(
        template.sections.get("U_ATTRIB_A", ())
    )

    theta = np.full((nx, ny), theta_default, dtype=np.float32)
    temperature = np.full((nx, ny), temperature_default, dtype=np.float32)

    for unode_id, value in theta_values.items():
        ix, iy = id_to_index[unode_id]
        theta[ix, iy] = float(value)

    for unode_id, value in temperature_values.items():
        ix, iy = id_to_index[unode_id]
        temperature[ix, iy] = float(value)

    return theta, temperature, template


def initialize_elle_phasefield(config: EllePhaseFieldConfig) -> tuple[object, object]:
    if config.nx < 1 or config.ny < 1:
        raise ValueError("nx and ny must be >= 1")

    x = jnp.arange(config.nx, dtype=jnp.float32)[:, None]
    y = jnp.arange(config.ny, dtype=jnp.float32)[None, :]
    cx = 0.5 * float(config.nx)
    cy = 0.5 * float(config.ny)
    inside = (x - cx) * (x - cx) + (y - cy) * (y - cy) < float(config.initial_radius_sq)
    theta = jnp.where(
        inside,
        jnp.asarray(float(config.initial_theta_inside), dtype=jnp.float32),
        jnp.asarray(float(config.initial_theta_outside), dtype=jnp.float32),
    )
    temperature = jnp.full(
        (config.nx, config.ny),
        fill_value=float(config.initial_temperature),
        dtype=jnp.float32,
    )
    return theta, temperature


def _angle_field(x_field, y_field):
    angle = jnp.arctan2(y_field, x_field)
    two_pi = jnp.asarray(2.0 * np.pi, dtype=angle.dtype)
    return jnp.mod(angle + two_pi, two_pi)


def _nine_point_laplacian(field, h: float):
    xp = jnp.roll(field, -1, axis=0)
    xm = jnp.roll(field, +1, axis=0)
    yp = jnp.roll(field, -1, axis=1)
    ym = jnp.roll(field, +1, axis=1)
    xyp = jnp.roll(xp, -1, axis=1)
    xym = jnp.roll(xp, +1, axis=1)
    xmy = jnp.roll(xm, -1, axis=1)
    xmm = jnp.roll(xm, +1, axis=1)
    return (
        2.0 * (xp + xm + yp + ym) + xyp + xym + xmy + xmm - 12.0 * field
    ) / (3.0 * h * h)


def _original_centered_difference(field, h: float, axis: int):
    return (jnp.roll(field, -1, axis=axis) - jnp.roll(field, +1, axis=axis)) / h


def _step_elle_phasefield(theta, temperature, config: EllePhaseFieldConfig):
    h = float(config.spatial_step)
    grad_theta_x = _original_centered_difference(theta, h, axis=0)
    grad_theta_y = _original_centered_difference(theta, h, axis=1)
    lap_theta = _nine_point_laplacian(theta, h)
    lap_temperature = _nine_point_laplacian(temperature, h)

    angle = _angle_field(grad_theta_x, grad_theta_y)
    epsilon = float(config.eps) * (
        1.0 + float(config.delta) * jnp.cos(float(config.aniso) * (angle - float(config.angle0)))
    )
    epsilon_prime = (
        -float(config.eps)
        * float(config.aniso)
        * float(config.delta)
        * jnp.sin(float(config.aniso) * (angle - float(config.angle0)))
    )

    ay = -epsilon * epsilon_prime * grad_theta_y
    ax = epsilon * epsilon_prime * grad_theta_x
    eps2 = epsilon * epsilon

    d_ay_dx = _original_centered_difference(ay, h, axis=0)
    d_ax_dy = _original_centered_difference(ax, h, axis=1)
    grad_eps2_x = _original_centered_difference(eps2, h, axis=0)
    grad_eps2_y = _original_centered_difference(eps2, h, axis=1)

    m_term = float(config.alpha) / np.pi * jnp.arctan(
        float(config.gamma) * (float(config.teq) - temperature)
    )
    reaction = theta * (1.0 - theta) * (theta - 0.5 + m_term)
    scalar_term = grad_eps2_x * grad_theta_x + grad_eps2_y * grad_theta_y

    theta_next = theta + (
        d_ay_dx + d_ax_dy + eps2 * lap_theta + scalar_term + reaction
    ) * float(config.dt) / float(config.tau)
    temperature_next = temperature + lap_temperature * float(config.dt) + float(config.latent_heat) * (
        theta_next - theta
    )
    theta_next = jnp.clip(theta_next, 0.0, 1.0)
    return theta_next, temperature_next


def run_elle_phasefield_simulation(
    config: EllePhaseFieldConfig,
    steps: int,
    save_every: int,
    on_snapshot: Callable[[int, object, object], None] | None = None,
    initial_state: tuple[object, object] | None = None,
) -> tuple[object, object, List[tuple[object, object]]]:
    if steps < 1:
        raise ValueError("steps must be >= 1")
    if save_every < 1:
        raise ValueError("save_every must be >= 1")

    if initial_state is None:
        theta, temperature = initialize_elle_phasefield(config)
    else:
        theta = jnp.asarray(initial_state[0], dtype=jnp.float32)
        temperature = jnp.asarray(initial_state[1], dtype=jnp.float32)
        if theta.shape != (config.nx, config.ny) or temperature.shape != (config.nx, config.ny):
            raise ValueError("initial_state shape does not match config grid")
    snapshots: List[tuple[object, object]] = []
    step_fn = jax.jit(
        lambda current_theta, current_temperature: _step_elle_phasefield(
            current_theta,
            current_temperature,
            config,
        )
    )

    for step in range(1, steps + 1):
        theta, temperature = step_fn(theta, temperature)
        if step % save_every == 0 or step == steps:
            snapshots.append((theta, temperature))
            if on_snapshot is not None:
                on_snapshot(step, theta, temperature)

    return theta, temperature, snapshots


def phasefield_statistics(theta, temperature, step: int | None = None) -> dict[str, float | int | list[float]]:
    theta_np = _as_numpy_2d(theta)
    temperature_np = _as_numpy_2d(temperature)
    interface_mask = (theta_np > 0.05) & (theta_np < 0.95)
    return {
        "step": step,
        "grid_shape": [int(theta_np.shape[0]), int(theta_np.shape[1])],
        "theta_range": [float(theta_np.min()), float(theta_np.max())],
        "temperature_range": [float(temperature_np.min()), float(temperature_np.max())],
        "solid_fraction": float((theta_np >= 0.5).mean()),
        "interface_fraction": float(interface_mask.mean()),
        "mean_theta": float(theta_np.mean()),
        "mean_temperature": float(temperature_np.mean()),
    }


def _grayscale_image(field: np.ndarray) -> np.ndarray:
    scaled = np.clip(np.asarray(field, dtype=np.float32), 0.0, 1.0)
    gray = np.round(255.0 * scaled).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2).transpose(1, 0, 2)


def _temperature_image(field: np.ndarray) -> np.ndarray:
    temp = np.asarray(field, dtype=np.float32)
    if float(temp.max()) - float(temp.min()) < 1e-12:
        normalized = np.zeros_like(temp)
    else:
        normalized = (temp - temp.min()) / (temp.max() - temp.min())
    red = np.round(255.0 * normalized).astype(np.uint8)
    blue = np.round(255.0 * (1.0 - normalized)).astype(np.uint8)
    green = np.round(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
    return np.stack([red, green, blue], axis=2).transpose(1, 0, 2)


def _default_phasefield_unodes(nx: int, ny: int) -> tuple[tuple[int, float, float], ...]:
    unodes: list[tuple[int, float, float]] = []
    unode_id = 0
    for iy in range(ny):
        y_coord = iy / float(ny)
        for ix in range(nx):
            x_coord = ix / float(nx)
            unodes.append((unode_id, x_coord, y_coord))
            unode_id += 1
    return tuple(unodes)


def _write_sparse_field_section(
    handle,
    section_name: str,
    field: np.ndarray,
    unodes: tuple[tuple[int, float, float], ...],
    *,
    default: float = 0.0,
    tol: float = 1e-12,
) -> None:
    x_values, y_values = _sorted_unique_coords(unodes)
    x_lookup = {round(value, 10): index for index, value in enumerate(x_values)}
    y_lookup = {round(value, 10): index for index, value in enumerate(y_values)}
    handle.write(f"{section_name}\n")
    handle.write(f"Default {float(default):.8e}\n")
    for unode_id, x_coord, y_coord in unodes:
        ix = x_lookup[round(float(x_coord), 10)]
        iy = y_lookup[round(float(y_coord), 10)]
        value = float(field[ix, iy])
        if abs(value - default) > tol:
            handle.write(f"{int(unode_id)} {value:.8e}\n")


def write_elle_phasefield_state(
    path: str | Path,
    theta,
    temperature,
    *,
    step: int | None = None,
    template: EllePhaseFieldTemplate | None = None,
) -> Path:
    theta_np = _as_numpy_2d(theta)
    temperature_np = _as_numpy_2d(temperature)
    if theta_np.shape != temperature_np.shape:
        raise ValueError("theta and temperature must have the same shape")

    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    nx, ny = theta_np.shape
    unodes = template.unodes if template is not None and template.unodes else _default_phasefield_unodes(nx, ny)
    if template is not None:
        template_nx, template_ny = len(_sorted_unique_coords(unodes)[0]), len(_sorted_unique_coords(unodes)[1])
        if (template_nx, template_ny) != (nx, ny):
            raise ValueError("template UNODES grid does not match theta/temperature shape")

    with path.open("w", encoding="utf-8") as handle:
        if template is not None:
            prefix_lines = list(template.prefix_lines)
            if prefix_lines:
                for line in prefix_lines:
                    handle.write(f"{line}\n")
            else:
                handle.write("# Created by python_jax_model ELLE phasefield export\n")
        else:
            handle.write("# Created by python_jax_model ELLE phasefield export\n")
            handle.write("OPTIONS\n")
            handle.write("SwitchDistance 2.50000000e-02\n")
            handle.write("MaxNodeSeparation 5.50000000e-02\n")
            handle.write("MinNodeSeparation 2.50000000e-02\n")
            handle.write("SpeedUp 1.00000000e+03\n")
            handle.write("CellBoundingBox 0.00000000e+00 0.00000000e+00\n")
            handle.write("                1.00000000e+00 0.00000000e+00 \n")
            handle.write("                1.00000000e+00 1.00000000e+00 \n")
            handle.write("                0.00000000e+00 1.00000000e+00 \n")
            handle.write("SimpleShearOffset 0.00000000e+00\n")
            handle.write("CumulativeSimpleShear 0.00000000e+00\n")
            handle.write("Timestep 3.15000000e+07\n")
            handle.write("UnitLength 1.00000000e-02\n")
            handle.write("Temperature 2.50000000e+01\n")
            handle.write("Pressure 1.00000000e+00\n")
            handle.write("BoundaryWidth 1.00000000e-09\n")
            handle.write("MassIncrement 0.00000000e+00\n")

        if step is not None:
            handle.write(f"# step={step}\n")

        section_order = list(template.section_order) if template is not None else []
        missing_sections = ["UNODES", "U_CONC_A", "U_ATTRIB_A"]
        for section_name in missing_sections:
            if section_name not in section_order:
                section_order.append(section_name)

        for section_name in section_order:
            if section_name == "UNODES":
                handle.write("UNODES\n")
                for unode_id, x_coord, y_coord in unodes:
                    handle.write(f"{int(unode_id)} {float(x_coord):.10f} {float(y_coord):.10f}\n")
            elif section_name == "U_CONC_A":
                _write_sparse_field_section(handle, "U_CONC_A", theta_np, unodes, default=0.0)
            elif section_name == "U_ATTRIB_A":
                _write_sparse_field_section(handle, "U_ATTRIB_A", temperature_np, unodes, default=0.0)
            else:
                if template is None:
                    continue
                handle.write(f"{section_name}\n")
                for line in template.sections.get(section_name, ()):
                    handle.write(f"{line}\n")

    return path


def save_elle_phasefield_artifacts(
    outdir: str | Path,
    step: int,
    theta,
    temperature,
    *,
    save_preview: bool = True,
    save_elle: bool = False,
    elle_template: EllePhaseFieldTemplate | None = None,
) -> dict[str, Path]:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    theta_np = _as_numpy_2d(theta)
    temperature_np = _as_numpy_2d(temperature)
    written: dict[str, Path] = {}

    theta_path = outdir_path / f"theta_{step:05d}.npy"
    temperature_path = outdir_path / f"temperature_{step:05d}.npy"
    np.save(theta_path, theta_np)
    np.save(temperature_path, temperature_np)
    written["theta"] = theta_path
    written["temperature"] = temperature_path

    stats_path = outdir_path / f"phasefield_stats_{step:05d}.json"
    with stats_path.open("w", encoding="utf-8") as handle:
        json.dump(phasefield_statistics(theta_np, temperature_np, step=step), handle, indent=2)
    written["stats"] = stats_path

    if save_preview:
        theta_preview_path = outdir_path / f"theta_preview_{step:05d}.ppm"
        temperature_preview_path = outdir_path / f"temperature_preview_{step:05d}.ppm"
        write_ppm(theta_preview_path, _grayscale_image(theta_np))
        write_ppm(temperature_preview_path, _temperature_image(temperature_np))
        written["theta_preview"] = theta_preview_path
        written["temperature_preview"] = temperature_preview_path

    if save_elle:
        elle_path = outdir_path / f"phasefield_state_{step:05d}.elle"
        write_elle_phasefield_state(
            elle_path,
            theta_np,
            temperature_np,
            step=step,
            template=elle_template,
        )
        written["elle"] = elle_path

    return written
