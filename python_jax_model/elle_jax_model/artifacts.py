from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np


def _as_numpy(phi) -> np.ndarray:
    phi_np = np.asarray(phi, dtype=np.float32)
    if phi_np.ndim != 3:
        raise ValueError("expected phi to have shape (num_grains, nx, ny)")
    return phi_np


def dominant_grain_map(phi) -> np.ndarray:
    phi_np = _as_numpy(phi)
    return np.argmax(phi_np, axis=0).astype(np.int32)


def boundary_mask(labels) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int32)
    if labels_np.ndim != 2:
        raise ValueError("expected labels to have shape (nx, ny)")
    return (
        (labels_np != np.roll(labels_np, +1, axis=0))
        | (labels_np != np.roll(labels_np, -1, axis=0))
        | (labels_np != np.roll(labels_np, +1, axis=1))
        | (labels_np != np.roll(labels_np, -1, axis=1))
    )


def snapshot_statistics(phi, step: int | None = None) -> dict[str, Any]:
    phi_np = _as_numpy(phi)
    labels = dominant_grain_map(phi_np)
    counts = np.bincount(labels.ravel(), minlength=phi_np.shape[0]).astype(np.int64)
    area_fractions = counts / labels.size
    boundary = boundary_mask(labels)
    sums = phi_np.sum(axis=0)

    grains: list[dict[str, Any]] = []
    for grain_id in range(phi_np.shape[0]):
        field = phi_np[grain_id]
        grains.append(
            {
                "grain_id": grain_id,
                "pixel_count": int(counts[grain_id]),
                "area_fraction": float(area_fractions[grain_id]),
                "mean_order_parameter": float(field.mean()),
                "max_order_parameter": float(field.max()),
            }
        )

    return {
        "step": step,
        "num_grains": int(phi_np.shape[0]),
        "grid_shape": [int(phi_np.shape[1]), int(phi_np.shape[2])],
        "value_range": [float(phi_np.min()), float(phi_np.max())],
        "sum_range": [float(sums.min()), float(sums.max())],
        "boundary_fraction": float(boundary.mean()),
        "active_grains": int((counts > 0).sum()),
        "grains": grains,
    }


def _label_palette(num_grains: int) -> np.ndarray:
    grain_ids = np.arange(num_grains, dtype=np.uint32)
    red = (grain_ids * 53 + 37) % 256
    green = (grain_ids * 97 + 17) % 256
    blue = (grain_ids * 193 + 71) % 256
    return np.stack([red, green, blue], axis=1).astype(np.uint8)


def preview_image(labels, boundary=None) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int32)
    if labels_np.ndim != 2:
        raise ValueError("expected labels to have shape (nx, ny)")

    palette = _label_palette(int(labels_np.max()) + 1)
    image = palette[labels_np].transpose(1, 0, 2)
    if boundary is not None:
        boundary_np = np.asarray(boundary, dtype=bool).transpose(1, 0)
        image = image.copy()
        image[boundary_np] = np.array([255, 255, 255], dtype=np.uint8)
    return image


def write_ppm(path: str | Path, image: np.ndarray) -> None:
    path = Path(path)
    image_np = np.asarray(image, dtype=np.uint8)
    if image_np.ndim != 3 or image_np.shape[2] != 3:
        raise ValueError("expected image to have shape (height, width, 3)")

    height, width, _ = image_np.shape
    with path.open("wb") as handle:
        handle.write(f"P6\n{width} {height}\n255\n".encode("ascii"))
        handle.write(np.ascontiguousarray(image_np).tobytes())


def save_snapshot_artifacts(
    outdir: str | Path,
    step: int,
    phi,
    *,
    save_order_parameter: bool = True,
    save_grain_ids: bool = True,
    save_boundary_mask: bool = True,
    save_stats: bool = True,
    save_preview: bool = True,
    save_elle: bool = False,
    tracked_topology: dict[str, Any] | None = None,
    save_topology: bool = False,
    mesh_state: dict[str, Any] | None = None,
    save_mesh: bool = False,
) -> dict[str, Path]:
    outdir_path = Path(outdir)
    outdir_path.mkdir(parents=True, exist_ok=True)

    phi_np = _as_numpy(phi)
    labels = dominant_grain_map(phi_np)
    boundary = boundary_mask(labels)
    written: dict[str, Path] = {}

    if save_order_parameter:
        order_path = outdir_path / f"order_parameter_{step:05d}.npy"
        np.save(order_path, phi_np)
        written["order_parameter"] = order_path

    if save_grain_ids:
        labels_path = outdir_path / f"grain_ids_{step:05d}.npy"
        np.save(labels_path, labels)
        written["grain_ids"] = labels_path

    if save_boundary_mask:
        boundary_path = outdir_path / f"boundary_mask_{step:05d}.npy"
        np.save(boundary_path, boundary.astype(np.uint8))
        written["boundary_mask"] = boundary_path

    if save_stats:
        stats_path = outdir_path / f"grain_stats_{step:05d}.json"
        with stats_path.open("w", encoding="utf-8") as handle:
            json.dump(snapshot_statistics(phi_np, step=step), handle, indent=2)
        written["grain_stats"] = stats_path

    if save_preview:
        preview_path = outdir_path / f"grain_preview_{step:05d}.ppm"
        write_ppm(preview_path, preview_image(labels, boundary=boundary))
        written["grain_preview"] = preview_path

    if save_topology and tracked_topology is not None:
        from .topology import write_topology_snapshot

        topology_path = outdir_path / f"topology_{step:05d}.json"
        write_topology_snapshot(topology_path, tracked_topology)
        written["topology"] = topology_path

    if save_mesh and mesh_state is not None:
        from .mesh import write_mesh_state

        mesh_path = outdir_path / f"mesh_{step:05d}.json"
        write_mesh_state(mesh_path, mesh_state)
        written["mesh"] = mesh_path

    if save_elle:
        from .elle_export import write_unode_elle

        elle_path = outdir_path / f"grain_unodes_{step:05d}.elle"
        write_unode_elle(
            elle_path,
            phi_np,
            step=step,
            tracked_topology=tracked_topology,
            mesh_state=mesh_state,
        )
        written["elle"] = elle_path

    return written
