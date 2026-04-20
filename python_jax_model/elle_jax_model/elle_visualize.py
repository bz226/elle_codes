from __future__ import annotations

import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from .artifacts import preview_image, write_ppm


@dataclass(frozen=True)
class ShowelleSettings:
    attribute: str | None = None
    vmin: float | None = None
    vmax: float | None = None
    overlay_boundaries: bool = True
    flynn_labels: bool = False


def _parse_elle_sections(path: str | Path) -> dict[str, tuple[str, ...]]:
    header_pattern = re.compile(r"^(?P<header>[A-Z][A-Z0-9_]*)(?:\s+[A-Z][A-Z0-9_]*)*$")
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        match = header_pattern.fullmatch(stripped)
        if match is not None:
            current_section = str(match.group("header"))
            sections.setdefault(current_section, [])
            continue
        if current_section is not None:
            sections[current_section].append(raw_line)
    return {name: tuple(lines) for name, lines in sections.items()}


def _parse_showelle_settings(path: str | Path) -> ShowelleSettings:
    attribute = None
    vmin = None
    vmax = None
    flynn_labels = False
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = [part.strip() for part in raw_line.split("=", 1)]
        if key == "Unode_Attribute":
            if value.upper().startswith("NONE"):
                attribute = None
            else:
                tokens = value.split()
                if tokens:
                    attribute = tokens[0]
                min_match = re.search(r"min=\s*([^\s]+)", value)
                max_match = re.search(r"max=\s*([^\s]+)", value)
                vmin = float(min_match.group(1)) if min_match else None
                vmax = float(max_match.group(1)) if max_match else None
        elif key == "Flynn_Labels":
            flynn_labels = value.strip() != "0"
    return ShowelleSettings(
        attribute=attribute,
        vmin=vmin,
        vmax=vmax,
        flynn_labels=flynn_labels,
    )


def _normalize_attribute_name(attribute: str | None) -> str | None:
    if attribute is None:
        return None
    attr = attribute.upper()
    if attr.startswith("U_"):
        return attr
    if attr == "CONC_A":
        return "U_CONC_A"
    if attr.startswith("ATTRIB_"):
        return f"U_{attr}"
    return attr


def _parse_sparse_values(lines: tuple[str, ...]) -> tuple[float, dict[int, float]]:
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


def _parse_unodes(lines: tuple[str, ...]) -> tuple[tuple[int, float, float], ...]:
    unodes = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 3:
            unodes.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return tuple(unodes)


def _parse_location(lines: tuple[str, ...]) -> dict[int, tuple[float, float]]:
    nodes: dict[int, tuple[float, float]] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 3:
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return nodes


def _parse_flynns(lines: tuple[str, ...]) -> list[dict[str, Any]]:
    flynns = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            flynn_id = int(parts[0])
            count = int(parts[1])
            node_ids = [int(value) for value in parts[2 : 2 + count]]
            flynns.append({"flynn_id": flynn_id, "node_ids": node_ids})
    return flynns


def _layout_from_unodes(unodes: tuple[tuple[int, float, float], ...]) -> dict[str, Any]:
    x_values = sorted({round(x_coord, 10) for _, x_coord, _ in unodes})
    y_values = sorted({round(y_coord, 10) for _, _, y_coord in unodes})
    xmin = min(float(x_coord) for _, x_coord, _ in unodes)
    xmax = max(float(x_coord) for _, x_coord, _ in unodes)
    ymin = min(float(y_coord) for _, _, y_coord in unodes)
    ymax = max(float(y_coord) for _, _, y_coord in unodes)
    structured = len(x_values) * len(y_values) == len(unodes)

    if structured:
        x_lookup = {value: index for index, value in enumerate(x_values)}
        y_lookup = {value: index for index, value in enumerate(y_values)}
        id_lookup = {
            int(unode_id): (x_lookup[round(x_coord, 10)], y_lookup[round(y_coord, 10)])
            for unode_id, x_coord, y_coord in unodes
        }
        return {
            "structured": True,
            "grid_shape": (len(x_values), len(y_values)),
            "id_lookup": id_lookup,
            "x_range": (xmin, xmax if len(x_values) > 1 else xmin + 1.0),
            "y_range": (ymin, ymax if len(y_values) > 1 else ymin + 1.0),
        }

    xspan = xmax - xmin if xmax > xmin else 1.0
    yspan = ymax - ymin if ymax > ymin else 1.0
    aspect = xspan / yspan if yspan > 1.0e-12 else 1.0
    target_cells = max(1, len(unodes))
    width = max(1, int(round(math.sqrt(target_cells * max(aspect, 1.0e-6)))))
    height = max(1, int(math.ceil(target_cells / width)))
    width = min(width, 1024)
    height = min(height, 1024)

    def to_cell(x_coord: float, y_coord: float) -> tuple[int, int]:
        ix = int(round((float(x_coord) - xmin) / xspan * (width - 1))) if width > 1 else 0
        iy = int(round((float(y_coord) - ymin) / yspan * (height - 1))) if height > 1 else 0
        return max(0, min(width - 1, ix)), max(0, min(height - 1, iy))

    id_lookup = {
        int(unode_id): to_cell(x_coord, y_coord)
        for unode_id, x_coord, y_coord in unodes
    }
    return {
        "structured": False,
        "grid_shape": (width, height),
        "id_lookup": id_lookup,
        "x_range": (xmin, xmax if xmax > xmin else xmin + 1.0),
        "y_range": (ymin, ymax if ymax > ymin else ymin + 1.0),
    }


def _field_from_sparse_unodes(
    unodes: tuple[tuple[int, float, float], ...],
    lines: tuple[str, ...],
) -> tuple[np.ndarray, tuple[int, int], dict[str, Any]]:
    layout = _layout_from_unodes(unodes)
    width, height = layout["grid_shape"]
    id_lookup = layout["id_lookup"]
    default, values = _parse_sparse_values(lines)
    field = np.full((width, height), default, dtype=np.float32)

    if layout["structured"]:
        for unode_id, value in values.items():
            if unode_id in id_lookup:
                ix, iy = id_lookup[unode_id]
                field[ix, iy] = float(value)
        return field, (width, height), layout

    sums = np.zeros((width, height), dtype=np.float64)
    counts = np.zeros((width, height), dtype=np.int32)
    for unode_id, _, _ in unodes:
        ix, iy = id_lookup[int(unode_id)]
        sums[ix, iy] += float(values.get(int(unode_id), default))
        counts[ix, iy] += 1
    filled = counts > 0
    field[filled] = (sums[filled] / counts[filled]).astype(np.float32)
    return field, (width, height), layout


def _is_integer_label_field(field: np.ndarray) -> bool:
    rounded = np.round(field)
    return bool(
        np.max(np.abs(field - rounded)) < 1e-5
        and len(np.unique(rounded)) <= max(256, field.size // 8)
    )


def _grayscale_image(field: np.ndarray, *, vmin: float | None, vmax: float | None) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    low = float(np.min(data)) if vmin is None else float(vmin)
    high = float(np.max(data)) if vmax is None else float(vmax)
    if high - low < 1e-12:
        normalized = np.zeros_like(data)
    else:
        normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
    gray = np.round(255.0 * normalized).astype(np.uint8)
    return np.stack([gray, gray, gray], axis=2).transpose(1, 0, 2)


def _heat_image(field: np.ndarray, *, vmin: float | None, vmax: float | None) -> np.ndarray:
    data = np.asarray(field, dtype=np.float32)
    low = float(np.min(data)) if vmin is None else float(vmin)
    high = float(np.max(data)) if vmax is None else float(vmax)
    if high - low < 1e-12:
        normalized = np.zeros_like(data)
    else:
        normalized = np.clip((data - low) / (high - low), 0.0, 1.0)
    red = np.round(255.0 * normalized).astype(np.uint8)
    blue = np.round(255.0 * (1.0 - normalized)).astype(np.uint8)
    green = np.round(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
    return np.stack([red, green, blue], axis=2).transpose(1, 0, 2)


def _draw_line(image: np.ndarray, x0: int, y0: int, x1: int, y1: int, color: np.ndarray) -> None:
    steps = int(max(abs(x1 - x0), abs(y1 - y0))) + 1
    xs = np.rint(np.linspace(x0, x1, steps)).astype(np.int32)
    ys = np.rint(np.linspace(y0, y1, steps)).astype(np.int32)
    height, width = image.shape[0], image.shape[1]
    valid = (xs >= 0) & (xs < width) & (ys >= 0) & (ys < height)
    image[ys[valid], xs[valid]] = color


_FONT_5X3 = {
    "0": ("111", "101", "101", "101", "111"),
    "1": ("010", "110", "010", "010", "111"),
    "2": ("111", "001", "111", "100", "111"),
    "3": ("111", "001", "111", "001", "111"),
    "4": ("101", "101", "111", "001", "001"),
    "5": ("111", "100", "111", "001", "111"),
    "6": ("111", "100", "111", "101", "111"),
    "7": ("111", "001", "010", "100", "100"),
    "8": ("111", "101", "111", "101", "111"),
    "9": ("111", "101", "111", "001", "111"),
    "-": ("000", "000", "111", "000", "000"),
    "+": ("000", "010", "111", "010", "000"),
    ".": ("000", "000", "000", "000", "010"),
    "e": ("000", "111", "101", "110", "111"),
    "E": ("111", "100", "111", "100", "111"),
    " ": ("000", "000", "000", "000", "000"),
}


def _draw_rect(image: np.ndarray, x0: int, y0: int, width: int, height: int, color: np.ndarray) -> None:
    x1 = min(image.shape[1], max(0, x0 + width))
    y1 = min(image.shape[0], max(0, y0 + height))
    x0 = max(0, x0)
    y0 = max(0, y0)
    if x1 <= x0 or y1 <= y0:
        return
    image[y0:y1, x0:x1] = color


def _draw_char(
    image: np.ndarray,
    x0: int,
    y0: int,
    char: str,
    *,
    color: tuple[int, int, int],
    pixel_size: int,
) -> None:
    glyph = _FONT_5X3.get(char, _FONT_5X3[" "])
    color_np = np.asarray(color, dtype=np.uint8)
    for row_index, row in enumerate(glyph):
        for col_index, value in enumerate(row):
            if value != "1":
                continue
            _draw_rect(
                image,
                x0 + col_index * pixel_size,
                y0 + row_index * pixel_size,
                pixel_size,
                pixel_size,
                color_np,
            )


def _draw_text(
    image: np.ndarray,
    x0: int,
    y0: int,
    text: str,
    *,
    color: tuple[int, int, int],
    pixel_size: int = 1,
    shadow: bool = False,
) -> None:
    cursor = x0
    for char in str(text):
        if shadow:
            _draw_char(image, cursor + pixel_size, y0 + pixel_size, char, color=(0, 0, 0), pixel_size=pixel_size)
        _draw_char(image, cursor, y0, char, color=color, pixel_size=pixel_size)
        cursor += 4 * pixel_size


def _text_dimensions(text: str, *, pixel_size: int) -> tuple[int, int]:
    return (max(0, len(text) * 4 - 1) * pixel_size, 5 * pixel_size)


def _map_point_to_pixel(
    x_coord: float,
    y_coord: float,
    *,
    width: int,
    height: int,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> tuple[int, int]:
    xmin, xmax = x_range
    ymin, ymax = y_range
    xspan = xmax - xmin if xmax > xmin else 1.0
    yspan = ymax - ymin if ymax > ymin else 1.0
    px = int(round((x_coord - xmin) / xspan * (width - 1)))
    py = int(round((y_coord - ymin) / yspan * (height - 1)))
    return px, py


def _overlay_flynn_boundaries(
    image: np.ndarray,
    flynns: list[dict[str, Any]],
    nodes: dict[int, tuple[float, float]],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    color: tuple[int, int, int],
) -> np.ndarray:
    if not flynns or not nodes:
        return image

    result = image.copy()
    width = result.shape[1]
    height = result.shape[0]
    color_np = np.asarray(color, dtype=np.uint8)

    for flynn in flynns:
        node_ids = flynn["node_ids"]
        if len(node_ids) < 2:
            continue
        points = [nodes[node_id] for node_id in node_ids if node_id in nodes]
        if len(points) < 2:
            continue
        for index, (x0, y0) in enumerate(points):
            x1, y1 = points[(index + 1) % len(points)]
            px0, py0 = _map_point_to_pixel(
                x0,
                y0,
                width=width,
                height=height,
                x_range=x_range,
                y_range=y_range,
            )
            px1, py1 = _map_point_to_pixel(
                x1,
                y1,
                width=width,
                height=height,
                x_range=x_range,
                y_range=y_range,
            )
            _draw_line(result, px0, py0, px1, py1, color_np)
    return result


def _overlay_flynn_labels(
    image: np.ndarray,
    flynns: list[dict[str, Any]],
    nodes: dict[int, tuple[float, float]],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    pixel_size: int,
) -> np.ndarray:
    if not flynns or not nodes:
        return image

    result = image.copy()
    height, width = result.shape[0], result.shape[1]
    for flynn in flynns:
        points = [nodes[node_id] for node_id in flynn["node_ids"] if node_id in nodes]
        if len(points) < 2:
            continue
        xs = [point[0] for point in points]
        ys = [point[1] for point in points]
        px, py = _map_point_to_pixel(
            float(np.mean(xs)),
            float(np.mean(ys)),
            width=width,
            height=height,
            x_range=x_range,
            y_range=y_range,
        )
        label = str(flynn["flynn_id"])
        text_width, text_height = _text_dimensions(label, pixel_size=pixel_size)
        text_x = min(max(0, px - text_width // 2), max(0, width - text_width))
        text_y = min(max(0, py - text_height // 2), max(0, height - text_height))
        _draw_text(
            result,
            text_x,
            text_y,
            label,
            color=(255, 255, 255),
            pixel_size=pixel_size,
            shadow=True,
        )
    return result


def _scale_image(image: np.ndarray, scale: int) -> np.ndarray:
    if scale <= 1:
        return image
    return np.repeat(np.repeat(image, scale, axis=0), scale, axis=1)


def _format_legend_value(value: float) -> str:
    if abs(value) < 1.0e-12:
        return "0"
    abs_value = abs(value)
    if 1.0e-3 <= abs_value < 1.0e3:
        return f"{value:.3f}".rstrip("0").rstrip(".")
    return f"{value:.2e}"


def _append_scalar_legend(
    image: np.ndarray,
    *,
    palette: str,
    vmin: float,
    vmax: float,
    text_scale: int,
) -> np.ndarray:
    height = image.shape[0]
    bar_width = max(10, 6 * text_scale)
    margin = max(4, 3 * text_scale)
    label_width = max(40, 22 * text_scale)
    legend_width = bar_width + margin * 3 + label_width
    legend = np.full((height, legend_width, 3), 255, dtype=np.uint8)

    normalized = np.linspace(1.0, 0.0, height, dtype=np.float32)
    if palette == "heat":
        red = np.round(255.0 * normalized).astype(np.uint8)
        blue = np.round(255.0 * (1.0 - normalized)).astype(np.uint8)
        green = np.round(255.0 * (1.0 - np.abs(2.0 * normalized - 1.0))).astype(np.uint8)
        colors = np.stack([red, green, blue], axis=1)
    else:
        gray = np.round(255.0 * normalized).astype(np.uint8)
        colors = np.stack([gray, gray, gray], axis=1)

    x0 = margin
    x1 = x0 + bar_width
    legend[:, x0:x1] = colors[:, np.newaxis, :]
    _draw_rect(legend, x0 - 1, 0, 1, height, np.array([0, 0, 0], dtype=np.uint8))
    _draw_rect(legend, x1, 0, 1, height, np.array([0, 0, 0], dtype=np.uint8))
    _draw_rect(legend, x0 - 1, 0, bar_width + 2, 1, np.array([0, 0, 0], dtype=np.uint8))
    _draw_rect(legend, x0 - 1, height - 1, bar_width + 2, 1, np.array([0, 0, 0], dtype=np.uint8))

    top_label = _format_legend_value(vmax)
    bottom_label = _format_legend_value(vmin)
    text_x = x1 + margin
    _draw_text(legend, text_x, margin, top_label, color=(0, 0, 0), pixel_size=text_scale)
    bottom_height = _text_dimensions(bottom_label, pixel_size=text_scale)[1]
    _draw_text(
        legend,
        text_x,
        max(margin, height - margin - bottom_height),
        bottom_label,
        color=(0, 0, 0),
        pixel_size=text_scale,
    )
    return np.concatenate([image, legend], axis=1)


def render_elle_file(
    elle_path: str | Path,
    *,
    outpath: str | Path | None = None,
    attribute: str = "auto",
    palette: str = "auto",
    showelle_in: str | Path | None = None,
    overlay_boundaries: bool = True,
    scale: int = 1,
    legend: bool = False,
    label_flynns: bool | None = None,
) -> dict[str, Any]:
    sections = _parse_elle_sections(elle_path)
    unodes = _parse_unodes(sections.get("UNODES", ()))
    nodes = _parse_location(sections.get("LOCATION", ()))
    flynns = _parse_flynns(sections.get("FLYNNS", ()))
    if not unodes:
        raise ValueError("ELLE file does not contain a UNODES section")
    if scale < 1:
        raise ValueError("scale must be at least 1")

    settings = _parse_showelle_settings(showelle_in) if showelle_in is not None else ShowelleSettings()
    chosen_attribute = attribute if attribute != "auto" else (settings.attribute or "auto")
    section_name = _normalize_attribute_name(chosen_attribute)
    chosen_label_flynns = settings.flynn_labels if label_flynns is None else bool(label_flynns)
    available_sections = [name for name in ("U_CONC_A", "U_ATTRIB_A", "U_ATTRIB_B") if name in sections]
    if section_name in {None, "AUTO"}:
        if "U_CONC_A" in sections:
            section_name = "U_CONC_A"
        elif available_sections:
            section_name = available_sections[0]
        else:
            raise ValueError("no supported unode attribute section found")

    field, grid_shape, layout = _field_from_sparse_unodes(unodes, sections[section_name])
    is_label = _is_integer_label_field(field) and section_name != "U_CONC_A"
    chosen_palette = palette
    if chosen_palette == "auto":
        if is_label:
            chosen_palette = "labels"
        elif section_name == "U_ATTRIB_A" and (float(field.min()) < 0.0 or float(field.max()) > 1.0):
            chosen_palette = "heat"
        else:
            chosen_palette = "gray"

    if chosen_palette == "labels":
        labels = np.rint(field).astype(np.int32)
        image = preview_image(labels)
        boundary_color = (255, 255, 255)
    elif chosen_palette == "heat":
        image = _heat_image(field, vmin=settings.vmin, vmax=settings.vmax)
        boundary_color = (255, 255, 255)
    else:
        image = _grayscale_image(field, vmin=settings.vmin, vmax=settings.vmax)
        boundary_color = (255, 0, 0)

    x_range = layout["x_range"]
    y_range = layout["y_range"]

    if overlay_boundaries and settings.overlay_boundaries:
        image = _overlay_flynn_boundaries(
            image,
            flynns,
            nodes,
            x_range=x_range,
            y_range=y_range,
            color=boundary_color,
        )

    image = _scale_image(image, scale)

    if chosen_label_flynns and flynns:
        image = _overlay_flynn_labels(
            image,
            flynns,
            nodes,
            x_range=x_range,
            y_range=y_range,
            pixel_size=max(1, min(3, scale)),
        )

    legend_applied = bool(legend and chosen_palette in {"gray", "heat"})
    if legend_applied:
        low = float(np.min(field)) if settings.vmin is None else float(settings.vmin)
        high = float(np.max(field)) if settings.vmax is None else float(settings.vmax)
        image = _append_scalar_legend(
            image,
            palette=chosen_palette,
            vmin=low,
            vmax=high,
            text_scale=max(1, min(3, scale)),
        )

    outpath_obj = Path(outpath) if outpath is not None else Path(elle_path).with_name(
        f"{Path(elle_path).stem}_preview.ppm"
    )
    outpath_obj.parent.mkdir(parents=True, exist_ok=True)
    write_ppm(outpath_obj, image)
    return {
        "elle_path": str(Path(elle_path)),
        "outpath": str(outpath_obj),
        "attribute": section_name,
        "palette": chosen_palette,
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "overlay_boundaries": bool(overlay_boundaries and settings.overlay_boundaries and bool(flynns)),
        "flynn_labels": bool(chosen_label_flynns and bool(flynns)),
        "legend": legend_applied,
        "scale": int(scale),
        "image_shape": [int(image.shape[0]), int(image.shape[1])],
        "num_flynns": int(len(flynns)),
    }
