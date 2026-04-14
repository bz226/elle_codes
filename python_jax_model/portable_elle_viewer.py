#!/usr/bin/env python3
from __future__ import annotations

import argparse
import base64
import hashlib
import html
import json
import math
import re
from pathlib import Path
from typing import Any


HEADER_PATTERN = re.compile(r"[A-Z][A-Z0-9_]*")


def _parse_elle_sections(path: str | Path) -> dict[str, list[str]]:
    sections: dict[str, list[str]] = {}
    current_section: str | None = None
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = raw_line.strip()
        if HEADER_PATTERN.fullmatch(stripped):
            current_section = stripped
            sections.setdefault(stripped, [])
            continue
        if current_section is not None:
            sections[current_section].append(raw_line)
    return sections


def _parse_showelle_settings(path: str | Path | None) -> dict[str, Any]:
    settings: dict[str, Any] = {
        "attribute": None,
        "vmin": None,
        "vmax": None,
        "flynn_labels": False,
    }
    if path is None:
        return settings
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        if "=" not in raw_line:
            continue
        key, value = [part.strip() for part in raw_line.split("=", 1)]
        if key == "Unode_Attribute":
            if value.upper().startswith("NONE"):
                settings["attribute"] = None
            else:
                tokens = value.split()
                if tokens:
                    settings["attribute"] = tokens[0]
                min_match = re.search(r"min=\s*([^\s]+)", value)
                max_match = re.search(r"max=\s*([^\s]+)", value)
                settings["vmin"] = float(min_match.group(1)) if min_match else None
                settings["vmax"] = float(max_match.group(1)) if max_match else None
        elif key == "Flynn_Labels":
            settings["flynn_labels"] = value.strip() != "0"
    return settings


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


def _parse_unodes(lines: list[str]) -> list[tuple[int, float, float]]:
    unodes: list[tuple[int, float, float]] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 3:
            unodes.append((int(parts[0]), float(parts[1]), float(parts[2])))
    return unodes


def _parse_sparse_values(lines: list[str]) -> tuple[float, dict[int, float]]:
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


def _parse_location(lines: list[str]) -> dict[int, tuple[float, float]]:
    nodes: dict[int, tuple[float, float]] = {}
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 3:
            nodes[int(parts[0])] = (float(parts[1]), float(parts[2]))
    return nodes


def _parse_flynns(lines: list[str]) -> list[dict[str, Any]]:
    flynns: list[dict[str, Any]] = []
    for raw_line in lines:
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if len(parts) >= 2:
            count = int(parts[1])
            node_ids = [int(value) for value in parts[2 : 2 + count]]
            flynns.append({"flynn_id": int(parts[0]), "node_ids": node_ids})
    return flynns


def _parse_cell_bounding_box(lines: list[str]) -> list[tuple[float, float]] | None:
    for index, raw_line in enumerate(lines):
        stripped = raw_line.strip()
        if not stripped or stripped.startswith("#"):
            continue
        parts = stripped.split()
        if not parts or parts[0] != "CellBoundingBox" or len(parts) < 3:
            continue
        corners: list[tuple[float, float]] = [(float(parts[1]), float(parts[2]))]
        for extra_line in lines[index + 1 : index + 4]:
            extra_parts = extra_line.strip().split()
            if len(extra_parts) >= 2:
                corners.append((float(extra_parts[0]), float(extra_parts[1])))
        if len(corners) >= 4:
            return corners[:4]
    return None


def _make_cell_transform(
    *,
    cell_bbox: list[tuple[float, float]] | None,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
) -> dict[str, Any]:
    if cell_bbox is not None and len(cell_bbox) >= 4:
        origin = cell_bbox[0]
        axis_u = (cell_bbox[1][0] - origin[0], cell_bbox[1][1] - origin[1])
        axis_v = (cell_bbox[3][0] - origin[0], cell_bbox[3][1] - origin[1])
        determinant = axis_u[0] * axis_v[1] - axis_u[1] * axis_v[0]
        if abs(determinant) > 1.0e-12:
            return {
                "origin": origin,
                "axis_u": axis_u,
                "axis_v": axis_v,
                "determinant": determinant,
                "periodic": True,
            }

    origin = (x_range[0], y_range[0])
    axis_u = (max(1.0e-12, x_range[1] - x_range[0]), 0.0)
    axis_v = (0.0, max(1.0e-12, y_range[1] - y_range[0]))
    determinant = axis_u[0] * axis_v[1]
    return {
        "origin": origin,
        "axis_u": axis_u,
        "axis_v": axis_v,
        "determinant": determinant,
        "periodic": False,
    }


def _point_to_cell(point: tuple[float, float], transform: dict[str, Any]) -> tuple[float, float]:
    dx = float(point[0]) - float(transform["origin"][0])
    dy = float(point[1]) - float(transform["origin"][1])
    determinant = float(transform["determinant"])
    axis_u = transform["axis_u"]
    axis_v = transform["axis_v"]
    u_coord = (dx * float(axis_v[1]) - dy * float(axis_v[0])) / determinant
    v_coord = (-dx * float(axis_u[1]) + dy * float(axis_u[0])) / determinant
    return (u_coord, v_coord)


def _cell_to_point(coords: tuple[float, float], transform: dict[str, Any]) -> tuple[float, float]:
    u_coord, v_coord = coords
    axis_u = transform["axis_u"]
    axis_v = transform["axis_v"]
    origin = transform["origin"]
    return (
        float(origin[0]) + float(u_coord) * float(axis_u[0]) + float(v_coord) * float(axis_v[0]),
        float(origin[1]) + float(u_coord) * float(axis_u[1]) + float(v_coord) * float(axis_v[1]),
    )


def _unwrap_cell_points(
    points: list[tuple[float, float]],
    *,
    periodic: bool,
) -> list[tuple[float, float]]:
    if not points:
        return []
    unwrapped = [points[0]]
    for point in points[1:]:
        prev_u, prev_v = unwrapped[-1]
        delta_u = float(point[0]) - prev_u
        delta_v = float(point[1]) - prev_v
        if periodic:
            delta_u -= round(delta_u)
            delta_v -= round(delta_v)
        unwrapped.append((prev_u + delta_u, prev_v + delta_v))
    return unwrapped


def _clip_segment_to_unit_square(
    start: tuple[float, float],
    end: tuple[float, float],
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    x0, y0 = float(start[0]), float(start[1])
    x1, y1 = float(end[0]), float(end[1])
    dx = x1 - x0
    dy = y1 - y0
    t0 = 0.0
    t1 = 1.0

    for p_value, q_value in ((-dx, x0), (dx, 1.0 - x0), (-dy, y0), (dy, 1.0 - y0)):
        if abs(p_value) < 1.0e-12:
            if q_value < 0.0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            if ratio > t1:
                return None
            if ratio > t0:
                t0 = ratio
        else:
            if ratio < t0:
                return None
            if ratio < t1:
                t1 = ratio

    clipped_start = (x0 + t0 * dx, y0 + t0 * dy)
    clipped_end = (x0 + t1 * dx, y0 + t1 * dy)
    return clipped_start, clipped_end


def _grid_from_unodes(unodes: list[tuple[int, float, float]]) -> tuple[list[float], list[float], dict[int, tuple[int, int]]]:
    x_values = sorted({round(x_coord, 10) for _, x_coord, _ in unodes})
    y_values = sorted({round(y_coord, 10) for _, _, y_coord in unodes})
    x_lookup = {value: index for index, value in enumerate(x_values)}
    y_lookup = {value: index for index, value in enumerate(y_values)}
    id_lookup = {
        int(unode_id): (x_lookup[round(x_coord, 10)], y_lookup[round(y_coord, 10)])
        for unode_id, x_coord, y_coord in unodes
    }
    return x_values, y_values, id_lookup


def _layout_from_unodes(unodes: list[tuple[int, float, float]]) -> dict[str, Any]:
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
    unodes: list[tuple[int, float, float]],
    lines: list[str],
) -> tuple[list[list[float]], tuple[int, int], dict[str, Any]]:
    layout = _layout_from_unodes(unodes)
    width, height = layout["grid_shape"]
    id_lookup = layout["id_lookup"]
    default, values = _parse_sparse_values(lines)
    field = [[default for _ in range(width)] for _ in range(height)]
    if layout["structured"]:
        for unode_id, value in values.items():
            if unode_id in id_lookup:
                ix, iy = id_lookup[unode_id]
                field[iy][ix] = float(value)
        return field, (width, height), layout

    sums = [[0.0 for _ in range(width)] for _ in range(height)]
    counts = [[0 for _ in range(width)] for _ in range(height)]
    for unode_id, _, _ in unodes:
        ix, iy = id_lookup[int(unode_id)]
        sums[iy][ix] += float(values.get(int(unode_id), default))
        counts[iy][ix] += 1
    for iy in range(height):
        for ix in range(width):
            if counts[iy][ix] > 0:
                field[iy][ix] = sums[iy][ix] / counts[iy][ix]
    return field, (width, height), layout


def _is_integer_label_field(field: list[list[float]]) -> bool:
    unique_values: set[int] = set()
    max_unique = max(256, (len(field) * len(field[0]) if field else 0) // 8)
    for row in field:
        for value in row:
            rounded = int(round(value))
            if abs(value - rounded) > 1.0e-5:
                return False
            unique_values.add(rounded)
            if len(unique_values) > max_unique:
                return False
    return True


def _default_palette_for(section_name: str, field: list[list[float]]) -> str:
    if section_name != "U_CONC_A" and _is_integer_label_field(field):
        return "labels"
    flat = [value for row in field for value in row]
    if section_name == "U_ATTRIB_A" and flat and (min(flat) < 0.0 or max(flat) > 1.0):
        return "heat"
    return "gray"


def _map_point_to_pixel(
    x_coord: float,
    y_coord: float,
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    width: int,
    height: int,
) -> list[int]:
    xspan = x_range[1] - x_range[0] if x_range[1] > x_range[0] else 1.0
    yspan = y_range[1] - y_range[0] if y_range[1] > y_range[0] else 1.0
    px = int(round((x_coord - x_range[0]) / xspan * max(0, width - 1)))
    py = int(round((y_coord - y_range[0]) / yspan * max(0, height - 1)))
    return [px, py]


def _serialize_flynns(
    flynns: list[dict[str, Any]],
    nodes: dict[int, tuple[float, float]],
    *,
    x_range: tuple[float, float],
    y_range: tuple[float, float],
    width: int,
    height: int,
    cell_bbox: list[tuple[float, float]] | None,
) -> list[dict[str, Any]]:
    transform = _make_cell_transform(cell_bbox=cell_bbox, x_range=x_range, y_range=y_range)
    periodic = bool(transform["periodic"])
    serialized: list[dict[str, Any]] = []
    for flynn in flynns:
        points = [nodes[node_id] for node_id in flynn["node_ids"] if node_id in nodes]
        if len(points) < 2:
            continue
        point_uv = [_point_to_cell(point, transform) for point in points]
        unwrapped_uv = _unwrap_cell_points(point_uv, periodic=periodic)
        paths: list[list[list[int]]] = []
        seen_segments: set[tuple[tuple[int, int], tuple[int, int]]] = set()

        segment_pairs = list(zip(unwrapped_uv, unwrapped_uv[1:]))
        closing_start = unwrapped_uv[-1]
        delta_u = float(point_uv[0][0]) - float(closing_start[0])
        delta_v = float(point_uv[0][1]) - float(closing_start[1])
        if periodic:
            delta_u -= round(delta_u)
            delta_v -= round(delta_v)
        segment_pairs.append((closing_start, (closing_start[0] + delta_u, closing_start[1] + delta_v)))

        shifts = (-1, 0, 1) if periodic else (0,)
        for start_uv, end_uv in segment_pairs:
            for shift_u in shifts:
                for shift_v in shifts:
                    clipped = _clip_segment_to_unit_square(
                        (start_uv[0] + shift_u, start_uv[1] + shift_v),
                        (end_uv[0] + shift_u, end_uv[1] + shift_v),
                    )
                    if clipped is None:
                        continue
                    clipped_points = []
                    for clipped_uv in clipped:
                        x_coord, y_coord = _cell_to_point(clipped_uv, transform)
                        clipped_points.append(
                            _map_point_to_pixel(
                                x_coord,
                                y_coord,
                                x_range=x_range,
                                y_range=y_range,
                                width=width,
                                height=height,
                            )
                        )
                    if clipped_points[0] == clipped_points[1]:
                        continue
                    segment = (tuple(clipped_points[0]), tuple(clipped_points[1]))
                    canonical = segment if segment[0] <= segment[1] else (segment[1], segment[0])
                    if canonical in seen_segments:
                        continue
                    seen_segments.add(canonical)
                    paths.append([clipped_points[0], clipped_points[1]])

        mean_u = sum(point[0] for point in unwrapped_uv) / len(unwrapped_uv)
        mean_v = sum(point[1] for point in unwrapped_uv) / len(unwrapped_uv)
        label_uv = (
            mean_u - math.floor(mean_u),
            mean_v - math.floor(mean_v),
        )
        label_x, label_y = _cell_to_point(label_uv, transform)
        serialized.append(
            {
                "flynn_id": int(flynn["flynn_id"]),
                "paths": paths,
                "label": _map_point_to_pixel(
                    label_x,
                    label_y,
                    x_range=x_range,
                    y_range=y_range,
                    width=width,
                    height=height,
                ),
            }
        )
    return serialized


def _encode_u16(values: list[int]) -> str:
    raw = bytearray()
    for value in values:
        raw.extend(int(value).to_bytes(2, byteorder="little", signed=False))
    return base64.b64encode(bytes(raw)).decode("ascii")


def _encode_u32(values: list[int]) -> str:
    raw = bytearray()
    for value in values:
        raw.extend(int(value).to_bytes(4, byteorder="little", signed=False))
    return base64.b64encode(bytes(raw)).decode("ascii")


def _encode_u8(values: list[int]) -> str:
    return base64.b64encode(bytes(values)).decode("ascii")


def _encode_field(
    field: list[list[float]],
    *,
    palette: str,
    display_range: tuple[float, float],
) -> dict[str, Any]:
    flat = [value for row in field for value in row]
    if palette == "labels":
        labels = [max(0, int(round(value))) for value in flat]
        max_label = max(labels) if labels else 0
        if max_label <= 65535:
            return {"kind": "labels_u16", "data": _encode_u16(labels), "max_label": int(max_label)}
        return {"kind": "labels_u32", "data": _encode_u32(labels), "max_label": int(max_label)}

    low, high = display_range
    span = high - low if high > low else 1.0
    quantized: list[int] = []
    for value in flat:
        normalized = max(0.0, min(1.0, (value - low) / span))
        quantized.append(int(round(255.0 * normalized)))
    return {"kind": "scalar_u8", "data": _encode_u8(quantized)}


def build_viewer_payload(
    elle_path: str | Path,
    *,
    attribute: str = "auto",
    palette: str = "auto",
    showelle_in: str | Path | None = None,
    overlay_boundaries: bool = True,
    label_flynns: bool | None = None,
    scale: int = 2,
    legend: bool = True,
) -> dict[str, Any]:
    sections = _parse_elle_sections(elle_path)
    unodes = _parse_unodes(sections.get("UNODES", []))
    nodes = _parse_location(sections.get("LOCATION", []))
    flynns = _parse_flynns(sections.get("FLYNNS", []))
    cell_bbox = _parse_cell_bounding_box(sections.get("OPTIONS", []))
    if not unodes:
        raise ValueError("ELLE file does not contain a UNODES section")
    if scale < 1:
        raise ValueError("scale must be at least 1")

    settings = _parse_showelle_settings(showelle_in)
    chosen_attribute = attribute if attribute != "auto" else (settings["attribute"] or "auto")
    section_name = _normalize_attribute_name(chosen_attribute)
    available_sections = [name for name in ("U_CONC_A", "U_ATTRIB_A", "U_ATTRIB_B") if name in sections]
    if section_name in {None, "AUTO"}:
        if "U_CONC_A" in sections:
            section_name = "U_CONC_A"
        elif available_sections:
            section_name = available_sections[0]
        else:
            raise ValueError("no supported unode attribute section found")

    field, grid_shape, layout = _field_from_sparse_unodes(unodes, sections[section_name])
    chosen_palette = palette if palette != "auto" else _default_palette_for(section_name, field)
    flat_values = [value for row in field for value in row]
    value_range = (
        float(min(flat_values)) if flat_values else 0.0,
        float(max(flat_values)) if flat_values else 0.0,
    )
    display_range = (
        value_range[0] if settings["vmin"] is None else float(settings["vmin"]),
        value_range[1] if settings["vmax"] is None else float(settings["vmax"]),
    )
    x_range = layout["x_range"]
    y_range = layout["y_range"]
    chosen_label_flynns = settings["flynn_labels"] if label_flynns is None else bool(label_flynns)

    return {
        "title": Path(elle_path).name,
        "elle_path": str(Path(elle_path)),
        "attribute": section_name,
        "palette": chosen_palette,
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "scale": int(scale),
        "legend": bool(legend and chosen_palette in {"gray", "heat"}),
        "overlay_boundaries": bool(overlay_boundaries and bool(flynns)),
        "flynn_labels": bool(chosen_label_flynns and bool(flynns)),
        "num_flynns": int(len(flynns)),
        "value_range": [value_range[0], value_range[1]],
        "display_range": [display_range[0], display_range[1]],
        "field_encoding": _encode_field(field, palette=chosen_palette, display_range=display_range),
        "flynns": _serialize_flynns(
            flynns,
            nodes,
            x_range=x_range,
            y_range=y_range,
            width=grid_shape[0],
            height=grid_shape[1],
            cell_bbox=cell_bbox,
        ),
    }


def _viewer_script_tag(
    payload_json: str,
    data_filename: str | None,
    *,
    data_version: str | None = None,
) -> str:
    if data_filename is None:
        return f"<script>window.ELLE_VIEWER_DATA = {payload_json};</script>"
    escaped = html.escape(data_filename, quote=True)
    if data_version:
        escaped = f"{escaped}?v={html.escape(data_version, quote=True)}"
    return f'<script src="{escaped}"></script>'


def _build_html_shell(
    title: str,
    *,
    data_filename: str | None,
    inline_payload: str | None = None,
    data_version: str | None = None,
) -> str:
    script_tag = (
        _viewer_script_tag(inline_payload or "{}", data_filename, data_version=data_version)
        if (data_filename or inline_payload)
        else ""
    )
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{html.escape(title)} Viewer</title>
  <style>
    :root {{
      --bg: #f5f0e8;
      --panel: #fffaf2;
      --ink: #1f1914;
      --muted: #73685b;
      --accent: #9e4d2e;
      --edge: #d9c9b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      min-height: 100vh;
      font-family: "Iowan Old Style", "Palatino Linotype", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, rgba(245, 214, 186, 0.5), transparent 26%),
        linear-gradient(180deg, #fbf8f2 0%, var(--bg) 100%);
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
      gap: 20px;
      padding: 20px;
    }}
    .panel {{
      background: rgba(255, 250, 242, 0.92);
      border: 1px solid var(--edge);
      border-radius: 18px;
      box-shadow: 0 14px 30px rgba(67, 49, 30, 0.08);
    }}
    .sidebar {{
      padding: 20px;
      display: grid;
      gap: 18px;
      align-content: start;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 8px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.05;
      max-width: 14ch;
    }}
    .subtle {{
      font-size: 14px;
      line-height: 1.45;
      color: var(--muted);
      word-break: break-word;
    }}
    .field {{
      display: grid;
      gap: 6px;
    }}
    label {{
      font-size: 12px;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    select, input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    select {{
      padding: 8px 10px;
      border-radius: 10px;
      border: 1px solid var(--edge);
      background: white;
      color: var(--ink);
      font: inherit;
    }}
    .checks {{
      display: grid;
      gap: 10px;
      padding: 14px;
      border-radius: 14px;
      border: 1px solid var(--edge);
      background: linear-gradient(180deg, rgba(255,255,255,0.82), rgba(240,228,214,0.72));
    }}
    .checks label {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
    }}
    .checks input {{
      width: 18px;
      height: 18px;
      accent-color: var(--accent);
    }}
    .meta {{
      display: grid;
      grid-template-columns: repeat(2, minmax(0, 1fr));
      gap: 10px;
    }}
    .meta-card {{
      padding: 12px;
      border-radius: 12px;
      background: white;
      border: 1px solid var(--edge);
    }}
    .meta-key {{
      font-size: 11px;
      letter-spacing: 0.1em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 6px;
    }}
    .meta-value {{
      font-size: 18px;
      font-weight: 600;
    }}
    .stage {{
      padding: 18px;
      display: grid;
      gap: 14px;
      min-width: 0;
    }}
    .toolbar {{
      display: flex;
      justify-content: space-between;
      align-items: center;
      gap: 12px;
      flex-wrap: wrap;
    }}
    .pill {{
      padding: 8px 12px;
      border-radius: 999px;
      background: #edd8c7;
      color: #7a3417;
      font-size: 13px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    .status {{
      padding: 8px 12px;
      border-radius: 999px;
      border: 1px solid var(--edge);
      background: rgba(255,255,255,0.85);
      color: #5c4435;
      font-family: "SFMono-Regular", "Cascadia Code", monospace;
      font-size: 12px;
    }}
    .viewer-wrap {{
      overflow: auto;
      padding: 18px;
      border-radius: 18px;
      border: 1px solid var(--edge);
      background:
        linear-gradient(45deg, rgba(255,255,255,0.86), rgba(244,236,227,0.92)),
        repeating-linear-gradient(
          45deg,
          rgba(158,77,46,0.04) 0px,
          rgba(158,77,46,0.04) 14px,
          rgba(255,255,255,0.08) 14px,
          rgba(255,255,255,0.08) 28px
        );
    }}
    #viewerCanvas {{
      image-rendering: pixelated;
      image-rendering: crisp-edges;
      background: white;
      border-radius: 8px;
      box-shadow: 0 10px 24px rgba(58, 43, 28, 0.18);
    }}
    #legendCanvas {{
      border-radius: 10px;
      border: 1px solid var(--edge);
      background: white;
    }}
    .foot {{
      display: flex;
      gap: 18px;
      flex-wrap: wrap;
      align-items: flex-start;
      color: var(--muted);
      font-size: 13px;
    }}
    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        max-width: none;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel sidebar">
      <div>
        <div class="eyebrow">Portable ELLE Viewer</div>
        <h1 id="titleLabel">{html.escape(title)}</h1>
        <p class="subtle" id="pathLabel"></p>
      </div>
      <div class="field">
        <label for="paletteSelect">Palette</label>
        <select id="paletteSelect">
          <option value="gray">Gray</option>
          <option value="heat">Heat</option>
          <option value="labels">Labels</option>
        </select>
      </div>
      <div class="field">
        <label for="zoomRange">Zoom</label>
        <input id="zoomRange" type="range" min="1" max="8" step="1">
      </div>
      <div class="checks">
        <label><span>Boundary Overlay</span><input id="boundariesToggle" type="checkbox"></label>
        <label><span>Flynn Labels</span><input id="labelsToggle" type="checkbox"></label>
        <label><span>Scalar Legend</span><input id="legendToggle" type="checkbox"></label>
      </div>
      <div class="meta">
        <div class="meta-card"><div class="meta-key">Attribute</div><div class="meta-value" id="attributeValue"></div></div>
        <div class="meta-card"><div class="meta-key">Grid</div><div class="meta-value" id="gridValue"></div></div>
        <div class="meta-card"><div class="meta-key">Flynns</div><div class="meta-value" id="flynnValue"></div></div>
        <div class="meta-card"><div class="meta-key">Range</div><div class="meta-value" id="rangeValue"></div></div>
      </div>
    </aside>
    <main class="panel stage">
      <div class="toolbar">
        <div class="pill" id="summaryPill"></div>
        <div class="status" id="hoverStatus">move over the field to inspect a cell</div>
      </div>
      <div class="viewer-wrap">
        <canvas id="viewerCanvas"></canvas>
      </div>
      <div class="foot">
        <canvas id="legendCanvas" width="92" height="260"></canvas>
        <div class="subtle">Keep the generated `.html` and `.data.js` files together. Open the HTML file in any browser and use scrollbars to pan when zoomed in.</div>
      </div>
    </main>
  </div>
  {script_tag}
  <script>
    const viewerData = window.ELLE_VIEWER_DATA;
    if (!viewerData) {{
      document.body.innerHTML = "<pre style='padding:20px;font-family:monospace'>Viewer data not found. Keep the HTML and .data.js files together.</pre>";
      throw new Error("window.ELLE_VIEWER_DATA is missing");
    }}

    const width = viewerData.grid_shape[0];
    const height = viewerData.grid_shape[1];
    const encoding = viewerData.field_encoding;

    function decodeBase64(data) {{
      const binary = atob(data);
      const bytes = new Uint8Array(binary.length);
      for (let index = 0; index < binary.length; index += 1) {{
        bytes[index] = binary.charCodeAt(index);
      }}
      return bytes;
    }}

    function decodeField(spec) {{
      const bytes = decodeBase64(spec.data);
      if (spec.kind === "labels_u16") {{
        const values = new Array(bytes.length / 2);
        for (let i = 0; i < values.length; i += 1) {{
          values[i] = bytes[i * 2] | (bytes[i * 2 + 1] << 8);
        }}
        return values;
      }}
      if (spec.kind === "labels_u32") {{
        const values = new Array(bytes.length / 4);
        for (let i = 0; i < values.length; i += 1) {{
          const base = i * 4;
          values[i] = (
            bytes[base] |
            (bytes[base + 1] << 8) |
            (bytes[base + 2] << 16) |
            (bytes[base + 3] << 24)
          ) >>> 0;
        }}
        return values;
      }}
      return Array.from(bytes);
    }}

    const field = decodeField(encoding);
    const flynns = viewerData.flynns;
    const canvas = document.getElementById("viewerCanvas");
    const ctx = canvas.getContext("2d");
    const legendCanvas = document.getElementById("legendCanvas");
    const legendCtx = legendCanvas.getContext("2d");
    const paletteSelect = document.getElementById("paletteSelect");
    const zoomRange = document.getElementById("zoomRange");
    const boundariesToggle = document.getElementById("boundariesToggle");
    const labelsToggle = document.getElementById("labelsToggle");
    const legendToggle = document.getElementById("legendToggle");
    const hoverStatus = document.getElementById("hoverStatus");

    document.getElementById("pathLabel").textContent = viewerData.elle_path;
    document.getElementById("attributeValue").textContent = viewerData.attribute;
    document.getElementById("gridValue").textContent = `${{width}} × ${{height}}`;
    document.getElementById("flynnValue").textContent = String(viewerData.num_flynns);
    document.getElementById("rangeValue").textContent =
      `${{viewerData.display_range[0].toFixed(3)}} → ${{viewerData.display_range[1].toFixed(3)}}`;

    const state = {{
      palette: viewerData.palette,
      scale: viewerData.scale,
      boundaries: viewerData.overlay_boundaries,
      labels: viewerData.flynn_labels,
      legend: viewerData.legend,
    }};

    paletteSelect.value = state.palette;
    zoomRange.value = String(state.scale);
    boundariesToggle.checked = state.boundaries;
    labelsToggle.checked = state.labels;
    legendToggle.checked = state.legend;

    function labelColor(label) {{
      const id = Math.max(0, Math.round(label));
      return [
        (id * 53 + 37) % 256,
        (id * 97 + 17) % 256,
        (id * 193 + 71) % 256,
      ];
    }}

    function clamp01(value) {{
      return Math.max(0, Math.min(1, value));
    }}

    function scalarColor(value, palette) {{
      const low = viewerData.display_range[0];
      const high = viewerData.display_range[1];
      const span = Math.max(1e-12, high - low);
      const t = clamp01((value - low) / span);
      if (palette === "heat") {{
        return [
          Math.round(255 * t),
          Math.round(255 * (1 - Math.abs(2 * t - 1))),
          Math.round(255 * (1 - t)),
        ];
      }}
      const gray = Math.round(255 * t);
      return [gray, gray, gray];
    }}

    function decodeScalar(index) {{
      const q = field[index];
      const low = viewerData.display_range[0];
      const high = viewerData.display_range[1];
      const span = Math.max(1e-12, high - low);
      return low + (q / 255) * span;
    }}

    function renderLegend() {{
      if (!(state.legend && (state.palette === "gray" || state.palette === "heat"))) {{
        legendCanvas.style.display = "none";
        return;
      }}
      legendCanvas.style.display = "block";
      legendCtx.clearRect(0, 0, legendCanvas.width, legendCanvas.height);
      for (let y = 0; y < legendCanvas.height; y += 1) {{
        const t = 1 - y / Math.max(1, legendCanvas.height - 1);
        const value = viewerData.display_range[0] + t * (viewerData.display_range[1] - viewerData.display_range[0]);
        const [r, g, b] = scalarColor(value, state.palette);
        legendCtx.fillStyle = `rgb(${{r}}, ${{g}}, ${{b}})`;
        legendCtx.fillRect(12, y, 22, 1);
      }}
      legendCtx.strokeStyle = "#3d3328";
      legendCtx.lineWidth = 1;
      legendCtx.strokeRect(11.5, 0.5, 23, legendCanvas.height - 1);
      legendCtx.fillStyle = "#3d3328";
      legendCtx.font = "12px monospace";
      legendCtx.fillText(viewerData.display_range[1].toExponential(2), 42, 14);
      legendCtx.fillText(viewerData.display_range[0].toExponential(2), 42, legendCanvas.height - 6);
    }}

    function render() {{
      const scale = Math.max(1, Number(state.scale) || 1);
      canvas.width = width * scale;
      canvas.height = height * scale;
      ctx.imageSmoothingEnabled = false;

      const base = document.createElement("canvas");
      base.width = width;
      base.height = height;
      const baseCtx = base.getContext("2d");
      const imageData = baseCtx.createImageData(width, height);
      const pixels = imageData.data;

      let offset = 0;
      for (let y = 0; y < height; y += 1) {{
        for (let x = 0; x < width; x += 1) {{
          const index = y * width + x;
          const [r, g, b] = state.palette === "labels"
            ? labelColor(field[index])
            : scalarColor(decodeScalar(index), state.palette);
          pixels[offset] = r;
          pixels[offset + 1] = g;
          pixels[offset + 2] = b;
          pixels[offset + 3] = 255;
          offset += 4;
        }}
      }}

      baseCtx.putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(base, 0, 0, canvas.width, canvas.height);

      if (state.boundaries) {{
        ctx.strokeStyle = state.palette === "gray" ? "#cf2e2e" : "#ffffff";
        ctx.lineWidth = Math.max(1, scale * 0.6);
        flynns.forEach((flynn) => {{
          const paths = flynn.paths || (flynn.points ? [flynn.points] : []);
          paths.forEach((path) => {{
            if (!path || path.length < 2) {{
              return;
            }}
            ctx.beginPath();
            path.forEach((point, index) => {{
              const drawX = point[0] * scale + scale / 2;
              const drawY = point[1] * scale + scale / 2;
              if (index === 0) {{
                ctx.moveTo(drawX, drawY);
              }} else {{
                ctx.lineTo(drawX, drawY);
              }}
            }});
            ctx.stroke();
          }});
        }});
      }}

      if (state.labels) {{
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = `${{Math.max(10, scale * 3.5)}}px monospace`;
        flynns.forEach((flynn) => {{
          const labelPoint = flynn.label || (() => {{
            const paths = flynn.paths || (flynn.points ? [flynn.points] : []);
            const firstPath = paths.find((path) => path && path.length > 0);
            return firstPath ? firstPath[0] : null;
          }})();
          if (!labelPoint) {{
            return;
          }}
          const cx = labelPoint[0] * scale + scale / 2;
          const cy = labelPoint[1] * scale + scale / 2;
          ctx.strokeStyle = "rgba(0, 0, 0, 0.7)";
          ctx.lineWidth = Math.max(2, scale * 0.5);
          ctx.strokeText(String(flynn.flynn_id), cx, cy);
          ctx.fillStyle = "#ffffff";
          ctx.fillText(String(flynn.flynn_id), cx, cy);
        }});
      }}

      renderLegend();
      document.getElementById("summaryPill").textContent =
        `${{viewerData.attribute}} · ${{state.palette}} · zoom ${{scale}}×`;
    }}

    paletteSelect.addEventListener("change", () => {{
      state.palette = paletteSelect.value;
      if (!(state.palette === "gray" || state.palette === "heat")) {{
        state.legend = false;
        legendToggle.checked = false;
      }}
      render();
    }});
    zoomRange.addEventListener("input", () => {{
      state.scale = Number(zoomRange.value);
      render();
    }});
    boundariesToggle.addEventListener("change", () => {{
      state.boundaries = boundariesToggle.checked;
      render();
    }});
    labelsToggle.addEventListener("change", () => {{
      state.labels = labelsToggle.checked;
      render();
    }});
    legendToggle.addEventListener("change", () => {{
      state.legend = legendToggle.checked;
      render();
    }});
    canvas.addEventListener("mousemove", (event) => {{
      const rect = canvas.getBoundingClientRect();
      const scale = Math.max(1, Number(state.scale) || 1);
      const x = Math.max(0, Math.min(width - 1, Math.floor((event.clientX - rect.left) / scale)));
      const y = Math.max(0, Math.min(height - 1, Math.floor((event.clientY - rect.top) / scale)));
      const index = y * width + x;
      const value = state.palette === "labels" ? field[index] : decodeScalar(index);
      hoverStatus.textContent = `cell (${{x}}, ${{y}}) value=${{Number(value).toFixed(6)}}`;
    }});
    canvas.addEventListener("mouseleave", () => {{
      hoverStatus.textContent = "move over the field to inspect a cell";
    }});

    render();
  </script>
</body>
</html>
"""


def write_viewer_bundle(
    elle_path: str | Path,
    *,
    outpath: str | Path | None = None,
    attribute: str = "auto",
    palette: str = "auto",
    showelle_in: str | Path | None = None,
    overlay_boundaries: bool = True,
    label_flynns: bool | None = None,
    scale: int = 2,
    legend: bool = True,
    single_file: bool = False,
) -> dict[str, Any]:
    payload = build_viewer_payload(
        elle_path,
        attribute=attribute,
        palette=palette,
        showelle_in=showelle_in,
        overlay_boundaries=overlay_boundaries,
        label_flynns=label_flynns,
        scale=scale,
        legend=legend,
    )
    outpath_obj = Path(outpath) if outpath is not None else Path(elle_path).with_name(
        f"{Path(elle_path).stem}_viewer.html"
    )
    outpath_obj.parent.mkdir(parents=True, exist_ok=True)

    payload_json = json.dumps(payload, separators=(",", ":"))
    if single_file:
        data_outpath = None
        html_text = _build_html_shell(payload["title"], data_filename=None, inline_payload=payload_json)
    else:
        data_outpath = outpath_obj.with_name(f"{outpath_obj.stem}.data.js")
        data_outpath.write_text(f"window.ELLE_VIEWER_DATA = {payload_json};\n", encoding="utf-8")
        data_version = hashlib.sha1(payload_json.encode("utf-8")).hexdigest()[:12]
        html_text = _build_html_shell(
            payload["title"],
            data_filename=data_outpath.name,
            data_version=data_version,
        )

    outpath_obj.write_text(html_text, encoding="utf-8")
    return {
        "elle_path": payload["elle_path"],
        "outpath": str(outpath_obj),
        "data_outpath": str(data_outpath) if data_outpath is not None else None,
        "attribute": payload["attribute"],
        "palette": payload["palette"],
        "grid_shape": payload["grid_shape"],
        "overlay_boundaries": payload["overlay_boundaries"],
        "flynn_labels": payload["flynn_labels"],
        "legend": payload["legend"],
        "scale": payload["scale"],
        "num_flynns": payload["num_flynns"],
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build a portable browser viewer for an ELLE file")
    parser.add_argument("elle_file", type=Path, help="Path to an .elle file")
    parser.add_argument("--out", type=Path, help="Output HTML path")
    parser.add_argument("--attribute", default="auto", help="Attribute to render, e.g. CONC_A or U_ATTRIB_A")
    parser.add_argument("--palette", choices=("auto", "gray", "heat", "labels"), default="auto")
    parser.add_argument("--scale", type=int, default=2, help="Initial zoom factor in the browser viewer")
    parser.add_argument("--no-legend", action="store_true", help="Disable the initial scalar legend state")
    parser.add_argument("--label-flynns", action="store_true", help="Enable flynn labels by default")
    parser.add_argument("--showelle-in", type=Path, help="Optional showelle.in file to reuse attribute/range settings")
    parser.add_argument("--no-boundaries", action="store_true", help="Disable boundary overlay by default")
    parser.add_argument("--single-file", action="store_true", help="Embed the data directly into the HTML file")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    result = write_viewer_bundle(
        args.elle_file,
        outpath=args.out,
        attribute=args.attribute,
        palette=args.palette,
        showelle_in=args.showelle_in,
        overlay_boundaries=not args.no_boundaries,
        label_flynns=True if args.label_flynns else None,
        scale=args.scale,
        legend=not args.no_legend,
        single_file=args.single_file,
    )
    summary = (
        f"viewer {result['elle_path']} -> {result['outpath']} "
        f"attribute={result['attribute']} palette={result['palette']} "
        f"grid={tuple(result['grid_shape'])} flynns={result['num_flynns']} "
        f"boundaries={int(result['overlay_boundaries'])} "
        f"labels={int(result['flynn_labels'])} legend={int(result['legend'])} scale={result['scale']}"
    )
    if result["data_outpath"] is not None:
        summary += f" data={result['data_outpath']}"
    print(summary)


if __name__ == "__main__":
    main()
