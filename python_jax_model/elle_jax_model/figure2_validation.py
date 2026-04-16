from __future__ import annotations

import json
import math
import re
from pathlib import Path
from typing import Any

import numpy as np

from .simulation import load_elle_label_seed


def _parse_step_from_name(path: str | Path) -> int | None:
    match = re.search(r"(\d+)(?=\.elle$)", Path(path).name)
    if match is None:
        return None
    return int(match.group(1))


def _periodic_component_areas(labels: np.ndarray) -> np.ndarray:
    labels_np = np.asarray(labels, dtype=np.int32)
    nx, ny = labels_np.shape
    visited = np.zeros((nx, ny), dtype=bool)
    areas: list[float] = []
    total_pixels = float(nx * ny)

    for ix in range(nx):
        for iy in range(ny):
            if visited[ix, iy]:
                continue
            label = int(labels_np[ix, iy])
            stack = [(ix, iy)]
            visited[ix, iy] = True
            pixel_count = 0

            while stack:
                cx, cy = stack.pop()
                pixel_count += 1
                for dx, dy in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    nx_i = (cx + dx) % nx
                    ny_i = (cy + dy) % ny
                    if visited[nx_i, ny_i] or int(labels_np[nx_i, ny_i]) != label:
                        continue
                    visited[nx_i, ny_i] = True
                    stack.append((nx_i, ny_i))

            areas.append(float(pixel_count) / total_pixels)

    return np.asarray(areas, dtype=np.float64)


def summarize_elle_label_area_distribution(
    elle_path: str | Path,
    *,
    attribute: str = "auto",
) -> dict[str, Any]:
    seed = load_elle_label_seed(elle_path, attribute=attribute)
    labels = np.asarray(seed["label_field"], dtype=np.int32)
    areas = _periodic_component_areas(labels)

    return {
        "path": str(Path(elle_path)),
        "step": _parse_step_from_name(elle_path),
        "attribute": str(seed["attribute"]),
        "grid_shape": [int(labels.shape[0]), int(labels.shape[1])],
        "source_label_count": int(seed["num_labels"]),
        "grain_count": int(areas.size),
        "mean_grain_area": float(areas.mean()) if areas.size else 0.0,
        "std_grain_area": float(areas.std()) if areas.size else 0.0,
        "median_grain_area": float(np.median(areas)) if areas.size else 0.0,
        "grain_area_fractions": [float(value) for value in areas.tolist()],
    }


def collect_elle_label_area_distributions(
    directory: str | Path,
    *,
    pattern: str = "*.elle",
    attribute: str = "auto",
) -> list[dict[str, Any]]:
    directory_path = Path(directory)
    snapshots = [
        summarize_elle_label_area_distribution(path, attribute=attribute)
        for path in sorted(directory_path.glob(pattern))
    ]
    return sorted(
        snapshots,
        key=lambda snapshot: (
            snapshot["step"] is None,
            snapshot["step"] if snapshot["step"] is not None else snapshot["path"],
        ),
    )


def _gaussian_kde(values: np.ndarray, grid: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    x_grid = np.asarray(grid, dtype=np.float64)
    if samples.size == 0 or x_grid.size == 0:
        return np.zeros_like(x_grid, dtype=np.float64)
    if samples.size == 1:
        bandwidth = max(float(x_grid[-1] - x_grid[0]) / 100.0, 1.0e-6)
    else:
        std = float(np.std(samples, ddof=1))
        q75, q25 = np.percentile(samples, [75.0, 25.0])
        iqr = float(q75 - q25)
        sigma = std
        if iqr > 0.0:
            sigma = min(std, iqr / 1.349) if std > 0.0 else iqr / 1.349
        bandwidth = 0.9 * sigma * (float(samples.size) ** (-0.2)) if sigma > 0.0 else 0.0
        if not np.isfinite(bandwidth) or bandwidth <= 0.0:
            bandwidth = max(float(x_grid[-1] - x_grid[0]) / 100.0, 1.0e-6)

    diffs = (x_grid[:, None] - samples[None, :]) / bandwidth
    density = np.exp(-0.5 * diffs * diffs).mean(axis=1) / (bandwidth * math.sqrt(2.0 * math.pi))
    return density.astype(np.float64)


def _histogram_density(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    samples = np.asarray(values, dtype=np.float64)
    edges = np.asarray(bin_edges, dtype=np.float64)
    if samples.size == 0 or edges.size < 2:
        return np.zeros(max(edges.size - 1, 0), dtype=np.float64)
    hist, _ = np.histogram(samples, bins=edges, density=True)
    return np.nan_to_num(hist, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float64)


def build_figure2_line_validation_report(
    *,
    reference_dir: str | Path,
    candidate_dir: str | Path,
    pattern: str = "*.elle",
    attribute: str = "auto",
    kde_points: int = 128,
) -> dict[str, Any]:
    reference = collect_elle_label_area_distributions(reference_dir, pattern=pattern, attribute=attribute)
    candidate = collect_elle_label_area_distributions(candidate_dir, pattern=pattern, attribute=attribute)
    reference_by_step = {snapshot["step"]: snapshot for snapshot in reference if snapshot["step"] is not None}
    candidate_by_step = {snapshot["step"]: snapshot for snapshot in candidate if snapshot["step"] is not None}
    matched_steps = sorted(set(reference_by_step) & set(candidate_by_step))

    reference_means: list[float] = []
    reference_stds: list[float] = []
    candidate_means: list[float] = []
    candidate_stds: list[float] = []

    all_areas: list[float] = []
    for step in matched_steps:
        all_areas.extend(reference_by_step[step]["grain_area_fractions"])
        all_areas.extend(candidate_by_step[step]["grain_area_fractions"])
        reference_means.append(float(reference_by_step[step]["mean_grain_area"]))
        reference_stds.append(float(reference_by_step[step]["std_grain_area"]))
        candidate_means.append(float(candidate_by_step[step]["mean_grain_area"]))
        candidate_stds.append(float(candidate_by_step[step]["std_grain_area"]))

    if all_areas:
        max_area = max(all_areas)
        grid_max = max(max_area * 1.05, 1.0e-6)
    else:
        grid_max = 1.0
    area_grid = np.linspace(0.0, grid_max, max(int(kde_points), 16), dtype=np.float64)
    histogram_bins = min(max(int(kde_points) // 4, 12), 28)
    histogram_edges = np.linspace(0.0, grid_max, histogram_bins + 1, dtype=np.float64)
    histogram_centers = 0.5 * (histogram_edges[:-1] + histogram_edges[1:])

    distributions: list[dict[str, Any]] = []
    for step in matched_steps:
        ref = reference_by_step[step]
        cand = candidate_by_step[step]
        ref_areas = np.asarray(ref["grain_area_fractions"], dtype=np.float64)
        cand_areas = np.asarray(cand["grain_area_fractions"], dtype=np.float64)
        distributions.append(
            {
                "step": int(step),
                "grain_area_grid": [float(value) for value in area_grid.tolist()],
                "grain_area_histogram_bin_edges": [float(value) for value in histogram_edges.tolist()],
                "grain_area_histogram_bin_centers": [float(value) for value in histogram_centers.tolist()],
                "reference": {
                    "grain_count": int(ref["grain_count"]),
                    "mean_grain_area": float(ref["mean_grain_area"]),
                    "std_grain_area": float(ref["std_grain_area"]),
                    "histogram_density": [float(value) for value in _histogram_density(ref_areas, histogram_edges).tolist()],
                    "kde": [float(value) for value in _gaussian_kde(ref_areas, area_grid).tolist()],
                },
                "candidate": {
                    "grain_count": int(cand["grain_count"]),
                    "mean_grain_area": float(cand["mean_grain_area"]),
                    "std_grain_area": float(cand["std_grain_area"]),
                    "histogram_density": [
                        float(value) for value in _histogram_density(cand_areas, histogram_edges).tolist()
                    ],
                    "kde": [float(value) for value in _gaussian_kde(cand_areas, area_grid).tolist()],
                },
            }
        )

    diffs = np.asarray(candidate_means, dtype=np.float64) - np.asarray(reference_means, dtype=np.float64)
    ref_scale = max(reference_means) if reference_means else 1.0
    mean_line = {
        "steps": [int(step) for step in matched_steps],
        "reference_mean_grain_area": [float(value) for value in reference_means],
        "reference_std_grain_area": [float(value) for value in reference_stds],
        "candidate_mean_grain_area": [float(value) for value in candidate_means],
        "candidate_std_grain_area": [float(value) for value in candidate_stds],
        "mae": float(np.mean(np.abs(diffs))) if diffs.size else float("nan"),
        "rmse": float(np.sqrt(np.mean(diffs * diffs))) if diffs.size else float("nan"),
        "normalized_rmse": (
            float(np.sqrt(np.mean(diffs * diffs)) / max(ref_scale, 1.0e-12))
            if diffs.size
            else float("nan")
        ),
    }

    return {
        "reference_dir": str(Path(reference_dir)),
        "candidate_dir": str(Path(candidate_dir)),
        "pattern": str(pattern),
        "attribute": str(attribute),
        "matched_steps": [int(step) for step in matched_steps],
        "reference_sequence": reference,
        "candidate_sequence": candidate,
        "figure2_like_validation": {
            "description": (
                "Figure-2-style static grain-growth validation using grain-area distributions, "
                "KDEs, and mean-grain-area evolution over time."
            ),
            "mean_grain_area_line": mean_line,
            "grain_area_distributions": distributions,
        },
    }


def write_figure2_line_validation_report(path: str | Path, report: dict[str, Any]) -> Path:
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return outpath


def _series_band_points(
    steps: list[int],
    means: list[float],
    stds: list[float],
    map_x,
    map_y,
) -> str:
    upper = [(map_x(step), map_y(mean + std)) for step, mean, std in zip(steps, means, stds)]
    lower = [(map_x(step), map_y(mean - std)) for step, mean, std in zip(steps, means, stds)]
    points = upper + list(reversed(lower))
    return " ".join(f"{x:.2f},{y:.2f}" for x, y in points)


def _series_line_path(steps: list[int], values: list[float], map_x, map_y) -> str:
    coords = [(map_x(step), map_y(value)) for step, value in zip(steps, values)]
    if not coords:
        return ""
    head = f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"
    tail = " ".join(f"L {x:.2f} {y:.2f}" for x, y in coords[1:])
    return f"{head} {tail}".strip()


def _xy_line_path(x_values: list[float], y_values: list[float], map_x, map_y) -> str:
    coords = [(map_x(x_value), map_y(y_value)) for x_value, y_value in zip(x_values, y_values)]
    if not coords:
        return ""
    head = f"M {coords[0][0]:.2f} {coords[0][1]:.2f}"
    tail = " ".join(f"L {x:.2f} {y:.2f}" for x, y in coords[1:])
    return f"{head} {tail}".strip()


def write_figure2_line_validation_html(
    path: str | Path,
    report: dict[str, Any],
    *,
    title: str = "Figure 2 style grain-area validation",
) -> Path:
    panel = report["figure2_like_validation"]["mean_grain_area_line"]
    distributions = report["figure2_like_validation"]["grain_area_distributions"]
    steps = [int(step) for step in panel["steps"]]
    ref_mean = [float(value) for value in panel["reference_mean_grain_area"]]
    ref_std = [float(value) for value in panel["reference_std_grain_area"]]
    cand_mean = [float(value) for value in panel["candidate_mean_grain_area"]]
    cand_std = [float(value) for value in panel["candidate_std_grain_area"]]

    width = 900
    height = 560
    margin_left = 80
    margin_right = 40
    margin_top = 70
    margin_bottom = 70
    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    if steps:
        min_step = min(steps)
        max_step = max(steps)
    else:
        min_step = 0
        max_step = 1
    if max_step == min_step:
        max_step = min_step + 1

    y_values = []
    for mean, std in zip(ref_mean, ref_std):
        y_values.extend([mean - std, mean + std])
    for mean, std in zip(cand_mean, cand_std):
        y_values.extend([mean - std, mean + std])
    y_values = [value for value in y_values if np.isfinite(value)]
    y_min = min(y_values) if y_values else 0.0
    y_max = max(y_values) if y_values else 1.0
    y_min = min(y_min, 0.0)
    if y_max <= y_min:
        y_max = y_min + 1.0
    y_pad = 0.08 * (y_max - y_min)
    y_min -= y_pad
    y_max += y_pad

    def map_x(step: int) -> float:
        return margin_left + (float(step - min_step) / float(max_step - min_step)) * plot_width

    def map_y(value: float) -> float:
        return margin_top + (float(y_max - value) / float(y_max - y_min)) * plot_height

    x_ticks = steps if steps else [0, 1]
    y_ticks = np.linspace(y_min, y_max, 5)

    ref_band = _series_band_points(steps, ref_mean, ref_std, map_x, map_y)
    cand_band = _series_band_points(steps, cand_mean, cand_std, map_x, map_y)
    ref_path = _series_line_path(steps, ref_mean, map_x, map_y)
    cand_path = _series_line_path(steps, cand_mean, map_x, map_y)

    ref_points = "\n".join(
        f'<circle cx="{map_x(step):.2f}" cy="{map_y(value):.2f}" r="3.5" fill="#c43c35" />'
        for step, value in zip(steps, ref_mean)
    )
    cand_points = "\n".join(
        f'<circle cx="{map_x(step):.2f}" cy="{map_y(value):.2f}" r="3.5" fill="#255f99" />'
        for step, value in zip(steps, cand_mean)
    )

    x_tick_svg = "\n".join(
        (
            f'<line x1="{map_x(step):.2f}" y1="{margin_top + plot_height:.2f}" '
            f'x2="{map_x(step):.2f}" y2="{margin_top + plot_height + 6:.2f}" stroke="#444" stroke-width="1" />'
            f'<text x="{map_x(step):.2f}" y="{margin_top + plot_height + 24:.2f}" text-anchor="middle" '
            f'font-size="12" fill="#333">{step}</text>'
        )
        for step in x_ticks
    )
    y_tick_svg = "\n".join(
        (
            f'<line x1="{margin_left - 6:.2f}" y1="{map_y(value):.2f}" '
            f'x2="{margin_left:.2f}" y2="{map_y(value):.2f}" stroke="#444" stroke-width="1" />'
            f'<line x1="{margin_left:.2f}" y1="{map_y(value):.2f}" '
            f'x2="{margin_left + plot_width:.2f}" y2="{map_y(value):.2f}" stroke="#ddd" stroke-width="1" />'
            f'<text x="{margin_left - 10:.2f}" y="{map_y(value) + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="#333">{value:.4f}</text>'
        )
        for value in y_ticks
    )

    panel_svg_width = 900
    panel_svg_height = 520
    panel_cols = 5
    panel_rows = max(int(math.ceil(max(len(distributions), 1) / panel_cols)), 1)
    panel_margin_left = 60
    panel_margin_right = 24
    panel_margin_top = 42
    panel_margin_bottom = 48
    panel_gap_x = 16
    panel_gap_y = 18
    panel_width = (
        panel_svg_width - panel_margin_left - panel_margin_right - (panel_cols - 1) * panel_gap_x
    ) / panel_cols
    panel_height = (
        panel_svg_height - panel_margin_top - panel_margin_bottom - (panel_rows - 1) * panel_gap_y
    ) / panel_rows

    kde_x_values = distributions[0]["grain_area_grid"] if distributions else [0.0, 1.0]
    kde_x_min = float(min(kde_x_values))
    kde_x_max = float(max(kde_x_values))
    if kde_x_max <= kde_x_min:
        kde_x_max = kde_x_min + 1.0
    hist_x_values = distributions[0]["grain_area_histogram_bin_edges"] if distributions else [0.0, 1.0]
    hist_x_min = float(min(hist_x_values))
    hist_x_max = float(max(hist_x_values))
    if hist_x_max <= hist_x_min:
        hist_x_max = hist_x_min + 1.0

    hist_y_max = 0.0
    for item in distributions:
        hist_y_max = max(
            hist_y_max,
            max(float(value) for value in item["reference"]["histogram_density"]),
            max(float(value) for value in item["candidate"]["histogram_density"]),
        )
    hist_y_max = max(hist_y_max * 1.08, 1.0)

    kde_y_max = 0.0
    for item in distributions:
        kde_y_max = max(
            kde_y_max,
            max(float(value) for value in item["reference"]["kde"]),
            max(float(value) for value in item["candidate"]["kde"]),
        )
    kde_y_max = max(kde_y_max * 1.08, 1.0)

    histogram_panels: list[str] = []
    kde_panels: list[str] = []
    for index, item in enumerate(distributions):
        row = index // panel_cols
        col = index % panel_cols
        panel_x = panel_margin_left + col * (panel_width + panel_gap_x)
        panel_y = panel_margin_top + row * (panel_height + panel_gap_y)

        def hist_map_x(value: float) -> float:
            return panel_x + ((float(value) - hist_x_min) / (hist_x_max - hist_x_min)) * panel_width

        def hist_map_y(value: float) -> float:
            return panel_y + panel_height - (float(value) / hist_y_max) * panel_height

        def kde_map_x(value: float) -> float:
            return panel_x + ((float(value) - kde_x_min) / (kde_x_max - kde_x_min)) * panel_width

        def kde_map_y(value: float) -> float:
            return panel_y + panel_height - (float(value) / kde_y_max) * panel_height

        histogram_edges = [float(value) for value in item["grain_area_histogram_bin_edges"]]
        histogram_ref = [float(value) for value in item["reference"]["histogram_density"]]
        histogram_cand = [float(value) for value in item["candidate"]["histogram_density"]]
        ref_hist_rects = []
        cand_hist_rects = []
        for left, right, ref_value, cand_value in zip(
            histogram_edges[:-1],
            histogram_edges[1:],
            histogram_ref,
            histogram_cand,
        ):
            x = hist_map_x(left)
            width_px = max(hist_map_x(right) - x, 1.0)
            ref_y = hist_map_y(ref_value)
            cand_y = hist_map_y(cand_value)
            ref_hist_rects.append(
                f'<rect x="{x:.2f}" y="{ref_y:.2f}" width="{width_px:.2f}" '
                f'height="{panel_y + panel_height - ref_y:.2f}" fill="#c43c35" opacity="0.28" />'
            )
            cand_hist_rects.append(
                f'<rect x="{x:.2f}" y="{cand_y:.2f}" width="{width_px:.2f}" '
                f'height="{panel_y + panel_height - cand_y:.2f}" fill="#255f99" opacity="0.28" />'
            )

        histogram_panels.append(
            f"""
      <g>
        <rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}" fill="#fff" stroke="#d7d7d7" stroke-width="1" />
        <line x1="{panel_x:.2f}" y1="{panel_y + panel_height:.2f}" x2="{panel_x + panel_width:.2f}" y2="{panel_y + panel_height:.2f}" stroke="#999" stroke-width="1" />
        <line x1="{panel_x:.2f}" y1="{panel_y:.2f}" x2="{panel_x:.2f}" y2="{panel_y + panel_height:.2f}" stroke="#999" stroke-width="1" />
        <text x="{panel_x + 8:.2f}" y="{panel_y + 16:.2f}" font-size="12" fill="#333">step {int(item['step'])}</text>
        <text x="{panel_x + 8:.2f}" y="{panel_y + 32:.2f}" font-size="11" fill="#555">ref n={int(item['reference']['grain_count'])}, cand n={int(item['candidate']['grain_count'])}</text>
        {''.join(ref_hist_rects)}
        {''.join(cand_hist_rects)}
      </g>"""
        )

        ref_kde_path = _xy_line_path(
            [float(value) for value in item["grain_area_grid"]],
            [float(value) for value in item["reference"]["kde"]],
            kde_map_x,
            kde_map_y,
        )
        cand_kde_path = _xy_line_path(
            [float(value) for value in item["grain_area_grid"]],
            [float(value) for value in item["candidate"]["kde"]],
            kde_map_x,
            kde_map_y,
        )

        kde_panels.append(
            f"""
      <g>
        <rect x="{panel_x:.2f}" y="{panel_y:.2f}" width="{panel_width:.2f}" height="{panel_height:.2f}" fill="#fff" stroke="#d7d7d7" stroke-width="1" />
        <line x1="{panel_x:.2f}" y1="{panel_y + panel_height:.2f}" x2="{panel_x + panel_width:.2f}" y2="{panel_y + panel_height:.2f}" stroke="#999" stroke-width="1" />
        <line x1="{panel_x:.2f}" y1="{panel_y:.2f}" x2="{panel_x:.2f}" y2="{panel_y + panel_height:.2f}" stroke="#999" stroke-width="1" />
        <text x="{panel_x + 8:.2f}" y="{panel_y + 16:.2f}" font-size="12" fill="#333">step {int(item['step'])}</text>
        <text x="{panel_x + 8:.2f}" y="{panel_y + 32:.2f}" font-size="11" fill="#555">ref n={int(item['reference']['grain_count'])}, cand n={int(item['candidate']['grain_count'])}</text>
        <path d="{ref_kde_path}" fill="none" stroke="#c43c35" stroke-width="2.0"></path>
        <path d="{cand_kde_path}" fill="none" stroke="#255f99" stroke-width="2.0"></path>
      </g>"""
        )

    panel_tick_svg = "\n".join(
        (
            f'<text x="{panel_margin_left + idx * (panel_width + panel_gap_x) + panel_width / 2:.2f}" '
            f'y="{panel_svg_height - 12:.2f}" text-anchor="middle" font-size="12" fill="#222">grain area fraction</text>'
        )
        for idx in range(min(panel_cols, max(len(distributions), 1)))
    )

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>{title}</title>
  <style>
    body {{
      font-family: Arial, sans-serif;
      margin: 24px;
      color: #222;
      background: #fafafa;
    }}
    .card {{
      background: white;
      border: 1px solid #ddd;
      border-radius: 10px;
      padding: 18px 20px;
      max-width: 980px;
      box-shadow: 0 1px 4px rgba(0,0,0,0.06);
    }}
    .meta {{
      margin: 0 0 14px 0;
      font-size: 14px;
      line-height: 1.5;
    }}
    .legend {{
      display: flex;
      gap: 18px;
      margin: 10px 0 0 0;
      font-size: 13px;
      flex-wrap: wrap;
    }}
    .legend span {{
      display: inline-flex;
      align-items: center;
      gap: 8px;
    }}
    .swatch {{
      width: 14px;
      height: 14px;
      border-radius: 3px;
      display: inline-block;
    }}
    code {{
      background: #f3f3f3;
      padding: 1px 5px;
      border-radius: 4px;
    }}
    .section-title {{
      margin: 22px 0 10px 0;
      font-size: 16px;
      color: #222;
    }}
  </style>
</head>
<body>
  <div class="card">
    <h2 style="margin-top:0;">{title}</h2>
    <p class="meta">
      Figure-2-style static grain-growth validation built from rasterized ELLE grain areas.<br />
      Reference: <code>{report['reference_dir']}</code><br />
      Candidate: <code>{report['candidate_dir']}</code><br />
      Matched steps: <code>{', '.join(str(step) for step in steps)}</code><br />
      Mean-line RMSE: <code>{panel['rmse']:.6f}</code>,
      normalized RMSE: <code>{panel['normalized_rmse']:.6f}</code>
    </p>
    <div class="section-title">Figure 2(a)-Style Grain-Area Distribution By Step</div>
    <svg width="{panel_svg_width}" height="{panel_svg_height}" viewBox="0 0 {panel_svg_width} {panel_svg_height}" role="img" aria-label="Figure 2 style grain-area distribution comparison">
      <rect x="0" y="0" width="{panel_svg_width}" height="{panel_svg_height}" fill="#ffffff" />
      <text x="{panel_margin_left:.2f}" y="20" font-size="13" fill="#222">Reference (red) vs candidate (blue) histogram densities, analogous to Liu &amp; Suckale Figure 2(a).</text>
      {''.join(histogram_panels)}
      {panel_tick_svg}
    </svg>
    <div class="section-title">Figure 2(b)-Style Grain-Area KDE Comparison By Step</div>
    <svg width="{panel_svg_width}" height="{panel_svg_height}" viewBox="0 0 {panel_svg_width} {panel_svg_height}" role="img" aria-label="Figure 2 style KDE comparison">
      <rect x="0" y="0" width="{panel_svg_width}" height="{panel_svg_height}" fill="#ffffff" />
      <text x="{panel_margin_left:.2f}" y="20" font-size="13" fill="#222">Reference (red) vs candidate (blue) grain-area KDEs, analogous to Liu &amp; Suckale Figure 2(b).</text>
      {''.join(kde_panels)}
      {panel_tick_svg}
    </svg>
    <div class="section-title">Figure 2(c)-Style Mean Grain Area Over Time</div>
    <svg width="{width}" height="{height}" viewBox="0 0 {width} {height}" role="img" aria-label="{title}">
      <rect x="0" y="0" width="{width}" height="{height}" fill="#ffffff" />
      <line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5" />
      <line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="#222" stroke-width="1.5" />
      {y_tick_svg}
      {x_tick_svg}
      <text x="{margin_left + plot_width / 2:.2f}" y="{height - 18:.2f}" text-anchor="middle" font-size="14" fill="#222">Step</text>
      <text x="24" y="{margin_top + plot_height / 2:.2f}" text-anchor="middle" font-size="14" fill="#222"
            transform="rotate(-90 24 {margin_top + plot_height / 2:.2f})">Mean grain area</text>
      <polygon points="{ref_band}" fill="#c43c35" opacity="0.14"></polygon>
      <polygon points="{cand_band}" fill="#255f99" opacity="0.18"></polygon>
      <path d="{ref_path}" fill="none" stroke="#c43c35" stroke-width="2.5"></path>
      <path d="{cand_path}" fill="none" stroke="#255f99" stroke-width="2.5"></path>
      {ref_points}
      {cand_points}
    </svg>
    <div class="legend">
      <span><i class="swatch" style="background:#c43c35;"></i>Reference histogram density</span>
      <span><i class="swatch" style="background:#255f99;"></i>Candidate histogram density</span>
      <span><i class="swatch" style="background:#c43c35;"></i>Reference KDE</span>
      <span><i class="swatch" style="background:#255f99;"></i>Candidate KDE</span>
      <span><i class="swatch" style="background:#c43c35;"></i>Reference mean ± std</span>
      <span><i class="swatch" style="background:#255f99;"></i>Candidate mean ± std</span>
    </div>
  </div>
</body>
</html>
"""
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(html, encoding="utf-8")
    return outpath
