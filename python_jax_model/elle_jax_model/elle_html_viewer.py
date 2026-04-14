from __future__ import annotations

import html
import json
from pathlib import Path
from typing import Any

import numpy as np

from portable_elle_viewer import write_viewer_bundle as _portable_write_viewer_bundle

from .elle_visualize import (
    ShowelleSettings,
    _field_from_sparse_unodes,
    _is_integer_label_field,
    _layout_from_unodes,
    _normalize_attribute_name,
    _parse_elle_sections,
    _parse_flynns,
    _parse_location,
    _parse_showelle_settings,
    _parse_unodes,
)


def _default_palette_for(section_name: str, field: np.ndarray) -> str:
    if _is_integer_label_field(field) and section_name != "U_CONC_A":
        return "labels"
    if section_name == "U_ATTRIB_A" and (float(field.min()) < 0.0 or float(field.max()) > 1.0):
        return "heat"
    return "gray"


def _serialize_flynns(
    flynns: list[dict[str, Any]],
    nodes: dict[int, tuple[float, float]],
) -> list[dict[str, Any]]:
    serialized: list[dict[str, Any]] = []
    for flynn in flynns:
        points = [nodes[node_id] for node_id in flynn["node_ids"] if node_id in nodes]
        if len(points) < 2:
            continue
        serialized.append(
            {
                "flynn_id": int(flynn["flynn_id"]),
                "points": [[round(float(x_coord), 8), round(float(y_coord), 8)] for x_coord, y_coord in points],
            }
        )
    return serialized


def _build_viewer_payload(
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
    available_sections = [name for name in ("U_CONC_A", "U_ATTRIB_A", "U_ATTRIB_B") if name in sections]
    if section_name in {None, "AUTO"}:
        if "U_CONC_A" in sections:
            section_name = "U_CONC_A"
        elif available_sections:
            section_name = available_sections[0]
        else:
            raise ValueError("no supported unode attribute section found")

    field, grid_shape = _field_from_sparse_unodes(unodes, sections[section_name])
    chosen_palette = palette if palette != "auto" else _default_palette_for(section_name, field)
    display_field = np.asarray(field, dtype=np.float32).transpose(1, 0)
    layout = _layout_from_unodes(unodes)
    chosen_label_flynns = settings.flynn_labels if label_flynns is None else bool(label_flynns)

    value_min = float(np.min(display_field)) if settings.vmin is None else float(settings.vmin)
    value_max = float(np.max(display_field)) if settings.vmax is None else float(settings.vmax)

    return {
        "title": Path(elle_path).name,
        "elle_path": str(Path(elle_path)),
        "attribute": section_name,
        "palette": chosen_palette,
        "grid_shape": [int(grid_shape[0]), int(grid_shape[1])],
        "image_shape": [int(display_field.shape[0]), int(display_field.shape[1])],
        "scale": int(scale),
        "legend": bool(legend and chosen_palette in {"gray", "heat"}),
        "overlay_boundaries": bool(overlay_boundaries and bool(flynns)),
        "flynn_labels": bool(chosen_label_flynns and bool(flynns)),
        "num_flynns": int(len(flynns)),
        "value_range": [float(np.min(display_field)), float(np.max(display_field))],
        "display_range": [value_min, value_max],
        "x_range": [float(layout["x_range"][0]), float(layout["x_range"][1])],
        "y_range": [float(layout["y_range"][0]), float(layout["y_range"][1])],
        "field": np.round(display_field, 6).tolist(),
        "flynns": _serialize_flynns(flynns, nodes),
    }


def _build_viewer_html(payload: dict[str, Any]) -> str:
    title = html.escape(payload["title"])
    data_json = json.dumps(payload, separators=(",", ":"))
    return f"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>{title} Viewer</title>
  <style>
    :root {{
      --bg: #f2efe7;
      --panel: #fffaf0;
      --ink: #1d1a16;
      --muted: #6c6458;
      --accent: #b0552b;
      --accent-soft: #ead8c4;
      --edge: #d4c7b8;
    }}
    * {{ box-sizing: border-box; }}
    body {{
      margin: 0;
      font-family: "Iowan Old Style", "Palatino Linotype", "Book Antiqua", serif;
      color: var(--ink);
      background:
        radial-gradient(circle at top right, #f7e7d3 0%, transparent 32%),
        linear-gradient(180deg, #f7f3ed 0%, var(--bg) 100%);
      min-height: 100vh;
    }}
    .layout {{
      display: grid;
      grid-template-columns: minmax(280px, 340px) minmax(0, 1fr);
      gap: 20px;
      padding: 20px;
      min-height: 100vh;
    }}
    .panel {{
      background: color-mix(in srgb, var(--panel) 90%, white);
      border: 1px solid var(--edge);
      border-radius: 18px;
      box-shadow: 0 14px 28px rgba(77, 60, 39, 0.08);
    }}
    .sidebar {{
      padding: 22px 20px;
      display: flex;
      flex-direction: column;
      gap: 18px;
    }}
    .eyebrow {{
      font-size: 12px;
      letter-spacing: 0.16em;
      text-transform: uppercase;
      color: var(--muted);
      margin-bottom: 10px;
    }}
    h1 {{
      margin: 0;
      font-size: 28px;
      line-height: 1.05;
      max-width: 12ch;
    }}
    .subtle {{
      color: var(--muted);
      font-size: 14px;
      line-height: 1.45;
      word-break: break-word;
    }}
    .controls {{
      display: grid;
      gap: 12px;
    }}
    .field {{
      display: grid;
      gap: 6px;
    }}
    label {{
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
      color: var(--muted);
    }}
    select,
    input[type="range"] {{
      width: 100%;
      accent-color: var(--accent);
    }}
    select {{
      padding: 9px 10px;
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
      background: linear-gradient(180deg, rgba(255,255,255,0.75), rgba(234,216,196,0.6));
      border-radius: 14px;
      border: 1px solid var(--edge);
    }}
    .checks label {{
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 12px;
      font-size: 13px;
      letter-spacing: 0.04em;
      text-transform: uppercase;
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
      min-width: 0;
      display: grid;
      gap: 14px;
      padding: 18px;
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
      background: var(--accent-soft);
      color: #7a3212;
      font-size: 13px;
      letter-spacing: 0.05em;
      text-transform: uppercase;
    }}
    .viewer-wrap {{
      min-height: 0;
      background:
        linear-gradient(45deg, rgba(255,255,255,0.86), rgba(243,235,226,0.92)),
        repeating-linear-gradient(
          45deg,
          rgba(176,85,43,0.04) 0px,
          rgba(176,85,43,0.04) 14px,
          rgba(255,255,255,0.08) 14px,
          rgba(255,255,255,0.08) 28px
        );
      border-radius: 18px;
      border: 1px solid var(--edge);
      overflow: auto;
      padding: 18px;
      display: grid;
      place-items: start;
    }}
    #viewerCanvas {{
      image-rendering: pixelated;
      image-rendering: crisp-edges;
      border-radius: 8px;
      box-shadow: 0 10px 22px rgba(53, 42, 28, 0.18);
      background: white;
    }}
    #legendCanvas {{
      border-radius: 10px;
      border: 1px solid var(--edge);
      background: white;
    }}
    .foot {{
      display: flex;
      justify-content: space-between;
      align-items: flex-start;
      gap: 18px;
      flex-wrap: wrap;
      color: var(--muted);
      font-size: 13px;
    }}
    .status {{
      font-family: "SFMono-Regular", "Cascadia Code", "Fira Code", monospace;
      font-size: 12px;
      color: #5d4636;
      background: rgba(255,255,255,0.8);
      border: 1px solid var(--edge);
      border-radius: 999px;
      padding: 8px 12px;
    }}
    @media (max-width: 980px) {{
      .layout {{
        grid-template-columns: 1fr;
      }}
      h1 {{
        max-width: none;
      }}
      .stage {{
        min-height: 70vh;
      }}
    }}
  </style>
</head>
<body>
  <div class="layout">
    <aside class="panel sidebar">
      <div>
        <div class="eyebrow">ELLE HTML Viewer</div>
        <h1>{title}</h1>
        <p class="subtle" id="pathLabel"></p>
      </div>

      <div class="controls">
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
      </div>

      <div class="meta">
        <div class="meta-card">
          <div class="meta-key">Attribute</div>
          <div class="meta-value" id="attributeValue"></div>
        </div>
        <div class="meta-card">
          <div class="meta-key">Grid</div>
          <div class="meta-value" id="gridValue"></div>
        </div>
        <div class="meta-card">
          <div class="meta-key">Flynns</div>
          <div class="meta-value" id="flynnValue"></div>
        </div>
        <div class="meta-card">
          <div class="meta-key">Range</div>
          <div class="meta-value" id="rangeValue"></div>
        </div>
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
        <div class="subtle">
          Scroll to pan when zoomed in. The browser handles panning naturally, so the viewer stays lightweight and portable.
        </div>
      </div>
    </main>
  </div>

  <script>
    const viewerData = {data_json};
    const width = viewerData.grid_shape[0];
    const height = viewerData.grid_shape[1];
    const field = viewerData.field;
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
      const g = Math.round(255 * t);
      return [g, g, g];
    }}

    function mapPoint(point) {{
      const [x, y] = point;
      const xSpan = Math.max(1e-12, viewerData.x_range[1] - viewerData.x_range[0]);
      const ySpan = Math.max(1e-12, viewerData.y_range[1] - viewerData.y_range[0]);
      return [
        ((x - viewerData.x_range[0]) / xSpan) * (width - 1),
        ((y - viewerData.y_range[0]) / ySpan) * (height - 1),
      ];
    }}

    function renderLegend() {{
      if (!(state.legend && (state.palette === "gray" || state.palette === "heat"))) {{
        legendCanvas.style.display = "none";
        return;
      }}
      legendCanvas.style.display = "block";
      const w = legendCanvas.width;
      const h = legendCanvas.height;
      legendCtx.clearRect(0, 0, w, h);
      for (let y = 0; y < h; y += 1) {{
        const t = 1 - y / Math.max(1, h - 1);
        const value = viewerData.display_range[0] + t * (viewerData.display_range[1] - viewerData.display_range[0]);
        const [r, g, b] = scalarColor(value, state.palette);
        legendCtx.fillStyle = `rgb(${{r}}, ${{g}}, ${{b}})`;
        legendCtx.fillRect(12, y, 22, 1);
      }}
      legendCtx.strokeStyle = "#3d3328";
      legendCtx.lineWidth = 1;
      legendCtx.strokeRect(11.5, 0.5, 23, h - 1);
      legendCtx.fillStyle = "#3d3328";
      legendCtx.font = "12px monospace";
      legendCtx.fillText(viewerData.display_range[1].toExponential(2), 42, 14);
      legendCtx.fillText(viewerData.display_range[0].toExponential(2), 42, h - 6);
    }}

    function render() {{
      const scale = Math.max(1, Number(state.scale) || 1);
      canvas.width = width * scale;
      canvas.height = height * scale;
      ctx.imageSmoothingEnabled = false;

      const baseCanvas = document.createElement("canvas");
      baseCanvas.width = width;
      baseCanvas.height = height;
      const baseCtx = baseCanvas.getContext("2d");
      const imageData = baseCtx.createImageData(width, height);
      const pixels = imageData.data;
      let offset = 0;

      for (let y = 0; y < height; y += 1) {{
        for (let x = 0; x < width; x += 1) {{
          const value = field[y][x];
          const [r, g, b] = state.palette === "labels"
            ? labelColor(value)
            : scalarColor(value, state.palette);
          pixels[offset] = r;
          pixels[offset + 1] = g;
          pixels[offset + 2] = b;
          pixels[offset + 3] = 255;
          offset += 4;
        }}
      }}

      baseCtx.putImageData(imageData, 0, 0);
      ctx.clearRect(0, 0, canvas.width, canvas.height);
      ctx.drawImage(baseCanvas, 0, 0, canvas.width, canvas.height);

      if (state.boundaries) {{
        ctx.strokeStyle = state.palette === "gray" ? "#cf2e2e" : "#ffffff";
        ctx.lineWidth = Math.max(1, scale * 0.6);
        flynns.forEach((flynn) => {{
          ctx.beginPath();
          flynn.points.forEach((point, index) => {{
            const [px, py] = mapPoint(point);
            const drawX = px * scale + scale / 2;
            const drawY = py * scale + scale / 2;
            if (index === 0) {{
              ctx.moveTo(drawX, drawY);
            }} else {{
              ctx.lineTo(drawX, drawY);
            }}
          }});
          ctx.closePath();
          ctx.stroke();
        }});
      }}

      if (state.labels) {{
        ctx.textAlign = "center";
        ctx.textBaseline = "middle";
        ctx.font = `${{Math.max(10, scale * 3.5)}}px monospace`;
        flynns.forEach((flynn) => {{
          let sx = 0;
          let sy = 0;
          flynn.points.forEach((point) => {{
            const [px, py] = mapPoint(point);
            sx += px;
            sy += py;
          }});
          const cx = (sx / flynn.points.length) * scale + scale / 2;
          const cy = (sy / flynn.points.length) * scale + scale / 2;
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
      const value = field[y][x];
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


def write_elle_html_viewer(
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
    return _portable_write_viewer_bundle(
        elle_path,
        outpath=outpath,
        attribute=attribute,
        palette=palette,
        showelle_in=showelle_in,
        overlay_boundaries=overlay_boundaries,
        label_flynns=label_flynns,
        scale=scale,
        legend=legend,
        single_file=single_file,
    )
