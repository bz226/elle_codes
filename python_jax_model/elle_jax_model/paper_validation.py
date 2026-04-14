from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .microstructure_validation import (
    collect_elle_microstructure_snapshots,
    summarize_liu_suckale_datasets,
)

_KNOWN_ICE_CORE_SITES = ("GISP2", "GRIP", "EGRIP", "EDML", "Siple Dome")


def _collapse_whitespace(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def _extract_pdf_text(path: str | Path) -> dict[str, Any]:
    try:
        from pypdf import PdfReader
    except ImportError as exc:  # pragma: no cover - exercised in CLI/runtime
        raise ImportError(
            "pypdf is required for paper extraction. Install it with "
            "`pip install pypdf` in the project environment."
        ) from exc

    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    page_texts = [_collapse_whitespace(page.extract_text() or "") for page in reader.pages]
    metadata = reader.metadata or {}
    return {
        "path": str(pdf_path),
        "pages": len(page_texts),
        "title": metadata.get("/Title") or pdf_path.stem,
        "author": metadata.get("/Author"),
        "page_texts": page_texts,
        "text": "\n".join(page_texts),
    }


def _first_snippet(text: str, queries: list[str], *, context: int = 260) -> str | None:
    lowered = text.lower()
    for query in queries:
        index = lowered.find(query.lower())
        if index >= 0:
            start = max(0, index - context)
            end = min(len(text), index + len(query) + context)
            return _collapse_whitespace(text[start:end])
    return None


def _extract_sample_counts(text: str) -> dict[str, int] | None:
    match = re.search(
        r"dataset contains\s+([\d,]+)\s+training samples,\s+([\d,]+)\s+validation samples,\s+and\s+([\d,]+)\s+testing samples",
        text,
        re.IGNORECASE,
    )
    if match is None:
        return None
    return {
        "training": int(match.group(1).replace(",", "")),
        "validation": int(match.group(2).replace(",", "")),
        "testing": int(match.group(3).replace(",", "")),
    }


def summarize_llorens_structure_from_text(text: str, *, title: str | None = None) -> dict[str, Any]:
    dual_layer = _first_snippet(text, ["flynns", "bnodes", "unodes", "two basic layers"])
    fft_elle = _first_snippet(text, ["FFT code", "ELLE", "stress and velocity fields"])
    gbm = _first_snippet(
        text,
        ["intracrystalline recovery", "GBM", "surface energy", "stored strain energy"],
    )
    return {
        "title": title or "Llorens et al. structure paper",
        "dual_layer_model": {
            "present": dual_layer is not None,
            "snippet": dual_layer,
            "expects": [
                "flynns boundary polygons",
                "bnodes boundary nodes",
                "unodes regular-grid material state",
            ],
        },
        "fft_elle_coupling": {
            "present": fft_elle is not None,
            "snippet": fft_elle,
            "expects": [
                "FFT deformation fields",
                "ELLE recrystallization/topology evolution",
                "stress and velocity transfer between scales",
            ],
        },
        "recrystallization_drivers": {
            "present": gbm is not None,
            "snippet": gbm,
            "expects": [
                "intracrystalline recovery",
                "grain-boundary migration",
                "surface-energy and stored-strain-energy forcing",
            ],
        },
    }


def summarize_llorens_structure(pdf_path: str | Path) -> dict[str, Any]:
    extracted = _extract_pdf_text(pdf_path)
    summary = summarize_llorens_structure_from_text(extracted["text"], title=extracted["title"])
    summary["path"] = extracted["path"]
    summary["pages"] = extracted["pages"]
    summary["author"] = extracted["author"]
    return summary


def summarize_liu_suckale_paper_from_text(text: str, *, title: str | None = None) -> dict[str, Any]:
    static_snippet = _first_snippet(
        text,
        ["Fan et al.", "grain size data were collected at seven time points", "Figure 2"],
    )
    dynamic_snippet = _first_snippet(
        text,
        ["synthetic ice in simple shear", "Qi et al.", "sweep angle"],
    )
    fno_grain_snippet = _first_snippet(
        text,
        ["11,932 training samples", "days 3.6, 60, 120, and 192", "mean grain area predicted by the FNO"],
    )
    fno_euler_snippet = _first_snippet(
        text,
        ["Fig. 5", "all three Euler angles", "day 97"],
    )
    vertical_column_snippet = _first_snippet(
        text,
        ["one-dimensional vertical ice column", "5.1. Coupling Introduces Depth Variability in Ice Properties"],
    )
    ice_core_snippet = _first_snippet(
        text,
        ["five ice-cores", "GISP2", "GRIP", "EGRIP", "EDML", "Siple Dome"],
    )
    sample_counts = _extract_sample_counts(text)
    ice_core_sites = [site for site in _KNOWN_ICE_CORE_SITES if site.lower() in text.lower()]

    return {
        "title": title or "Liu and Suckale multiscale paper",
        "microscale_benchmarks": [
            {
                "name": "static_recrystallization_high_temperature",
                "status": "detected" if static_snippet is not None else "missing",
                "target_metrics": [
                    "grain area distribution",
                    "grain area KDE",
                    "mean grain area vs time",
                ],
                "snippet": static_snippet,
            },
            {
                "name": "dynamic_recrystallization_simple_shear",
                "status": "detected" if dynamic_snippet is not None else "missing",
                "target_metrics": [
                    "c-axis pole-figure structure",
                    "sweep-angle histogram",
                    "temperature dependence of fabric evolution",
                ],
                "snippet": dynamic_snippet,
            },
        ],
        "surrogate_benchmarks": [
            {
                "name": "grain_size_fno",
                "status": "detected" if fno_grain_snippet is not None else "missing",
                "target_metrics": [
                    "grain area distribution snapshots",
                    "mean grain area scatter against microscale model",
                ],
                "sample_counts": sample_counts,
                "snippet": fno_grain_snippet,
            },
            {
                "name": "euler_angle_fno",
                "status": "detected" if fno_euler_snippet is not None else "missing",
                "target_metrics": [
                    "spatial Euler-angle fields",
                    "feature-level fabric evolution",
                ],
                "sample_counts": sample_counts,
                "snippet": fno_euler_snippet,
            },
        ],
        "macro_benchmarks": [
            {
                "name": "vertical_ice_column_coupling",
                "status": "detected" if vertical_column_snippet is not None else "missing",
                "target_metrics": [
                    "depth-dependent strain rate",
                    "depth-dependent stress",
                    "coupled vs uncoupled sensitivity",
                ],
                "snippet": vertical_column_snippet,
            },
            {
                "name": "five_ice_core_grain_size_profiles",
                "status": "detected" if ice_core_snippet is not None else "missing",
                "target_metrics": [
                    "grain size vs depth",
                    "site-by-site mismatch identification",
                ],
                "sites": ice_core_sites,
                "snippet": ice_core_snippet,
            },
        ],
    }


def summarize_liu_suckale_paper(pdf_path: str | Path) -> dict[str, Any]:
    extracted = _extract_pdf_text(pdf_path)
    summary = summarize_liu_suckale_paper_from_text(extracted["text"], title=extracted["title"])
    summary["path"] = extracted["path"]
    summary["pages"] = extracted["pages"]
    summary["author"] = extracted["author"]
    return summary


def assess_current_rewrite_against_papers() -> dict[str, Any]:
    return {
        "llorens_alignment": [
            {
                "target": "ELLE dual-layer flynns/bnodes/unodes structure",
                "status": "implemented",
                "evidence": [
                    "python_jax_model/elle_jax_model/elle_export.py",
                    "python_jax_model/elle_jax_model/topology.py",
                    "python_jax_model/elle_jax_model/elle_visualize.py",
                ],
                "notes": "The rewrite can export, track, and view ELLE-style flynn and unode state.",
            },
            {
                "target": "FFT deformation fields coupled directly into ELLE recrystallization",
                "status": "missing",
                "evidence": [
                    "TwoWayIceModel_Release/elle/example/launch.sh",
                    "llorens2016jg.pdf",
                ],
                "notes": "The rewrite does not yet reproduce the original FFT-driven mechanics pipeline.",
            },
            {
                "target": "Recovery + GBM driven by surface and stored strain energy",
                "status": "partial",
                "evidence": [
                    "python_jax_model/elle_jax_model/mesh.py",
                    "python_jax_model/elle_jax_model/simulation.py",
                ],
                "notes": "There is topology maintenance and mesh feedback, but not the full original ELLE constitutive workflow.",
            },
        ],
        "liu_suckale_alignment": [
            {
                "target": "Static grain-growth benchmark metrics",
                "status": "partial",
                "evidence": [
                    "python_jax_model/elle_jax_model/microstructure_validation.py",
                    "python_jax_model/elle_jax_model/phasefield_compare.py",
                ],
                "notes": "We can measure grain area distributions and mean grain size, but have not yet reproduced Fan et al. directly.",
            },
            {
                "target": "Dynamic fabric benchmark metrics",
                "status": "partial",
                "evidence": [
                    "python_jax_model/elle_jax_model/microstructure_validation.py",
                ],
                "notes": "Orientation histograms exist, but we have not yet recreated the Qi et al. simple-shear benchmark.",
            },
            {
                "target": "FNO release-example reproduction",
                "status": "implemented_for_release_examples",
                "evidence": [
                    "TwoWayIceModel_Release/data/README.md",
                    "TwoWayIceModel_Release/example/train_grainsize.py",
                    "TwoWayIceModel_Release/example/train_euler.py",
                ],
                "notes": "The supplement data and training scripts are available locally and can be rerun.",
            },
            {
                "target": "Vertical-column multiscale coupling",
                "status": "missing",
                "evidence": [
                    "JCP__Grain_size___Emma_Liu__Copy_ (1).pdf",
                ],
                "notes": "Our rewrite does not yet include the macroscale Stokes-flow coupling used in the paper.",
            },
            {
                "target": "Five-site ice-core grain-size profile comparison",
                "status": "missing",
                "evidence": [
                    "JCP__Grain_size___Emma_Liu__Copy_ (1).pdf",
                ],
                "notes": "We do not yet generate or compare depth-resolved ice-core grain-size profiles.",
            },
        ],
    }


def build_paper_validation_report(
    *,
    llorens_pdf: str | Path,
    liu_pdf: str | Path,
    reference_dir: str | Path | None = None,
    data_dir: str | Path | None = None,
    pattern: str = "*.elle",
) -> dict[str, Any]:
    report: dict[str, Any] = {
        "llorens_structure": summarize_llorens_structure(llorens_pdf),
        "liu_suckale_paper": summarize_liu_suckale_paper(liu_pdf),
        "rewrite_assessment": assess_current_rewrite_against_papers(),
    }

    if reference_dir is not None:
        report["reference_sequence"] = collect_elle_microstructure_snapshots(reference_dir, pattern=pattern)
    if data_dir is not None:
        report["liu_suckale_datasets"] = summarize_liu_suckale_datasets(data_dir)

    return report


def write_paper_validation_report(path: str | Path, report: dict[str, Any]) -> Path:
    outpath = Path(path)
    outpath.parent.mkdir(parents=True, exist_ok=True)
    outpath.write_text(json.dumps(report, indent=2, sort_keys=True), encoding="utf-8")
    return outpath
