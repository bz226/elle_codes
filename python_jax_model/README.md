# ELLE Python/JAX Prototype Rewrite

This folder contains a **clean-room Python rewrite prototype** inspired by ELLE's grain-growth style process drivers (for example `processes/growth/gg.main.cc`).

It does **not** modify or replace the original C/C++ codebase. Instead, it demonstrates a modern, vectorized modeling path using JAX.

## What this prototype includes

- A small 2D phase-field grain-growth simulator in JAX (with NumPy fallback if JAX is unavailable).
- A reproducible random initial-condition generator.
- A simulation runner that can export snapshots to `.npy` files.
- Minimal unit tests for deterministic setup and shape checks.

## Why this is a practical migration path

A full rewrite of the entire ELLE ecosystem is large, but **yes, it is possible** to incrementally rewrite process-by-process.
This prototype shows one such process-level migration:

1. Keep old C/C++ binaries untouched.
2. Reimplement one physics kernel in Python/JAX.
3. Validate behavior against known ELLE outputs.
4. Expand coverage process-by-process.

## Quick start

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r python_jax_model/requirements.txt
python python_jax_model/run_simulation.py --steps 300 --save-every 60 --outdir python_jax_model/output
```

## Output

The runner saves `order_parameter_XXXXX.npy` files where each file is a `(num_grains, nx, ny)` tensor.
You can post-process these arrays for analysis or visualization.
