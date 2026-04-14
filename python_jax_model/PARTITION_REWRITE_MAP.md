# Partition Rewrite Map

This note tracks how the faithful NumPy branch maps to the original ELLE
`GBMUnodeUpdate.cc` mass-transfer path and what still needs revision.

## Source Functions

- Original source: [basecode/GBMUnodeUpdate.cc](/home/bz1229682991/research/Elle/newcode/elle/basecode/GBMUnodeUpdate.cc)
- Main routines:
  - `Partition(...)`
  - `RemoveSweptMass(...)`
  - `PartitionMass(...)`
  - `EnrichUnodes(...)`

## Current Python Mapping

- Faithful branch entrypoint:
  - [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
  - `_apply_segment_mass_partition(...)`
- Core helper:
  - [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
  - `_entry_partition_terms(...)`

## What Is Now Fairly Faithful

- Swept triangles are subdivided into increment records.
- Each increment carries separate:
  - `sweep_mask`
  - `enrich_mask`
  - `reassigned_mask`
- The ledger now keeps ELLE-style channels:
  - `sweep_weight_total_0 / 1 / 2`
  - `enrich_weight_total_0 / 1 / 2`
  - `swept_mass_0 / 1`
  - `enrich_mass_0 / 1`
- The helper computes and exposes:
  - `conc_s`, `conc_s1`
  - `conc_e`, `conc_e1`
  - `conc_b`, `conc_b_f`, `conc_s_f`, `conc_e_f`
  - `mass_chge_s`, `mass_chge_e`, `mass_chge_b`
  - `swept_area_frac`, `enrich_area_frac`
- `put_mass` is now applied back through enrich-support weights, which is much
  closer to the original `EnrichUnodes(...)` stage than the older generic
  candidate-cell redistribution.
- the truthful node-aware branch now uses a literal Python
  `PartitionMass(...)`-style node solve for final node concentration and total
  `put_mass`
- `U_CONC_*` and `N_CONC_*` now share one ledger, so node and unode updates are
  at least driven by the same increment bookkeeping.

## What Is Still Approximate

- `RemoveSweptMass(...)`:
  - Original ELLE removes mass using `sum_j Weights[j][i]` over the combined
    increment set.
  - Python currently removes mass pointwise from the swept side using aggregated
    swept-support weights. This is close in spirit, but not yet a literal
    `Weights[j][i]` matrix rewrite.

- `PartitionMass(...)`:
  - Original ELLE performs one node-level solve:
    - `mass = total_swept_mass + total_gb_mass`
    - `gb_mass_f = mass * (total_gb_area_f/2) * pc / (total_swept_area + total_gb_area_f/2)`
    - `put_mass[i] = total_put_mass * swept_area[i] / total_swept_area`
  - Python now uses that node-level solve in the truthful node-aware branch, but
    still keeps richer per-entry bookkeeping alongside it for diagnostics.

- `EnrichUnodes(...)`:
  - Original ELLE applies `p_mass[j] * Weights[j][i] / Total_Weights[j]` over
    the whole unode list.
  - Python now uses enrich-support weights directly, but still does so from the
    per-entry ledger rather than a single dense `Weights[j][i]` table.

- Reassigned-point handling:
  - The visible original code still has the reassigned-point exclusion branches
    commented out.
  - Python mirrors that structure, but it does not yet model any hidden library
    behavior beyond the visible source.

## Highest-Priority Revisions

1. Replace the current swept-mass removal with a denser ELLE-style
   `Weights[j][i]` accumulation over the combined unode list.

2. Keep comparing the literal node-level `PartitionMass(...)` helper against the
   richer per-entry assembly path, so the truthful branch stays aligned with
   both visible ELLE routines.

3. Refactor enrich redistribution to use explicit `p_mass[j]` arrays and
   `Total_Weights[j]`, so the Python code structurally matches
   `EnrichUnodes(...)` instead of only matching its effect.

4. Build a small regression harness around one synthetic node with three
   increments and compare:
   - swept mass totals
   - node concentration
   - per-increment `put_mass`
   between ELLE formulas and Python formulas.

## Lower-Priority Revisions

- Decide whether to preserve the richer Python ledger as debug output even after
  a stricter ELLE-style solver is added.
- Separate the “faithful mesh-only branch” tests from the older calibrated
  analogue branch more clearly.
- Add a benchmark using a real `U_CONC_*` active ELLE case, since `fine_foam`
  mostly validates geometry and labels rather than mass transfer.

## Current Best Reading Order

1. [basecode/GBMUnodeUpdate.cc](/home/bz1229682991/research/Elle/newcode/elle/basecode/GBMUnodeUpdate.cc)
2. [elle_jax_model/mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py)
3. [tests/test_simulation.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/tests/test_simulation.py)
