# Legacy Reference Bundles

This folder holds compact golden-reference bundles extracted from original
old-solver ELLE outputs.

These bundles are meant for automated regression tests in the faithful branch.
They store hashes, counts, and compact field summaries rather than committing
large raw `.elle` checkpoints into the Python subtree.

The builder CLI is:

```bash
python python_jax_model/build_legacy_reference_bundle.py \
  --source-name old_solver_name \
  --checkpoint stage_name=/path/to/checkpoint.elle \
  --json-out python_jax_model/legacy_reference/testdata/reference.json
```

The comparison CLI is:

```bash
python python_jax_model/validate_legacy_reference_bundle.py \
  --reference-json python_jax_model/legacy_reference/testdata/reference.json \
  --checkpoint stage_name=/path/to/checkpoint.elle \
  --json-out python_jax_model/validation/reference_compare.json
```

For before/after stage parity, there is now a transition-level contract too:

```bash
python python_jax_model/build_legacy_reference_transition.py \
  --before /path/to/before.elle \
  --after /path/to/after.elle \
  --checkpoint-name recovery_stage0 \
  --json-out python_jax_model/legacy_reference/testdata/recovery_stage0_transition.json
```

```bash
python python_jax_model/validate_legacy_reference_transition.py \
  --reference-json python_jax_model/legacy_reference/testdata/recovery_stage0_transition.json \
  --before /path/to/candidate_before.elle \
  --after /path/to/candidate_after.elle \
  --json-out python_jax_model/validation/recovery_stage0_transition_compare.json
```

The transition contract is meant for stage-by-stage parity checks where full
snapshot hashes are too coarse. It records:

- label-field change fraction and changed-pixel hash
- before/after grain-count and mean-area summaries
- per-field changed-row counts
- mean/max absolute deltas
- dense delta hashes for each tracked field

For swept-unode fidelity work there is also a swept-site view in
[legacy_reference.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/legacy_reference.py):

- `extract_legacy_reference_swept_unode_transition(...)`
- `compare_legacy_reference_swept_unode_transition(...)`

That contract restricts comparison to unodes whose label ownership changed
between the before/after states, which makes it useful for checking legacy
Euler reassignment and density-reset behavior on swept sites.

The shipped old-solver pair
[fine_foam_step001.elle](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step001.elle)
to
[fine_foam_step002.elle](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step002.elle)
now exercises both swept `U_EULER_3` and swept `U_DISLOCDEN` directly through
that interface.

The first committed reference in this folder is:

- [fft_example_step0_reference.json](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/legacy_reference/testdata/fft_example_step0_reference.json)

It is extracted from:

- [inifft001.elle](/home/bz1229682991/research/Elle/newcode/elle/processes/fft/example/step0/inifft001.elle)

The first committed transition-level reference in this folder is:

- [fine_foam_outerstep_001_to_002_transition.json](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/legacy_reference/testdata/fine_foam_outerstep_001_to_002_transition.json)

It is extracted from the shipped old-solver outer-step pair:

- [fine_foam_step001.elle](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step001.elle)
- [fine_foam_step002.elle](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step002.elle)
