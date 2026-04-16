# Legacy Solver Spec

This document is the compact source-to-rewrite map for the faithful GBM branch.
It exists so each fidelity change can point to one old-code contract instead of
relying on memory or proxy behavior.

## Core Contract

The faithful path is defined by the original ELLE + FFT solver chain:

- GBM driver:
  - [FS_gbm_pp_fft.main.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.main.cc)
  - [FS_gbm_pp_fft.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.elle.cc)
- node motion:
  - [FS_movenode_pp_fft.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc)
- topology repair:
  - [FS_topocheck_builtin.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_topocheck_builtin.cc)
- swept-unode update:
  - [GBMUnodeUpdate.cc](/home/bz1229682991/research/Elle/newcode/elle/basecode/GBMUnodeUpdate.cc)
- recovery:
  - [FS_recovery.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_recrystallisation/FS_recovery/FS_recovery.elle.cc)
- FFT bridges:
  - [FS_elle2fft.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_elle2fft/FS_elle2fft.cc)
  - [FS_fft2elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_fft2elle/FS_fft2elle.cc)

## Behavior Map

| Faithful behavior | Old source contract | Python target |
| --- | --- | --- |
| ELLE seed loading for flynns, bnodes, unodes, and options | `basecode/file.cc`, `FS_gbm_pp_fft.elle.cc` | [faithful_config.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_config.py), [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py) |
| Faithful stage loop | `GBMGrowth()` and outer driver in `FS_gbm_pp_fft.elle.cc` | [gbm_faithful.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/gbm_faithful.py), [faithful_runtime.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/faithful_runtime.py) |
| Randomized node motion with immediate topology repair | `FS_movenode_pp_fft.cc`, `FS_topocheck_builtin.cc` | [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py) |
| Mobility law from phase pairs, temperature, and misorientation | `phase_db.txt`, `GetBoundaryMobility`, `Get2BoundaryMobility` | pending faithful mobility module |
| Swept-unode ownership, `PartitionMass`, and `EnrichUnodes` | `GBMUnodeUpdate.cc` | [mesh.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/mesh.py) |
| Recovery and lattice-orientation reset logic | `FS_recovery.elle.cc`, `symmetry.symm` | pending faithful recovery module |
| FFT deformation import/export and stress/slip transfer | `FS_elle2fft.cc`, `FS_fft2elle.cc` | pending faithful FFT bridge module |
| Snapshot comparison against old code | old checkpoint `.elle` files and side outputs | [legacy_reference.py](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/elle_jax_model/legacy_reference.py) |

## Reference Comparison

The first committed fixture is now paired with a comparison CLI:

```bash
python python_jax_model/validate_legacy_reference_bundle.py \
  --reference-json python_jax_model/legacy_reference/testdata/fft_example_step0_reference.json \
  --checkpoint step0=/path/to/candidate.elle \
  --json-out python_jax_model/validation/reference_compare.json
```

This gives the faithful branch a stable yes/no contract plus focused diffs for:

- mesh hashes and counts
- label-grid hashes and grain statistics
- compact field summaries and dense value hashes

There is now also a transition-level contract for faithful stage parity:

```bash
python python_jax_model/build_legacy_reference_transition.py \
  --before /path/to/legacy_before.elle \
  --after /path/to/legacy_after.elle \
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

This transition contract is the right tool for direct GBM-stage and
recovery-stage parity work because it records how a snapshot changes, not just
what the endpoints look like in isolation.

## Reference Bundle Contract

The committed legacy bundle format should include, for each checkpoint:

- mesh summary:
  - node count
  - flynn count
  - node-position hash
  - flynn-connectivity hash
- label summary:
  - label source
  - grid shape
  - label hash
  - grain count
  - mean/std grain area
  - grain-area hash
- field summaries:
  - section name
  - entity type (`unode`, `node`, or `flynn`)
  - component count
  - default value
  - explicit count
  - component statistics
  - dense value hash

This is intentionally compact. It is not a full archival copy of the old output.
It is the testable contract used to detect regressions while the faithful branch
is being ported.

## First Committed Reference

The first compact committed old-code reference case is:

- [fft_example_step0_reference.json](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/legacy_reference/testdata/fft_example_step0_reference.json)

It is derived from the original old solver output:

- [inifft001.elle](/home/bz1229682991/research/Elle/newcode/elle/processes/fft/example/step0/inifft001.elle)

This case is useful because it already contains:

- flynns and node geometry
- unodes
- `U_EULER_3`
- `U_DISLOCDEN`
- imported `U_ATTRIB_*` fields

That makes it a good first anchor for the faithful GBM branch even before the
full deformation -> GBM -> recovery -> outer-loop checkpoint chain is scripted.

## First Committed Transition Reference

The first compact committed old-code transition case is:

- [fine_foam_outerstep_001_to_002_transition.json](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/legacy_reference/testdata/fine_foam_outerstep_001_to_002_transition.json)

It is derived from the shipped old solver outer-step pair:

- [fine_foam_step001.elle](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step001.elle)
- [fine_foam_step002.elle](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step002.elle)

This is not a pure recovery-stage fixture yet, but it is the first real old
before/after transition contract for the `fine_foam` workflow. It gives the
faithful branch a direct way to compare how one saved outer step changes:

- unode grain ownership
- orientation and density fields
- flynn and node field counts/shapes
- the overall changed-pixel fraction in `U_ATTRIB_C`
