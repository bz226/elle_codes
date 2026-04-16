# Original ELLE Learning Guide

This note is a practical reading guide for the original ELLE codebase in:

- `../basecode`
- `../processes`
- `../FS_Codes`

It is meant to answer:

1. What are the main parts of the original code?
2. In what order should you read them?
3. What does each important file mean in the `fine_foam` / GBM workflow?

This guide is intentionally biased toward the recrystallization / grain-boundary
migration path, because that is the path we are trying to rewrite faithfully.

For a direct parameter-by-parameter mapping between the Python rewrite and
original ELLE GBM, see
[ORIGINAL_ELLE_GBM_PARAMETER_MAP.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/ORIGINAL_ELLE_GBM_PARAMETER_MAP.md).
For a workflow-based explanation of `U_ATTRIB_*` meanings, see
[ELLE_UNODE_ATTRIBUTE_GUIDE.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/ELLE_UNODE_ATTRIBUTE_GUIDE.md).

## 1. The Core Mental Model

The original ELLE code is easiest to understand if you treat it as two coupled
representations:

- Boundary-network representation:
  - flynns
  - boundary nodes
  - topology changes
  - grain-boundary geometry
- Material-state representation:
  - unodes
  - scalar attributes
  - concentrations
  - per-point material state

For the GBM-style workflow, the basic sequence is:

1. get driving information
2. move boundary nodes
3. repair topology
4. update unodes from the moved boundaries

That is the most important thing to keep in mind. The original GBM path is not
primarily a dense PDE solver. It is a moving-boundary / topology workflow with
unode updates layered onto it.

## 2. The Best Reading Order

If your goal is understanding the original `fine_foam` / GBM chain, read in
this order.

### Pass 1: Workflow First

Start here:

1. `../FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.elle.cc`
2. `../FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc`
3. `../FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc`
4. `../basecode/GBMUnodeUpdate.cc`

Why this order:

- `FS_gbm_pp_fft.elle.cc` shows the high-level process flow.
- `FS_movenode_pp_fft.cc` shows how nodes actually move.
- `FS_topocheck.elle.cc` shows how bad geometry/topology is repaired.
- `GBMUnodeUpdate.cc` shows how unodes are reassigned after boundaries move.

If you only read four files first, read those.

### Pass 2: Data Structures

Then read the ELLE core data-model files:

1. `../basecode/flynnarray.h`
2. `../basecode/flynnarray.cc`
3. `../basecode/flynns.cc`
4. `../basecode/nodes.cc`
5. `../basecode/regions.cc`
6. `../basecode/unodes.cc`
7. `../basecode/file.cc`

Why:

- these files define what a flynn, node, region, and unode actually are
- they tell you what the process files are manipulating

### Pass 3: Supporting Math and Runtime

Then read:

1. `../basecode/general.cc`
2. `../basecode/general.h`
3. `../basecode/parseopts.c`
4. `../basecode/runopts.cc`
5. `../basecode/attribute.cc`
6. `../basecode/attribarray.cc`

Why:

- these files explain shared helper logic
- runtime options
- attribute storage
- geometry utilities that the process code depends on

### Pass 4: FFT Coupling

Only after the above, read the FFT bridge:

1. `../FS_Codes/FS_elle2fft/FS_elle2fft.cc`
2. `../FS_Codes/FS_fft2elle/FS_fft2elle.cc`
3. `../FS_Codes/FS_fft2elle_strainanalysis/FS_fft2elle_strainanalysis.cc`

Why:

- these explain how ELLE exchanges information with the external FFT solver
- they matter for full physical fidelity, but they are not the right entry
  point for learning the codebase

## 3. What The Important Files Mean

This section is the quickest “what am I looking at?” reference.

### `FS_gbm_pp_fft.elle.cc`

Role:

- the main GBM process driver

What to look for:

- process setup
- reading runtime parameters
- stage ordering
- calls into node motion
- calls into topology maintenance
- calls into unode updates

What it means:

- this file is the process orchestrator
- if you want the overall logic, this is the most important file

### `FS_movenode_pp_fft.cc`

Role:

- the actual node-motion law

What to look for:

- `GGMoveNode_all`
- `GetMoveDir`
- `MoveDNode`
- `MoveTNode`
- trial energy sampling around a node
- mobility-weighted denominators
- timestep clamping

What it means:

- this file translates driving force into actual node displacement
- if you want to know “why does a boundary move this way?”, this is the file

### `FS_topocheck.elle.cc`

Role:

- geometry and topology cleanup

What to look for:

- deletion of small flynns
- split checks
- angle checks
- coincident-node handling
- node-spacing maintenance

What it means:

- ELLE does not assume the mesh stays healthy by itself
- this file is the guardrail that keeps the boundary network valid

### `GBMUnodeUpdate.cc`

Role:

- transfer from moved boundaries back to unodes and node fields

What to look for:

- swept-region logic
- `Partition`
- `PartitionMass`
- `EnrichUnodes`
- reassigned / enrich / swept categories
- concentration and mass bookkeeping

What it means:

- this is the bridge between boundary motion and the material field
- for faithful rewriting, this is one of the hardest and most important files

### `flynnarray.*`, `flynns.cc`, `regions.cc`

Role:

- flynn storage and region relationships

What to look for:

- how flynns are indexed
- how neighbors are discovered
- how node cycles define flynns

What it means:

- these files define the grain topology data model

### `nodes.cc`

Role:

- node-level storage and adjacency behavior

What to look for:

- neighbor lookups
- node activity
- triple vs double node logic
- node position access

What it means:

- if a process refers to node neighbors, active nodes, or node plotting, it is
  relying on this layer

### `unodes.cc`

Role:

- unode storage and lookup

What to look for:

- how unodes are stored
- how attributes are attached
- how unodes are associated with grains/regions

What it means:

- this is the regular-grid material-state side of ELLE

### `file.cc`

Role:

- ELLE file read/write behavior

What to look for:

- how `UNODES`, `FLYNNS`, `LOCATION`, and attributes are serialized

What it means:

- use this when you need to understand exact file semantics rather than process
  semantics

### `general.cc`

Role:

- shared geometry / math / utility helpers

What to look for:

- vector math
- geometry utilities
- periodic plotting helpers
- generic support functions used across processes

What it means:

- this file is important, but not the best entry point
- read it after you already understand the workflow

### `parseopts.c`

Role:

- command-line option parsing and process/runtime configuration

What to look for:

- which runtime flags exist
- how options get attached to a process

What it means:

- useful for understanding setup and reproducibility
- not the file to study first if your question is about the physics

## 4. The `fine_foam` / GBM Workflow Map

If you want to mentally trace one timestep of the original GBM chain, this is
the rough map:

1. state and options are loaded
2. GBM driver prepares process data
3. for each relevant node:
   - evaluate trial positions
   - compute surface / stored-energy tendencies
   - build a movement direction
   - clamp timestep if necessary
4. topology cleanup is applied:
   - merge/delete/split/switch/check geometry
5. unodes and node fields are updated from the moved boundaries
6. the new ELLE state is written or passed to the next process

Files involved:

- driver:
  - `../FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.elle.cc`
- motion:
  - `../FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc`
- topology:
  - `../FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc`
- unodes:
  - `../basecode/GBMUnodeUpdate.cc`

## 5. What To Ignore At First

These are important eventually, but they are not the right first stop if
you’re learning the GBM path:

- most viewer code
- most utility binaries
- unrelated processes under `../processes`
- generic build files
- MATLAB plotting scripts

You can come back to those later.

## 6. A Good Way To Read The Code

For each file, ask four questions:

1. Is this a driver, a physics rule, a topology rule, or infrastructure?
2. Does it primarily update nodes, flynns, or unodes?
3. Is it deciding energy, geometry, or ownership?
4. Is it ELLE-core behavior or process-specific behavior?

Those questions help separate “what ELLE is” from “what this specific process is
doing with ELLE.”

## 7. Recommended Study Sessions

### Session 1: Learn the overall flow

Read:

- `../FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.elle.cc`
- `../FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc`

Goal:

- understand how a GBM step is orchestrated

### Session 2: Learn topology maintenance

Read:

- `../FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc`
- `../basecode/nodes.cc`
- `../basecode/flynns.cc`

Goal:

- understand how ELLE keeps the boundary network valid

### Session 3: Learn unode transfer

Read:

- `../basecode/GBMUnodeUpdate.cc`
- `../basecode/unodes.cc`

Goal:

- understand how moved boundaries become updated unode state

### Session 4: Learn runtime and I/O

Read:

- `../basecode/file.cc`
- `../basecode/parseopts.c`
- `../basecode/general.cc`

Goal:

- understand how processes are configured and how ELLE files are interpreted

## 8. If You Only Want The Most Important Files

If you only have time for a short pass, read these five:

1. `../FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_gbm_pp_fft.elle.cc`
2. `../FS_Codes/FS_recrystallisation/FS_gbm_pp_fft/FS_movenode_pp_fft.cc`
3. `../FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc`
4. `../basecode/GBMUnodeUpdate.cc`
5. `../basecode/flynnarray.cc`

That set gives you:

- process flow
- motion
- topology
- unode transfer
- data structure basics

## 9. How This Relates To Our Rewrite

If you compare the original code to our Python rewrite:

- the closest direct process port is:
  - `elle_jax_model/elle_phasefield.py`
- the closest structural GBM rewrite branch is:
  - `elle_jax_model/mesh.py`
  - `run_truthful_numpy.py`
- the best current numerical `fine_foam` match is still the calibrated analogue
  branch:
  - `elle_jax_model/simulation.py`
  - `elle_jax_model/calibration.py`

So when reading original ELLE, the most useful mindset is:

- for process structure, compare against the truthful NumPy branch
- for benchmark closeness, compare against the calibrated branch

## 10. Suggested Next Companion Note

After this guide, the next most useful document would be a file-by-file
annotated checklist just for:

- `FS_gbm_pp_fft.elle.cc`
- `FS_movenode_pp_fft.cc`
- `FS_topocheck.elle.cc`
- `GBMUnodeUpdate.cc`

That would be the right next step if you want a more detailed study guide.
