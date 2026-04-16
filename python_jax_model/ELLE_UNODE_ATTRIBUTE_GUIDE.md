# ELLE Unode Attribute Guide

This note answers two related questions:

1. What do the unode attributes mean in the shipped `fine_foam` workflow?
2. How should we figure out the meaning of `U_ATTRIB_*` in a general ELLE
   workflow?

The short version is:

- `U_ATTRIB_A`, `U_ATTRIB_B`, `U_ATTRIB_C`, ... are generic storage slots.
- Their meaning is not fixed globally.
- You should identify their meaning from the workflow that wrote the file.

For the broader GBM reading order, see
[ORIGINAL_ELLE_LEARNING_GUIDE.md](/home/bz1229682991/research/Elle/newcode/elle/python_jax_model/ORIGINAL_ELLE_LEARNING_GUIDE.md).

## 1. Quick Answer For `fine_foam`

For the shipped `fine_foam` example, the important mappings are:

| Attribute | Meaning in `fine_foam` | Evidence |
| --- | --- | --- |
| `U_ATTRIB_A` | normalized strain rate | imported from `tex.out` column `4` by `importFFTdata_florian -u 4 5 6 7 12` |
| `U_ATTRIB_B` | normalized stress | same import step, column `5` |
| `U_ATTRIB_C` | flynn / grain ID of the unode | used by GBM/topocheck as the ownership field |
| `U_ATTRIB_D` | basal activity | same import step, column `6` |
| `U_ATTRIB_E` | prismatic activity | same import step, column `7` |
| `U_DISLOCDEN` | dislocation density | handled separately by FFT bridge code |

Main references:

- [launch.sh](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/launch.sh)
- [importFFTdata_florian/readme](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_importFFTdata_florian/readme)
- [importFFTdata_florian.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_importFFTdata_florian/importFFTdata_florian.cc)
- [FS_topocheck.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc)
- [FS_flynn2unode_attribute.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_flynn2unode_attribute/FS_flynn2unode_attribute.cc)

## 2. Why `U_ATTRIB_*` Names Are Tricky

ELLE has two kinds of attribute names:

- semantic names with a built-in intent:
  - `U_DISLOCDEN`
  - `U_EULER_3`
  - `U_VISCOSITY`
- generic slots:
  - `U_ATTRIB_A`
  - `U_ATTRIB_B`
  - `U_ATTRIB_C`
  - `U_ATTRIB_D`
  - `U_ATTRIB_E`
  - `U_ATTRIB_F`

The semantic names are usually stable.

The generic `U_ATTRIB_*` names are not. Different utilities use them for
different physics or bookkeeping. So the slot name alone is not enough.

Examples from this codebase:

- in `fine_foam`, `U_ATTRIB_A` is normalized strain rate
- in GBM/topocheck, `U_ATTRIB_C` is unode flynn ID
- in strain-analysis utilities, `U_ATTRIB_A` can mean a finite deformation
  tensor component or a derived scalar
- in some debug paths, `U_ATTRIB_A` or `U_ATTRIB_C` are just temporary markers

So the right question is not:

- "What does `U_ATTRIB_A` mean?"

The right question is:

- "What process wrote `U_ATTRIB_A` in this workflow?"

## 3. How To Determine Meaning In A General ELLE Workflow

This is the recommended order.

### Step 1: Read the launch script or process chain

Look at the exact order of processes that wrote the file.

For `fine_foam`, this is visible in
[launch.sh](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/launch.sh):

1. `FS_elle2fft`
2. `FFT_vs128`
3. `FS_fft2elle`
4. `importFFTdata_florian`
5. `FS_flynn2unode_attribute`
6. recovery / nucleation / GBM / topocheck loops

That already tells us that `U_ATTRIB_*` may have been touched by several
different processes.

### Step 2: Check the file header

The file header tells you what process wrote the final file, but not
necessarily which process first created every attribute.

Example:

- [fine_foam_step001.elle](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step001.elle)
  says it was created by `FS_flynn2unode_attribute`

That means `FS_flynn2unode_attribute` wrote the final file. It does not mean
every attribute in that file originated there.

### Step 3: Find where the attribute is written

Search for:

- `ElleSetUnodeAttribute(..., U_ATTRIB_A)`
- `ElleSetUnodeAttribute(..., U_ATTRIB_C)`
- `ElleInitUnodeAttribute(U_ATTRIB_A)`
- `ElleGetUnodeAttribute(..., U_ATTRIB_A)`

The key distinction is:

- `Init` only means "this slot exists"
- `Set` usually tells you the actual meaning

### Step 4: Read the process readme and user options

Many ELLE utilities map command-line user data directly onto attribute slots.

For `fine_foam`, [importFFTdata_florian/readme](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_importFFTdata_florian/readme)
is decisive:

- `A: Transfer to U_ATTRIB_A`
- `B: Transfer to U_ATTRIB_B`
- `C: Transfer to U_ATTRIB_D`
- `D: Transfer to U_ATTRIB_E`
- `E: Transfer to U_ATTRIB_F`

Then [launch.sh](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/launch.sh)
passes `-u 4 5 6 7 12`, so we know which `tex.out` columns are used.

### Step 5: Inspect the values in the `.elle` file

The numbers themselves are often a strong clue.

Examples:

- if values are integer-like and match flynn IDs, the field is probably an
  ownership label
- if values are continuous and look like stresses, rates, angles, or
  concentrations, the field is probably a scalar material field

In `fine_foam_step001.elle`:

- [U_ATTRIB_C](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step001.elle#L77464)
  has values like `126`, `129`, `4`
- [U_ATTRIB_A](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/results/fine_foam_step001.elle#L93772)
  has continuous values like `1.4665`, `0.7728`, `4.2793`

That is exactly what we would expect from:

- `U_ATTRIB_C` = grain ID
- `U_ATTRIB_A` = imported scalar field

### Step 6: Prefer workflow meaning over slot-name meaning

If two files disagree about what `U_ATTRIB_A` "means", the workflow wins.

The slot is generic.
The process that wrote it defines its physical meaning in that run.

## 4. The `fine_foam` Case, Step By Step

Here is the actual reasoning for `fine_foam`.

### `U_ATTRIB_C`

This one is clear:

- [FS_topocheck.elle.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_topocheck/FS_topocheck.elle.cc)
  says `U_ATTRIB_C` stores the ID of the flynn containing the unode
- [FS_flynn2unode_attribute.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_flynn2unode_attribute/FS_flynn2unode_attribute.cc)
  checks and writes flynn IDs into `U_ATTRIB_C`
- the values in the file are integer-like flynn IDs

So in `fine_foam`:

- `U_ATTRIB_C` = unode flynn / grain ownership

### `U_ATTRIB_A`

This one needed more tracing.

At first glance it looks ambiguous because:

- `FS_flynn2unode_attribute` can copy `F_ATTRIB_A -> U_ATTRIB_A`
- `FS_elle2fft` contains a line that writes `pts[i][6] -> U_ATTRIB_A`
- other utilities use `U_ATTRIB_A` for unrelated scalar fields

But the workflow resolves it:

1. [launch.sh](/home/bz1229682991/research/Elle/TwoWayIceModel_Release/elle/example/launch.sh)
   runs `importFFTdata_florian -u 4 5 6 7 12`
2. [importFFTdata_florian/readme](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_importFFTdata_florian/readme)
   says slot `A` maps to `U_ATTRIB_A`
3. [importFFTdata_florian.cc](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_importFFTdata_florian/importFFTdata_florian.cc)
   writes `val[opt_1]` into `U_ATTRIB_A`
4. in that same code, option `4` is documented as normalized strain rate
5. the same launch script later runs `FS_flynn2unode_attribute -u 1 -n`
6. [FS_flynn2unode_attribute/readme](/home/bz1229682991/research/Elle/newcode/elle/FS_Codes/FS_utilities/FS_flynn2unode_attribute/readme)
   shows `-u 1` only activates the first transfer slot, which is `VISCOSITY`,
   not `F_ATTRIB_A`

So the best reading is:

- `U_ATTRIB_A` in `fine_foam` = FFT-imported normalized strain rate

## 5. A Good Working Rule

When you see `U_ATTRIB_*` in ELLE, use this rule:

- treat it as a generic storage slot first
- then identify the last meaningful process that wrote it
- then confirm with the value pattern in the file

That approach is much safer than assuming:

- `U_ATTRIB_A` always means one specific physics quantity
- `U_ATTRIB_C` always means one specific physics quantity

Sometimes it will be true in one workflow, but not in another.

## 6. Practical Checklist

Use this checklist when decoding a new ELLE file:

1. Read the workflow script.
2. Read the file header.
3. Search for `ElleSetUnodeAttribute(..., U_ATTRIB_X)`.
4. Read the relevant process readmes.
5. Inspect the actual values in the `.elle` file.
6. Prefer process-specific meaning over generic slot-name intuition.

If two competing interpretations remain, trust:

1. the launch script
2. the writing code path
3. the data pattern in the file

in that order.
