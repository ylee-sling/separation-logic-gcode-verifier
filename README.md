# Neuro-Symbolic G-code Evaluator-Refiner

A Python prototype of the pipeline in *Correct-by-Construction G-Code Generation*
([Paper 2](#references), arXiv:2605.10568), built on *Separation Logic for Verifying Physical
Collisions of CNC Programs* ([Paper 1](#references), arXiv:2605.10437). "Paper 1" and "Paper 2"
refer to these two papers throughout this README and the code comments.

An LLM drafts G-code. A deterministic Separation Logic prover checks every motion block on a
discrete Spatial Heap in Z³. A collision (a *Spatial Data Race*) is returned to the LLM as a
structured `B_conflict` prompt. The loop repeats until the whole program is proved or the iteration
budget runs out.

## Architecture

```
            ┌──────────── RAG grounding (cad_extractor.py) ─────────────┐
 STEP file ─┤ B-Rep faces → analytic features → AABB → S_grid → GridBox ├─► Environment heap
            └───────────────────────────┬───────────────────────────────┘   + OBSTACLE_BOUNDS
                                        │ system prompt (Fig. 4)
                                        ▼
  intent ──► Generator (LLM_Agent) ──► G-code ──► Parser (parser.py)
                  ▲                                │  Store + Decimal → Z³, V_path = path ⊕ V_tool ⊕ B_ε
                  │                                ▼
                  │                    Evaluator (sl_prover.py)
                  │                     {P} C {Q} per block, V_path ∗ (C_env ∪ C_stock)
                  │                                │
                  │      FAULT (SpatialDataRace)   │   PROVED for every block
                  └── Refiner (sl_feedback_generator.py) ◄─┘        └──► verified program
                      Fig. 5 B_conflict prompt
```

| Module | Role |
|---|---|
| `sl_feedback_generator.py` | Refiner: `ConflictRecord` (+ `from_boxes`), B_conflict, the Figure 5 prompt |
| `cad_extractor.py` | Grounding: STEP → B-Rep features → AABB, `s_grid`, `GridBox`, RAG `OBSTACLE_BOUNDS` |
| `parser.py` | Discretization: Store, `path_box` / `path_lin`, Minkowski footprints, `VoxelRegion` |
| `sl_prover.py` | Evaluator: Spatial Heap, separating conjunction, G00/G01 rules |
| `main_loop.py` | Orchestrator: system prompt, `LLM_Agent` stub, Evaluator-Refiner loop |

`main_loop.run_neuro_symbolic_pipeline()` runs these steps:

1. **Generator.** `LLM_Agent.generate_code()` returns a draft. Today this is a scripted stub, and a
   real model call replaces only this method.
2. **Parser.** It checks that N numbers increase strictly in steps of 10. It then compiles the
   program into `MotionStep`s, each carrying `V_start`, `V_final`, `V_path` and `v_sweep`. The
   parser *fails closed*: G02/G03, G28/G30, G41/G42, G53, G55–G59, G92, canned cycles, rotary
   axes, M98/M99, G01 without F, and axis words without a motion mode all raise `GCodeError`. The
   error goes back to the LLM in the Figure 5 format.
3. **Evaluator.** `SLProver` executes each step against the heap and raises `SpatialDataRace` on
   the first conflict. A fault leaves the heap unchanged and halts the prover until
   `reset()` / `verify()`.
4. **Terminate** on a full proof, **or Refiner.** On a fault, `generate_sl_feedback()` turns the
   `ConflictRecord` into the prompt, which is appended to the history, and the loop repeats. The
   cap is `MAX_ITERATIONS = 5`.

## The Spatial Heap

Space is discretized at μ cells per mm with `s_grid(x) = ⌊x·μ⌋`, computed in `Decimal`. G-code
numbers are parsed as `Decimal`, never `float`. A small float-noise snap (1e-7 mm) applies to CAD
coordinates only. Grid boxes back-project to mm as `[lo/μ, (hi+1)/μ]`, rounded outward.

Every voxel is in exactly one of four states:

| State | Contents | Represented as |
|---|---|---|
| **Tool** | the current tool footprint | `VoxelRegion` (union of `GridBox`) |
| **Environment** | fixtures, clamps, machine travel limits (outside the workspace W) | labelled `GridBox`es, stored **nominal** |
| **Stock** | material still to be cut | pairwise disjoint `GridBox`es in an X-bucketed `_BoxIndex` |
| **Empty** | everything else | implicit |

Footprints, obstacles and stock are always unions of axis-aligned boxes. Voxels are never
enumerated on a hot path. For scale, the nominal Clamp_1 box claims 48,401,101 voxels at μ = 10,
and a single G00 in the mock run sweeps 230,632,963. The separating conjunction is a box-overlap
test. Removing material is box subtraction (`subtract_boxes`), and the stock fragments stay
disjoint.

Rules per block:

- **G00 (rapid):** `V_path ∗ (C_env ∪ C_stock)` must hold. A rapid may touch neither fixtures nor
  material.
- **G01 (feed):** `V_path ∗ C_env` must hold. Stock in the swept region is cut and moves to Empty.

## Minkowski padding: what is padded and why

**1. ε is applied to the tool only.**
The parser builds `V_path = path ⊕ V_tool ⊕ B_ε`. The heap stores obstacles *nominal*
(`PaddedObstacle.nominal_grid`). The padded box (`padded_grid`) is used only to show the LLM
pre-expanded `OBSTACLE_BOUNDS` in the RAG prompt. Padding both sides would apply the margin
twice, 2ε, and reject safe programs.

Example with ε = 2 mm, μ = 10 (from `python cad_extractor.py`):

| Clamp_1 (#006, R20 at (-245, 0, -100)) | Value |
|---|---|
| Nominal AABB (mm) | X [-265, -225] Y [-20, 20] Z [-100, -70] |
| Heap box, nominal (Z³) | X [-2650, -2250] Y [-200, 200] Z [-1000, -700] |
| ε in voxels | ⌈2.0 × 10⌉ = 20 |
| RAG box, padded (Z³) | X [-2670, -2230] Y [-220, 220] Z [-1020, -680] |
| RAG bounds shown to the LLM (mm) | X [-267.000, -222.900] Y [-22.000, 22.100] Z [-102.000, -67.900] |

Clamp_1 is a **synthetic mock**. Figure 3 gives only the radius and centre of surface #006, and
the mock *assumes* a +Z axis, a 30 mm height and a convex boss. Reading the real
`nist_ctc_01_asme1_rd.stp` gives R20 at (-245, 0, -100), matching Figure 3, but the face is a
**concave 180° half-bore of the part** with V ∈ [50, 100], so Z ∈ [-50, 0]. It is not a clamp.
The mock stays as a fixed regression scenario. Real fixtures come from a fixture STEP file
(see [Fixtures](#fixtures-from-a-step-file)).

**2. G01 dilates by ε + 1 (`bresenham_margin=1`).**
G01 footprints follow a Bresenham line. A Bresenham path can skip a cell that the continuous
segment actually crosses (the `parser.py` docstring has an example). One extra voxel of dilation
covers every such cell, so the discrete check stays sound for the continuous motion. This shows
up in the G01 footprint: `N10 G01 X20 Y10 Z5` gives `V_path X ∈ [-21, 221]`, while the G00 boxes
use exactly ±20. `bresenham_margin=0` exists only to reproduce the paper's formula.

**3. Material removal uses the nominal sweep (`MotionStep.v_sweep`, `removal="nominal"`).**
A G01 cut removes `path ⊕ V_tool` *without* ε. Removing the ε-padded volume (`removal="padded"`,
the paper's formula) would erase material that is still physically there. The `sl_prover`
self-check shows this with a 0.9 mm wall (narrower than ε) next to a cut, followed by a rapid
straight through the wall:

```
removal='nominal' -> FAULT at N50, Stock race within X ∈ [100, 108], Y ∈ [380, 420], Z ∈ [-70, -30]
removal='padded'  -> PROVED, because the wall was erased from the heap by the ε-padded cut
```

Padded removal "proves" a program that would crash into the wall.

## Feedback format (Figure 5)

All feedback, including parser and block-number rejections, uses three headers:
`[System Error]`, `[Conflict Domain]` and `[Actionable Directive]`. This is the real prompt that
`python main_loop.py` produces in iteration 1 (tool T01, R5 × 30 mm flat end mill):

```
Autogenerated SL Feedback Prompt
[System Error] Spatial Data Race Detected
The Separation Logic verifier encountered a physical collision at the following instruction:
N20 G00 X-245.0 Y0.0 Z-80.0

[Conflict Domain]
The requested swept volume intersects with the restricted Environment Heap (Clamp_1) within the following integer bounds:
X ∈ [-2521, -2250], Y ∈ [-71, 171], Z ∈ [-820, -700]
Back-projected machine bounds (mm, μ = 10; grid cell i spans [i/μ, (i+1)/μ), rounded outward):
X ∈ [-252.100, -224.900], Y ∈ [-7.100, 17.200], Z ∈ [-82.000, -69.900]

[Actionable Directive]
Regenerate the toolpath between lines N10 and N30. You must route the tool to strictly avoid the stated conflict bounds.
The bounds apply to the full tool envelope (tool geometry dilated by the safety margin ε); moving only the programmed tool-center coordinates to the edge of the bounds is not sufficient.
Consider increasing the Z-axis clearance height (G00 Z...) prior to lateral XY translation.
```

With a point tool, the same block (`sl_prover.py` mock) gives
`B_conflict = X [-2470, -2250] Y [-20, 120] Z [-820, -700]`. The near miss `G00 X-25.0 Y0.0 Z-80.0`,
about 200 mm short of the clamp, is proved safe.

In the mock run, the stub generator retracts to `Z50.0` before the lateral rapid, and the program
is **verified on iteration 2** (2 generator calls, 3 motion blocks, 0 stock voxels removed).

## Installation

The core is pure standard-library Python **≥ 3.10**. On ≤ 3.9 the built-in `parser` module can
shadow `parser.py`. OpenCASCADE is needed only for `cad_extractor.extract_step()`, and pytest only
for the tests.

Recommended setup, a conda-forge environment with Python 3.11, pythonocc-core and pytest:

```sh
conda env create -f environment.yml
conda activate sl-gcode
python -c "import OCC; print(OCC.VERSION)"
```

On Apple Silicon with an x86_64 (Intel-prefix) conda, create a native env with
`CONDA_SUBDIR=osx-arm64 conda env create -f environment.yml`.

Without OpenCASCADE: `pip install -r requirements-dev.txt`. Everything except
`tests/test_occ_extraction.py` (which is skipped) still runs.

**`cadquery-ocp` is not a drop-in replacement.** It exposes `OCP.*` instead of `OCC.Core.*`, and
its static methods carry an `_s` suffix (`TopoDS.Face_s`, `BRepBndLib.AddOptimal_s`). Supporting
it would need a backend switch inside `_load_occ()` / `_classify_face()`.

## Running

```sh
# Self-checks: each must end with "[self-check] all assertions passed"
python sl_feedback_generator.py
python cad_extractor.py        # mock extraction of nist_ctc_01 surface #006
python parser.py
python sl_prover.py
python main_loop.py            # end-to-end Evaluator-Refiner run

# Tests (pytest.ini points pytest at tests/)
pytest -q

# Real STEP extraction: report + RAG payload
python cad_extractor.py part.stp --epsilon 2.0 --mu 10 --json rag.json

# Fixture STEP: one AABB obstacle per solid
python cad_extractor.py fixtures.stp --solids --json rag.json
```

### Fixtures from a STEP file

Clamps, vise jaws and fixture plates are mostly planar, so no analytic face bounds them.
`extract_solids()` turns each solid of a fixture STEP file into one `SolidFeature`. Its AABB
comes from `BRepBndLib::AddOptimal` on the exact geometry (no triangulation), enlarged by the
shape tolerance. The box contains the solid, so it is conservative. It goes through the same
`pad_feature()` as analytic faces: the nominal box goes into the Environment heap and the padded
box into `OBSTACLE_BOUNDS`.

The fixtures must be modelled in G54, which is the CAD origin. Run `--solids` first to see the
solid indices, then select the fixtures and name them:

```python
config = PipelineConfig(
    fixture_step_file="setup.stp",
    fixture_solids={1: "Vise_jaw_L", 2: "Vise_jaw_R"},  # omit the part's own solid; None = all
    machine_limits_mm=(("-450", "450"), ("-250", "250"), ("-150", "150")),
)
run_neuro_symbolic_pipeline(intent, config)
```

`home_mm` sets the tool-tip position before the first block. The default is the G54 origin. A
part that covers the origin needs a raised home, or Σ0 is unsatisfiable (`HeapInitError`).

When `fixture_step_file` is set without `step_file`, the mock Clamp_1 is not added. Only solids
become obstacles. Wireframe or PMI geometry in the file is ignored. The NIST CTC-01 file, for
example, has one solid with the box X [-400, 400] Y [-225, 225] Z [-100, 50] plus a separate
wireframe compound that reaches X [-590, 580]. That part exceeds the default
`machine_limits_mm` in Y (±200).

### Example: CTC-01 in a vise (demonstration setup)

`python examples/ctc01_vise_setup.py` builds `examples/ctc01_vise_setup.stp` from the real
CTC-01 solid plus three made-up fixture boxes, then runs the loop. The fixtures are two vise
jaws at Y [-265, -225] and Y [225, 265] (X [-150, 150], Z [-100, -50]) and a base plate at
Z [-130, -100]. The part's AABB is loaded as Stock. The machine travels X ±500, Y ±350,
Z [-150, 200], with home at Z150 and SAFE_Z at Z100.

Output of the run:

```
N80  G00  FAULT, separating conjunction fails against Environment (Vise_jaw_L)
          B_conflict = X ∈ [-71, 71], Y ∈ [-2521, -2379], Z ∈ [-620, -500]
...
{Σ0} C_SL {Σ_final} proved for all 7 motion blocks; 16,612,113 stock voxels removed
✔ Correct-by-Construction Verified (iteration 2, 2 generator calls)
```

Writing the drafts showed one rule that matters in practice. After a G01 cut, the tool must leave
the material **with G01**. Only the nominal sweep is removed, so the material within ε of the
tool is still in the heap, and a G00 retract from the cut is rejected as a Stock race. That is
invariant 3 working as intended.

`tests/test_integration.py` needs no OpenCASCADE. `tests/test_occ_extraction.py` builds solids in
OpenCASCADE, writes them to STEP and checks `extract_step()` against the known geometry. It
covers a boss and a bore (convexity), Euler V − E + F = 2, and the AABB of a tilted cylinder
against OCC's optimal bounding box. It also covers per-solid fixture AABBs (including a rotated
solid) and a full pipeline run grounded from a generated fixture STEP.

The OCC path agrees with OpenCASCADE's own solid classifier on NIST CTC-01. All 57 cylinder faces
get the same convexity from both. The file's unique counts are 433 / 468 / 139, so
V − E + F = 104 rather than 2, because its 433 vertex entities lie at only 297 distinct points
(duplicate, unshared vertices in the STEP). The explorer visits match Figure 3 exactly:
1734 / 856 / 139.

## Test data

`nist_ctc_01_asme1_rd.stp` is the NIST standard reference model used in Paper 2, Figure 3
("Truncated extraction report for the NIST standard reference model (nist_ctc_01_asme1_rd.stp)").
It is Combined Test Case 1 (CTC-01, geometry only) from the NIST MBE PMI Validation and
Conformance Testing Project. It is included unmodified, for testing only.

> National Institute of Standards and Technology (NIST). MBE PMI Validation and Conformance
> Testing Project.
> https://www.nist.gov/ctl/smart-connected-systems-division/smart-connected-manufacturing-systems-group/mbe-pmi-0

## Known gaps (not bugs)

- Part faces: only cylindrical and spherical faces become obstacles. Fixtures are covered by
  per-solid AABBs (`--solids`, `fixture_step_file`), which over-approximate non-box solids.
- The part itself is not loaded as Stock from STEP yet. Give `stock_mm` boxes by hand.
- Cells the tool only partly cuts are counted as removed (error < 1/μ mm).
- `LLM_Agent` is a scripted stub. A real model call replaces only `generate_code()`.
- The Figure 3 counts in Paper 2 (1734 / 856 / 139) are `TopExp_Explorer` visits, not unique
  sub-shapes.
- G54 work coordinates equal the CAD origin. There is no work-offset transform yet.

## References

- **[Paper 1]** Y. Lee. *Separation Logic for Verifying Physical Collisions of CNC Programs.*
  arXiv:2605.10437, 2026. https://arxiv.org/abs/2605.10437
- **[Paper 2]** Y. Lee. *Correct-by-Construction G-Code Generation: A Neuro-Symbolic Approach via
  Separation Logic.* arXiv:2605.10568, 2026. https://arxiv.org/abs/2605.10568
- NIST CTC-01 test model: see [Test data](#test-data).

## License

Copyright (C) 2026 Yeonseok Lee

This program is free software: you can redistribute it and/or modify it under the terms of the
GNU General Public License as published by the Free Software Foundation, either version 3 of the
License, or (at your option) any later version. It is distributed in the hope that it will be
useful, but WITHOUT ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS
FOR A PARTICULAR PURPOSE. See [LICENSE](LICENSE) for the full text.

The GPL covers the code and documentation of this repository. `nist_ctc_01_asme1_rd.stp` is
third-party data from NIST (see [Test data](#test-data)) and is not covered by this license.
