# Neuro-Symbolic G-code Evaluator-Refiner

Python prototype of the pipeline in Lee, *Correct-by-Construction G-Code Generation* (arXiv:2605.10568),
built on Lee, *Separation Logic for Verifying Physical Collisions of CNC Programs* (arXiv:2605.10437).
An LLM drafts G-code. A deterministic Separation Logic prover checks it on a discrete Spatial Heap in Z^3.
Collisions (Spatial Data Races) go back to the LLM as B_conflict feedback until the program is proved.

## Modules (dependency order)

| File | Role | Imports from the project |
|---|---|---|
| `sl_feedback_generator.py` | Refiner: `ConflictRecord` (+ `from_boxes`), B_conflict, Figure 5 prompt | nothing |
| `cad_extractor.py` | Grounding: STEP → B-Rep features → AABB, `s_grid`, `GridBox`, RAG `OBSTACLE_BOUNDS` | nothing (OpenCASCADE only inside `_load_occ()`) |
| `parser.py` | Discretization: Store, `path_box` / `path_lin`, Minkowski footprints, `VoxelRegion` | `cad_extractor` |
| `sl_prover.py` | Evaluator: Spatial Heap, separating conjunction, G00/G01 rules | `cad_extractor`, `parser`, `sl_feedback_generator` |
| `main_loop.py` | Orchestrator: system prompt, `LLM_Agent` stub, Evaluator-Refiner loop | all four |

## Commands

- Self-checks. Each must end with `[self-check] all assertions passed`:
  `python sl_feedback_generator.py`, `python cad_extractor.py`, `python parser.py`, `python sl_prover.py`, `python main_loop.py`
- Tests: `pytest -q`
- Real STEP extraction: `python cad_extractor.py <file.stp> --epsilon 2.0 --mu 10 --json rag.json`
- Fixture STEP (one AABB per solid): `python cad_extractor.py <fixtures.stp> --solids --json rag.json`

## Environment

- Python ≥ 3.10. On ≤ 3.9 the built-in `parser` module shadows `parser.py`.
- The core is pure standard-library Python. OpenCASCADE is needed only for `extract_step()`, and `pytest` only for tests.
- `cad_extractor.py` targets **pythonocc-core** (`OCC.Core.*`, distributed through conda-forge).
  **`cadquery-ocp` is not a drop-in replacement.** It uses the `OCP.*` namespace and `_s`-suffixed static methods.
  If OCP has to be used, add a backend switch inside `_load_occ()` / `_classify_face()` only. Change nothing else for it.

## Design invariants (do not change without explicit approval)

These were decided on purpose, and several deviate from the papers for soundness.
A change that "simplifies" one of them is a safety regression, not a cleanup.

1. **ε is applied to the tool only.** The parser builds `V_path = path ⊕ V_tool ⊕ B_ε`.
   The heap stores obstacles **nominal** (`PaddedObstacle.nominal_grid`).
   The padded box (`padded_grid`) is only for the RAG prompt. Never pad both.
2. **G01 dilates by ε + 1** (`bresenham_margin=1`). Bresenham can skip cells the continuous segment crosses.
   `bresenham_margin=0` exists only to reproduce the paper's formula.
3. **Material removal uses `MotionStep.v_sweep`** (path ⊕ V_tool, no ε), not `v_path` (`SLProver(removal="nominal")`).
   Removing the padded volume erases real material from the model; the thin-wall self-check proves it.
4. **Discretization is exact.** `s_grid(x) = ⌊x·μ⌋` uses `Decimal`. G-code numbers are parsed as `Decimal`, never `float`.
   The 1e-7 mm float-noise snapping applies to CAD coordinates only.
5. **Grid → mm back-projection is `[lo/μ, (hi+1)/μ]`, rounded outward.** `cad_extractor` and `sl_feedback_generator` must agree.
6. **Footprints, obstacles and stock are unions of `GridBox`.** Never enumerate voxels on a hot path.
   `to_voxel_set()` is for small tests and is size-guarded. Stock fragments stay pairwise disjoint in the X-bucketed index.
7. **The parser fails closed.** These raise `GCodeError` and are never skipped or approximated silently:
   G02/G03, G28/G30, G41/G42, G53, G55–G59, G92, canned cycles, A/B/C/U/V/W, M98/M99,
   G01 without F, and axis words without a motion mode.
8. **A fault leaves the heap unchanged** and halts the prover until `reset()` / `verify()`.
9. **All LLM feedback uses the Figure 5 headers** `[System Error]`, `[Conflict Domain]`, `[Actionable Directive]`,
   including parser and block-number rejections.
10. **Generated programs use strictly increasing N numbers in steps of 10.**
11. **G54 work coordinates = CAD origin.** There is no work-offset transform yet.

## Testing rules

- If a self-check or test fails after a change, fix the change. Do not edit an assertion to make it pass
  unless the user agrees the expectation was wrong.
- Reference values:
  - The mock collision is `N20 G00 X-245.0 Y0.0 Z-80.0` after `N10 G01 X20.0 Y10.0 Z5.0 F100`, with ε = 2.0 mm, μ = 10.
    - Point tool: B_conflict = X[-2470, -2250] Y[-20, 120] Z[-820, -700].
    - T01 R5 × 30 mm end mill (`main_loop`): B_conflict = X[-2521, -2250] Y[-71, 171] Z[-820, -700].
  - `G00 X-25.0 Y0.0 Z-80.0` stops about 200 mm short of Clamp_1 and must be **proved safe**.
  - `nist_ctc_01_asme1_rd.stp` is NIST CTC-01 (the Paper 2 Figure 3 reference model), from the NIST MBE PMI
    Validation and Conformance Testing Project. Keep it unmodified, and keep the citation in README.md "Test data".
  - Clamp_1 is a **synthetic** scenario. The real CTC-01 surface #006 is a concave half-bore at Z[-50, 0], not a clamp.
    Keep the mock and its values as a regression test. Real fixtures come from `fixture_step_file`.
- The OpenCASCADE path (`extract_step()`, `_classify_face()`, `_convexity()`, `_count_subshapes()`, `extract_solids()`)
  is covered by `tests/test_occ_extraction.py` against generated STEP files (pythonocc-core 8.0.1.1).

## Known gaps (not bugs)

- Only cylindrical and spherical part faces become obstacles. Fixtures use per-solid AABBs (`extract_solids()`).
- The part is not loaded as Stock from STEP. `stock_mm` boxes are given by hand.
- Cells the tool only partly cuts are counted as removed (error < 1/μ mm).
- `LLM_Agent` is a scripted stub. A real model call replaces only `generate_code()`.
- The Figure 3 counts in Paper 2 are TopExp_Explorer visits, not unique sub-shapes.
