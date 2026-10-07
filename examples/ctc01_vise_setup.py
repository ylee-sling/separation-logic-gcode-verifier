#!/usr/bin/env python3
"""
ctc01_vise_setup.py -- Example fixture setup around the real NIST CTC-01 part
=============================================================================

A DEMONSTRATION setup, not a real machine. It shows the pipeline grounded from a fixture
STEP file (per-solid AABBs) instead of the synthetic Clamp_1 mock.

Setup, in G54 = CAD origin (mm):
  Solid #001  CTC-01 part (from nist_ctc_01_asme1_rd.stp)  X [-400, 400] Y [-225, 225] Z [-100, 50]
              -> Stock (its AABB), so G00 may not enter it and G01 cuts it
  Solid #002  Vise_jaw_L    X [-150, 150] Y [-265, -225] Z [-100, -50]
  Solid #003  Vise_jaw_R    X [-150, 150] Y [ 225,  265] Z [-100, -50]
  Solid #004  Base_plate    X [-450, 450] Y [-300,  300] Z [-130, -100]
Machine travel X ±500, Y ±350, Z [-150, 200]; SAFE_Z Z100; tool T01 R5 × 30 mm.
The tool starts at X0 Y0 Z150: the G54 origin lies inside the part, so Σ0 would be unsatisfiable.

The scripted generator retracts from the facing cut with G01 (a G00 would touch the material
left within ε of the tool) and then plunges a rapid onto the left jaw; the prover traps it, the
B_conflict feedback goes back, and the second draft parks beside the jaw instead.

Usage (needs pythonocc-core):
    python examples/ctc01_vise_setup.py            # build examples/ctc01_vise_setup.stp, run the loop
"""

from __future__ import annotations

import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.dirname(HERE)
sys.path.insert(0, REPO)

from cad_extractor import extract_solids  # noqa: E402
from main_loop import LLM_Agent, PipelineConfig, PipelineResult, run_neuro_symbolic_pipeline  # noqa: E402

NIST_STEP = os.path.join(REPO, "nist_ctc_01_asme1_rd.stp")
SETUP_STEP = os.path.join(HERE, "ctc01_vise_setup.stp")

# (name, corner, size) of each fixture box, in solid order #002..#004.
FIXTURES = (
    ("Vise_jaw_L", (-150.0, -265.0, -100.0), (300.0, 40.0, 50.0)),
    ("Vise_jaw_R", (-150.0, 225.0, -100.0), (300.0, 40.0, 50.0)),
    ("Base_plate", (-450.0, -300.0, -130.0), (900.0, 600.0, 30.0)),
)
PART_SOLID = 1

INTENT = (
    "Take a light facing pass along Y0 at Z48 from X-380 to X380, then park the tool "
    "at X0 next to the left vise jaw at Z-60 for probing."
)

DRAFT_COLLIDING = """\
```gcode
N10 G21 G90 G54
N20 G00 Z100.0
N30 G00 X-380.0 Y0.0
N40 G01 Z48.0 F200
N50 G01 X380.0 F500
N60 G01 Z100.0 F1000
N70 G00 X0.0 Y-245.0
N80 G00 Z-60.0
N90 M30
```"""

DRAFT_CORRECTED = """\
The rapid at N80 plunged onto Vise_jaw_L. The corrected program parks outside the jaw's
conflict bounds (Y-300 clears Y-265 by more than the tool radius plus ε). The retract out of
the cut stays G01 (N60): material within ε of the tool is still there after a nominal cut.
```gcode
N10 G21 G90 G54
N20 G00 Z100.0
N30 G00 X-380.0 Y0.0
N40 G01 Z48.0 F200
N50 G01 X380.0 F500
N60 G01 Z100.0 F1000
N70 G00 X0.0 Y-300.0
N80 G00 Z-60.0
N90 M30
```"""


def build_setup_step(nist_step: str = NIST_STEP, out_path: str = SETUP_STEP) -> str:
    """Write the part solid plus the fixture boxes to one STEP file (solid order as documented)."""
    from OCC.Core.BRep import BRep_Builder
    from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox
    from OCC.Core.IFSelect import IFSelect_RetDone
    from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Reader, STEPControl_Writer
    from OCC.Core.TopAbs import TopAbs_SOLID
    from OCC.Core.TopExp import TopExp_Explorer
    from OCC.Core.TopoDS import TopoDS_Compound
    from OCC.Core.gp import gp_Pnt

    reader = STEPControl_Reader()
    if reader.ReadFile(nist_step) != IFSelect_RetDone:
        raise RuntimeError(f"cannot read {nist_step}")
    reader.TransferRoots()
    explorer = TopExp_Explorer(reader.OneShape(), TopAbs_SOLID)  # the solid only, not the wireframe
    if not explorer.More():
        raise RuntimeError(f"no solid in {nist_step}")
    part = explorer.Current()

    setup, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(setup)
    builder.Add(setup, part)
    for _, corner, size in FIXTURES:
        builder.Add(setup, BRepPrimAPI_MakeBox(gp_Pnt(*corner), *size).Shape())

    writer = STEPControl_Writer()
    writer.Transfer(setup, STEPControl_AsIs)
    if writer.Write(out_path) != IFSelect_RetDone:
        raise RuntimeError(f"cannot write {out_path}")
    return out_path


def setup_config(setup_step: str = SETUP_STEP) -> PipelineConfig:
    """Fixtures are solids #002..#004; the part (#001) becomes Stock through its AABB."""
    part = next(s for s in extract_solids(setup_step) if s.index == PART_SOLID)
    return PipelineConfig(
        machine_limits_mm=(("-500", "500"), ("-350", "350"), ("-150", "200")),
        safe_z_mm="100.0",
        home_mm=("0", "0", "150"),
        fixture_step_file=setup_step,
        fixture_solids={index: name for index, (name, _, _) in enumerate(FIXTURES, start=2)},
        stock_mm=(part.aabb,),
    )


def run(verbose: bool = True, setup_step: str = SETUP_STEP) -> PipelineResult:
    config = setup_config(setup_step)
    agent = LLM_Agent("", [DRAFT_COLLIDING, DRAFT_CORRECTED])
    return run_neuro_symbolic_pipeline(INTENT, config, agent, verbose=verbose)


if __name__ == "__main__":
    print(f"Building {os.path.relpath(build_setup_step(), REPO)} from {os.path.basename(NIST_STEP)}\n")
    result = run()

    first, second = result.iterations
    assert result.verified and len(result.iterations) == 2
    assert first.outcome == "SPATIAL_DATA_RACE" and first.record.resource_label == "Vise_jaw_L"
    assert first.record.line_number == "N80" and first.record.motion_mode == "G00"
    assert second.outcome == "PROVED"
    print("\n[example] verified on iteration 2")
