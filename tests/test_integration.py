"""End-to-end checks of the pipeline that need no OpenCASCADE.

Reference values come from CLAUDE.md ("Testing rules"): ε = 2.0 mm, μ = 10, Clamp_1 = CAD
surface #006 (R20 cylinder at (-245, 0, -100), V ∈ [0, 30]).
"""

import subprocess
import sys
from pathlib import Path

import pytest

from cad_extractor import AnalyticFeature, GridBox, pad_feature
from main_loop import MAX_ITERATIONS, _DRAFT_COLLIDING, LLM_Agent, run_neuro_symbolic_pipeline
from parser import Parser
from sl_feedback_generator import (
    HEADER_CONFLICT_DOMAIN,
    HEADER_DIRECTIVE,
    HEADER_SYSTEM_ERROR,
    generate_sl_feedback,
)
from sl_prover import STOCK, SLProver, SpatialDataRace

REPO = Path(__file__).resolve().parent.parent
MU, EPSILON = 10, "2.0"
FEED = "N10 G01 X20.0 Y10.0 Z5.0 F100"
INTENT = "Finishing pass to X20 Y10 at Z5, then park at X-245 Y0 next to Clamp_1."
MODULES = ["sl_feedback_generator", "cad_extractor", "parser", "sl_prover", "main_loop"]
HEADERS = (HEADER_SYSTEM_ERROR, HEADER_CONFLICT_DOMAIN, HEADER_DIRECTIVE)


@pytest.fixture
def clamp_prover() -> SLProver:
    clamp = pad_feature(
        AnalyticFeature(6, "CYLINDER", (-245.0, 0.0, -100.0), 20.0, (0.0, 0.0, 1.0), (0.0, 30.0)),
        float(EPSILON), MU, "Clamp_1",
    )
    prover = SLProver(MU)
    prover.add_environment(clamp.nominal_grid, clamp.name)  # nominal: ε lives on the tool
    return prover


def compile_(gcode: str):
    return Parser(MU, EPSILON).compile(gcode)


@pytest.mark.parametrize("module", MODULES)
def test_self_check_exits_zero(module):
    proc = subprocess.run(
        [sys.executable, f"{module}.py"], cwd=REPO, capture_output=True, text=True, timeout=600,
    )
    assert proc.returncode == 0, proc.stderr[-2000:]
    assert "[self-check] all assertions passed" in proc.stdout


def test_g01_feed_is_proved(clamp_prover):
    result = clamp_prover.verify(compile_(FEED))
    assert result.proved and result.steps_verified == 1


def test_g00_into_clamp_is_a_spatial_data_race(clamp_prover):
    program = compile_(f"{FEED}\nN20 G00 X-245.0 Y0.0 Z-80.0")
    clamp_prover.initialize(program.v_home)
    clamp_prover.execute(program.steps[0])
    with pytest.raises(SpatialDataRace) as info:
        clamp_prover.execute(program.steps[1])
    race = info.value
    assert race.b_conflict == GridBox(-2470, -2250, -20, 120, -820, -700)
    assert race.record.line_number == "N20" and race.record.motion_mode == "G00"
    assert race.record.resource_label == "Clamp_1"
    feedback = generate_sl_feedback(race.record)
    assert all(h in feedback for h in HEADERS)
    assert "X ∈ [-2470, -2250], Y ∈ [-20, 120], Z ∈ [-820, -700]" in feedback


def test_g00_short_of_clamp_is_proved(clamp_prover):
    result = clamp_prover.verify(compile_(f"{FEED}\nN20 G00 X-25.0 Y0.0 Z-80.0"))
    assert result.proved and result.steps_verified == 2


def test_pipeline_verifies_on_iteration_two():
    result = run_neuro_symbolic_pipeline(INTENT, verbose=False)
    assert result.verified and len(result.iterations) == 2
    first, second = result.iterations
    assert first.outcome == "SPATIAL_DATA_RACE" and second.outcome == "PROVED"
    box = first.record.bounding_box  # T01 R5 × 30 mm end mill
    assert (box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max) == (
        -2521, -2250, -71, 171, -820, -700)
    assert all(h in first.feedback for h in HEADERS)


def test_stubborn_agent_stops_after_max_iterations():
    stubborn = LLM_Agent("", [_DRAFT_COLLIDING])
    result = run_neuro_symbolic_pipeline(INTENT, agent=stubborn, verbose=False)
    assert not result.verified and result.gcode is None
    assert stubborn.calls == MAX_ITERATIONS
    assert len(result.iterations) == MAX_ITERATIONS
    assert all(r.outcome == "SPATIAL_DATA_RACE" for r in result.iterations)


def test_thin_wall_nominal_removal_faults_padded_proves():
    # 0.9 mm wall (< ε), cut 1 mm beside it, then a rapid straight through the wall.
    wall = GridBox(100, 108, -500, 500, -200, 0)
    program = compile_("N10 G00 Z5.0\nN20 G00 X9.0 Y-40.0\nN30 G01 Z-5.0 F100\nN40 G01 Y40.0\nN50 G00 X15.0")
    verdicts = {}
    for policy in ("nominal", "padded"):
        prover = SLProver(MU, removal=policy)
        prover.add_stock(wall)
        verdicts[policy] = prover.verify(program)
    nominal = verdicts["nominal"]
    assert not nominal.proved and nominal.failed_step.block == "N50"
    assert nominal.record.violated_resource == STOCK
    assert verdicts["padded"].proved
