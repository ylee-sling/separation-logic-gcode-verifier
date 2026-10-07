#!/usr/bin/env python3
"""
main_loop.py -- The Orchestrator
================================

The Neuro-Symbolic Evaluator-Refiner loop (Lee, "Correct-by-Construction G-Code Generation:
A Neuro-Symbolic Approach via Separation Logic", Fig. 1-2, Sec. 3.3-3.4, 5.3).

    ground_workspace()      cad_extractor  ->  padded RAG bounds + nominal heap boxes
    build_system_prompt()   Fig. 4 constraints, strict N-numbered blocks
    loop (MAX_ITERATIONS):
      Step 1  Generator     LLM_Agent.generate_code(history)            -> candidate G-code
      Step 2  Parser        parser.Parser.compile()                       -> MotionSteps in Z^3
      Step 3  Evaluator     sl_prover.SLProver.execute() per step         -> {P} C {Q} or fault
      Step 4  Terminate     every triple proved                           -> verified G-code
      Step 5  Refiner       SpatialDataRace -> generate_sl_feedback()     -> appended to history

The conversation is cumulative, as in Paper 2, Sec. 5.3: the user intent, every flawed draft
and every feedback prompt stay in the history. The model therefore always sees the original
goal, what it tried and exactly where that failed.

Besides Spatial Data Races, the loop also feeds back two kinds of rejection, using the same
three Figure 5 headers:
  * a GCodeError from the Parser (arcs, canned cycles, missing feed, ... cannot be exhausted
    into Z^3, so they are rejected rather than skipped), and
  * a block-number violation (every block must carry a strictly increasing N word, since the
    Refiner localizes faults by N number).
Configuration errors that no G-code edit can fix (HeapInitError, HeapStateError) abort the run.

The Generator is a deterministic stub. LLM_Agent.generate_code() is the only place a real
model call needs to go; its inputs (system prompt plus alternating user/assistant messages)
already have the shape a chat-completion API expects.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from decimal import Decimal
from typing import Dict, List, Mapping, Optional, Sequence, Tuple

from cad_extractor import (
    AABB,
    AnalyticFeature,
    GridBox,
    PaddedObstacle,
    extract_step,
    format_obstacle_bounds,
    grid_box_of,
    pad_features,
)
from parser import CompiledProgram, GCodeError, Parser, ToolVolume
from sl_feedback_generator import (
    HEADER_CONFLICT_DOMAIN,
    HEADER_DIRECTIVE,
    HEADER_SYSTEM_ERROR,
    PROMPT_TITLE,
    ConflictRecord,
    generate_sl_feedback,
)
from sl_prover import SLProver, SpatialDataRace, StepOutcome

__all__ = [
    "MAX_ITERATIONS",
    "PipelineConfig",
    "Workspace",
    "LLM_Agent",
    "IterationRecord",
    "PipelineResult",
    "ground_workspace",
    "build_system_prompt",
    "extract_gcode",
    "check_block_numbers",
    "format_rejection_feedback",
    "run_neuro_symbolic_pipeline",
]

MAX_ITERATIONS = 5
BLOCK_STEP = 10  # N10, N20, ... matches generate_sl_feedback's default regeneration window

Message = Dict[str, str]


# ======================================================================================
# Configuration and grounding
# ======================================================================================
def _mock_clamp_feature() -> AnalyticFeature:
    """Surface #006 of the NIST CTC-01 report (Paper 2, Fig. 3). Axis +Z and the 30 mm axial
    span are assumed, as in the cad_extractor mock."""
    return AnalyticFeature(
        index=6, kind="CYLINDER", location=(-245.0, 0.0, -100.0), radius=20.0,
        axis=(0.0, 0.0, 1.0), axial_range=(0.0, 30.0), angular_span_deg=360.0, convexity="convex",
    )


@dataclass(frozen=True)
class PipelineConfig:
    """Everything the loop needs. Lengths are mm strings, so they stay exact Decimals.

    fixture_features : face index -> obstacle name. Only these analytic features become
                       Environment. A real STEP part also yields cylinders for its own bores,
                       which are not fixtures. None means every extracted feature becomes one.
    step_file        : STEP file to ground from; None uses the mock Clamp_1 (no OpenCASCADE needed).
    """

    mu: int = 10
    epsilon_mm: str = "2.0"
    machine_limits_mm: Tuple[Tuple[str, str], ...] = (("-400", "400"), ("-200", "200"), ("-150", "150"))
    tool_id: str = "T01"
    tool_radius_mm: str = "5.0"
    tool_length_mm: str = "30.0"
    safe_z_mm: str = "50.0"
    step_file: Optional[str] = None
    fixture_features: Optional[Mapping[int, str]] = field(default_factory=lambda: {6: "Clamp_1"})
    stock_mm: Tuple[AABB, ...] = ()
    max_iterations: int = MAX_ITERATIONS


@dataclass(frozen=True)
class Workspace:
    """The grounded machine: heap contents for the prover and their padded RAG view."""

    config: PipelineConfig
    obstacles: Tuple[PaddedObstacle, ...]
    workspace_grid: GridBox
    stock_grid: Tuple[GridBox, ...]
    tool: ToolVolume
    epsilon_vox: int

    def new_parser(self) -> Parser:
        return Parser(self.config.mu, self.config.epsilon_mm, self.tool)

    def new_prover(self) -> SLProver:
        prover = SLProver(self.config.mu, workspace=self.workspace_grid)
        for obstacle in self.obstacles:
            prover.add_environment(obstacle.nominal_grid, obstacle.name)  # nominal: ε lives on the tool
        for box in self.stock_grid:
            prover.add_stock(box)
        return prover


def ground_workspace(config: PipelineConfig) -> Workspace:
    """Physical grounding (Paper 2, Sec. 3.2): B-Rep features -> padded obstacles, plus the
    discretized machine limits, stock and tool."""
    if config.step_file:
        features = extract_step(config.step_file).features()
    else:
        features = [_mock_clamp_feature()]
    if config.fixture_features is not None:
        features = [f for f in features if f.index in config.fixture_features]
    obstacles = tuple(pad_features(features, float(config.epsilon_mm), config.mu, names=config.fixture_features))

    (x0, x1), (y0, y1), (z0, z1) = config.machine_limits_mm
    limits = AABB(float(x0), float(x1), float(y0), float(y1), float(z0), float(z1))
    return Workspace(
        config=config,
        obstacles=obstacles,
        workspace_grid=grid_box_of(limits, config.mu),
        stock_grid=tuple(grid_box_of(box, config.mu) for box in config.stock_mm),
        tool=ToolVolume.flat_end_mill(config.tool_radius_mm, config.tool_length_mm, config.mu),
        epsilon_vox=obstacles[0].epsilon_vox if obstacles else 0,
    )


# ======================================================================================
# System prompt (Paper 2, Fig. 4)
# ======================================================================================
def build_system_prompt(workspace: Workspace) -> str:
    """The RAG-grounded system prompt: Fig. 4 constraints plus the output rules the verifier needs."""
    c = workspace.config
    (x0, x1), (y0, y1), (z0, z1) = c.machine_limits_mm
    lines = [
        "You are a G-code generator for a 3-axis CNC mill. Every program you write is checked by a",
        "Separation Logic verifier before it may run: any block whose swept volume (tool plus safety",
        "margin ε) intersects an obstacle is rejected and returned to you with the conflicting bounds.",
        "",
        "System Constraints:",
        f"MACHINE_LIMITS: X ∈ [{x0}, {x1}], Y ∈ [{y0}, {y1}], Z ∈ [{z0}, {z1}]",
        "WORK_OFFSET: G54 (identical to the CAD origin)",
        f"ACTIVE_TOOL: {c.tool_id} (Flat Endmill, Radius={c.tool_radius_mm}mm, Length={c.tool_length_mm}mm; "
        "coordinates are tool-tip positions)",
        f"SAFE_Z_RETRACT: Z{c.safe_z_mm}",
        format_obstacle_bounds(workspace.obstacles, float(c.epsilon_mm)),
    ]
    if workspace.stock_grid:
        lines.append(f"STOCK_BOUNDS (machinable with G01 only; G00 must stay ε = {c.epsilon_mm}mm clear):")
        for k, box in enumerate(workspace.stock_grid, start=1):
            lines.append(f"- Stock_{k}: {box.dilate(workspace.epsilon_vox).format_mm(c.mu)}")
    lines += [
        "",
        "Output Rules:",
        f"1. Start every block with an N number, strictly increasing in steps of {BLOCK_STEP} "
        "(N10, N20, N30, ...). Blocks without N numbers are rejected.",
        "2. Use only: G00, G01 (with a feed rate F), G90/G91, G20/G21, G17, G54, G94, M03/M05/M30, S, T.",
        "   No arcs (G02/G03), canned cycles, G28/G30, cutter compensation (G41/G42) or rotary axes.",
        "3. OBSTACLE_BOUNDS already include ε. Additionally keep the tool axis at least the tool radius "
        f"({c.tool_radius_mm} mm) horizontally clear of them, unless the tool tip is above their Z maximum.",
        f"4. Before any lateral XY rapid, retract to SAFE_Z_RETRACT (G00 Z{c.safe_z_mm}).",
        "5. Return the complete program in one ```gcode fenced block.",
        "",
        "Directive: Route all G00 rapid movements to strictly avoid the OBSTACLE_BOUNDS.",
    ]
    return "\n".join(lines)


# ======================================================================================
# LLM interface (stub)
# ======================================================================================
_DRAFT_COLLIDING = """\
Here is the program: a finishing pass to X20 Y10, then the tool parks beside Clamp_1.
```gcode
N05 G21 G90 G54
N10 G01 X20.0 Y10.0 Z5.0 F100
N20 G00 X-245.0 Y0.0 Z-80.0
N30 M30
```"""

_DRAFT_CORRECTED = """\
The rapid at N20 descended into Clamp_1's conflict bounds. The corrected program retracts to the
safe Z height first and approaches X-245 Y0 from above, so the tool never enters those bounds.
```gcode
N05 G21 G90 G54
N10 G01 X20.0 Y10.0 Z5.0 F100
N20 G00 Z50.0
N30 G00 X-245.0 Y0.0
N40 M30
```"""


class LLM_Agent:
    """The Generator. A deterministic state machine standing in for a real model.

    State 0 returns a draft. Each time the latest user message is verifier feedback (it
    carries the [System Error] header), the agent advances to its next scripted refinement,
    and stays on the last one once the script runs out. A production version replaces the
    body of generate_code() with a chat-completion call that sends `self.system_prompt` and
    `conversation_history`.
    """

    def __init__(self, system_prompt: str, scripted_responses: Sequence[str] = (_DRAFT_COLLIDING, _DRAFT_CORRECTED)) -> None:
        if not scripted_responses:
            raise ValueError("the stub needs at least one scripted response")
        self.system_prompt = system_prompt
        self._responses = list(scripted_responses)
        self._state = 0
        self.calls = 0

    def generate_code(self, conversation_history: List[Message]) -> str:
        if not conversation_history or conversation_history[-1]["role"] != "user":
            raise ValueError("generate_code expects the history to end with a user message")
        self.calls += 1
        if HEADER_SYSTEM_ERROR in conversation_history[-1]["content"]:
            self._state = min(self._state + 1, len(self._responses) - 1)
        return self._responses[self._state]


# ======================================================================================
# Output hygiene and rejection feedback
# ======================================================================================
_FENCE = re.compile(r"```[ \t]*[\w+-]*[ \t]*\n(.*?)```", re.DOTALL)
_N_PREFIX = re.compile(r"^\s*N(\d+)(?!\d)", re.IGNORECASE)


def extract_gcode(response: str) -> str:
    """The first fenced code block of the reply, or the whole reply if it has no fence."""
    match = _FENCE.search(response)
    body = match.group(1) if match else response
    return "\n".join(line.rstrip() for line in body.strip().splitlines())


def check_block_numbers(gcode: str) -> Optional[Tuple[str, str]]:
    """Every non-empty, non-comment line needs an N word, strictly increasing.
    Returns (offending line, reason), or None if the program complies."""
    previous: Optional[int] = None
    for line in gcode.splitlines():
        text = line.strip()
        if not text or text == "%" or text.startswith(("(", ";")):
            continue
        match = _N_PREFIX.match(text)
        if not match:
            return text, "the block has no N number, so a fault in it could not be localized"
        number = int(match.group(1))
        if previous is not None and number <= previous:
            return text, f"N{number} does not increase after N{previous}"
        previous = number
    return None


def _window(block: Optional[str]) -> Optional[Tuple[str, str]]:
    match = _N_PREFIX.match(block or "")
    if not match:
        return None
    number = int(match.group(1))
    return f"N{max(0, number - BLOCK_STEP)}", f"N{number + BLOCK_STEP}"


def format_rejection_feedback(kind: str, instruction: str, domain: str, block: Optional[str], directive: str) -> str:
    """Non-spatial rejections, in the same three-section Figure 5 structure as the SL feedback."""
    window = _window(block)
    regenerate = (
        f"Regenerate the toolpath between lines {window[0]} and {window[1]}."
        if window else "Regenerate the complete program."
    )
    return "\n".join([
        PROMPT_TITLE,
        f"{HEADER_SYSTEM_ERROR} {kind}",
        "The Separation Logic parser could not exhaust the following instruction into discrete spatial sets:",
        instruction,
        "",
        HEADER_CONFLICT_DOMAIN,
        domain,
        "",
        HEADER_DIRECTIVE,
        f"{regenerate} {directive}",
    ])


# ======================================================================================
# The Evaluator-Refiner loop
# ======================================================================================
@dataclass(frozen=True)
class IterationRecord:
    iteration: int
    gcode: str
    outcome: str                                  # PROVED | SPATIAL_DATA_RACE | PARSER_REJECTION | BLOCK_NUMBERING
    feedback: Optional[str] = None
    record: Optional[ConflictRecord] = None
    outcomes: Tuple[StepOutcome, ...] = ()


@dataclass(frozen=True)
class PipelineResult:
    verified: bool
    gcode: Optional[str]
    iterations: Tuple[IterationRecord, ...]
    history: Tuple[Message, ...]
    program: Optional[CompiledProgram] = None


def _log(verbose: bool, text: str = "") -> None:
    if verbose:
        print(text)


def _indent(text: str, prefix: str = "      ") -> str:
    return "\n".join((prefix + line) if line else prefix.rstrip() for line in text.splitlines())


def run_neuro_symbolic_pipeline(
    intent: str,
    config: PipelineConfig = PipelineConfig(),
    agent: Optional[LLM_Agent] = None,
    *,
    verbose: bool = True,
    output_path: Optional[str] = None,
) -> PipelineResult:
    """Run Generator -> Parser -> Evaluator -> Refiner until a program is proved or the budget runs out."""
    workspace = ground_workspace(config)
    system_prompt = build_system_prompt(workspace)
    agent = agent or LLM_Agent(system_prompt)
    parser = workspace.new_parser()
    prover = workspace.new_prover()
    mu = config.mu

    _log(verbose, "=" * 78)
    _log(verbose, " NEURO-SYMBOLIC EVALUATOR-REFINER PIPELINE")
    _log(verbose, "=" * 78)
    _log(verbose, f"Grounding : μ = {mu} (voxel {Decimal(1) / mu} mm), ε = {config.epsilon_mm} mm "
                  f"(ε_vox = {workspace.epsilon_vox}), tool {config.tool_id} R{config.tool_radius_mm} "
                  f"L{config.tool_length_mm} ({len(workspace.tool.region.boxes)} voxel rows)")
    for obstacle in workspace.obstacles:
        _log(verbose, f"            Environment {obstacle.name} ({obstacle.source}) nominal heap box "
                      f"{obstacle.nominal_grid.format()}")
    _log(verbose, f"            Workspace W {workspace.workspace_grid.format()}")
    _log(verbose, "\n--- System prompt (RAG context injection) ---")
    _log(verbose, _indent(system_prompt, "  "))
    _log(verbose, "\n--- User intent ---")
    _log(verbose, _indent(intent, "  "))

    history: List[Message] = [{"role": "user", "content": intent}]
    records: List[IterationRecord] = []

    for iteration in range(1, config.max_iterations + 1):
        _log(verbose, f"\n[Iteration {iteration}/{config.max_iterations}]")

        # Step 1: Generator
        response = agent.generate_code(history)
        history.append({"role": "assistant", "content": response})
        gcode = extract_gcode(response)
        blocks = [line for line in gcode.splitlines() if line.strip()]
        _log(verbose, f"  Step 1  Generator : {len(blocks)} blocks received")
        _log(verbose, _indent(gcode, "            | "))

        def reject(outcome: str, feedback: str, record: Optional[ConflictRecord] = None) -> None:
            history.append({"role": "user", "content": feedback})
            records.append(IterationRecord(iteration, gcode, outcome, feedback, record))
            _log(verbose, "  Step 5  Refiner   : feedback appended to the conversation history")
            _log(verbose, _indent(feedback, "            > "))

        numbering = check_block_numbers(gcode)
        if numbering is not None:
            line, reason = numbering
            _log(verbose, f"  Step 2  Parser    : REJECTED, block numbering ({reason})")
            reject("BLOCK_NUMBERING", format_rejection_feedback(
                "Missing or Non-Increasing Block Number", line,
                f"Faults are localized by N number, and {reason}.", None,
                f"Prefix every block with an N number, strictly increasing in steps of {BLOCK_STEP} (N10, N20, ...).",
            ))
            continue

        # Step 2: Parser (continuous Store -> discrete Z^3 sets)
        try:
            program = parser.compile(gcode)
        except GCodeError as error:
            _log(verbose, f"  Step 2  Parser    : REJECTED at {error.block or f'line {error.line_index}'} ({error.reason})")
            reject("PARSER_REJECTION", format_rejection_feedback(
                "Unverifiable Instruction", error.raw,
                f"No spatial footprint can be derived for this block: {error.reason}.", error.block,
                "Use only the instructions allowed by the output rules, and give every G01 a feed rate.",
            ))
            continue
        _log(verbose, f"  Step 2  Parser    : {len(program.steps)} motion blocks exhausted into Z^3 "
                      f"(Store discarded; prover sees only V_start, V_final, V_path)")

        # Step 3: Evaluator (SL Prover, one triple per motion block)
        prover.initialize(program.v_home)  # Σ0; HeapInitError is a configuration fault and propagates
        outcomes: List[StepOutcome] = []
        try:
            for step in program.steps:
                outcomes.append(prover.execute(step))
                premise = "V_path ∗ (C_env ∪ C_stock)" if step.command == "G00" else "V_path ∗ C_env"
                _log(verbose, f"  Step 3  Evaluator : {step.label:<4} {step.command}  "
                              f"{premise} holds -> triple proved  "
                              f"(|V_path| = {step.v_path.voxel_count:,} voxels)")
        except SpatialDataRace as race:
            record = race.record
            label = f" ({record.resource_label})" if record.resource_label else ""
            _log(verbose, f"  Step 3  Evaluator : {race.step.label:<4} {race.step.command}  "
                          f"FAULT, separating conjunction fails against {record.violated_resource}{label}")
            _log(verbose, f"            B_conflict = {race.b_conflict.format()}  "
                          f"(mm: {race.b_conflict.format_mm(mu)})")
            reject("SPATIAL_DATA_RACE", generate_sl_feedback(record), record)
            continue

        # Step 4: Terminate on proof
        records.append(IterationRecord(iteration, gcode, "PROVED", outcomes=tuple(outcomes)))
        removed = sum(o.removed_voxels for o in outcomes)
        _log(verbose, f"  Step 4  Terminate : {{Σ0}} C_SL {{Σ_final}} proved for all "
                      f"{len(program.steps)} motion blocks; {removed:,} stock voxels removed")
        _log(verbose, "\n" + "=" * 78)
        _log(verbose, f" ✔ Correct-by-Construction Verified (iteration {iteration}, "
                      f"{agent.calls} generator call{'s' if agent.calls != 1 else ''})")
        _log(verbose, "=" * 78)
        _log(verbose, _indent(gcode, "  "))
        if output_path:
            with open(output_path, "w", encoding="utf-8") as handle:
                handle.write(gcode + "\n")
            _log(verbose, f"\nVerified program written to {output_path}")
        return PipelineResult(True, gcode, tuple(records), tuple(history), program)

    _log(verbose, "\n" + "=" * 78)
    _log(verbose, f" ✘ NOT VERIFIED after {config.max_iterations} iterations; no program may be deployed")
    _log(verbose, "=" * 78)
    return PipelineResult(False, None, tuple(records), tuple(history))


# ======================================================================================
# End-to-end mock run
# ======================================================================================
if __name__ == "__main__":
    INTENT = (
        "Run a finishing pass from the origin to X20 Y10 at Z5 with feed F100, "
        "then park the tool at X-245 Y0 next to Clamp_1 for inspection."
    )
    result = run_neuro_symbolic_pipeline(INTENT)

    # ---- Self-checks ----------------------------------------------------------------
    first, second = result.iterations
    assert result.verified and len(result.iterations) == 2
    assert first.outcome == "SPATIAL_DATA_RACE" and first.record.resource_label == "Clamp_1"
    assert first.record.line_number == "N20" and first.record.motion_mode == "G00"
    assert first.record.bounding_box.x_min == -2521 and first.record.bounding_box.z_max == -700
    assert all(h in first.feedback for h in (HEADER_SYSTEM_ERROR, HEADER_CONFLICT_DOMAIN, HEADER_DIRECTIVE))
    assert "between lines N10 and N30" in first.feedback
    assert second.outcome == "PROVED" and result.gcode == extract_gcode(_DRAFT_CORRECTED)
    assert [m["role"] for m in result.history] == ["user", "assistant", "user", "assistant"]
    assert result.history[2]["content"] == first.feedback          # the B_conflict prompt was appended

    quiet = dict(verbose=False)
    grounded = ground_workspace(PipelineConfig())

    # MAX_ITERATIONS safeguard: a generator that never fixes the crash is stopped after 5 calls.
    stubborn = LLM_Agent("", [_DRAFT_COLLIDING])
    stopped = run_neuro_symbolic_pipeline(INTENT, agent=stubborn, **quiet)
    assert not stopped.verified and stopped.gcode is None
    assert stubborn.calls == MAX_ITERATIONS and len(stopped.iterations) == MAX_ITERATIONS

    # Parser rejection (an arc) and block-number rejection both loop back with Fig. 5-structured feedback.
    arc = "```gcode\nN10 G00 Z50.0\nN20 G02 X10.0 Y10.0 I5.0 J0.0\n```"
    unnumbered = "```gcode\nG00 Z50.0\nG00 X-245.0 Y0.0\n```"
    agent = LLM_Agent("", [arc, unnumbered, _DRAFT_CORRECTED])
    mixed = run_neuro_symbolic_pipeline(INTENT, agent=agent, **quiet)
    assert mixed.verified and [r.outcome for r in mixed.iterations] == ["PARSER_REJECTION", "BLOCK_NUMBERING", "PROVED"]
    assert "between lines N10 and N30" in mixed.iterations[0].feedback
    assert "circular interpolation" in mixed.iterations[0].feedback

    # Leaving the machine envelope is a Spatial Data Race against the workspace bounds.
    runaway = "```gcode\nN10 G00 Z50.0\nN20 G00 X-450.0\n```"
    out = run_neuro_symbolic_pipeline(INTENT, agent=LLM_Agent("", [runaway, _DRAFT_CORRECTED]), **quiet)
    assert out.verified and out.iterations[0].record.resource_label == "Machine travel limits"

    # The system prompt carries the Fig. 4 elements and the RAG bounds produced by the extractor.
    prompt = build_system_prompt(grounded)
    assert "N10, N20, N30" in prompt and "ACTIVE_TOOL: T01" in prompt and "SAFE_Z_RETRACT: Z50.0" in prompt
    assert grounded.obstacles[0].rag_line() in prompt

    print("\n[self-check] all assertions passed")
