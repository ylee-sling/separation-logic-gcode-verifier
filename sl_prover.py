#!/usr/bin/env python3
"""
sl_prover.py -- The Verifier
============================

The SL Prover half of the Parser-Prover Handshake (Lee, "Separation Logic for Verifying
Physical Collisions of CNC Programs", Sec. 6; Lee, "Correct-by-Construction G-Code
Generation", Sec. 4-5.1).

It keeps the Spatial Heap h : Z^3 ⇀ {Tool, Environment, Stock, Empty} and runs each
MotionStep from parser.py through the G00/G01 inference rules. A step either proves its
triple {P} C {Q} and mutates the heap, or it fails the separating conjunction. The failure
is trapped as a Spatial Data Race carrying a ConflictRecord for sl_feedback_generator.py.

    parser.CompiledProgram --SLProver.verify()--> ProofResult (proved | ConflictRecord)

Heap representation
-------------------
Nothing is stored per voxel.
    Environment : list of (GridBox, label), static, e.g. cad_extractor's nominal_grid.
    Stock       : pairwise-disjoint GridBox fragments in an X-bucketed index, mutated by
                  material removal. A cut only tests the fragments near it.
    Tool        : the VoxelRegion the tool currently owns (V_final of the last step).
    Empty       : implicit, W \\ (Environment ∪ Stock ∪ Tool), where W is the optional workspace.
The separating conjunction R(A, θ1) ∗ R(B, θ2) holds iff A ∩ B = ∅ (Axiom, Eq. 10/13). With A
and B as box unions this is a pairwise box test, and the intersections are exactly
V_conflict = dom(h_swept) ∩ dom(h_env) (Paper 2, Eq. 10). B_conflict (Eq. 11) is their
bounding box, so no voxel list is ever built.

Inference rules (Paper 1, Eq. 11-12; Paper 2, Eq. 7-8)
------------------------------------------------------
    every step : assert R(V_start, Tool), i.e. the tool is where the step says it starts.
    G00        : V_path ∩ (C_env ∪ C_stock) = ∅  ⟹  Tool := V_final, and the rest of V_path is Empty.
    G01        : V_path ∩ C_env = ∅               ⟹  Stock := Stock \\ V_cut; Tool := V_final.
    workspace  : if W is given, V_path ⊆ W. Voxels outside W are not in dom(h), so claiming
                 them is reported as Environment ("Machine travel limits").
A fault leaves the heap untouched and halts the prover (deterministic abort) until reset.

Material removal (deviation from the paper, configurable)
---------------------------------------------------------
The paper removes C_stock ∩ V_path, where V_path is ε-padded. That marks real material within ε
of every cut as Empty. A wall thinner than ε beside a cut disappears from the heap entirely,
and a later G00 straight through it then verifies. The self-check reproduces exactly this.

The default is therefore removal="nominal", which removes C_stock ∩ V_sweep, where
V_sweep = path ⊕ V_tool is the nominal swept volume without ε. Residual error: cells the tool
only partly cuts are marked Empty, an error below one voxel (1/μ mm).
removal="padded" reproduces the paper.

Σ0 conventions
--------------
Environment and Stock that overlap (e.g. a clamp face touching the stock face lands in the same
boundary cells) cannot both own a voxel. Environment keeps it, since it is non-machinable, which
is the conservative choice. The home tool region must be disjoint from both; otherwise Σ0 is
unsatisfiable and HeapInitError is raised.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

from cad_extractor import GridBox
from parser import CompiledProgram, MotionStep, VoxelRegion
from sl_feedback_generator import ConflictRecord

__all__ = [
    "TOOL",
    "ENVIRONMENT",
    "STOCK",
    "EMPTY",
    "box_difference",
    "subtract_boxes",
    "separating_conjunction",
    "SpatialDataRace",
    "HeapStateError",
    "HeapInitError",
    "ProverHalted",
    "StepOutcome",
    "ProofResult",
    "SLProver",
]

TOOL, ENVIRONMENT, STOCK, EMPTY = "Tool", "Environment", "Stock", "Empty"
WORKSPACE_LABEL = "Machine travel limits"


# ======================================================================================
# Box algebra
# ======================================================================================
def box_difference(a: GridBox, b: GridBox) -> List[GridBox]:
    """a \\ b as at most six pairwise-disjoint boxes (X slabs, then Y, then Z)."""
    i = a.intersection(b)
    if i is None:
        return [a]
    pieces: List[GridBox] = []
    if a.x_min < i.x_min:
        pieces.append(GridBox(a.x_min, i.x_min - 1, a.y_min, a.y_max, a.z_min, a.z_max))
    if i.x_max < a.x_max:
        pieces.append(GridBox(i.x_max + 1, a.x_max, a.y_min, a.y_max, a.z_min, a.z_max))
    if a.y_min < i.y_min:
        pieces.append(GridBox(i.x_min, i.x_max, a.y_min, i.y_min - 1, a.z_min, a.z_max))
    if i.y_max < a.y_max:
        pieces.append(GridBox(i.x_min, i.x_max, i.y_max + 1, a.y_max, a.z_min, a.z_max))
    if a.z_min < i.z_min:
        pieces.append(GridBox(i.x_min, i.x_max, i.y_min, i.y_max, a.z_min, i.z_min - 1))
    if i.z_max < a.z_max:
        pieces.append(GridBox(i.x_min, i.x_max, i.y_min, i.y_max, i.z_max + 1, a.z_max))
    return pieces


def subtract_boxes(boxes: Sequence[GridBox], cuts: Sequence[GridBox]) -> List[GridBox]:
    """(∪ boxes) \\ (∪ cuts). Disjoint input boxes stay disjoint in the output."""
    result = list(boxes)
    for cut in cuts:
        next_result: List[GridBox] = []
        for box in result:
            if box.intersection(cut) is None:
                next_result.append(box)
            else:
                next_result.extend(box_difference(box, cut))
        result = next_result
    return result


def _overlaps(a: GridBox, b: GridBox) -> bool:
    return (
        a.x_min <= b.x_max and b.x_min <= a.x_max
        and a.y_min <= b.y_max and b.y_min <= a.y_max
        and a.z_min <= b.z_max and b.z_min <= a.z_max
    )


class _BoxIndex:
    """Pairwise-disjoint boxes bucketed by X range.

    Material removal splits Stock into many fragments, and each cut touches only the few
    near it. Bucketing turns "test every fragment" into "test the fragments in the cut's
    X buckets", and keeps the total volume current in O(1).
    """

    def __init__(self, boxes: Sequence[GridBox] = (), bucket: int = 64) -> None:
        self._bucket = bucket
        self._boxes: Dict[int, GridBox] = {}
        self._buckets: Dict[int, set] = {}
        self._next_id = 0
        self.volume = 0
        for box in boxes:
            self.add(box)

    def _keys(self, box: GridBox) -> range:
        return range(box.x_min // self._bucket, box.x_max // self._bucket + 1)

    def add(self, box: GridBox) -> None:
        key = self._next_id
        self._next_id += 1
        self._boxes[key] = box
        self.volume += box.voxel_count
        for bucket in self._keys(box):
            self._buckets.setdefault(bucket, set()).add(key)

    def _remove(self, key: int) -> GridBox:
        box = self._boxes.pop(key)
        self.volume -= box.voxel_count
        for bucket in self._keys(box):
            members = self._buckets[bucket]
            members.discard(key)
            if not members:
                del self._buckets[bucket]
        return box

    def overlapping(self, box: GridBox) -> List[int]:
        candidates: set = set()
        for bucket in self._keys(box):
            candidates |= self._buckets.get(bucket, set())
        return sorted(k for k in candidates if _overlaps(self._boxes[k], box))

    def intersections(self, region: VoxelRegion) -> Tuple[GridBox, ...]:
        """region ∩ (∪ boxes), as boxes."""
        pieces: List[GridBox] = []
        for box in region.boxes:
            for key in self.overlapping(box):
                pieces.append(box.intersection(self._boxes[key]))  # type: ignore[arg-type]
        return tuple(pieces)

    def subtract(self, region: VoxelRegion) -> None:
        """Remove every voxel of `region`; surviving fragments stay pairwise disjoint."""
        for cut in region.boxes:
            for key in self.overlapping(cut):
                for piece in box_difference(self._remove(key), cut):
                    self.add(piece)

    def boxes(self) -> Tuple[GridBox, ...]:
        return tuple(self._boxes[k] for k in sorted(self._boxes))


def _bbox(boxes: Sequence[GridBox]) -> GridBox:
    return GridBox(
        min(b.x_min for b in boxes), max(b.x_max for b in boxes),
        min(b.y_min for b in boxes), max(b.y_max for b in boxes),
        min(b.z_min for b in boxes), max(b.z_max for b in boxes),
    )


def separating_conjunction(claim: VoxelRegion, owned: Sequence[GridBox]) -> Tuple[GridBox, ...]:
    """Evaluate R(claim, θ1) ∗ R(owned, θ2).

    Returns V_conflict = claim ∩ owned as boxes (they may overlap). An empty tuple means the
    two domains are disjoint and the separating conjunction holds. A non-empty result means
    R(C1, θ1) ∗ R(C2, θ2) ⇒ false: a Spatial Data Race.
    """
    reach = claim.bbox
    pieces: List[GridBox] = []
    for other in owned:
        if other.intersection(reach) is None:
            continue
        for box in claim.boxes:
            piece = box.intersection(other)
            if piece is not None:
                pieces.append(piece)
    return tuple(pieces)


# ======================================================================================
# Results and errors
# ======================================================================================
class SpatialDataRace(Exception):
    """A step's footprint claims voxels already owned by a restricted resource."""

    def __init__(self, record: ConflictRecord, step: MotionStep, conflict_boxes: Tuple[GridBox, ...]) -> None:
        self.record = record
        self.step = step
        self.conflict_boxes = conflict_boxes
        self.b_conflict = _bbox(conflict_boxes)
        label = f" ({record.resource_label})" if record.resource_label else ""
        super().__init__(
            f"Spatial Data Race at {step.label}: {step.command} footprint intersects "
            f"{record.violated_resource}{label} within {self.b_conflict.format()}"
        )


class HeapStateError(RuntimeError):
    """A step does not fit the heap (e.g. R(V_start, Tool) fails). This is a pipeline error, not a collision."""


class HeapInitError(ValueError):
    """Σ0 is unsatisfiable, e.g. the home tool region overlaps a fixture or the stock."""


class ProverHalted(RuntimeError):
    """The prover is in the fault state; call reset() or verify() to start over."""


@dataclass(frozen=True)
class StepOutcome:
    label: str
    command: str
    removed_voxels: int        # Stock -> Empty transitions in this step (G01 only)
    stock_voxels_after: int


@dataclass(frozen=True)
class ProofResult:
    proved: bool
    steps_total: int
    outcomes: Tuple[StepOutcome, ...]
    record: Optional[ConflictRecord] = None
    failed_step: Optional[MotionStep] = None
    b_conflict: Optional[GridBox] = None

    @property
    def steps_verified(self) -> int:
        return len(self.outcomes)


# ======================================================================================
# The prover
# ======================================================================================
class SLProver:
    """Discrete memory manager for the Spatial Heap.

    Parameters
    ----------
    mu        : grid multiplier μ (only used to label ConflictRecords for mm back-projection).
    workspace : optional bounded workspace W in Z^3 (e.g. machine travel limits).
    removal   : "nominal" (default, removes Stock ∩ V_sweep) or "padded" (paper: Stock ∩ V_path).
    """

    def __init__(self, mu: int, *, workspace: Optional[GridBox] = None, removal: str = "nominal") -> None:
        if isinstance(mu, bool) or not isinstance(mu, int) or mu <= 0:
            raise ValueError(f"grid multiplier μ must be a positive integer, got {mu!r}")
        if removal not in ("nominal", "padded"):
            raise ValueError(f"removal must be 'nominal' or 'padded', got {removal!r}")
        self.mu = mu
        self.workspace = workspace
        self.removal = removal
        self._env: List[Tuple[GridBox, Optional[str]]] = []
        self._initial_stock: List[GridBox] = []
        self._stock = _BoxIndex()
        self._tool: Optional[VoxelRegion] = None
        self._fault: Optional[SpatialDataRace] = None

    # ---- heap configuration ---------------------------------------------------------
    def add_environment(self, box: GridBox, label: Optional[str] = None) -> None:
        """Register a static, non-machinable obstacle (use the NOMINAL box; ε lives on the tool)."""
        self._env.append((box, label))
        self.reset()

    def add_stock(self, box: GridBox) -> None:
        """Register raw material. Overlaps with earlier stock are merged into disjoint pieces."""
        self._initial_stock.extend(subtract_boxes([box], self._initial_stock))
        self.reset()

    def reset(self) -> None:
        """Back to the configured Σ0 without a tool: Stock restored, fault cleared."""
        env_boxes = [box for box, _ in self._env]
        self._stock = _BoxIndex(subtract_boxes(self._initial_stock, env_boxes))  # Environment keeps shared voxels
        self._tool = None
        self._fault = None

    # ---- inspection -----------------------------------------------------------------
    @property
    def environment(self) -> Tuple[Tuple[GridBox, Optional[str]], ...]:
        return tuple(self._env)

    @property
    def stock(self) -> Tuple[GridBox, ...]:
        return self._stock.boxes()

    @property
    def stock_voxel_count(self) -> int:
        return self._stock.volume  # fragments are pairwise disjoint

    @property
    def tool(self) -> Optional[VoxelRegion]:
        return self._tool

    @property
    def fault(self) -> Optional[SpatialDataRace]:
        return self._fault

    def status_at(self, voxel: Tuple[int, int, int]) -> Optional[str]:
        """h(c): the owner of one voxel, or None if c ∉ dom(h) (outside W)."""
        if self.workspace is not None and not self.workspace.contains(voxel):
            return None
        if self._tool is not None and self._tool.contains(voxel):
            return TOOL
        if any(box.contains(voxel) for box, _ in self._env):
            return ENVIRONMENT
        x, y, z = voxel
        if self._stock.overlapping(GridBox(x, x, y, y, z, z)):
            return STOCK
        return EMPTY

    # ---- Σ0 -------------------------------------------------------------------------
    def initialize(self, v_home: VoxelRegion) -> None:
        """Establish Σ0 = R(V_home, Tool) ∗ R(C_env, Environment) ∗ R(C_stock, Stock) ∗ R(C_empty, Empty)."""
        self.reset()
        env_hits = separating_conjunction(v_home, [box for box, _ in self._env])
        stock_hits = self._stock.intersections(v_home)
        outside = self._outside_workspace(v_home)
        if env_hits or stock_hits or outside:
            what = ENVIRONMENT if env_hits else STOCK if stock_hits else WORKSPACE_LABEL
            box = _bbox(env_hits or stock_hits or outside)
            raise HeapInitError(f"Σ0 is unsatisfiable: the home tool region overlaps {what} within {box.format()}")
        self._tool = v_home

    # ---- inference rules ------------------------------------------------------------
    def _outside_workspace(self, region: VoxelRegion) -> Tuple[GridBox, ...]:
        if self.workspace is None:
            return ()
        pieces: List[GridBox] = []
        for box in region.boxes:
            pieces.extend(box_difference(box, self.workspace))
        return tuple(pieces)

    def _trap(self, step: MotionStep, resource: str, boxes: Tuple[GridBox, ...], label: Optional[str]) -> None:
        record = ConflictRecord.from_boxes(
            line_number=step.block or "",
            command=step.raw,
            conflict_boxes=boxes,
            violated_resource=resource,
            multiplier=self.mu,
            resource_label=label,
            motion_mode=step.command,
        )
        self._fault = SpatialDataRace(record, step, boxes)
        raise self._fault

    def execute(self, step: MotionStep) -> StepOutcome:
        """Apply one G00/G01 triple. Returns the outcome, or raises SpatialDataRace (heap unchanged)."""
        if self._fault is not None:
            raise ProverHalted(f"prover is in the fault state after {self._fault.step.label}; reset() first")
        if self._tool is None:
            raise HeapStateError("the heap has no tool; call initialize(v_home) or verify(program)")

        # assert R(V_start, Tool)
        if step.v_start != self._tool:
            raise HeapStateError(
                f"{step.label}: assert R(V_start, Tool) failed; the step starts at "
                f"{step.v_start.bbox.format()} but the tool owns {self._tool.bbox.format()}"
            )

        # V_path must stay inside dom(h).
        outside = self._outside_workspace(step.v_path)
        if outside:
            self._trap(step, ENVIRONMENT, outside, WORKSPACE_LABEL)

        # Common premise of G00 and G01: V_path ∗ C_env, so the footprint is disjoint from Environment.
        env_pieces: List[GridBox] = []
        labels: List[str] = []
        for box, label in self._env:
            hits = separating_conjunction(step.v_path, [box])
            if hits:
                env_pieces.extend(hits)
                if label and label not in labels:
                    labels.append(label)
        if env_pieces:
            self._trap(step, ENVIRONMENT, tuple(env_pieces), ", ".join(labels) or None)

        removed = 0
        if step.command == "G00":
            # Rapid premise: V_path ∩ C_stock = ∅ (non-cutting moves travel through Empty only).
            stock_hits = self._stock.intersections(step.v_path)
            if stock_hits:
                self._trap(step, STOCK, stock_hits, None)
        elif step.command == "G01":
            # Material removal: Stock ∩ V_cut becomes Empty.
            cut = step.v_sweep if self.removal == "nominal" else step.v_path
            before = self.stock_voxel_count
            self._stock.subtract(cut)
            removed = before - self.stock_voxel_count
        else:  # pragma: no cover - the parser only emits G00/G01
            raise HeapStateError(f"{step.label}: no inference rule for {step.command}")

        # Postcondition: R(V_final, Tool); V_path \ V_final is now Empty (implicitly).
        self._tool = step.v_final
        return StepOutcome(step.label, step.command, removed, self.stock_voxel_count)

    def verify(self, program: CompiledProgram) -> ProofResult:
        """Prove {Σ0} C_SL {Σ_final} for a whole program, stopping at the first Spatial Data Race."""
        self.initialize(program.v_home)
        outcomes: List[StepOutcome] = []
        for step in program.steps:
            try:
                outcomes.append(self.execute(step))
            except SpatialDataRace as race:
                return ProofResult(False, len(program.steps), tuple(outcomes), race.record, step, race.b_conflict)
        if self._tool != program.v_end:  # Σ_final ≜ R(V_end, Tool) ∗ true
            raise HeapStateError("postcondition R(V_end, Tool) does not hold after the last step")
        return ProofResult(True, len(program.steps), tuple(outcomes))


# ======================================================================================
# Mock run
# ======================================================================================
if __name__ == "__main__":
    from cad_extractor import AnalyticFeature, pad_feature
    from parser import Parser
    from sl_feedback_generator import generate_sl_feedback

    MU, EPSILON = 10, "2.0"

    # Clamp_1 from the cad_extractor mock. By our decision, the heap holds the NOMINAL box;
    # the padded box is the RAG view, and ε is applied to the tool by the parser.
    clamp = pad_feature(
        AnalyticFeature(6, "CYLINDER", (-245.0, 0.0, -100.0), 20.0, (0.0, 0.0, 1.0), (0.0, 30.0)),
        float(EPSILON), MU, "Clamp_1",
    )
    prover = SLProver(MU)
    prover.add_environment(clamp.nominal_grid, clamp.name)

    def run(title: str, gcode: str) -> ProofResult:
        program = Parser(MU, EPSILON).compile(gcode)
        result = prover.verify(program)
        print(f"\n{title}")
        for step in program.steps:
            if result.failed_step is step:
                print(f"  {step.raw:<34} -> FAULT: Spatial Data Race")
                break
            print(f"  {step.raw:<34} -> proved; V_path {step.v_path.bbox.format()}")
        verdict = "PROVED" if result.proved else "FAULT"
        print(f"  Result: {verdict} ({result.steps_verified}/{result.steps_total} steps)")
        return result

    print(f"=== SL PROVER MOCK: μ = {MU}, ε = {EPSILON} mm on the tool ===")
    print("Spatial Heap Σ0")
    print(f"  Environment: {clamp.name} nominal {clamp.nominal_grid.format()}")
    print(f"               (RAG view, padded: {clamp.padded_grid.format()})")
    print(f"  Stock      : none;  Tool: V_home {Parser(MU, EPSILON).compile('').v_home.bbox.format()}")

    as_written = run(
        "Program A, as specified (X-25.0 mm is 200 mm short of the clamp at X ≈ -245 mm):",
        "N10 G01 X20.0 Y10.0 Z5.0 F100\nN20 G00 X-25.0 Y0.0 Z-80.0",
    )
    on_clamp = run(
        "Program B, rapid aimed at the clamp centre (X-245.0):",
        "N10 G01 X20.0 Y10.0 Z5.0 F100\nN20 G00 X-245.0 Y0.0 Z-80.0",
    )

    # Catch the race directly through execute(), the way main_loop.py can stream steps.
    steps = Parser(MU, EPSILON).compile("N10 G01 X20.0 Y10.0 Z5.0 F100\nN20 G00 X-245.0 Y0.0 Z-80.0")
    prover.initialize(steps.v_home)
    try:
        for step in steps.steps:
            prover.execute(step)
        raise AssertionError("the race was not trapped")
    except SpatialDataRace as race:
        record = race.record
        b = race.b_conflict
        print("\nCaught SpatialDataRace -> ConflictRecord")
        print(f"  line / command    : {record.line_number} / {record.command}")
        print(f"  motion mode       : {record.motion_mode}")
        print(f"  violated resource : {record.violated_resource} ({record.resource_label})")
        print(f"  conflict boxes    : {len(record.conflict_boxes)}")
        print(f"  B_conflict (Z^3)  : {b.format()}")
        print(f"  B_conflict (mm)   : {b.format_mm(MU)}")
        print("  heap after fault  : unchanged, tool still at", prover.tool.bbox.format())
        print("\n" + generate_sl_feedback(record))

    # ---- Self-checks ----------------------------------------------------------------
    assert as_written.proved and as_written.steps_verified == 2
    assert not on_clamp.proved and on_clamp.steps_verified == 1
    assert on_clamp.b_conflict == GridBox(-2470, -2250, -20, 120, -820, -700)
    assert record.bounding_box.x_min == -2470 and record.bounding_box.z_max == -700
    try:
        prover.execute(steps.steps[1])
        raise AssertionError("a faulted prover accepted another step")
    except ProverHalted:
        pass

    # Out-of-order step: R(V_start, Tool) must hold.
    prover.initialize(steps.v_home)
    try:
        prover.execute(steps.steps[1])
        raise AssertionError("a step was accepted from the wrong start position")
    except HeapStateError:
        pass

    # Workspace W: leaving dom(h) is reported as Environment (machine travel limits).
    bounded = SLProver(MU, workspace=GridBox(-100, 5000, -100, 5000, -100, 5000))
    out = bounded.verify(Parser(MU, EPSILON).compile("N10 G00 X-30.0"))
    assert not out.proved and out.record.resource_label == WORKSPACE_LABEL
    assert out.b_conflict.x_min == -320 and out.b_conflict.x_max == -101

    # Stock: G01 removes material, G00 into stock is a race, Σ0 gives Environment precedence.
    block = GridBox(100, 300, -100, 100, -100, 0)           # 20 x 20 x 10 mm block, top at Z0
    shop = SLProver(MU)
    shop.add_stock(block)
    shop.add_environment(GridBox(250, 320, -100, 100, -150, -100), "Vise jaw")  # shares the Z-100 layer
    assert shop.status_at((260, 0, -100)) == ENVIRONMENT and shop.status_at((260, 0, -99)) == STOCK
    cut = shop.verify(Parser(MU, EPSILON).compile(
        "N10 G00 X5.0 Y0.0 Z5.0\nN20 G01 Z-5.0 F100\nN30 G01 X25.0\nN40 G01 Z5.0\nN50 G00 X5.0"))
    assert cut.proved, cut.record
    assert shop.status_at((150, 0, -50)) == EMPTY and shop.status_at((150, 30, -50)) == STOCK
    # A point tool removes single voxel lines: N30 cuts X100..250 at Z-50 (151 voxels) and the N40
    # retract cuts X250, Z-49..0 (50 more, since Z-50 was already removed).
    assert [o.removed_voxels for o in cut.outcomes] == [0, 0, 151, 50, 0]
    rapid_into_stock = shop.verify(Parser(MU, EPSILON).compile("N10 G00 X15.0 Y0.0 Z-2.0"))
    assert not rapid_into_stock.proved and rapid_into_stock.record.violated_resource == STOCK

    # Thin wall (0.9 mm < ε) beside a cut: the paper's padded removal erases the wall from the
    # heap, so a later rapid straight through it verifies. Nominal removal keeps it and traps it.
    wall = GridBox(100, 108, -500, 500, -200, 0)
    thin_wall_program = Parser(MU, EPSILON).compile(
        "N10 G00 Z5.0\nN20 G00 X9.0 Y-40.0\nN30 G01 Z-5.0 F100\nN40 G01 Y40.0\nN50 G00 X15.0")
    verdicts = {}
    for policy in ("nominal", "padded"):
        checker = SLProver(MU, removal=policy)
        checker.add_stock(wall)
        verdicts[policy] = checker.verify(thin_wall_program)
    assert not verdicts["nominal"].proved and verdicts["nominal"].failed_step.block == "N50"
    assert verdicts["nominal"].record.violated_resource == STOCK
    assert verdicts["padded"].proved
    # The indexed Stock agrees with plain voxel-set arithmetic (differential test).
    import random

    rng = random.Random(5)

    def rand_box(span: int) -> GridBox:
        lo = [rng.randint(-span, span) for _ in range(3)]
        return GridBox(lo[0], lo[0] + rng.randint(0, 5), lo[1], lo[1] + rng.randint(0, 5),
                       lo[2], lo[2] + rng.randint(0, 5))

    for _ in range(150):
        disjoint: List[GridBox] = []
        for box in (rand_box(6) for _ in range(3)):   # same normalisation as add_stock()
            disjoint.extend(subtract_boxes([box], disjoint))
        index = _BoxIndex(disjoint, bucket=4)
        reference = set()
        for b in index.boxes():
            reference.update(b.iter_voxels())
        for _ in range(rng.randint(1, 4)):
            cut_region = VoxelRegion(tuple(rand_box(8) for _ in range(rng.randint(1, 3))))
            probe_region = VoxelRegion((rand_box(8),))
            expected_hits = reference & set(probe_region.to_voxel_set())
            got_hits = set()
            for piece in index.intersections(probe_region):
                got_hits.update(piece.iter_voxels())
            assert got_hits == expected_hits
            index.subtract(cut_region)
            reference -= set(cut_region.to_voxel_set())
            current = set()
            for b in index.boxes():
                current.update(b.iter_voxels())
            assert current == reference and index.volume == len(reference)

    print("\nThin-wall check (0.9 mm wall, cut 1 mm away, ε = 2 mm, then G00 through the wall):")
    print(f"  removal='nominal' -> FAULT at N50, Stock race within {verdicts['nominal'].b_conflict.format()}")
    print("  removal='padded'  -> PROVED, because the wall was erased from the heap by the ε-padded cut")

    print("\n[self-check] all assertions passed")
