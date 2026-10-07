#!/usr/bin/env python3
"""
parser.py -- The Discretization Engine
======================================

The Parser half of the Parser-Prover Handshake (Lee, "Separation Logic for Verifying
Physical Collisions of CNC Programs", Sec. 2.2, 3-5; Lee, "Correct-by-Construction G-Code
Generation", Sec. 3.4.1, 4.2-4.3).

It reads G-code, keeps every continuous quantity in the Store (s), and emits only discrete
spatial literals. The SL Prover never sees a coordinate in mm, a feed rate or a mode flag.
It receives a MotionStep per motion block, holding V_start, V_final and V_path as subsets
of Z^3.

    G-code text --Parser.iter_steps()--> MotionStep*  --> sl_prover.py
                   (Store s: position, feed, modes, #variables)

Footprint construction
----------------------
    c          = S_grid(p) = ⌊p·μ⌋                              (cad_extractor.s_grid, exact Decimal)
    V_tool^ε   = V_tool ⊕ B_ε                                   (Paper 2, Eq. 5-6)
    V_start    = {c_start} ⊕ V_tool^ε,   V_final = {c_final} ⊕ V_tool^ε
    G00:  V_path = path_box(c_start, c_final) ⊕ V_tool ⊕ B_ε      (Paper 1, Eq. 4)
    G01:  V_path = path_lin(c_start, c_final) ⊕ V_tool ⊕ B_(ε+1)  (Paper 1, Eq. 2, with one extra voxel)
    V_sweep = path ⊕ V_tool                                       (Paper 1, Eq. 7: nominal swept volume, no ε)

V_path is what the prover checks for collisions. V_sweep is the region the prover treats as
material actually removed by a G01. Removing the ε-padded V_path instead would delete real
material within ε of every cut.

Why G01 gets one extra voxel
----------------------------
Bresenham picks one voxel per step along the driving axis, but the continuous segment
between two real points can cross cells that Bresenham skips. Example (μ = 1): the segment
from (0.9, 0) to (2.0, 1.95) passes through cell (1, 0), and Bresenham from (0,0) to (2,1)
returns {(0,0), (1,1), (2,1)}. With ε = 0 the paper's footprint would miss a real contact.

Every cell the continuous segment touches lies within L∞ distance 1 of some Bresenham
voxel. Bresenham's minor-axis error is at most ½. The continuous point stays within ½ of
the nearest step along each axis. The real endpoints sit anywhere inside their cells, which
adds up to one cell. So path_lin ⊕ B_1 covers the true swept cells.

Adding that voxel keeps the full ε as physical clearance instead of silently spending one
voxel of it on discretization error. `bresenham_margin=0` reproduces the paper's formula
exactly. G00 needs no extra margin: the continuous rapid stays inside the box spanned by its
endpoints, and since ⌊·⌋ is monotone, that box's cells are exactly path_box(c_start, c_final).

V_start and V_final carry the ε envelope (Paper 1's V_tool is "padded by safety margin ε",
Fig. 3). This gives V_start ⊆ V_path, and step k's V_final equals step k+1's V_start.
Per our earlier decision, only the tool is padded here; obstacles enter the heap nominal
(cad_extractor.PaddedObstacle.nominal_grid).

Representation
--------------
A footprint is a VoxelRegion, a union of GridBoxes. Minkowski sums of boxes are boxes,
so every footprint is exact with no per-voxel enumeration, and intersecting it with the
box-shaped Environment is O(#boxes). Enumerating instead would be costly: one padded G00
in the mock is already 526,071 voxels. `to_voxel_set()` gives the strict set when it is small.

Supported G-code (fail-closed)
------------------------------
Motion G00, G01. Modes G90/G91 (absolute/incremental), G20/G21 (inch/mm).
No-ops: G4 G17 G18 G19 G40 G43 G49 G54 G61 G64 G80 G94; M-codes except M98/M99; S T H P D O words.
Store assignments `#name = value` and word references `X#name` (Paper 1, Sec. 4.1).
Everything else that could move the tool or change its frame is rejected with a GCodeError
rather than skipped: arcs, G28/G30, cutter compensation, other work offsets, canned cycles,
rotary axes, subprograms. A verifier that ignored a motion it cannot model would not be
sound. Coordinates are taken in G54, which is assumed to coincide with the CAD origin.

Note: on Python <= 3.9 the built-in `parser` module shadows this file when it is imported
as `parser`. Python 3.10+ (which removed that module) is required for `import parser`.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from decimal import Decimal
from functools import cached_property
from numbers import Integral
from typing import Dict, FrozenSet, Iterable, Iterator, List, Optional, Sequence, Tuple, Union

from cad_extractor import GridBox, epsilon_to_voxels, s_grid

__all__ = [
    "Voxel",
    "VoxelRegion",
    "ToolVolume",
    "Store",
    "MotionStep",
    "CompiledProgram",
    "GCodeError",
    "Parser",
    "path_lin",
    "path_box",
]

Voxel = Tuple[int, int, int]
_MM_PER_INCH = Decimal("25.4")


# ======================================================================================
# Discrete trajectories (Paper 1, Sec. 3.3-3.5)
# ======================================================================================
def path_lin(c_start: Voxel, c_target: Voxel) -> List[Voxel]:
    """3D Bresenham line from c_start to c_target, both included (Paper 1, Eq. 2-3).

    The coordinate with the largest |Δ| drives the loop. The two minor coordinates are
    stepped by integer error terms, so the result is pure Z^3 arithmetic: a 26-connected
    sequence of max|Δ|+1 voxels whose minor-axis deviation from the ideal line is at most ½.
    """
    x, y, z = (int(v) for v in c_start)
    x1, y1, z1 = (int(v) for v in c_target)
    dx, dy, dz = abs(x1 - x), abs(y1 - y), abs(z1 - z)
    sx = 1 if x1 > x else -1
    sy = 1 if y1 > y else -1
    sz = 1 if z1 > z else -1
    points: List[Voxel] = [(x, y, z)]

    if dx >= dy and dx >= dz:  # X drives
        e_y, e_z = 2 * dy - dx, 2 * dz - dx
        for _ in range(dx):
            x += sx
            if e_y >= 0:
                y += sy
                e_y -= 2 * dx
            if e_z >= 0:
                z += sz
                e_z -= 2 * dx
            e_y += 2 * dy
            e_z += 2 * dz
            points.append((x, y, z))
    elif dy >= dx and dy >= dz:  # Y drives
        e_x, e_z = 2 * dx - dy, 2 * dz - dy
        for _ in range(dy):
            y += sy
            if e_x >= 0:
                x += sx
                e_x -= 2 * dy
            if e_z >= 0:
                z += sz
                e_z -= 2 * dy
            e_x += 2 * dx
            e_z += 2 * dz
            points.append((x, y, z))
    else:  # Z drives
        e_x, e_y = 2 * dx - dz, 2 * dy - dz
        for _ in range(dz):
            z += sz
            if e_x >= 0:
                x += sx
                e_x -= 2 * dz
            if e_y >= 0:
                y += sy
                e_y -= 2 * dz
            e_x += 2 * dx
            e_y += 2 * dy
            points.append((x, y, z))
    return points


def path_box(c_start: Voxel, c_target: Voxel) -> GridBox:
    """Worst-case Cartesian box for G00 rapid travel (Paper 1, Eq. 4).

    path_box(a, b) = {c ∈ Z^3 | min(a, b) ≤ c ≤ max(a, b)}, taken pointwise, since the
    controller may move the axes uncoordinated.
    """
    (ax, ay, az), (bx, by, bz) = c_start, c_target
    return GridBox(min(ax, bx), max(ax, bx), min(ay, by), max(ay, by), min(az, bz), max(az, bz))


# ======================================================================================
# Voxel regions: unions of boxes
# ======================================================================================
def _box_key(box: GridBox) -> Tuple[int, ...]:
    return (box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max)


def _box_sum(a: GridBox, b: GridBox) -> GridBox:
    """a ⊕ b for two boxes, which is again a box: [a_lo + b_lo, a_hi + b_hi] per axis."""
    return GridBox(
        a.x_min + b.x_min, a.x_max + b.x_max,
        a.y_min + b.y_min, a.y_max + b.y_max,
        a.z_min + b.z_min, a.z_max + b.z_max,
    )


def _union_length(spans: List[Tuple[int, int]]) -> int:
    """Total length of a union of half-open integer intervals."""
    spans.sort()
    total = 0
    lo, hi = spans[0]
    for s_lo, s_hi in spans[1:]:
        if s_lo > hi:
            total += hi - lo
            lo, hi = s_lo, s_hi
        elif s_hi > hi:
            hi = s_hi
    return total + (hi - lo)


def _union_area_yz(boxes: Sequence[GridBox]) -> int:
    ys = sorted({b.y_min for b in boxes} | {b.y_max + 1 for b in boxes})
    area = 0
    for y0, y1 in zip(ys, ys[1:]):
        spans = [(b.z_min, b.z_max + 1) for b in boxes if b.y_min <= y0 and b.y_max + 1 >= y1]
        if spans:
            area += (y1 - y0) * _union_length(spans)
    return area


def _merge_intervals(spans: List[Tuple[int, int]]) -> Tuple[Tuple[int, int], ...]:
    """Union of inclusive integer intervals, with touching intervals fused ([0,3] + [4,6] = [0,6])."""
    spans.sort()
    merged: List[Tuple[int, int]] = []
    lo, hi = spans[0]
    for s_lo, s_hi in spans[1:]:
        if s_lo <= hi + 1:
            if s_hi > hi:
                hi = s_hi
        else:
            merged.append((lo, hi))
            lo, hi = s_lo, s_hi
    merged.append((lo, hi))
    return tuple(merged)


def _compact_xy(boxes: Sequence[GridBox], z_min: int, z_max: int) -> List[GridBox]:
    """Disjoint re-decomposition of boxes that share one Z range: sweep X, merge the active Y
    intervals per slab, and fuse neighbouring slabs whose intervals are identical."""
    starts: Dict[int, List[int]] = {}
    ends: Dict[int, List[int]] = {}
    for k, box in enumerate(boxes):
        starts.setdefault(box.x_min, []).append(k)
        ends.setdefault(box.x_max + 1, []).append(k)
    active: set = set()
    out: List[GridBox] = []
    pending: Optional[Tuple[int, Tuple[Tuple[int, int], ...]]] = None
    for x in sorted(set(starts) | set(ends)):
        for k in ends.get(x, ()):
            active.discard(k)
        for k in starts.get(x, ()):
            active.add(k)
        intervals = _merge_intervals([(boxes[k].y_min, boxes[k].y_max) for k in active]) if active else ()
        if pending is not None and pending[1] == intervals:
            continue
        if pending is not None:
            x_start, spans = pending
            out.extend(GridBox(x_start, x - 1, y0, y1, z_min, z_max) for y0, y1 in spans)
        pending = (x, intervals) if intervals else None
    return out


def _union_volume(boxes: Sequence[GridBox]) -> int:
    """Exact voxel count of a union of boxes (sweep over X slabs, then Y strips, then Z spans)."""
    if len(boxes) == 1:
        return boxes[0].voxel_count
    xs = sorted({b.x_min for b in boxes} | {b.x_max + 1 for b in boxes})
    volume = 0
    for x0, x1 in zip(xs, xs[1:]):
        active = [b for b in boxes if b.x_min <= x0 and b.x_max + 1 >= x1]
        if active:
            volume += (x1 - x0) * _union_area_yz(active)
    return volume


@dataclass(frozen=True)
class VoxelRegion:
    """A finite subset of Z^3, stored as a union of (possibly overlapping) GridBoxes."""

    boxes: Tuple[GridBox, ...]

    def __post_init__(self) -> None:
        boxes = tuple(self.boxes)
        if not boxes:
            raise ValueError("a VoxelRegion needs at least one box")
        for box in boxes:
            if not isinstance(box, GridBox):
                raise TypeError(f"VoxelRegion boxes must be GridBox instances, got {type(box).__name__}")
        object.__setattr__(self, "boxes", tuple(sorted(set(boxes), key=_box_key)))

    # ---- construction ---------------------------------------------------------------
    @classmethod
    def from_box(cls, box: GridBox) -> "VoxelRegion":
        return cls((box,))

    @classmethod
    def from_point(cls, voxel: Voxel) -> "VoxelRegion":
        x, y, z = voxel
        return cls((GridBox(x, x, y, y, z, z),))

    @classmethod
    def from_trajectory(cls, voxels: Sequence[Voxel]) -> "VoxelRegion":
        """Merge consecutive voxels that advance along a single axis into one box per run.
        An axis-aligned move becomes a single box."""
        if not voxels:
            raise ValueError("trajectory is empty")
        boxes: List[GridBox] = []
        run_start = prev = tuple(voxels[0])
        run_axis: Optional[int] = None
        for voxel in voxels[1:]:
            voxel = tuple(voxel)
            moved = [i for i in range(3) if voxel[i] != prev[i]]
            if len(moved) == 1 and abs(voxel[moved[0]] - prev[moved[0]]) == 1 and run_axis in (None, moved[0]):
                run_axis = moved[0]
            else:
                boxes.append(path_box(run_start, prev))
                run_start, run_axis = voxel, None
            prev = voxel
        boxes.append(path_box(run_start, prev))
        return cls(tuple(boxes))

    # ---- algebra --------------------------------------------------------------------
    def minkowski(self, other: "VoxelRegion") -> "VoxelRegion":
        """self ⊕ other, exact: the union of the pairwise box sums."""
        return VoxelRegion(tuple(_box_sum(a, b) for a in self.boxes for b in other.boxes))

    def dilate(self, k: int) -> "VoxelRegion":
        """self ⊕ B_k."""
        return VoxelRegion(tuple(box.dilate(k) for box in self.boxes))

    def compact(self) -> "VoxelRegion":
        """Exact re-decomposition into fewer boxes (same voxels). Boxes are grouped by Z range,
        and each group's XY union is rebuilt by an X sweep. Boxes within a group come out
        disjoint, while different groups may still overlap. A Minkowski sum of a Bresenham
        path with a 103-row end mill shrinks from ~50,000 overlapping boxes to about a thousand."""
        groups: Dict[Tuple[int, int], List[GridBox]] = {}
        for box in self.boxes:
            groups.setdefault((box.z_min, box.z_max), []).append(box)
        out: List[GridBox] = []
        for (z_min, z_max), boxes in groups.items():
            out.extend(_compact_xy(boxes, z_min, z_max))
        return VoxelRegion(tuple(out))

    def translate(self, offset: Voxel) -> "VoxelRegion":
        dx, dy, dz = offset
        return VoxelRegion(tuple(
            GridBox(b.x_min + dx, b.x_max + dx, b.y_min + dy, b.y_max + dy, b.z_min + dz, b.z_max + dz)
            for b in self.boxes
        ))

    # ---- queries --------------------------------------------------------------------
    @cached_property
    def bbox(self) -> GridBox:
        return GridBox(
            min(b.x_min for b in self.boxes), max(b.x_max for b in self.boxes),
            min(b.y_min for b in self.boxes), max(b.y_max for b in self.boxes),
            min(b.z_min for b in self.boxes), max(b.z_max for b in self.boxes),
        )

    @cached_property
    def voxel_count(self) -> int:
        """Exact |region| (overlaps counted once)."""
        return _union_volume(self.boxes)

    def contains(self, voxel: Voxel) -> bool:
        return any(box.contains(voxel) for box in self.boxes)

    def intersects(self, box: GridBox) -> bool:
        return box.intersection(self.bbox) is not None and any(b.intersection(box) is not None for b in self.boxes)

    def intersections(self, box: GridBox) -> Tuple[GridBox, ...]:
        """The nonempty pieces of self ∩ box (as boxes; they may overlap)."""
        pieces = (b.intersection(box) for b in self.boxes)
        return tuple(p for p in pieces if p is not None)

    def to_voxel_set(self, max_voxels: int = 2_000_000) -> FrozenSet[Voxel]:
        """The strict discrete set. Guarded, because footprints grow as (2ε+1)^3."""
        if self.voxel_count > max_voxels:
            raise MemoryError(
                f"region holds {self.voxel_count:,} voxels (limit {max_voxels:,}); "
                "use the box representation instead"
            )
        voxels = set()
        for box in self.boxes:
            voxels.update(box.iter_voxels())
        return frozenset(voxels)


# ======================================================================================
# Tool volume
# ======================================================================================
def _check_mu(mu: int) -> int:
    if isinstance(mu, bool) or not isinstance(mu, Integral) or mu <= 0:
        raise ValueError(f"grid multiplier μ must be a positive integer (μ ∈ Z+), got {mu!r}")
    return int(mu)


@dataclass(frozen=True)
class ToolVolume:
    """V_tool: voxel offsets relative to the cell holding the tool control point (tool tip centre).

    The offsets are conservative for any control point inside that cell. Every cell the real
    tool can touch while its control point lies anywhere in [0, 1/μ)^3 is included. This is
    what makes {c} ⊕ V_tool cover the physical tool and not just a sampled copy of it.
    """

    name: str
    region: VoxelRegion

    @classmethod
    def point(cls) -> "ToolVolume":
        return cls("point", VoxelRegion.from_point((0, 0, 0)))

    @classmethod
    def box(cls, half_x: int, half_y: int, half_z: int) -> "ToolVolume":
        return cls(f"box ±({half_x}, {half_y}, {half_z}) voxels",
                   VoxelRegion.from_box(GridBox(-half_x, half_x, -half_y, half_y, -half_z, half_z)))

    @classmethod
    def flat_end_mill(cls, radius_mm: Union[str, float, Decimal], length_mm: Union[str, float, Decimal], mu: int) -> "ToolVolume":
        """Flat end mill of radius r and length L, tip at the control point, extending along +Z.

        In units of h = 1/μ, the control-point cell is [0,1)^3 and the tool occupies
        [0,1)^3 ⊕ (disc(r) × [0, L]). Along Z the covered cells are exactly 0 ≤ k < 1 + L·μ.
        Column (i, j) is kept if its closed distance from the unit square is ≤ r·μ, which can
        include at most one extra boundary ring when r·μ is an integer. That errs on the safe side.
        """
        mu = _check_mu(mu)
        r_u = Decimal(str(radius_mm)) * mu
        l_u = Decimal(str(length_mm)) * mu
        if r_u <= 0 or l_u < 0:
            raise ValueError(f"end mill needs radius > 0 and length >= 0, got r={radius_mm!r}, L={length_mm!r}")
        k_max = math.ceil(l_u + 1) - 1
        r_sq = r_u * r_u
        reach = math.floor(r_u) + 1
        boxes = []
        for i in range(-reach, reach + 1):
            d_i = max(0, abs(i) - 1)
            remaining = r_sq - d_i * d_i
            if remaining < 0:
                continue
            j = math.isqrt(math.floor(remaining)) + 1  # largest |j| with (|j|-1)^2 <= remaining
            boxes.append(GridBox(i, i, -j, j, 0, k_max))
        return cls(f"flat end mill R{radius_mm} L{length_mm}", VoxelRegion(tuple(boxes)))


# ======================================================================================
# Store, outputs and errors
# ======================================================================================
@dataclass
class Store:
    """The Store s: Var -> values, used by the Parser only (Paper 1, Sec. 2.3, Eq. 5).

    Positions are exact Decimals in mm, in G54 work coordinates (= CAD origin).
    """

    position: Tuple[Decimal, Decimal, Decimal] = (Decimal(0), Decimal(0), Decimal(0))
    feed: Optional[Decimal] = None          # mm/min (inch input is converted)
    motion_mode: Optional[str] = None       # "G00" | "G01"
    absolute: bool = True                   # G90 (True) / G91 (False)
    inch: bool = False                      # G20 (True) / G21 (False)
    variables: Dict[str, Decimal] = field(default_factory=dict)


@dataclass(frozen=True)
class MotionStep:
    """One motion block, fully exhausted into Z^3 (the input to one SL triple).

    index       : 0-based position among the emitted steps.
    line_index  : 1-based source line.
    block       : N word as written ("N10"), or None.
    raw         : source text of the block (for feedback prompts only; never evaluated).
    command     : "G00" or "G01".
    c_start     : S_grid of the start position (tool control point).
    c_final     : S_grid of the target position.
    trajectory  : Bresenham voxels for G01; empty for G00.
    path        : discrete path before padding (path_lin or path_box).
    v_start, v_final, v_path : footprints, see the module docstring.
    v_sweep     : path ⊕ V_tool without ε, the nominal material-removal region.
    """

    index: int
    line_index: int
    block: Optional[str]
    raw: str
    command: str
    c_start: Voxel
    c_final: Voxel
    trajectory: Tuple[Voxel, ...]
    path: VoxelRegion
    v_start: VoxelRegion
    v_final: VoxelRegion
    v_path: VoxelRegion
    v_sweep: VoxelRegion

    @property
    def label(self) -> str:
        return self.block or f"line {self.line_index}"


@dataclass(frozen=True)
class CompiledProgram:
    """The Parser's complete output for {Σ0} C_SL {Σ_final} (Paper 1, Sec. 5.1).

    Environment and Stock are added to Σ0 by the prover, from cad_extractor's nominal boxes.
    """

    mu: int
    epsilon_vox: int
    bresenham_margin: int
    tool_name: str
    home: Voxel
    v_home: VoxelRegion                 # R(V_start, Tool) in Σ0
    steps: Tuple[MotionStep, ...]
    final_position: Voxel
    v_end: VoxelRegion                  # R(V_end, Tool) in Σ_final


class GCodeError(ValueError):
    """A block the Parser refuses to exhaust (unsupported, malformed or underspecified)."""

    def __init__(self, line_index: int, block: Optional[str], raw: str, reason: str) -> None:
        self.line_index = line_index
        self.block = block
        self.raw = raw
        self.reason = reason
        where = f"line {line_index}" + (f" ({block})" if block else "")
        super().__init__(f"{where}: {reason} -- {raw!r}")


# ======================================================================================
# G-code vocabulary
# ======================================================================================
_PAREN_COMMENT = re.compile(r"\([^()]*\)")
_NUMBER = r"[+-]?(?:\d+\.?\d*|\.\d+)"
_VAR_NAME = r"(?:[A-Za-z_]\w*|\d+)"
_WORD = re.compile(rf"\s*([A-Za-z])\s*({_NUMBER}|#\s*{_VAR_NAME})\s*")
_ASSIGNMENT = re.compile(rf"^(?:N(\d+)\s*)?#\s*({_VAR_NAME})\s*=\s*({_NUMBER}|#\s*{_VAR_NAME})\s*$", re.IGNORECASE)

_G_NOOP = {4, 17, 18, 19, 40, 43, 49, 54, 61, 64, 80, 94}
_G_REJECT = {
    2: "circular interpolation (G02/G03) is not modelled; approximate the arc with G01 segments",
    3: "circular interpolation (G02/G03) is not modelled; approximate the arc with G01 segments",
    28: "reference-point return travels through a machine-defined path",
    30: "reference-point return travels through a machine-defined path",
    41: "cutter radius compensation changes the toolpath geometry",
    42: "cutter radius compensation changes the toolpath geometry",
    53: "machine-coordinate moves bypass the G54 frame",
    92: "G92 redefines the coordinate frame",
    **{code: "only G54 (the CAD origin) is supported as a work offset" for code in (55, 56, 57, 58, 59)},
    **{code: "canned cycles expand into motions that are not modelled" for code in (73, 74, 76, 81, 82, 83, 84, 85, 86, 87, 88, 89)},
}
_M_REJECT = {98: "subprogram calls are not followed", 99: "subprogram returns are not followed"}
_NOOP_LETTERS = set("STHPDO")
_REJECT_LETTERS = {
    **{letter: "rotary axes are not modelled by the 3-axis Parser" for letter in "ABC"},
    **{letter: "secondary linear axes are not modelled" for letter in "UVW"},
    **{letter: "arc/cycle parameters imply unsupported motion" for letter in "IJKRQL"},
}


def _integral_code(value: Decimal) -> Optional[int]:
    return int(value) if value == value.to_integral_value() else None


# ======================================================================================
# The Parser
# ======================================================================================
class Parser:
    """Exhausts continuous G-code kinematics into discrete MotionSteps.

    Parameters
    ----------
    mu               : grid multiplier μ (voxel edge = 1/μ mm).
    epsilon_mm       : safety margin ε in mm, converted to ε_vox = ⌈ε·μ⌉.
    tool             : V_tool (default: a point).
    home_mm          : initial control-point position (default X0 Y0 Z0).
    bresenham_margin : extra dilation for G01 (default 1, which is sound; 0 is the paper's formula).
    """

    def __init__(
        self,
        mu: int,
        epsilon_mm: Union[str, float, Decimal],
        tool: Optional[ToolVolume] = None,
        *,
        home_mm: Tuple[Union[str, float, Decimal], ...] = ("0", "0", "0"),
        bresenham_margin: int = 1,
    ) -> None:
        self.mu = _check_mu(mu)
        self.epsilon_mm = Decimal(str(epsilon_mm))
        self.epsilon_vox = epsilon_to_voxels(self.epsilon_mm, self.mu)
        if isinstance(bresenham_margin, bool) or not isinstance(bresenham_margin, Integral) or bresenham_margin < 0:
            raise ValueError(f"bresenham_margin must be a non-negative integer, got {bresenham_margin!r}")
        self.bresenham_margin = int(bresenham_margin)
        self.tool = tool or ToolVolume.point()
        self.home_mm = tuple(Decimal(str(v)) for v in home_mm)
        if len(self.home_mm) != 3:
            raise ValueError("home_mm needs exactly three coordinates")

        e, m = self.epsilon_vox, self.bresenham_margin
        self._tool_eps = self.tool.region.dilate(e)          # V_tool ⊕ B_ε
        self._tool_eps_lin = self.tool.region.dilate(e + m)  # V_tool ⊕ B_(ε+margin)
        self.store = Store()
        self._steps_emitted = 0
        self.reset()

    # ---- state ----------------------------------------------------------------------
    def reset(self) -> None:
        """Re-initialise the Store: tool at home, no feed, no motion mode, G90 G21."""
        self.store = Store(position=self.home_mm)
        self._steps_emitted = 0

    def _grid(self, point_mm: Tuple[Decimal, Decimal, Decimal]) -> Voxel:
        # G-code numerals are exact decimals, so no float-noise snapping is applied here.
        return tuple(s_grid(v, self.mu, snap_tol_mm=0) for v in point_mm)  # type: ignore[return-value]

    @property
    def position_voxel(self) -> Voxel:
        return self._grid(self.store.position)

    def tool_region_at(self, c: Voxel) -> VoxelRegion:
        """{c} ⊕ V_tool ⊕ B_ε."""
        return self._tool_eps.translate(c)

    # ---- public entry points --------------------------------------------------------
    def iter_steps(self, program: Union[str, Iterable[str]]) -> Iterator[MotionStep]:
        """Yield one MotionStep per motion block, continuing from the current Store."""
        lines = program.splitlines() if isinstance(program, str) else program
        for line_index, text in enumerate(lines, start=1):
            step = self.parse_block(text, line_index)
            if step is not None:
                yield step

    def compile(self, program: Union[str, Iterable[str]]) -> CompiledProgram:
        """Reset the Store, parse the whole program and return the exhausted triple data."""
        self.reset()
        home = self.position_voxel
        steps = tuple(self.iter_steps(program))
        final = self.position_voxel
        return CompiledProgram(
            mu=self.mu,
            epsilon_vox=self.epsilon_vox,
            bresenham_margin=self.bresenham_margin,
            tool_name=self.tool.name,
            home=home,
            v_home=self.tool_region_at(home),
            steps=steps,
            final_position=final,
            v_end=self.tool_region_at(final),
        )

    # ---- block parsing --------------------------------------------------------------
    def _value(self, token: str, fail) -> Decimal:
        token = token.strip()
        if token.startswith("#"):
            name = token[1:].strip().upper()
            if name not in self.store.variables:
                fail(f"variable #{name} is used before it is assigned")
            return self.store.variables[name]
        return Decimal(token)

    def parse_block(self, text: str, line_index: int = 0) -> Optional[MotionStep]:
        """Update the Store with one block; return a MotionStep if the block moves the tool."""
        raw = text.strip()
        block: Optional[str] = None

        def fail(reason: str):
            raise GCodeError(line_index, block, raw, reason)

        # Comments: ';' to end of line, and (...) groups.
        body = raw.split(";", 1)[0]
        body = _PAREN_COMMENT.sub(" ", body)
        if "(" in body or ")" in body:
            fail("unbalanced parenthesis comment")
        body = body.strip()
        if not body or body == "%":
            return None

        # Store assignment: #name = value (Paper 1, Sec. 4.1 "Assignment").
        if "=" in body:
            match = _ASSIGNMENT.match(body)
            if not match:
                fail("only '#name = <number or #name>' assignments are supported")
            if match.group(1):
                block = f"N{match.group(1)}"
            name = match.group(2).upper()
            self.store.variables[name] = self._value(match.group(3), fail)
            return None

        # Tokenize into words; the whole block must be consumed.
        words: List[Tuple[str, str]] = []
        pos = 0
        while pos < len(body):
            match = _WORD.match(body, pos)
            if not match:
                fail(f"unrecognised text at {body[pos:pos + 12]!r}")
            words.append((match.group(1).upper(), match.group(2)))
            pos = match.end()

        for letter, token in words:
            if letter == "N":
                if block is not None:
                    fail("more than one N word in a block")
                if not re.fullmatch(r"\d+", token):
                    fail("N word must be a non-negative integer")
                block = f"N{token}"

        axes: Dict[str, Decimal] = {}
        motion: Optional[str] = None
        units: Optional[bool] = None
        distance: Optional[bool] = None
        feed: Optional[Decimal] = None

        for letter, token in words:
            if letter == "N":
                continue
            value = self._value(token, fail)
            if letter == "G":
                code = _integral_code(value)
                if code in (0, 1):
                    if motion is not None:
                        fail("more than one motion command in a block")
                    motion = "G00" if code == 0 else "G01"
                elif code in (90, 91):
                    distance = code == 90
                elif code in (20, 21):
                    units = code == 20
                elif code in _G_NOOP:
                    pass
                else:
                    fail(_G_REJECT.get(code, f"G{token} is not supported by the SL Parser"))
            elif letter == "M":
                code = _integral_code(value)
                if code is None or code in _M_REJECT:
                    fail(_M_REJECT.get(code, f"M{token} is not supported"))
            elif letter in "XYZ":
                if letter in axes:
                    fail(f"{letter} appears twice in one block")
                axes[letter] = value
            elif letter == "F":
                if value <= 0:
                    fail("feed rate F must be positive")
                feed = value
            elif letter in _NOOP_LETTERS:
                pass
            elif letter in _REJECT_LETTERS:
                fail(_REJECT_LETTERS[letter])
            else:
                fail(f"word {letter}{token} is not supported")

        # Modal updates take effect before the motion in the same block.
        store = self.store
        if units is not None:
            store.inch = units
        if distance is not None:
            store.absolute = distance
        scale = _MM_PER_INCH if store.inch else Decimal(1)
        if feed is not None:
            store.feed = feed * scale
        if motion is not None:
            store.motion_mode = motion

        if not axes:
            return None
        if store.motion_mode is None:
            fail("axis words without an active motion mode (G00 or G01)")
        if store.motion_mode == "G01" and store.feed is None:
            fail("G01 requires a feed rate (F) before or in this block")

        start = store.position
        target = tuple(
            (axes[a] * scale if store.absolute else start[i] + axes[a] * scale) if a in axes else start[i]
            for i, a in enumerate("XYZ")
        )
        store.position = target  # type: ignore[assignment]
        return self._emit(store.motion_mode, start, target, line_index, block, raw)

    def _emit(self, command: str, start_mm, target_mm, line_index: int, block: Optional[str], raw: str) -> MotionStep:
        c_start, c_final = self._grid(start_mm), self._grid(target_mm)
        if command == "G00":
            trajectory: Tuple[Voxel, ...] = ()
            path = VoxelRegion.from_box(path_box(c_start, c_final))
            v_path = path.minkowski(self._tool_eps).compact()
        else:
            trajectory = tuple(path_lin(c_start, c_final))
            path = VoxelRegion.from_trajectory(trajectory)
            v_path = path.minkowski(self._tool_eps_lin).compact()
        step = MotionStep(
            index=self._steps_emitted,
            line_index=line_index,
            block=block,
            raw=raw,
            command=command,
            c_start=c_start,
            c_final=c_final,
            trajectory=trajectory,
            path=path,
            v_start=self.tool_region_at(c_start),
            v_final=self.tool_region_at(c_final),
            v_path=v_path,
            v_sweep=path.minkowski(self.tool.region).compact(),
        )
        self._steps_emitted += 1
        return step


# ======================================================================================
# Mock run
# ======================================================================================
if __name__ == "__main__":
    import random
    from fractions import Fraction

    MU, EPSILON = 10, "2.0"
    PROGRAM = """\
N10 G00 X10.0 Y0.0 Z5.0
N20 G01 X20.0 Y10.0 Z5.0 F100
"""

    parser = Parser(MU, EPSILON)                              # V_tool = point, sound G01 margin
    paper = Parser(MU, EPSILON, bresenham_margin=0)           # Paper's formula, for comparison
    program = parser.compile(PROGRAM)
    paper_steps = paper.compile(PROGRAM).steps

    print(f"=== PARSER MOCK: μ = {MU}, ε = {EPSILON} mm (ε_vox = {program.epsilon_vox}), "
          f"V_tool = {program.tool_name}, G01 margin = {program.bresenham_margin} voxel ===")
    print(f"Home: X0 Y0 Z0 -> c = {program.home}; V_home bounds: {program.v_home.bbox.format()}")
    for step, paper_step in zip(program.steps, paper_steps):
        kind = "path_box" if step.command == "G00" else f"path_lin ({len(step.trajectory)} Bresenham voxels)"
        print()
        print(f"[{step.label}] {step.raw}")
        print(f"   command           : {step.command} via {kind}")
        print(f"   c_start -> c_final: {step.c_start} -> {step.c_final}")
        print(f"   path              : {step.path.voxel_count:,} voxels in {len(step.path.boxes)} box(es); "
              f"bounds {step.path.bbox.format()}")
        print(f"   V_start bounds    : {step.v_start.bbox.format()}   ({step.v_start.voxel_count:,} voxels)")
        print(f"   V_final bounds    : {step.v_final.bbox.format()}   ({step.v_final.voxel_count:,} voxels)")
        print(f"   V_path bounds     : {step.v_path.bbox.format()}")
        print(f"   V_path (mm)       : {step.v_path.bbox.format_mm(MU)}")
        print(f"   |V_path|          : {step.v_path.voxel_count:,} voxels "
              f"(paper formula without margin: {paper_step.v_path.voxel_count:,})")
    store = parser.store
    print()
    print(f"Store after N20 (never passed to the prover): position = "
          f"X{store.position[0]} Y{store.position[1]} Z{store.position[2]}, F = {store.feed}, "
          f"mode = {store.motion_mode}")
    print(f"Σ_final tool region: {program.v_end.bbox.format()}")

    # ---- Self-checks ----------------------------------------------------------------
    rng_c = random.Random(11)
    g00, g01 = program.steps
    assert (g00.c_start, g00.c_final) == ((0, 0, 0), (100, 0, 50))
    assert g00.v_path.bbox == GridBox(-20, 120, -20, 20, -20, 70) and g00.v_path.voxel_count == 141 * 41 * 91
    assert (g01.c_start, g01.c_final) == ((100, 0, 50), (200, 100, 50)) and len(g01.trajectory) == 101
    assert g01.v_path.bbox == GridBox(79, 221, -21, 121, 29, 71)
    assert g01.v_path.voxel_count == 43 * (43 * 43 + 100 * 85)         # diagonal sweep of 43^3 cubes
    assert paper_steps[1].v_path.voxel_count == 41 * (41 * 41 + 100 * 81)
    assert g00.v_final == g01.v_start                                   # heap hand-over between steps
    for step in program.steps:
        assert all(step.v_path.intersects(b) for b in step.v_start.boxes)
        # V_sweep (nominal) lies inside V_path (padded): adding it does not grow the union.
        assert VoxelRegion(step.v_path.boxes + step.v_sweep.boxes).voxel_count == step.v_path.voxel_count

    # compact() is exact: the voxel set is unchanged.
    for _ in range(200):
        boxes = []
        for _ in range(rng_c.randint(1, 8)):
            lo = [rng_c.randint(-6, 6) for _ in range(3)]
            boxes.append(GridBox(lo[0], lo[0] + rng_c.randint(0, 4), lo[1], lo[1] + rng_c.randint(0, 4),
                                 lo[2], lo[2] + rng_c.choice([0, 2])))
        region = VoxelRegion(tuple(boxes))
        packed = region.compact()
        assert packed.to_voxel_set() == region.to_voxel_set()

    rng = random.Random(2026)

    # Bresenham invariants.
    for _ in range(500):
        a = tuple(rng.randint(-30, 30) for _ in range(3))
        b = tuple(rng.randint(-30, 30) for _ in range(3))
        line = path_lin(a, b)
        n = max(abs(b[i] - a[i]) for i in range(3))
        assert line[0] == a and line[-1] == b and len(line) == n + 1
        assert all(max(abs(p[i] - q[i]) for i in range(3)) == 1 for p, q in zip(line, line[1:]))
        for k, p in enumerate(line):  # deviation from the ideal lattice line is at most 1/2 (exact)
            t = Fraction(k, n) if n else Fraction(0)
            assert all(abs(p[i] - (a[i] + t * (b[i] - a[i]))) <= Fraction(1, 2) for i in range(3))

    # The paper's formula misses a crossed cell; the one-voxel margin covers it.
    probe = Parser(1, 0, bresenham_margin=0)
    probe.parse_block("G00 X0.9 Y0 Z0")
    raw_step = probe.parse_block("G01 X2.0 Y1.95 Z0 F100")
    assert not raw_step.v_path.contains((1, 0, 0))
    probe = Parser(1, 0)
    probe.parse_block("G00 X0.9 Y0 Z0")
    assert probe.parse_block("G01 X2.0 Y1.95 Z0 F100").v_path.contains((1, 0, 0))

    # Soundness by sampling: every cell touched by the continuous tool lies inside V_path.
    mill = ToolVolume.flat_end_mill("0.35", "0.6", MU)
    for tool, eps in ((ToolVolume.point(), 0), (mill, 0), (mill, "0.1")):
        for _ in range(60):
            checker = Parser(MU, eps, tool)
            p0 = [Decimal(rng.randint(-4000, 4000)) / 1000 for _ in range(3)]
            p1 = [Decimal(rng.randint(-4000, 4000)) / 1000 for _ in range(3)]
            mode = rng.choice(["G00", "G01"])
            checker.parse_block("G00 X{} Y{} Z{}".format(*p0))
            step = checker.parse_block("{} X{} Y{} Z{} F100".format(mode, *p1))
            for _ in range(300):
                t = rng.random()
                if mode == "G01":
                    q = [float(p0[i]) + t * float(p1[i] - p0[i]) for i in range(3)]
                else:  # any point of the rapid's bounding box is reachable
                    q = [rng.uniform(float(min(p0[i], p1[i])), float(max(p0[i], p1[i]))) for i in range(3)]
                if tool is mill:  # random point of the cylinder
                    ang, rad = rng.uniform(0, 2 * math.pi), 0.35 * math.sqrt(rng.random())
                    q = [q[0] + rad * math.cos(ang), q[1] + rad * math.sin(ang), q[2] + rng.uniform(0, 0.6)]
                assert step.v_path.contains(tuple(math.floor(v * MU) for v in q)), (mode, p0, p1, q)

    # Exact union count agrees with brute-force enumeration.
    small = Parser(MU, "0.3")
    small.parse_block("G00 X0 Y0 Z0")
    s = small.parse_block("G01 X1.3 Y-0.7 Z0.4 F50")
    assert s.v_path.voxel_count == len(s.v_path.to_voxel_set())

    # Modes: G91 incremental, G20 inch, #variables.
    modal = Parser(MU, 0)
    modal.compile("")
    modal.parse_block("G91 G00 X1.25")
    modal.parse_block("X1.25")
    assert modal.position_voxel == (25, 0, 0)
    modal.parse_block("G90 G20 G00 X1")
    assert modal.position_voxel == (254, 0, 0)
    modal.parse_block("G21")
    modal.parse_block("#SAFE = 50")
    modal.parse_block("#Z1 = #SAFE")
    assert modal.parse_block("G00 Z#Z1").c_final == (254, 0, 500)

    # Fail-closed: unsupported or underspecified blocks raise.
    for bad in ("G02 X1 Y1 I1 J0", "G01 X5", "X5", "G28 X0", "G00 A90", "G55 G00 X0", "G01 X1 F0",
                "G00 X1 X2", "G21 #SAFE = 50", "G00 X(unclosed", "G00 Z#UNDEFINED"):
        try:
            Parser(MU, 0).compile(bad)
        except GCodeError:
            continue
        raise AssertionError(f"accepted unsupported block {bad!r}")

    print("\n[self-check] all assertions passed")
