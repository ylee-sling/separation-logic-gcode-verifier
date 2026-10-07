#!/usr/bin/env python3
"""
cad_extractor.py -- The Grounding Engine
========================================

Physical grounding stage of the Neuro-Symbolic Evaluator-Refiner pipeline
(Lee, "Correct-by-Construction G-Code Generation: A Neuro-Symbolic Approach via
Separation Logic", Sec. 3.2-3.3, Listing 1, Figures 3-4).

Pipeline position
-----------------
    STEP file --extract_step()--> ExtractionReport --pad_features()--> PaddedObstacle[]
        |                                                                   |
        |  nominal_grid  --> parser.py / sl_prover.py   (Spatial Heap C_env, Environment)
        |  padded_grid   --> main_loop.py               (RAG OBSTACLE_BOUNDS, Fig. 4)

What this module adds to Listing 1
----------------------------------
1. The axial extent of each cylinder. gp_Cylinder.Location() (Listing 1's "Center XYZ")
   is the origin of the surface's local frame. That is one point on the axis, often an end
   of the face, and it says nothing about how long the cylinder is. A box built from center
   and radius alone can therefore miss most of the obstacle. This module also reads the
   axis direction and the face's V-parameter range, which for a cylinder is the signed
   axial distance from Location(). The resulting AABB is exact for any axis orientation.

2. Unique topology counts. Listing 1 counts TopExp_Explorer visits, and the explorer visits
   a shared sub-shape once per parent. An edge shared by two faces is counted twice, and a
   vertex once per edge occurrence. This module reports both numbers. The unique count uses
   TopTools_IndexedMapOfShape, which compares with IsSame and ignores orientation.

3. One padding, applied in one place. B_ε is a symmetric cube, so for any tool set T and
   environment set E in Z^3:

        (T ⊕ B_ε) ∩ E ≠ ∅    ⟺    T ∩ (E ⊕ B_ε) ≠ ∅

   The prover pads the tool (V_path = path ⊕ V_tool ⊕ B_ε, Paper 2 Eq. 6), and the RAG
   prompt shows the padded obstacle (Paper 2 Sec. 3.3.2). By the identity above these are
   the same constraint. The Spatial Heap must therefore receive the NOMINAL obstacle
   (`nominal_grid`). Padding the heap as well would apply ε twice, and the prover would
   then reject paths that the RAG prompt told the LLM were legal.

Discretization convention (shared with sl_feedback_generator.py)
----------------------------------------------------------------
S_grid(x) = ⌊x·μ⌋, so integer i stands for the cell [i/μ, (i+1)/μ). A continuous box
[a, b] touches cells ⌊aμ⌋..⌊bμ⌋, and a grid box [lo, hi] back-projects to the mm range
[lo/μ, (hi+1)/μ]. The RAG bounds are the back-projection of the padded grid box. This
keeps the numbers the LLM sees identical to what the prover enforces, at the cost of
upper bounds that end one cell past the continuous value (e.g. -222.9 rather than -223.0).

Units: STEPControl_Reader converts to the session length unit, which is millimetres by
default (xstep.cascade.unit = MM), so every continuous value here is in mm.

Usage
-----
    python cad_extractor.py                                 # mock run (no pythonocc needed)
    python cad_extractor.py part.stp --epsilon 2.0 --mu 10  # real extraction
    python cad_extractor.py part.stp --json rag_obstacles.json
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
from dataclasses import dataclass, field
from decimal import ROUND_HALF_EVEN, Decimal
from numbers import Integral, Real
from types import SimpleNamespace
from typing import Dict, Iterable, Iterator, List, Mapping, Optional, Sequence, Tuple, Union

__all__ = [
    "Vec3",
    "AABB",
    "GridBox",
    "AnalyticFeature",
    "SolidFeature",
    "SurfaceRecord",
    "TopologyMetrics",
    "ExtractionReport",
    "PaddedObstacle",
    "StepReadError",
    "SURFACE_LABELS",
    "CONFUSION_MM",
    "s_grid",
    "grid_box_of",
    "epsilon_to_voxels",
    "cylinder_aabb",
    "sphere_aabb",
    "extract_step",
    "extract_solids",
    "pad_feature",
    "pad_features",
    "format_obstacle_bounds",
    "build_rag_payload",
]

Vec3 = Tuple[float, float, float]
Number = Union[int, float, Decimal, str]

# OpenCASCADE Precision::Confusion(): distances below this are geometrically the same.
CONFUSION_MM = 1e-7

# Parameter values at or beyond this magnitude are treated as unbounded
# (OpenCASCADE uses Precision::Infinite() = 2e100).
_UNBOUNDED = 1e50

# Human-readable labels, matching the SURFACE_TYPES table in Listing 1.
SURFACE_LABELS: Dict[str, str] = {
    "PLANE": "PLANE (Flat Surface)",
    "CYLINDER": "CYLINDER (Curved Tube)",
    "CONE": "CONE (Tapered Surface)",
    "SPHERE": "SPHERE (Ball Surface)",
    "TORUS": "TORUS (Donut/O-Ring Surface)",
    "BEZIER": "BEZIER SURFACE (Smooth Freeform)",
    "BSPLINE": "B-SPLINE / NURBS SURFACE (Complex Engineering Surface)",
    "REVOLUTION": "SURFACE OF REVOLUTION",
    "EXTRUSION": "SURFACE OF EXTRUSION",
    "OFFSET": "OFFSET SURFACE",
    "OTHER": "UNKNOWN COMPLEX SURFACE",
}

# Analytic surfaces whose parameters are extracted and turned into obstacles.
_ANALYTIC_KINDS = ("CYLINDER", "SPHERE")


class StepReadError(RuntimeError):
    """OpenCASCADE could not read or transfer the STEP file."""


# ======================================================================================
# Small numeric helpers
# ======================================================================================
def _check_mu(mu: int) -> int:
    if isinstance(mu, bool) or not isinstance(mu, Integral) or mu <= 0:
        raise ValueError(f"grid multiplier μ must be a positive integer (μ ∈ Z+), got {mu!r}")
    return int(mu)


def _to_decimal(value: Number) -> Decimal:
    """Exact decimal form of a coordinate. Floats go through repr(), the shortest round-trip form."""
    if isinstance(value, Decimal):
        result = value
    elif isinstance(value, str):
        result = Decimal(value.strip())
    elif isinstance(value, Real) and not isinstance(value, bool):
        result = Decimal(repr(float(value)))
    else:
        raise TypeError(f"expected a real number, got {value!r}")
    if not result.is_finite():
        raise ValueError(f"coordinate must be finite, got {value!r}")
    return result


def _fmt(value: float, decimals: int = 3) -> str:
    """Fixed-point formatting that never prints '-0.000'."""
    text = f"{float(value):.{decimals}f}"
    if text.startswith("-") and float(text) == 0.0:
        text = text[1:]
    return text


def _fmt_vec(vec: Sequence[float], decimals: int = 3) -> str:
    return "(" + ", ".join(_fmt(c, decimals) for c in vec) + ")"


def _unit(vec: Sequence[float]) -> Vec3:
    x, y, z = (float(c) for c in vec)
    norm = math.sqrt(x * x + y * y + z * z)
    if norm == 0.0 or not math.isfinite(norm):
        raise ValueError(f"axis direction must be a finite non-zero vector, got {tuple(vec)!r}")
    return (x / norm, y / norm, z / norm)


def _cell_lower_mm(index: int, mu: int, decimals: int) -> Decimal:
    """Lower mm edge of grid cell `index` (index/μ), rounded toward -∞."""
    return Decimal((index * 10 ** decimals) // mu).scaleb(-decimals)


def _cell_upper_mm(index: int, mu: int, decimals: int) -> Decimal:
    """Upper mm edge of grid cell `index` ((index+1)/μ), rounded toward +∞."""
    return Decimal(-((-(index + 1) * 10 ** decimals) // mu)).scaleb(-decimals)


# ======================================================================================
# Discretization: S_grid and B_ε
# ======================================================================================
def s_grid(value_mm: Number, mu: int, *, snap_tol_mm: float = CONFUSION_MM) -> int:
    """S_grid(x) = ⌊x·μ⌋ (Paper 1, Sec. 4.2), computed in exact decimal arithmetic.

    B-Rep coordinates carry floating-point noise such as -244.99999999999997. A value within
    `snap_tol_mm` of a cell boundary is snapped to that boundary first, so the noise cannot
    add a spurious cell. The default tolerance is Precision::Confusion(), below which
    OpenCASCADE treats two points as identical. Pass snap_tol_mm=0 for a pure floor.
    """
    mu = _check_mu(mu)
    scaled = _to_decimal(value_mm) * mu
    nearest = scaled.to_integral_value(rounding=ROUND_HALF_EVEN)
    if snap_tol_mm > 0 and abs(scaled - nearest) <= _to_decimal(snap_tol_mm) * mu:
        return int(nearest)
    return math.floor(scaled)


def epsilon_to_voxels(epsilon_mm: Number, mu: int) -> int:
    """Discrete safety margin ε_vox = ⌈ε·μ⌉ (rounded up, so the margin is never smaller than asked).

    Paper 1 (Sec. 3.1) defines ε ∈ Z+ in voxel units, while Paper 2 (Sec. 3.3.2) states
    ε = 2.0 mm. This converts the latter into the former.
    """
    mu = _check_mu(mu)
    eps = _to_decimal(epsilon_mm)
    if eps < 0:
        raise ValueError(f"safety margin ε must be >= 0, got {epsilon_mm!r}")
    return math.ceil(eps * mu)


# ======================================================================================
# Boxes
# ======================================================================================
@dataclass(frozen=True)
class AABB:
    """A continuous axis-aligned bounding box in mm."""

    x_min: float
    x_max: float
    y_min: float
    y_max: float
    z_min: float
    z_max: float

    def __post_init__(self) -> None:
        for axis, lo, hi in self.axes():
            if not (math.isfinite(lo) and math.isfinite(hi)):
                raise ValueError(f"AABB {axis} bounds must be finite, got [{lo}, {hi}]")
            if lo > hi:
                raise ValueError(f"AABB {axis} bounds are inverted: [{lo}, {hi}]")

    def axes(self) -> Tuple[Tuple[str, float, float], ...]:
        return (("X", self.x_min, self.x_max), ("Y", self.y_min, self.y_max), ("Z", self.z_min, self.z_max))

    def minkowski_pad(self, epsilon_mm: float) -> "AABB":
        """Return self ⊕ B_ε, where B_ε = {c : ‖c‖∞ ≤ ε} (Paper 1, Eq. 1; Paper 2, Eq. 5).

        B_ε is an axis-aligned cube, so the Minkowski sum of a box with B_ε is exactly the
        box grown by ε on every face. This is not an approximation.
        """
        eps = float(epsilon_mm)
        if eps < 0 or not math.isfinite(eps):
            raise ValueError(f"safety margin ε must be finite and >= 0, got {epsilon_mm!r}")
        return AABB(
            self.x_min - eps, self.x_max + eps,
            self.y_min - eps, self.y_max + eps,
            self.z_min - eps, self.z_max + eps,
        )

    def format(self, decimals: int = 3) -> str:
        return ", ".join(f"{a} ∈ [{_fmt(lo, decimals)}, {_fmt(hi, decimals)}]" for a, lo, hi in self.axes())

    def as_dict(self) -> Dict[str, List[float]]:
        return {a.lower(): [lo + 0.0, hi + 0.0] for a, lo, hi in self.axes()}


@dataclass(frozen=True)
class GridBox:
    """An axis-aligned box of voxels in Z^3. Both ends of each range are inclusive."""

    x_min: int
    x_max: int
    y_min: int
    y_max: int
    z_min: int
    z_max: int

    def __post_init__(self) -> None:
        for axis, lo, hi in self.axes():
            if isinstance(lo, bool) or isinstance(hi, bool) or not isinstance(lo, Integral) or not isinstance(hi, Integral):
                raise TypeError(f"GridBox {axis} bounds must be integers, got [{lo!r}, {hi!r}]")
            if lo > hi:
                raise ValueError(f"GridBox {axis} bounds are inverted: [{lo}, {hi}]")

    def axes(self) -> Tuple[Tuple[str, int, int], ...]:
        return (("X", self.x_min, self.x_max), ("Y", self.y_min, self.y_max), ("Z", self.z_min, self.z_max))

    def dilate(self, k: int) -> "GridBox":
        """Return self ⊕ B_k in Z^3 (discrete Minkowski sum with the L∞ ball of radius k)."""
        if isinstance(k, bool) or not isinstance(k, Integral) or k < 0:
            raise ValueError(f"dilation radius must be a non-negative integer, got {k!r}")
        return GridBox(
            self.x_min - k, self.x_max + k,
            self.y_min - k, self.y_max + k,
            self.z_min - k, self.z_max + k,
        )

    @property
    def voxel_count(self) -> int:
        return (
            (self.x_max - self.x_min + 1)
            * (self.y_max - self.y_min + 1)
            * (self.z_max - self.z_min + 1)
        )

    def contains(self, voxel: Sequence[int]) -> bool:
        x, y, z = voxel
        return (
            self.x_min <= x <= self.x_max
            and self.y_min <= y <= self.y_max
            and self.z_min <= z <= self.z_max
        )

    def intersection(self, other: "GridBox") -> Optional["GridBox"]:
        """Box intersection (empty -> None). An O(1) disjointness test, so the heap can store
        obstacles as boxes instead of enumerating millions of voxels."""
        lo_x, hi_x = max(self.x_min, other.x_min), min(self.x_max, other.x_max)
        lo_y, hi_y = max(self.y_min, other.y_min), min(self.y_max, other.y_max)
        lo_z, hi_z = max(self.z_min, other.z_min), min(self.z_max, other.z_max)
        if lo_x > hi_x or lo_y > hi_y or lo_z > hi_z:
            return None
        return GridBox(lo_x, hi_x, lo_y, hi_y, lo_z, hi_z)

    def iter_voxels(self) -> Iterator[Tuple[int, int, int]]:
        for x in range(self.x_min, self.x_max + 1):
            for y in range(self.y_min, self.y_max + 1):
                for z in range(self.z_min, self.z_max + 1):
                    yield (x, y, z)

    def back_project(self, mu: int, decimals: int = 3) -> Tuple[Tuple[Decimal, Decimal], ...]:
        """mm bounds [lo/μ, (hi+1)/μ] per axis, rounded outward to `decimals` places."""
        mu = _check_mu(mu)
        if decimals < 0:
            raise ValueError(f"decimals must be >= 0, got {decimals!r}")
        return tuple(
            (_cell_lower_mm(lo, mu, decimals), _cell_upper_mm(hi, mu, decimals))
            for _, lo, hi in self.axes()
        )

    def format(self) -> str:
        return ", ".join(f"{a} ∈ [{lo}, {hi}]" for a, lo, hi in self.axes())

    def format_mm(self, mu: int, decimals: int = 3) -> str:
        bounds = self.back_project(mu, decimals)
        return ", ".join(f"{a} ∈ [{lo:f}, {hi:f}]" for (a, _, _), (lo, hi) in zip(self.axes(), bounds))

    def as_dict(self) -> Dict[str, List[int]]:
        return {a.lower(): [lo, hi] for a, lo, hi in self.axes()}


def grid_box_of(box: AABB, mu: int, *, snap_tol_mm: float = CONFUSION_MM) -> GridBox:
    """The smallest GridBox whose cells cover `box`: [⌊min·μ⌋, ⌊max·μ⌋] per axis."""
    return GridBox(
        s_grid(box.x_min, mu, snap_tol_mm=snap_tol_mm), s_grid(box.x_max, mu, snap_tol_mm=snap_tol_mm),
        s_grid(box.y_min, mu, snap_tol_mm=snap_tol_mm), s_grid(box.y_max, mu, snap_tol_mm=snap_tol_mm),
        s_grid(box.z_min, mu, snap_tol_mm=snap_tol_mm), s_grid(box.z_max, mu, snap_tol_mm=snap_tol_mm),
    )


# ======================================================================================
# Analytic geometry -> nominal AABB
# ======================================================================================
def cylinder_aabb(location: Vec3, axis: Vec3, radius: float, axial_range: Tuple[float, float]) -> AABB:
    """Exact AABB of a finite cylinder for any axis orientation.

    The cylinder is {p + R·w : p on the axis segment, w ⊥ d, |w| ≤ 1}, with
    p ∈ [location + v0·d, location + v1·d]. Along world axis i the disc of radius R
    perpendicular to d extends by R·√(1 - d_i²). The box is therefore the segment's box
    grown by that amount per axis. A partial cylinder (angular span < 360°) lies inside
    this box, so the result is a conservative superset in that case.
    """
    r = float(radius)
    if not (math.isfinite(r) and r > 0):
        raise ValueError(f"cylinder radius must be finite and > 0, got {radius!r}")
    d = _unit(axis)
    v0, v1 = sorted(float(v) for v in axial_range)
    if not (math.isfinite(v0) and math.isfinite(v1)):
        raise ValueError(f"cylinder axial range must be finite, got {axial_range!r}")
    p0 = tuple(float(location[i]) + v0 * d[i] for i in range(3))
    p1 = tuple(float(location[i]) + v1 * d[i] for i in range(3))
    bounds: List[float] = []
    for i in range(3):
        spread = r * math.sqrt(max(0.0, 1.0 - d[i] * d[i]))
        bounds.extend((min(p0[i], p1[i]) - spread, max(p0[i], p1[i]) + spread))
    return AABB(*bounds)


def sphere_aabb(center: Vec3, radius: float) -> AABB:
    """AABB of a sphere (a conservative superset for a partial spherical face)."""
    r = float(radius)
    if not (math.isfinite(r) and r > 0):
        raise ValueError(f"sphere radius must be finite and > 0, got {radius!r}")
    cx, cy, cz = (float(c) for c in center)
    return AABB(cx - r, cx + r, cy - r, cy + r, cz - r, cz + r)


@dataclass(frozen=True)
class AnalyticFeature:
    """Exact parameters of one analytic face extracted from the B-Rep.

    index            : 1-based face index in traversal order (the "Surface #006" numbering).
    kind             : "CYLINDER" or "SPHERE".
    location         : Listing 1's "Center XYZ", the origin of the surface's local frame.
                       For a cylinder this is a point on the axis, not necessarily the
                       midpoint of the face.
    radius           : Radius in mm.
    axis             : Unit axis direction (cylinders only).
    axial_range      : (v0, v1), the signed distances along `axis` from `location` that bound
                       the trimmed face (cylinders only).
    angular_span_deg : Angular extent of the face around the axis (360 for a full cylinder).
    convexity        : "convex" (boss/pin, material inside the surface), "concave"
                       (bore/hole, material outside), or None if unknown. Derived from the
                       face orientation, so it is only meaningful for closed solids.
    """

    index: int
    kind: str
    location: Vec3
    radius: float
    axis: Optional[Vec3] = None
    axial_range: Optional[Tuple[float, float]] = None
    angular_span_deg: Optional[float] = None
    convexity: Optional[str] = None

    def __post_init__(self) -> None:
        if self.kind not in _ANALYTIC_KINDS:
            raise ValueError(f"kind must be one of {_ANALYTIC_KINDS}, got {self.kind!r}")
        object.__setattr__(self, "location", tuple(float(c) for c in self.location))
        if self.kind == "CYLINDER":
            if self.axis is None or self.axial_range is None:
                raise ValueError("a CYLINDER feature needs both `axis` and `axial_range`")
            object.__setattr__(self, "axis", _unit(self.axis))
            object.__setattr__(self, "axial_range", tuple(sorted(float(v) for v in self.axial_range)))
        if self.convexity not in (None, "convex", "concave"):
            raise ValueError(f"convexity must be 'convex', 'concave' or None, got {self.convexity!r}")

    def axial_endpoints(self) -> Optional[Tuple[Vec3, Vec3]]:
        if self.kind != "CYLINDER":
            return None
        (v0, v1), d, loc = self.axial_range, self.axis, self.location
        return (
            tuple(loc[i] + v0 * d[i] for i in range(3)),
            tuple(loc[i] + v1 * d[i] for i in range(3)),
        )

    def nominal_aabb(self) -> AABB:
        if self.kind == "CYLINDER":
            return cylinder_aabb(self.location, self.axis, self.radius, self.axial_range)
        return sphere_aabb(self.location, self.radius)


@dataclass(frozen=True)
class SolidFeature:
    """One solid of a fixture STEP file, bounded by its AABB.

    Clamps, vise jaws and fixture plates are mostly planar, so no analytic face bounds them.
    The whole solid becomes the obstacle instead. An AABB contains its solid, so this is
    conservative: it can only reject more programs, never accept a colliding one.

    index : 1-based solid index in TopExp_Explorer order (the "Solid #001" numbering).
    aabb  : AABB of the solid in mm, enlarged by the shape tolerance (see extract_solids()).
    """

    index: int
    aabb: AABB
    kind: str = field(default="SOLID", init=False)
    convexity: Optional[str] = field(default=None, init=False)

    def nominal_aabb(self) -> AABB:
        return self.aabb


@dataclass(frozen=True)
class SurfaceRecord:
    """One face visited during traversal (one line group of the Figure 3 report)."""

    index: int
    kind: str
    feature: Optional[AnalyticFeature] = None
    note: Optional[str] = None

    @property
    def label(self) -> str:
        return SURFACE_LABELS.get(self.kind, SURFACE_LABELS["OTHER"])


@dataclass(frozen=True)
class TopologyMetrics:
    """Topology counts. `*_visits` reproduce Listing 1 (TopExp_Explorer occurrences); the plain
    fields count distinct sub-shapes (IsSame). None means unknown, as in the mock."""

    vertices: Optional[int]
    edges: Optional[int]
    faces: Optional[int]
    vertex_visits: Optional[int]
    edge_visits: Optional[int]
    face_visits: Optional[int]


@dataclass(frozen=True)
class ExtractionReport:
    source: str
    topology: TopologyMetrics
    surfaces: Tuple[SurfaceRecord, ...]

    def features(self, kinds: Iterable[str] = _ANALYTIC_KINDS) -> List[AnalyticFeature]:
        wanted = set(kinds)
        return [s.feature for s in self.surfaces if s.feature is not None and s.feature.kind in wanted]

    def format_text(self, decimals: int = 3) -> str:
        """Plain-text report in the layout of Figure 3."""

        def count_line(label: str, unique: Optional[int], visits: Optional[int]) -> str:
            shown = str(unique) if unique is not None else "n/a"
            extra = f"   (explorer visits: {visits})" if visits is not None else ""
            return f" {label}{shown}{extra}"

        topo = self.topology
        lines = [
            f"Loading STEP file: {self.source}...",
            "",
            "========================================",
            "       STEP FILE TOPOLOGY METRICS       ",
            "========================================",
            count_line("Total Unique Vertices (Points): ", topo.vertices, topo.vertex_visits),
            count_line("Total Unique Edges (Lines):     ", topo.edges, topo.edge_visits),
            count_line("Total Unique Faces (Surfaces):  ", topo.faces, topo.face_visits),
            "========================================",
            "",
            "--- SURFACE GEOMETRY DETECTED ---",
        ]
        for surface in self.surfaces:
            lines.append(f" Surface #{surface.index:03d}: {surface.label}")
            feature = surface.feature
            if feature is not None:
                lines.append(f"    -> Dimension: Radius = {_fmt(feature.radius, 2)} mm")
                lines.append(f"    -> Center XYZ: {_fmt_vec(feature.location, decimals)}")
                if feature.kind == "CYLINDER":
                    v0, v1 = feature.axial_range
                    span = "" if feature.angular_span_deg is None else f", span {_fmt(feature.angular_span_deg, 1)}°"
                    lines.append(
                        f"    -> Axis: {_fmt_vec(feature.axis, decimals)}, "
                        f"V ∈ [{_fmt(v0, decimals)}, {_fmt(v1, decimals)}] mm{span}"
                    )
                if feature.convexity is not None:
                    side = "boss/pin" if feature.convexity == "convex" else "bore/hole"
                    lines.append(f"    -> Side: {feature.convexity} ({side})")
                lines.append(f"    -> Nominal AABB: {feature.nominal_aabb().format(decimals)}")
            if surface.note:
                lines.append(f"    -> Note: {surface.note}")
        listed = len(self.surfaces)
        if topo.face_visits is not None and topo.face_visits > listed:
            lines.append(f" ... [{topo.face_visits - listed} further surfaces not listed] ...")
        lines += ["", "Analysis Complete."]
        return "\n".join(lines)


# ======================================================================================
# OpenCASCADE extraction (pythonocc-core, imported lazily)
# ======================================================================================
def _load_occ() -> SimpleNamespace:
    """Import pythonocc-core only when a real STEP file is read, so every other part of this
    module (and parser.py, which reuses s_grid) works without OpenCASCADE installed."""
    try:
        import OCC.Core.GeomAbs as GeomAbs
        from OCC.Core.Bnd import Bnd_Box
        from OCC.Core.BRepAdaptor import BRepAdaptor_Surface
        from OCC.Core.IFSelect import IFSelect_RetDone
        from OCC.Core.STEPControl import STEPControl_Reader
        from OCC.Core.TopAbs import (
            TopAbs_EDGE,
            TopAbs_FACE,
            TopAbs_FORWARD,
            TopAbs_REVERSED,
            TopAbs_SOLID,
            TopAbs_VERTEX,
        )
        from OCC.Core.TopExp import TopExp_Explorer
        from OCC.Core.TopTools import TopTools_IndexedMapOfShape
    except ImportError as exc:
        raise ImportError(
            "extract_step() needs pythonocc-core "
            "(conda install -c conda-forge pythonocc-core). "
            "The AABB, padding and RAG functions in this module work without it."
        ) from exc

    # TopoDS downcast: `topods.Face` in pythonocc >= 7.7.1, `topods_Face` before that.
    # Recent releases also auto-downcast, so a missing helper is harmless.
    face_cast = None
    try:
        from OCC.Core.TopoDS import topods

        face_cast = topods.Face
    except ImportError:
        try:
            from OCC.Core.TopoDS import topods_Face as face_cast  # type: ignore[no-redef]
        except ImportError:
            face_cast = None

    # BRepBndLib: `brepbndlib.AddOptimal` in pythonocc >= 7.7.1, `brepbndlib_AddOptimal` before that.
    try:
        from OCC.Core.BRepBndLib import brepbndlib

        add_optimal = brepbndlib.AddOptimal
    except ImportError:
        from OCC.Core.BRepBndLib import brepbndlib_AddOptimal as add_optimal  # type: ignore[no-redef]

    geomabs_names = {
        "GeomAbs_Plane": "PLANE",
        "GeomAbs_Cylinder": "CYLINDER",
        "GeomAbs_Cone": "CONE",
        "GeomAbs_Sphere": "SPHERE",
        "GeomAbs_Torus": "TORUS",
        "GeomAbs_BezierSurface": "BEZIER",
        "GeomAbs_BSplineSurface": "BSPLINE",
        "GeomAbs_SurfaceOfRevolution": "REVOLUTION",
        "GeomAbs_SurfaceOfExtrusion": "EXTRUSION",
        "GeomAbs_OffsetSurface": "OFFSET",
        "GeomAbs_OtherSurface": "OTHER",
    }
    surface_kinds = {
        getattr(GeomAbs, name): kind for name, kind in geomabs_names.items() if hasattr(GeomAbs, name)
    }

    return SimpleNamespace(
        Bnd_Box=Bnd_Box,
        add_optimal=add_optimal,
        BRepAdaptor_Surface=BRepAdaptor_Surface,
        IFSelect_RetDone=IFSelect_RetDone,
        STEPControl_Reader=STEPControl_Reader,
        TopAbs_EDGE=TopAbs_EDGE,
        TopAbs_FACE=TopAbs_FACE,
        TopAbs_FORWARD=TopAbs_FORWARD,
        TopAbs_REVERSED=TopAbs_REVERSED,
        TopAbs_SOLID=TopAbs_SOLID,
        TopAbs_VERTEX=TopAbs_VERTEX,
        TopExp_Explorer=TopExp_Explorer,
        TopTools_IndexedMapOfShape=TopTools_IndexedMapOfShape,
        face_cast=face_cast,
        surface_kinds=surface_kinds,
    )


def _count_subshapes(occ: SimpleNamespace, shape, kind) -> Tuple[int, int]:
    """(unique count, explorer visits) for one sub-shape type."""
    unique = occ.TopTools_IndexedMapOfShape()
    visits = 0
    explorer = occ.TopExp_Explorer(shape, kind)
    while explorer.More():
        unique.Add(explorer.Current())
        visits += 1
        explorer.Next()
    return unique.Extent(), visits


def _convexity(occ: SimpleNamespace, face, position) -> Optional[str]:
    """Material side of a cylindrical or spherical face.

    For a right-handed local frame (gp_Ax3.Direct()), the natural surface normal
    D1U × D1V points away from the axis or centre. A REVERSED face flips it. Faces of a
    closed solid have outward normals (away from the material), so the face is convex
    (a boss or pin) exactly when Direct() agrees with FORWARD.
    """
    try:
        orientation = face.Orientation()
        if orientation not in (occ.TopAbs_FORWARD, occ.TopAbs_REVERSED):
            return None  # INTERNAL / EXTERNAL faces have no material side
        forward = orientation == occ.TopAbs_FORWARD
        return "convex" if bool(position.Direct()) == forward else "concave"
    except Exception:  # pragma: no cover - defensive against binding differences
        return None


def _classify_face(occ: SimpleNamespace, face, index: int) -> SurfaceRecord:
    adaptor = occ.BRepAdaptor_Surface(face)  # Restriction=True: parameters bounded by the face
    kind = occ.surface_kinds.get(adaptor.GetType(), "OTHER")

    if kind == "CYLINDER":
        cylinder = adaptor.Cylinder()
        position = cylinder.Position()
        loc, direction = cylinder.Location(), position.Direction()
        v0, v1 = adaptor.FirstVParameter(), adaptor.LastVParameter()
        u0, u1 = adaptor.FirstUParameter(), adaptor.LastUParameter()
        if not all(math.isfinite(v) and abs(v) < _UNBOUNDED for v in (v0, v1)):
            return SurfaceRecord(index, kind, note="unbounded axial parameter range; no AABB derived")
        span = min(360.0, math.degrees(abs(u1 - u0))) if math.isfinite(u1 - u0) else None
        feature = AnalyticFeature(
            index=index,
            kind="CYLINDER",
            location=(loc.X(), loc.Y(), loc.Z()),
            radius=cylinder.Radius(),
            axis=(direction.X(), direction.Y(), direction.Z()),
            axial_range=(v0, v1),
            angular_span_deg=span,
            convexity=_convexity(occ, face, position),
        )
        return SurfaceRecord(index, kind, feature)

    if kind == "SPHERE":
        sphere = adaptor.Sphere()
        loc = sphere.Location()
        feature = AnalyticFeature(
            index=index,
            kind="SPHERE",
            location=(loc.X(), loc.Y(), loc.Z()),
            radius=sphere.Radius(),
            convexity=_convexity(occ, face, sphere.Position()),
        )
        return SurfaceRecord(index, kind, feature)

    return SurfaceRecord(index, kind)


def _read_step(occ: SimpleNamespace, file_path: str):
    """STEPControl_Reader -> ReadFile -> TransferRoots -> OneShape (Listing 1, step 1)."""
    if not os.path.isfile(file_path):
        raise FileNotFoundError(f"STEP file not found: {file_path}")
    reader = occ.STEPControl_Reader()
    status = reader.ReadFile(file_path)
    if status != occ.IFSelect_RetDone:
        raise StepReadError(f"OpenCASCADE failed to parse this STEP file (status {status}): {file_path}")
    reader.TransferRoots()
    shape = reader.OneShape()
    if shape is None or shape.IsNull():
        raise StepReadError(f"STEP file transferred no geometry: {file_path}")
    return shape


def extract_step(file_path: str) -> ExtractionReport:
    """Load a STEP file and extract its B-Rep topology and analytic surface parameters.

    The workflow follows Listing 1 (Paper 2, Sec. 3.2.1):
      1. Initialization and root transfer: STEPControl_Reader -> ReadFile -> TransferRoots -> OneShape.
      2. Topological traversal: TopExp_Explorer over vertices, edges and faces.
      3. Geometric adaptation: BRepAdaptor_Surface classifies each face; cylinders and spheres
         get their exact parameters (plus axis and axial range for cylinders).
    """
    occ = _load_occ()
    shape = _read_step(occ, file_path)

    n_vertices, v_visits = _count_subshapes(occ, shape, occ.TopAbs_VERTEX)
    n_edges, e_visits = _count_subshapes(occ, shape, occ.TopAbs_EDGE)
    n_faces, f_visits = _count_subshapes(occ, shape, occ.TopAbs_FACE)
    topology = TopologyMetrics(n_vertices, n_edges, n_faces, v_visits, e_visits, f_visits)

    surfaces: List[SurfaceRecord] = []
    explorer = occ.TopExp_Explorer(shape, occ.TopAbs_FACE)
    index = 0
    while explorer.More():
        index += 1
        face = explorer.Current()
        if occ.face_cast is not None:
            try:
                face = occ.face_cast(face)
            except Exception:  # pragma: no cover - already a TopoDS_Face
                pass
        surfaces.append(_classify_face(occ, face, index))
        explorer.Next()

    return ExtractionReport(os.path.basename(file_path), topology, tuple(surfaces))


def extract_solids(file_path: str) -> Tuple[SolidFeature, ...]:
    """One AABB per solid in a fixture STEP file (clamps, vise jaws, fixture plates).

    The box comes from BRepBndLib::AddOptimal on the exact geometry (no triangulation), enlarged
    by the shape tolerance, so it contains the solid even when the file's tolerances are loose.
    Assembly instances are visited with their placement, so each placed copy is its own solid.
    """
    occ = _load_occ()
    shape = _read_step(occ, file_path)
    solids: List[SolidFeature] = []
    explorer = occ.TopExp_Explorer(shape, occ.TopAbs_SOLID)
    while explorer.More():
        box = occ.Bnd_Box()
        occ.add_optimal(explorer.Current(), box, False, True)  # exact geometry, + shape tolerance
        explorer.Next()
        if box.IsVoid():
            raise StepReadError(f"solid #{len(solids) + 1:03d} has an empty bounding box: {file_path}")
        x0, y0, z0, x1, y1, z1 = box.Get()
        solids.append(SolidFeature(len(solids) + 1, AABB(x0, x1, y0, y1, z0, z1)))
    return tuple(solids)


# ======================================================================================
# Minkowski padding -> RAG obstacle bounds
# ======================================================================================
@dataclass(frozen=True)
class PaddedObstacle:
    """One obstacle, in every form the downstream modules need.

    nominal_aabb : exact AABB of the extracted feature (mm).
    padded_aabb  : nominal_aabb ⊕ B_ε in R^3 (mm), the continuous Minkowski bound of Paper 2.
    nominal_grid : S_grid(nominal_aabb) in Z^3. This goes into the Spatial Heap as Environment.
    padded_grid  : nominal_grid ⊕ B_εvox in Z^3. Its back-projection is what the RAG prompt shows.
    """

    name: str
    feature: Union[AnalyticFeature, SolidFeature]
    epsilon_mm: float
    epsilon_vox: int
    mu: int
    nominal_aabb: AABB
    padded_aabb: AABB
    nominal_grid: GridBox
    padded_grid: GridBox

    @property
    def source(self) -> str:
        return f"#{self.feature.index:03d}"

    def rag_bounds_mm(self, decimals: int = 3) -> Tuple[Tuple[Decimal, Decimal], ...]:
        return self.padded_grid.back_project(self.mu, decimals)

    def rag_line(self, decimals: int = 3) -> str:
        """One OBSTACLE_BOUNDS entry in the Figure 4 format."""
        origin = "CAD Solid" if self.feature.kind == "SOLID" else "CAD Feature"
        return (
            f"- {self.name} (Derived from {origin} {self.source}): "
            f"{self.padded_grid.format_mm(self.mu, decimals)}"
        )

    def to_dict(self, decimals: int = 3) -> Dict[str, object]:
        f = self.feature
        parameters: Dict[str, object] = {}
        if f.kind in _ANALYTIC_KINDS:
            parameters.update(center_xyz=[c + 0.0 for c in f.location], radius_mm=f.radius)
        if f.kind == "CYLINDER":
            parameters.update(
                axis=[c + 0.0 for c in f.axis],
                axial_range_mm=list(f.axial_range),
                angular_span_deg=f.angular_span_deg,
            )
        rag = {
            axis.lower(): [float(lo), float(hi)]
            for (axis, _, _), (lo, hi) in zip(self.padded_grid.axes(), self.rag_bounds_mm(decimals))
        }
        return {
            "name": self.name,
            "source_feature": self.source,
            "surface_type": f.kind,
            "convexity": f.convexity,
            "parameters": parameters,
            "epsilon_mm": self.epsilon_mm,
            "epsilon_vox": self.epsilon_vox,
            "mu": self.mu,
            "nominal_aabb_mm": self.nominal_aabb.as_dict(),
            "padded_aabb_mm": self.padded_aabb.as_dict(),
            "nominal_grid": self.nominal_grid.as_dict(),
            "padded_grid": self.padded_grid.as_dict(),
            "rag_bounds_mm": rag,
        }


def pad_feature(
    feature: Union[AnalyticFeature, SolidFeature], epsilon_mm: float, mu: int, name: Optional[str] = None
) -> PaddedObstacle:
    """Apply the safety margin ε to one extracted feature or solid (Paper 2, Sec. 3.3.2)."""
    mu = _check_mu(mu)
    eps_vox = epsilon_to_voxels(epsilon_mm, mu)
    nominal = feature.nominal_aabb()
    nominal_grid = grid_box_of(nominal, mu)
    return PaddedObstacle(
        name=name or f"{'Fixture' if feature.kind == 'SOLID' else 'Obstacle'}_{feature.index:03d}",
        feature=feature,
        epsilon_mm=float(epsilon_mm),
        epsilon_vox=eps_vox,
        mu=mu,
        nominal_aabb=nominal,
        padded_aabb=nominal.minkowski_pad(float(epsilon_mm)),
        nominal_grid=nominal_grid,
        padded_grid=nominal_grid.dilate(eps_vox),
    )


def pad_features(
    features: Iterable[Union[AnalyticFeature, SolidFeature]],
    epsilon_mm: float,
    mu: int,
    *,
    names: Optional[Mapping[int, str]] = None,
    include_concave: bool = True,
) -> List[PaddedObstacle]:
    """Pad every extracted feature. `names` maps a face (or solid) index to a label (e.g. {6: "Clamp_1"}).

    Concave faces (bores) are kept by default. Treating a hole as an obstacle is
    conservative, whereas dropping a boss would be unsafe.
    """
    names = dict(names or {})
    return [
        pad_feature(f, epsilon_mm, mu, names.get(f.index))
        for f in features
        if include_concave or f.convexity != "concave"
    ]


def format_obstacle_bounds(obstacles: Sequence[PaddedObstacle], epsilon_mm: float, decimals: int = 3) -> str:
    """The OBSTACLE_BOUNDS block of the RAG system prompt (Paper 2, Figure 4)."""
    header = f"OBSTACLE_BOUNDS (Pre-expanded via Bε margin, ε = {repr(float(epsilon_mm))}mm):"
    if not obstacles:
        return header + "\n- (none)"
    return "\n".join([header] + [o.rag_line(decimals) for o in obstacles])


def build_rag_payload(obstacles: Sequence[PaddedObstacle], epsilon_mm: float, mu: int, decimals: int = 3) -> Dict[str, object]:
    """Structured form of the obstacle bounds, for main_loop.py or for JSON export."""
    return {
        "epsilon_mm": float(epsilon_mm),
        "epsilon_vox": epsilon_to_voxels(epsilon_mm, mu),
        "mu": _check_mu(mu),
        "obstacle_bounds_block": format_obstacle_bounds(obstacles, epsilon_mm, decimals),
        "obstacles": [o.to_dict(decimals) for o in obstacles],
    }


# ======================================================================================
# CLI and mock run
# ======================================================================================
def _compact_json(payload: object) -> str:
    """Indented JSON with flat arrays (coordinates, bounds) kept on a single line."""
    import re

    text = json.dumps(payload, indent=2, ensure_ascii=False)
    return re.sub(
        r"\[\s+([^\[\]{}]*?)\s+\]",
        lambda m: "[" + re.sub(r",\s+", ", ", m.group(1)) + "]",
        text,
    )


def _run_mock(epsilon_mm: float, mu: int) -> None:
    """Simulated extraction of the NIST CTC-01 surfaces shown in Paper 2, Figure 3.

    Figure 3 reports only Location() and the radius. The axis (+Z) and the 30 mm axial span
    below are ASSUMED so that the full AABB can be shown.
    """
    import random

    def cylinder(index: int, center: Vec3, radius: float) -> SurfaceRecord:
        return SurfaceRecord(index, "CYLINDER", AnalyticFeature(
            index=index, kind="CYLINDER", location=center, radius=radius,
            axis=(0.0, 0.0, 1.0), axial_range=(0.0, 30.0), angular_span_deg=360.0, convexity="convex",
        ))

    surfaces = (
        SurfaceRecord(1, "PLANE"), SurfaceRecord(2, "PLANE"), SurfaceRecord(3, "PLANE"),
        SurfaceRecord(4, "PLANE"), SurfaceRecord(5, "PLANE"),
        cylinder(6, (-245.0, 0.0, -100.0), 20.0),
        SurfaceRecord(7, "PLANE"),
        cylinder(8, (245.0, -15.0, -100.0), 10.0),
        SurfaceRecord(9, "PLANE"),
        cylinder(10, (245.0, 15.0, -100.0), 10.0),
    )
    # Figure 3's counts are explorer visits; the unique counts are unknown for the mock.
    topology = TopologyMetrics(None, None, None, vertex_visits=1734, edge_visits=856, face_visits=139)
    report = ExtractionReport("nist_ctc_01_asme1_rd.stp [MOCK]", topology, surfaces)

    print(report.format_text())
    print()

    names = {6: "Clamp_1", 8: "Clamp_2", 10: "Clamp_3"}
    obstacles = pad_features(report.features(), epsilon_mm, mu, names=names)
    clamp = obstacles[0]
    f = clamp.feature

    print(f"--- MINKOWSKI PADDING: Surface {clamp.source} -> {clamp.name} (ε = {epsilon_mm} mm, μ = {mu}) ---")
    print(f" 1. Extracted parameters     : Center {_fmt_vec(f.location)}, R = {_fmt(f.radius)} mm,")
    print(f"                               axis {_fmt_vec(f.axis)}, V ∈ [{_fmt(f.axial_range[0])}, {_fmt(f.axial_range[1])}] mm (assumed)")
    print(f" 2. Nominal AABB (mm)        : {clamp.nominal_aabb.format()}")
    print(f" 3. AABB ⊕ B_ε (mm)          : {clamp.padded_aabb.format()}")
    print(f" 4. Nominal grid box (Z^3)   : {clamp.nominal_grid.format()}   -> Spatial Heap C_env")
    print(f" 5. ε in voxels              : ⌈{epsilon_mm} × {mu}⌉ = {clamp.epsilon_vox}")
    print(f" 6. Grid box ⊕ B_εvox (Z^3)  : {clamp.padded_grid.format()}")
    print(f" 7. RAG bounds (mm)          : {clamp.padded_grid.format_mm(mu)}   -> LLM prompt")
    print(f"    Voxels claimed by the nominal clamp: {clamp.nominal_grid.voxel_count:,}")
    print()
    print(format_obstacle_bounds(obstacles, epsilon_mm))
    print()
    print("--- STRUCTURED PAYLOAD (Clamp_1) ---")
    print(_compact_json(clamp.to_dict()))

    # ---- Self-checks (these assume the default ε = 2.0 mm and μ = 10) ------------------
    if (float(epsilon_mm), mu) == (2.0, 10):
        assert clamp.nominal_aabb == AABB(-265.0, -225.0, -20.0, 20.0, -100.0, -70.0)
        assert clamp.padded_aabb == AABB(-267.0, -223.0, -22.0, 22.0, -102.0, -68.0)
        assert clamp.nominal_grid == GridBox(-2650, -2250, -200, 200, -1000, -700)
        assert clamp.epsilon_vox == 20
        assert clamp.padded_grid == GridBox(-2670, -2230, -220, 220, -1020, -680)
        assert clamp.rag_line() == (
            "- Clamp_1 (Derived from CAD Feature #006): "
            "X ∈ [-267.000, -222.900], Y ∈ [-22.000, 22.100], Z ∈ [-102.000, -67.900]"
        )

    # S_grid: exact decimal floor, noise snapping, negative coordinates.
    assert s_grid(2.3, 10) == 23                   # the float product 22.999... must not floor to 22
    assert s_grid(-244.99999999999997, 10) == -2450
    assert s_grid(-0.05, 10) == -1
    assert s_grid("12.345", 1000) == 12345
    assert epsilon_to_voxels(0.25, 10) == 3        # ⌈2.5⌉: the margin is rounded up, never down

    # Cylinder AABB is exact for a tilted axis: every sampled surface point lies inside the box,
    # and the sampled extremes reach the box faces.
    loc, axis, radius, v_range = (1.0, 2.0, 3.0), (1.0, 1.0, 1.0), 5.0, (-4.0, 10.0)
    box = cylinder_aabb(loc, axis, radius, v_range)
    d = _unit(axis)
    e1 = _unit((d[1], -d[0], 0.0))
    e2 = (d[1] * e1[2] - d[2] * e1[1], d[2] * e1[0] - d[0] * e1[2], d[0] * e1[1] - d[1] * e1[0])
    samples = [
        tuple(loc[i] + v * d[i] + radius * (math.cos(t) * e1[i] + math.sin(t) * e2[i]) for i in range(3))
        for v in (v_range[0], 3.0, v_range[1])
        for t in (2 * math.pi * k / 3600 for k in range(3600))
    ]
    for axis_index, (_, lo, hi) in enumerate(box.axes()):
        coords = [p[axis_index] for p in samples]
        assert lo - 1e-9 <= min(coords) and max(coords) <= hi + 1e-9
        assert abs(min(coords) - lo) < 1e-4 and abs(max(coords) - hi) < 1e-4

    # Padding the obstacle in the prompt is the same constraint as padding the tool in the prover:
    #   (T ⊕ B_k) ∩ E ≠ ∅  ⟺  T ∩ (E ⊕ B_k) ≠ ∅
    rng = random.Random(7)
    for _ in range(300):
        lo = [rng.randint(-5, 3) for _ in range(3)]
        env = GridBox(lo[0], lo[0] + rng.randint(0, 3), lo[1], lo[1] + rng.randint(0, 3), lo[2], lo[2] + rng.randint(0, 3))
        tool = [tuple(rng.randint(-9, 9) for _ in range(3)) for _ in range(rng.randint(1, 4))]
        k = rng.randint(0, 2)
        cube = [(a, b, c) for a in range(-k, k + 1) for b in range(-k, k + 1) for c in range(-k, k + 1)]
        padded_tool_hits = any(env.contains((t[0] + a, t[1] + b, t[2] + c)) for t in tool for a, b, c in cube)
        tool_hits_padded_env = any(env.dilate(k).contains(t) for t in tool)
        assert padded_tool_hits == tool_hits_padded_env

    # Fixture solids (per-solid AABB) go through the same padding: nominal box for the heap,
    # padded box for the RAG prompt.
    jaw = pad_feature(SolidFeature(1, AABB(-300.0, -280.0, -50.0, 50.0, -100.0, -60.0)), 2.0, 10, "Vise_jaw")
    assert jaw.nominal_grid == GridBox(-3000, -2800, -500, 500, -1000, -600)
    assert jaw.padded_grid == GridBox(-3020, -2780, -520, 520, -1020, -580)
    assert jaw.rag_line() == (
        "- Vise_jaw (Derived from CAD Solid #001): "
        "X ∈ [-302.000, -277.900], Y ∈ [-52.000, 52.100], Z ∈ [-102.000, -57.900]"
    )
    assert jaw.to_dict()["surface_type"] == "SOLID" and jaw.to_dict()["parameters"] == {}
    assert pad_feature(SolidFeature(2, AABB(0, 1, 0, 1, 0, 1)), 2.0, 10).name == "Fixture_002"

    print("\n[self-check] all assertions passed")


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = argparse.ArgumentParser(description="Extract B-Rep obstacles from a STEP file and pad them by ε.")
    parser.add_argument("step_file", nargs="?", help="STEP file to analyse; omit to run the mock")
    parser.add_argument("--epsilon", type=float, default=2.0, help="safety margin ε in mm (default 2.0)")
    parser.add_argument("--mu", type=int, default=10, help="grid multiplier μ (default 10, i.e. 0.1 mm voxels)")
    parser.add_argument("--json", metavar="PATH", help="write the RAG obstacle payload to this JSON file")
    parser.add_argument("--solids", action="store_true",
                        help="fixture file: one AABB obstacle per solid instead of per analytic face")
    args = parser.parse_args(argv)

    if args.step_file is None:
        _run_mock(args.epsilon, args.mu)
        return 0

    if args.solids:
        solids = extract_solids(args.step_file)
        print(f"Loading STEP file: {args.step_file}...\n\n--- SOLIDS DETECTED: {len(solids)} ---")
        for solid in solids:
            print(f" Solid #{solid.index:03d}: AABB {solid.aabb.format()}")
        obstacles = pad_features(solids, args.epsilon, args.mu)
    else:
        obstacles = _report_faces(args.step_file, args.epsilon, args.mu)
    print()
    print(format_obstacle_bounds(obstacles, args.epsilon))

    if args.json:
        with open(args.json, "w", encoding="utf-8") as handle:
            handle.write(_compact_json(build_rag_payload(obstacles, args.epsilon, args.mu)) + "\n")
        print(f"\nSaved RAG obstacle payload to: {args.json}")
    return 0


def _report_faces(step_file: str, epsilon_mm: float, mu: int) -> List[PaddedObstacle]:
    """Listing 1 face report (printed and saved next to the cwd), then the padded analytic faces."""
    report = extract_step(step_file)
    text = report.format_text()
    print(text)

    # Like Listing 1, also save the plain-text report in the current working directory.
    base = os.path.splitext(os.path.basename(step_file))[0]
    report_path = os.path.join(os.getcwd(), base + "_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(text + "\n")
    print(f"\nSaved plain text report to: {report_path}")
    return pad_features(report.features(), epsilon_mm, mu)


if __name__ == "__main__":
    sys.exit(main())
