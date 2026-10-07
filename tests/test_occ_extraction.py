"""First real run of the OpenCASCADE path: extract_step(), _classify_face(), _convexity(),
_count_subshapes(). Solids with known geometry are built in OCC, written to STEP and read back.

Skipped when pythonocc-core is not installed.
"""

import math

import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI")

from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.gp import gp_Ax2, gp_Dir, gp_Pnt

from cad_extractor import extract_step

try:  # pythonocc >= 7.7.1
    from OCC.Core.BRepBndLib import brepbndlib

    _add_optimal = brepbndlib.AddOptimal
except ImportError:  # older releases
    from OCC.Core.BRepBndLib import brepbndlib_AddOptimal as _add_optimal

TOL = 1e-6


def to_step(shape, tmp_path, name):
    path = tmp_path / f"{name}.stp"
    writer = STEPControl_Writer()
    writer.Transfer(shape, STEPControl_AsIs)
    assert writer.Write(str(path)) == IFSelect_RetDone
    return str(path)


def cylinder(origin, axis, radius, height):
    return BRepPrimAPI_MakeCylinder(gp_Ax2(gp_Pnt(*origin), gp_Dir(*axis)), radius, height).Shape()


def assert_vec(actual, expected, tol=TOL):
    assert all(math.isclose(a, e, abs_tol=tol) for a, e in zip(actual, expected)), (actual, expected)


def aabb_tuple(box):
    return (box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max)


def test_cylinder_boss(tmp_path):
    centre, radius, height = (10.0, -5.0, 3.0), 7.5, 12.0
    report = extract_step(to_step(cylinder(centre, (0, 0, 1), radius, height), tmp_path, "boss"))
    (feature,) = report.features()
    assert feature.kind == "CYLINDER"
    assert math.isclose(feature.radius, radius, abs_tol=TOL)
    assert_vec(feature.axis, (0.0, 0.0, 1.0))
    assert_vec(feature.location, centre)
    v0, v1 = feature.axial_range
    assert math.isclose(v1 - v0, height, abs_tol=TOL)
    assert_vec(aabb_tuple(feature.nominal_aabb()), (2.5, 17.5, -12.5, 2.5, 3.0, 15.0))
    assert feature.convexity == "convex"


def test_bore_is_concave(tmp_path):
    block = BRepPrimAPI_MakeBox(gp_Pnt(-20, -20, 0), 40, 40, 20).Shape()
    drill = cylinder((0, 0, -5), (0, 0, 1), 5.0, 30.0)   # extends past both faces
    report = extract_step(to_step(BRepAlgoAPI_Cut(block, drill).Shape(), tmp_path, "bore"))
    bores = [f for f in report.features() if math.isclose(f.radius, 5.0, abs_tol=TOL)]
    assert len(bores) == 1
    bore = bores[0]
    assert bore.convexity == "concave"
    assert_vec(aabb_tuple(bore.nominal_aabb()), (-5.0, 5.0, -5.0, 5.0, 0.0, 20.0))


def test_cylinder_topology_counts(tmp_path):
    report = extract_step(to_step(cylinder((0, 0, 0), (0, 0, 1), 3.0, 8.0), tmp_path, "plain"))
    t = report.topology
    assert (t.vertices, t.edges, t.faces) == (2, 3, 3)
    assert t.vertices - t.edges + t.faces == 2
    assert t.vertex_visits >= t.vertices
    assert t.edge_visits >= t.edges
    assert t.face_visits >= t.faces


def test_tilted_cylinder_aabb_matches_occ_optimal_box(tmp_path):
    shape = cylinder((1.0, 2.0, 3.0), (1, 2, 3), 4.0, 25.0)
    report = extract_step(to_step(shape, tmp_path, "tilted"))
    (feature,) = report.features()
    reference = Bnd_Box()
    _add_optimal(shape, reference, False, False)  # exact geometry, no triangulation or tolerance
    xmin, ymin, zmin, xmax, ymax, zmax = reference.Get()
    assert_vec(aabb_tuple(feature.nominal_aabb()), (xmin, xmax, ymin, ymax, zmin, zmax), tol=1e-3)
