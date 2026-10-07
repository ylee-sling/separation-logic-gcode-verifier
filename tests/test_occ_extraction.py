"""First real run of the OpenCASCADE path: extract_step(), _classify_face(), _convexity(),
_count_subshapes(). Solids with known geometry are built in OCC, written to STEP and read back.

Skipped when pythonocc-core is not installed.
"""

import math

import pytest

pytest.importorskip("OCC.Core.BRepPrimAPI")

from OCC.Core.BRep import BRep_Builder
from OCC.Core.BRepAlgoAPI import BRepAlgoAPI_Cut
from OCC.Core.BRepBuilderAPI import BRepBuilderAPI_Transform
from OCC.Core.BRepPrimAPI import BRepPrimAPI_MakeBox, BRepPrimAPI_MakeCylinder
from OCC.Core.Bnd import Bnd_Box
from OCC.Core.IFSelect import IFSelect_RetDone
from OCC.Core.STEPControl import STEPControl_AsIs, STEPControl_Writer
from OCC.Core.TopoDS import TopoDS_Compound
from OCC.Core.gp import gp_Ax1, gp_Ax2, gp_Dir, gp_Pnt, gp_Trsf

from cad_extractor import extract_solids, extract_step
from main_loop import PipelineConfig, build_system_prompt, ground_workspace, run_neuro_symbolic_pipeline

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


def box(corner, size):
    return BRepPrimAPI_MakeBox(gp_Pnt(*corner), *size).Shape()


def compound(*shapes):
    result, builder = TopoDS_Compound(), BRep_Builder()
    builder.MakeCompound(result)
    for shape in shapes:
        builder.Add(result, shape)
    return result


def test_fixture_solids_get_one_aabb_each(tmp_path):
    angle = math.radians(30)
    rotation = gp_Trsf()
    rotation.SetRotation(gp_Ax1(gp_Pnt(0, 0, 0), gp_Dir(0, 0, 1)), angle)
    rotated = BRepBuilderAPI_Transform(box((100, 0, 0), (20, 10, 5)), rotation, True).Shape()
    fixtures = compound(
        box((-300, -50, -100), (20, 100, 40)),     # vise jaw 1
        box((280, -50, -100), (20, 100, 40)),      # vise jaw 2
        rotated,                                   # strap clamp rotated 30° about Z
    )
    solids = extract_solids(to_step(fixtures, tmp_path, "fixtures"))
    assert [s.index for s in solids] == [1, 2, 3]
    assert all(s.kind == "SOLID" for s in solids)
    assert_vec(aabb_tuple(solids[0].aabb), (-300, -280, -50, 50, -100, -60), tol=1e-5)
    assert_vec(aabb_tuple(solids[1].aabb), (280, 300, -50, 50, -100, -60), tol=1e-5)

    c, s_ = math.cos(angle), math.sin(angle)
    corners = [(x * c - y * s_, x * s_ + y * c, z) for x in (100, 120) for y in (0, 10) for z in (0, 5)]
    expected = tuple(f(p[i] for p in corners) for i in range(3) for f in (min, max))
    got = aabb_tuple(solids[2].aabb)
    lows, highs = got[0::2], got[1::2]
    assert all(lo <= e for lo, e in zip(lows, expected[0::2]))      # contains the solid
    assert all(hi >= e for hi, e in zip(highs, expected[1::2]))
    assert_vec(got, expected, tol=1e-3)                              # and is tight


def test_pipeline_grounded_from_fixture_step(tmp_path):
    clamp = box((-265, -20, -100), (40, 40, 30))   # same envelope as the mock Clamp_1
    part = box((-100, -100, -100), (200, 200, 50))  # in the file but not a fixture
    path = to_step(compound(clamp, part), tmp_path, "setup")
    config = PipelineConfig(fixture_step_file=path, fixture_solids={1: "Clamp_1"})
    result = run_neuro_symbolic_pipeline("park next to Clamp_1", config, verbose=False)
    assert result.verified and len(result.iterations) == 2
    first = result.iterations[0]
    assert first.outcome == "SPATIAL_DATA_RACE" and first.record.resource_label == "Clamp_1"
    workspace = ground_workspace(config)
    assert [o.name for o in workspace.obstacles] == ["Clamp_1"]   # no mock clamp, part filtered out
    prompt = build_system_prompt(workspace)
    assert "- Clamp_1 (Derived from CAD Solid #001): " in prompt and "#002" not in prompt


def test_ctc01_vise_example(tmp_path):
    import sys
    from pathlib import Path

    repo = Path(__file__).resolve().parent.parent
    nist = repo / "nist_ctc_01_asme1_rd.stp"
    if not nist.is_file():
        pytest.skip("nist_ctc_01_asme1_rd.stp is not in the repository root")
    sys.path.insert(0, str(repo / "examples"))
    try:
        import ctc01_vise_setup as example
    finally:
        sys.path.pop(0)
    setup = example.build_setup_step(str(nist), str(tmp_path / "setup.stp"))
    assert [s.index for s in extract_solids(setup)] == [1, 2, 3, 4]
    result = example.run(verbose=False, setup_step=setup)
    assert result.verified and len(result.iterations) == 2
    first = result.iterations[0]
    assert first.record.resource_label == "Vise_jaw_L" and first.record.line_number == "N80"
    box = first.record.bounding_box
    assert (box.x_min, box.x_max, box.y_min, box.y_max, box.z_min, box.z_max) == (
        -71, 71, -2521, -2379, -620, -500)
