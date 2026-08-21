import math

import numpy as np
import pytest
from PIL import Image

from semantic_digital_twin.adapters.usd.exceptions import (
    UnsupportedUsdGeometryTypeError,
    UnsupportedUsdPhysicsJointTypeError,
    UsdPhysicsJointMissingChildBodyError,
)
from semantic_digital_twin.adapters.usd.parser import USDParser, UsdMeshShapeBuilder
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.usd_semantics import UsdSemanticLabels
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Cylinder, Sphere
from semantic_digital_twin.world_description.world_entity import Body

from .usd_stages import (
    PXR_AVAILABLE,
    USD_SEMANTICS_AVAILABLE,
    build_jointless_stage_with_a_default_prim,
    build_jointless_stage_with_multiple_top_level_prims,
    build_jointless_stage_with_unsupported_geometry,
    build_single_joint_stage,
    build_single_joint_stage_with_mass,
    build_single_joint_stage_with_semantic_labels,
    build_stage_with_ambiguous_root_and_a_joint,
    build_stage_with_joint_missing_body1,
    build_stage_with_mesh_targeted_body0,
    build_stage_with_primitive_shapes,
    build_stage_with_scaled_mesh,
    build_stage_with_textured_mesh,
)

if PXR_AVAILABLE:
    from pxr import Gf, UsdGeom

pytestmark = pytest.mark.skipif(
    not PXR_AVAILABLE, reason="usd-core (pxr) not installed"
)


def parse(stage) -> World:
    return USDParser(stage=stage, prefix="object").parse()


# %% joints


def test_parse_connects_a_revolute_joint():
    stage = build_single_joint_stage("RevoluteJoint")
    world = parse(stage)

    assert len(world.bodies) == 2  # root + child
    [connection] = world.connections
    assert isinstance(connection, RevoluteConnection)


def test_parse_connects_a_prismatic_joint():
    stage = build_single_joint_stage("PrismaticJoint")
    world = parse(stage)

    [connection] = world.connections
    assert isinstance(connection, PrismaticConnection)


def test_parse_connects_a_fixed_joint():
    stage = build_single_joint_stage("FixedJoint")
    world = parse(stage)

    [connection] = world.connections
    assert isinstance(connection, FixedConnection)


def test_parse_inverts_the_joints_local_pos1_into_connection_t_child():
    # UsdPhysics.Joint documents localPos1/localRot1 as the joint frame's pose relative
    # to body1 (child_T_connection) - the inverse of what Connection needs
    # (connection_T_child). A pure translation makes a missing inversion visible as a
    # sign flip: authoring local_pos1=(1, 0, 0) must place the child's own origin at
    # (-1, 0, 0) relative to the connection frame, not (1, 0, 0).
    stage = build_single_joint_stage("FixedJoint", local_pos1=(1.0, 0.0, 0.0))
    world = parse(stage)

    [connection] = world.connections
    translation = connection.connection_T_child_expression.to_np()[:3, 3]
    np.testing.assert_allclose(translation, [-1.0, 0.0, 0.0], atol=1e-6)


def test_parse_raises_on_an_unsupported_joint_type_instead_of_building_an_incomplete_world():
    # An unrecognized joint type name must not be silently skipped like a non-joint
    # prim would be, or a stage using only unsupported joints builds a World with no
    # bodies but the root - not an error, just quietly missing every link.
    stage = build_single_joint_stage("SphericalJoint")

    with pytest.raises(UnsupportedUsdPhysicsJointTypeError) as excinfo:
        parse(stage)

    assert excinfo.value.joint_type == "PhysicsSphericalJoint"


def test_parse_swaps_inverted_revolute_limits():
    # Seen on a real ArtVIP object (one blade of a pair of scissors): the USD data
    # itself authors lower > upper.
    stage = build_single_joint_stage("RevoluteJoint", lower_limit=30.0, upper_limit=0.0)
    world = parse(stage)

    [connection] = world.connections
    assert connection.raw_dof.limits.lower.position == pytest.approx(0.0)
    assert connection.raw_dof.limits.upper.position == pytest.approx(math.radians(30.0))


def test_parse_resolves_a_mesh_targeted_body0_to_its_enclosing_link():
    # body0 pointing directly at a link's mesh prim (rather than its enclosing Xform)
    # must not create a second, disconnected Body for the same physical link.
    stage = build_stage_with_mesh_targeted_body0()
    world = parse(stage)

    assert len(world.bodies) == 3  # root + carcass + handle
    assert world.root is not None


def test_parse_raises_on_a_joint_with_no_body1_target():
    stage = build_stage_with_joint_missing_body1()

    with pytest.raises(UsdPhysicsJointMissingChildBodyError) as excinfo:
        parse(stage)

    assert excinfo.value.joint_path == "/object/joint"


# %% joint-less stages


def test_parse_builds_a_single_body_world_for_a_stage_without_joints():
    stage = build_jointless_stage_with_a_default_prim()
    world = parse(stage)

    assert len(world.bodies) == 1
    assert len(world.connections) == 0
    assert len(world.root.visual.shapes) == 1


def test_parse_connects_a_joint_to_a_synthetic_root_with_an_ambiguous_root_prim():
    # A joint's unset body0 never depended on the stage having an identifiable root
    # prim: it works the same against a synthetic root as against a real one.
    stage = build_stage_with_ambiguous_root_and_a_joint()
    world = parse(stage)

    [connection] = world.connections
    assert isinstance(connection, FixedConnection)
    assert connection.parent is world.root


def test_parse_attaches_multiple_top_level_prims_to_a_synthetic_root():
    # A joint-less stage's ambiguous root is not an error the way one with a joint
    # graph to name is: each top-level prim becomes its own freely posable body instead.
    stage = build_jointless_stage_with_multiple_top_level_prims()
    world = parse(stage)

    assert len(world.bodies) == 3  # synthetic root + object_a + object_b
    non_root_bodies = [body for body in world.bodies if body is not world.root]
    assert {body.name.name for body in non_root_bodies} == {"object_a", "object_b"}
    for body in non_root_bodies:
        assert len(body.visual.shapes) == 1

    [connection_a, connection_b] = world.connections
    assert isinstance(connection_a, Connection6DoF)
    assert isinstance(connection_b, Connection6DoF)
    assert connection_a.parent is world.root
    assert connection_b.parent is world.root


def test_parse_raises_on_unsupported_geometry_instead_of_silently_dropping_it():
    # A Cone prim is a UsdGeom.Gprim this parser does not build a Shape for - it must
    # not be silently skipped like a non-geometry Xform/Scope/Camera prim would be, or
    # a stage using it builds a World quietly missing that piece of geometry.
    stage = build_jointless_stage_with_unsupported_geometry()

    with pytest.raises(UnsupportedUsdGeometryTypeError) as excinfo:
        parse(stage)

    assert excinfo.value.geometry_type == "Cone"


# %% shapes


def test_create_mesh_shape_applies_a_non_uniform_scale():
    stage = build_stage_with_scaled_mesh(scale=(2.0, 3.0, 4.0))
    mesh_prim = stage.GetPrimAtPath("/object/scaled/mesh")
    body = Body(name=PrefixedName("test_body"))

    shape = UsdMeshShapeBuilder(mesh_prim, Gf.Matrix4d(1), body).build()

    vertices = shape.unscaled_mesh.vertices
    np.testing.assert_allclose(vertices.max(axis=0), [2.0, 3.0, 0.0], atol=1e-5)


def test_parse_builds_a_box_for_a_cube_prim():
    stage = build_stage_with_primitive_shapes(cube_scale=(1.0, 2.0, 3.0))
    world = parse(stage)

    [box] = [shape for shape in world.root.visual.shapes if isinstance(shape, Box)]
    np.testing.assert_allclose([box.scale.x, box.scale.y, box.scale.z], [2.0, 4.0, 6.0])


def test_parse_builds_a_sphere_for_a_sphere_prim():
    stage = build_stage_with_primitive_shapes()
    world = parse(stage)

    [sphere] = [
        shape for shape in world.root.visual.shapes if isinstance(shape, Sphere)
    ]
    assert sphere.radius == pytest.approx(0.5)


def test_parse_builds_a_cylinder_for_a_z_axis_cylinder_prim():
    stage = build_stage_with_primitive_shapes(cylinder_axis="Z")
    world = parse(stage)

    [cylinder] = [
        shape for shape in world.root.visual.shapes if isinstance(shape, Cylinder)
    ]
    assert cylinder.width == pytest.approx(1.0)
    assert cylinder.height == pytest.approx(2.0)


def test_parse_aligns_an_x_axis_cylinder_prim_to_the_shapes_local_z_axis():
    stage = build_stage_with_primitive_shapes(cylinder_axis="X")
    world = parse(stage)

    [cylinder] = [
        shape for shape in world.root.visual.shapes if isinstance(shape, Cylinder)
    ]
    rotation = cylinder.origin.to_np()[:3, :3]
    local_z_in_link_frame = rotation @ np.array([0.0, 0.0, 1.0])
    np.testing.assert_allclose(local_z_in_link_frame, [1.0, 0.0, 0.0], atol=1e-6)


def test_parse_applies_scale_along_the_cylinders_own_axis_not_the_shapes():
    # scale is authored in the prim's own local axes, unaffected by the alignment
    # rotation that reorients an X/Y-axis cylinder to the shape's local Z: for an
    # X-axis cylinder, scale[0] is height and scale[1]/scale[2] are radial, not
    # scale[2]/scale[0:2] as for the default Z-axis case.
    stage = build_stage_with_primitive_shapes(
        cylinder_axis="X", cylinder_scale=(5.0, 1.0, 1.0)
    )
    world = parse(stage)

    [cylinder] = [
        shape for shape in world.root.visual.shapes if isinstance(shape, Cylinder)
    ]
    assert cylinder.height == pytest.approx(2.0 * 5.0)
    assert cylinder.width == pytest.approx(1.0)


# %% inertials


def test_parse_inertial_reads_mass_api_properties():
    stage = build_single_joint_stage_with_mass(
        mass=2.0, center_of_mass=(0.1, 0.2, 0.3), diagonal_inertia=(1.0, 2.0, 3.0)
    )
    link_prim = stage.GetPrimAtPath("/object/child")
    body = Body(name=PrefixedName("test_body"))

    inertial = USDParser._parse_inertial(link_prim, body)

    assert inertial is not None
    assert inertial.mass == pytest.approx(2.0)
    np.testing.assert_allclose(
        inertial.center_of_mass.to_np()[:3], [0.1, 0.2, 0.3], atol=1e-6
    )


def test_parse_inertial_is_none_without_mass_api():
    stage = build_single_joint_stage("FixedJoint")
    link_prim = stage.GetPrimAtPath("/object/child")
    body = Body(name=PrefixedName("test_body"))

    assert USDParser._parse_inertial(link_prim, body) is None


def test_parse_sets_body_inertial_from_mass_api():
    stage = build_single_joint_stage_with_mass(mass=2.0)
    world = parse(stage)

    [child] = [body for body in world.bodies if body is not world.root]
    assert child.inertial.mass == pytest.approx(2.0)


# %% semantics


@pytest.mark.skipif(
    not USD_SEMANTICS_AVAILABLE, reason="UsdSemantics not available in this usd-core"
)
def test_read_semantic_labels_reads_every_taxonomy():
    stage = build_single_joint_stage_with_semantic_labels()
    link_prim = stage.GetPrimAtPath("/object/child")

    labels_by_taxonomy = USDParser._read_semantic_labels(link_prim)

    assert labels_by_taxonomy == {
        "class": ["chair", "furniture"],
        "category": ["seating"],
    }


def test_read_semantic_labels_is_empty_without_any_applied():
    stage = build_single_joint_stage("FixedJoint")
    link_prim = stage.GetPrimAtPath("/object/child")

    assert USDParser._read_semantic_labels(link_prim) == {}


@pytest.mark.skipif(
    not USD_SEMANTICS_AVAILABLE, reason="UsdSemantics not available in this usd-core"
)
def test_parse_attaches_one_usd_semantic_labels_annotation_per_taxonomy():
    stage = build_single_joint_stage_with_semantic_labels()
    world = parse(stage)

    [child] = [body for body in world.bodies if body is not world.root]
    annotations = [
        annotation
        for annotation in world.semantic_annotations
        if isinstance(annotation, UsdSemanticLabels)
    ]
    labels_by_taxonomy = {
        annotation.taxonomy: annotation.labels for annotation in annotations
    }
    assert all(annotation.root is child for annotation in annotations)
    assert labels_by_taxonomy == {
        "class": ["chair", "furniture"],
        "category": ["seating"],
    }


def test_parse_attaches_no_annotation_for_an_unlabelled_body():
    stage = build_single_joint_stage("FixedJoint")
    world = parse(stage)

    assert not any(
        isinstance(annotation, UsdSemanticLabels)
        for annotation in world.semantic_annotations
    )


# %% materials


@pytest.fixture
def texture_file(tmp_path):
    path = tmp_path / "wood.png"
    Image.new("RGB", (2, 2), color=(120, 80, 40)).save(path)
    return str(path)


def test_diffuse_texture_path_resolves_the_bound_texture(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")

    resolved = UsdMeshShapeBuilder._diffuse_texture_path(mesh_prim)

    assert resolved == texture_file


def test_diffuse_texture_path_is_none_without_a_bound_material(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh_prim = stage.DefinePrim("/object/unbound_mesh", "Mesh")

    resolved = UsdMeshShapeBuilder._diffuse_texture_path(unbound_mesh_prim)

    assert resolved is None


def test_uv_coordinates_reads_the_per_point_st_primvar(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")

    uv = UsdMeshShapeBuilder._uv_coordinates(mesh_prim)

    np.testing.assert_array_equal(uv, [[0, 0], [1, 0], [1, 1], [0, 1]])


def test_uv_coordinates_is_none_without_an_st_primvar(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh_prim = stage.DefinePrim("/object/unbound_mesh", "Mesh")

    uv = UsdMeshShapeBuilder._uv_coordinates(unbound_mesh_prim)

    assert uv is None


def test_create_mesh_shape_applies_the_bound_texture(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    mesh_prim = stage.GetPrimAtPath("/object/mesh")
    body = Body(name=PrefixedName("test_body"))

    shape = UsdMeshShapeBuilder(mesh_prim, Gf.Matrix4d(1), body).build()

    mesh = shape.unscaled_mesh
    assert mesh.visual.kind == "texture"
    assert mesh.visual.uv is not None


def test_create_mesh_shape_has_no_texture_without_a_bound_material(texture_file):
    stage = build_stage_with_textured_mesh(texture_file)
    unbound_mesh = UsdGeom.Mesh.Define(stage, "/object/unbound_mesh")
    unbound_mesh.CreatePointsAttr([(0, 0, 0), (1, 0, 0), (1, 1, 0)])
    unbound_mesh.CreateFaceVertexCountsAttr([3])
    unbound_mesh.CreateFaceVertexIndicesAttr([0, 1, 2])
    body = Body(name=PrefixedName("test_body"))

    shape = UsdMeshShapeBuilder(unbound_mesh.GetPrim(), Gf.Matrix4d(1), body).build()

    assert shape.unscaled_mesh.visual.kind != "texture"


# %% triangulation


def test_triangulate_keeps_a_triangle_as_is():
    faces = UsdMeshShapeBuilder._triangulate([3], [0, 1, 2])
    np.testing.assert_array_equal(faces, [[0, 1, 2]])


def test_triangulate_fans_a_quad_into_two_triangles():
    faces = UsdMeshShapeBuilder._triangulate([4], [0, 1, 2, 3])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3]])


def test_triangulate_fans_a_pentagon_into_three_triangles():
    faces = UsdMeshShapeBuilder._triangulate([5], [0, 1, 2, 3, 4])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [0, 2, 3], [0, 3, 4]])


def test_triangulate_handles_multiple_faces_of_different_sizes():
    # A triangle followed by a quad, sharing no vertex indices.
    faces = UsdMeshShapeBuilder._triangulate([3, 4], [0, 1, 2, 3, 4, 5, 6])
    np.testing.assert_array_equal(faces, [[0, 1, 2], [3, 4, 5], [3, 5, 6]])
