"""
Unit tests for :func:`cramera.body_geometry.measure_body` and
:func:`~cramera.body_geometry.rounded_scale`.

Every shape is attached to a real, single-body ``World`` rather than a duck-typed mimic:
``ShapeCollection.scale`` transforms each shape's bounding box through its reference
frame, which requires that frame to belong to an actual world.
"""

from __future__ import annotations

import pytest
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom
from semantic_digital_twin.world_description.geometry import (
    Box,
    Cylinder,
    Mesh,
    Scale,
    Sphere,
)
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
    Point3,
    Pose,
)
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body, Region
from typing_extensions import Any, Optional

from cramera.body_geometry import (
    measure_body,
    mesh_file_of,
    NumericPose,
    pose_label,
    position_label,
    rounded_pose,
    rounded_scale,
)


# %% fixtures
def _body_with_shapes(
    visual: Optional[ShapeCollection] = None,
    collision: Optional[ShapeCollection] = None,
) -> Body:
    """
    A real ``Body``, registered in its own single-body ``World``.

    ``ShapeCollection.scale`` reads each shape's reference frame back to a world via
    :meth:`Body.__post_init__`'s wiring, so a bare, unregistered ``Body`` is not enough.
    """
    body = Body(
        name=PrefixedName("object"),
        visual=visual if visual is not None else ShapeCollection(),
        collision=collision if collision is not None else ShapeCollection(),
    )
    world = World()
    with world.modify_world():
        world.add_body(body)
    return body


def _refuse_to_build(*args: Any, **kwargs: Any) -> Any:
    """
    Stand in for a symbolic call that a published read must never make.
    """
    raise AssertionError("the read path built a symbolic expression")


# %% measure_body
def test_of_measures_a_box():
    body = _body_with_shapes(
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.2, 0.3, 0.4))])
    )
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.2, 0.3, 0.4])


def test_of_measures_a_mesh():
    """
    A unit mesh scaled by ``Scale(0.2, 0.3, 0.4)`` measures to that scale.

    True whether read directly off the ``Mesh.scale`` field (the pre-fix path) or from
    the scaled geometry's own bounding box (the post-fix path) - both agree here.
    """
    body = _body_with_shapes(
        collision=ShapeCollection(
            shapes=[Mesh.box(extents=(1.0, 1.0, 1.0), scale=Scale(0.2, 0.3, 0.4))]
        )
    )
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.2, 0.3, 0.4])


def test_of_measures_a_sphere():
    """
    A sphere has no ``.scale`` attribute; before the fix this reported ``None``.
    """
    body = _body_with_shapes(collision=ShapeCollection(shapes=[Sphere(radius=0.5)]))
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([1.0, 1.0, 1.0])


def test_of_measures_a_cylinder():
    """
    A cylinder has no ``.scale`` attribute; before the fix this reported ``None``.
    """
    body = _body_with_shapes(
        collision=ShapeCollection(shapes=[Cylinder(width=0.4, height=0.6)])
    )
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.4, 0.4, 0.6])


def test_of_returns_none_when_the_body_has_no_shapes():
    body = _body_with_shapes()
    assert measure_body(body) is None


def test_of_prefers_visual_over_collision():
    body = _body_with_shapes(
        visual=ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))]),
        collision=ShapeCollection(shapes=[Box(scale=Scale(0.9, 0.9, 0.9))]),
    )
    extent = measure_body(body)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.1, 0.1, 0.1])


# %% regions
def _region_with_area(shapes: ShapeCollection) -> Region:
    """
    A real ``Region``, fixed to the root of its own world.

    A region's shapes are measured through their reference frame just as a body's are,
    so the region has to live in an actual world too.
    """
    region = Region(name=PrefixedName("landmark"), area=shapes)
    world = World()
    root = Body(name=PrefixedName("world"))
    with world.modify_world():
        world.add_body(root)
        world.add_connection(
            FixedConnection(
                parent=root,
                child=region,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.5, 0.0, 0.0
                ),
            )
        )
    return region


def test_of_measures_a_regions_area():
    region = _region_with_area(
        ShapeCollection(shapes=[Box(scale=Scale(0.2, 0.3, 0.4))])
    )
    extent = measure_body(region)
    assert [extent.x, extent.y, extent.z] == pytest.approx([0.2, 0.3, 0.4])


def test_mesh_file_of_reads_a_regions_area_mesh(tmp_path):
    mesh_file = tmp_path / "hole_marker.obj"
    mesh_file.write_text("o marker\n")
    region = _region_with_area(ShapeCollection(shapes=[Mesh(filename=str(mesh_file))]))
    assert mesh_file_of(region) == str(mesh_file)


def test_rounded_pose_reads_a_regions_world_pose():
    region = _region_with_area(
        ShapeCollection(shapes=[Box(scale=Scale(0.1, 0.1, 0.1))])
    )
    assert rounded_pose(region) == [0.5, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]


# %% rounded_scale
def test_rounded_scale_rounds_each_axis():
    assert rounded_scale(Scale(x=0.123456, y=1.0, z=2.987654), 3) == [0.123, 1.0, 2.988]


# %% rounded_pose
def test_rounded_pose_reports_position_then_quaternion():
    """
    The pose is published in the ``[x, y, z, qx, qy, qz, qw]`` order the viewer reads,
    which is the order semantic_digital_twin's own conversion produces.
    """
    body = _body_with_shapes()
    assert rounded_pose(body, 5) == pytest.approx(
        [
            *body.global_pose.to_position().to_np().tolist()[:3],
            *body.global_pose.to_quaternion().to_np().tolist(),
        ]
    )


# %% display labels
def test_pose_label_reports_the_position_before_the_orientation():
    """
    A pose reads as its position followed by its quaternion, so a query answer naming a
    target is readable without the reader knowing the field order.
    """
    pose = Pose.from_xyz_rpy(1.0, 2.0, 3.0)

    assert pose_label(pose) == "(1.00, 2.00, 3.00) q(0.00, 0.00, 0.00, 1.00)"


def test_pose_label_positions_agree_with_position_label():
    """
    Both labels format coordinates the same way, so a position and the position part of
    a pose never disagree in an answer that shows both.
    """
    pose = Pose.from_xyz_rpy(0.125, -1.5, 0.0)

    assert pose_label(pose).startswith(position_label(Point3(0.125, -1.5, 0.0)))


# %% poses read out into plain numbers
def test_a_numeric_pose_carries_the_poses_own_coordinates():
    """
    Reading a pose out is what makes it safe to hand to another thread, so the numbers
    it keeps have to be the pose's own.
    """
    pose = Pose.from_xyz_rpy(1.5, -2.0, 0.25)

    read_out = NumericPose.of_pose(pose)

    assert read_out.position == (1.5, -2.0, 0.25)
    assert read_out.quaternion == (0.0, 0.0, 0.0, 1.0)


def test_a_numeric_pose_is_free_of_the_symbolic_pose_it_was_read_from():
    """
    A value still holding a CasADi expression would be evaluated again by whichever
    thread renders it, which is the whole hazard being avoided.
    """
    read_out = NumericPose.of_pose(Pose.from_xyz_rpy(1.0, 2.0, 3.0))

    assert all(type(value) is float for value in read_out.position)
    assert all(type(value) is float for value in read_out.quaternion)


def test_a_numeric_pose_reads_exactly_as_the_pose_it_was_read_from():
    """
    Recording a pose as numbers must not change how an answer showing it reads.
    """
    pose = Pose.from_xyz_rpy(0.125, -1.5, 0.0)

    assert NumericPose.of_pose(pose).label == pose_label(pose)


def test_rounded_pose_rounds_every_value():
    world = World()
    root = Body(name=PrefixedName("world"))
    body = Body(name=PrefixedName("object"))
    with world.modify_world():
        degrees_of_freedom = {
            component: DegreeOfFreedom(name=PrefixedName(component))
            for component in ("x", "y", "z", "qx", "qy", "qz", "qw")
        }
        for degree_of_freedom in degrees_of_freedom.values():
            world.add_degree_of_freedom(degree_of_freedom)
        world.add_connection(
            Connection6DoF(parent=root, child=body, **degrees_of_freedom)
        )
        world.state[degrees_of_freedom["qw"].id].position = 1.0
        world.state[degrees_of_freedom["x"].id].position = 0.123456789

    assert rounded_pose(body, 3)[0] == 0.123


def test_rounded_pose_builds_no_transformation_matrix(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """
    Publishing runs while a demo plans, and wrapping forward kinematics in a
    transformation matrix is what would make publishing a pose a CasADi call.
    """
    world = World()
    root = Body(name=PrefixedName("world"))
    body = Body(name=PrefixedName("object"))
    with world.modify_world():
        world.add_connection(
            FixedConnection(
                parent=root,
                child=body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    0.5, 1.5, -0.25, 0.3, -1.1, 2.4
                ),
            )
        )
    expected = rounded_pose(body)
    monkeypatch.setattr(World, "compute_forward_kinematics", _refuse_to_build)

    assert rounded_pose(body) == expected
