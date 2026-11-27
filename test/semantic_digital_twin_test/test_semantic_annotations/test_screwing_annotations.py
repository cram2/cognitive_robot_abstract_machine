"""
Tests for the :class:`Bolt` and :class:`Screwdriver` semantic annotations.
"""

from __future__ import annotations

import numpy as np

from krrood.adapters.json_serializer import from_json, to_json
from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.part_whole.relationship_marker import (
    IsPartWholeRelationship,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bolt,
    BoltHead,
    BoltShaft,
    Screwdriver,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body

# %% world setup


def _build_world_with_body(*, name: str) -> tuple[World, Body]:
    """
    Create a world containing a root body and a child body attached to it.

    :param name: Name of the child body.
    :return: The world and spawned child body.
    """
    world = World()
    root = Body(name=PrefixedName(name="world"))
    body = Body(name=PrefixedName(name=name))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)
        world.add_kinematic_structure_entity(body)
        world.add_connection(FixedConnection(parent=root, child=body))
    return world, body


def _assert_axial_top_down_candidates(
    *,
    candidates: list,
    expected_graspable,
    expected_frame: Body,
    expected_count: int,
) -> None:
    """
    Assert that ``candidates`` are evenly rotated top-down grasps along local ``-X`` at
    ``expected_frame``'s origin.

    :param candidates: Generated grasp candidates to inspect.
    :param expected_graspable: Annotation expected on every candidate.
    :param expected_frame: Reference frame expected on every candidate pose.
    :param expected_count: Number of expected candidates.
    """
    assert len(candidates) == expected_count
    expected_angles = np.linspace(0.0, 2.0 * np.pi, expected_count, endpoint=False)
    for candidate, angle in zip(candidates, expected_angles):
        assert candidate.graspable is expected_graspable
        assert candidate.grasp_pose.reference_frame is expected_frame
        np.testing.assert_allclose(
            candidate.grasp_pose.position.to_np()[:3],
            [0.0, 0.0, 0.0],
            atol=1e-9,
        )
        rotation_matrix = candidate.grasp_pose.rotation_matrix.to_np()
        np.testing.assert_allclose(rotation_matrix[:3, 0], [-1.0, 0.0, 0.0], atol=1e-9)
        expected_orientation = RotationMatrix.from_vectors(
            x=Vector3(-1, 0, 0),
            y=Vector3(0.0, float(np.cos(angle)), float(np.sin(angle))),
        )
        np.testing.assert_allclose(
            rotation_matrix,
            expected_orientation.to_np(),
            atol=1e-9,
        )


# %% Bolt tests


def test_bolt_models_head_and_shaft_as_part_whole_relationships() -> None:
    """
    Verify :class:`Bolt` models :class:`BoltHead` and :class:`BoltShaft` as part-whole
    relationships and mounts both parts under its root body.
    """
    assert IsPartWholeRelationship.of_field(clazz=Bolt, field_name="head") is not None
    assert IsPartWholeRelationship.of_field(clazz=Bolt, field_name="shaft") is not None

    world, bolt_body = _build_world_with_body(name="bolt")
    head_body = Body(name=PrefixedName(name="bolt_head"))
    shaft_body = Body(name=PrefixedName(name="bolt_shaft"))
    with world.modify_world():
        world.add_kinematic_structure_entity(head_body)
        world.add_kinematic_structure_entity(shaft_body)
        world.add_connection(
            FixedConnection(
                parent=bolt_body,
                child=head_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.025, reference_frame=bolt_body
                ),
            )
        )
        world.add_connection(FixedConnection(parent=bolt_body, child=shaft_body))
        bolt = Bolt(root=bolt_body)
        head = BoltHead(root=head_body)
        shaft = BoltShaft(root=shaft_body)
        world.add_semantic_annotations([bolt, head, shaft])
        bolt.add(part=head)
        bolt.add(part=shaft)

    assert bolt.head is head
    assert bolt.shaft is shaft
    assert bolt.parts == [head, shaft]
    assert head.root.parent_kinematic_structure_entity is bolt.root
    assert shaft.root.parent_kinematic_structure_entity is bolt.root


def test_bolt_grasp_candidates_delegate_to_bolt_head() -> None:
    """
    Verify :meth:`Bolt.grasp_candidates` delegates to :attr:`Bolt.head` when set.
    """
    world, bolt_body = _build_world_with_body(name="bolt")
    head_body = Body(name=PrefixedName(name="bolt_head"))
    with world.modify_world():
        world.add_kinematic_structure_entity(head_body)
        world.add_connection(
            FixedConnection(
                parent=bolt_body,
                child=head_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.025, reference_frame=bolt_body
                ),
            )
        )
        bolt = Bolt(root=bolt_body)
        head = BoltHead(root=head_body)
        world.add_semantic_annotations([bolt, head])
        bolt.add(part=head)

    assert bolt.grasped_part() is head
    _assert_axial_top_down_candidates(
        candidates=bolt.grasp_candidates(),
        expected_graspable=head,
        expected_frame=head_body,
        expected_count=head.grasp_candidate_count,
    )


def test_bolt_head_default_grasp_candidates_follow_grasp_candidate_count() -> None:
    """
    Verify :meth:`BoltHead.default_grasp_candidates` respects
    :attr:`BoltHead.grasp_candidate_count`.
    """
    world, head_body = _build_world_with_body(name="bolt_head")
    head = BoltHead(root=head_body, grasp_candidate_count=6)
    with world.modify_world():
        world.add_semantic_annotations([head])

    _assert_axial_top_down_candidates(
        candidates=head.default_grasp_candidates(),
        expected_graspable=head,
        expected_frame=head_body,
        expected_count=6,
    )


def test_bolt_falls_back_to_own_root_when_head_is_missing() -> None:
    """
    Verify :meth:`Bolt.grasp_candidates` falls back to ``bolt.root`` when no
    :attr:`Bolt.head` is attached.
    """
    world, body = _build_world_with_body(name="bolt")
    bolt = Bolt(root=body, grasp_candidate_count=6)
    with world.modify_world():
        world.add_semantic_annotations([bolt])

    assert bolt.grasped_part() is bolt
    _assert_axial_top_down_candidates(
        candidates=bolt.grasp_candidates(),
        expected_graspable=bolt,
        expected_frame=body,
        expected_count=6,
    )


def test_bolt_with_head_and_shaft_survives_json_round_trip() -> None:
    """
    Verify a :class:`Bolt` with :class:`BoltHead` and :class:`BoltShaft` parts survives
    world modification JSON serialization.
    """
    world, bolt_body = _build_world_with_body(name="bolt")
    head_body = Body(name=PrefixedName(name="bolt_head"))
    shaft_body = Body(name=PrefixedName(name="bolt_shaft"))
    with world.modify_world():
        world.add_kinematic_structure_entity(head_body)
        world.add_kinematic_structure_entity(shaft_body)
        world.add_connection(FixedConnection(parent=bolt_body, child=head_body))
        world.add_connection(FixedConnection(parent=bolt_body, child=shaft_body))
        bolt = Bolt(root=bolt_body)
        head = BoltHead(root=head_body)
        shaft = BoltShaft(root=shaft_body)
        world.add_semantic_annotations([bolt, head, shaft])
        bolt.add(part=head)
        bolt.add(part=shaft)

    other_world = World()
    tracker = WorldEntityWithIDKwargsTracker()
    for block in world.get_world_model_manager().model_modification_blocks:
        sent = from_json(to_json(block), **tracker.create_kwargs())
        with other_world.modify_world():
            sent.apply(world=other_world)

    [arrived_bolt] = other_world.get_semantic_annotations_by_type(Bolt)
    assert isinstance(arrived_bolt.head, BoltHead)
    assert isinstance(arrived_bolt.shaft, BoltShaft)


# %% Screwdriver tests


def test_screwdriver_default_grasp_candidates_rotate_about_axis_at_grasp_height() -> (
    None
):
    """
    Verify :meth:`Screwdriver.default_grasp_candidates` returns rotated handle grasps at
    :attr:`Screwdriver.grasp_height` along local +Z.
    """
    world, body = _build_world_with_body(name="screwdriver")
    default_screwdriver = Screwdriver(root=body)
    assert default_screwdriver.grasp_height == 0.025

    grasp_height = 0.03
    screwdriver = Screwdriver(root=body, grasp_height=grasp_height)
    with world.modify_world():
        world.add_semantic_annotations([screwdriver])

    candidates = screwdriver.default_grasp_candidates()
    assert len(candidates) == screwdriver.grasp_candidate_count

    expected_yaws = np.linspace(
        0.0, 2.0 * np.pi, screwdriver.grasp_candidate_count, endpoint=False
    )
    for candidate, yaw in zip(candidates, expected_yaws):
        assert candidate.graspable is screwdriver
        assert candidate.grasp_pose.reference_frame is body
        np.testing.assert_allclose(
            candidate.grasp_pose.position.to_np()[:3],
            [0.0, 0.0, grasp_height],
            atol=1e-9,
        )
        rotation_matrix = candidate.grasp_pose.rotation_matrix.to_np()
        np.testing.assert_allclose(rotation_matrix[:3, 0], [0.0, 0.0, -1.0], atol=1e-9)
        expected_orientation = RotationMatrix.from_rpy(yaw=float(yaw), pitch=np.pi / 2)
        np.testing.assert_allclose(
            rotation_matrix,
            expected_orientation.to_np(),
            atol=1e-9,
        )


def test_screwdriver_default_grasp_candidates_follow_grasp_candidate_count() -> None:
    """
    Verify :meth:`Screwdriver.default_grasp_candidates` respects
    :attr:`Screwdriver.grasp_candidate_count`.
    """
    world, body = _build_world_with_body(name="screwdriver")
    screwdriver = Screwdriver(root=body, grasp_candidate_count=6)
    with world.modify_world():
        world.add_semantic_annotations([screwdriver])

    candidates = screwdriver.default_grasp_candidates()
    assert len(candidates) == 6
    expected_yaws = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    for candidate, yaw in zip(candidates, expected_yaws):
        expected_orientation = RotationMatrix.from_rpy(yaw=float(yaw), pitch=np.pi / 2)
        np.testing.assert_allclose(
            candidate.grasp_pose.rotation_matrix.to_np(),
            expected_orientation.to_np(),
            atol=1e-9,
        )


def test_screwdriver_tool_alignment_uses_tip_body_and_target_frame() -> None:
    """
    Verify :meth:`Screwdriver.tool_alignment` uses the dedicated tip body and handles
    both :class:`Body` and :class:`Pose` targets.
    """
    world, handle_body = _build_world_with_body(name="screwdriver")
    tip_body = Body(name=PrefixedName(name="screwdriver_end_effector"))
    target_body = Body(name=PrefixedName(name="bolt"))
    with world.modify_world():
        world.add_kinematic_structure_entity(tip_body)
        world.add_connection(
            FixedConnection(
                parent=handle_body,
                child=tip_body,
                parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                    z=-0.05, reference_frame=handle_body
                ),
            )
        )
        world.add_kinematic_structure_entity(target_body)
        world.add_connection(FixedConnection(parent=world.root, child=target_body))
        screwdriver = Screwdriver(root=handle_body)
        world.add_semantic_annotations([screwdriver])

    assert screwdriver.get_tool_frame() is tip_body

    for target in (target_body, Pose(reference_frame=target_body)):
        pairs = screwdriver.tool_alignment(target)
        assert len(pairs) == 1
        assert pairs[0].tip_normal.reference_frame is tip_body
        assert pairs[0].reference_normal.reference_frame is target_body
        np.testing.assert_allclose(
            pairs[0].tip_normal.to_np()[:3], [0.0, 0.0, 1.0], atol=1e-9
        )
        np.testing.assert_allclose(
            pairs[0].reference_normal.to_np()[:3], [0.0, 0.0, 1.0], atol=1e-9
        )


def test_screwdriver_tool_alignment_falls_back_to_root_without_tip_body() -> None:
    """
    Verify :meth:`Screwdriver.tool_alignment` falls back to ``root`` when no dedicated
    tip body exists.
    """
    world, handle_body = _build_world_with_body(name="screwdriver")
    target_body = Body(name=PrefixedName(name="target"))
    with world.modify_world():
        world.add_kinematic_structure_entity(target_body)
        world.add_connection(FixedConnection(parent=world.root, child=target_body))
        screwdriver = Screwdriver(root=handle_body)
        world.add_semantic_annotations([screwdriver])

    assert screwdriver.get_tool_frame() is handle_body
    pairs = screwdriver.tool_alignment(target_body)
    assert len(pairs) == 1
    assert pairs[0].tip_normal.reference_frame is handle_body
    assert pairs[0].reference_normal.reference_frame is target_body
