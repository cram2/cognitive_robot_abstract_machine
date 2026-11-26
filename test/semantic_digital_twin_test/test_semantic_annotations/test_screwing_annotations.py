"""
Tests for the :class:`Bolt` and :class:`Screwdriver` semantic annotations.
"""

from __future__ import annotations

import numpy as np

from krrood_spatial_math.spatial_types import (
    HomogeneousTransformationMatrix,
    Pose,
    RotationMatrix,
    Vector3,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Bolt,
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


# %% Bolt tests


def test_bolt_default_grasp_candidates_rotate_about_axis_at_head_offset() -> None:
    """
    Verify :meth:`Bolt.default_grasp_candidates` returns rotated top-down grasps at
    :attr:`Bolt.head_offset` along local +X.
    """
    world, body = _build_world_with_body(name="bolt")
    default_bolt = Bolt(root=body)
    assert default_bolt.head_offset == 0.025

    custom_offset = 0.04
    bolt = Bolt(root=body, head_offset=custom_offset)
    with world.modify_world():
        world.add_semantic_annotations([bolt])

    candidates = bolt.default_grasp_candidates()
    assert len(candidates) == bolt.grasp_candidate_count

    expected_angles = np.linspace(
        0.0, 2.0 * np.pi, bolt.grasp_candidate_count, endpoint=False
    )
    for candidate, angle in zip(candidates, expected_angles):
        assert candidate.graspable is bolt
        assert candidate.grasp_pose.reference_frame is body
        np.testing.assert_allclose(
            candidate.grasp_pose.position.to_np()[:3],
            [custom_offset, 0.0, 0.0],
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


def test_bolt_default_grasp_candidates_follow_grasp_candidate_count() -> None:
    """
    Verify :meth:`Bolt.default_grasp_candidates` respects
    :attr:`Bolt.grasp_candidate_count`.
    """
    world, body = _build_world_with_body(name="bolt")
    bolt = Bolt(root=body, grasp_candidate_count=6)
    with world.modify_world():
        world.add_semantic_annotations([bolt])

    candidates = bolt.default_grasp_candidates()
    assert len(candidates) == 6
    expected_angles = np.linspace(0.0, 2.0 * np.pi, 6, endpoint=False)
    for candidate, angle in zip(candidates, expected_angles):
        expected_orientation = RotationMatrix.from_vectors(
            x=Vector3(-1, 0, 0),
            y=Vector3(0.0, float(np.cos(angle)), float(np.sin(angle))),
        )
        np.testing.assert_allclose(
            candidate.grasp_pose.rotation_matrix.to_np(),
            expected_orientation.to_np(),
            atol=1e-9,
        )


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
        assert pairs[0].goal_normal.reference_frame is target_body
        np.testing.assert_allclose(
            pairs[0].tip_normal.to_np()[:3], [0.0, 0.0, 1.0], atol=1e-9
        )
        np.testing.assert_allclose(
            pairs[0].goal_normal.to_np()[:3], [0.0, 0.0, 1.0], atol=1e-9
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
    assert pairs[0].goal_normal.reference_frame is target_body
