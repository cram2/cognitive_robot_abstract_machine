"""
Simox poses retain their meaning through the standard grasp interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

from coraplex.external_interfaces.simox_grasp_planner import (
    SimoxConfiguration,
    SimoxGraspPlanner,
    SimoxResponse,
    SimoxRequest,
    SimoxService,
)
from coraplex.external_interfaces.exceptions import (
    InvalidSimoxResponse,
    SimoxPlanningFailed,
)
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix, Pose

# %% service boundary


@dataclass
class RecordedGrasps(SimoxService):
    """
    Return a recorded response without a ROS session.
    """

    response: SimoxResponse
    """Response received by the planner."""

    def plan(self, request: SimoxRequest) -> SimoxResponse:
        assert request.object_model_path.exists()
        return self.response


@pytest.fixture
def configuration() -> SimoxConfiguration:
    """
    The PR2 calibration used by the external model.
    """
    return SimoxConfiguration.for_pr2(Path("pr2.xml"))


# %% pose and metadata contract


def test_candidate_preserves_tool_goal_and_follows_object(
    pr2_apartment_context: tuple, configuration: SimoxConfiguration
) -> None:
    world, robot, _ = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    arm = robot.get_right_arm_if_specified()
    base_T_object = world.transform(milk.root.global_pose, robot.root)
    base_T_simox = Pose.from_xyz_rpy(0.6, -0.2, 0.9, roll=0.4, pitch=0.7, yaw=-0.3)
    response = SimoxResponse([base_T_simox], [0.7], [False])
    planner = SimoxGraspPlanner(configuration, RecordedGrasps(response))

    results = planner.plan_grasps_for_body(milk, arm, robot.root)

    assert len(results) == 1
    result = results[0]
    assert type(result.candidate) is GraspCandidate
    assert result.candidate.graspable is milk
    assert result.candidate.grasp_pose.reference_frame is milk.root
    assert result.quality == 0.7
    assert result.is_force_closure is False
    expected = base_T_simox.to_np() @ configuration.simox_T_tool.to_np()
    reconstructed = (
        base_T_object.to_np()
        @ arm.end_effector.tool_frame_goal_for_grasp(
            result.candidate.grasp_pose
        ).to_np()
    )
    np.testing.assert_allclose(reconstructed, expected, atol=1e-8)
    relative_pose = result.candidate.grasp_pose.to_np().copy()
    with world.modify_world():
        milk.root.parent_connection.origin = (
            HomogeneousTransformationMatrix.from_xyz_rpy(
                1.0, 0.5, 0.8, yaw=1.2, reference_frame=world.root
            )
        )
    np.testing.assert_allclose(result.candidate.grasp_pose.to_np(), relative_pose)
    np.testing.assert_allclose(
        result.candidate.world_T_grasp.to_np(),
        milk.root.global_pose.to_np() @ relative_pose,
    )


def test_response_ranking_preserves_metadata(
    pr2_apartment_context: tuple, configuration: SimoxConfiguration
) -> None:
    world, robot, _ = pr2_apartment_context
    milk = world.get_semantic_annotations_by_type(Milk)[0]
    response = SimoxResponse([Pose(), Pose()], [0.2, 0.9], [False, True])
    planner = SimoxGraspPlanner(configuration, RecordedGrasps(response))
    results = planner.plan_grasps_for_body(
        milk, robot.get_right_arm_if_specified(), robot.root
    )
    assert [result.quality for result in results] == [0.9, 0.2]
    assert [result.is_force_closure for result in results] == [True, False]


@pytest.mark.parametrize(
    "response",
    [
        SimoxResponse([Pose()], [], [True]),
        SimoxResponse([Pose()], [float("nan")], [True]),
    ],
)
def test_malformed_response_is_rejected(
    pr2_apartment_context: tuple,
    configuration: SimoxConfiguration,
    response: SimoxResponse,
) -> None:
    world, robot, _ = pr2_apartment_context
    planner = SimoxGraspPlanner(configuration, RecordedGrasps(response))
    with pytest.raises(InvalidSimoxResponse):
        planner.plan_grasps_for_body(
            world.get_semantic_annotations_by_type(Milk)[0],
            robot.get_right_arm_if_specified(),
            robot.root,
        )


def test_empty_response_is_rejected(
    pr2_apartment_context: tuple, configuration: SimoxConfiguration
) -> None:
    world, robot, _ = pr2_apartment_context
    planner = SimoxGraspPlanner(
        configuration, RecordedGrasps(SimoxResponse([], [], []))
    )
    with pytest.raises(SimoxPlanningFailed):
        planner.plan_grasps_for_body(
            world.get_semantic_annotations_by_type(Milk)[0],
            robot.get_right_arm_if_specified(),
            robot.root,
        )
