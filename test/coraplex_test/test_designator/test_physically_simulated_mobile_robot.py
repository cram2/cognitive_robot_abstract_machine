"""
Tests for a mobile robot performing a plan in a physical simulation, where its own
servos move its joints and its drive moves its base across the floor.
"""

from __future__ import annotations

import pytest

from ...plan_running import perform_in_physics, robot_extensions
from ...pytest_environment import runs_in_continuous_integration

from coraplex.robot_plans.actions.core.navigation import NavigateAction
from coraplex.robot_plans.actions.core.robot_body import (
    MoveTorsoAction,
    ParkArmsAction,
)
from cramph.composites import Sequence
from semantic_digital_twin.datastructures.definitions import TorsoState
from krrood_spatial_math.spatial_types import Pose3Dimensional


@pytest.mark.skipif(
    not runs_in_continuous_integration(), reason="MuJoCo tests only run in CI"
)
def test_the_hsrb_raises_its_torso_and_drives_to_where_it_is_sent(
    simulated_hsrb,
) -> None:
    """
    In a physical simulation the HSR raises its torso to the joint positions asked for
    and drives to where it is sent.
    """
    hsrb, simulation = simulated_hsrb
    world = hsrb._world
    goal = Pose3Dimensional.from_xyz_rpy(
        1.0, 0.5, 0.0, yaw=0.5, reference_frame=world.root
    )
    raised = hsrb.mobile_base.torso.get_joint_state_by_type(TorsoState.HIGH)
    plan = Sequence(
        nodes=[
            ParkArmsAction(arms=hsrb.all_arms),
            MoveTorsoAction(torso_state=TorsoState.HIGH),
            NavigateAction(planar_goal_pose=goal),
        ]
    )

    perform_in_physics(plan, robot_extensions(hsrb), simulation)

    world_T_base = world.compute_forward_kinematics_np(world.root, hsrb.root)
    simulated_torso_lift = {
        connection.name.name: simulation.simulator.get_joint_value(
            connection.name.name
        ).result
        for connection in raised.connections
    }
    assert world_T_base[:3, 3].tolist() == pytest.approx(
        goal.position.to_np()[:3].tolist(), abs=0.02
    )
    assert simulated_torso_lift == pytest.approx(
        {
            connection.name.name: target
            for connection, target in zip(raised.connections, raised.target_values)
        },
        abs=0.02,
    )
