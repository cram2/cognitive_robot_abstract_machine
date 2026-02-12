import json

import numpy as np
import pytest

from giskardpy.executor import Executor
from giskardpy.model.collision_matrix_manager import (
    CollisionRequest,
    CollisionAvoidanceTypes,
)
from giskardpy.motion_statechart.context import BuildContext
from giskardpy.motion_statechart.data_types import (
    LifeCycleValues,
    ObservationStateValues,
)
from giskardpy.motion_statechart.exceptions import (
    NodeNotFoundError,
)
from giskardpy.motion_statechart.goals.templates import Sequence
from giskardpy.motion_statechart.graph_node import (
    TrinaryCondition,
    EndMotion,
    CancelMotion,
)
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.motion_statechart import (
    MotionStatechart,
    LifeCycleState,
    ObservationState,
)
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList
from giskardpy.motion_statechart.test_nodes.test_nodes import (
    ConstTrueNode,
    TestNestedGoal,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.utils.utils import limits_from_urdf_joint
from krrood.symbolic_math.symbolic_math import (
    trinary_logic_and,
    trinary_logic_not,
    trinary_logic_or,
)
from semantic_digital_twin.adapters.ros.world_fetcher import (
    FetchWorldServer,
    fetch_world_from_service,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedom,
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.world_entity import Body


def test_cart_goal_simple(pr2_world_setup: World, rclpy_node):
    tip = pr2_world_setup.get_kinematic_structure_entity_by_name("base_footprint")
    root = pr2_world_setup.get_kinematic_structure_entity_by_name("odom_combined")
    tip_goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
        pos_x=-0.2, reference_frame=tip
    )

    msc = MotionStatechart()
    cart_goal = CartesianPose(
        root_link=root,
        tip_link=tip,
        goal_pose=tip_goal,
    )
    msc.add_node(cart_goal)
    end = EndMotion()
    msc.add_node(end)
    end.start_condition = cart_goal.observation_variable

    json_data = msc.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    fetcher = FetchWorldServer(node=rclpy_node, world=pr2_world_setup)
    pr2_world_copy = fetch_world_from_service(
        rclpy_node,
    )

    tracker = WorldEntityWithIDKwargsTracker.from_world(pr2_world_copy)
    kwargs = tracker.create_kwargs()
    msc_copy = MotionStatechart.from_json(new_json_data, **kwargs)

    kin_sim = Executor(
        world=pr2_world_copy,
        controller_config=QPControllerConfig.create_with_simulation_defaults(),
    )

    kin_sim.compile(motion_statechart=msc_copy)
    kin_sim.tick_until_end()

    fk = pr2_world_copy.compute_forward_kinematics_np(
        pr2_world_copy.get_kinematic_structure_entity_by_name(root.name),
        pr2_world_copy.get_kinematic_structure_entity_by_name(tip.name),
    )
    assert np.allclose(fk, tip_goal, atol=cart_goal.threshold)
    fetcher.close()
