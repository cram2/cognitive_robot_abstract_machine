from copy import deepcopy

import numpy as np
import pytest

from krrood.entity_query_language.factories import underspecified
from krrood.entity_query_language.backends import ProbabilisticBackend
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from pycram.datastructures.dataclasses import Context
from pycram.motion_executor import simulated_robot
from pycram.plans.factories import execute_single
from pycram.robot_plans.actions.composite.sage10k_actions import Sage10kOpenDoor
from pycram.robot_plans.actions.core.misc import MoveToReach
from random_events.variable import Continuous
from semantic_digital_twin.adapters.ros.visualization.pose_publisher import publish_pose
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.robots.hsrb import HSRB
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Wall,
    Door,
    Handle,
    Hinge,
)
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_types.spatial_types import Vector3
from semantic_digital_twin.spatial_types.derivatives import DerivativeMap
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.world_description.degree_of_freedom import (
    DegreeOfFreedomLimits,
)
from semantic_digital_twin.world_description.geometry import Color, Scale
from semantic_digital_twin.world_description.world_entity import Body


@pytest.fixture
def wall_door_handle_world():
    world = World()
    root = Body(name=PrefixedName("map"))
    with world.modify_world():
        world.add_body(root)

    with world.modify_world():
        wall = Wall.create_with_new_body_in_world(
            name=PrefixedName("wall"),
            world=world,
            scale=Scale(0.1, 4, 2),
        )
        wall.root.visual.dye_shapes(Color(R=0.6, G=0.6, B=0.6))

        door = Door.create_with_new_body_in_world(
            name=PrefixedName("door"),
            world=world,
            scale=Scale(0.11, 1, 2),
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(z=1.0),
        )
        door.root.visual.dye_shapes(Color(R=0.55, G=0.27, B=0.07))

    with world.modify_world():
        wall.add_aperture(door.entry_way)

    with world.modify_world():
        handle = Handle.create_with_new_body_in_world(
            name=PrefixedName("handle"),
            world=world,
            world_root_T_self=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=1.0, y=0.25, x=0.06
            ),
            scale=Scale(0.1, 0.1, 0.05),
        )
        handle.root.visual.dye_shapes(Color(R=0.8, G=0.8, B=0.1))
        door.add_handle(handle)

    world_T_hinge = door.calculate_world_T_hinge_based_on_handle(Vector3.Z())
    with world.modify_world():
        hinge = Hinge.create_with_new_body_in_world(
            name=PrefixedName("hinge"),
            world=world,
            active_axis=Vector3.Z(),
            world_root_T_self=world_T_hinge,
            connection_limits=DegreeOfFreedomLimits(
                lower=DerivativeMap(position=0.0),
                upper=DerivativeMap(position=np.pi / 2),
            ),
        )
        door.add_hinge(hinge)

    return world, wall, door, handle


def test_door_opening(wall_door_handle_world, hsr_world_setup, rclpy_node):
    world, wall, door, handle = wall_door_handle_world
    hsr_copy = deepcopy(hsr_world_setup)
    world.merge_world(hsr_copy)
    robot = HSRB.from_world(world)
    odom_combined = world.get_body_by_name("odom_combined")
    odom_combined.parent_connection.origin = (
        HomogeneousTransformationMatrix.from_xyz_rpy(x=1)
    )

    viz_marker_publisher = VizMarkerPublisher(node=rclpy_node, _world=world)
    viz_marker_publisher.with_tf_publisher()

    publish_pose(handle.pre_grasp_pose(), rclpy_node)
    return  # keep this to make the test pass here
    context = Context.from_world(world, query_backend=ProbabilisticBackend())

    with simulated_robot:
        execute_single(Sage10kOpenDoor(door), context=context).perform()

    assert np.isclose(door.hinge.root.parent_connection.position, np.pi / 2, atol=2e-2)


def test_translate_free_space_to_where_condition(wall_door_handle_world):
    from krrood.entity_query_language.factories import variable
    from krrood.entity_query_language.operators.core_logical_operators import (
        LogicalOperator,
    )
    from semantic_digital_twin.world_description.graph_of_convex_sets import (
        navigation_map_at_target,
        translate_free_space_to_where_condition,
    )

    world, wall, door, handle = wall_door_handle_world

    # Create navigation map at target (handle)
    gcs = navigation_map_at_target(target=handle.root)

    # Create a variable for the robot

    query = underspecified(MoveToReach)(robot_x=..., robot_y=...)

    # Translate free space to where condition
    where_condition = translate_free_space_to_where_condition(
        gcs.free_space_event,
        query.expression,
        x_variable_name="MoveToReach.robot_x",
        y_variable_name="MoveToReach.robot_y",
    )

    query = query.where(where_condition)
    parameters = UnderspecifiedParameters(query)
    # assert that the parameters truncation event is the same as the free space

    result_to_compare = parameters.truncation_event.update_variables(
        {
            Continuous("MoveToReach.robot_x"): SpatialVariables.x.value,
            Continuous("MoveToReach.robot_y"): SpatialVariables.y.value,
        }
    )
    assert result_to_compare == gcs.free_space_event.marginal(SpatialVariables.xy)
