import numpy as np

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import ObservationStateValues
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.grasp_bar import GraspBar
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world import World


def test_grasp_bar(pr2_world_state_reset: World, rclpy_node):
    VizMarkerPublisher(
        _world=pr2_world_state_reset, node=rclpy_node
    ).with_tf_publisher()
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")

    bar_center = Point3(0.6, -0.3, 1.0, reference_frame=root)
    bar_axis = Vector3.Z(reference_frame=root)
    tip_grasp_axis = Vector3.X(reference_frame=tip)

    msc = MotionStatechart()
    grasp = GraspBar(
        root_link=root,
        tip_link=tip,
        tip_grasp_axis=tip_grasp_axis,
        bar_center=bar_center,
        bar_axis=bar_axis,
        bar_length=0.4,
    )
    msc.add_node(grasp)
    msc.add_node(EndMotion.when_true(grasp))

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    assert grasp.observation_state == ObservationStateValues.TRUE

    # Tip must end up within the bar segment and aligned with the bar axis.
    root_P_tip = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=Point3(0, 0, 0, reference_frame=tip)
    ).to_np()[:3]
    bar_center_np = bar_center.to_np()[:3]
    distance = np.linalg.norm(np.cross(root_P_tip - bar_center_np, [0, 0, 1]))
    assert distance <= grasp.threshold + 1e-2
