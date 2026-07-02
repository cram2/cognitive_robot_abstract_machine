import random

import numpy as np

from giskardpy.executor import Executor, SimulationPacer
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import (
    ObservationStateValues,
    DefaultWeights,
)
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.monitors.payload_monitors import (
    CountControlCycles,
    CountSimulationTimeSeconds,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.wiggle_insert import WiggleInsert
from giskardpy.ros_executor import Ros2Executor
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world import World
from test.semantic_digital_twin_test.test_orm.test_orm import hsr_world_state_reset


def _hole_at_current_tip(world: World, tip, root) -> Point3:
    root_P_tip = world.transform(
        target_frame=root, spatial_object=Point3(0, 0, 0, reference_frame=tip)
    ).to_np()[:3]
    return Point3(*root_P_tip, reference_frame=root)


def test_wiggle_insert_reaches_hole(pr2_world_state_reset: World, rclpy_node):
    VizMarkerPublisher(
        _world=pr2_world_state_reset, node=rclpy_node
    ).with_tf_publisher()
    random.seed(0)
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    hole_point = _hole_at_current_tip(pr2_world_state_reset, tip, root)

    msc = MotionStatechart()
    wiggle = WiggleInsert(
        root_link=root,
        tip_link=tip,
        hole_point=hole_point,
        hole_normal=Vector3.Z(reference_frame=root),
        noise_translation=0.01,
        noise_angle=0.05,
        down_velocity=0.05,
        threshold=0.02,
    )
    msc.add_node(wiggle)
    msc.add_node(EndMotion.when_true(wiggle))

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_state_reset))
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    assert wiggle.observation_state == ObservationStateValues.TRUE


def test_wiggle_insert_basis_uses_root_frame_hole_normal(pr2_world_state_reset: World):
    """
    The wiggle basis vectors must be perpendicular to hole_normal as expressed in root_link,
    since the wiggle noise they span is added to a root_link-frame point (root_P_hole).
    """
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    hole_point = _hole_at_current_tip(pr2_world_state_reset, tip, root)
    hole_normal_in_tip_frame = Vector3.Z(reference_frame=tip)

    wiggle = WiggleInsert(
        root_link=root,
        tip_link=tip,
        hole_point=hole_point,
        hole_normal=hole_normal_in_tip_frame,
    )
    context = MotionStatechartContext(world=pr2_world_state_reset)
    wiggle.build(context)

    expected_normal_in_root = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=hole_normal_in_tip_frame
    ).to_np()[:3]

    assert np.isclose(np.dot(wiggle._v1, expected_normal_in_root), 0.0, atol=1e-9)
    assert np.isclose(np.dot(wiggle._v2, expected_normal_in_root), 0.0, atol=1e-9)


def test_wiggle_insert_on_tick_updates_noise(pr2_world_state_reset: World):
    random.seed(1)
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    hole_point = _hole_at_current_tip(pr2_world_state_reset, tip, root)

    msc = MotionStatechart()
    wiggle = WiggleInsert(
        root_link=root,
        tip_link=tip,
        hole_point=hole_point,
        hole_normal=Vector3.Z(reference_frame=root),
        noise_translation=0.05,
        noise_angle=0.2,
        down_velocity=0.05,
    )
    msc.add_node(wiggle)
    msc.add_node(EndMotion.when_true(wiggle))

    context = MotionStatechartContext(world=pr2_world_state_reset)
    kin_sim = Executor(context)
    kin_sim.compile(motion_statechart=msc)

    kin_sim.tick()
    first_translation = wiggle._rand_translation.evaluate().flatten()[:3].copy()
    kin_sim.tick()
    second_translation = wiggle._rand_translation.evaluate().flatten()[:3].copy()

    # The injected noise must change between control cycles.
    assert not np.allclose(first_translation, second_translation)


def test_wiggle_insert(hsr_world_state_reset):
    goal_state = {
        "arm_flex_joint": -1.5,
        "arm_lift_joint": 0.5,
        "arm_roll_joint": 0.0,
        "head_pan_joint": 0.0,
        "head_tilt_joint": 0.0,
        "wrist_flex_joint": -1.5,
        "wrist_roll_joint": 0.0,
    }

    hpl = hsr_world_state_reset.get_body_by_name("hand_gripper_tool_frame")
    root_link = hsr_world_state_reset.get_body_by_name("map")

    hole_point = Point3(x=0.5, z=0.3, reference_frame=root_link)

    msc = MotionStatechart()
    msc.add_node(
        motion := Sequence(
            [
                SetSeedConfiguration(
                    seed_configuration=JointState.from_str_dict(
                        goal_state, world=hsr_world_state_reset
                    )
                ),
                Parallel(
                    [
                        wiggle := WiggleInsert(
                            name="wiggle",
                            root_link=root_link,
                            tip_link=hpl,
                            hole_point=hole_point,
                            weight=DefaultWeights.WEIGHT_BELOW_CA,
                        ),
                        barrier := Parallel(
                            [
                                CountSimulationTimeSeconds(seconds=5),
                                CartesianPosition(
                                    root_link=root_link,
                                    tip_link=hpl,
                                    goal_point=Point3(
                                        x=0.5, z=0.4, reference_frame=root_link
                                    ),
                                    weight=DefaultWeights.WEIGHT_ABOVE_CA,
                                ),
                            ],
                        ),
                    ]
                ),
            ]
        )
    )
    barrier.end_condition = barrier.observation_variable
    msc.add_node(EndMotion.when_true(motion))

    kin_sim = Executor(
        MotionStatechartContext(world=hsr_world_state_reset),
        pacer=SimulationPacer(real_time_factor=1),
    )
    kin_sim.compile(motion_statechart=msc)
    kin_sim.tick_until_end()

    assert motion.observation_state == ObservationStateValues.TRUE
    assert not np.allclose(wiggle._current_vector, np.zeros(3))
