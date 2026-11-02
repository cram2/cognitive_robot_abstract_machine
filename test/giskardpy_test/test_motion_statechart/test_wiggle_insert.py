import random

import numpy as np

from cramph.executor import SimulationPacer
from cramph.data_types import ObservationStateValues
from giskardpy.motion_statechart.weights import DefaultWeights
from cramph.composites import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from cramph.monitors import CountSimulationTimeSeconds
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPosition
from giskardpy.motion_statechart.tasks.wiggle_insert import WiggleInsert
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.hsrb import HSRBJoint
from semantic_digital_twin.spatial_types import Point3, Vector3
from semantic_digital_twin.world import World
from test.semantic_digital_twin_test.test_orm.test_orm import hsr_world_state_reset
from giskardpy.motion_control import MotionControl
from cramph.context import StatechartContext
from cramph.executor import StatechartExecutor
from ..motion_control_context import add_to_statechart_with_motion_control


def _hole_at_current_tip(world: World, tip, root) -> Point3:
    root_P_tip = world.transform(
        target_frame=root, spatial_object=Point3(0, 0, 0, reference_frame=tip)
    ).to_np()[:3]
    return Point3(*root_P_tip, reference_frame=root)


def test_wiggle_insert_reaches_hole(pr2_world_state_reset: World, rclpy_node):
    VizMarkerPublisher(_world=pr2_world_state_reset, node=rclpy_node)
    random.seed(0)
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    hole_point = _hole_at_current_tip(pr2_world_state_reset, tip, root)

    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
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
    statechart.add_node(wiggle)
    statechart.add_node(EndMotion.when_true(wiggle))

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert wiggle.observation_state == ObservationStateValues.TRUE


def test_wiggle_insert_basis_uses_root_frame_hole_normal(pr2_world_state_reset: World):
    """
    The wiggle basis vectors must be perpendicular to hole_normal as expressed in
    root_link, since the wiggle noise they span is added to a root_link-frame point
    (root_P_hole).
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
    add_to_statechart_with_motion_control(wiggle, pr2_world_state_reset)
    wiggle.build()

    expected_normal_in_root = pr2_world_state_reset.transform(
        target_frame=root, spatial_object=hole_normal_in_tip_frame
    ).to_np()[:3]

    assert np.isclose(
        np.dot(wiggle._perpendicular_basis_first, expected_normal_in_root),
        0.0,
        atol=1e-9,
    )
    assert np.isclose(
        np.dot(wiggle._perpendicular_basis_second, expected_normal_in_root),
        0.0,
        atol=1e-9,
    )


def test_wiggle_insert_on_tick_updates_noise(pr2_world_state_reset: World):
    random.seed(1)
    tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    root = pr2_world_state_reset.get_kinematic_structure_entity_by_name("odom_combined")
    hole_point = _hole_at_current_tip(pr2_world_state_reset, tip, root)

    motion_control = MotionControl()
    executor = StatechartExecutor(
        context=StatechartContext(world=pr2_world_state_reset),
        extensions=[motion_control],
    )
    statechart = Statechart(context=executor.context)
    wiggle = WiggleInsert(
        root_link=root,
        tip_link=tip,
        hole_point=hole_point,
        hole_normal=Vector3.Z(reference_frame=root),
        noise_translation=0.05,
        noise_angle=0.2,
        down_velocity=0.05,
    )
    statechart.add_node(wiggle)
    statechart.add_node(EndMotion.when_true(wiggle))

    executor.compile(statechart=statechart)

    executor.tick()
    first_translation = wiggle._random_translation.evaluate().flatten()[:3].copy()
    executor.tick()
    second_translation = wiggle._random_translation.evaluate().flatten()[:3].copy()

    # The injected noise must change between control cycles.
    assert not np.allclose(first_translation, second_translation)


def test_wiggle_insert(hsr_world_state_reset):
    goal_state = {
        HSRBJoint.ARM_FLEX: -1.5,
        HSRBJoint.ARM_LIFT: 0.5,
        HSRBJoint.ARM_ROLL: 0.0,
        HSRBJoint.HEAD_PAN: 0.0,
        HSRBJoint.HEAD_TILT: 0.0,
        HSRBJoint.WRIST_FLEX: -1.5,
        HSRBJoint.WRIST_ROLL: 0.0,
    }

    hpl = hsr_world_state_reset.get_body_by_name("hand_gripper_tool_frame")
    root_link = hsr_world_state_reset.get_body_by_name("map")

    hole_point = Point3(x=0.5, z=0.3, reference_frame=root_link)

    executor = StatechartExecutor(
        context=StatechartContext(world=hsr_world_state_reset),
        pacer=SimulationPacer(real_time_factor=1),
        extensions=[MotionControl()],
    )
    statechart = Statechart(context=executor.context)
    statechart.add_node(
        motion := Sequence(
            nodes=[
                SetSeedConfiguration(
                    seed_configuration=JointState.from_str_dict(
                        goal_state, world=hsr_world_state_reset
                    )
                ),
                Parallel(
                    nodes=[
                        wiggle := WiggleInsert(
                            name="wiggle",
                            root_link=root_link,
                            tip_link=hpl,
                            hole_point=hole_point,
                            weight=DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE,
                        ),
                        barrier := Parallel(
                            nodes=[
                                CountSimulationTimeSeconds(seconds=5),
                                CartesianPosition(
                                    root_link=root_link,
                                    tip_link=hpl,
                                    goal_point=Point3(
                                        x=0.5, z=0.4, reference_frame=root_link
                                    ),
                                    weight=DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE,
                                ),
                            ],
                        ),
                    ]
                ),
            ]
        )
    )
    barrier.success_condition = barrier.observes_true
    statechart.add_node(EndMotion.when_true(motion))

    executor.compile(statechart=statechart)
    executor.tick_until_end()

    assert motion.last_observation_state == ObservationStateValues.TRUE
    assert not np.allclose(wiggle._current_vector, np.zeros(3))
