import json
import time

from giskardpy.executor import Executor
from giskardpy.motion_statechart.context import MotionStatechartContext
from giskardpy.motion_statechart.data_types import DefaultWeights
from giskardpy.motion_statechart.goals.collision_avoidance import SelfCollisionAvoidance
from giskardpy.motion_statechart.goals.templates import Sequence, Parallel
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.monitors import LocalMinimumReached
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from semantic_digital_twin.adapters.ros.world_fetcher import (
    FetchWorldServer,
    fetch_world_from_service,
)
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.robots.pr2 import PR2
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World


def to_and_from_json(motion_statechart: MotionStatechart, target_world: World):
    json_data = motion_statechart.to_json()
    json_str = json.dumps(json_data)
    new_json_data = json.loads(json_str)

    tracker = WorldEntityWithIDKwargsTracker.from_world(target_world)
    kwargs = tracker.create_kwargs()
    return MotionStatechart.from_json(new_json_data, **kwargs)


def test_execute_collision_goal_in_fetched_world(rclpy_node, pr2_world_state_reset):
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(PR2)[0]
    fetcher = FetchWorldServer(node=rclpy_node, world=pr2_world_state_reset)

    pr2_world_copy = fetch_world_from_service(
        rclpy_node,
    )

    time.sleep(2)

    fetched_pr2 = pr2_world_copy.get_semantic_annotations_by_type(PR2)[0]

    r_tip = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    base_footprint = pr2_world_state_reset.get_kinematic_structure_entity_by_name(
        "base_footprint"
    )

    msc = MotionStatechart()
    msc.add_node(
        Sequence(
            [
                SetSeedConfiguration(
                    seed_configuration=JointState.from_str_dict(
                        {
                            "r_elbow_flex_joint": -1.43286344265,
                            "r_forearm_roll_joint": -1.26465060073,
                            "r_shoulder_lift_joint": 0.47990329056,
                            "r_shoulder_pan_joint": -0.281272240139,
                            "r_upper_arm_roll_joint": -0.528415402668,
                            "r_wrist_flex_joint": -1.18811419869,
                            "r_wrist_roll_joint": 2.26884630124,
                        },
                        world=pr2_world_state_reset,
                    )
                ),
                Parallel(
                    [
                        CartesianPose(
                            root_link=base_footprint,
                            tip_link=r_tip,
                            goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                                0.2, reference_frame=r_tip
                            ),
                            weight=DefaultWeights.WEIGHT_ABOVE_CA,
                        ),
                        SelfCollisionAvoidance(robot=pr2),
                    ]
                ),
            ]
        )
    )
    msc.add_node(local_min := LocalMinimumReached())
    msc.add_node(EndMotion.when_true(local_min))

    msc_copy = to_and_from_json(msc, pr2_world_copy)

    kin_sim = Executor(MotionStatechartContext(world=pr2_world_copy))
    kin_sim.compile(motion_statechart=msc_copy)

    kin_sim.tick_until_end(500)
