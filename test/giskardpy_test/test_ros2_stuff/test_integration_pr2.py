from __future__ import division

import asyncio
from dataclasses import dataclass, field
from time import sleep

import numpy as np
import pytest

from giskardpy.middleware.ros2.robot_interface_config import (
    StandAloneRobotInterfaceConfig,
)
from giskardpy.middleware.ros2.server_config import GiskardServerConfig
from giskardpy.middleware.ros2.scripts.iai_robots.pr2.configs import (
    WorldWithPR2Config,
)
from giskardpy.middleware.ros2.giskard import Giskard
from semantic_digital_twin.adapters.ros.rospy import wait_for_future_to_complete
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import (
    GiskardTester,
    compare_points,
)
from cramph.exceptions import EmptyStatechartError
from giskardpy.motion_statechart.graph_node import EndMotion
from cramph.monitors import CountSeconds
from cramph.statechart import Statechart
from giskardpy.motion_statechart.tasks.cartesian_tasks import (
    CartesianPose,
)
from giskardpy.motion_statechart.tasks.joint_tasks import (
    JointPositionList,
    JointState,
)
from giskardpy.qp.qp_controller_config import QPControllerConfig
from giskardpy.middleware.ros2.exceptions import (
    ClientDisconnectedError,
    ExecutionCanceledException,
    WorldModelModifiedDuringMotionError,
)
from krrood.symbolic_math import symbolic_math as sm
from semantic_digital_twin.robots.pr2 import PR2Joint
from semantic_digital_twin.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
)
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

from .test_client_presence import wait_until


@dataclass
class PR2Tester(GiskardTester):
    better_pose_left = {
        PR2Joint.LEFT_SHOULDER_PAN: 1.9652,
        PR2Joint.LEFT_SHOULDER_LIFT: -0.26499,
        PR2Joint.LEFT_UPPER_ARM_ROLL: 1.3837,
        PR2Joint.LEFT_ELBOW_FLEX: -2.12,
        PR2Joint.LEFT_FOREARM_ROLL: 16.99,
        PR2Joint.LEFT_WRIST_FLEX: -0.10001,
        PR2Joint.LEFT_WRIST_ROLL: 0,
    }

    right_tip: KinematicStructureEntity = field(init=False)
    base_footprint: KinematicStructureEntity = field(init=False)
    map: KinematicStructureEntity = field(init=False)

    def __post_init__(self):
        super().__post_init__()
        self.right_tip = self.api.world.get_kinematic_structure_entity_by_name(
            "r_gripper_tool_frame"
        )
        self.base_footprint = self.api.world.get_kinematic_structure_entity_by_name(
            "base_footprint"
        )
        self.map = self.api.world.get_kinematic_structure_entity_by_name("map")

    def setup_giskard(self) -> Giskard:
        robot_desc = load_xacro(
            "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
        )
        return Giskard(
            world_config=WorldWithPR2Config(urdf=robot_desc),
            robot_interface_config=StandAloneRobotInterfaceConfig(),
            server_config=GiskardServerConfig(
                debug_mode=True,
                plot_trajectory=True,
            ),
            qp_controller_config=QPControllerConfig(
                target_frequency=20,
            ),
        )


@pytest.fixture()
def robot():
    c = PR2Tester()
    try:
        yield c
    finally:
        print("tear down")
        c.close()


@pytest.fixture()
def default_joint_state(default_pr2_pose: dict[str, float]) -> dict[str, float]:
    return default_pr2_pose


@pytest.fixture()
def pocky_pose_setup(giskard_factory, pocky_pr2_pose: dict[str, float]) -> PR2Tester:
    return giskard_factory(pocky_pr2_pose)


# %% motions through the action server


class TestJointGoals:
    def test_joint_goal(self, giskard: PR2Tester):
        js = {
            PR2Joint.TORSO_LIFT: 0.2999225173357618,
            PR2Joint.HEAD_PAN: 0.041880780651479044,
            PR2Joint.HEAD_TILT: -0.37,
            PR2Joint.RIGHT_UPPER_ARM_ROLL: -0.9487714747527726,
            PR2Joint.RIGHT_SHOULDER_PAN: -1.0047307505973626,
            PR2Joint.RIGHT_SHOULDER_LIFT: 0.48736790658811985,
            PR2Joint.RIGHT_FOREARM_ROLL: -14.895833882874182,
            PR2Joint.RIGHT_ELBOW_FLEX: -1.392377908925028,
            PR2Joint.RIGHT_WRIST_FLEX: -0.4548695149411013,
            PR2Joint.RIGHT_WRIST_ROLL: 0.11426798984097819,
            PR2Joint.LEFT_UPPER_ARM_ROLL: 1.7383062350263658,
            PR2Joint.LEFT_SHOULDER_PAN: 1.8799810286792007,
            PR2Joint.LEFT_SHOULDER_LIFT: 0.011627231224188975,
            PR2Joint.LEFT_FOREARM_ROLL: 312.67276414458695,
            PR2Joint.LEFT_ELBOW_FLEX: -2.0300928925694675,
            PR2Joint.LEFT_WRIST_FLEX: -0.1,
            PR2Joint.LEFT_WRIST_ROLL: -6.062015047706399,
        }
        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(js, giskard.api.world),
            )
        )
        statechart.add_node(EndMotion.when_true(joint_goal))
        giskard.api.compile(statechart)
        giskard.api.execute()
        sleep(0.1)
        for joint, goal in js.items():
            connection = giskard.api.world.get_connection_by_name(joint)
            actual = giskard.api.world.state[connection.dof.id].position
            if (
                isinstance(connection, RevoluteConnection)
                and not connection.dof.has_position_limits()
            ):
                assert sm.shortest_angular_distance(actual, goal).to_np()[0] < 0.01
            else:
                assert np.isclose(
                    actual,
                    goal,
                    atol=1e-2,
                ), f"expected {goal} got {actual} for {connection.name}"


class TestCartGoals:
    def test_cart_goal_1eef(self, giskard: PR2Tester):
        tip_goal = HomogeneousTransformationMatrix.from_xyz_quaternion(
            pos_x=-0.2, reference_frame=giskard.right_tip
        )

        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            cart_goal := CartesianPose(
                root_link=giskard.base_footprint,
                tip_link=giskard.right_tip,
                goal_pose=tip_goal,
            )
        )
        statechart.add_node(EndMotion.when_true(cart_goal))
        giskard.api.compile(statechart)
        giskard.api.execute()


# %% world synchronization


def test_bodies_added_relative_to_the_gripper_are_placed_there_on_the_server(
    pocky_pose_setup: PR2Tester,
) -> None:
    """
    Bodies placed relative to the gripper end up at the same place relative to it in the
    server's world, which needs that world to be in step with the robot's state.
    """
    r_tip = pocky_pose_setup.right_tip
    poses_relative_to_r_tip = {
        "box": HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.08, reference_frame=r_tip
        ),
        "b1": HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.2, reference_frame=r_tip
        ),
        "bl": HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.15, y=0.04, reference_frame=r_tip
        ),
        "br": HomogeneousTransformationMatrix.from_xyz_rpy(
            x=0.15, y=-0.04, reference_frame=r_tip
        ),
    }
    pocky_pose_setup.add_box_to_world(
        name="box",
        size=(0.2, 0.05, 0.05),
        parent_link=r_tip,
        pose=poses_relative_to_r_tip["box"],
    )
    pocky_pose_setup.add_box_to_world(
        "b1", (0.01, 0.2, 0.2), pose=poses_relative_to_r_tip["b1"]
    )
    pocky_pose_setup.add_box_to_world(
        "bl", (0.1, 0.01, 0.2), pose=poses_relative_to_r_tip["bl"]
    )
    pocky_pose_setup.add_box_to_world(
        "br", (0.1, 0.01, 0.2), pose=poses_relative_to_r_tip["br"]
    )

    server_world = pocky_pose_setup.world
    server_r_tip = server_world.get_kinematic_structure_entity_by_name(
        "r_gripper_tool_frame"
    )
    for name, pose in poses_relative_to_r_tip.items():
        body = server_world.get_kinematic_structure_entity_by_name(name)
        compare_points(
            actual_point=server_world.compute_forward_kinematics(
                server_r_tip, body
            ).position,
            desired_point=pose.position,
            decimal=4,
        )


# %% action server events


class TestActionServerEvents:
    def test_wrong_params1(self, giskard: PR2Tester):
        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            joint_goal := JointPositionList(
                goal_state=HomogeneousTransformationMatrix(),
            )
        )
        statechart.add_node(EndMotion.when_true(joint_goal))
        with pytest.raises(AttributeError):
            giskard.api.compile(statechart)
            giskard.api.execute()

        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            joint_goal := JointPositionList(
                goal_state=JointState.from_str_dict(
                    giskard.better_pose_left, world=giskard.api.world
                ),
            )
        )
        statechart.add_node(EndMotion.when_true(joint_goal))
        giskard.api.compile(statechart)
        giskard.api.execute()

    @pytest.mark.asyncio
    async def test_cancel_with_new_goal(self, giskard: PR2Tester):
        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=100, reference_frame=giskard.base_footprint
                ),
            )
        )

        giskard.api.compile(statechart)
        goal_accepted_future = giskard.api.execute_async()
        wait_for_future_to_complete(goal_accepted_future)

        await asyncio.sleep(2)

        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            cart_goal := CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    y=1, reference_frame=giskard.base_footprint
                ),
            )
        )
        statechart.add_node(EndMotion.when_true(cart_goal))

        giskard.api.compile(statechart)
        goal_accepted_future = giskard.api.execute_async()
        wait_for_future_to_complete(goal_accepted_future)

        await giskard.api.get_result()

    @pytest.mark.asyncio
    async def test_interrupt(self, giskard: PR2Tester):
        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5, reference_frame=giskard.base_footprint
                ),
            )
        )

        giskard.api.compile(statechart)
        goal_accepted_future = giskard.api.execute_async()
        wait_for_future_to_complete(goal_accepted_future)

        await asyncio.sleep(2)
        wait_for_future_to_complete(giskard.api.cancel_goal_async())

        with pytest.raises(ExecutionCanceledException):
            await giskard.api.get_result()

    @pytest.mark.asyncio
    async def test_a_goal_whose_client_stops_announcing_itself_is_aborted(
        self, giskard: PR2Tester
    ):
        """
        Nobody is waiting for a motion whose client died, so it is stopped instead of
        being driven to its end.

        The motion ends by itself after a while, so that a client that is not noticed
        leaving fails this test instead of keeping the goal running forever.
        """
        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5, reference_frame=giskard.base_footprint
                ),
            )
        )
        statechart.add_node(give_up := CountSeconds(seconds=30))
        statechart.add_node(EndMotion.when_true(give_up))
        giskard.api.compile(statechart)
        giskard.api.announce_presence()
        goal_accepted_future = giskard.api.execute_async()
        wait_for_future_to_complete(goal_accepted_future)
        client_watchdog = giskard.giskard.motion_server.client_watchdog
        assert wait_until(
            lambda: client_watchdog.presence.watched_client == giskard.api.client,
            timeout=30.0,
        )

        giskard.api.heartbeat_publisher.stop()

        with pytest.raises(ClientDisconnectedError) as disconnect:
            await giskard.api.get_result()
        assert disconnect.value.client == giskard.api.client

    def test_empty_goal(self, giskard: PR2Tester):
        giskard.api.compile(Statechart(context=giskard.api.context))
        with pytest.raises(EmptyStatechartError):
            giskard.api.execute()

    @pytest.mark.asyncio
    async def test_world_model_modification_terminates_the_motion(
        self, giskard: PR2Tester
    ):
        """
        The running motion was compiled against the structure of the world, so a body
        added by another process ends it instead of being applied underneath it.
        """
        statechart = Statechart(context=giskard.api.context)
        statechart.add_node(
            CartesianPose(
                root_link=giskard.map,
                tip_link=giskard.base_footprint,
                goal_pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                    x=0.5, reference_frame=giskard.base_footprint
                ),
            )
        )

        giskard.api.compile(statechart)
        goal_accepted_future = giskard.api.execute_async()
        wait_for_future_to_complete(goal_accepted_future)

        await asyncio.sleep(1)

        giskard.add_box_to_world(
            name="box",
            size=(0.05, 0.01, 0.15),
            pose=HomogeneousTransformationMatrix.from_xyz_rpy(
                z=0.06, reference_frame=giskard.right_tip
            ),
            parent_link=giskard.right_tip,
        )

        with pytest.raises(WorldModelModifiedDuringMotionError):
            await giskard.api.get_result()

        await asyncio.sleep(1)

        # the modification is applied once the motion is over
        assert giskard.api.world.get_kinematic_structure_entity_by_name("box")
        assert giskard.giskard.executor.context.world.get_kinematic_structure_entity_by_name(
            "box"
        )
