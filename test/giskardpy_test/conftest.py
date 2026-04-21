import numpy as np
import pytest

from giskardpy.middleware.ros2 import rospy
from giskardpy.middleware.ros2.utils.utils import load_xacro
from giskardpy.middleware.ros2.utils.utils_for_tests import GiskardTester
from giskardpy.motion_statechart.graph_node import EndMotion
from giskardpy.motion_statechart.monitors.overwrite_state_monitors import (
    SetSeedConfiguration,
    SetOdometry,
)
from giskardpy.motion_statechart.motion_statechart import MotionStatechart
from krrood.symbolic_math.symbolic_math import trinary_logic_and
from semantic_digital_twin.datastructures.joint_state import JointState
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import Vector3, HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    RevoluteConnection,
    FixedConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import (
    Body,
)


@pytest.fixture()
def better_pr2_pose():
    return {
        "r_shoulder_pan_joint": -1.7125,
        "r_shoulder_lift_joint": -0.25672,
        "r_upper_arm_roll_joint": -1.46335,
        "r_elbow_flex_joint": -2.12,
        "r_forearm_roll_joint": 1.76632,
        "r_wrist_flex_joint": -0.10001,
        "r_wrist_roll_joint": 0.05106,
        "l_shoulder_pan_joint": 1.9652,
        "l_shoulder_lift_joint": -0.26499,
        "l_upper_arm_roll_joint": 1.3837,
        "l_elbow_flex_joint": -2.12,
        "l_forearm_roll_joint": 16.99,
        "l_wrist_flex_joint": -0.10001,
        "l_wrist_roll_joint": 0,
        "torso_lift_joint": 0.2,
        "l_gripper_l_finger_joint": 0.55,
        "r_gripper_l_finger_joint": 0.55,
        "head_pan_joint": 0,
        "head_tilt_joint": 0,
    }


@pytest.fixture(scope="function")
def pr2_with_box(pr2_world_copy) -> World:
    with pr2_world_copy.modify_world():
        box = Body(
            name=PrefixedName("box"),
            visual=ShapeCollection(shapes=[Box(scale=Scale(1, 1, 1))]),
            collision=ShapeCollection(shapes=[Box(scale=Scale(1, 1, 1))]),
        )
        root_C_box = FixedConnection(
            parent=pr2_world_copy.root,
            child=box,
            parent_T_connection_expression=HomogeneousTransformationMatrix.from_xyz_rpy(
                x=1.2, z=0.3, reference_frame=pr2_world_copy.root
            ),
        )
        pr2_world_copy.add_connection(root_C_box)
    return pr2_world_copy


@pytest.fixture()
def mini_world():
    world = World()
    with world.modify_world():
        body = Body(name=PrefixedName("root"))
        body2 = Body(name=PrefixedName("tip"))
        connection = RevoluteConnection.create_with_dofs(
            world=world, parent=body, child=body2, axis=Vector3.Z()
        )
        world.add_connection(connection)
    return world


@pytest.fixture(scope="function")
def init_rospy():

    rospy.init_node("giskard")
    rospy.node.get_logger().info("init ros")

    try:
        yield None
    finally:
        print("kill ros")
        # Cleanly reset TF and shutdown ROS2 node/executor
        rospy.shutdown()


@pytest.fixture()
def giskard_factory(init_rospy, robot: GiskardTester):
    def _create_giskard(seed_joint_state: dict[str, float]) -> GiskardTester:
        parse_seed_joint_state = {
            robot.api.world.get_connection_by_name(name): target
            for name, target in seed_joint_state.items()
        }
        msc = MotionStatechart()

        initial_config = SetSeedConfiguration(
            name="initial configuration",
            seed_configuration=JointState.from_mapping(parse_seed_joint_state),
        )
        msc.add_node(initial_config)

        if robot.has_odometry_joint():
            base_goal = HomogeneousTransformationMatrix(
                reference_frame=robot.api.world.root
            )
            base_pose_reached = SetOdometry(name="initial pose", base_pose=base_goal)
            msc.add_node(base_pose_reached)
            done = trinary_logic_and(
                initial_config.observation_variable,
                base_pose_reached.observation_variable,
            )
        else:
            done = initial_config.observation_variable
        end = EndMotion(name="end")
        msc.add_node(end)
        end.start_condition = done
        robot.api.execute(msc)
        return robot

    return _create_giskard


@pytest.fixture()
def giskard(giskard_factory, default_joint_state):
    return giskard_factory(default_joint_state)


@pytest.fixture()
def giskard_better_pose(giskard_factory, better_pose):
    return giskard_factory(better_pose)


@pytest.fixture()
def kitchen_setup(giskard_better_pose: GiskardTester) -> GiskardTester:
    giskard_better_pose.default_env_name = "iai_kitchen"
    kitchen_urdf = load_xacro(
        "package://iai_kitchen/urdf_obj/iai_kitchen_python.urdf.xacro"
    )
    giskard_better_pose.add_urdf_to_world(
        name=giskard_better_pose.default_env_name,
        urdf=kitchen_urdf,
        pose=HomogeneousTransformationMatrix(
            reference_frame=giskard_better_pose.api.world.root
        ),
    )
    return giskard_better_pose


@pytest.fixture()
def dlr_kitchen_setup(better_pose: GiskardTester) -> GiskardTester:
    raise NotImplementedError("DLR kitchen setup is not yet implemented")


@pytest.fixture()
def apartment_setup(giskard_better_pose: GiskardTester) -> GiskardTester:
    giskard_better_pose.default_env_name = "iai_apartment"
    kitchen_urdf = load_xacro("package://iai_apartment/urdf/apartment.urdf")
    giskard_better_pose.add_urdf_to_world(
        name=giskard_better_pose.default_env_name,
        urdf=kitchen_urdf,
        pose=HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.5,
            y=1.4,
            yaw=np.pi,
            reference_frame=giskard_better_pose.api.world.root,
        ),
    )
    return giskard_better_pose
