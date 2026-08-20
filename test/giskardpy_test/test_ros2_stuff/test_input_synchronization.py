"""
Tests for the synchronizers that write ROS topics and tf frames into the world state.
"""

from __future__ import annotations

from dataclasses import dataclass, fields
from typing import Any, Optional

import pytest
from geometry_msgs.msg import PoseStamped
from nav_msgs.msg import Odometry
from numpy.testing import assert_allclose
from sensor_msgs.msg import JointState

from giskardpy.middleware.ros2.exceptions import (
    AlreadyTrackedByTfFrameError,
    ConnectionCannotBeTrackedByTfFrameError,
    UnboundMessageTypeError,
)
from giskardpy.middleware.ros2.input_synchronization import (
    LatestJointStateSynchronizer,
    OdometrySynchronizer,
    PendingJointStateSynchronizer,
    TfFrameSynchronizer,
    TopicInputSynchronizer,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
    OmniDrive,
)
from semantic_digital_twin.world_description.world_entity import Body

# %% mimics and helpers


@dataclass
class UnparameterizedTopicSynchronizer(TopicInputSynchronizer):
    """
    A topic synchronizer that never bound the type of its messages.
    """

    def apply_message(self, message: Any) -> None:
        pass


@dataclass
class RecordedTransformLookup:
    """
    Answers every lookup with one recorded transform, standing in for tf.
    """

    parent_T_child: PoseStamped
    """
    The transform handed back to the caller.
    """

    def lookup_pose(self, target_frame: str, source_frame: str) -> PoseStamped:
        return self.parent_T_child


def latest_message_field_type(synchronizer_type: type) -> Any:
    """
    The declared type of the buffered message of a synchronizer class.
    """
    [message_field] = [
        field for field in fields(synchronizer_type) if field.name == "latest_message"
    ]
    return message_field.type


def joint_state_message(joint_name: str, position: float) -> JointState:
    """
    A joint state message that reports one position for one joint.
    """
    message = JointState()
    message.name = [joint_name]
    message.position = [position]
    return message


def odometry_message(pose: HomogeneousTransformationMatrix) -> Odometry:
    """
    An odometry message that reports the given pose.
    """
    quaternion = pose.to_rotation_matrix().to_quaternion().to_np()
    position = pose.to_position().to_np()
    message = Odometry()
    message.pose.pose.position.x = float(position[0])
    message.pose.pose.position.y = float(position[1])
    message.pose.pose.position.z = float(position[2])
    message.pose.pose.orientation.x = float(quaternion[0])
    message.pose.pose.orientation.y = float(quaternion[1])
    message.pose.pose.orientation.z = float(quaternion[2])
    message.pose.pose.orientation.w = float(quaternion[3])
    return message


def make_pose_stamped(
    position_x: float, position_y: float, position_z: float
) -> PoseStamped:
    """
    A translation-only pose message at the given coordinates.
    """
    pose_stamped = PoseStamped()
    pose_stamped.pose.position.x = position_x
    pose_stamped.pose.position.y = position_y
    pose_stamped.pose.position.z = position_z
    pose_stamped.pose.orientation.w = 1.0
    return pose_stamped


@pytest.fixture()
def omni_drive_world() -> World:
    """
    A world whose root is connected to a base body by an omni drive.
    """
    world = World()
    with world.modify_world():
        root = Body(name=PrefixedName("root"))
        base = Body(name=PrefixedName("base"))
        world.add_connection(
            OmniDrive.create_with_dofs(world=world, parent=root, child=base)
        )
    return world


@pytest.fixture()
def tracked_connection(world_with_two_bodies):
    """
    A world whose two bodies are joined by a tracked 6 degree of freedom connection.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = Connection6DoF.create_with_dofs(world, parent, child)
        world.add_connection(connection)
    return world, connection


# %% message type resolution


def test_joint_state_synchronizers_read_joint_state_messages():
    assert PendingJointStateSynchronizer.message_type() is JointState
    assert LatestJointStateSynchronizer.message_type() is JointState


def test_odometry_synchronizer_reads_odometry_messages():
    assert OdometrySynchronizer.message_type() is Odometry


def test_synchronizer_without_bound_message_type_is_rejected():
    with pytest.raises(UnboundMessageTypeError):
        UnparameterizedTopicSynchronizer.message_type()
    with pytest.raises(UnboundMessageTypeError):
        TopicInputSynchronizer.message_type()


def test_joint_state_synchronizers_buffer_joint_state_messages():
    assert (
        latest_message_field_type(PendingJointStateSynchronizer) == Optional[JointState]
    )
    assert (
        latest_message_field_type(LatestJointStateSynchronizer) == Optional[JointState]
    )


def test_odometry_synchronizer_buffers_odometry_messages():
    assert latest_message_field_type(OdometrySynchronizer) == Optional[Odometry]


# %% writing joint states


def test_pending_joint_state_synchronizer_writes_a_message_once(
    init_rospy, mini_world: World
):
    [connection] = mini_world.connections
    synchronizer = PendingJointStateSynchronizer(
        world=mini_world, topic_name="joint_states"
    )
    synchronizer.latest_message = joint_state_message(connection.name.name, 0.42)

    assert synchronizer.apply() is True
    assert mini_world.state[connection.raw_dof.id].position == 0.42
    assert synchronizer.apply() is False


def test_latest_joint_state_synchronizer_rewrites_its_message_every_cycle(
    init_rospy, mini_world: World
):
    [connection] = mini_world.connections
    synchronizer = LatestJointStateSynchronizer(
        world=mini_world, topic_name="joint_states"
    )
    synchronizer.latest_message = joint_state_message(connection.name.name, 0.42)

    assert synchronizer.apply() is True
    mini_world.state[connection.raw_dof.id].position = 1.0
    assert synchronizer.apply() is True
    assert mini_world.state[connection.raw_dof.id].position == 0.42


def test_synchronizer_writes_nothing_without_a_message(init_rospy, mini_world: World):
    [connection] = mini_world.connections
    position_before_apply = mini_world.state[connection.raw_dof.id].position
    synchronizer = PendingJointStateSynchronizer(
        world=mini_world, topic_name="joint_states"
    )

    assert synchronizer.apply() is False
    assert mini_world.state[connection.raw_dof.id].position == position_before_apply


# %% writing the base pose


def test_odometry_synchronizer_writes_the_pose_into_the_drive(
    init_rospy, omni_drive_world: World
):
    connection = omni_drive_world.get_connection_by_name("root_T_base")
    expected_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
        x=1.5, y=-2.5, yaw=0.75
    )
    synchronizer = OdometrySynchronizer(
        world=omni_drive_world, topic_name="odom", connection=connection
    )
    synchronizer.latest_message = odometry_message(expected_pose)

    assert synchronizer.apply() is True
    assert_allclose(
        connection.origin.to_np().astype(float),
        expected_pose.to_np().astype(float),
        atol=1e-9,
    )


# %% writing tf into the world


def test_apply_writes_the_looked_up_transform_into_the_connection(
    init_rospy, tracked_connection
):
    """
    The origin the synchronizer assigns has to carry the frames of the connection it
    writes to, because the setter of a 6 degree of freedom connection converts the
    transform into the parent frame and rejects one without a reference frame.
    """
    world, connection = tracked_connection
    synchronizer = TfFrameSynchronizer(world=world)
    synchronizer.tf_wrapper = RecordedTransformLookup(
        parent_T_child=make_pose_stamped(1.0, -2.0, 0.5)
    )
    synchronizer.track(connection, tf_parent_frame="map", tf_child_frame="odom")

    wrote_something = synchronizer.apply()

    assert wrote_something
    assert_allclose(
        connection.origin,
        HomogeneousTransformationMatrix.from_xyz_rpy(
            x=1.0, y=-2.0, z=0.5, reference_frame=connection.parent
        ),
        atol=1e-9,
    )


def test_apply_writes_nothing_without_a_tracked_connection(
    init_rospy, world_with_two_bodies
):
    """
    A synchronizer that tracks nothing must report that it did not write, so the loop
    around it does not recompute the forward kinematics for no reason.
    """
    world, _, _ = world_with_two_bodies
    synchronizer = TfFrameSynchronizer(world=world)

    assert not synchronizer.apply()


# %% rejecting connections it cannot write


def test_tracking_a_connection_twice_is_rejected(init_rospy, tracked_connection):
    """
    A second pair of frames for the same connection would silently overwrite the first.
    """
    world, connection = tracked_connection
    synchronizer = TfFrameSynchronizer(world=world)
    synchronizer.track(connection, tf_parent_frame="map", tf_child_frame="odom")

    with pytest.raises(AlreadyTrackedByTfFrameError):
        synchronizer.track(
            connection, tf_parent_frame="map", tf_child_frame="base_link"
        )


def test_tracking_a_connection_without_six_degrees_of_freedom_is_rejected(
    init_rospy, world_with_two_bodies
):
    """
    Only a connection with all six degrees of freedom can follow an arbitrary transform.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = FixedConnection(parent=parent, child=child)
        world.add_connection(connection)
    synchronizer = TfFrameSynchronizer(world=world)

    with pytest.raises(ConnectionCannotBeTrackedByTfFrameError):
        synchronizer.track(connection, tf_parent_frame="map", tf_child_frame="odom")
