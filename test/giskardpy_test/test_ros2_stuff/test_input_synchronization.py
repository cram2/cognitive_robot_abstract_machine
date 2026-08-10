from dataclasses import dataclass

import pytest
from geometry_msgs.msg import PoseStamped
from numpy.testing import assert_allclose

from giskardpy.middleware.ros2.exceptions import (
    AlreadyTrackedByTfFrameError,
    ConnectionCannotBeTrackedByTfFrameError,
)
from giskardpy.middleware.ros2.input_synchronization import TfFrameSynchronizer
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.connections import (
    Connection6DoF,
    FixedConnection,
)

# %% test doubles


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
def tracked_connection(world_with_two_bodies):
    """
    A world whose two bodies are joined by a tracked 6 degree of freedom connection.
    """
    world, parent, child = world_with_two_bodies
    with world.modify_world():
        connection = Connection6DoF.create_with_dofs(world, parent, child)
        world.add_connection(connection)
    return world, connection


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
