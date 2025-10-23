from __future__ import division

from dataclasses import field
from typing import Dict, Optional, Union, Type, Tuple

import semantic_digital_twin.spatial_types.spatial_types as cas
from giskardpy.data_types.exceptions import GoalInitalizationException
from giskardpy.god_map import god_map
from giskardpy.motion_statechart.data_types import ObservationState
from giskardpy.motion_statechart.monitors.monitors import PayloadMonitor
from giskardpy.utils.decorators import validated_dataclass
from giskardpy.utils.math import axis_angle_from_quaternion
from semantic_digital_twin.world_description.connections import OmniDrive, ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import Connection


@validated_dataclass
class SetSeedConfiguration(PayloadMonitor):
    """
    Overwrite the configuration of the world to allow starting the planning from a different state.
    CAUTION! don't use this to overwrite the robot's state outside standalone mode!
    :param seed_configuration: maps joint name to float
    :param group_name: if joint names are not unique, it will search in this group for matches.
    """

    seed_configuration: Dict[ActiveConnection1DOF, Union[int, float]]

    def __post_init__(self):
        self.seed_configuration = {
            connection.dof.name: v for connection, v in self.seed_configuration.items()
        }
        if self.name is None:
            self.name = (
                f"{str(self.__class__.__name__)}/{list(self.seed_configuration.keys())}"
            )

    def __call__(self):
        for dof_name, initial_joint_value in self.seed_configuration.items():
            god_map.world.state[dof_name].position = initial_joint_value
        god_map.world.notify_state_change()
        self.state = ObservationState.true


@validated_dataclass
class SetOdometry(PayloadMonitor):
    base_pose: cas.TransformationMatrix
    _odom_joints: Tuple[Type[Connection], ...] = field(default=(OmniDrive,), init=False)
    odom_connection: Optional[OmniDrive] = None

    def __post_init__(self):
        if self.name is None:
            self.name = f"{self.__class__.__name__}/{self.odom_connection}"
        if self.odom_connection is None:
            drive_connections = god_map.world.get_connections_by_type(self._odom_joints)
            if len(drive_connections) == 0:
                raise GoalInitalizationException("No drive joints in world")
            elif len(drive_connections) == 1:
                self.odom_connection = drive_connections[0]
            else:
                raise GoalInitalizationException(
                    "Multiple drive joint found in world, please set 'group_name'"
                )

    def __call__(self):
        parent_T_pose_ref = cas.TransformationMatrix(
            god_map.world.compute_forward_kinematics_np(
                self.odom_connection.parent, self.base_pose.reference_frame
            )
        )
        parent_T_pose = parent_T_pose_ref @ self.base_pose
        position = parent_T_pose.to_position().to_np()
        orientation = parent_T_pose.to_rotation_matrix().to_quaternion().to_np()
        god_map.world.state[self.odom_connection.x.name].position = position[0]
        god_map.world.state[self.odom_connection.y.name].position = position[1]
        axis, angle = axis_angle_from_quaternion(
            orientation[0], orientation[1], orientation[2], orientation[3]
        )
        if axis[-1] < 0:
            angle = -angle
        # if isinstance(self.brumbrum_joint, OmniDrivePR22):
        #     god_map.world.state[self.brumbrum_joint.yaw1_vel.name].position = 0
        #     god_map.world.state[self.brumbrum_joint.yaw.name].position = angle
        # else:
        god_map.world.state[self.odom_connection.yaw.name].position = angle
        god_map.world.notify_state_change()
        self.state = ObservationState.true
