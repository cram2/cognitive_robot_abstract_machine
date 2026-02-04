from __future__ import annotations

import math
from copy import deepcopy
from dataclasses import dataclass
from typing import Tuple

import numpy as np
from scipy.spatial.transform import Rotation as R
from typing_extensions import Optional, Union, List

from semantic_digital_twin.robots.abstract_robot import Manipulator, AbstractRobot
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.world_description.world_entity import Body
from .dataclasses import Rotations
from .enums import Grasp, AxisIdentifier, ApproachDirection, VerticalAlignment
from .pose import PoseStamped, PyCramVector3
from ..has_parameters import HasParameters
from ..tf_transformations import quaternion_multiply
from ..utils import translate_pose_along_local_axis


@dataclass
class GraspDescription:
    """
    Describes a grasp configuration for a manipulator the description consists of the approach direction (the side from
    which to grasp e.g. FRONT, LEFT, etc and the vertical alignment (TOP, BOTTOM).
    """

    approach_direction: ApproachDirection
    """
    The direction from which the body should be grasped. These are the four directions in the x-y plane (FRONT, BACK, LEFT, RIGHT).
    """
    vertical_alignment: VerticalAlignment

    """
    The alignment of the gripper with the body in the z-axis (TOP, BOTTOM).
    """

    manipulator: Manipulator
    """
    The manipulator that is used to grasp the body.
    """

    rotate_gripper: bool = False
    """
    Rotate the gripper by 90 degrees.
    """

    manipulation_offset: float = 0.05
    """
    The offset between the center of the pose in the grasp sequence
    """

    def _pose_sequence(self, pose: PoseStamped, body: Body = None, reverse: bool = False) -> List[PoseStamped]:
        """
        Calculates the pose sequence to grasp something at the pose if the body is given its geometry is also taken into
        account. The pose sequence consists of 3 poses: one in front of the body (taking body geometry into account),
        one at the center of the body, and the last one above the body to lift it.

        :param pose: The pose around which the pose sequence should be centered.
        :param body: The body of the grasp.
        :param reverse: If the sequence should be reversed.
        :return: The pose sequence.
        """
        pose_frame = pose.frame_id

        world = pose_frame._world

        grasp_orientation = self.grasp_orientation()

        if body:
            bb_in_frame = body.collision.as_bounding_box_collection_in_frame(body).bounding_box()

            approach_axis = np.array(self.approach_direction.axis.value, dtype=np.bool)

            # Pre-pose calculation
            offset = (np.array(bb_in_frame.dimensions)[approach_axis] / 2 + self.manipulation_offset)[0]
        else:
            offset = 0

        pre_pose = PoseStamped.from_list(pose.position.to_list(), pose.orientation.to_list(), frame=pose_frame)
        pre_pose.rotate_by_quaternion(grasp_orientation)
        pre_pose = translate_pose_along_local_axis(pre_pose, self.manipulation_axis(), -offset)

        grasp_pose = deepcopy(pose)
        grasp_pose.rotate_by_quaternion(grasp_orientation)

        # Lift pose calculation
        lift_pose_map = PoseStamped.from_spatial_type(world.transform(pose.to_spatial_type(), world.root))
        lift_pose_map.position.z += self.manipulation_offset

        lift_pose =  PoseStamped.from_spatial_type(world.transform(lift_pose_map.to_spatial_type(), pose_frame))
        lift_pose.rotate_by_quaternion(grasp_orientation)

        sequence = [pre_pose, grasp_pose, lift_pose]

        if reverse:
            sequence.reverse()
        return sequence

    def grasp_pose_sequence(self, body: Body):
        """
        Calculates the pose sequence to grasp the body. The sequence is 3 poses, one in front of the body (taking body
        geometry into account), one at the center of the body, and the last one above the body to lift it.

        :param body: The body of the grasp.
        :return: The pose sequence.
        """
        return self._pose_sequence(PoseStamped.from_list(frame=body), body)

    def place_pose_sequence(self, pose: PoseStamped) -> List[PoseStamped]:
        """
        Calculates the pose sequence to place a body at the given pose. Assumes that the manipulator is holding a body
        which is being placed.

        :param pose: The pose at which the body in the manipulator should be placed
        :return: The pose sequence.
        """
        body =  self.manipulator.tool_frame.child_kinematic_structure_entities[0]
        return self._pose_sequence(pose, body, reverse=True)

    def manipulation_axis(self) -> List[float]:
        """
        Axis of the manipulator that is manipulating the body. Translates the x-axis of the global frame to how the
        manipulator is rotated.

        :returns: The axis of the manipulator that is manipulating the body.
        """
        return self.calculate_manipulator_axis(AxisIdentifier.X)


    def lift_axis(self) -> List[float]:
        """
        Axis of the manipulator that is lifting the body. Translates the z-axis of the global frame to how the
        manipulator is rotated.

        :returns: The axis of the manipulator that is lifting the body.
        """
        return self.calculate_manipulator_axis(AxisIdentifier.Z)


    def calculate_manipulator_axis(self, axis: AxisIdentifier) -> List[float]:
        """
        Calculates the corresponding axis of the manipulator for a given axis of the body.

        :param axis: The axis of the body as a list of [x, y, z] indices.
        :return: The corresponding axis of the manipulator as a list of [x, y, z] values.
        """
        axis_list = axis.value
        front_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            *axis_list, reference_frame=self.manipulator._world.root
        )

        grasp_pose = HomogeneousTransformationMatrix.from_xyz_quaternion(0, 0, 0, *self.manipulator.front_facing_orientation.to_np(), reference_frame=self.manipulator._world.root)
        world = self.manipulator._world

        front_global = world.transform(front_pose, world.root)

        grasp_global = world.transform(grasp_pose, world.root)

        t = grasp_global.inverse().to_np() @ front_global.to_np()

        return t[:3, 3].astype(int).tolist()

    def grasp_orientation(self):
        """
        The orientation of the grasp. Takes into account the approach direction and vertical
        alignment.
        """
        rotation = Rotations.SIDE_ROTATIONS[self.approach_direction]
        rotation = quaternion_multiply(
            rotation, Rotations.VERTICAL_ROTATIONS[self.vertical_alignment]
        )
        rotation = quaternion_multiply(
            rotation, Rotations.HORIZONTAL_ROTATIONS[self.rotate_gripper]
        )

        orientation = quaternion_multiply(rotation, self.manipulator.front_facing_orientation.to_np())

        norm = math.sqrt(sum(comp**2 for comp in orientation))
        orientation = [comp / norm for comp in orientation]

        return orientation

    def edge_offset(self, body: Body) -> float:
        """
        The offset between the center of the body and its edge in the direction of the approach axis.

        :param body: The body to calculate the edge offset for.
        :return: The edge offset.
        """
        rim_direction_index = self.approach_direction.value[0].value.index(1)

        rim_offset = (
                body.collision.as_bounding_box_collection_in_frame(body)
                .bounding_box()
                .dimensions[rim_direction_index]
                / 2
        )
        return rim_offset

    def grasp_pose(self, body: Body, grasp_edge: bool = False) -> PoseStamped:
        """
        The pose for the given manipulator to grasp the body in the frame of the body.

        :param body: The body to grasp.
        :param grasp_edge: Indicates if the pose should be for the edge of the body or the center.
        :return: The pose of the body in the body frame.
        """
        edge_offset = -self.edge_offset(body) if grasp_edge else 0
        orientation = self.grasp_orientation()
        grasp_pose = PoseStamped().from_list([edge_offset, 0, 0], orientation, frame=body)

        return grasp_pose

    @classmethod
    def calculate_grasp_descriptions(
            cls,
            manipulator: Manipulator,
            pose: PoseStamped,
            grasp_alignment: Optional[PreferredGraspAlignment] = None,
    ) -> List[GraspDescription]:
        """
        This method determines the possible grasp configurations (approach axis and vertical alignment) of the body,
        taking into account the bodies orientation, position, and whether the gripper should be rotated by 90°.

        :param manipulator: The manipulator to use.
        :param grasp_alignment: An optional PreferredGraspAlignment object that specifies preferred grasp axis,
        :param pose: The pose of the object to be grasped.

        :return: A sorted list of GraspDescription instances representing all grasp permutations.
        """
        world = manipulator._world
        objectTmap = PoseStamped.from_spatial_type(world.transform(pose.to_spatial_type(), world.root))

        robot_pose = PoseStamped.from_spatial_type(manipulator._robot.root.global_pose)

        if grasp_alignment:
            side_axis = grasp_alignment.preferred_axis
            vertical = grasp_alignment.with_vertical_alignment
            rotated_gripper = grasp_alignment.with_rotated_gripper
        else:
            side_axis, vertical, rotated_gripper = (
                AxisIdentifier.Undefined,
                False,
                False,
            )

        object_to_robot_vector_world = objectTmap.position.vector_to_position(
            robot_pose.position
        )
        orientation = objectTmap.orientation.to_list()

        mapRobject = R.from_quat(orientation).as_matrix()
        objectRmap = mapRobject.T

        object_to_robot_vector_local = objectRmap.dot(
            object_to_robot_vector_world.to_numpy()
        )
        vector_x, vector_y, vector_z = object_to_robot_vector_local

        vector_side = PyCramVector3(vector_x, vector_y, np.nan)
        side_faces = GraspDescription.calculate_closest_faces(vector_side, side_axis)

        vector_vertical = PyCramVector3(np.nan, np.nan, vector_z)
        if vertical:
            vertical_faces = GraspDescription.calculate_closest_faces(vector_vertical)
        else:
            vertical_faces = [VerticalAlignment.NoAlignment]

        grasp_configs = [
            GraspDescription(
                approach_direction=side,
                vertical_alignment=top_face,
                rotate_gripper=rotated_gripper,
                manipulator=manipulator,
            )
            for top_face in vertical_faces
            for side in side_faces
        ]

        return grasp_configs

    @staticmethod
    def calculate_closest_faces(
            pose_to_robot_vector: PyCramVector3,
            specified_grasp_axis: AxisIdentifier = AxisIdentifier.Undefined,
    ) -> Union[
        Tuple[ApproachDirection, ApproachDirection],
        Tuple[VerticalAlignment, VerticalAlignment],
    ]:
        """
        Determines the faces of the object based on the input vector.

        If `specified_grasp_axis` is None, it calculates the primary and secondary faces based on the vector's magnitude
        determining which sides of the object are most aligned with the robot. This will either be the x, y plane for side faces
        or the z axis for top/bottom faces.
        If `specified_grasp_axis` is provided, it only considers the specified axis and calculates the faces aligned
        with that axis.

        :param pose_to_robot_vector: A 3D vector representing one of the robot's axes in the pose's frame, with
                              irrelevant components set to np.nan.
        :param specified_grasp_axis: Specifies a specific axis (e.g., X, Y, Z) to focus on.

        :return: A tuple of two Grasp enums representing the primary and secondary faces.
        """
        all_axes = [AxisIdentifier.X, AxisIdentifier.Y, AxisIdentifier.Z]

        if not specified_grasp_axis == AxisIdentifier.Undefined:
            valid_axes = [specified_grasp_axis]
        else:
            valid_axes = [
                axis
                for axis in all_axes
                if not np.isnan(pose_to_robot_vector.to_list()[axis.value.index(1)])
            ]

        object_to_robot_vector = np.array(pose_to_robot_vector.to_list()) + 1e-9
        sorted_axes = sorted(
            valid_axes,
            key=lambda axis: abs(object_to_robot_vector[axis.value.index(1)]),
            reverse=True,
        )

        primary_axis: AxisIdentifier = sorted_axes[0]
        primary_sign = int(np.sign(object_to_robot_vector[primary_axis.value.index(1)]))

        primary_axis_class = (
            VerticalAlignment if primary_axis == AxisIdentifier.Z else ApproachDirection
        )
        primary_face = primary_axis_class.from_axis_direction(
            primary_axis, primary_sign
        )

        if len(sorted_axes) > 1:
            secondary_axis: AxisIdentifier = sorted_axes[1]
            secondary_sign = int(
                np.sign(object_to_robot_vector[secondary_axis.value.index(1)])
            )
        else:
            secondary_axis: AxisIdentifier = primary_axis
            secondary_sign = -primary_sign

        secondary_axis_class = (
            VerticalAlignment
            if secondary_axis == AxisIdentifier.Z
            else ApproachDirection
        )
        secondary_face = secondary_axis_class.from_axis_direction(
            secondary_axis, secondary_sign
        )

        return primary_face, secondary_face

@dataclass
class PreferredGraspAlignment:
    """
    Description of the preferred grasp alignment for an object.
    """

    preferred_axis: Optional[AxisIdentifier]
    """
    The preferred axis, X, Y, or Z, for grasping the object, or None if not specified.
    """

    with_vertical_alignment: bool
    """
    Indicates if the object should be grasped with a vertical alignment.
    """

    with_rotated_gripper: bool
    """
    Indicates if the gripper should be rotated by 90° around X.
    """
