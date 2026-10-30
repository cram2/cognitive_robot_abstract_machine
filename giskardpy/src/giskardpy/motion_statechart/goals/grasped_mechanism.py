from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from krrood.symbolic_math.symbolic_math import trinary_logic_and
from semantic_digital_twin.spatial_types.spatial_types import Pose
from semantic_digital_twin.world_description.connections import ActiveConnection1DOF
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)
from cramph.node import EndedByOwner
from giskardpy.motion_statechart.weights import DefaultWeights
from cramph.node import CompositeNode, NodeArtifacts
from giskardpy.motion_statechart.tasks.cartesian_tasks import CartesianPose
from giskardpy.motion_statechart.tasks.joint_tasks import JointPositionList, JointState


@dataclass(eq=False, repr=False, kw_only=True)
class ArticulateGraspedMechanism(EndedByOwner, CompositeNode):
    """
    Drive the degree of freedom of a 1-dof mechanism in an environment to a position
    while keeping the end effector fixed relative to the grasped part.

    Assumes that the grasped part (e.g. a handle or a bottle cap) has already been
    grasped. Works with any mechanism whose grasped part hangs below an
    :class:`ActiveConnection1DOF`, e.g. drawers, doors, or screw caps.
    """

    tip_link: KinematicStructureEntity
    """
    End effector that is grasping the handle.
    """

    environment_link: KinematicStructureEntity
    """
    Name of the handle that was grasped.
    """

    goal_joint_state: float
    """
    The position the degree of freedom of the mechanism is driven to, clamped to its
    limits once this goal expands.
    """

    mechanism_weight: float = DefaultWeights.WEIGHT_BELOW_COLLISION_AVOIDANCE
    """
    Weight of the goal driving the degree of freedom of the mechanism.

    Below collision avoidance, because following a mechanism contorts the arm against
    whatever is around it, and at a higher weight the solver buys the trajectory by
    pushing the arm through what is in its way.
    """

    grasp_weight: float = DefaultWeights.WEIGHT_ABOVE_COLLISION_AVOIDANCE
    """
    Weight of the goal keeping the end effector fixed relative to the grasped part.

    .. warning:: Unlike most goals, this one is weighted above collision avoidance on
        purpose. Holding on to the grasped part has to matter more than both moving the
        mechanism and keeping clear of collisions: at a lower weight the solver buys
        clearance by letting the end effector drift off the grasped part, and moves the
        mechanism without the end effector holding on to it.
    """

    def expand(self) -> None:
        self.connection = self.environment_link.get_first_parent_connection_of_type(
            ActiveConnection1DOF
        )
        limits = self.connection.dof.limits
        self.goal_joint_state = float(
            np.clip(self.goal_joint_state, limits.lower.position, limits.upper.position)
        )

        self._add_children_to_statechart(
            [
                JointPositionList(
                    name="hinge goal",
                    goal_state=JointState.from_mapping(
                        {self.connection: self.goal_joint_state}
                    ),
                    weight=self.mechanism_weight,
                ),
                CartesianPose(
                    name="hold handle",
                    root_link=self.environment_link,
                    tip_link=self.tip_link,
                    goal_pose=Pose(reference_frame=self.tip_link),
                    weight=self.grasp_weight,
                ),
            ]
        )

    def build_artifacts(self) -> NodeArtifacts:
        """
        Build an observation that is True once both the degree of freedom and the grip
        on the grasped part reached their goals.

        Both parts are created here and end only when this goal ends, so each is judged
        by what it observes now and stops counting once it drifts away from its goal
        again.
        """
        return NodeArtifacts(
            observation=trinary_logic_and(
                *[node.observation_variable for node in self.nodes]
            )
        )
