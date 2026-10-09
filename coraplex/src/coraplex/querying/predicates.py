from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from typing_extensions import List, Callable, Tuple

from krrood.entity_query_language.predicate import Predicate, RenderedFields
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Adjective,
    clause,
    Copula,
    Noun,
    predicate_clause,
)
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.grasping.grasp_candidates import GraspCandidate
from semantic_digital_twin.spatial_types.spatial_types import Pose2D
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
    Body,
)


@dataclass
class GripperOccupancy:
    """
    Base class for predicates that check the gripper occupancy.
    """

    end_effector: EndEffector
    """
    Semantic annotation for the gripper that should be evaluated.
    """

    def check_manipulator_occupancy(
        self, condition: Callable[List[Body], bool]
    ) -> bool:
        """
        Checks the occupancy of the gripper against a condition.

        The condition gets the list of bodies below the tool frame in the kinematic
        structure and returns a boolean.

        :param condition: The condition that should be evaluated.
        :return: True if the condition is satisfied, False otherwise.
        """
        bodies_below_tool_frame = (
            self.end_effector._world.get_kinematic_structure_entities_of_branch(
                self.end_effector.tool_frame
            )
        )
        if self.end_effector.tool_frame in bodies_below_tool_frame:
            bodies_below_tool_frame.remove(self.end_effector.tool_frame)
        return condition(bodies_below_tool_frame)


@dataclass
class GripperIsFree(GripperOccupancy, Predicate):
    """
    Checks if the gripper is holding something.

    Checks this by looking at the kinematic structure of the end_effector.
    """

    def __call__(self) -> bool:
        return self.check_manipulator_occupancy(lambda bodies: len(bodies) == 0)

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["end_effector"]), Copula(), Adjective("free"))


@dataclass
class GripperIsNotFree(GripperOccupancy, Predicate):
    """
    Checks if the gripper is free at the moment, so it can be used to grab something.

    This is checked by looking at the kinematic structure.
    """

    def __call__(self) -> bool:
        return self.check_manipulator_occupancy(lambda bodies: len(bodies) != 0)

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["end_effector"]), Copula(), Adjective("occupied"))


@dataclass
class GripperHolds(GripperOccupancy, Predicate):
    """
    Checks if the gripper holds a particular body.

    This is checked by looking at the kinematic structure below the tool frame, so a
    gripper holding something else does not count.
    """

    body: Body
    """
    The body the gripper is asked to hold.
    """

    def __call__(self) -> bool:
        return self.check_manipulator_occupancy(lambda bodies: self.body in bodies)

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<end effector> holds <body>"*.
        """
        return predicate_clause(cls, Noun(fields["end_effector"]), Noun(fields["body"]))


@dataclass(eq=False)
class ToolFrameIsAtGrasp(Predicate):
    """
    Whether an end effector's tool frame stands where a grasp is, within a tolerance.

    The tool frame reaches a grasp at the grasp's own position, which for a grasp on a
    rim or a handle lies away from the object's origin.
    """

    end_effector: EndEffector
    """
    The end effector whose tool frame is asked about.
    """

    grasp: GraspCandidate
    """
    The grasp the tool frame is to stand at.
    """

    tolerance: float = 0.03
    """
    How far, in meters along each axis, the tool frame may stand from the grasp.
    """

    def __call__(self) -> bool:
        world_T_grasp = self.grasp.world_T_grasp.to_np()
        world_T_tool = self.end_effector.tool_frame.global_transform.to_np()
        return bool(
            np.allclose(world_T_tool[:3, 3], world_T_grasp[:3, 3], atol=self.tolerance)
        )

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<end effector>'s tool frame is at <grasp>"*.
        """
        return predicate_clause(
            cls, Noun(fields["end_effector"]), Noun(fields["grasp"])
        )


@dataclass(eq=False)
class IsAmongTheClosestGraspsTo(Predicate):
    """
    Whether a grasp is among the grasps closest to where the robot stands.

    Grasps are ranked by their horizontal distance from the standing pose, then by the
    angle between their approach direction and the direction from the standing pose to
    them, then by their order in :attr:`grasps`. Left open, :attr:`standing_position` is
    grounded to the standing poses that suit a fixed grasp.
    """

    grasp: GraspCandidate
    """
    The grasp that is asked about.
    """

    standing_position: Pose2D
    """
    Where the robot stands while taking the grasp.
    """

    grasps: List[GraspCandidate]
    """
    The grasps on the same object that :attr:`grasp` is ranked among.

    A grasp not among them ranks after those as close as it.
    """

    number_of_grasps: int = 3
    """
    How many of :attr:`grasps` count as the closest.
    """

    def __call__(self) -> bool:
        world = self.grasp.graspable.root._world

        # Transform to np for speed, as this is called a lot
        world_P_standing = world.transform(
            self.standing_position.pose, world.root
        ).position.to_np()
        world_T_object = self.grasp.graspable.root.global_transform.to_np()

        position = next(
            (index for index, grasp in enumerate(self.grasps) if grasp is self.grasp),
            len(self.grasps),
        )
        ranked = [*self.grasps[:position], self.grasp, *self.grasps[position + 1 :]]
        closeness = self._closeness(self.grasp, world_T_object, world_P_standing)
        # Grasps exactly as close as this one rank by the order they are listed in, so
        # no more of them count as the closest than were asked for.
        ranked_ahead = sum(
            1
            for index, other_closeness in enumerate(
                self._closeness(grasp, world_T_object, world_P_standing)
                for grasp in ranked
            )
            if other_closeness < closeness
            or (other_closeness == closeness and index < position)
        )
        return ranked_ahead < self.number_of_grasps

    @staticmethod
    def _closeness(
        grasp: GraspCandidate,
        world_T_object: NDArray[np.float64],
        world_P_standing: NDArray[np.float64],
    ) -> Tuple[float, float]:
        """
        Computes a tuple of the horizontal distance from the standing pose to `grasp`,
        and the angle between the direction `grasp` is approached along and the
        direction from the standing pose to it.

        :param grasp: A grasp on the object.
        :param world_T_object: The object's root frame in the world frame.
        :param world_P_standing: Where the robot stands, as a homogeneous point in the
            world frame.
        :return: the tuple of the horizontal distance and the angle.
        """
        world_T_grasp = world_T_object @ grasp.grasp_pose.to_np()
        world_V_standing_to_grasp = world_T_grasp[:, 3] - world_P_standing
        horizontal_distance = np.linalg.norm(world_V_standing_to_grasp[:2])
        distance = np.linalg.norm(world_V_standing_to_grasp)
        if distance == 0:
            # Standing exactly at the grasp leaves no direction to approach it from.
            return float(horizontal_distance), 0.0
        cosine = world_T_grasp[:, 0] @ world_V_standing_to_grasp / distance
        return float(horizontal_distance), float(np.arccos(np.clip(cosine, -1.0, 1.0)))

    @classmethod
    def _verbalization_fragment_(cls, fields: RenderedFields) -> VerbalizationFragment:
        """
        :param fields: The rendered fragment for each field.
        :return: The clause *"<grasp> is among the closest grasps to <standing
            position>"*.
        """
        return predicate_clause(
            cls, Noun(fields["grasp"]), Noun(fields["standing_position"])
        )
