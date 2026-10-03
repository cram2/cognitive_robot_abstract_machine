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
from semantic_digital_twin.semantic_annotations.mixins import GraspCandidate
from semantic_digital_twin.spatial_types.spatial_types import Pose
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

    def check_man_occupancy(self, condition: Callable[List[Body], bool]) -> bool:
        """
        Checks the occupancy of the gripper against a condition.

        The condition get the list of bodies that are under the TCP in the kinematic
        structure and returns a boolean.

        :param condition: The condition that should be evaluated.
        :return: True if the condition is satisfied, False otherwise.
        """
        bodies_under_tcp = (
            self.end_effector._world.get_kinematic_structure_entities_of_branch(
                self.end_effector.tool_frame
            )
        )
        if self.end_effector.tool_frame in bodies_under_tcp:
            bodies_under_tcp.remove(self.end_effector.tool_frame)
        return condition(bodies_under_tcp)


@dataclass
class GripperIsFree(GripperOccupancy, Predicate):
    """
    Checks if the gripper is holding something.

    Checks this by looking at the kinematic structure of the end_effector.
    """

    def __call__(self) -> bool:
        return self.check_man_occupancy(lambda bodies: len(bodies) == 0)

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
        return self.check_man_occupancy(lambda bodies: len(bodies) != 0)

    @classmethod
    def _verbalization_fragment_(cls, fields):
        return clause(Noun(fields["end_effector"]), Copula(), Adjective("occupied"))


@dataclass(eq=False)
class IsAmongTheClosestGraspsTo(Predicate):
    """
    Whether a grasp is among the grasps closest to where the robot stands.

    Grasps are ranked by their horizontal distance from the standing pose, grasps at
    the same distance by the angle between the direction they are approached along and
    the direction from the standing pose to them, and grasps tied on both by the order
    of :attr:`grasps`.

    .. note:: With the grasp fixed and the standing pose left open, it chooses the
        standing poses that suit that grasp instead.
    """

    grasp: GraspCandidate
    """
    The grasp that is asked about.
    """

    standing_position: Pose
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
        world_P_standing = world.transform(self.standing_position, world.root).to_np()[
            :, 3
        ]
        world_T_object = self.grasp.graspable.root.global_transform.to_np()

        listed = any(grasp is self.grasp for grasp in self.grasps)
        ranked = self.grasps if listed else [*self.grasps, self.grasp]
        # Sorting is stable, so grasps exactly as close as one another keep the order
        # they are listed in, and no more of them count as the closest than were asked for.
        closest = sorted(
            ranked,
            key=lambda grasp: self._closeness(grasp, world_T_object, world_P_standing),
        )[: self.number_of_grasps]
        return any(grasp is self.grasp for grasp in closest)

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
        world_T_grasp = world_T_object @ grasp.root_T_grasp.to_np()
        world_V_standing_to_grasp = world_T_grasp[:, 3] - world_P_standing
        horizontal_distance = np.linalg.norm(world_V_standing_to_grasp[:2])
        cosine = (
            world_T_grasp[:, 0]
            @ world_V_standing_to_grasp
            / np.linalg.norm(world_V_standing_to_grasp)
        )
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
