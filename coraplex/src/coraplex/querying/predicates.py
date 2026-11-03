from dataclasses import dataclass

from typing_extensions import Callable

from krrood.entity_query_language.predicate import Predicate, RenderedFields
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Noun,
    predicate_clause,
)
from semantic_digital_twin.robots.robot_parts import EndEffector
from semantic_digital_twin.world_description.world_entity import Body


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
        self, condition: Callable[list[Body], bool]
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
