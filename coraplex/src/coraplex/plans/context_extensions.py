from __future__ import annotations

from dataclasses import dataclass, field


from cramph.context import ContextExtension
from krrood.entity_query_language.backends import (
    EntityQueryLanguageGenerativeBackend,
    QueryBackend,
)
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

# %% what a plan's nodes read from the context


@dataclass
class RobotAccess(ContextExtension):
    """
    Gives the nodes of a statechart the robot that performs the plan.
    """

    robot: AbstractRobot
    """
    The robot performing the plan.
    """

    @property
    def controlled_root(self) -> KinematicStructureEntity:
        """
        :return: The topmost entity the robot may move, which a Cartesian goal is
            expressed relative to. Driving the base while manipulating moves the robot
            relative to the world, so the world root is the only frame that holds still
            then; otherwise the robot's own root does.
        """
        if (
            isinstance(self.robot, HasMobileBase)
            and self.robot.mobile_base.full_body_controlled
        ):
            return self.robot._world.root
        return self.robot.root


@dataclass
class StatementGrounding(ContextExtension):
    """
    How the underspecified statements of a plan are grounded into actions.
    """

    query_backend: QueryBackend = field(
        default_factory=EntityQueryLanguageGenerativeBackend
    )
    """
    The backend answering the underspecified statements.

    Defaults to the deterministic generative backend, since underspecified actions are
    constructed (generated).
    """

    candidates_to_try: int = 50
    """
    How many candidates an underspecified step tries before giving up, unless the step
    has a limit of its own.
    """

    sampling_seed: int | None = 0
    """
    Seed for the locations the plan's actions sample, so a run repeats; ``None`` samples
    afresh each run.
    """
