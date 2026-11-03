from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, fields

from typing_extensions import Any

from coraplex.plans.context_extensions import RobotAccess
from coraplex.plans.plan_transformation import PlanRewriting
from krrood.ormatic.utils import classproperty
from cramph.node import SucceedsOnObservingTrue, FailsOnObservingFalse
from cramph.composites import ControlStructureNode, Parallel, Sequence, TryAll
from cramph.context import ContextExtension
from cramph.node import (
    CompositeNode,
    NodeArtifacts,
    StatechartNode,
    expanded_child_field,
)
from giskardpy.motion_statechart.graph_node import MotionStatechartNode
from giskardpy.motion_statechart.tasks.cartesian_tasks import HoldPose
from krrood.class_diagrams.attribute_introspector import DataclassOnlyIntrospector
from krrood.patterns.field_metadata import JSONMetadata
from krrood.symbolic_math.symbolic_math import Scalar, trinary_if_cases
from semantic_digital_twin.collision_checking.collision_matrix import CollisionRule
from semantic_digital_twin.robots.robot_part_mixins import HasMobileBase
from semantic_digital_twin.robots.robot_parts import AbstractRobot
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

logger = logging.getLogger(__name__)


@dataclass(eq=False, repr=False, kw_only=True)
class Action(CompositeNode, ABC):
    """
    Something a robot does, described by the parameters it is given and run as a node of
    a statechart.

    It ends itself once it did what it does, so an action can be a step of another one.
    What it does takes one of a few shapes: a :class:`GoalAction` reaches a single goal,
    while a :class:`SequentialAction`, :class:`ParallelAction` or :class:`TryAllAction`
    is the control structure node of the steps it creates.

    A parameter may also be given as an underspecified statement of its value, such as
    ``a(...)``, which is grounded when the action runs.
    """

    temporary_collision_rules: list[CollisionRule] = field(
        default_factory=list,
        init=False,
        repr=False,
        metadata=JSONMetadata(serialize=True).as_dict(),
    )
    """
    The collision rules in force while this action runs, next to the temporary rules of
    everything else running at the same time, as created by
    :meth:`create_temporary_collision_rules` when the action is expanded.

    They belong to the action rather than to a node beside each of its steps, so they
    also hold between two steps. Sent along with the action, because a receiver does not
    expand it again.
    """

    @classproperty
    def context_extensions_required_to_expand(
        cls,
    ) -> tuple[type[ContextExtension], ...]:
        return super().context_extensions_required_to_expand + (RobotAccess,)

    @property
    def designator_parameter(self) -> dict[str, Any]:
        """
        :return: Every parameter this action was given, by the name it was given under:
            the public fields a caller sets that the subclasses of :class:`Action`
            declare.
        """
        statechart_fields = {action_field.name for action_field in fields(Action)}
        return {
            discovered.public_name: getattr(self, discovered.field.name)
            for discovered in DataclassOnlyIntrospector().discover(type(self))
            if discovered.field.init and discovered.field.name not in statechart_fields
        }

    @property
    def robot(self) -> AbstractRobot:
        """
        :return: The robot performing this action.
        """
        return self.context.require_extension(RobotAccess).robot

    @property
    def controlled_root(self) -> KinematicStructureEntity:
        """
        :return: The topmost entity this action's motions may move the robot relative
            to, which its Cartesian goals are expressed in.
        """
        return self.context.require_extension(RobotAccess).controlled_root

    def keep_base_still(self) -> list[MotionStatechartNode]:
        """
        :return: The task holding the robot's base where it stands, empty when the robot
            has no mobile base or may move its whole body.

        A goal relative to the robot is bound when it starts, so a base that moves
        afterwards, which collision avoidance does to buy clearance, carries the goal
        away from its target.
        """
        robot = self.robot
        if (
            not isinstance(robot, HasMobileBase)
            or robot.mobile_base.full_body_controlled
        ):
            return []
        return [
            HoldPose(name="hold base", root_link=self.world.root, tip_link=robot.root)
        ]

    def expand(self) -> None:
        """
        Creates what this action does below it, once the
        :class:`~coraplex.plans.plan_transformation.PlanRewriting` of :attr:`context`,
        if any, configured this action.
        """
        rewriting = self.context.get_extension(PlanRewriting)
        if rewriting is not None:
            rewriting.configure(self)
        self.temporary_collision_rules = self.create_temporary_collision_rules()
        self._create_children()

    def create_temporary_collision_rules(self) -> list[CollisionRule]:
        """
        :return: New collision rules to keep in force while this action runs, none by
            default. New ones for each action, because rules are taken out of force by
            identity, so an action inside another one with equal rules takes out only its
            own.
        """
        return []

    def on_start(self) -> None:
        super().on_start()
        if self.temporary_collision_rules:
            self.context.world.collision_manager.put_temporary_rules_in_force(
                self.temporary_collision_rules
            )

    def on_end(self) -> None:
        self._take_temporary_collision_rules_out_of_force()
        super().on_end()

    def on_reset(self) -> None:
        self._take_temporary_collision_rules_out_of_force()
        super().on_reset()

    def cleanup(self) -> None:
        """
        Takes :attr:`temporary_collision_rules` out of force, so a run that stopped
        while this action ran leaves none of them behind.
        """
        self._take_temporary_collision_rules_out_of_force()
        super().cleanup()

    def _take_temporary_collision_rules_out_of_force(self) -> None:
        """
        Takes :attr:`temporary_collision_rules` out of force, which skips those that are
        not in force, so a reset that ends no run changes nothing.
        """
        if self.temporary_collision_rules:
            self.context.world.collision_manager.take_temporary_rules_out_of_force(
                self.temporary_collision_rules
            )

    @abstractmethod
    def _create_children(self) -> None:
        """
        Puts the nodes doing what this action does below it, in the shape of its kind of
        action.
        """


@dataclass(eq=False, repr=False, kw_only=True)
class GoalAction(Action, SucceedsOnObservingTrue, FailsOnObservingFalse, ABC):
    """
    An action reaching a single goal, the node :meth:`create_goal` creates, which stays
    a node of its own below the action.

    The action reaches its goal once that node did, and declares itself failed as soon
    as the node failed. A goal whose owner decides its success, such as a single task,
    is ended by the action once it observes True.
    """

    goal: StatechartNode | None = expanded_child_field(default=None)
    """
    The child reaching the goal of this action, None until the action was expanded.
    """

    @abstractmethod
    def create_goal(self) -> StatechartNode:
        """
        :return: A new node reaching the goal of this action. Called once, when the
            action is expanded.
        """

    def _create_children(self) -> None:
        """
        Puts the goal below this action and ends it once it is at its goal, unless it
        decides its own success.
        """
        self.goal = self.create_goal()
        self._add_child_to_statechart(self.goal)
        self._succeed_at_its_goal(self.goal)

    def build_artifacts(self) -> NodeArtifacts:
        """
        Report how the goal ended, or nothing while it has not.
        """
        return NodeArtifacts(
            observation=trinary_if_cases(
                cases=[
                    (self.goal.is_succeeded, Scalar.const_true()),
                    (self.goal.is_failed_or_interrupted, Scalar.const_false()),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


@dataclass(eq=False, repr=False, kw_only=True)
class ActionOfSteps(Action, ControlStructureNode, ABC):
    """
    An action that is the control structure node of its own steps, the nodes
    :meth:`create_steps` creates, so the steps are its children and the control
    structure node it also is decides how they run and when they are done.
    """

    nodes: list[StatechartNode] = field(default_factory=list, init=False, repr=False)
    """
    The steps of this action, which :meth:`create_steps` creates when the action is
    expanded, and which plan transformations may add to.
    """

    @abstractmethod
    def create_steps(self) -> list[StatechartNode]:
        """
        :return: New nodes doing what this action does, in order. Called once, when the
            action is expanded.
        """

    def _create_children(self) -> None:
        """
        Puts the steps below this action and lets the control structure node wire them.
        """
        self.nodes = [self._as_node(step) for step in self.create_steps()]
        self.adopt_handed_children()


@dataclass(eq=False, repr=False, kw_only=True)
class SequentialAction(ActionOfSteps, Sequence, ABC):
    """
    An action running its steps one after another, done once the last one succeeded.
    """


@dataclass(eq=False, repr=False, kw_only=True)
class ParallelAction(ActionOfSteps, Parallel, ABC):
    """
    An action running its steps side by side, done once all of them are at their goals
    together.
    """

    minimum_success: int | None = field(default=None, init=False, repr=False)
    """
    None, so every step has to be at its goal; how many have to be is up to the action,
    not a parameter it is described by.
    """

    @property
    def inherent_success_condition(self) -> Scalar:
        """
        Differs from :attr:`~cramph.node.EndedByOwner.inherent_success_condition` of a
        parallel in ending the action once it observes True, so it can be a step of
        another one.
        """
        return self.observes_true


@dataclass(eq=False, repr=False, kw_only=True)
class TryAllAction(ActionOfSteps, TryAll, ABC):
    """
    An action running its steps side by side as alternatives, done once one of them
    succeeded.
    """
