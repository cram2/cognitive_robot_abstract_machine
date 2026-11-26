"""
A plan that detects two milks, then takes one and puts it down as one statement: both
steps work on the same milk, and the next milk is tried as long as one cannot be put
down; as one action holding both steps, and as a sequence of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from coraplex.plans.failures import PlanFailure
from coraplex.robot_plans.actions.base import SequentialAction
from cramph.composites import Sequence, StatementNode
from cramph.context import ContextExtension
from cramph.data_types import LifeCycleValue
from cramph.monitors import CountTicks
from cramph.node import StatechartNode
from cramph.threaded_nodes import FunctionCall
from krrood.entity_query_language.factories import a, variable
from krrood.entity_query_language.query.match import Match
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.semantic_annotations.semantic_annotations import Milk
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.world_entity import Body
from ...plan_running import run_plan

# %% mimics of detecting, taking and putting down milks


@dataclass(kw_only=True)
class MilkDetection:
    """
    Stands in for a detection that finds milks the world does not know yet.
    """

    world: World
    """
    The world the detected milks are added to.
    """

    milk_names: list[PrefixedName]
    """
    The names of the detected milks, in the order they are added.
    """

    def add_detected_milks(self) -> None:
        """
        Add a milk for each of :attr:`milk_names` to :attr:`world`.
        """
        with self.world.modify_world():
            for name in self.milk_names:
                body = Body(name=name)
                self.world.add_kinematic_structure_entity(body)
                self.world.add_connection(
                    FixedConnection(parent=self.world.root, child=body)
                )
                self.world.add_semantic_annotation(Milk(root=body))


@dataclass(eq=False, repr=False, kw_only=True)
class TakeMilkAction(SequentialAction):
    """
    Takes a milk, which always succeeds.
    """

    milk: Milk
    """
    The milk taken.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [CountTicks(ticks=1)]


@dataclass(eq=False, repr=False, kw_only=True)
class PutDownMilkAction(SequentialAction):
    """
    Puts a milk down, which fails for every milk but one.
    """

    milk: Milk
    """
    The milk put down.
    """

    placeable_milk_name: PrefixedName
    """
    The name of the only milk this action succeeds for.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [FunctionCall(function=self._put_down, failure_types=(PlanFailure,))]

    def _put_down(self) -> None:
        """
        :raises PlanFailure: If :attr:`milk` is not the milk that can be put down.
        """
        if self.milk.root.name != self.placeable_milk_name:
            raise PlanFailure()


@dataclass(eq=False, repr=False, kw_only=True)
class TakeAndPutDownMilkAction(SequentialAction):
    """
    Takes a milk and puts it down.
    """

    taking: TakeMilkAction
    """
    The step taking the milk.
    """

    putting_down: PutDownMilkAction
    """
    The step putting the milk down.
    """

    def create_steps(self) -> list[StatechartNode]:
        return [self.taking, self.putting_down]


# %% choosing among the detected milks


@dataclass(kw_only=True)
class DetectedMilks:
    """
    Two milks a plan detects, the one of them that can be put down, and the variable
    ranging over the milks of the world, which holds a milk of its own already.
    """

    detection: MilkDetection
    """
    The detection adding the milks.
    """

    placeable_milk_name: PrefixedName
    """
    The name of the only milk that can be put down.
    """

    @property
    def milk(self) -> Milk:
        """
        :return: A new variable ranging over the milks of the world, read when the
            statement using it is grounded.
        """
        return variable(Milk, domain=self.detection.world.semantic_annotations)


@pytest.fixture(params=[0, 1], ids=["first_placeable", "second_placeable"])
def detected_milks(
    apartment_world_pr2_copy_with_context, request: pytest.FixtureRequest
) -> DetectedMilks:
    """
    :return: The milks the plan detects in the apartment, with the first or the second
        of them the one that can be put down.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    milk_names = [PrefixedName("first_milk"), PrefixedName("second_milk")]
    return DetectedMilks(
        detection=MilkDetection(world=world, milk_names=milk_names),
        placeable_milk_name=milk_names[request.param],
    )


def run_detection_then(
    *,
    statement: Match,
    detected_milks: DetectedMilks,
    extensions: list[ContextExtension],
) -> StatementNode:
    """
    Run a plan that detects the milks and then grounds and runs `statement`.

    :return: The node that grounded `statement`.
    """
    plan = Sequence(
        nodes=[
            FunctionCall(function=detected_milks.detection.add_detected_milks),
            statement,
        ]
    )
    run_plan(plan, extensions)
    assert plan.life_cycle_state == LifeCycleValue.SUCCEEDED
    [grounding] = [node for node in plan.nodes if isinstance(node, StatementNode)]
    return grounding


def test_one_action_taking_and_putting_down_the_same_milk_tries_the_next_milk(
    apartment_world_pr2_copy_with_context, detected_milks: DetectedMilks
) -> None:
    """
    The action is grounded after the detection, so its candidates include the detected
    milks; its steps share the milk, and each milk that fails to be put down in its
    trial gives way to the next.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    milk = detected_milks.milk
    statement = a(TakeAndPutDownMilkAction)(
        taking=a(TakeMilkAction)(milk=milk),
        putting_down=a(PutDownMilkAction)(
            milk=milk, placeable_milk_name=detected_milks.placeable_milk_name
        ),
    )

    grounding = run_detection_then(
        statement=statement, detected_milks=detected_milks, extensions=extensions
    )

    [chosen] = grounding.grounded_nodes
    assert [
        (type(step), step.milk.root.name, step.milk.root._world is world)
        for step in (chosen.taking, chosen.putting_down)
    ] == [
        (TakeMilkAction, detected_milks.placeable_milk_name, True),
        (PutDownMilkAction, detected_milks.placeable_milk_name, True),
    ]


def test_a_sequence_taking_and_putting_down_the_same_milk_tries_the_next_milk(
    apartment_world_pr2_copy_with_context, detected_milks: DetectedMilks
) -> None:
    """
    As
    :func:`test_one_action_taking_and_putting_down_the_same_milk_tries_the_next_milk`,
    with the steps given as a sequence instead of an action.
    """
    world, robot, extensions = apartment_world_pr2_copy_with_context
    milk = detected_milks.milk
    statement = a(Sequence)(
        nodes=[
            a(TakeMilkAction)(milk=milk),
            a(PutDownMilkAction)(
                milk=milk, placeable_milk_name=detected_milks.placeable_milk_name
            ),
        ]
    )

    grounding = run_detection_then(
        statement=statement, detected_milks=detected_milks, extensions=extensions
    )

    [chosen] = grounding.grounded_nodes
    assert [
        (type(step), step.milk.root.name, step.milk.root._world is world)
        for step in chosen.nodes
    ] == [
        (TakeMilkAction, detected_milks.placeable_milk_name, True),
        (PutDownMilkAction, detected_milks.placeable_milk_name, True),
    ]
