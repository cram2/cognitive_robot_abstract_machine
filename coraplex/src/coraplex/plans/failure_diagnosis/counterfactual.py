"""
Asks of a task that stopped approaching its goal whether it would have reached it had it
been free to move the degrees of freedom an earlier node left where they are.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum

from typing_extensions import Any

from coraplex.plans.underspecified import ActionTrial, TrialBuild
from cramph.node import NodeRebuild, StatechartNode
from giskardpy.motion_statechart.mixins import HasKinematicChainParameter
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.kinematic_chain import KinematicChain

# %% the answer of the counterfactual


class StallCounterfactual(StrEnum):
    """
    What trying a stalled task again, free to move what it only read, showed.
    """

    REACHED = "reached its goal once free to move what an earlier node left"
    STILL_STALLED = "stalled again although free to move what an earlier node left"
    NOT_TRIED = "could not be tried free of what an earlier node left"


# %% trying the task free of its inputs


@dataclass(kw_only=True)
class TaskFreedOfItsInputs(NodeRebuild, TrialBuild):
    """
    Builds a stalled task anew onto a copy of the world, with its kinematic chain
    reaching down to the root of that world, so it may move every degree of freedom
    between the root and its tip.
    """

    task: HasKinematicChainParameter
    """
    The task that stalled.
    """

    _world: World | None = field(default=None, init=False, repr=False)
    """
    The copy the task is built onto.
    """

    def build_for_trial(self, *, world: World) -> StatechartNode:
        self._world = world
        freed = self.task.rebuilt_by(rebuild=self)
        freed.kinematic_chain = KinematicChain(
            root=world.root, tip=freed.kinematic_chain.tip
        )
        return freed

    def carry_over_conditions(self) -> None:
        """
        The freed task runs alone, so it carries over no condition.
        """

    def rebuilt(self, *, node: StatechartNode) -> StatechartNode | None:
        return node.rebuilt_by(rebuild=self)

    def rebuilt_children(
        self, *, owner: StatechartNode, children: list[StatechartNode]
    ) -> list[StatechartNode]:
        return [child.rebuilt_by(rebuild=self) for child in children]

    def rebound(self, *, value: Any) -> Any:
        return self._world.rebind_world_entities(value)

    def __str__(self) -> str:
        """
        :return: The stalled task and that it may move its whole chain.
        """
        return f"{self.task.name} free to move its whole chain from the world root"


def try_stalled_task_freed(
    *, task: StatechartNode, trial: ActionTrial | None
) -> StallCounterfactual:
    """
    :param task: A task that stopped approaching its goal.
    :param trial: The trial to try it in, None if the plan has none.
    :return: Whether the task reaches its goal once free to move its whole chain.
    """
    if trial is None or not isinstance(task, HasKinematicChainParameter):
        return StallCounterfactual.NOT_TRIED
    if trial.succeeds(rebuild=TaskFreedOfItsInputs(task=task)):
        return StallCounterfactual.REACHED
    return StallCounterfactual.STILL_STALLED
