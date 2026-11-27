"""
The diagnoses of a plan failure: each says how the run where the failure started failed,
and which other runs caused it.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import StrEnum

from typing_extensions import TYPE_CHECKING, Self

from coraplex.plans.failure_diagnosis.causal_graph import RootCause
from coraplex.plans.failure_diagnosis.counterfactual import StallCounterfactual
from coraplex.plans.failures import PlanFailure
from coraplex.robot_plans.actions.base import Action
from cramph.composites import StatementNode
from cramph.state_history import NodeRun
from cramph.world_access import WorldAccess
from krrood.patterns.classproperty import classproperty
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

if TYPE_CHECKING:
    from coraplex.plans.failure_diagnosis.diagnosis import FailureDiagnoser

# %% what a diagnosis could say about the cause


class Verdict(StrEnum):
    """
    What a diagnosis could say about the cause.
    """

    ONE_CAUSING_NODE = "one other node caused it"
    SEVERAL_CANDIDATES = "one of several other nodes caused it"
    NO_OTHER_NODE = "no other node explains it"


# %% a diagnosis


@dataclass(kw_only=True)
class FailureDiagnosis(PlanFailure, ABC):
    """
    Why a run of a plan failed, in terms of other runs.

    Each subclass is one way the run where the failure started can fail.
    """

    reported_run: NodeRun
    """
    The run the failure was reported for.
    """

    origin_run: NodeRun
    """
    The run where the failure started.
    """

    root_causes: list[RootCause]
    """
    The runs that caused the failure, the most likely first.
    """

    @classmethod
    @abstractmethod
    def diagnose_if_failed_this_way(
        cls, *, reported_run: NodeRun, origin_run: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        """
        :param reported_run: A run a failure is reported for.
        :param origin_run: The run where that failure started.
        :param diagnoser: The diagnoser of the statechart the runs belong to.
        :return: The diagnosis of the failure if `origin_run` failed this way, else
            None.
        """

    @classproperty
    @abstractmethod
    def failure_description(cls) -> str:
        """
        :return: How the origin failed, as the verb phrase following it.
        """

    @property
    def verdict(self) -> Verdict:
        """
        :return: What this diagnosis could say about the cause.
        """
        causing_nodes = {root.causing_run.node for root in self.root_causes}
        if not causing_nodes:
            return Verdict.NO_OTHER_NODE
        if len(causing_nodes) == 1:
            return Verdict.ONE_CAUSING_NODE
        return Verdict.SEVERAL_CANDIDATES

    def degrees_of_freedom_left_by_earlier_nodes(
        self, *, access: WorldAccess
    ) -> frozenset[DegreeOfFreedom]:
        """
        :param access: What the node of :attr:`origin_run` does to the world.
        :return: What :attr:`origin_run` read whose state its statechart was given from
            before it.
        """
        return access.read_degrees_of_freedom

    @staticmethod
    def describe_run(*, run: NodeRun) -> str:
        """
        :param run: A run.
        :return: `run`, with the action holding its node if that is another node.
        """
        action = Action.enclosing_action_of(node=run.node)
        if action is None or action is run.node:
            return str(run)
        return f"{run} in {action.name}"

    def report_lines(self) -> list[str]:
        """
        :return: The reported run, the origin run and how it failed, one per line.
        """
        return [
            f"reported {self.describe_run(run=self.reported_run)}",
            f"origin   {self.describe_run(run=self.origin_run)} "
            f"{self.failure_description}",
        ]

    def error_message(self) -> str:
        """
        :return: The reported run, the origin run and how it failed, the verdict and
            each cause with the edges from it to the origin run, one per line.
        """
        lines = [*self.report_lines(), f"verdict  {self.verdict.value}"]
        for rank, root in enumerate(self.root_causes, start=1):
            lines.append(f"cause {rank}  {self.describe_run(run=root.causing_run)}")
            lines.extend(
                f"           {edge.cause.node.name} {edge.describe_relation()} "
                f"{edge.effect.node.name}"
                for edge in root.edges_to_origin
            )
        return "\n".join(lines)

    def suggest_correction(self) -> str:
        if not self.root_causes:
            return ""
        return "Repair the node of the first cause before trying the failed one again."


# %% the ways the origin of a failure can fail


@dataclass(kw_only=True)
class TaskStalled(FailureDiagnosis):
    """
    The origin is a motion task that stopped approaching its goal.
    """

    counterfactual: StallCounterfactual | None = None
    """
    What trying the task again, free to move what an earlier node left or with its
    joints moved away from the position limits blocking it, showed; None if it was not
    tried again.
    """

    @classmethod
    def diagnose_if_failed_this_way(
        cls, *, reported_run: NodeRun, origin_run: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        if origin_run.node not in diagnoser.stalled_nodes:
            return None
        root_causes, counterfactual = diagnoser.find_causes_of_stall(stalled=origin_run)
        return cls(
            reported_run=reported_run,
            origin_run=origin_run,
            root_causes=root_causes,
            counterfactual=counterfactual,
        )

    @classproperty
    def failure_description(cls) -> str:
        return "stopped approaching its goal"

    def degrees_of_freedom_left_by_earlier_nodes(
        self, *, access: WorldAccess
    ) -> frozenset[DegreeOfFreedom]:
        """
        Differs from
        :meth:`FailureDiagnosis.degrees_of_freedom_left_by_earlier_nodes` in leaving
        out what the task moves or holds itself.
        """
        return access.read_degrees_of_freedom - access.moved_or_held_degrees_of_freedom

    def report_lines(self) -> list[str]:
        lines = super().report_lines()
        if self.counterfactual is not None:
            lines.append(f"trial    {self.counterfactual.value}")
        return lines


@dataclass(kw_only=True)
class TaskCollided(FailureDiagnosis):
    """
    The origin moved bodies closer to each other than collision avoidance allows.
    """

    @classmethod
    def diagnose_if_failed_this_way(
        cls, *, reported_run: NodeRun, origin_run: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        if origin_run.node not in diagnoser.colliding_nodes:
            return None
        return cls(
            reported_run=reported_run,
            origin_run=origin_run,
            root_causes=diagnoser.find_causes_of_collision(colliding=origin_run),
        )

    @classproperty
    def failure_description(cls) -> str:
        return "brought bodies closer than collision avoidance allows"


@dataclass(kw_only=True)
class StatementRanOutOfCandidates(FailureDiagnosis):
    """
    The origin is an underspecified statement none of whose candidates succeeded.
    """

    @classmethod
    def diagnose_if_failed_this_way(
        cls, *, reported_run: NodeRun, origin_run: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        if not isinstance(origin_run.node, StatementNode):
            return None
        return cls(
            reported_run=reported_run,
            origin_run=origin_run,
            root_causes=diagnoser.find_causes_shared_by_candidates(
                statement=origin_run
            ),
        )

    @classproperty
    def failure_description(cls) -> str:
        return "ran out of candidates to try"


@dataclass(kw_only=True)
class NodeFailedByACondition(FailureDiagnosis):
    """
    The origin was failed or interrupted by a condition reading another node.
    """

    @classmethod
    def diagnose_if_failed_this_way(
        cls, *, reported_run: NodeRun, origin_run: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self | None:
        if not diagnoser.edges_from_conditions_that_ended(run=origin_run):
            return None
        return cls(
            reported_run=reported_run,
            origin_run=origin_run,
            root_causes=diagnoser.find_causes_of_failing_condition(run=origin_run),
        )

    @classproperty
    def failure_description(cls) -> str:
        return "was failed by a condition reading another node"


@dataclass(kw_only=True)
class NodeFailedOnItsOwn(FailureDiagnosis):
    """
    The origin failed on what it observed itself; this way of failing applies to every
    origin.
    """

    @classmethod
    def diagnose_if_failed_this_way(
        cls, *, reported_run: NodeRun, origin_run: NodeRun, diagnoser: FailureDiagnoser
    ) -> Self:
        return cls(
            reported_run=reported_run,
            origin_run=origin_run,
            root_causes=diagnoser.find_causes_of_observation(run=origin_run),
        )

    @classproperty
    def failure_description(cls) -> str:
        return "failed on what it observed itself"
