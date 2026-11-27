"""
Explains the failure of a plan by the nodes that caused it, from the causal graph of its
statechart.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import singledispatchmethod

from coraplex.plans.failure_diagnosis.causal_graph import RunCausalGraph, RootCause
from coraplex.plans.failure_diagnosis.counterfactual import (
    PositionLimitRetry,
    RetryFromEarlierPositions,
    RetryWithJointsUnwound,
    StallCounterfactual,
    retry_task_with_chain_from_world_root,
)
from coraplex.plans.failure_diagnosis.exceptions import (
    FailureDiagnosis,
    NodeFailedByACondition,
    NodeFailedOnItsOwn,
    StatementRanOutOfCandidates,
    TaskCollided,
    TaskStalled,
)
from coraplex.plans.failures import (
    EmptyUnderspecified,
    MotionMadeNoProgress,
    MotionViolatedCollisionAvoidance,
)
from coraplex.plans.underspecified import (
    ActionTrial,
    FailedTrial,
    UnderspecifiedChildChooser,
)
from cramph.causal_edges import ConditionEdge, CausalEdge, WorldStateEdge
from cramph.composites import ChildChooserAccess, StatementNode
from cramph.data_types import TransitionKind
from cramph.node import StatechartNode
from cramph.node_runs import StatechartRuns
from cramph.run_start_positions import RunStartPositions
from cramph.state_history import NodeRun
from cramph.statechart import Statechart
from giskardpy.motion_statechart.conflict_edges import ConflictEdge
from giskardpy.motion_statechart.exceptions import PositionLimitBlockedError
from giskardpy.motion_statechart.monitors.progress_monitors import BlockedPositionLimit
from krrood.patterns.called_positionally import called_positionally
from semantic_digital_twin.world_description.degree_of_freedom import DegreeOfFreedom

# %% diagnosing


@dataclass(kw_only=True)
class FailureDiagnoser:
    """
    Diagnoses failures of a statechart from its causal graph.
    """

    causal_graph: RunCausalGraph
    """
    The causal graph of the statechart.
    """

    action_trial: ActionTrial | None = None
    """
    The trial the plan tries its candidates in, which also tries the counterfactuals of
    a diagnosis; None if the plan has none.
    """

    position_limit_retries: list[PositionLimitRetry] = field(default_factory=list)
    """
    The ways to retry a task stalled at position limits, tried in this order until one
    reaches its goal.
    """

    stalled_nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The nodes a motion failure reported as no longer approaching their goal.
    """

    colliding_nodes: list[StatechartNode] = field(default_factory=list, init=False)
    """
    The nodes moving bodies a motion failure reported as too close to each other.
    """

    colliding_degrees_of_freedom: frozenset[DegreeOfFreedom] = field(
        default=frozenset(), init=False
    )
    """
    The degrees of freedom that moved the bodies a motion failure reported as too close
    to each other.
    """

    blocked_position_limits: list[BlockedPositionLimit] = field(
        default_factory=list, init=False
    )
    """
    The position limits a motion failure reported as blocking the stalled nodes.
    """

    diagnosis_kinds: list[type[FailureDiagnosis]] = field(
        default_factory=lambda: [
            TaskStalled,
            TaskCollided,
            StatementRanOutOfCandidates,
            NodeFailedByACondition,
        ]
    )
    """
    The ways an origin can fail, tried in order; an origin that fails none of these ways
    failed on its own, see :class:`NodeFailedOnItsOwn`.
    """

    @classmethod
    def from_statechart(cls, *, statechart: Statechart) -> FailureDiagnoser:
        """
        :param statechart: A statechart that ran, at least in part.
        :return: A diagnoser over its causal graph, trying counterfactuals in the trial
            its plan tries candidates in, from where its runs found what they move.
        """
        access = statechart.context.get_extension(extension_type=ChildChooserAccess)
        choosers = [] if access is None else access.choosers
        trial = next(
            (
                chooser.trial
                for chooser in choosers
                if isinstance(chooser, UnderspecifiedChildChooser)
            ),
            None,
        )
        causal_graph = RunCausalGraph.from_statechart(statechart=statechart)
        retries: list[PositionLimitRetry] = [
            RetryWithJointsUnwound(world=statechart.context.world)
        ]
        run_start_positions = statechart.context.get_extension(
            extension_type=RunStartPositions
        )
        if run_start_positions is not None:
            retries.append(
                RetryFromEarlierPositions(
                    world_changes=causal_graph.world_changes,
                    run_start_positions=run_start_positions,
                )
            )
        return cls(
            causal_graph=causal_graph,
            action_trial=trial,
            position_limit_retries=retries,
        )

    @property
    def runs(self) -> StatechartRuns:
        """
        :return: The runs of the statechart.
        """
        return self.causal_graph.runs

    # %% where a failure is reported

    def diagnose_failure(self, *, failure: Exception | None) -> list[FailureDiagnosis]:
        """
        Forgets what an earlier failure reported before reading `failure`.

        :param failure: What the plan failed with, None if it ended without an
            exception.
        :return: One diagnosis per run the failure is reported for.
        """
        self.stalled_nodes = []
        self.colliding_nodes = []
        self.colliding_degrees_of_freedom = frozenset()
        self.blocked_position_limits = []
        return self.diagnose_reported_runs(failure)

    @singledispatchmethod
    @called_positionally
    def diagnose_reported_runs(
        self, failure: Exception | None
    ) -> list[FailureDiagnosis]:
        """
        :param failure: What the plan failed with, None if it ended without an
            exception.
        :return: One diagnosis per run the failure is reported for.
        """
        return [
            self.diagnose_run(reported_run=run) for run in self.innermost_failed_runs()
        ]

    @diagnose_reported_runs.register
    @called_positionally
    def _(self, failure: MotionMadeNoProgress) -> list[FailureDiagnosis]:
        """
        :return: One diagnosis per task the failure reports as no longer approaching
            its goal.
        """
        self.stalled_nodes = list(failure.no_progress.stalled_tasks) or [
            failure.no_progress.progress_monitor.monitored_node
        ]
        if isinstance(failure.no_progress, PositionLimitBlockedError):
            self.blocked_position_limits = failure.no_progress.blocked_position_limits
        return [
            self.diagnose_run(reported_run=self.runs.latest_run_of_node(node=node))
            for node in self.stalled_nodes
        ]

    @diagnose_reported_runs.register
    @called_positionally
    def _(self, failure: MotionViolatedCollisionAvoidance) -> list[FailureDiagnosis]:
        """
        :return: One diagnosis per run moving a body the failure reports as too close
            to another.
        """
        self.colliding_degrees_of_freedom = (
            failure.degrees_of_freedom_moving_colliding_bodies
        )
        self.colliding_nodes = [
            run.node
            for run in self.active_runs_moving(
                degrees_of_freedom=self.colliding_degrees_of_freedom
            )
        ]
        return [
            self.diagnose_run(reported_run=self.runs.latest_run_of_node(node=node))
            for node in self.colliding_nodes
        ]

    @diagnose_reported_runs.register
    @called_positionally
    def _(self, failure: EmptyUnderspecified) -> list[FailureDiagnosis]:
        """
        :return: The diagnosis of the statement that ran out of candidates.
        """
        return [
            self.diagnose_run(
                reported_run=self.runs.latest_run_of_node(node=failure.node)
            )
        ]

    def diagnose_node(self, *, node: StatechartNode) -> FailureDiagnosis:
        """
        :param node: A node whose latest run failed.
        :return: Why it failed.
        """
        return self.diagnose_run(reported_run=self.runs.latest_run_of_node(node=node))

    def innermost_failed_runs(self) -> list[NodeRun]:
        """
        :return: The failed runs no failed run of a node they hold explains and no later
            run of their node replaced, the earliest first.
        """
        failed = [
            run
            for run in self.runs.all_runs
            if run.failed and run == self.runs.latest_run_of_node(node=run.node)
        ]
        return sorted(
            (run for run in failed if not self.failed_runs_below(run=run)),
            key=lambda run: run.last_tick,
        )

    def failed_runs_below(self, *, run: NodeRun) -> list[NodeRun]:
        """
        :param run: A run.
        :return: The failed runs of nodes below `run`'s node within it, the earliest
            first.
        """
        descendants = set(run.node.descendants)
        return sorted(
            (
                other
                for other in self.runs.all_runs
                if other.failed
                and other.node in descendants
                and run.contains(other=other)
            ),
            key=lambda other: other.last_tick,
        )

    def active_runs_moving(
        self, *, degrees_of_freedom: frozenset[DegreeOfFreedom]
    ) -> list[NodeRun]:
        """
        :param degrees_of_freedom: The degrees of freedom that moved bodies that came
            too close to each other.
        :return: The runs active at the newest tick that move one of them, or every
            active run that moves anything if none does.
        """
        active = [
            run
            for run in self.runs.all_runs
            if run.last_tick >= self.runs.newest_tick
            and self.causal_graph.world_access_of_nodes[
                run.node
            ].moved_degrees_of_freedom
        ]
        moving = [
            run
            for run in active
            if self.causal_graph.world_access_of_nodes[
                run.node
            ].moved_degrees_of_freedom
            & degrees_of_freedom
        ]
        return moving or active

    # %% from the report to the cause

    def failure_origin_of(self, *, run: NodeRun) -> NodeRun:
        """
        :param run: A run the failure is reported for.
        :return: The run where the failure started: the earliest failed run below it
            that no failed run below that one explains, `run` itself if there is none.
        """
        if isinstance(run.node, StatementNode):
            return run
        inside = self.failed_runs_below(run=run)
        if not inside:
            return run
        return self.failure_origin_of(run=inside[0])

    def diagnose_run(self, *, reported_run: NodeRun) -> FailureDiagnosis:
        """
        :param reported_run: A run the failure is reported for.
        :return: Why it failed, by the first of :attr:`diagnosis_kinds` its origin
            failed in.
        """
        origin_run = self.failure_origin_of(run=reported_run)
        for kind in self.diagnosis_kinds:
            diagnosis = kind.diagnose_if_failed_this_way(
                reported_run=reported_run, origin_run=origin_run, diagnoser=self
            )
            if diagnosis is not None:
                return diagnosis
        return NodeFailedOnItsOwn.diagnose_if_failed_this_way(
            reported_run=reported_run, origin_run=origin_run, diagnoser=self
        )

    def edges_from_conditions_that_ended(self, *, run: NodeRun) -> list[ConditionEdge]:
        """
        :param run: A run.
        :return: The control edges into `run` from its fail and interrupt conditions
            that held on the tick it ended.
        """
        if run.end_tick is None:
            return []
        fired = {
            condition.kind
            for condition in run.node.conditions
            if condition.kind in (TransitionKind.FAIL, TransitionKind.INTERRUPT)
            and condition.was_true_at_tick(
                history=self.runs.statechart.state_history, tick=run.end_tick
            )
        }
        return [
            edge
            for edge in self.causal_graph.edges_into_run(run=run)
            if isinstance(edge, ConditionEdge) and edge.transition_kind in fired
        ]

    # %% the causes of each kind of failure

    def find_causes_of_stall(
        self, *, stalled: NodeRun
    ) -> tuple[list[RootCause], StallCounterfactual | None]:
        """
        A task stalls when another task drives its degrees of freedom at once, when an
        earlier run left a joint it moves at a position limit, or when an earlier run
        left degrees of freedom it cannot move where they are. The latter two hold only
        if the task, retried with that joint moved back or free to move them, succeeds.

        :param stalled: The run that stopped approaching its goal.
        :return: The runs that made it stall, the most likely first, and what trying it
            free of what an earlier run left showed, None if that was not asked.
        """
        edges = self.causal_graph.edges_into_run(run=stalled)
        access = self.causal_graph.world_access_of_nodes[stalled.node]
        conflicts = [
            edge
            for edge in edges
            if isinstance(edge, ConflictEdge)
            and edge.degrees_of_freedom & access.moved_degrees_of_freedom
        ]
        if conflicts:
            return [
                RootCause(causing_run=edge.cause, edges_to_origin=(edge,))
                for edge in conflicts
            ], None
        blocked = [
            limit
            for limit in self.blocked_position_limits
            if limit.task is stalled.node
        ]
        blocked_degrees_of_freedom = frozenset(
            limit.degree_of_freedom for limit in blocked
        )
        limit_counterfactual = None
        for retry in self.position_limit_retries:
            counterfactual = retry.try_in(
                trial=self.action_trial,
                stalled=stalled,
                blocked_position_limits=blocked,
            )
            if counterfactual is None:
                continue
            if counterfactual == retry.reached:
                return self.last_plan_runs_moving(
                    degrees_of_freedom=blocked_degrees_of_freedom, statement=stalled
                ), counterfactual
            limit_counterfactual = counterfactual
        unmovable = (
            access.read_degrees_of_freedom - access.moved_or_held_degrees_of_freedom
        )
        left_by_earlier = [
            RootCause(causing_run=edge.cause, edges_to_origin=(edge,))
            for edge in self.sorted_by_latest_cause(edges=edges)
            if isinstance(edge, WorldStateEdge) and edge.degrees_of_freedom & unmovable
        ]
        if not left_by_earlier:
            return [], limit_counterfactual
        counterfactual = retry_task_with_chain_from_world_root(
            task=stalled.node, trial=self.action_trial
        )
        if counterfactual == StallCounterfactual.STILL_STALLED:
            return [], counterfactual
        return left_by_earlier, counterfactual

    def find_causes_of_collision(self, *, colliding: NodeRun) -> list[RootCause]:
        """
        :param colliding: A run moving a body that came too close to another.
        :return: The last runs before it that moved a degree of freedom moving one of
            the bodies that came too close, among those it does not drive itself.
        """
        return self.last_plan_runs_moving(
            degrees_of_freedom=self.colliding_degrees_of_freedom
            - self.causal_graph.world_access_of_nodes[
                colliding.node
            ].moved_or_held_degrees_of_freedom,
            statement=colliding,
        )

    def find_causes_of_observation(self, *, run: NodeRun) -> list[RootCause]:
        """
        :param run: A run that failed on what it observed or aimed for.
        :return: The runs that left what it read where it was, the latest first.
        """
        return [
            RootCause(causing_run=edge.cause, edges_to_origin=(edge,))
            for edge in self.sorted_by_latest_cause(
                edges=self.causal_graph.edges_into_run(run=run)
            )
            if isinstance(edge, WorldStateEdge)
        ]

    def find_causes_of_failing_condition(self, *, run: NodeRun) -> list[RootCause]:
        """
        :param run: A run a fail or interrupt condition ended.
        :return: The runs that caused what the condition read, through the nodes it
            read; a read node that nothing else explains is the cause itself.
        """
        root_causes = []
        for edge in self.edges_from_conditions_that_ended(run=run):
            upstream = self.find_causes_of_observation(run=edge.cause)
            if not upstream:
                root_causes.append(
                    RootCause(causing_run=edge.cause, edges_to_origin=(edge,))
                )
                continue
            root_causes.extend(
                RootCause(
                    causing_run=root.causing_run,
                    edges_to_origin=(*root.edges_to_origin, edge),
                )
                for root in upstream
            )
        return root_causes

    def find_causes_shared_by_candidates(
        self, *, statement: NodeRun
    ) -> list[RootCause]:
        """
        :param statement: A run of a statement that ran out of candidates.
        :return: The causes every failed candidate shares, whether it failed in the
            plan or in its trial, or the causes of any of them if they share none.
        """
        per_candidate = [
            *(
                self.diagnose_run(reported_run=child).root_causes
                for child in self.runs.all_runs
                if child.failed
                and child.node.parent_node is statement.node
                and statement.contains(other=child)
            ),
            *(
                self.find_causes_of_trial(
                    failed_trial=failed_trial, statement=statement
                )
                for failed_trial in self.failed_trials_during(statement=statement)
            ),
        ]
        nodes_per_candidate = [
            {root.causing_run.node for root in roots} for roots in per_candidate
        ]
        shared = (
            set.intersection(*nodes_per_candidate) if nodes_per_candidate else set()
        )
        roots = [root for roots in per_candidate for root in roots]
        kept = (
            [root for root in roots if root.causing_run.node in shared]
            if shared
            else roots
        )
        return list({root.causing_run: root for root in kept}.values())

    def failed_trials_during(self, *, statement: NodeRun) -> list[FailedTrial]:
        """
        :param statement: A run of a statement.
        :return: The failed trials that tried candidates of the statement during this
            run.
        """
        if self.action_trial is None:
            return []
        return [
            failed_trial
            for failed_trial in self.action_trial.failed_trials
            if statement.node in failed_trial.tried_statements
            and statement.start_tick <= failed_trial.tick_in_plan <= statement.last_tick
        ]

    def find_causes_of_trial(
        self, *, failed_trial: FailedTrial, statement: NodeRun
    ) -> list[RootCause]:
        """
        A trial starts from the world the plan left, so what it read and no node of its
        own changed was left by the plan's runs before the statement. Only the recorded
        history of the trial is read; no trial runs again.

        :param failed_trial: A failed trial of a candidate of `statement`.
        :param statement: The run of the statement.
        :return: The runs of the plan that left what the trial read, for every failure
            of the trial no node of its own explains.
        """
        trial_diagnoser = FailureDiagnoser(
            causal_graph=RunCausalGraph.from_statechart(
                statechart=failed_trial.statechart
            )
        )
        root_causes = []
        for diagnosis in trial_diagnoser.diagnose_failure(failure=failed_trial.failure):
            if diagnosis.root_causes:
                continue
            access = trial_diagnoser.causal_graph.world_access_of_nodes[
                diagnosis.origin_run.node
            ]
            root_causes.extend(
                self.last_plan_runs_moving(
                    degrees_of_freedom=diagnosis.degrees_of_freedom_left_by_earlier_nodes(
                        access=access
                    ),
                    statement=statement,
                )
            )
        return root_causes

    def last_plan_runs_moving(
        self, *, degrees_of_freedom: frozenset[DegreeOfFreedom], statement: NodeRun
    ) -> list[RootCause]:
        """
        :param degrees_of_freedom: Degrees of freedom of a copy of the plan's world.
        :param statement: A run of the plan.
        :return: The last runs of the plan before `statement` that moved them, matched
            by their ids.
        """
        changes = self.causal_graph.world_changes
        left_by: dict[NodeRun, set[DegreeOfFreedom]] = {}
        for degree_of_freedom in degrees_of_freedom:
            mover = changes.last_run_that_moved(
                degree_of_freedom_id=degree_of_freedom.id, reader=statement
            )
            if mover is not None:
                left_by.setdefault(mover, set()).add(degree_of_freedom)
        return [
            RootCause(
                causing_run=mover,
                edges_to_origin=(
                    WorldStateEdge(
                        cause=mover,
                        effect=statement,
                        degrees_of_freedom=frozenset(moved),
                    ),
                ),
            )
            for mover, moved in left_by.items()
        ]

    @staticmethod
    def sorted_by_latest_cause(*, edges: list[CausalEdge]) -> list[CausalEdge]:
        """
        :return: `edges`, those from the latest cause first.
        """
        return sorted(edges, key=lambda edge: edge.cause.last_tick, reverse=True)
