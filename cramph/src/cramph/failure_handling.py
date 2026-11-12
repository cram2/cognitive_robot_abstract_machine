"""
Recovering a plan from the failure of one of its nodes while it runs: the strategies
that change the plan, and the nodes that run them once a node failed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from cramph.composites import CompositeNodeEndingItsChildren
from cramph.node import (
    NodeArtifacts,
    StatechartNode,
    SucceedsOnObservingTrue,
    expanded_child_field,
)
from krrood.symbolic_math.symbolic_math import (
    Scalar,
    logic_and,
    trinary_if_cases,
)

# %% failure handling strategies


@dataclass(kw_only=True)
class FailureHandlingStrategy(ABC):
    """
    Changes a plan so that it recovers from the failure of one of its nodes.

    .. note:: A strategy is sent through JSON together with the node holding it, so its
        private fields start anew on the receiver.
    """

    @abstractmethod
    def handle_failure(self, failed_node: StatechartNode) -> None:
        """
        Changes the plan so that it recovers from the failure of `failed_node`.

        Leaving that node failed means giving up on it.

        :param failed_node: The node that failed.
        """


# %% reacting to a failure


@dataclass(eq=False, repr=False, kw_only=True)
class FailureHandlingNode(SucceedsOnObservingTrue, StatechartNode):
    """
    Starts once :attr:`observed_node` failed and runs :attr:`failure_handling_strategy`
    on it, in a block that may rewrite compiled nodes, see
    :meth:`~cramph.statechart.Statechart.modify`.

    It succeeds on the tick after it started, which says the strategy ran and not that it
    recovered, and is ready again once :attr:`observed_node` was reset.

    Its conditions may only read a sibling, and control structure nodes react to a failed
    child on their own, so inside a plan use :class:`HandledOnFailure` instead.

    .. warning:: A strategy that changes the structure of the plan, for example by
        inserting nodes, only works in a locally ticked statechart.
    """

    observed_node: StatechartNode
    """
    The node whose failure starts this node.
    """

    failure_handling_strategy: FailureHandlingStrategy
    """
    Handles the failure of :attr:`observed_node` when this node starts.
    """

    def __post_init__(self) -> None:
        super().__post_init__()
        self.start_condition = self.observed_node.is_failed
        self.reset_condition = self.observed_node.is_not_started

    def build_artifacts(self) -> NodeArtifacts:
        return NodeArtifacts(observation=Scalar.const_true())

    def on_start(self) -> None:
        with self.statechart.modify(rewrites_compiled_nodes=True):
            self.failure_handling_strategy.handle_failure(self.observed_node)


@dataclass(repr=False, eq=False, kw_only=True)
class HandledOnFailure(CompositeNodeEndingItsChildren):
    """
    Runs a node next to the :class:`FailureHandlingNode` observing it, so that a failure
    of that node stays inside this node for as long as it is being recovered from.

    Its observation turns True once the node succeeded, and False once the node is still
    failed after the strategy ran.

    A node that cannot fail on its own is rejected, since it would never be handled.

    .. warning:: A strategy that changes the structure of the plan, for example by
        inserting nodes, only works in a locally ticked statechart.
    """

    handled_node: StatechartNode
    """
    The node to run and to recover when it fails.
    """

    failure_handling_strategy: FailureHandlingStrategy
    """
    See :attr:`FailureHandlingNode.failure_handling_strategy`.
    """

    failure_handler: FailureHandlingNode = expanded_child_field()
    """
    The node running :attr:`failure_handling_strategy`.
    """

    def expand(self) -> None:
        """
        Add the handled node, ended once it is at its goal, and the node observing it.
        """
        self._check_can_decide(self.handled_node)
        self._add_child_to_statechart(self.handled_node)
        self._succeed_at_its_goal(self.handled_node)
        self.failure_handler = FailureHandlingNode(
            name=f"{self.name}/failure handler",
            observed_node=self.handled_node,
            failure_handling_strategy=self.failure_handling_strategy,
        )
        self._add_child_to_statechart(self.failure_handler)

    def check_children(self) -> None:
        """
        Rejects a node that cannot fail, since it would never be handled.
        """
        self._check_child_can_fail(self.handled_node)

    def build_artifacts(self) -> NodeArtifacts:
        """
        Report success, giving up, or neither.

        A failed node alone decides nothing, since the strategy may still run it again.
        """
        return NodeArtifacts(
            observation=trinary_if_cases(
                cases=[
                    (self.handled_node.is_succeeded, Scalar.const_true()),
                    (
                        logic_and(
                            self.handled_node.is_failed,
                            self.failure_handler.is_terminated,
                        ),
                        Scalar.const_false(),
                    ),
                ],
                else_result=Scalar.const_trinary_unknown(),
            )
        )


# %% recovering


@dataclass(kw_only=True)
class LimitedRecovery(FailureHandlingStrategy, ABC):
    """
    A failure handling strategy that tries to recover at most :attr:`maximum_attempts`
    times and gives up after that.

    .. note:: The attempts are not counted anew when a node holding the failed node is
        reset.
    """

    maximum_attempts: int
    """
    How often this strategy tries to recover.
    """

    _attempts_made: int = field(default=0, init=False, repr=False)
    """
    How often this strategy tried to recover so far.
    """

    def handle_failure(self, failed_node: StatechartNode) -> None:
        if self._attempts_made >= self.maximum_attempts:
            return
        self._attempts_made += 1
        self.recover(failed_node)

    @abstractmethod
    def recover(self, failed_node: StatechartNode) -> None:
        """
        Changes the plan so that it recovers from the failure of `failed_node`.

        :param failed_node: The node that failed.
        """


@dataclass(kw_only=True)
class RetryFailedNode(LimitedRecovery):
    """
    Recovers by running the failed node again.
    """

    def recover(self, failed_node: StatechartNode) -> None:
        failed_node.statechart.request_reset(node=failed_node)
