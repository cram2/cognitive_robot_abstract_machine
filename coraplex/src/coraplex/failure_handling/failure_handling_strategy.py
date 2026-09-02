from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field

from typing_extensions import ClassVar, Optional, TYPE_CHECKING, Type

from coraplex.datastructures.enums import TaskStatus
from coraplex.plans.factories import execute_single
from coraplex.plans.failures import PlanFailure

if TYPE_CHECKING:
    from coraplex.plans.plan_node import ActionLike, PlanNode, UnderspecifiedNode

# %% resolutions


@dataclass
class FailureResolution(ABC):
    """
    The decision a :class:`FailureHandlingStrategy` made about how plan execution
    continues after a failure.

    A resolution interprets itself at the node that is handling the failure:
    :meth:`apply` *returning* means "the failure was dealt with here", :meth:`apply`
    *escalating* hands it to the node above, which applies the resolution again. The
    handling node never branches on resolution or node types.
    """

    failure: PlanFailure
    """
    The refined failure this resolution resolves.
    """

    @abstractmethod
    def apply(self, node: PlanNode) -> None:
        """
        Interpret this resolution at the given node.

        Returning deals with the failure there; escalating the carried failure hands it
        to the node above, which applies this resolution again.

        :param node: The node that is applying this resolution.
        """

    def propagate(self, node: PlanNode) -> None:
        """
        Record the carried failure on the node and escalate it along the plan tree.

        The carried failure keeps a reference to this resolution, so the node it is
        escalated to applies the already decided resolution instead of consulting the
        handler again.

        :param node: The node the failure passes through.
        :raises PlanFailure: Once escalation reaches the root of the plan.
        """
        node.status = TaskStatus.FAILED
        node.reason = self.failure
        self.failure.resolution = self
        node.escalate(self.failure)


@dataclass
class Propagate(FailureResolution):
    """
    Give up on handling: the carried failure escalates through every enclosing node and
    finally out of the plan.
    """

    def apply(self, node: PlanNode) -> None:
        self.propagate(node)


@dataclass
class TargetedResolution(FailureResolution):
    """
    A resolution that re-runs one specific node: it escalates through every node below
    the target and returns once the target applies it.
    """

    target_node: PlanNode
    """
    The node whose work is run again.
    """

    def apply(self, node: PlanNode) -> None:
        if node is self.target_node:
            self.failure.resolution = None
            return
        self.propagate(node)


@dataclass
class RetryNode(TargetedResolution):
    """
    Run the target's work again as it is, typically after a recovery sub-plan repaired
    the situation the failure described.
    """


@dataclass
class Reparameterize(TargetedResolution):
    """
    Run an enclosing underspecified node again, which advances it to its next action
    candidate.
    """

    target_node: UnderspecifiedNode
    """
    The underspecified node that generates a fresh action candidate when it runs again.
    """


# %% strategies


@dataclass
class FailureHandlingStrategy(ABC):
    """
    Decides how plan execution continues after a refined failure.

    A strategy declares the failure type it handles; the
    :class:`~coraplex.failure_handling.failure_handler.FailureHandler` selects the most
    specific applicable strategy. Attempt bookkeeping (for example maximum retries)
    lives in strategy instances.
    """

    handled_failure_type: ClassVar[Type[PlanFailure]] = PlanFailure
    """
    The failure type this strategy resolves.
    """

    def applies(self, failure: PlanFailure) -> bool:
        """
        :param failure: The refined failure to check.
        :return: Whether this strategy can resolve the failure.
        """
        return isinstance(failure, self.handled_failure_type)

    def retried_node(self, failure: PlanFailure) -> PlanNode:
        """
        :param failure: The refined failure to find a retry target for.
        :return: The node a targeted resolution runs again: the action the failure
            happened in, or the failing node itself when no action encloses it.

        A single motion of an action is rarely meaningful on its own, so the whole
        action is repeated rather than the node that happened to raise.
        """
        return failure.action_node or failure.node

    @abstractmethod
    def resolve(self, failure: PlanFailure) -> FailureResolution:
        """
        Decide how execution continues after the failure.

        :param failure: The refined failure to resolve.
        :return: The resolution the handling nodes apply along the plan tree.
        """


# %% recovery-plan strategies


@dataclass
class RecoveryPlanStrategy(FailureHandlingStrategy, ABC):
    """
    A strategy that recovers by performing real robot actions before execution
    continues.

    The recovery sub-plan runs as a separate plan sharing the failing plan's
    :class:`~coraplex.datastructures.dataclasses.Context` (same world and robot).

    ..note:: Recording the recovery sub-plan inside the failing plan's tree (via
        :meth:`~coraplex.plans.plan_node.PlanNode.mount_subplan`) is follow-up work.
    """

    _recovering: bool = field(init=False, default=False)
    """
    Whether this strategy is currently performing its recovery plan, which guards
    against recovering from a failure of the recovery itself.
    """

    @abstractmethod
    def recovery_plan(self, failure: PlanFailure) -> Optional[ActionLike]:
        """
        Build the recovery sub-plan for the failure.

        :param failure: The refined failure to recover from.
        :return: The plan to perform before execution continues, or None if no recovery
            is possible.
        """

    @abstractmethod
    def resolution_after_recovery(self, failure: PlanFailure) -> FailureResolution:
        """
        :param failure: The refined failure the recovery plan just repaired.
        :return: The resolution applied after the recovery plan succeeded, typically a
            :class:`RetryNode` targeting the failing action.
        """

    def resolve(self, failure: PlanFailure) -> FailureResolution:
        if self._recovering:
            return Propagate(failure=failure)
        recovery_plan = self.recovery_plan(failure)
        if recovery_plan is None:
            return Propagate(failure=failure)
        return self.perform_recovery(recovery_plan, failure)

    def perform_recovery(
        self, recovery_plan: ActionLike, failure: PlanFailure
    ) -> FailureResolution:
        """
        Perform the recovery plan in the failing plan's context.

        The recovery plan runs as a separate plan; the context is handed back to the
        failing plan afterwards. A failure of the recovery itself is linked to the
        propagated failure as its cause.

        :param recovery_plan: The plan to perform before execution continues.
        :param failure: The refined failure the recovery plan repairs.
        :return: The follow-up resolution, or a :class:`Propagate` of the original
            failure when the recovery failed.
        """
        context = failure.context
        failing_plan = context.plan
        recovery_root = execute_single(recovery_plan, context=context)
        self._recovering = True
        try:
            recovery_root.perform()
        except PlanFailure as recovery_failure:
            failure.__cause__ = recovery_failure
            return Propagate(failure=failure)
        finally:
            self._recovering = False
            context.plan = failing_plan
        return self.resolution_after_recovery(failure)
