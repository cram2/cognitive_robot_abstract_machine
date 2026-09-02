from __future__ import annotations

from dataclasses import dataclass, field
from threading import Lock

from typing_extensions import Dict, TYPE_CHECKING

if TYPE_CHECKING:
    from coraplex.plans.plan_node import PlanNode

# %% attempt bookkeeping


@dataclass
class AttemptBudget:
    """
    How often a strategy may still run one node's work again.

    :meth:`~coraplex.plans.plan_node.PlanNode.perform` retries as long as a resolution
    reaches its target, so a strategy that keeps returning a targeted resolution never
    terminates against a deterministic failure. Counting per node keeps a plan that
    fails in several places recoverable everywhere.
    """

    maximum_attempts: int = 3
    """
    How many attempts each node is granted.
    """

    _granted_by_node: Dict[PlanNode, int] = field(default_factory=dict, init=False)
    """
    How many attempts each node has been granted so far, keyed by node identity.
    """

    _lock: Lock = field(default_factory=Lock, init=False)
    """
    Guards the bookkeeping, because the children of a parallel node consult the same
    strategy instance from their worker threads.
    
    TODO: Look if this is still relevant since parallel should be pardes to giskard and not worker threads
    """

    def grant(self, node: PlanNode) -> bool:
        """
        Consume one of the node's attempts.

        :param node: The node whose work would be run again.
        :return: Whether the node may run again.
        """
        with self._lock:
            granted = self._granted_by_node.get(node, 0)
            if granted >= self.maximum_attempts:
                return False
            self._granted_by_node[node] = granted + 1
            return True
