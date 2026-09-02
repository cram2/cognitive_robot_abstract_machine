from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan


@dataclass(eq=False)
class PlanEntity:
    """
    A base class for entities that are managed by a plan.

    ..note:: Entities are compared by identity. A generated equality would compare two
        entities by the plan that manages them, which makes every node of a plan equal
        to every other one while :class:`~coraplex.plans.plan_node.PlanNode` hashes by
        identity.
    """

    plan: Optional[Plan] = field(kw_only=True, default=None)
