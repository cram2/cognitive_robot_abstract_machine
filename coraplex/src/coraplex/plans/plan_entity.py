from __future__ import annotations

from dataclasses import dataclass, field
from typing_extensions import TYPE_CHECKING

if TYPE_CHECKING:
    from coraplex.plans.plan import Plan


@dataclass
class PlanEntity:
    """
    A base class for entities that are managed by a plan.
    """

    plan: Plan | None = field(kw_only=True, default=None)
