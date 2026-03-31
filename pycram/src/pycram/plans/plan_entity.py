from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from pycram.plans.plan import Plan


@dataclass
class PlanEntity:
    """
    A base class for entities that are managed by a plan.
    """

    plan: Optional[Plan] = field(kw_only=True, default=None)
