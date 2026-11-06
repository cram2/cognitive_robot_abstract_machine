from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional
from uuid import UUID


@dataclass(kw_only=True)
class ClashingTarget:
    id: UUID


@dataclass(kw_only=True)
class ClashingEntity:
    clashing_target_id: UUID
    clashing_target: Optional[ClashingTarget] = field(default=None, repr=False)
