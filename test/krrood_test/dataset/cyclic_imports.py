from __future__ import annotations

from typing_extensions import Optional, TYPE_CHECKING
from dataclasses import dataclass, field

from krrood.symbol_graph.symbol_graph import Symbol

if TYPE_CHECKING:
    from .example_classes import Pose


@dataclass
class PoseAnnotation(Symbol):
    name: str
    pose: Optional["Pose"] = field(default=None, repr=False, kw_only=True)
