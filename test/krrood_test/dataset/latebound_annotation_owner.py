from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .latebound_annotation_type import LateBoundAnnotationType


@dataclass
class OwnerWithLateBoundAnnotation:
    """A dataclass annotated with a ``TYPE_CHECKING``-only type from another module."""

    value: Optional[LateBoundAnnotationType] = field(default=None)
