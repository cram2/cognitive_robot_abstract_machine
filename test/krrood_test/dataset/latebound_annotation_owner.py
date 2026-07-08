from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Optional, TYPE_CHECKING

if TYPE_CHECKING:
    from .latebound_annotation_type import LateBoundAnnotationType
    from .latebound_annotation_secondary_type import LateBoundAnnotationSecondaryType


@dataclass
class OwnerWithLateBoundAnnotation:
    """A dataclass annotated with ``TYPE_CHECKING``-only types from other modules."""

    value: Optional[LateBoundAnnotationType] = field(default=None)
    secondary_value: Optional[LateBoundAnnotationSecondaryType] = field(default=None)
