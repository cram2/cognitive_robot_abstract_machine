"""
A base class whose annotation is only importable under ``TYPE_CHECKING``.

Mirrors a root class every mapped dataclass inherits from, annotating a field with a
type its module imports only for type checking.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import TYPE_CHECKING, Optional

if TYPE_CHECKING:
    from .latebound_annotation_type import LateBoundAnnotationType


@dataclass
class BaseWithLateBoundAnnotation:
    """
    Root of the hierarchy that carries the late-bound annotation.
    """

    value: Optional[LateBoundAnnotationType] = None
    """
    The annotation whose name resolves only from this module's own imports.
    """
