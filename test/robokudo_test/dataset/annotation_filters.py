"""
Serializable predicates for annotator configuration checks.
"""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass

from robokudo.types.annotation import Classification
from robokudo.types.core import Annotation


# %% Classification filtering
@dataclass(kw_only=True)
class ClassificationFilter:
    """
    Accept annotations classifying an object with the configured label.
    """

    classname: str
    """
    Classification label accepted by this predicate.
    """

    def __call__(self, annotation: Annotation) -> bool:
        """
        Whether the annotation classifies an object with the selected label.
        """
        return (
            isinstance(annotation, Classification)
            and annotation.classname == self.classname
        )
