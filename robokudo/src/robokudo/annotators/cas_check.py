"""
CAS condition checking utilities.

This module provides annotators for checking conditions in the Common Analysis System (CAS).
The annotators follow these principles:

* Return SUCCESS if condition is true
* Return FAILURE if condition is false
* Never return RUNNING status

.. note::
   For RUNNING status checks, use CASCondition from cas_condition.py instead.
"""

from __future__ import annotations

import functools

# %% Imports
from dataclasses import dataclass, field

from py_trees.common import Status
from typing_extensions import Callable

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS
from robokudo.exceptions import CASCheckConfigurationError, CASCheckFailed
from robokudo.types.scene import ObjectHypothesis


@dataclass(kw_only=True, eq=False)
class CASCheckFunction(BaseAnnotator):
    """
    Function-based CAS condition checker.

    Evaluates a given function that checks conditions in the CAS and returns:

    * SUCCESS if function returns True
    * FAILURE if function returns False

    .. warning::
       Will raise an exception if initialized without a function.
    """

    function: Callable[[CAS], bool] | None = None
    """
    Condition evaluated against the current CAS.
    """

    raise_with_str: str = ""
    """
    Failure message; an empty value returns a failure status instead.
    """

    def __post_init__(self) -> None:
        """
        Initialize annotator resources after
        :meth:`robokudo.annotators.core.BaseAnnotator.__post_init__`.
        """
        super().__post_init__()
        if self.function is None:
            raise CASCheckConfigurationError(component_name=self.name)

    def update(self) -> Status:
        """
        Check the CAS condition.

        :return: SUCCESS if condition is True, FAILURE if False
        :raises CASCheckFailed: If raise_with_str is set and condition is False
        """
        cas = self.get_cas()
        if self.function(cas):
            return Status.SUCCESS
        else:
            if self.raise_with_str != "":
                raise CASCheckFailed(reason=self.raise_with_str)
            else:
                return Status.FAILURE


def any_of_type_present(annotation_type: type, cas: CAS) -> bool:
    """
    Check if any annotations of a specific type exist in the CAS.

    :param annotation_type: Type of annotation to check for
    :param cas: Common Analysis System instance
    :return: True if at least one annotation exists, False otherwise
    """
    annotations = cas.filter_annotations_by_type(annotation_type)
    return len(annotations) > 0


@dataclass(kw_only=True, eq=False)
class CASCheckAnnotationTypeExists(CASCheckFunction):
    """
    Check for existence of specific annotation types.

    Specialized CASCheckFunction that:

    * Checks for presence of specific annotation types
    * Returns SUCCESS if at least one annotation exists
    * Returns FAILURE if no annotations exist
    """

    annotation_type: type | None = None
    """
    Annotation type whose presence satisfies the condition.
    """

    function: Callable[[CAS], bool] | None = field(init=False, repr=False)
    """
    Condition specialized to the requested annotation type.
    """

    def __post_init__(self) -> None:
        """
        Specialize :meth:`CASCheckFunction.__post_init__` with the configured annotation
        condition.
        """
        self.function = functools.partial(any_of_type_present, self.annotation_type)
        super().__post_init__()


@dataclass(kw_only=True, eq=False)
class CASCheckObjectHypothesisExists(CASCheckAnnotationTypeExists):
    """
    Check for existence of ObjectHypothesis annotations.

    Specialized CASCheckAnnotationTypeExists that:

    * Specifically checks for ObjectHypothesis annotations
    * Returns SUCCESS if any ObjectHypothesis exists
    * Returns FAILURE if no ObjectHypothesis exists
    """

    annotation_type: type = field(default=ObjectHypothesis, init=False)
    """
    Object-hypothesis annotation required by this checker.
    """
