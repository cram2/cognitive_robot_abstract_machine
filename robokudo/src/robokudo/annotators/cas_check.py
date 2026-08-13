"""CAS condition checking utilities.

This module provides annotators for checking conditions in the Common Analysis System (CAS).
The annotators follow these principles:

* Return SUCCESS if condition is true
* Return FAILURE if condition is false
* Never return RUNNING status

.. note::
   For RUNNING status checks, use CASCondition from cas_condition.py instead.
"""

import functools

from py_trees.common import Status
from typing_extensions import Callable, Type, Optional

from robokudo.annotators.core import BaseAnnotator
from robokudo.cas import CAS
from robokudo.types.scene import ObjectHypothesis


class CASCheckFunc(BaseAnnotator):
    """Function-based CAS condition checker.

    Evaluates a given function that checks conditions in the CAS and returns:

    * SUCCESS if function returns True
    * FAILURE if function returns False

    .. warning::
       Will raise an exception if initialized without a function.
    """

    def __init__(
        self,
        name: str = "CASCheckFunc",
        func: Optional[Callable[[CAS], bool]] = None,
        raise_with_str: str = "",
    ) -> None:
        """Initialize the CAS condition checker.

        :param name: Name of this node in the behavior tree, defaults to "CASCheckFunc"
        :param func: Function that evaluates CAS conditions, must return bool, defaults to None
        :param raise_with_str: Error message to raise on failure, empty string disables raising, defaults to ""
        :raises Exception: If func is None
        """
        super(CASCheckFunc, self).__init__(name=name)
        if func is None:
            raise Exception(
                "CASCheckFunc needs a function to work properly. Please pass 'func'."
            )
        self.func = func
        self.raise_with_str = raise_with_str

    def update(self) -> Status:
        """Check the CAS condition.

        :return: SUCCESS if condition is True, FAILURE if False
        :raises Exception: If raise_with_str is set and condition is False
        """
        cas = self.get_cas()
        if self.func(cas):
            return Status.SUCCESS
        else:
            if self.raise_with_str != "":
                raise Exception(self.raise_with_str)
            else:
                return Status.FAILURE


def any_of_type_present(annotation_type: Type, cas: CAS) -> bool:
    """Check if any annotations of a specific type exist in the CAS.

    :param annotation_type: Type of annotation to check for
    :param cas: Common Analysis System instance
    :return: True if at least one annotation exists, False otherwise
    """
    annotations = cas.filter_annotations_by_type(annotation_type)
    return len(annotations) > 0


class CASCheckAnnotationTypeExists(CASCheckFunc):
    """Check for existence of specific annotation types.

    Specialized CASCheckFunc that:

    * Checks for presence of specific annotation types
    * Returns SUCCESS if at least one annotation exists
    * Returns FAILURE if no annotations exist
    """

    def __init__(
        self,
        name: str = "CASCheckAnnotationTypeExists",
        annotation_type: Optional[Type] = None,
        raise_with_str: str = "",
    ) -> None:
        """Initialize annotation type checker.

        :param name: Name of this node in the behavior tree, defaults to "CASCheckAnnotationTypeExists"
        :param annotation_type: Type of annotation to check for, defaults to None
        :param raise_with_str: Error message to raise on failure, empty string disables raising, defaults to ""
        """
        func = functools.partial(any_of_type_present, annotation_type)
        super(CASCheckAnnotationTypeExists, self).__init__(
            name=name, func=func, raise_with_str=raise_with_str
        )


class CASCheckOHExists(CASCheckAnnotationTypeExists):
    """Check for existence of ObjectHypothesis annotations.

    Specialized CASCheckAnnotationTypeExists that:

    * Specifically checks for ObjectHypothesis annotations
    * Returns SUCCESS if any ObjectHypothesis exists
    * Returns FAILURE if no ObjectHypothesis exists
    """

    def __init__(
        self, name: str = "CASCheckOHExists", raise_with_str: str = ""
    ) -> None:
        """Initialize ObjectHypothesis checker.

        :param name: Name of this node in the behavior tree, defaults to "CASCheckOHExists"
        :param raise_with_str: Error message to raise on failure, empty string disables raising, defaults to ""
        """
        super(CASCheckOHExists, self).__init__(
            name=name, annotation_type=ObjectHypothesis, raise_with_str=raise_with_str
        )
