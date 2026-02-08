from __future__ import annotations

from typing_extensions import Optional, Union, Type, Iterable, Callable, Any

from .result_quantification_constraint import (
    ResultQuantificationConstraint,
)
from .symbolic import (
    An,
    The,
    SetOf,
    Entity,
    Selectable,
    Max,
    Min,
    Sum,
    Average,
    Count,
    ResultProcessor,
)
from .utils import T


def an(
    entity_: Union[SetOf[T], Entity[T], T, Iterable[T], Type[T]],
    quantification: Optional[ResultQuantificationConstraint] = None,
) -> Union[An[T], T]:
    """
    Select all values satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :param quantification: Optional quantification constraint.
    :return: A quantifier representing "an" element.
    :rtype: An[T]
    """
    return _apply_result_processor(
        An, entity_, _quantification_constraint_=quantification
    )


a = an
"""
This is an alias to accommodate for words not starting with vowels.
"""


def the(
    entity_: Union[SetOf[T], Entity[T], T, Iterable[T], Type[T]],
) -> Union[The[T], T]:
    """
    Select the unique value satisfying the given entity description.

    :param entity_: An entity or a set expression to quantify over.
    :return: A quantifier representing "an" element.
    :rtype: The[T]
    """
    return _apply_result_processor(The, entity_)


def max(
    variable: Selectable[T],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Max[T]]:
    """
    Maps the variable values to their maximum value.

    :param variable: The variable for which the maximum value is to be found.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Max object that can be evaluated to find the maximum value.
    """
    return _apply_result_processor(
        Max, variable, _key_func_=key, _default_value_=default, _distinct_=distinct
    )


def min(
    variable: Selectable[T],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Min[T]]:
    """
    Maps the variable values to their minimum value.

    :param variable: The variable for which the minimum value is to be found.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Min object that can be evaluated to find the minimum value.
    """
    return _apply_result_processor(
        Min, variable, _key_func_=key, _default_value_=default, _distinct_=distinct
    )


def sum(
    variable: Union[T, Selectable[T]],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Sum[T]]:
    """
    Computes the sum of values produced by the given variable.

    :param variable: The variable for which the sum is calculated.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Sum object that can be evaluated to find the sum of values.
    """
    return _apply_result_processor(
        Sum, variable, _key_func_=key, _default_value_=default, _distinct_=distinct
    )


def average(
    variable: Union[Selectable[T], Any],
    key: Optional[Callable] = None,
    default: Optional[T] = None,
    distinct: bool = False,
) -> Union[T, Average[T]]:
    """
    Computes the sum of values produced by the given variable.

    :param variable: The variable for which the sum is calculated.
    :param key: A function that extracts a comparison key from each variable value.
    :param default: The value returned when the iterable is empty.
    :param distinct: Whether to only consider distinct values.
    :return: A Sum object that can be evaluated to find the sum of values.
    """
    return _apply_result_processor(
        Average, variable, _key_func_=key, _default_value_=default, _distinct_=distinct
    )


def count(
    variable: Optional[Selectable[T]] = None, distinct: bool = False
) -> Union[T, Count[T]]:
    """
    Count the number of values produced by the given variable.

    :param variable: The variable for which the count is calculated, if not given, the count of all results (by group)
     is returned.
    :param distinct: Whether to only consider distinct values.
    :return: A Count object that can be evaluated to count the number of values.
    """
    return _apply_result_processor(Count, variable, _distinct_=distinct)


def _apply_result_processor(
    result_processor: Type[ResultProcessor[T]],
    variable: Optional[Selectable[T]] = None,
    **result_processor_kwargs,
) -> Union[T, ResultProcessor[T]]:
    """
    Applies the result processor to the given variable.

    :param result_processor: The result processor to apply to the variable.
    :param variable: The variable for which the result processor is applied.
    :param **result_processor_kwargs: The keyword arguments for the result processor.
    :return: The result processor instance.
    """
    return result_processor(_child_=variable, **result_processor_kwargs)
