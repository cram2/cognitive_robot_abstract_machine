"""
Utilities for hashing, rendering, and general helpers used by the symbolic query engine.
"""

from __future__ import annotations

import inspect
import itertools
import re
from functools import lru_cache

from typing_extensions import (
    Set,
    Any,
    TypeVar,
    List,
    Dict,
    Callable,
    Iterator,
    Union,
    Type,
    Tuple,
    TYPE_CHECKING,
    Hashable,
    Iterable,
    Optional,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import (
        OperationResult,
        SymbolicExpression,
    )


class IDGenerator:
    """
    A class that generates incrementing, unique IDs and caches them for every object
    this is called on.
    """

    _counter = 0
    """
    The counter of the unique IDs.
    """

    def __call__(self, obj: Any) -> int:
        """
        Creates a unique ID and caches it for every object this is called on.

        :param obj: The object to generate a unique ID for, must be hashable.
        :return: The unique ID.
        """
        self._counter += 1
        return self._counter


def generate_combinations(
    generators_dict: Dict[str, Iterable[Any]],
) -> Iterator[Dict[str, Any]]:
    """
    Yield all combinations of generator values as keyword-argument dictionaries.

    :param generators_dict: A mapping of names to iterables of candidate values.
    :return: An iterator of dictionaries, each mapping names to one combination of
        values.
    """
    for combination in itertools.product(*generators_dict.values()):
        yield dict(zip(generators_dict.keys(), combination))


def make_list(value: Any) -> List:
    """
    Make a list from a value.

    :param value: The value to make a list from.
    :return: A list wrapping the value, or a one-element list if not iterable.
    """
    return list(value) if is_iterable(value) else [value]


def is_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable.

    :param obj: The object to check.
    :return:``True`` if the object is a non-string, non-type iterable, ``False``
        otherwise.
    """
    return callable(getattr(obj, "__iter__", None)) and not isinstance(
        obj, (str, type, bytes, bytearray)
    )


def make_tuple(value: Any) -> Any:
    """
    Make a tuple from a value.

    :param value: The value to make a tuple from.
    :return: A tuple wrapping the value, or a one-element tuple if not iterable.
    """
    return tuple(value) if is_iterable(value) else (value,)


def make_set(value: Any) -> Set:
    """
    Make a set from a value.

    :param value: The value to make a set from.
    :return: A set wrapping the value, or a one-element set if not iterable.
    """
    return set(value) if is_iterable(value) else {value}


def cartesian_product_while_passing_the_bindings_around(
    expressions: Iterable[SymbolicExpression],
    sources: Optional[OperationResult],
) -> Iterator[OperationResult]:
    """
    Evaluate the symbolic expressions by generating combinations of values from their
    evaluation generators while passing the bindings from the previous evaluated
    generator to the next.

    :param expressions: The symbolic expressions to evaluate.
    :param sources: The current OperationResult carrying bindings, or None.
    :return: An Iterable of Bindings for each combination of values.
    """

    def _make_stage(
        inner_expression: SymbolicExpression,
    ) -> Callable[[Optional[OperationResult]], Iterator[OperationResult]]:
        """
        Create a new evaluation stage for the given expression that combines the
        bindings from the previous stage with the current evaluation result.

        A stage is a function that takes a previous result and returns an iterator of
        results.
        """

        def stage(prev: Optional[OperationResult]) -> Iterator[OperationResult]:
            """
            Evaluate the inner expression and combine its bindings with the previous
            stage's bindings.
            """
            for result in inner_expression._evaluate_(prev):
                if prev is not None:
                    result.update(prev)
                yield result

        return stage

    expression_evaluation_generators = [
        _make_stage(expression) for expression in expressions
    ]

    yield from chain_stages(expression_evaluation_generators, sources)


T = TypeVar("T")


def chain_stages(
    stages: List[Callable[[Optional[OperationResult]], Iterator[OperationResult]]],
    initial: Optional[OperationResult],
) -> Iterator[OperationResult]:
    """
    Chains a sequence of stages into a single pipeline.

    This function takes a list of computational stages and an initial binding, passing
    the result of each computation stage to the next one. It produces an iterator of
    bindings by applying each stage in sequence to the current binding.

    :param stages: A list of stages where each stage is a callable that accepts a
        Binding and produces an iterator of Bindings.
    :param initial: The initial binding to start the computation with.
    :return: An iterator over the bindings resulting from applying all stages in
        sequence.
    """

    def evaluate_next_stage_or_yield(
        stage_index: int, current_result: Optional[OperationResult]
    ) -> Iterator[OperationResult]:
        """
        Recursively evaluates the next stage or yields the current binding if all stages
        are done.

        :param stage_index: The index of the current stage.
        :param current_result: The current binding to be processed.
        """
        if stage_index == len(stages):
            yield current_result
            return
        for next_result in stages[stage_index](current_result):
            yield from evaluate_next_stage_or_yield(stage_index + 1, next_result)

    yield from evaluate_next_stage_or_yield(0, initial)


@lru_cache
def get_function_argument_names(function: Callable) -> List[str]:
    """
    :param function: A function to inspect
    :return: The argument names of the function
    """
    return list(inspect.signature(function).parameters.keys())


def merge_args_and_kwargs(
    function_or_class: Callable,
    args: Tuple[Any, ...],
    kwargs: Dict[str, Any],
    ignore_first: bool = False,
) -> Dict[str, Any]:
    """
    Merge the arguments and keyword-arguments of a function/class into a dict of
    keyword-arguments. If a class is passed, the arguments are assumed to be the
    `__init__` arguments.

    :param function_or_class: The function/class to get the argument names from
    :param args: The arguments passed to the function
    :param kwargs: The keyword arguments passed to the function
    :param ignore_first: Whether to ignore the first argument or not. Use this when
        `function_or_class` contains something like `self`
    :return: The dict of assigned keyword-arguments.
    """
    starting_index = 1 if ignore_first else 0
    function_or_class = (
        function_or_class.__init__
        if inspect.isclass(function_or_class)
        else function_or_class
    )
    all_kwargs = {
        name: arg
        for name, arg in zip(
            get_function_argument_names(function_or_class)[starting_index:],
            args,
        )
    }
    all_kwargs.update(kwargs)
    return all_kwargs


def convert_args_and_kwargs_into_hashable_key(
    dictionary: Dict[str, Any],
) -> Tuple[Any, ...]:
    """
    Generates a hashable key from the dictionary.

    The key is a tuple of sorted (key, value) pairs. If a value is a dictionary or a
    set, it is converted to a frozenset of its items. If a value is a list, it is
    converted to a tuple.

    :param dictionary: The keyword arguments to generate the key from.
    :return: The generated key as a tuple.
    """
    key = []
    for k, v in dictionary.items():
        if isinstance(v, dict):
            v = frozenset(v.items())
        elif isinstance(v, set):
            v = frozenset(v)
        elif isinstance(v, list):
            v = tuple(v)
        key.append((k, v))
    return tuple(sorted(key))


def camel_case_to_words(name: str) -> str:
    """
    Convert a CamelCase class name to space-separated lowercase words.

    :param name: CamelCase identifier string.
    :return: Space-separated lowercase words.

    Examples::

        camel_case_to_words("HasRole")     # → "has role"
        camel_case_to_words("IsReachable") # → "is reachable"
    """
    return re.sub(r"([A-Z])", r" \1", name).strip().lower()
