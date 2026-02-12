from __future__ import annotations

import inspect
from functools import lru_cache

"""
Utilities for hashing, rendering, and general helpers used by the
symbolic query engine.
"""
import itertools

try:
    import six
except ImportError:
    six = None

try:
    from graphviz import Source
except ImportError:
    Source = None

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
)

if TYPE_CHECKING:
    from krrood.entity_query_language.symbolic import Bindings


class IDGenerator:
    """
    A class that generates incrementing, unique IDs and caches them for every object this is called on.
    """

    _counter = 0
    """
    The counter of the unique IDs.
    """

    # @lru_cache(maxsize=None)
    def __call__(self, obj: Any) -> int:
        """
        Creates a unique ID and caches it for every object this is called on.

        :param obj: The object to generate a unique ID for, must be hashable.
        :return: The unique ID.
        """
        self._counter += 1
        return self._counter


def lazy_iterate_dicts(dict_of_iterables):
    """Generator that yields dicts with one value from each iterable"""
    for values in zip(*dict_of_iterables.values()):
        yield dict(zip(dict_of_iterables.keys(), values))


def generate_combinations(generators_dict):
    """Yield all combinations of generator values as keyword arguments"""
    for combination in itertools.product(*generators_dict.values()):
        yield dict(zip(generators_dict.keys(), combination))


def make_list(value: Any) -> List:
    """
    Make a list from a value.

    :param value: The value to make a list from.
    """
    return list(value) if is_iterable(value) else [value]


def is_iterable(obj: Any) -> bool:
    """
    Check if an object is iterable.

    :param obj: The object to check.
    """
    return hasattr(obj, "__iter__") and not isinstance(
        obj, (str, type, bytes, bytearray)
    )


def make_tuple(value: Any) -> Any:
    """
    Make a tuple from a value.
    """
    return tuple(value) if is_iterable(value) else (value,)


def make_set(value: Any) -> Set:
    """
    Make a set from a value.

    :param value: The value to make a set from.
    """
    return set(value) if is_iterable(value) else {value}


T = TypeVar("T")


def chain_stages(
    stages: List[Callable[[Bindings], Iterator[Bindings]]], initial: Bindings
) -> Iterator[Bindings]:
    """
    Chains a sequence of stages into a single pipeline.

    This function takes a list of computational stages and an initial binding, passing the
    result of each computation stage to the next one. It produces an iterator of bindings
    by applying each stage in sequence to the current binding.

    :param stages: A list of stages where each stage is a callable that accepts
        a Binding and produces an iterator of bindings.
    :param initial: The initial binding to start the computation with.

    :return: An iterator over the bindings resulting from applying all
        stages in sequence.
    """

    def evaluate_next_stage_or_yield(i: int, b: Bindings) -> Iterator[Bindings]:
        """
        Recursively evaluates the next stage or yields the current binding if all stages are done.

        :param i: The index of the current stage.
        :param b: The current binding to be processed.
        """
        if i == len(stages):
            yield b
            return
        for b2 in stages[i](b):
            yield from evaluate_next_stage_or_yield(i + 1, b2)

    yield from evaluate_next_stage_or_yield(0, initial)


@lru_cache
def get_function_argument_names(function: Callable) -> List[str]:
    """
    :param function: A function to inspect
    :return: The argument names of the function
    """
    return list(inspect.signature(function).parameters.keys())


def merge_args_and_kwargs(
    function_or_class: Union[Callable, Type], args, kwargs, ignore_first: bool = False
) -> Dict[str, Any]:
    """
    Merge the arguments and keyword-arguments of a function/class into a dict of keyword-arguments.
    If a class is passed, the arguments are assumed to be the `__init__` arguments.

    :param function_or_class: The function/class to get the argument names from
    :param args: The arguments passed to the function
    :param kwargs: The keyword arguments passed to the function
    :param ignore_first: Whether to ignore the first argument or not.
    Use this when `function_or_class` contains something like `self`
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


def convert_args_and_kwargs_into_a_hashable_key(
    dictionary: Dict[str, Any],
) -> Tuple[Any, ...]:
    """
    Generates a hashable key from the dictionary. The key is a tuple of sorted (key, value) pairs.
    If a value is a dictionary or a set, it is converted to a frozenset of its items.
    If a value is a list, it is converted to a tuple.

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


def ensure_hashable(obj) -> Hashable:
    """
    :return: The object itself if it is hashable, otherwise its id.
    """
    if not is_hashable(obj):
        return id(obj)
    return obj


def is_hashable(obj):
    """
    Checks if an object is hashable by attempting to compute its hash.

    :param obj: The object to check.
    :return: True if the object is hashable, False otherwise.
    """
    try:
        hash(obj)
        return True
    except TypeError:
        return False
