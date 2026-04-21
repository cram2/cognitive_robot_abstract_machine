from __future__ import division

from collections import defaultdict
from copy import deepcopy
from dataclasses import fields, dataclass
from functools import wraps
from time import time
from typing import Any, TypeVar
from typing import Callable, get_type_hints, get_origin, Union, get_args

# I only do this, because otherwise test/test_integration_pr2.py::TestWorldManipulation::test_unsupported_options
# fails on github actions

T = TypeVar("T", bound=Callable)


def memoize(function: T) -> T:
    memo = function.memo = {}

    @wraps(function)
    def wrapper(*args: Any, **kwargs: Any) -> T:
        key = (args, frozenset(kwargs.items()))
        try:
            return memo[key]
        except KeyError:
            rv = function(*args, **kwargs)
            memo[key] = rv
            return rv

    return wrapper  # type: ignore


def memoize_with_counter(reset_after: int):
    def memoize(function: T) -> T:
        memo = function.memo = {}
        function.__counter = 0

        @wraps(function)
        def wrapper(*args, **kwargs):
            key = (args, frozenset(kwargs.items()))
            try:
                hit = memo[key]
                if function.__counter >= reset_after:
                    raise KeyError
                else:
                    function.__counter += 1
                    return hit
            except KeyError:
                function.__counter = 1
                rv = function(*args, **kwargs)
                memo[key] = rv
                return rv

        return wrapper

    return memoize


def record_time(function: T) -> T:
    # return function
    function_name = function.__name__

    @wraps(function)
    def wrapper(*args, **kwargs):
        self = args[0]
        if not hasattr(self, "__times"):
            setattr(self, "__times", defaultdict(list))
        start_time = time()
        result = function(*args, **kwargs)
        time_delta = time() - start_time
        self.__times[function_name].append(time_delta)
        return result

    return wrapper


# %% these two decorators automatically add a state variable to an object that prevents multiple calls for off on pairs
def toggle_on(state_var: str):
    def decorator(func: T) -> T:
        def wrapper(self, *args, **kwargs) -> T:
            if getattr(self, state_var, False):
                return
            setattr(self, state_var, True)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def toggle_off(state_var: str):
    def decorator(func: T) -> T:
        def wrapper(self, *args, **kwargs) -> T:
            if not getattr(self, state_var, True):
                return
            setattr(self, state_var, False)
            return func(self, *args, **kwargs)

        return wrapper

    return decorator


def _check_type(value, expected_type):
    """Check if value matches expected_type, handling Union, List, etc."""
    # Handle None case
    if value is None:
        return expected_type is type(None) or (
            get_origin(expected_type) is Union and type(None) in get_args(expected_type)
        )

    # Get the origin and args of the type hint
    origin = get_origin(expected_type)
    args = get_args(expected_type)

    # Handle Union types (including Optional which is Union[T, None])
    if origin is Union:
        return any(_check_type(value, arg) for arg in args)

    # Handle List, Set, Tuple, etc.
    if origin is list:
        if not isinstance(value, list):
            return False
        if args:  # If List[SomeType] was specified
            return all(_check_type(item, args[0]) for item in value)
        return True

    if origin is tuple:
        if not isinstance(value, tuple):
            return False
        if args:
            if len(args) == 2 and args[1] is ...:  # Tuple[int, ...]
                return all(_check_type(item, args[0]) for item in value)
            else:  # Tuple[int, str, float]
                return len(value) == len(args) and all(
                    _check_type(v, t) for v, t in zip(value, args)
                )
        return True

    if origin is set:
        if not isinstance(value, set):
            return False
        if args:
            return all(_check_type(item, args[0]) for item in value)
        return True

    if origin is dict:
        if not isinstance(value, dict):
            return False
        if len(args) == 2:  # Dict[KeyType, ValueType]
            return all(_check_type(k, args[0]) for k in value.keys()) and all(
                _check_type(v, args[1]) for v in value.values()
            )
        return True

    # Handle regular types (int, str, custom classes, etc.)
    if origin is None:
        try:
            return isinstance(value, expected_type)
        except TypeError:
            # Fallback for types that can't be used with isinstance
            return type(value) == expected_type

    # For other generic types, just check the origin
    try:
        return isinstance(value, origin)
    except TypeError:
        return False


def validate_types(cls):
    """Class decorator that adds type validation to dataclasses."""
    original_post_init = getattr(cls, "__post_init__", None)

    def __post_init__(self):
        # Validate types FIRST
        type_hints = get_type_hints(type(self))
        for field_info in fields(self):
            field_name = field_info.name

            if not field_info.init:
                continue

            if field_name in type_hints:
                expected_type = type_hints[field_name]
                if expected_type == float:
                    expected_type = Union[int, float]
                value = getattr(self, field_name)
                if not _check_type(value, expected_type):
                    expected_str = str(expected_type).replace("typing.", "")
                    raise TypeError(
                        f"Field '{field_name}' in '{self.__class__.__name__}' expected {expected_str}, "
                        f"got {type(value).__name__} = {str(value)}"
                    )

        # THEN call the original __post_init__ if it existed
        if original_post_init is not None:
            original_post_init(self)

    cls.__post_init__ = __post_init__
    return cls


def validated_dataclass(cls):
    """Combines @dataclass and @validate_types for convenience."""
    return validate_types(dataclass(cls))
