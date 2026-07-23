"""Serialisation of type hints and values to Python source-code strings."""

from __future__ import annotations

import enum
import inspect
from typing import (
    Any,
    Callable,
    Dict,
    ForwardRef,
    List,
    Optional,
    Set,
    Tuple,
    Type,
    get_args,
    get_origin,
)

from krrood.class_diagrams.utils import get_type_hints_of_object
from krrood.utils import is_builtin_type, is_typing_type

# %%
# Type-hint serialisation

def _extract_types(type_hint: Any, seen: Optional[Set[type]] = None) -> Set[type]:
    """Recursively extract all base types from a type hint."""
    if seen is None:
        seen = set()

    if type_hint in seen or isinstance(type_hint, str):
        return seen

    if isinstance(type_hint, ForwardRef):
        return seen

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin:
        seen.add(origin)
        for arg in args:
            _extract_types(arg, seen)
    elif isinstance(type_hint, type):
        seen.add(type_hint)

    return seen


def stringify_type_hint(type_hint: Any) -> str:
    """Recursively convert a type hint to its string representation.

    Handles :class:`~typing.ForwardRef`, generic aliases (e.g. ``List[int]``),
    builtins, and qualified names.  This is the **single canonical** function
    for converting type hints to source-code strings — use it everywhere
    instead of manual ``t.__name__`` concatenation.

    :param type_hint: The type hint to convert.
    :returns: A Python source-code string for the type.
    """
    if isinstance(type_hint, str):
        return type_hint

    if isinstance(type_hint, ForwardRef):
        return type_hint.__forward_arg__

    origin = get_origin(type_hint)
    args = get_args(type_hint)

    if origin is not None:
        origin_str = getattr(origin, "__name__", str(origin)).capitalize()
        args_str = ", ".join(stringify_type_hint(arg) for arg in args)
        return f"{origin_str}[{args_str}]"

    if isinstance(type_hint, type):
        if type_hint.__module__ == "builtins":
            return type_hint.__name__
        return f"{type_hint.__qualname__}"

    return str(type_hint)


def value_to_source(value: object) -> str:
    """Convert a Python value to its source-code representation.

    Handles: ``None``, booleans, integers, floats, strings, enum members,
    and type objects.  Falls back to ``repr(value)`` for unrecognized types.

    :param value: The Python value to convert.
    :returns: A source-code string.
    """
    if value is None:
        return "None"
    if isinstance(value, enum.Enum):
        return f"{type(value).__name__}.{value.name}"
    if isinstance(value, type) or is_typing_type(value):
        return stringify_type_hint(value)
    return repr(value)


def get_types_to_import_from_type_hints(hints: List[Type]) -> List[Type]:
    """Extract importable types from a list of type hints.

    :param hints: A list of type hints to extract types from.
    :return: A set of types that need to be imported.
    """
    seen_types = _extract_types(None, set())
    for hint in hints:
        _extract_types(hint, seen_types)

    to_import: List[Type] = []
    for type_ in seen_types:
        if isinstance(type_, ForwardRef) or isinstance(type_, str):
            continue
        if not is_builtin_type(type_) and type_ not in to_import:
            to_import.append(type_)

    return to_import


def get_types_to_import_from_function_type_hints(function: Callable) -> List[Type]:
    """Extract importable types from a function's annotations.

    :param function: The function to extract type hints from.
    :returns: A list of types that need to be imported.
    """
    hints = get_type_hints_of_object(function)

    sig = inspect.signature(function)
    all_hints = list(hints.values())
    if sig.return_annotation != inspect.Signature.empty:
        all_hints.append(sig.return_annotation)

    for param in sig.parameters.values():
        if param.annotation != inspect.Parameter.empty:
            all_hints.append(param.annotation)

    return get_types_to_import_from_type_hints(all_hints)
