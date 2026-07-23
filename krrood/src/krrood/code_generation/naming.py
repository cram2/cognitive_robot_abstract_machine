"""Naming-convention helpers for generated Python identifiers."""

from __future__ import annotations

import re
import inflection

# %%
# Naming utilities


def to_snake_case(name: str) -> str:
    """Convert any string to snake_case.

    :param name: The string to convert.
    :return: The snake_case string.
    """
    if len(name) == 0:
        return name
    return inflection.underscore(name)


def to_camel_case(name: str) -> str:
    """Convert snake_case to CamelCase. E.g. ``'my_func'`` → ``'MyFunc'``."""
    if len(name) == 0:
        return name
    return inflection.camelize(name, True)


def camel_case_to_lower_camel_case(name: str) -> str:
    """Convert a CamelCase name to a lowerCamelCase name.

    E.g. ``"Distance"`` → ``"distance"``, ``"MyDistance"`` → ``"myDistance"``.
    """
    if len(name) == 0:
        return name
    return inflection.camelize(name, False)