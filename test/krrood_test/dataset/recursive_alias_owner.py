"""
A class annotated with a recursive alias of another package, whose module never binds
the name the alias refers to itself by.
"""

from __future__ import annotations

from dataclasses import dataclass

from . import recursive_alias_package


@dataclass(kw_only=True)
class RecursiveAliasOwner:
    """
    An object that holds a value of a recursive alias.
    """

    tree: recursive_alias_package.RecursiveAliasTree
    """
    A whole number or a nested list of them.
    """
