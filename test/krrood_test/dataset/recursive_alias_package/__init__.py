"""
A package that re-exports a recursive alias from a module of its own, so that the name
the alias refers to itself by is bound in that module only.
"""

from .definitions import RecursiveAliasTree
