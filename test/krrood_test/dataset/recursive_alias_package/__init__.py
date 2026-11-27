"""
A package that re-exports a recursive alias from a module of its own, the way
:mod:`optax` re-exports ``OptState``, so that the name the alias refers to is bound in
that module only.
"""

from .definitions import RecursiveAliasTree
