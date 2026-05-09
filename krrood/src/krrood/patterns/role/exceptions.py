"""
Exceptions for the role transformer subsystem.
"""

from __future__ import annotations


class RoleTransformerError(ValueError):
    """Raised when a parsed node has an unexpected type during role transformation."""


class MissingRoleMixinsError(RuntimeError):
    """Raised when one or more role mixin files are absent from disk.

    This typically means the offline generation step has not been run yet.
    Fix by executing::

        krrood-generate-role-mixins <package> [<package> ...]
    """
