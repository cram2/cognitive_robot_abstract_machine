"""
Role type classification for wrapped classes.
"""

from __future__ import annotations

import enum

from krrood.class_diagrams.class_diagram import WrappedClass, WrappedSpecializedGeneric
from krrood.patterns.property_delegator import PropertyDelegator
from krrood.patterns.role.role import Role


class MethodType(enum.Enum):
    """Classification of a method's type."""

    NORMAL = enum.auto()
    """A standard instance method."""

    CLASS_METHOD = enum.auto()
    """A method decorated with @classmethod."""

    FACTORY_METHOD = enum.auto()
    """A @classmethod that returns Self."""

    def is_class_method(self) -> bool:
        """Return True if this is a class method or factory method."""
        return self in (MethodType.CLASS_METHOD, MethodType.FACTORY_METHOD)


class RoleType(enum.Enum):
    """Classification of a class within the role or delegation hierarchy."""

    PRIMARY = enum.auto()
    """A primary role that directly inherits from Role or updates the role taker type."""

    SUB_ROLE = enum.auto()
    """A role that inherits from another role."""

    SPECIALIZED_ROLE_FOR = enum.auto()
    """A synthetic role created when a role updates its taker type."""

    DELEGATOR = enum.auto()
    """A PropertyDelegator subclass that is not a Role."""

    NOT_A_ROLE = enum.auto()
    """A class that is neither a role nor a property delegator."""

    @staticmethod
    def get_role_type(wrapped_class: WrappedClass) -> RoleType:
        """Return the role type of the given wrapped class.

        :param wrapped_class: The wrapped class to classify.
        :return: The corresponding RoleType value.
        """
        if isinstance(wrapped_class, WrappedSpecializedGeneric):
            return RoleType.NOT_A_ROLE

        if not issubclass(wrapped_class.clazz, PropertyDelegator):
            return RoleType.NOT_A_ROLE

        if not issubclass(wrapped_class.clazz, Role):
            return RoleType.DELEGATOR

        is_direct_role = any(
            p is Role or (p.__origin__ is Role if hasattr(p, "__origin__") else False)
            for p in wrapped_class.clazz.__bases__
        )

        if is_direct_role:
            return RoleType.PRIMARY
        if wrapped_class.clazz.updates_role_taker_type():
            return RoleType.SPECIALIZED_ROLE_FOR
        return RoleType.SUB_ROLE
