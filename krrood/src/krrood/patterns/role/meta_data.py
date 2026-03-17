from __future__ import annotations

import enum

from krrood.class_diagrams.class_diagram import WrappedClass, WrappedSpecializedGeneric
from krrood.patterns.role.role import Role


class RoleType(enum.Enum):
    """
    Enum representing the different types of roles.
    """

    PRIMARY = enum.auto()
    """
    A primary role that directly inherits from Role or updates the role taker type.
    """

    SUB_ROLE = enum.auto()
    """
    A role that inherits from another role.
    """

    SPECIALIZED_ROLE_FOR = enum.auto()
    """
    A synthetic role created when a role updates its taker type.
    """

    NOT_A_ROLE = enum.auto()
    """
    A class that is not a role.
    """

    @staticmethod
    def get_role_type(wrapped_class: WrappedClass) -> RoleType:
        """
        Determines the role type of a wrapped class.

        :param wrapped_class: The wrapped class.
        :return: The role type.
        """
        if isinstance(wrapped_class, WrappedSpecializedGeneric) or not issubclass(
                wrapped_class.clazz, Role
        ):
            return RoleType.NOT_A_ROLE

        # Local check for primary roles: must be a direct subclass of Role
        is_direct_role = any(
            p is Role or (getattr(p, "__origin__", None) is Role)
            for p in wrapped_class.clazz.__bases__
        )

        if is_direct_role:
            return RoleType.PRIMARY
        elif wrapped_class.clazz.updates_role_taker_type():
            return RoleType.SPECIALIZED_ROLE_FOR

        return RoleType.SUB_ROLE
