from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Type

from krrood.exceptions import DataclassException


@dataclass
class DelegatedFactoryMethodError(DataclassException):
    """
    Raised when a role-taker factory method is invoked through a role.

    A factory classmethod constructs an instance of the role taker, so delegating it through a
    role would return a bare role taker and silently drop the role. The call is refused to keep
    that mistake loud instead of quiet.
    """

    role_type: Type
    """
    The role type the factory method was accessed through.
    """

    taker_type: Type
    """
    The role-taker type that declares the factory method.
    """

    method_name: str
    """
    The name of the delegated factory method.
    """

    def error_message(self) -> str:
        return (
            f"{self.taker_type.__name__}.{self.method_name}() is a factory method; calling it "
            f"through {self.role_type.__name__} would build a bare {self.taker_type.__name__} and "
            f"drop the role."
        )

    def suggest_correction(self) -> str:
        return (
            f"Either override {self.method_name}() on {self.role_type.__name__} to return a "
            f"proper role, or call it on the role taker explicitly via .role_taker or "
            f".root_persistent_entity."
        )


@dataclass
class RoleAttributeNotDeclaredError(DataclassException):
    """
    Raised when assigning a name that the role does not declare as one of its own fields.

    Assignments target the role itself and only its declared fields may be set, so a write cannot
    silently shadow a role-taker attribute.
    """

    role_type: Type
    """
    The role class the assignment was attempted on.
    """

    attribute_name: str
    """
    The name that was assigned.
    """

    def error_message(self) -> str:
        return (
            f"{self.role_type.__name__} declares no field named '{self.attribute_name}', and a "
            f"role can only assign its own declared fields."
        )

    def suggest_correction(self) -> str:
        return (
            f"Declare '{self.attribute_name}' as a field on {self.role_type.__name__}, or assign "
            f"through .role_taker to change the underlying entity."
        )
