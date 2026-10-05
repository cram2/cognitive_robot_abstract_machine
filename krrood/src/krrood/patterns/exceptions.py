from __future__ import annotations

import inspect
from dataclasses import dataclass

from typing_extensions import Any, Callable, List, Type

from krrood.exceptions import DataclassException


@dataclass
class DelegatedFactoryMethodError(DataclassException):
    """
    Raised when a role-taker factory method is invoked through a role.

    A factory classmethod constructs an instance of the role taker, so delegating it
    through a role would return a bare role taker and silently drop the role. The call
    is refused to keep that mistake loud instead of quiet.
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
    Raised when assigning a name that the role does not declare as one of its own
    fields.

    Assignments target the role itself and only its declared fields may be set, so a
    write cannot silently shadow a role-taker attribute.
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


@dataclass
class UnmemoizableOwnerError(DataclassException):
    """
    Raised when a memoized call's owner cannot hold a cache.

    A memoization cache lives exactly as long as the object whose results it caches, so
    an owner whose lifetime cannot be observed has nowhere to keep one.
    """

    owner: Any
    """
    The receiver of the memoized call.
    """

    function_name: str
    """
    The name of the memoized function that was called.
    """

    def error_message(self) -> str:
        return (
            f"{self.function_name}() was called on a {type(self.owner).__name__}, which cannot be "
            f"weakly referenced and therefore cannot own a memoization cache."
        )

    def suggest_correction(self) -> str:
        return (
            f"Memoize on an object whose lifetime can be tracked, or cache "
            f"{self.function_name}() with functools.lru_cache instead."
        )


@dataclass
class KeywordNamesNoFactoryParameter(DataclassException, TypeError):
    """
    Raised when a keyword argument given for construction names no parameter of the
    factory it is given to, which would otherwise be lost without a trace.
    """

    factory: Callable[..., Any]
    """
    The factory the keyword argument was given to.
    """

    keyword: str
    """
    The keyword argument that names no parameter of the factory.
    """

    def error_message(self) -> str:
        return (
            f"{self.factory.__qualname__} has no parameter named {self.keyword!r}, so it "
            f"cannot be constructed with it."
        )

    def suggest_correction(self) -> str:
        parameter_names = ", ".join(inspect.signature(self.factory).parameters)
        return (
            f"Check the spelling of {self.keyword!r}; the parameters of "
            f"{self.factory.__qualname__} are: {parameter_names}."
        )


@dataclass
class AmbiguousRuleError(DataclassException):
    """
    Two or more rules of a family are equally specific for the same subject, a collision
    that would otherwise resolve silently by registration order.

    Surfaced as an error so an accidental overlap is caught rather than masked.
    """

    subject: Any
    """
    What the rules were asked about when the collision occurred.
    """

    candidates: List[Type]
    """
    The equally specific rule classes that collided.
    """

    def error_message(self) -> str:
        names = ", ".join(sorted(candidate.__name__ for candidate in self.candidates))
        return f"{names} are equally specific for {self.subject!r}."

    def suggest_correction(self) -> str:
        return (
            "Make the colliding guards mutually exclusive, or have one rule subclass the other "
            "to declare it the more-specific special case."
        )
