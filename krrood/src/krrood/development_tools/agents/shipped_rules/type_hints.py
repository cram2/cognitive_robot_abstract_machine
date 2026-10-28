"""
The rules on type hints.
"""

from __future__ import annotations


import typing_extensions
from typing_extensions import Any, ClassVar, Generic

from krrood.development_tools.agents.rule import CodeReference, Section
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric


class TypeHintsRules(Section):
    """
    The rules on type hints.
    """

    ANNOTATE_EVERYTHING = (
        "Every parameter and return value has an accurate type hint, "
        f"{CodeReference(Any).name} and `-> "
        "None` included"
    )

    BUILTIN_GENERICS = (
        "Use builtin generics and `X | None` (`list[int]`, `Pose | None`), never "
        "`typing.List`, `Dict`, `Optional` or `Union`; import every other typing "
        f"construct from {CodeReference(typing_extensions).path}, never from `typing`"
    )

    FUTURE_ANNOTATIONS = (
        "Use `from __future__ import annotations` instead of wrapping types in strings"
    )

    BOUND_GENERIC_PARAMETERS = (
        "When each class in a family declares the type it handles, carry that type "
        f"as a bound generic parameter, not a {CodeReference(ClassVar).name}, so it is part of the "
        "signature and cannot disagree with a separate attribute: bind it in each "
        "member (`class MemberOfFamily(Family[ConcreteType])`)"
    )

    NARROW_FIELD_TYPES_IN_GENERIC_SUBCLASSES = (
        f"Generic classes inherit {CodeReference(Generic).name} and then "
        f"{CodeReference(SubClassSafeGeneric).path}, which narrows field types "
        "when a subclass binds a parameter. Read a bound type with "
        f"{CodeReference(SubClassSafeGeneric.get_type_of_generic_parameter).name}, "
        "never via `__orig_bases__`. Such classes cannot be frozen dataclasses"
    )
