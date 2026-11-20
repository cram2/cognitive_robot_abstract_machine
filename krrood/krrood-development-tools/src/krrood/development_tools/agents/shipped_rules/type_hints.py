"""
The rules on type hints.
"""

from __future__ import annotations


import typing_extensions
from typing_extensions import Any, ClassVar, Generic

from krrood.development_tools.agents.rule import CodeReference, Section
from krrood.development_tools.agents.ruff_rules import (
    CheckedModules,
    RuffRule,
    RuffRuleChecker,
    RuffSetting,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric


class TypeHintsRules(Section):
    """
    The rules on type hints.
    """

    ANNOTATE_EVERYTHING = (
        "Every parameter and return value has a type hint, "
        f"{CodeReference(target=Any).name} and `-> None` included",
        (),
        RuffRuleChecker(
            lint_rules=(
                RuffRule.MISSING_TYPE_FUNCTION_ARGUMENT,
                RuffRule.MISSING_TYPE_ARGS,
                RuffRule.MISSING_TYPE_KWARGS,
                RuffRule.MISSING_RETURN_TYPE_UNDOCUMENTED_PUBLIC_FUNCTION,
                RuffRule.MISSING_RETURN_TYPE_PRIVATE_FUNCTION,
                RuffRule.MISSING_RETURN_TYPE_SPECIAL_METHOD,
                RuffRule.MISSING_RETURN_TYPE_STATIC_METHOD,
                RuffRule.MISSING_RETURN_TYPE_CLASS_METHOD,
            ),
            modules=CheckedModules.OUTSIDE_TESTS,
        ),
    )

    ACCURATE_TYPE_HINTS = (
        "Every type hint is accurate: it names the types the code takes and returns"
    )

    BUILTIN_GENERICS = (
        "Use builtin generics and `X | None` (`list[int]`, `Pose | None`), never "
        "`typing.List`, `Dict`, `Optional` or `Union`; import every other typing "
        f"construct from {CodeReference(target=typing_extensions).path}, never from `typing`",
        (),
        RuffRuleChecker(
            lint_rules=(
                RuffRule.NON_PEP585_ANNOTATION,
                RuffRule.NON_PEP604_ANNOTATION_UNION,
                RuffRule.NON_PEP604_ANNOTATION_OPTIONAL,
                RuffRule.BANNED_API,
            ),
            settings=(RuffSetting.TYPING_BANNED,),
        ),
    )

    FUTURE_ANNOTATIONS = (
        "Use `from __future__ import annotations` instead of wrapping types in strings",
        (),
        RuffRuleChecker(
            lint_rules=(RuffRule.MISSING_REQUIRED_IMPORT, RuffRule.QUOTED_ANNOTATION),
            settings=(RuffSetting.FUTURE_ANNOTATIONS_REQUIRED,),
        ),
    )

    BOUND_GENERIC_PARAMETERS = (
        "When each class in a family declares the type it handles, carry that type "
        f"as a bound generic parameter, not a {CodeReference(target=ClassVar).name}, so it is part of the "
        "signature and cannot disagree with a separate attribute: bind it in each "
        "member (`class MemberOfFamily(Family[ConcreteType])`)"
    )

    NARROW_FIELD_TYPES_IN_GENERIC_SUBCLASSES = (
        f"Generic classes inherit {CodeReference(target=Generic).name} and then "
        f"{CodeReference(target=SubClassSafeGeneric).path}, which narrows field types "
        "when a subclass binds a parameter. Read a bound type with "
        f"{CodeReference(target=SubClassSafeGeneric.get_type_of_generic_parameter).name}, "
        "never via `__orig_bases__`. Such classes cannot be frozen dataclasses"
    )
