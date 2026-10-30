"""
The rules on the design of classes, methods and modules.
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from pathlib import Path

from typing_extensions import ClassVar, Protocol

import krrood
from krrood.adapters import json_serializer
from krrood.adapters.json_serializer import (
    DataclassJSONSerializer,
    SubclassJSONSerializer,
)
from krrood.development_tools.agents.rule import CodeReference, Section
from krrood.exceptions import DataclassException
from krrood.ormatic.utils import classproperty
from krrood.patterns.path_enumeration import PathEnumeration


class DesignPrinciplesRules(Section):
    """
    The rules on the design of classes and modules.
    """

    OBJECT_ORIENTED_AND_SOLID = (
        "Use strictly object-oriented design and always apply the SOLID principles:",
        (
            (
                "Single Responsibility: each class and method does one thing. If a "
                "method has cyclomatic complexity in the hundreds, refactor"
            ),
            "Open/Closed: open for extension, closed for modification",
            (
                "Liskov Substitution: subtypes are substitutable for their base types "
                "without breaking behaviour"
            ),
            (
                "Interface Segregation: many small interfaces over one large one. An "
                "interface is an explicit superclass or mixin, never a "
                f"{CodeReference(Protocol).name}"
            ),
            (
                "Dependency Inversion: depend on abstractions, not concrete "
                "implementations"
            ),
        ),
    )

    MODULAR_AND_DECOUPLED = "Keep code modular and decoupled"

    NO_YAGNI = "Eliminate YAGNI smells"

    HARD_TO_MISUSE_INTERFACES = "Make interfaces hard to misuse"


class MethodsRules(Section):
    """
    The rules on methods.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DesignPrinciplesRules

    EXTRACT_METHODS_THAT_EARN_THEIR_ABSTRACTION = (
        "Every method hides what actually runs from the reader, so extract one "
        "only when its abstraction is worth more than reading the executed code in "
        "place"
    )

    ONE_LEVEL_OF_ABSTRACTION = (
        "Keep every statement in a method on the same level of abstraction: a "
        "method either strings together named steps or carries out one step's "
        "detail, never both"
    )

    NO_FORWARDING_METHODS = (
        "Never write a method that only forwards to or renames another call. A "
        "one-line method is allowed only when it implements an abstract method or "
        f"a {CodeReference(classproperty).name}, caches its result, or removes duplication - and in the "
        "last case, first try to abstract the duplicated code itself"
    )

    INLINE_SINGLE_CALLER_METHODS = (
        "A method with a single caller must earn its place under these rules; "
        "otherwise inline it"
    )

    GUARD_CLAUSES = (
        "Reduce nesting and complexity with guard clauses: handle alternative "
        "outputs first by inverting conditions and returning early, so the main "
        "branch holds the main output and the biggest compute. A single level of "
        "abstraction never justifies deep nesting; split the method instead"
    )


class ExceptionModellingRules(Section):
    """
    The rules on raising and declaring exceptions.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DesignPrinciplesRules

    NEVER_TRY_EXCEPT = (
        "Never use try-except; a program in an illegal state raises an appropriate "
        "exception"
    )

    DECLARE_AS_DATACLASSES = "Create meaningful custom exceptions as dataclasses"

    PLACE_IN_THE_CLOSEST_MODULE = (
        "Put each exception in the `exceptions.py` closest to the code that raises it:",
        (
            (
                "An exception raised only within one subpackage goes in that "
                "subpackage's `exceptions.py`; create the file if the subpackage has "
                "none"
            ),
            (
                "An exception raised across several subpackages goes in the "
                "`exceptions.py` of their nearest common package"
            ),
        ),
    )

    NO_ASSERT_OUTSIDE_TESTS = (
        "Never use `assert` outside tests: Python drops it when run with `-O`, so "
        "the check silently disappears. Raise a custom exception instead"
    )

    SUBCLASS_THE_KRROOD_BASE_CLASS = (
        f"Custom exceptions subclass {CodeReference(DataclassException).path}, "
        f"implementing its {CodeReference(DataclassException.error_message).name} "
        f"and {CodeReference(DataclassException.suggest_correction).name}"
    )


class ConstantsAndClassLevelValuesRules(Section):
    """
    The rules on constants and values shared by a class.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DesignPrinciplesRules

    NO_GLOBAL_OR_CLASS_VARIABLES = (
        "Never use global variables, module-level constants or "
        f"{CodeReference(ClassVar).name}. Instead:",
        (
            "A fixed set of related values is a module-level enum",
            (
                "A default a caller may want to change is a parameter of the method "
                "that uses it if only that method does, otherwise a dataclass field "
                "with that default. For example, a `Heater` takes `target_temperature: "
                "float = 20.0` as a field instead of declaring "
                "`DEFAULT_TARGET_TEMPERATURE: ClassVar[float] = 20.0`"
            ),
            f"A constant that belongs to a class is a {CodeReference(classproperty).name}",
            (
                f"{CodeReference(ClassVar).name} is allowed only for state that explicitly has to be shared "
                "and mutable across every instance of a class. This almost never "
                "applies"
            ),
        ),
    )

    USE_KRROOD_CLASSPROPERTY = (
        f"Use the {CodeReference(classproperty).name} "
        f"{CodeReference(krrood).name} provides"
    )


class StructuredDataRules(Section):
    """
    The rules on representing data with structure instead of bare literals.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DesignPrinciplesRules

    PREFER_STRUCTURED_DATA = (
        "Structured data is the default over bare strings, hardcoded values and "
        "meaningless numbers: reach for the structured form first and justify the "
        "literal, never the other way round"
    )

    NAMED_STRINGS_ARE_ENUMS = (
        "A string that names a fixed thing - a payload key, a state, a label, a "
        "filename, an environment variable, a command flag, a status - is a "
        f"{CodeReference(StrEnum).name} member. A value spelled in two places has no single source to "
        "rename, and nothing fails when the two drift apart"
    )

    TYPED_ENUM_VALUES = (
        "When the values are more than text - paths, numbers - give the enum "
        "values of that type, mixing the type in "
        f"({CodeReference(IntEnum).name}, {CodeReference(StrEnum).name})"
    )

    NO_MAGIC_NUMBERS = (
        "A magic number becomes an enum member, a field default or a "
        f"{CodeReference(classproperty).name}; a bare literal that carries meaning is unreadable where "
        "it is used and unsearchable everywhere else"
    )

    MIRROR_FOREIGN_DATA = (
        "Data whose shape someone else controls - an API response, a configuration "
        "file - is mirrored in dataclasses and parsed by a `from_json` "
        "classmethod, so the field names and the access path into the payload are "
        "written once"
    )

    NO_POSITIONAL_TUPLES = (
        "A tuple whose positions carry meaning becomes a dataclass, or an enum "
        "when the positions are a fixed set of alternatives rather than fields"
    )

    LONG_LITERALS_IN_FILES = (
        "A long literal document - a query, a template, a schema - lives in a file "
        "of its own type and is read in, never embedded as a string"
    )

    SUBCLASS_PATH_ENUMERATION = (
        f"A path enum subclasses {CodeReference(PathEnumeration).path}, so every "
        f"path derived from a member is a plain {CodeReference(Path).name}; its "
        "members' `name` is the path's last component, so read the member name "
        "as `_name_`"
    )


class SerializationRules(Section):
    """
    The rules on converting our own classes to and from a stored form.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return DesignPrinciplesRules

    ROUND_TRIP_JSON_WITH_KRROOD = (
        "JSON that our own classes round-trip goes through "
        f"{CodeReference(json_serializer).path}: use "
        f"{CodeReference(DataclassJSONSerializer).name} wherever it can "
        "(de)serialize the class; where a class needs more, subclass "
        f"{CodeReference(SubclassJSONSerializer).name} and build on "
        f"{CodeReference(DataclassJSONSerializer).name}'s "
        f"{CodeReference(DataclassJSONSerializer.to_json).name}/"
        f"{CodeReference(DataclassJSONSerializer.from_json).name}. Never "
        "hand-write field-by-field "
        f"{CodeReference(SubclassJSONSerializer.to_json).name}/"
        f"{CodeReference(SubclassJSONSerializer.from_json).name}"
    )
