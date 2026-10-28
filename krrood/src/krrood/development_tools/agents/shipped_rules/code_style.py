"""
The rules on code style and naming.
"""

from __future__ import annotations

import dataclasses
from enum import Enum


from krrood.development_tools.agents.rule import CodeReference, Section
from krrood.ormatic.utils import classproperty


class CodeStyleRules(Section):
    """
    The rules on the shape of code.
    """

    USE_DATACLASSES = "Always use dataclasses"

    DIVIDE_FILES_INTO_SECTIONS = (
        "Divide every file, source and test, into sections with `# %% <short "
        "description>` headers (e.g. `# %% same-noun disambiguation`), never "
        "decorative box-drawing dividers"
    )

    GROUP_PRIMITIVES_INTO_DATACLASSES = (
        "A group of primitives that travels together, or a return type that keeps "
        "being repeated, becomes a dataclass"
    )

    NEVER_DUPLICATE_CODE = (
        "Never duplicate code. Never put methods in a catch-all module such as "
        "`utils.py`; move them onto the class that owns the behaviour"
    )

    ACCESS_ATTRIBUTES_DIRECTLY = (
        f"Access attributes with `.`, never with {CodeReference(getattr).name}, and "
        "never wrap "
        "attribute access in try-except"
    )

    NO_MUTABLE_DEFAULT_ARGUMENTS = "Never use mutable objects as default arguments"

    NEVER_REIMPLEMENT = (
        "Never reimplement what the codebase or an existing dependency already "
        "provides. When searching for it, never cut off the output of grep or "
        "other searches; narrow the search instead"
    )


class NamingRules(Section):
    """
    The rules on naming.
    """

    @classproperty
    def parent(cls) -> type[Section]:
        return CodeStyleRules

    CORRECT_SIMPLE_DESCRIPTIVE_NAMES = (
        "Names are technically correct, simple and descriptive, in that order. An "
        "inaccurate name is worse than a vague one, because a reader who trusts it "
        "stops reading"
    )

    NAMES_READ_ON_THEIR_OWN = (
        "A name stays correct and understandable when read on its own, without its "
        "class or a keyword argument next to it - after `value = "
        "instance.attribute`, in a log line, in a traceback, or when passed on "
        "positionally. A `Pipe` field `size` says nothing once it leaves the "
        "class; `inner_diameter` does. A generic word that would fit anything "
        "never passes this test"
    )

    USE_PLAIN_WORDS = (
        "Use the plain word every reader already knows. Use a specialist, "
        "metaphorical or in-house term only where it is genuinely the precise "
        "word, never as shorthand between the people who were in the discussion"
    )

    USE_THE_DOMAIN_WORD = (
        "Where the domain or file format already has a word for something, use "
        "that word"
    )

    NEVER_ABBREVIATE = "Never abbreviate an identifier"

    NAME_WHAT_NOT_HOW = (
        "A name says *what* a thing is or does - never *how* it does it, *when* it "
        "runs, or the layer or mechanism it is built on. Keep it short"
    )

    NO_FOREIGN_VOCABULARY = (
        "A name whose meaning has to be looked up elsewhere is wrong. Never adopt "
        "another system's vocabulary as an identifier of ours: name the thing for "
        "what it is here, and explain a foreign shape in the docstring"
    )

    VERB_AND_NOUN_PHRASES = (
        "Methods are verb phrases for what they do; classes and attributes are "
        "noun phrases for what they are. Name a field for its subject, not for the "
        "shape of its value"
    )

    ONE_NAME_PER_OPERATION = (
        "One operation has one name throughout a module. Where callers depend on "
        "that name, declare it in a base class instead of leaving it a convention "
        "each class is trusted to follow"
    )

    NAME_ENUM_MEMBERS_FOR_THEIR_SITUATION = (
        "Name an enum member for the situation it means, not for the function it "
        "dispatches to or the text it renders"
    )

    NO_REPEATED_WORDS = (
        "Never repeat the enclosing type's name in its members, or the same word "
        "twice within one name"
    )

    NEVER_SHADOW_BOUND_NAMES = (
        "Never take an identifier the language or something in scope already "
        f"binds: {CodeReference(Enum).name} reserves `name`, a parameter `field` "
        f"shadows {CodeReference(dataclasses.field).path}, a field shadows a method of the same name. These "
        "fail at runtime or silently, not at import"
    )

    FINISH_RENAMES = (
        "A rename is finished only when every reader of the old name, docstrings "
        "and comments included, uses the new one and the tests pass. A mechanical "
        "rename across a file is exactly where a method and a field converge on "
        "one name"
    )

    SUSPECT_VAGUE_CODE = (
        "When no honest specific name exists, suspect the code, not your "
        "vocabulary: a thing that can only be described vaguely usually has no "
        "single subject (a container for whatever its caller passed, a function "
        "doing two jobs). Remove it instead of hunting for a better word"
    )
