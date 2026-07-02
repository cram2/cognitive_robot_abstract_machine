from __future__ import annotations

from enum import StrEnum


class Separator(StrEnum):
    """How the parts of a phrase are joined in the rendered output — the closed set of inline
    separators a :class:`~krrood.entity_query_language.verbalization.fragments.base.PhraseFragment`
    uses (consistent with the formatter-side ``BulletStyle`` / ``IndentSize`` enums).  A
    :class:`StrEnum`, so a member is itself the join string (``separator.join(parts)`` just works).
    """

    SPACE = " "
    """A single space between words (the default)."""
    NONE = ""
    """No separator — parts abut directly (the orthography pass owns the spacing)."""
    COMMA = ", "
    """A comma and space for inline coordinated lists (*"a, b, or c"*) and tuple selections."""


class GrammaticalNumber(StrEnum):
    """Grammatical number of a noun or verb (singular vs. plural)."""

    SINGULAR = "singular"
    """A single entity."""
    PLURAL = "plural"
    """More than one entity."""

    @classmethod
    def of(cls, is_plural: bool) -> GrammaticalNumber:
        """
        :param is_plural: Whether the number is plural.
        :return: ``PLURAL`` when *is_plural* else ``SINGULAR``.

        >>> GrammaticalNumber.of(True)
        <GrammaticalNumber.PLURAL: 'plural'>
        >>> GrammaticalNumber.of(False)
        <GrammaticalNumber.SINGULAR: 'singular'>
        """
        return cls.PLURAL if is_plural else cls.SINGULAR


class Definiteness(StrEnum):
    """Grammatical definiteness of a noun phrase (*"a/an"* vs. *"the"* vs. no determiner)."""

    BARE = "bare"
    """No determiner — a numbered label (*"Robot 2"*) or a bare predicate noun."""
    INDEFINITE = "indefinite"
    """First, non-specific mention — *"a/an Robot"*, or a bare plural *"Robots"*."""
    DEFINITE = "definite"
    """Identifiable or subsequent mention — *"the Robot"* / *"the Robots"*."""
    UNIQUE = "unique"
    """A uniqueness-quantified first mention — *"the unique Robot"*."""


class Spacing(StrEnum):
    """Orthographic spacing of a token relative to its neighbours."""

    NONE = "none"
    """Spaced on both sides like a normal word (the default)."""
    LEFT = "left"
    """No space *before* this token — it hugs the preceding token (*","* → *"x,"*)."""
    RIGHT = "right"
    """No space *after* this token — it hugs the following token (*"("* → *"(x"*)."""
