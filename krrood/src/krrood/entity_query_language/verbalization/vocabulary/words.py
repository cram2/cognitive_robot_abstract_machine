from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, StrEnum
from typing_extensions import ClassVar

from krrood.entity_query_language.verbalization.fragments.base import (
    RoleFragment,
    Fragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (  # noqa: F401  (re-export)
    Spacing,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole

# ``Number`` is defined in ``fragments.features`` (below both this lexicon and the fragment
# tree, to avoid a cycle) and re-exported here for the ``for_number`` selectors and importers.


@dataclass(frozen=True)
class PlainWord:
    """
    A neutral word/phrase with no semantic role.
    """

    text: str
    """The raw English text (e.g. ``"of"``, ``"and"``, ``"the"``, ``"ascending"``)."""

    def as_fragment(self) -> WordFragment:
        """:return: A neutral word fragment."""
        return WordFragment(text=self.text)


@dataclass(frozen=True)
class PunctuationWord(PlainWord):
    """
    A punctuation token carrying an orthographic spacing feature, so the orthography pass drops
    the space adjacent to it.
    """

    spacing: Spacing = Spacing.NONE
    """Which side hugs its neighbour (``LEFT`` for *","* / *")"*; ``RIGHT`` for *"("*)."""

    def as_fragment(self) -> WordFragment:
        """:return: A role-less word fragment carrying this token's spacing."""
        return WordFragment(text=self.text, spacing=self.spacing)


@dataclass(frozen=True)
class RoleWord:
    """
    A word/phrase carrying a fixed semantic role.

    Subclasses set ``_role_`` as a ``ClassVar`` so it never appears in ``__init__``.
    """

    text: str
    """The raw English text."""

    _role_: ClassVar[SemanticRole]
    """The semantic role for this word type (set by each subclass)."""

    def as_fragment(self) -> RoleFragment:
        """:return: A role-tagged fragment."""
        return RoleFragment(text=self.text, role=self._role_)


class KeyWord(RoleWord):
    """EQL structure keyword with the ``KEYWORD`` role."""

    _role_ = SemanticRole.KEYWORD


class LogicalWord(RoleWord):
    """Logical connective with the ``LOGICAL`` role."""

    _role_ = SemanticRole.LOGICAL


class ChildForm(StrEnum):
    """How an aggregation word verbalizes its child expression."""

    PLURAL = "plural"
    """The child is rendered in plural form, joined by *"of"*: *"sum of amounts of
    BankTransactions"*. The bare aggregation noun (*"sum"*) is the phrase head, so a repeat mention
    reduces cleanly to *"the sum"*."""
    SINGULAR = "singular"
    """The child is rendered singular, joined by *"of"*: *"maximum of the amount of …"*."""
    NONE = "none"
    """The aggregation word takes no child (e.g. ``"count of all"``); it renders bare."""


@dataclass(frozen=True)
class AggregationWord(RoleWord):
    """
    Aggregation phrase with the ``AGGREGATION`` role.
    """

    _role_ = SemanticRole.AGGREGATION
    child_form: ChildForm = ChildForm.PLURAL
    """Controls how the child expression is verbalized."""


class OperatorWord(RoleWord):
    """Comparison operator phrase with the ``OPERATOR`` role."""

    _role_ = SemanticRole.OPERATOR


class PronounWord(RoleWord):
    """
    A coreference pronoun (e.g. *"its"*) standing in for a previously introduced variable.
    Carries the ``VARIABLE`` role so it is coloured like the variable it refers to.
    """

    _role_ = SemanticRole.VARIABLE


@dataclass(frozen=True)
class OperatorPhrase:
    """
    All eight text variants for one comparison operator, co-located.

    The three boolean flags ``negated``, ``compact``, and ``temporal`` select among the eight
    fields.
    """

    standard: str
    """Default form (e.g. ``"is greater than"``)."""

    compact: str
    """Copula-less form used in HAVING clauses (e.g. ``"greater than"``)."""

    negated: str
    """Negated form (e.g. ``"is not greater than"``)."""

    negated_compact: str
    """Negated copula-less form (e.g. ``"not greater than"``)."""

    temporal: str = ""
    """Temporal standard form (e.g. ``"is after"``)."""

    temporal_compact: str = ""
    """Temporal compact form (e.g. ``"after"``)."""

    temporal_negated: str = ""
    """Temporal negated form (e.g. ``"is no later than"``)."""

    temporal_negated_compact: str = ""
    """Temporal negated compact form (e.g. ``"no later than"``)."""

    def select(
        self, *, negated: bool = False, compact: bool = False, temporal: bool = False
    ) -> OperatorWord:
        """
        Select the appropriate operator word for the given flag combination.

        Falls back to the standard form when the corresponding temporal field is empty (e.g. for
        operators without temporal variants).

        :param negated: Use the negated variant.
        :param compact: Use the copula-less variant (for HAVING clauses).
        :param temporal: Use the temporal variant (for datetime comparisons).
        :return: The appropriate operator word.
        """
        if temporal:
            if negated and compact:
                text = self.temporal_negated_compact
            elif negated:
                text = self.temporal_negated
            elif compact:
                text = self.temporal_compact
            else:
                text = self.temporal
        elif negated and compact:
            text = self.negated_compact
        elif negated:
            text = self.negated
        elif compact:
            text = self.compact
        else:
            text = self.standard
        return OperatorWord(text=text or self.standard)


class VocabEnum(Enum):
    """
    Enum mixin for enums whose member values are word instances.

    Delegates ``as_fragment()`` and ``text`` to the dataclass stored in ``.value`` so callers
    never write ``.value`` explicitly.
    """

    def as_fragment(self) -> Fragment:
        """:return: The fragment representing this vocabulary item."""
        return self.value.as_fragment()

    @property
    def text(self) -> str:
        """:return: The raw English text of this vocabulary item."""
        return self.value.text
