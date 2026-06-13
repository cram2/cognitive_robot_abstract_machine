from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing_extensions import Iterable, List, Optional, Set
import uuid

from krrood.entity_query_language.verbalization.fragments.base import (
    map_structural_children,
    NounPhrase,
    PossessiveChain,
    SubjectScope,
    Fragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    possessive_path,
    pronominal_path,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Pronouns


@dataclass(frozen=True)
class SubjectFrame:
    """One entry of the coreference pass's subject stack — the current pronoun-eligible subject."""

    subject_id: Optional[uuid.UUID]
    """The subject's referent id, or ``None`` for a scope with no single subject (e.g. ``SetOf``)."""

    number: Number
    """The subject's grammatical number — selects *"its"* (singular) vs. *"their"* (plural)."""


@dataclass
class CoreferenceProcessor:
    """
    Resolve every referring noun phrase in document order (first / repeat / pronoun) — the one
    place the discourse (coreference) decision is made.

    A referring expression is named differently each time it appears: an indefinite first mention
    (*"a Robot"*), a definite subsequent mention (*"the Robot"*), or a pronoun (*"its …"*) when it
    is the current discourse subject. Rules emit the first-mention form; this pass walks the
    finished tree in document order, tracking which referents have been introduced and which is
    the current subject, and downgrades every repeat mention to a definite reference (dropping the
    first-mention modifiers, keeping the head label) or a pronoun.

    Stateful per pass: the walk threads ``_seen`` and ``_subject_stack``.

    Reference: Reiter & Dale (2000) — referring-expression generation as a microplanning subtask;
    Gatt & Reiter (2009), SimpleNLG — ordered realisation stages.
    """

    _seen: Set[uuid.UUID] = field(init=False, default_factory=set)
    """Referent ids already mentioned at the current point of the walk."""

    _subject_stack: List[SubjectFrame] = field(init=False, default_factory=list)
    """Stack of :class:`SubjectFrame` entries — the number selects *"its"*/*"their"*."""

    def process(
        self,
        fragment: Fragment,
        already_seen: Optional[Iterable[uuid.UUID]] = None,
    ) -> Fragment:
        """
        :param fragment: Root of the fragment tree.
        :param already_seen: Referents introduced by *prior* builds sharing the same context
            (so the same expression verbalized twice against one context reads *"a Robot"* then
            *"the Robot"*).  These are treated as already-mentioned before the walk begins.
        :return: A new tree with referring noun phrases resolved and ``SubjectScope`` markers
            stripped.
        """
        self._seen = set(already_seen or ())
        self._subject_stack = []
        return self._walk(fragment)

    def _walk(self, fragment: Fragment) -> Fragment:
        """Document-order rebuild, threading the accumulating discourse state."""
        match fragment:
            case SubjectScope(
                subject_id=subject_id, child=child, subject_number=subject_number
            ):
                self._subject_stack.append(SubjectFrame(subject_id, subject_number))
                try:
                    return self._walk(child)
                finally:
                    self._subject_stack.pop()
            case NounPhrase():
                return self._noun_phrase(fragment)
            case PossessiveChain():
                return self._possessive_chain(fragment)
            case _:
                rebuilt = map_structural_children(fragment, self._walk)
                return rebuilt if rebuilt is not None else fragment

    def _possessive_chain(self, possessive_chain: PossessiveChain) -> Fragment:
        """:return: The chain as *"its/their …"* when its root is the current subject (the
        pronoun agreeing with the subject's number — *"their"* for a plural population), else as
        the possessive *"the … of <root>"* (resolving the root noun phrase for first/subsequent
        mention)."""
        if self._pronominalises(possessive_chain):
            subject_number = self._subject_stack[-1].number
            pronoun = (
                Pronouns.THEIR if subject_number is Number.PLURAL else Pronouns.ITS
            )
            return pronominal_path(possessive_chain.parts, pronoun.as_fragment())
        return possessive_path(
            possessive_chain.parts, self._walk(possessive_chain.root_fragment)
        )

    def _pronominalises(self, possessive_chain: PossessiveChain) -> bool:
        """:return: ``True`` when the chain root is the current, already-introduced, non-numbered subject."""
        if (
            possessive_chain.root_referent_id is None
            or possessive_chain.root_referent_id not in self._seen
        ):
            return False
        if (
            not self._subject_stack
            or self._subject_stack[-1].subject_id != possessive_chain.root_referent_id
        ):
            return False
        # A numbered root ("Robot 2") renders BARE and is never pronominalised.
        return not (
            isinstance(possessive_chain.root_fragment, NounPhrase)
            and possessive_chain.root_fragment.definiteness is Definiteness.BARE
        )

    def _noun_phrase(self, noun_phrase: NounPhrase) -> Fragment:
        """Every mention (singular or plural) marks its referent introduced.  Only a **repeat
        singular** mention is downgraded to a definite reference — dropping the first-mention
        modifiers and keeping the head label (*"a Robot, where …"* → *"the Robot"*).  A plural
        mention (*"Robots"*) only introduces the referent (it never carries an article), and a
        ``BARE`` numbered label (*"Robot 2"*) never downgrades.

        :return: The resolved referring noun phrase (first / repeat), or the non-referring noun
            phrase rebuilt around its recursed children.
        """
        if noun_phrase.referent_id is None:
            return self._rebuilt(noun_phrase)
        repeat = noun_phrase.referent_id in self._seen
        self._seen.add(noun_phrase.referent_id)
        downgrade = (
            repeat
            and noun_phrase.number is Number.SINGULAR
            and noun_phrase.definiteness is not Definiteness.BARE
        )
        if downgrade:
            return NounPhrase(
                head=self._walk(noun_phrase.head),
                number=noun_phrase.number,
                definiteness=Definiteness.DEFINITE,
                referent_id=noun_phrase.referent_id,
            )
        return self._rebuilt(noun_phrase)

    def _rebuilt(self, noun_phrase: NounPhrase) -> NounPhrase:
        """Rebuild *noun_phrase* with its head and modifiers recursed (document order preserved)."""
        return replace(
            noun_phrase,
            head=self._walk(noun_phrase.head),
            modifiers=[self._walk(modifier) for modifier in noun_phrase.modifiers],
        )
