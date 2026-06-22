from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing_extensions import Dict, Iterable, List, Optional, Set, Tuple
import uuid

from krrood.entity_query_language.verbalization.fragments.base import (
    map_structural_children,
    NounPhrase,
    PossessiveChain,
    Fragment,
)
from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    attribute_fragment,
    possessive_path,
    pronominal_path,
)
from krrood.entity_query_language.verbalization.rendering.discourse import (
    DiscourseView,
    EMPTY_DISCOURSE,
)


@dataclass
class SubjectFrame:
    """One entry of the coreference pass's subject stack — the current pronoun-eligible subject."""

    subject_id: Optional[uuid.UUID]
    """The subject's referent id, or ``None`` for a scope with no single subject (e.g. ``SetOf``)."""

    number: Number = Number.SINGULAR
    """The subject's grammatical number — selects *"its"* (singular) vs. *"their"* (plural). Filled
    from the subject's own noun phrase when the pass walks it (rules supply no number)."""


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

    The discourse subject is **not** marked by the rules: each fragment carries the EQL node it was
    built from (its ``source``), and this pass asks the :class:`DiscourseView` who the focus of a
    query-sourced fragment is. So a scope opens wherever a query node's fragment appears, and the
    rules emit only structure + identity.

    Stateful per pass: the walk threads ``_seen`` and ``_subject_stack``.

    Reference: Reiter & Dale (2000) — referring-expression generation as a microplanning subtask;
    Gatt & Reiter (2009), SimpleNLG — ordered realisation stages.
    """

    discourse: DiscourseView = EMPTY_DISCOURSE
    """The focus-per-scope view, consulted to open a scope at a query-sourced fragment."""

    numbered_labels: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Disambiguation numbers (*"Robot 1"*) for referents the rules cannot label themselves — a
    relational referent's relative clause is built in the microplanner, with no access to the
    referring service, so the number is stamped on here instead."""

    _seen: Set[uuid.UUID] = field(init=False, default_factory=set)
    """Referent ids already mentioned at the current point of the walk."""

    _subject_stack: List[SubjectFrame] = field(init=False, default_factory=list)
    """Stack of :class:`SubjectFrame` entries — the number selects *"its"*/*"their"*."""

    _center: Optional[uuid.UUID] = field(init=False, default=None)
    """The local centre — the grammatical subject of the immediately preceding clause (see
    :meth:`_chain_topic`). The next chain's attribute on that referent pronominalises to *"its …"*;
    a clause about something else (or one headed by an attribute) clears it, so *"its"* binds only
    to the referent that was the subject just before (centering theory, Grosz/Joshi/Weinstein
    1995)."""

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
        :return: A new tree with referring noun phrases resolved.
        """
        self._seen = set(already_seen or ())
        self._subject_stack = []
        self._center = None
        return self._walk(fragment)

    def _walk(self, fragment: Fragment) -> Fragment:
        """Document-order rebuild, threading the accumulating discourse state.

        A fragment built from a query node opens a discourse scope whose focus the
        :class:`DiscourseView` supplies (``None`` suppresses pronominalisation, e.g. a set-of).
        """
        if self.discourse.is_scope(fragment.source):
            self._subject_stack.append(
                SubjectFrame(self.discourse.focus_of(fragment.source))
            )
            try:
                return self._dispatch(fragment)
            finally:
                self._subject_stack.pop()
        return self._dispatch(fragment)

    def _dispatch(self, fragment: Fragment) -> Fragment:
        """Resolve *fragment* by kind (the scope, if any, is already on the stack)."""
        match fragment:
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
        mention).

        The built chain is walked, not returned raw: a relational hop emits a referring noun phrase
        (*"the Robot to which it is assigned"*), so the walk resolves its first/repeat mention — a
        second mention of the same navigation reduces to a bare *"the Robot"*.

        After resolving, the centre advances to this clause's subject — unless the clause was itself
        an *"its …"* continuation of the current centre, which *keeps* it (a centering CONTINUE), so
        a run of attributes on one referent stays uniformly pronominal (*"its battery … its
        power"*) rather than mixing *"its battery … the power of the Robot"*."""
        reduced_quantity = self._reduced_selected_quantity(possessive_chain)
        if reduced_quantity is not None:
            resolved = reduced_quantity
        elif (built := self._relational_possessive(possessive_chain)) is not None:
            # An "its …" continuation keeps the centre it referred to.
            return self._walk(built)
        elif self._pronominalises(possessive_chain):
            subject_number = self._subject_stack[-1].number
            resolved = self._walk(
                pronominal_path(possessive_chain.parts, subject_number)
            )
        else:
            resolved = self._walk(
                possessive_path(possessive_chain.parts, possessive_chain.root_fragment)
            )
        self._center = self._chain_topic(possessive_chain.parts)
        return resolved

    def _reduced_selected_quantity(
        self, possessive_chain: PossessiveChain
    ) -> Optional[Fragment]:
        """A query's selected / measured quantity (an aggregation's measured attribute) spells out
        in full where it is first named — *"the average of the battery of the Robot to which a
        Mission is assigned"* — and a later mention of that same quantity (a WHERE on the very
        attribute being aggregated) reduces to a bare *"the battery"* rather than repeating it.

        :return: The bare definite attribute on a repeat of the selected quantity, or ``None`` when
            the chain is not a selected quantity (or is its first mention, recorded here).
        """
        if not self.discourse.is_selected_quantity(possessive_chain.node_id):
            return None
        if possessive_chain.node_id not in self._seen:
            self._seen.add(possessive_chain.node_id)
            return None
        return NounPhrase(
            head=attribute_fragment(possessive_chain.parts[-1]),
            definiteness=Definiteness.DEFINITE,
        )

    @staticmethod
    def _outermost_relation(parts: List[PathStep]) -> Optional[Tuple[uuid.UUID, List]]:
        """:return: ``(referent_id, tail)`` for the chain's outermost relational hop — the related
        entity it introduces and the attributes hanging off it — or ``None`` when the chain has no
        relational hop (or it carries no referent id)."""
        for index in reversed(range(len(parts))):
            if parts[index].is_relation:
                referent_id = parts[index].relation.referent_id
                return (
                    None if referent_id is None else (referent_id, parts[index + 1 :])
                )
        return None

    def _chain_topic(self, parts: List[PathStep]) -> Optional[uuid.UUID]:
        """:return: The referent the clause just said is grammatically *about* — the head of its
        subject noun phrase — to carry forward as the local centre for *"its …"*.

        A relational referent is the subject head only when nothing is predicated of an attribute of
        it: a boolean predicative (*"the Robot to which it is assigned is operational"*) or a bare
        relational mention. A genitive attribute (*"the **battery** of the Robot …"*) is itself the
        subject head and displaces the referent — so a following attribute of that referent is not
        *"its power"* (which a reader would bind to the battery) but spells out *"the power of the
        Robot"*. This is also why an aggregation's measured quantity (*"the average of the battery
        …"*) clears the centre: the battery heads the phrase, not the Robot. ``None`` when no
        relational referent heads the clause."""
        relation = self._outermost_relation(parts)
        if relation is None:
            return None
        referent_id, tail = relation
        return referent_id if not tail else None

    def _relational_possessive(
        self, possessive_chain: PossessiveChain
    ) -> Optional[Fragment]:
        """An attribute reached *through* the local centre reads as *"its <attribute>"* rather than
        re-naming the referent — after *"the Robot to which it is assigned is operational"* a
        following *"the battery of the Robot"* becomes *"its battery"*.

        The centre is the *subject* of the clause just said (see :meth:`_chain_topic`), so *"its"*
        binds where a reader binds it — to that subject — and never to an attribute that merely
        happened to be named more recently.

        :return: The pronominalised tail, or ``None`` when the chain's relational referent is not the
            current centre (leaving the subject / possessive forms to apply).
        """
        relation = self._outermost_relation(possessive_chain.parts)
        if relation is None:
            return None
        referent_id, tail = relation
        if referent_id != self._center or not tail:
            return None
        return pronominal_path(tail, Number.SINGULAR)

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
        """Every mention (singular or plural) marks its referent introduced.  A repeat **singular**
        mention is reduced to its head — dropping the first-mention modifiers and keeping the head
        label (*"a Robot, where …"* → *"the Robot"*, *"Robot 1 to which …"* → *"Robot 1"*).  A
        plural mention (*"Robots"*) only introduces the referent (it never carries an article).
        Relational referents are first numbered (*"Robot 1"*) when their type collides.

        :return: The resolved referring noun phrase (first / repeat), or the non-referring noun
            phrase rebuilt around its recursed children.
        """
        if noun_phrase.referent_id is None:
            return self._rebuilt(noun_phrase)
        noun_phrase = self._numbered(noun_phrase)
        self._record_subject_number(noun_phrase)
        repeat = noun_phrase.referent_id in self._seen
        self._seen.add(noun_phrase.referent_id)
        if repeat and noun_phrase.number is Number.SINGULAR:
            return self._reduced(noun_phrase)
        return self._rebuilt(noun_phrase)

    def _numbered(self, noun_phrase: NounPhrase) -> NounPhrase:
        """Stamp a disambiguation number on a referent the rules could not label themselves — a
        relational referent arrives as a definite *"the Robot to which …"*, and becomes a bare
        *"Robot 1"* clause when its type collides. A rule-labelled referent (already ``BARE``) is
        left untouched, so this never re-numbers a variable.

        :return: The numbered noun phrase, or *noun_phrase* unchanged when it has no number.
        """
        label = self.numbered_labels.get(noun_phrase.referent_id)
        if label is None or noun_phrase.definiteness is Definiteness.BARE:
            return noun_phrase
        return replace(
            noun_phrase,
            head=replace(noun_phrase.head, text=label),
            definiteness=Definiteness.BARE,
        )

    def _reduced(self, noun_phrase: NounPhrase) -> Fragment:
        """:return: A repeat mention reduced to its head — the first-mention modifiers dropped — as a
        bare label (*"Robot 1"*) when numbered, else a definite reference (*"the Robot"*).
        """
        return NounPhrase(
            head=self._walk(noun_phrase.head),
            number=noun_phrase.number,
            definiteness=(
                Definiteness.BARE
                if noun_phrase.definiteness is Definiteness.BARE
                else Definiteness.DEFINITE
            ),
            referent_id=noun_phrase.referent_id,
        )

    def _record_subject_number(self, noun_phrase: NounPhrase) -> None:
        """If this noun phrase *is* an enclosing scope's subject, record its grammatical number on
        that frame — the subject is rendered before the chains that refer to it, so the number
        (*"its"*/*"their"*) is known by the time pronominalisation is decided. Rules supply none.
        """
        for frame in reversed(self._subject_stack):
            if frame.subject_id == noun_phrase.referent_id:
                frame.number = noun_phrase.number
                return

    def _rebuilt(self, noun_phrase: NounPhrase) -> NounPhrase:
        """Rebuild *noun_phrase* with its head and modifiers recursed (document order preserved).

        A referring noun phrase is the discourse subject *of its own modifiers*: a restrictive
        modifier predicates over the head, so a chain rooted at the head pronominalises (*"a Robot
        whose battery exceeds its threshold"*). This is inferred from structure — the modifiers
        slot — so rules never mark the scope."""
        head = self._walk(noun_phrase.head)
        if noun_phrase.referent_id is None or not noun_phrase.subject_of_modifiers:
            modifiers = [self._walk(modifier) for modifier in noun_phrase.modifiers]
        else:
            self._subject_stack.append(
                SubjectFrame(noun_phrase.referent_id, noun_phrase.number)
            )
            try:
                modifiers = [self._walk(modifier) for modifier in noun_phrase.modifiers]
            finally:
                self._subject_stack.pop()
        return replace(noun_phrase, head=head, modifiers=modifiers)
