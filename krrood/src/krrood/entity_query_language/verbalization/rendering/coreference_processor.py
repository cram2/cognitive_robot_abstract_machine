from __future__ import annotations

from dataclasses import dataclass, field, replace
from typing_extensions import Dict, Iterable, List, Optional, Set, Tuple
import uuid

from krrood.entity_query_language.verbalization.fragments.base import (
    Clause,
    map_structural_children,
    NounPhrase,
    OwnedAttributes,
    PhraseFragment,
    PossessiveChain,
    VerbalizationFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    attribute_fragment,
    chain_head_number,
    coordinated_genitive,
    possessive_path,
    pronominal_path,
)
from krrood.entity_query_language.verbalization.rendering.discourse import (
    DiscourseView,
    EMPTY_DISCOURSE,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Pronouns,
    Punctuation,
)
from krrood.entity_query_language.verbalization.rendering.passes import RealizationPass


@dataclass
class SubjectFrame:
    """One entry of the coreference pass's subject stack — the current pronoun-eligible subject."""

    subject_id: Optional[uuid.UUID]
    """The subject's referent id, or ``None`` for a scope with no single subject (e.g. ``SetOf``)."""

    number: GrammaticalNumber = GrammaticalNumber.SINGULAR
    """The subject's grammatical number — selects *"its"* (singular) vs. *"their"* (plural). Filled
    from the subject's own noun phrase when the pass walks it (rules supply no number)."""


@dataclass
class CoreferenceProcessor(RealizationPass):
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

    Reference: :cite:t:`reiter2000building` — referring-expression generation as a microplanning subtask;
    :cite:t:`gatt2009simplenlg` — ordered realisation stages.

    Both chains rooted at the Employee subject are downgraded to the pronoun *its*:

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(an(entity(employee).where(employee.salary > employee.starting_salary)))
    'Find an Employee such that its salary is greater than its starting_salary'
    """

    discourse: DiscourseView = EMPTY_DISCOURSE
    """The focus-per-scope view, consulted to open a scope at a query-sourced fragment."""

    numbered_labels: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Disambiguation numbers (*"Robot 1"*) for referents the rules cannot label themselves — a
    relational referent's relative clause is built in the microplanner, with no access to the
    referring service, so the number is stamped on here instead."""

    previously_introduced_referents: Iterable[uuid.UUID] = ()
    """Referents introduced by *prior* builds sharing the same context, so the same expression
    verbalized twice against one context reads *"a Robot"* then *"the Robot"* — treated as
    already-mentioned before the walk begins."""

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

    def process(self, fragment: VerbalizationFragment) -> VerbalizationFragment:
        """
        :param fragment: Root of the fragment tree.
        :return: A new tree with referring noun phrases resolved (first / repeat / pronoun), seeded
            with the referents from prior builds (:attr:`previously_introduced_referents`).

        Verbalizing one expression twice against a shared context reads *"a Robot"* then *"the
        Robot"* — the second build's referents were seeded from the first:

        >>> from krrood.entity_query_language.verbalization.context import MicroplanningServices
        >>> from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> robot = variable(Robot, [])
        >>> query = a(entity(robot))
        >>> shared = MicroplanningServices.from_expression(robot)
        >>> flatten_fragment_to_plain_text(EQLVerbalizer().build(query, shared))
        'Find a Robot'
        >>> flatten_fragment_to_plain_text(EQLVerbalizer().build(query, shared))
        'Find the Robot'
        """
        self._seen = set(self.previously_introduced_referents)
        self._subject_stack = []
        self._center = None
        return self._walk(fragment)

    def _walk(self, fragment: VerbalizationFragment) -> VerbalizationFragment:
        """Document-order rebuild, threading the accumulating discourse state.

        A fragment built from a query node opens a discourse scope whose focus the
        :class:`DiscourseView` supplies (``None`` suppresses pronominalisation, e.g. a set-of).

        Here it is :meth:`_walk` that opens the Mission's scope, so the *it* in *the Robot to which
        it is assigned* resolves to the in-scope Mission subject (with no scope open, that *it* would
        instead re-name the whole *a Mission*):

        >>> mission = variable(Mission, [])
        >>> verbalize_expression(a(entity(mission).where(mission.assigned_to.battery > 50)))
        'Find a Mission such that the battery of the Robot to which it is assigned is greater than 50'
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

    def _dispatch(self, fragment: VerbalizationFragment) -> VerbalizationFragment:
        """Resolve *fragment* by kind (the scope, if any, is already on the stack).

        Its contribution is the routing: each possessive chain (*its salary*, *its starting_salary*)
        is sent to :meth:`_possessive_chain`; a plain structural node would instead be rebuilt around
        its recursed children.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).where(employee.salary > employee.starting_salary)))
        'Find an Employee such that its salary is greater than its starting_salary'
        """
        match fragment:
            case NounPhrase():
                return self._noun_phrase(fragment)
            case PossessiveChain():
                return self._possessive_chain(fragment)
            case OwnedAttributes():
                return self._owned_attributes(fragment)
            case Clause():
                return self._subject_clause(fragment)
            case PhraseFragment(parts=[PossessiveChain(), *_]):
                return self._predicate_clause(fragment)
            case _:
                rebuilt = map_structural_children(fragment, self._walk)
                return rebuilt if rebuilt is not None else fragment

    def _predicate_clause(self, clause: PhraseFragment) -> VerbalizationFragment:
        """A clause led by its subject chain (*"<subject> <copula> <value>"* — a comparator
        predicate or a *"… is between …"* range) whose finite copula agrees with that subject.

        The build tags the copula singular; coreference is what may realise the subject as a plural
        population (*"their batteries"*), distributing a scalar attribute over the quantified
        variable. Since that population reading is decided here — with the pronominalisation — the
        copula's agreement is decided here too, in one place, rather than pre-baked into the
        quantifier rule (which cannot know whether the body's grammatical subject is the variable).
        A singular subject leaves the copula untouched, so every plain predicate is unaffected.

        Both surfaces agree off the same plural subject (*"are"* / *"are between"*):

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(for_all(robot, robot.battery > 0))
        'for all Robots, their batteries are greater than 0'
        >>> robot = variable(Robot, [])
        >>> verbalize_expression(for_all(robot, and_(robot.battery > 10, robot.battery < 90)))
        'for all Robots, their batteries are between 10 and 90'
        """
        subject_number = self._clause_subject_number(clause.parts[0])
        rebuilt = map_structural_children(clause, self._walk)
        if subject_number is GrammaticalNumber.PLURAL:
            return self._with_agreed_copula(rebuilt, subject_number)
        return rebuilt

    def _clause_subject_number(
        self, subject: VerbalizationFragment
    ) -> GrammaticalNumber:
        """:return: The grammatical number the clause's subject is realised with — plural only when
        a pronominalised chain distributes a scalar leaf over a plural population (*"their
        batteries"*); singular for every other subject (a deeper chain, a non-pronominalised one, or
        a plain noun phrase the build already agreed)."""
        if not isinstance(subject, PossessiveChain) or not self._pronominalises(
            subject
        ):
            return GrammaticalNumber.SINGULAR
        return chain_head_number(subject.parts, self._subject_stack[-1].number)

    def _subject_clause(self, clause: Clause) -> VerbalizationFragment:
        """A part-of-speech predicate clause (*"<subject> <verb/copula> …"* built by
        :func:`~krrood.entity_query_language.verbalization.vocabulary.parts_of_speech.clause`).

        When the leading constituent is the current discourse subject, it pronominalises to the
        nominative *"it"* / *"they"* and the finite verb or copula agrees with that subject's number
        (*"is"* → *"are"*, *"works"* → *"work"*); otherwise the clause is rebuilt unchanged, so a
        plain predicate and a top-level mention are untouched. A clause whose subject is a navigation
        chain defers to :meth:`_predicate_clause` (the *"its/their …"* path).

        The body's subject is the quantified population, so it reads *"they"* and the copula agrees:

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(for_all(robot, IsReachable(robot)))
        'for all Robots, they are reachable'
        """
        subject = clause.parts[0]
        if isinstance(subject, PossessiveChain):
            return self._predicate_clause(clause)
        number = self._subject_pronoun_number(subject)
        if number is None:
            rebuilt = map_structural_children(clause, self._walk)
            return rebuilt if rebuilt is not None else clause
        pronoun = Pronouns.nominative(number).as_fragment()
        agreed_rest = [
            self._agree_finite(self._walk(part), number) for part in clause.parts[1:]
        ]
        return replace(clause, parts=[pronoun, *agreed_rest])

    def _subject_pronoun_number(
        self, subject: VerbalizationFragment
    ) -> Optional[GrammaticalNumber]:
        """:return: The number to pronominalise the clause subject with — the in-scope subject's
        number when *subject* is the current, already-introduced discourse subject, else ``None``
        (leaving the subject as its first/repeat noun-phrase mention)."""
        if not isinstance(subject, NounPhrase) or subject.referent_id is None:
            return None
        if subject.referent_id not in self._seen:
            return None
        if (
            not self._subject_stack
            or self._subject_stack[-1].subject_id != subject.referent_id
        ):
            return None
        return self._subject_stack[-1].number

    @staticmethod
    def _with_agreed_copula(
        clause: PhraseFragment, number: GrammaticalNumber
    ) -> VerbalizationFragment:
        """:return: *clause* with its finite copula tagged *number* (*"is"* → *"are"* once the
        morphology pass realises it). Only the operator slot's leading copula inflects — the subject
        and value (and any copula nested in a relative clause on either) are left untouched.
        """
        return replace(
            clause,
            parts=[
                CoreferenceProcessor._agree_finite(part, number)
                for part in clause.parts
            ],
        )

    _FINITE_ROLES = (SemanticRole.OPERATOR, SemanticRole.VERB)
    """The clause roles a finite predicate agrees through — the copula / comparison operator and a
    lexical verb."""

    @staticmethod
    def _agree_finite(
        part: VerbalizationFragment, number: GrammaticalNumber
    ) -> VerbalizationFragment:
        """:return: *part* re-tagged with *number* when it is the clause's finite slot — an
        ``OPERATOR`` or ``VERB`` leaf, or a phrase led by one (the factored *"is greater than"*) —
        else *part* unchanged. The copula inflects (*"is"* → *"are"*) and a lexical verb agrees
        (*"works"* → *"work"*); a non-copula operator (*"contains"*) is tagged too but the morphology
        pass leaves it be, so this never has to single the finite word out by text."""
        if (
            isinstance(part, RoleFragment)
            and part.role in CoreferenceProcessor._FINITE_ROLES
        ):
            return replace(part, number=number)
        leads_with_finite = (
            isinstance(part, PhraseFragment)
            and part.parts
            and isinstance(part.parts[0], RoleFragment)
            and part.parts[0].role in CoreferenceProcessor._FINITE_ROLES
        )
        if leads_with_finite:
            return replace(
                part,
                parts=[replace(part.parts[0], number=number), *part.parts[1:]],
            )
        return part

    def _owned_attributes(self, owned: OwnedAttributes) -> VerbalizationFragment:
        """:return: the owner's attributes as the possessive *"its/their <attrs>"* when the owner is
        the current subject, else the genitive *"the <attrs> of <owner>"* — the same pronominalise vs.
        spell-out choice :meth:`_possessive_chain` makes for a navigation chain, but for a coordinated
        attribute list whose owner has no further hops (e.g. a *"predict"* point on the selection).

        >>> verbalize_expression(underspecified(Robot)(name="R2", battery=...))
        "Generate a Robot and predict its battery value given that its name is 'R2'"
        """
        if self._owner_is_subject(owned):
            possessive = Pronouns.possessive(
                self._subject_stack[-1].number
            ).as_fragment()
            return PhraseFragment(parts=[possessive, self._walk(owned.attributes)])
        return coordinated_genitive(
            [self._walk(owned.attributes)], self._walk(owned.owner_fragment)
        )

    def _owner_is_subject(self, owned: OwnedAttributes) -> bool:
        """:return: whether *owned*'s owner is the current, already-introduced, non-numbered subject —
        the gate selecting the possessive *"its <attrs>"* over the genitive (the
        :class:`OwnedAttributes` analogue of :meth:`_pronominalises`)."""
        if owned.owner_referent_id is None or owned.owner_referent_id not in self._seen:
            return False
        if (
            not self._subject_stack
            or self._subject_stack[-1].subject_id != owned.owner_referent_id
        ):
            return False
        return not (
            isinstance(owned.owner_fragment, NounPhrase)
            and owned.owner_fragment.definiteness is Definiteness.BARE
        )

    def _possessive_chain(
        self, possessive_chain: PossessiveChain
    ) -> VerbalizationFragment:
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
        power"*) rather than mixing *"its battery … the power of the Robot"*.

        Its contribution here is choosing the chain's form: it renders this one through the relational
        hop as *the Robot to which it is assigned* (not a pronoun — the Robot is not the subject) and
        advances the centre to that Robot:

        >>> mission = variable(Mission, [])
        >>> verbalize_expression(a(entity(mission).where(mission.assigned_to.battery > 50)))
        'Find a Mission such that the battery of the Robot to which it is assigned is greater than 50'
        """
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
    ) -> Optional[VerbalizationFragment]:
        """A query's selected / measured quantity (an aggregation's measured attribute) spells out
        in full where it is first named — *"the average of the battery of the Robot to which a
        Mission is assigned"* — and a later mention of that same quantity (a WHERE on the very
        attribute being aggregated) reduces to a bare *"the battery"* rather than repeating it.

        :return: The bare definite attribute on a repeat of the selected quantity, or ``None`` when
            the chain is not a selected quantity (or is its first mention, recorded here).

        The measured ``battery`` spells out in full where the aggregation names it, and the WHERE on
        that same attribute reduces to a bare *"the battery"*:

        >>> mission = variable(Mission, [])
        >>> verbalize_expression(an(entity(max(mission.assigned_to.battery)).where(mission.assigned_to.battery > 5)))
        'Find the maximum of the battery of the Robot to which a Mission is assigned such that the battery is greater than 5'
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
        relational hop (or it carries no referent id).

        It singles out the *assigned_to* hop and the Robot it introduces — the related entity the
        pass goes on to centre on (so a later attribute of that Robot can read *its …*). The *it* in
        *to which it is assigned* is the Mission subject, pied-piped into the relative clause:

        >>> mission = variable(Mission, [])
        >>> verbalize_expression(a(entity(mission).where(mission.assigned_to.battery > 50)))
        'Find a Mission such that the battery of the Robot to which it is assigned is greater than 50'
        """
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
        relational referent heads the clause.

        After *the Robot to which it is assigned is operational* (a boolean predicative, so the Robot
        heads the clause), :meth:`_chain_topic` carries that Robot forward as the centre — which is
        why the next clause's attribute reads *its battery* rather than *the battery of the Robot*:

        >>> mission = variable(Mission, [])
        >>> verbalize_expression(a(entity(mission).where(and_(mission.assigned_to.operational, mission.assigned_to.battery > 50))))
        'Find a Mission such that the Robot to which it is assigned is operational, and its battery is greater than 50'
        """
        relation = self._outermost_relation(parts)
        if relation is None:
            return None
        referent_id, tail = relation
        return referent_id if not tail else None

    def _relational_possessive(
        self, possessive_chain: PossessiveChain
    ) -> Optional[VerbalizationFragment]:
        """An attribute reached *through* the local centre reads as *"its <attribute>"* rather than
        re-naming the referent — after *"the Robot to which it is assigned is operational"* a
        following *"the battery of the Robot"* becomes *"its battery"*.

        The centre is the *subject* of the clause just said (see :meth:`_chain_topic`), so *"its"*
        binds where a reader binds it — to that subject — and never to an attribute that merely
        happened to be named more recently.

        :return: The pronominalised tail, or ``None`` when the chain's relational referent is not the
            current centre (leaving the subject / possessive forms to apply).

        This is the step that produces the second clause's *its battery*: the Robot was made the
        centre by the first clause, so its following attribute pronominalises instead of repeating
        *the battery of the Robot*:

        >>> mission = variable(Mission, [])
        >>> verbalize_expression(a(entity(mission).where(and_(mission.assigned_to.operational, mission.assigned_to.battery > 50))))
        'Find a Mission such that the Robot to which it is assigned is operational, and its battery is greater than 50'
        """
        relation = self._outermost_relation(possessive_chain.parts)
        if relation is None:
            return None
        referent_id, tail = relation
        if referent_id != self._center or not tail:
            return None
        return pronominal_path(tail, GrammaticalNumber.SINGULAR)

    def _pronominalises(self, possessive_chain: PossessiveChain) -> bool:
        """:return: ``True`` when the chain root is the current, already-introduced, non-numbered subject.

        It is the gate that decides pronominalisation: returning ``True`` for each chain rooted at the
        Employee subject is what renders *its salary* / *its starting_salary* instead of *the salary
        of the Employee*.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).where(employee.salary > employee.starting_salary)))
        'Find an Employee such that its salary is greater than its starting_salary'
        """
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

    def _noun_phrase(self, noun_phrase: NounPhrase) -> VerbalizationFragment:
        """Every mention (singular or plural) marks its referent introduced.  A repeat **singular**
        mention is reduced to its head — dropping the first-mention modifiers and keeping the head
        label (*"a Robot, where …"* → *"the Robot"*, *"Robot 1, to which …,"* → *"Robot 1"*).  A
        plural mention (*"Robots"*) only introduces the referent (it never carries an article).
        Relational referents are first numbered (*"Robot 1"*) when their type collides.

        :return: The resolved referring noun phrase (first / repeat), or the non-referring noun
            phrase rebuilt around its recursed children.

        It is the entry that marks each Robot introduced and applies numbering — here both Robots are
        first mentions of one colliding type, so it hands them to :meth:`_numbered` and the result is
        *Robot 1* / *Robot 2* (a lone Robot would stay *a Robot*):

        >>> robot_one, robot_two = variable(Robot, []), variable(Robot, [])
        >>> verbalize_expression(a(entity(robot_one).where(robot_one.battery > robot_two.battery)))
        'Find Robot 1 whose battery is greater than the battery of Robot 2'
        """
        if noun_phrase.referent_id is None:
            return self._rebuilt(noun_phrase)
        noun_phrase = self._numbered(noun_phrase)
        self._record_subject_number(noun_phrase)
        repeat = noun_phrase.referent_id in self._seen
        self._seen.add(noun_phrase.referent_id)
        if repeat and noun_phrase.number is GrammaticalNumber.SINGULAR:
            return self._reduced(noun_phrase)
        return self._rebuilt(noun_phrase)

    def _numbered(self, noun_phrase: NounPhrase) -> NounPhrase:
        """Stamp a disambiguation number on a referent the rules could not label themselves — a
        relational referent arrives as a definite *"the Robot to which …"*, and becomes a bare
        *"Robot 1"* clause when its type collides. A rule-labelled referent (already ``BARE``) is
        left untouched, so this never re-numbers a variable.

        :return: The numbered noun phrase, or *noun_phrase* unchanged when it has no number.

        It is the step that supplies the digits: the two same-type Robots collide, so it stamps each
        with its disambiguation label — the *1* and *2* in *Robot 1* / *Robot 2*:

        >>> robot_one, robot_two = variable(Robot, []), variable(Robot, [])
        >>> verbalize_expression(a(entity(robot_one).where(robot_one.battery > robot_two.battery)))
        'Find Robot 1 whose battery is greater than the battery of Robot 2'
        """
        label = self.numbered_labels.get(noun_phrase.referent_id)
        if label is None or noun_phrase.definiteness is Definiteness.BARE:
            return noun_phrase
        # A numbered head is identified by its number, so a relative clause on it is non-restrictive
        # and is set off by commas. The trailing comma never dangles: numbering requires two same-type
        # relational referents, which only arise inside a predicate (genitive owner / "is <attribute>"
        # subject), so following text always supplies the post-comma space.
        modifiers = (
            self._comma_framed(noun_phrase.modifiers)
            if noun_phrase.relative_clause
            else noun_phrase.modifiers
        )
        return replace(
            noun_phrase,
            head=replace(noun_phrase.head, text=label),
            definiteness=Definiteness.BARE,
            modifiers=modifiers,
        )

    @staticmethod
    def _comma_framed(
        modifiers: List[VerbalizationFragment],
    ) -> List[VerbalizationFragment]:
        """:return: *modifiers* set off by a leading and trailing comma, so a non-restrictive relative
        clause reads *"Robot 1, to which its primary is assigned, is …"*. The comma's
        :attr:`Spacing.LEFT` lets the orthography pass hug it to the preceding token."""
        comma = Punctuation.COMMA.as_fragment()
        return [comma, *modifiers, comma]

    def _reduced(self, noun_phrase: NounPhrase) -> VerbalizationFragment:
        """:return: A repeat mention reduced to its head — the first-mention modifiers dropped — as a
        bare label (*"Robot 1"*) when numbered, else a definite reference (*"the Robot"*).

        Verbalizing one expression twice against a shared context reduces the repeat to *"the Robot"*:

        >>> from krrood.entity_query_language.verbalization.context import MicroplanningServices
        >>> from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
        >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
        >>> robot = variable(Robot, [])
        >>> query, shared = a(entity(robot)), MicroplanningServices.from_expression(robot)
        >>> _ = EQLVerbalizer().build(query, shared)
        >>> flatten_fragment_to_plain_text(EQLVerbalizer().build(query, shared))
        'Find the Robot'
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

        A plural subject records a plural number, so its possessive reads *"their"*:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).ordered_by(employee.salary)))
        'Report Employees ordered by their salaries from lowest to highest'
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
        slot — so rules never mark the scope.

        Its contribution is pushing the Employee as the subject *of its own modifiers*, so the chains
        inside the WHERE pronominalise to *its salary* / *its starting_salary* rather than re-naming
        the Employee:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).where(employee.salary > employee.starting_salary)))
        'Find an Employee such that its salary is greater than its starting_salary'
        """
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
