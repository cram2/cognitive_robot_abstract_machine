from __future__ import annotations

import operator
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing_extensions import Dict, List, Optional, Set, Tuple, TYPE_CHECKING

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import (
    InstantiatedVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.value_lexicon import type_noun
from krrood.entity_query_language.verbalization.relational_attributes import (
    relational_verb,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Keywords
from krrood.entity_query_language.query.aggregation_structure import (
    aggregation_source_root,
    selected_aggregator,
)
from krrood.patterns.field_metadata import GrammarMetadata

if TYPE_CHECKING:
    from krrood.entity_query_language.verbalization.grammar.conditions.placement import (
        RestrictionFragments,
    )


def referring_noun_with_restrictions(
    variable: Variable,
    selected_type: str,
    definiteness: Definiteness,
    restriction: Optional[RestrictionFragments],
) -> NounPhrase:
    """:return: a referring noun phrase for *variable* carrying *restriction* as post-nominal
    modifiers — the shared assembly behind a nested entity noun (*"a Mission with priority greater
    than 2"* / *"a Worker whose tasks contains a Task"*).

    The pieces attach in reading order: inline modifiers and relative clauses hug the noun, the
    coordinated *"whose"* block and a *"where"*-headed residual follow. A referring noun phrase is the
    subject of its own modifiers, so the coreference pass pronominalises later mentions correctly.
    """
    modifiers: List[VerbalizationFragment] = []
    if restriction is not None:
        modifiers.extend(restriction.inline_modifiers)
        modifiers.extend(restriction.relative_clauses)
        if restriction.whose is not None:
            modifiers.append(restriction.whose)
        if restriction.residual is not None:
            modifiers.append(
                PhraseFragment(
                    parts=[Keywords.WHERE.as_fragment(), restriction.residual]
                )
            )
    return NounPhrase(
        head=RoleFragment.for_variable(selected_type, variable),
        definiteness=definiteness,
        referent_id=variable._id_ if isinstance(variable, Variable) else None,
        modifiers=modifiers,
    )


# %% Operand head-noun resolution


@dataclass(frozen=True)
class ParentEdge:
    """
    One structural edge from a child node up to a parent it is reachable through in an expression
    graph.
    """

    parent: SymbolicExpression
    """The parent node this edge descends from."""

    field_name: Optional[str] = None
    """The declared predicate field the child fills on *parent*, when *parent* is an
    :class:`InstantiatedVariable` and this edge is one of its declared fields; ``None`` for every
    other structural edge."""


def _child_edges_by_id(
    expression: SymbolicExpression,
) -> Dict[uuid.UUID, List[ParentEdge]]:
    """:return: every node's parent edges in *expression*, keyed by the child's ``_id_``. A
    :class:`ParentEdge`'s field name is set only when the parent is an :class:`InstantiatedVariable`
    and the edge is one of its declared fields (so the same variable filling two fields on one
    predicate is two distinct edges, via :attr:`InstantiatedVariable._child_vars_`; every other
    structural edge carries no field name).

    Used to tell an operand that fills exactly one predicate field, and appears nowhere else, from
    a variable also referenced elsewhere (a query subject, an ``==``-constrained pair) — the
    condition :func:`operand_head_noun` uses to decide whether a field-derived noun applies.

    Deduplicates by parent id first: :attr:`SymbolicExpression._all_expressions_` revisits a
    shared node once per reachable path, so a parent reached twice would otherwise double-count
    its own child edges.
    """
    edges: Dict[uuid.UUID, List[ParentEdge]] = defaultdict(list)
    visited_parents: Set[uuid.UUID] = set()
    for node in expression._all_expressions_:
        if node._id_ in visited_parents:
            continue
        visited_parents.add(node._id_)
        if isinstance(node, InstantiatedVariable):
            for field_name, child in node._child_vars_.items():
                edges[child._id_].append(ParentEdge(node, field_name))
        else:
            for child in node._children_:
                edges[child._id_].append(ParentEdge(node))
    return edges


def _sole_predicate_field(edges: List[ParentEdge]) -> Optional[Tuple[type, str]]:
    """:return: ``(predicate_class, field_name)`` when *edges* is exactly one edge and that edge is
    a named predicate field, else ``None``. This is the structural (not heuristic) condition for
    treating a variable as an anonymous operand of that one field: it must be reachable through
    exactly that field and nowhere else in the expression.
    """
    if len(edges) != 1:
        return None
    edge = edges[0]
    if edge.field_name is None or not isinstance(edge.parent, InstantiatedVariable):
        return None
    return edge.parent._type_, edge.field_name


def operand_head_noun(node: Variable, edges: List[ParentEdge]) -> str:
    """:return: the head noun naming *node*, resolved in order of decreasing specificity:

    1. *node*'s own type noun, when its type is informative (a concrete class other than the bare
       ``object`` placeholder) — the type is the default identifier for a referring expression
       (Dale & Reiter's Incremental Algorithm includes the type attribute unconditionally) and
       *always* wins once known, so a genuinely typed operand (``HasType(a_body, Apple)`` →
       *"a Body"*) is never overridden by a field's metadata;
    2. only once the type carries no information: the owning predicate field's declared
       :attr:`~krrood.patterns.field_metadata.GrammarMetadata.display_name` — explicit lexical
       metadata, checked only when *node* fills exactly that one field and appears nowhere else
       (:func:`_sole_predicate_field`);
    3. the sole owning field's name itself (verbatim, underscores read as spaces), when no
       metadata is declared either;
    4. ``"object"`` as the last resort (no sole field at all).

    A variable referenced anywhere besides that one field (a query subject, an
    ``==``-constrained pair) never reaches steps 2 or 3 — it always keeps its type-named identity,
    because a tracked entity is identified by its category, not by the role it happens to fill
    here.

    :param node: The referent variable.
    :param edges: *node*'s parent edges, from :func:`_child_edges_by_id`.

    >>> operand_head_noun(variable(Robot, []), [])
    'Robot'

    A sole predicate field only ever supplies a noun once the type gives none — a typed operand
    keeps its type noun even filling a field of ``IsReachable`` (whose ``location`` field is
    declared ``object``, an uninformative type on its own):

    >>> reachable = IsReachable(variable(Robot, []))
    >>> operand_head_noun(variable(Robot, []), [ParentEdge(reachable, "location")])
    'Robot'

    An untyped (``object``) operand falls through: to the field's declared
    :attr:`~krrood.patterns.field_metadata.GrammarMetadata.display_name` when one is set, else to
    the field name itself, else to ``"object"`` once there is no sole field to name it by:

    >>> operand_head_noun(variable(object, []), [ParentEdge(reachable, "location")])
    'location'
    >>> operand_head_noun(variable(object, []), [])
    'object'

    A variable reachable through more than one edge is not a sole predicate operand, so it never
    reads a field name — only its type, or ``"object"`` as the last resort:

    >>> operand_head_noun(
    ...     variable(object, []), [ParentEdge(reachable, "location"), ParentEdge(reachable, "location")]
    ... )
    'object'
    """
    type_ = node._type_
    if isinstance(type_, type) and type_ is not object:
        return type_noun(type_)
    sole_field = _sole_predicate_field(edges)
    if sole_field is None:
        return "object"
    predicate_class, field_name = sole_field
    metadata = GrammarMetadata.of_field(predicate_class, field_name)
    if metadata is not None and metadata.display_name is not None:
        return metadata.display_name
    return field_name.replace("_", " ")


# %% Same-noun disambiguation


@dataclass(frozen=True)
class Distinguisher:
    """
    The determiner-level feature distinguishing one member of a same-noun group of ≥ 2 distinct
    referents from the others — mirrors :attr:`~…fragments.base.NounPhrase.alternative` /
    :attr:`~…fragments.base.NounPhrase.ordinal`.
    """

    alternative: bool = False
    """``True`` for the second member of a same-noun *pair* — realised as *"another"* / *"the
    other"*."""

    ordinal: Optional[int] = None
    """A member's position within a same-noun group of three or more, counting the first member as
    position 1 — so a value of 2 realises as an ordinal word (*"a second …"*), 3 as *"a third
    …"*, and so on."""


@dataclass
class DistinguisherIndex:
    """
    Assigns each same-noun group's members a distinguishing feature the first time the coreference
    pass encounters them, in discourse (document) order.

    The pre-scan (:meth:`ReferringExpressions._group_referents_by_noun`) only knows which
    canonical referents share a noun and how many there are — it walks the expression graph, not
    the rendered text, so it cannot know which referent will be *said* first (folding,
    coordination, and shared-subject reduction all rewrite the mention order). Deciding "who is
    plain and who is another" is therefore deferred to the coreference pass, which walks the
    realised tree in the order it will actually be read.

    References:

    * :cite:t:`reiter2000building` — referring-expression generation as a discourse-order
      decision, not an input-structure-order one.
    * :cite:t:`gundel1993givenness` — the given/new distinction behind the indefinite
      *"another"* (a fresh alternative) vs. the definite *"the other"* (an already-identified
      one) that :attr:`Distinguisher.alternative` realises.
    """

    canonical_of: Dict[uuid.UUID, uuid.UUID] = field(default_factory=dict)
    """Every referent id's canonical id — an ``==``-unified group collapses to one (see
    :func:`referent_aliases`); a referent in no identity is its own canonical."""

    noun_of_canonical: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Each canonical's resolved head noun."""

    group_size: Dict[str, int] = field(default_factory=dict)
    """How many distinct canonicals share each noun."""

    _assigned_position: Dict[uuid.UUID, int] = field(default_factory=dict, repr=False)
    """Each canonical's position within its noun group, counting the first member as position 0,
    filled lazily on first encounter."""

    _next_position: Dict[str, int] = field(default_factory=dict, repr=False)
    """The next unassigned position for each noun group."""

    def distinguisher_for(self, referent_id: uuid.UUID) -> Optional[Distinguisher]:
        """:return: the distinguishing feature for *referent_id*, or ``None`` when it is alone in
        its noun group (or shares no group at all). The first call for a given canonical assigns
        its position, in call order; every later call for the same canonical (a repeat mention)
        returns the same feature, so re-mentioning a referent keeps the distinguisher its first
        mention was assigned.

        :param referent_id: A referent's own id (before canonicalisation).
        """
        canonical_id = self.canonical_of.get(referent_id, referent_id)
        noun = self.noun_of_canonical.get(canonical_id)
        if noun is None or self.group_size.get(noun, 1) < 2:
            return None
        position = self._assigned_position.get(canonical_id)
        if position is None:
            position = self._next_position.get(noun, 0)
            self._assigned_position[canonical_id] = position
            self._next_position[noun] = position + 1
        if position == 0:
            return None
        if self.group_size[noun] == 2:
            return Distinguisher(alternative=True)
        return Distinguisher(ordinal=position + 1)


@dataclass
class _HeadNounGrouping:
    """
    Numberable referents grouped under their canonical entity (``==``-unified referents collapse
    to one) and then by resolved head noun, recording each noun's canonicals in first-encounter
    (pre-scan) order — feeding both the first-mention noun text
    (:meth:`ReferringExpressions.head_noun_of`) and the group structure the coreference pass later
    disambiguates by determiner (:class:`DistinguisherIndex`).
    """

    canonicals_by_noun: Dict[str, List[uuid.UUID]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """Each noun's canonical entities, in first-encounter order — a noun with two or more is
    disambiguated."""

    members_by_canonical: Dict[uuid.UUID, List[uuid.UUID]] = field(
        default_factory=lambda: defaultdict(list)
    )
    """Every original referent id that maps to a given canonical entity."""

    _noun_of_canonical: Dict[uuid.UUID, str] = field(default_factory=dict)
    """The noun a canonical was first registered under (guards against re-listing it)."""

    def add(self, referent_id: uuid.UUID, canonical: uuid.UUID, noun: str) -> None:
        """Record *referent_id* as a member of *canonical*, registering *canonical* under *noun*
        on first sight — an ``==``-unified canonical whose members would otherwise resolve
        different nouns keeps the first-encountered one, a single deterministic rule."""
        if canonical not in self._noun_of_canonical:
            self._noun_of_canonical[canonical] = noun
            self.canonicals_by_noun[noun].append(canonical)
        self.members_by_canonical[canonical].append(referent_id)

    def head_nouns(self) -> Dict[uuid.UUID, str]:
        """:return: every member referent id mapped to its canonical's resolved head noun."""
        return {
            member: noun
            for noun, canonicals in self.canonicals_by_noun.items()
            for canonical in canonicals
            for member in self.members_by_canonical[canonical]
        }

    def distinguisher_index(self) -> DistinguisherIndex:
        """:return: the :class:`DistinguisherIndex` for these groups, with no positions assigned
        yet — the coreference pass assigns those lazily, in discourse order."""
        canonical_of = {
            member: canonical
            for canonical, members in self.members_by_canonical.items()
            for member in members
        }
        return DistinguisherIndex(
            canonical_of=canonical_of,
            noun_of_canonical=dict(self._noun_of_canonical),
            group_size={
                noun: len(canonicals)
                for noun, canonicals in self.canonicals_by_noun.items()
            },
        )


@dataclass(frozen=True)
class NounForm:
    """
    The first-mention determiner form for a variable noun: its definiteness and label.
    """

    definiteness: Definiteness
    """
    Always ``INDEFINITE`` (*"a Robot"*) — the disambiguating determiner (if any) is decided
    later, by the coreference pass, once discourse order is known.
    """

    label: str
    """
    The display label for the noun head.
    """


@dataclass
class ReferringExpressions:
    """
    Pre-computed referring-expression state for a verbalization pass: each referent's resolved
    head noun and the set of referents already introduced.

    This is the referring-expression generation subtask of microplanning: deciding a referent's
    head noun (:func:`operand_head_noun`), between an indefinite first mention (*"a Robot"*), a
    definite subsequent mention (*"the Robot"*), a determiner-distinguished form when its noun is
    shared with another referent (*"another Robot"* / *"a second Robot"*), and a pronoun (*"its
    …"*) when a chain is rooted at the current discourse subject.

    References:

    * :cite:t:`reiter2000building` — referring-expression generation as a microplanning subtask.
    * :cite:t:`dale1995gricean` — interpreting the Gricean maxims in referring-expression generation.
    """

    seen: Set[uuid.UUID] = field(default_factory=set)
    """
    Referent ``_id_`` s introduced so far, used to seed coreference across builds
    sharing this context, so verbalizing the same expression twice reads *"a Robot"*
    then *"the Robot"*.
    """

    head_nouns: Dict[uuid.UUID, str] = field(default_factory=dict)
    """
    Maps referent ``_id_`` → resolved head noun, pre-computed before verbalization begins
    (:func:`operand_head_noun`).
    """

    distinguishers: DistinguisherIndex = field(default_factory=DistinguisherIndex)
    """
    Same-noun-group disambiguation, assigned lazily in discourse order by the coreference pass.
    """

    @classmethod
    def from_expression(cls, expression: SymbolicExpression) -> ReferringExpressions:
        """
        :param expression: Root EQL expression or query to scan.
        :return: An instance with the head-noun map pre-built for *expression*.

        >>> first, second = variable(Robot, []), variable(Robot, [])
        >>> referring = ReferringExpressions.from_expression(an(set_of(first, second)))
        >>> referring.head_nouns[first._id_]
        'Robot'
        """
        grouping = cls._group_referents_by_noun(expression)
        return cls(
            head_nouns=grouping.head_nouns(),
            distinguishers=grouping.distinguisher_index(),
        )

    @classmethod
    def _group_referents_by_noun(
        cls, expression: SymbolicExpression
    ) -> _HeadNounGrouping:
        """:return: Every numberable referent grouped under its canonical entity, then its
        resolved head noun, in encounter order. Literal nodes, already-seen ids, and aggregation
        sources are skipped; an ``==``-unified pair shares one canonical (so it is named once).
        """
        if isinstance(expression, Query):
            expression.build()
        suppressed = cls._aggregation_source_ids(expression)
        aliases = referent_aliases(expression)
        edges_by_id = _child_edges_by_id(expression)
        grouping = _HeadNounGrouping()
        seen: Set[uuid.UUID] = set()
        for node in expression._all_expressions_:
            noun = cls._resolve_head_noun(node, edges_by_id)
            if noun is None or node._id_ in suppressed or node._id_ in seen:
                continue
            seen.add(node._id_)
            grouping.add(node._id_, aliases.get(node._id_, node._id_), noun)
        return grouping

    @staticmethod
    def _aggregation_source_ids(expression: SymbolicExpression) -> Set[uuid.UUID]:
        """
        Such a variable denotes a population to aggregate over, not a specific entity, so it must not
        consume a disambiguating distinguisher — otherwise the outer subject would pick up a
        spurious *"another"* with no matching plain mention, and a constrained aggregation scope
        would read *"among another BankTransaction"* rather than *"among BankTransactions"*.

        :param expression: Root expression to scan.
        :return: The ``_id_`` of every variable that serves as the source population of an
            aggregation sub-query (e.g. the ``BankTransaction`` behind ``max(t.amount_details.amount)``),
            to exclude from disambiguation.

        >>> transaction = variable(BankTransaction, [])
        >>> source = an(entity(max(transaction.amount_details.amount)))
        >>> transaction._id_ in ReferringExpressions._aggregation_source_ids(source)
        True
        """
        ids: Set[uuid.UUID] = set()
        for node in expression._all_expressions_:
            if isinstance(node, Entity) and selected_aggregator(node) is not None:
                root = aggregation_source_root(node)
                if root is not None:
                    ids.add(root._id_)
        return ids

    @staticmethod
    def _is_relational_referent(node: SymbolicExpression) -> bool:
        """:return: whether *node* is a relational attribute hop (a verb-named hop such as
        ``assigned_to``, whose relative clause names a distinct entity that must be told apart
        from other same-noun entities).

        :param node: The expression node to test.
        """
        return (
            isinstance(node, Attribute)
            and relational_verb(node._attribute_name_) is not None
        )

    @staticmethod
    def _is_numberable(node: SymbolicExpression) -> bool:
        """:return: whether *node* denotes a numberable entity — a (non-literal) variable, or a
        relational attribute hop. Independent of *which* noun it resolves to
        (:meth:`_resolve_head_noun`) — used where only entity identity matters.

        :param node: The expression node to test.
        """
        if isinstance(node, Variable) and not isinstance(node, Literal):
            return True
        return ReferringExpressions._is_relational_referent(node)

    @staticmethod
    def _resolve_head_noun(
        node: SymbolicExpression,
        edges_by_id: Dict[uuid.UUID, List[Tuple[SymbolicExpression, Optional[str]]]],
    ) -> Optional[str]:
        """:return: The head noun a node is disambiguated (and later referred to) under — the
        operand-aware resolution (:func:`operand_head_noun`) for a (non-literal) variable, or the
        *value* type noun for a relational attribute. ``None`` for anything that does not denote a
        numberable entity.

        :param node: The expression node to resolve a head noun for.
        :param edges_by_id: Every node's parent edges in the same expression, from
            :func:`_child_edges_by_id` — looked up for *node* and passed on to
            :func:`operand_head_noun`.

        >>> ReferringExpressions._resolve_head_noun(variable(Robot, []), {})
        'Robot'
        """
        if isinstance(node, Variable) and not isinstance(node, Literal):
            return operand_head_noun(node, edges_by_id.get(node._id_, []))
        if ReferringExpressions._is_relational_referent(node):
            value_type = node._type_
            return type_noun(value_type) if isinstance(value_type, type) else None
        return None

    @staticmethod
    def _variable_type_label(variable: Variable) -> str:
        """:return: The display noun for *variable*'s type — the friendly type noun when it carries a
        ``_type_`` (*"Integer"* for ``int``), else its own class name."""
        return (
            type_noun(variable._type_)
            if variable._type_
            else variable.__class__.__name__
        )

    def head_noun_of(self, variable: Variable) -> str:
        """:return: *variable*'s resolved head noun — pre-computed in the pre-scan, or its type
        label as a fallback for a variable outside the pre-scanned expression."""
        return self.head_nouns.get(variable._id_, self._variable_type_label(variable))

    def noun_for_parts(self, variable: Variable) -> NounForm:
        """
        :param variable: A variable instance.
        :return: The first-mention :class:`NounForm` for *variable* — always ``INDEFINITE``; a
            determiner-distinguished form (if this noun turns out to be shared with another
            referent) is decided later, by the coreference pass.

        >>> ReferringExpressions().noun_for_parts(variable(Robot, [])).label
        'Robot'
        """
        self.seen.add(variable._id_)
        return NounForm(Definiteness.INDEFINITE, self.head_noun_of(variable))


def _entity_referent_id(node: SymbolicExpression) -> Optional[uuid.UUID]:
    """:return: the referent id of an entity-denoting node — a free variable, or a relational hop
    (the related entity it introduces) — else ``None``. The same notion of a numberable referent
    :meth:`ReferringExpressions._is_numberable` uses, so identity and disambiguation agree on what
    counts as an entity."""
    if not ReferringExpressions._is_numberable(node):
        return None
    return node._id_


def referent_aliases(expression: SymbolicExpression) -> Dict[uuid.UUID, uuid.UUID]:
    """
    :param expression: Root expression to scan.
    :return: A map from each entity referent id that participates in an identity to its canonical
        id. An ``==`` constraint between two entity referents (``m.assigned_to == r``) makes them one
        entity, so disambiguation and coreference can treat the pair as a single referent (*"a
        Robot"*, not *"a Robot"* / *"another Robot"*). Referents in no identity do not appear (each
        is its own canonical).

    >>> robot, mission = variable(Robot, []), variable(Mission, [])
    >>> aliases = referent_aliases(and_(mission.assigned_to == robot, mission.priority > 2))
    >>> len(set(aliases.values()))  # the relation and the variable collapse to one entity
    1
    """
    parent: Dict[uuid.UUID, uuid.UUID] = {}

    def find(node_id: uuid.UUID) -> uuid.UUID:
        parent.setdefault(node_id, node_id)
        root = node_id
        while parent[root] != root:
            root = parent[root]
        while parent[node_id] != root:
            parent[node_id], node_id = root, parent[node_id]
        return root

    def union(first: uuid.UUID, second: uuid.UUID) -> None:
        parent[find(second)] = find(first)

    for node in expression._all_expressions_:
        if isinstance(node, Comparator) and node.operation is operator.eq:
            left = _entity_referent_id(node.left)
            right = _entity_referent_id(node.right)
            if left is not None and right is not None:
                union(left, right)
    return {node_id: find(node_id) for node_id in parent}
