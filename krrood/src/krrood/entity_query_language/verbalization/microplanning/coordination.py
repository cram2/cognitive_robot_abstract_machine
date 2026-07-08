"""
Coordination — the conjunction-reduction (aggregation) microplanning task.

Two stages live here, top to bottom:

* **EQL-level conjunct reduction** — the :class:`ConjunctReducer` pass folds a flat conjunct list
  (e.g. an ``AND``'s operands) into one where recognizable groups become first-class fold *artifacts*
  (:class:`RangeFold`, :class:`CoindexedFold`) that the grammar then renders. Each fold is a
  :class:`ConjunctFold` strategy in the reducer's ordered registry — adding a fold is a new strategy,
  nothing else changes (open/closed). This is the *one* place a caller goes to simplify conjuncts.
* **VerbalizationFragment-level coordination builders** — :func:`build_between` and :func:`oxford_comma`
  (re-exported from the fragment layer) assemble already-rendered pieces into a coordinated phrase.

References: :cite:t:`reiter2000building` and :cite:t:`dalianis1999aggregation` — aggregation realised via coordination /
conjunction reduction.
"""

from __future__ import annotations

import operator
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing_extensions import (
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    TypeAlias,
    TypeVar,
    Union,
)

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_comma,
    PhraseFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    RangePhrases,
    SetMembership,
    copula_with,
)

#: Hashable identity of a pure attribute chain: ``(root variable id, ((name, owner), …))``.
ChainKey: TypeAlias = Tuple

#: The comparison operators a co-indexed group folds over. Equality additionally licenses the
#: natural *"… have the same …"* surface (see :func:`coindexed_natural_parts`); the others read in
#: the faithful *"… are <op> those of …"* form. ``ne``/``contains``/temporal never fold.
COINDEXED_OPERATORS: Tuple[Callable, ...] = (
    operator.eq,
    operator.gt,
    operator.lt,
    operator.ge,
    operator.le,
)


# %% fold artifacts (the vocabulary the pass produces)


@dataclass
class RangeFold:
    """A folded lower/upper bound pair on one attribute chain."""

    chain_expression: SymbolicExpression
    """The shared attribute chain (e.g. ``t.booking_date``)."""

    lower_expression: SymbolicExpression
    """The lower-bound value expression (the ``>=`` / ``>`` right operand)."""

    upper_expression: SymbolicExpression
    """The upper-bound value expression (the ``<=`` / ``<`` right operand)."""


@dataclass
class CoindexedFold:
    """A group of comparators that compare the *same* leaf attributes across two shared prefixes —
    e.g. ``p.begin.month == p.end.month`` and ``p.begin.year == p.end.year`` — reduced to one node.

    This is conjunction reduction over co-indexed comparisons: the two prefixes (``p.begin`` /
    ``p.end``) and the operator are shared, and only the terminal attribute varies, so the shared
    structure is said once (*"the begin and end of its period have the same month and year"*).
    """

    operation: Callable
    """The shared comparison operator (e.g. ``operator.eq``)."""

    terminals: List[Tuple[str, type]]
    """The *distinct* co-indexed terminal attributes ``(name, owner)``, in first-occurrence order —
    the *"month and year"* the comparators range over. A leaf compared more than once is one
    attribute, so it is listed once (*"have the same name"*, never *"… name and name"*)."""

    left_prefix_expression: SymbolicExpression
    """An exemplar of the left chain minus its terminal hop (e.g. ``p.begin``)."""

    right_prefix_expression: SymbolicExpression
    """An exemplar of the right chain minus its terminal hop (e.g. ``p.end``)."""


@dataclass
class SharedSubjectComparisons:
    """Two or more value comparators on the *same* subject chain, factored so the subject and its
    copula are said once and only the predicate tails are coordinated — *"the battery of a Robot is
    greater than 50 or less than 10"* instead of repeating *"the battery of … is …"* per
    disjunct.

    This is coordination reduction over a shared subject: the disjunctive analogue of
    :class:`RangeFold` (which is itself the special case of two complementary bounds folded to
    *"between"*), generalised to any comparison operators.
    """

    subject_expression: SymbolicExpression
    """The shared left-hand attribute chain (e.g. ``robot.battery``), rendered once via the normal
    recursion."""

    comparators: List[Comparator]
    """The comparators in source order; each contributes its operator-and-value tail (*"greater than
    50"*), coordinated under the shared subject."""


@dataclass
class SharedSubjectConjunction:
    """The conjunctive analogue of :class:`SharedSubjectComparisons`: two or more value comparisons
    on the *same bare variable*, said once as one shared-subject main clause — *"an Integer is
    between 1 and 10 and not 5"* — rather than repeating the subject per conjunct.

    Scoped to a *bare variable* subject (not an attribute chain), so the shared subject and copula
    lead a single clause; an attribute-chain conjunction (*"the battery of a Robot"*) names a
    different subject per conjunct and so keeps its per-clause surface.
    """

    subject_expression: SymbolicExpression
    """The shared bare variable, rendered once via the normal recursion (*"an Integer"*)."""

    tails: List[Union[Comparator, RangeFold]]
    """The predicate tails in source order — a value :class:`Comparator` (its operator-and-value
    tail, *"not 5"*) or a folded :class:`RangeFold` (a *"between low and high"* tail), so a
    complementary bound pair reads *"between 1 and 10"* within the clause."""


@dataclass(frozen=True)
class CoindexedNaturalParts:
    """The pieces of the natural *"the <a> and <b> of <shared> have the same …"* rendering."""

    shared_prefix_expression: SymbolicExpression
    """The chain the two prefixes share (e.g. ``p.period``), rendered via the normal recursion."""

    left_hop: Tuple[str, type]
    """The left prefix's distinguishing final hop ``(name, owner)`` (e.g. ``begin``)."""

    right_hop: Tuple[str, type]
    """The right prefix's distinguishing final hop ``(name, owner)`` (e.g. ``end``)."""


# %% the conjunct-reduction pass (high-level entry)

FoldNode: TypeAlias = Union[SymbolicExpression, RangeFold, CoindexedFold]
"""A node the verbalization fold dispatches over: either a real EQL expression or a synthetic
coordination artifact (:class:`RangeFold` / :class:`CoindexedFold`) produced by conjunct reduction."""

ConjunctList: TypeAlias = List[FoldNode]


class ConjunctFold(ABC):
    """One coordination fold: recognise a foldable group within a list of sibling conjuncts and
    collapse it to a single fold artifact, leaving everything else (and its order) intact.

    A fold is order-preserving and idempotent on already-folded items, so the reducer can apply its
    folds in sequence without interference.
    """

    @abstractmethod
    def apply(self, conjuncts: ConjunctList) -> ConjunctList:
        """:param conjuncts: The (possibly already partly-folded) conjunct list.
        :return: The list with every group this fold recognises collapsed to its artifact.
        """


class RangeBoundFold(ConjunctFold):
    """Fold a complementary lower/upper bound pair on one chain into a :class:`RangeFold`
    (*"x is between low and high"*)."""

    def apply(self, conjuncts: ConjunctList) -> ConjunctList:
        """:return: *conjuncts* with each complementary bound pair folded to a :class:`RangeFold`.

        >>> robot = variable(Robot, [])
        >>> isinstance(RangeBoundFold().apply([robot.battery >= 20, robot.battery <= 80])[0], RangeFold)
        True
        """
        return fold_range_pairs(conjuncts)


class CoindexedComparisonFold(ConjunctFold):
    """Fold a group of co-indexed comparisons across two shared prefixes into a
    :class:`CoindexedFold` (*"the begin and end … have the same month and year"*)."""

    def apply(self, conjuncts: ConjunctList) -> ConjunctList:
        """:return: *conjuncts* with each co-indexed comparison group folded to a
        :class:`CoindexedFold`.

        >>> import dataclasses
        >>> Span = dataclasses.make_dataclass("Span", [("low", int), ("high", int)])
        >>> Pair = dataclasses.make_dataclass("Pair", [("left", Span), ("right", Span)])
        >>> pair = variable(Pair, [])
        >>> folded = CoindexedComparisonFold().apply([pair.left.low == pair.right.low, pair.left.high == pair.right.high])
        >>> isinstance(folded[0], CoindexedFold)
        True
        """
        return fold_coindexed_groups(conjuncts)


@dataclass
class ConjunctReducer:
    """The single home of WHERE-conjunct simplification: applies its ordered :class:`ConjunctFold`
    registry to a flat conjunct list. The default registry folds range pairs, then co-indexed
    groups; the two operate on disjoint patterns, so the order only fixes a stable result.
    """

    folds: List[ConjunctFold] = field(
        default_factory=lambda: [RangeBoundFold(), CoindexedComparisonFold()]
    )
    """The folds applied, in order — the open/closed registry (append a strategy to extend)."""

    def reduce(self, conjuncts: List[SymbolicExpression]) -> ConjunctList:
        """:param conjuncts: A flat conjunct list (e.g. an ``AND``'s operands).
        :return: The list with each fold's recognised groups reduced to artifacts.

        >>> robot = variable(Robot, [])
        >>> len(ConjunctReducer().reduce([robot.battery >= 20, robot.battery <= 80]))
        1
        """
        items: ConjunctList = list(conjuncts)
        for fold in self.folds:
            items = fold.apply(items)
        return items


def reduce_conjuncts(conjuncts: List[SymbolicExpression]) -> ConjunctList:
    """
    Reduce a flat conjunct list by every coordination fold — the single entry every
    conjunct-rendering caller uses, so none has to know a fold exists.

    :param conjuncts: A flat list of conjuncts (e.g. the operands of an ``AND``).
    :return: The reduced list (raw expressions interleaved with range / co-indexed folds).

    >>> robot = variable(Robot, [])
    >>> [type(item).__name__ for item in reduce_conjuncts([robot.battery >= 20, robot.battery <= 80])]
    ['RangeFold']
    """
    return ConjunctReducer().reduce(conjuncts)


# %% range-bound fold


class _RangeBound(Enum):
    """Internal marker for the direction of a bound comparison in range folding."""

    LOWER = auto()
    UPPER = auto()


def chain_key(expression: SymbolicExpression) -> Optional[ChainKey]:
    """:return: The hashable identity of a pure attribute chain — ``(root_id, ((name, owner),
    …))`` — or ``None``.

    >>> robot = variable(Robot, [])
    >>> chain_key(robot.battery) == chain_key(robot.battery)
    True
    >>> chain_key(robot) is None
    True
    """
    if not isinstance(expression, MappedVariable):
        return None
    chain, root = walk_chain(expression)
    if not isinstance(root, Variable):
        return None
    parts = []
    for node in chain:
        if not isinstance(node, Attribute):
            return None  # only pure attribute chains fold cleanly
        parts.append((node._attribute_name_, node._owner_class_))
    return (root._id_, tuple(parts))


def subject_key(expression: SymbolicExpression) -> Optional[ChainKey]:
    """:return: The hashable identity of a comparison's subject — :func:`chain_key` for a pure
    attribute chain, or ``(variable_id, ())`` for a bare variable — so a fold can group bounds on a
    bare variable (*"an Integer"*) as readily as on a chain. ``None`` for anything else.

    >>> x = variable(int, [])
    >>> subject_key(x) == subject_key(x)
    True
    >>> subject_key(variable(Robot, []).battery) is not None
    True
    """
    key = chain_key(expression)
    if key is not None:
        return key
    if isinstance(expression, Variable) and not isinstance(expression, Literal):
        return (expression._id_, ())
    return None


def _classify(conjunct: SymbolicExpression) -> Optional[Tuple[ChainKey, _RangeBound]]:
    """
    :param conjunct: A candidate conjunct.
    :return: ``(chain_key, _RangeBound)`` when *conjunct* is a bound comparison, else ``None``.

    >>> robot = variable(Robot, [])
    >>> _classify(robot.battery >= 20)[1] is _RangeBound.LOWER
    True
    >>> _classify(robot) is None
    True
    """
    if not isinstance(conjunct, Comparator):
        return None
    key = subject_key(conjunct.left)
    if key is None:
        return None
    if conjunct.operation in (operator.gt, operator.ge):
        return key, _RangeBound.LOWER
    if conjunct.operation in (operator.lt, operator.le):
        return key, _RangeBound.UPPER
    return None


def fold_range_pairs(
    conjuncts: List[SymbolicExpression],
) -> List[Union[SymbolicExpression, RangeFold]]:
    """
    Fold complementary lower/upper bound comparisons on the same chain into range items,
    preserving the order of everything else.

    Direction (not position) decides which operand is the lower vs upper bound, so ``t.x <= high``
    written before ``t.x >= low`` still yields ``between low and high``. Folding is greedy: the first
    unfolded bound in one direction pairs with the first arriving opposite-direction bound on the
    same chain.

    :param conjuncts: A flat list of conjuncts (e.g. the operands of an ``AND``).
    :return: A list whose items are either the original expressions or range folds.

    The fold surfaces as a *"between … and …"* rendering:

    >>> transaction = variable(BankTransaction, [])
    >>> verbalize_expression(an(entity(transaction).where(
    ...     transaction.booking_date >= datetime.datetime(2026, 5, 15),
    ...     transaction.booking_date <= datetime.datetime(2026, 5, 30))))
    'Find a BankTransaction whose booking_date is between May 15, 2026 and May 30, 2026'
    """
    classifications = [_classify(conjunct) for conjunct in conjuncts]
    slots: List[Union[SymbolicExpression, RangeFold]] = list(conjuncts)
    dropped = [False] * len(conjuncts)
    # chain_key -> indices of bounds awaiting a complement (always one direction at a time).
    awaiting: Dict[ChainKey, List[int]] = {}
    for i, classification in enumerate(classifications):
        if classification is None:
            continue
        key, bound = classification
        queue = awaiting.setdefault(key, [])
        # A waiting bound of the opposite direction → fold the pair; else enqueue and wait.
        if queue and classifications[queue[0]][1] is not bound:
            j = queue.pop(0)
            lower, upper = (
                (conjuncts[j], conjuncts[i])
                if bound is _RangeBound.UPPER
                else (conjuncts[i], conjuncts[j])
            )
            slots[min(i, j)] = RangeFold(
                chain_expression=lower.left,
                lower_expression=lower.right,
                upper_expression=upper.right,
            )
            dropped[max(i, j)] = True
        else:
            queue.append(i)
    return [slot for index, slot in enumerate(slots) if not dropped[index]]


def has_pair(conjuncts: List[SymbolicExpression]) -> bool:
    """
    :param conjuncts: A flat list of conjuncts.
    :return: ``True`` when range folding would produce at least one range fold.

    >>> robot = variable(Robot, [])
    >>> has_pair([robot.battery >= 20, robot.battery <= 80])
    True
    >>> has_pair([robot.battery >= 20])
    False
    """
    return any(isinstance(item, RangeFold) for item in fold_range_pairs(conjuncts))


# %% co-indexed fold


def _attribute_pair(node: SymbolicExpression) -> Optional[Tuple[str, type]]:
    """:return: ``(name, owner)`` when *node* is an ``Attribute`` hop, else ``None``.

    >>> terminal, _ = walk_chain(variable(LoveBirds, []).bird_1.name)
    >>> _attribute_pair(terminal[-1]) == ("name", Bird)
    True
    """
    if isinstance(node, Attribute):
        return (node._attribute_name_, node._owner_class_)
    return None


def _terminal_attribute(expression: SymbolicExpression) -> Optional[Attribute]:
    """:return: The leaf ``Attribute`` of a ``MappedVariable`` chain (e.g. ``month`` of
    ``p.begin.month``), or ``None`` when the chain does not end in an attribute.

    >>> _terminal_attribute(variable(LoveBirds, []).bird_1.name)._attribute_name_
    'name'
    """
    chain, _ = walk_chain(expression)
    if chain and isinstance(chain[-1], Attribute):
        return chain[-1]
    return None


def coindexed_signature(
    conjunct: SymbolicExpression,
) -> Optional[Tuple[Tuple, Tuple[str, type]]]:
    """
    Recognise a co-indexed comparison — a comparator whose two sides compare the *same* leaf
    attribute across two attribute-chain prefixes (``p.begin.month == p.end.month``).

    The guard is strict: it only fires on a foldable operator, on two pure attribute chains with an
    identical terminal attribute on both sides, and on prefixes that are themselves pure attribute
    chains. Anything else returns ``None`` (→ no fold).

    :param conjunct: A candidate conjunct.
    :return: ``((operation, left_prefix_key, right_prefix_key), (terminal_name, terminal_owner))``
        — the grouping signature and the co-indexed leaf — or ``None``.

    >>> lovebirds = variable(LoveBirds, [])
    >>> coindexed_signature(lovebirds.bird_1.name == lovebirds.bird_2.name)[1] == ("name", Bird)
    True
    >>> coindexed_signature(lovebirds.bird_1.name != lovebirds.bird_2.name) is None
    True
    """
    if not isinstance(conjunct, Comparator):
        return None
    if conjunct.operation not in COINDEXED_OPERATORS:
        return None
    left_terminal = _terminal_attribute(conjunct.left)
    right_terminal = _terminal_attribute(conjunct.right)
    if left_terminal is None or right_terminal is None:
        return None
    leaf = (left_terminal._attribute_name_, left_terminal._owner_class_)
    if leaf != (right_terminal._attribute_name_, right_terminal._owner_class_):
        return None  # the compared leaves must be the same (co-indexed) attribute
    left_prefix_key = chain_key(conjunct.left._child_)
    right_prefix_key = chain_key(conjunct.right._child_)
    if left_prefix_key is None or right_prefix_key is None:
        return None
    return (conjunct.operation, left_prefix_key, right_prefix_key), leaf


def fold_coindexed_groups(
    items: List[Union[SymbolicExpression, RangeFold]],
) -> List[Union[SymbolicExpression, RangeFold, CoindexedFold]]:
    """
    Fold groups of co-indexed comparators (``p.begin.X == p.end.X`` for several ``X``) on the same
    prefixes and operator into one :class:`CoindexedFold`, preserving the order of everything else.

    Runs over the already range-folded list (range folds and prior co-indexed folds pass through
    untouched), so the two reductions compose without interfering. A group of fewer than two
    comparators is never folded, and a leaf compared more than once across the same prefixes is one
    co-indexed attribute (its terminal is listed once).

    :param items: A list of conjuncts, possibly already containing :class:`RangeFold` items.
    :return: The list with each co-indexed group reduced to a single fold at the group's first
        position.

    >>> import dataclasses
    >>> Span = dataclasses.make_dataclass("Span", [("low", int), ("high", int)])
    >>> Pair = dataclasses.make_dataclass("Pair", [("left", Span), ("right", Span)])
    >>> pair = variable(Pair, [])
    >>> [term[0] for term in fold_coindexed_groups(
    ...     [pair.left.low == pair.right.low, pair.left.high == pair.right.high])[0].terminals]
    ['low', 'high']
    >>> [term[0] for term in fold_coindexed_groups(  # a repeated leaf collapses to one terminal
    ...     [pair.left.low == pair.right.low, pair.left.low == pair.right.low])[0].terminals]
    ['low']
    """
    signatures = [
        (
            None
            if isinstance(item, (RangeFold, CoindexedFold))
            else coindexed_signature(item)
        )
        for item in items
    ]
    slots: List[Union[SymbolicExpression, RangeFold, CoindexedFold]] = list(items)
    dropped = [False] * len(items)
    groups: Dict[Tuple, List[int]] = {}
    for index, signature in enumerate(signatures):
        if signature is None:
            continue
        groups.setdefault(signature[0], []).append(index)
    for signature, indices in groups.items():
        if len(indices) < 2:
            continue  # a lone co-indexed comparison says itself; nothing to factor
        exemplar = items[indices[0]]
        slots[indices[0]] = CoindexedFold(
            operation=signature[0],
            # Distinct terminals only: a leaf compared more than once across the same prefixes is one
            # co-indexed attribute, so *"have the same name and name"* collapses to *"… the same name"*.
            terminals=list(dict.fromkeys(signatures[index][1] for index in indices)),
            left_prefix_expression=exemplar.left._child_,
            right_prefix_expression=exemplar.right._child_,
        )
        for index in indices[1:]:
            dropped[index] = True
    return [slot for index, slot in enumerate(slots) if not dropped[index]]


# %% owner grouping (shared aggregation primitive)

_Item = TypeVar("_Item")
_Payload = TypeVar("_Payload")

#: Classify a sibling item for owner-grouping: ``(owner, payload)`` when the item belongs to an
#: owner (the owner is identified by its ``_id_``), or ``None`` when it does not group.
OwnerClassifier: TypeAlias = Callable[
    [_Item], Optional[Tuple["SymbolicExpression", _Payload]]
]


@dataclass(frozen=True)
class OwnerGroup:
    """A group of sibling items that share one owner — the unit of owner-based aggregation, shared
    by the match construction-pattern grouping (a position's x/y/z) and the ``set_of`` selection
    grouping (the department and salary of one Employee)."""

    owner: SymbolicExpression
    """The shared owner expression (the position, the selected variable)."""

    items: List[object]
    """The grouped payloads, in source order (attribute assignments, terminal hops, …)."""


def group_by_owner(
    items: List[_Item], classify: OwnerClassifier
) -> Tuple[List[OwnerGroup], List[_Item]]:
    """
    Group *items* by their owner's id — all items of one owner together, owners in first-seen order
    — partitioning out the items that do not classify.

    Used by the match planner: a construction pattern's single-hop equalities aggregate per object
    (``position.x/y/z`` group under ``position``), while multi-hop / non-equality conditions fall to
    the ungrouped remainder.

    :param items: The sibling items to group.
    :param classify: Maps an item to ``(owner, payload)`` or ``None`` (does not group).
    :return: ``(groups, ungrouped)`` — the owner groups and the items that did not classify.

    >>> employee = variable(Employee, [])
    >>> groups, ungrouped = group_by_owner(["department", "salary"], lambda item: (employee, item))
    >>> groups[0].items, ungrouped
    (['department', 'salary'], [])
    """
    builders: Dict[object, Tuple[SymbolicExpression, List[object]]] = {}
    order: List[object] = []
    ungrouped: List[_Item] = []
    for item in items:
        decomposed = classify(item)
        if decomposed is None:
            ungrouped.append(item)
            continue
        owner, payload = decomposed
        if owner._id_ not in builders:
            builders[owner._id_] = (owner, [])
            order.append(owner._id_)
        builders[owner._id_][1].append(payload)
    groups = [
        OwnerGroup(owner=builders[key][0], items=builders[key][1]) for key in order
    ]
    return groups, ungrouped


def group_consecutive_by_owner(
    items: List[_Item], classify: OwnerClassifier
) -> List[Union[OwnerGroup, _Item]]:
    """
    Replace each maximal *consecutive* run of two-or-more items sharing one owner with a single
    :class:`OwnerGroup`, leaving every other item (a lone item, or one that does not classify) in
    place — the order-preserving analogue of :func:`reduce_conjuncts` for owner aggregation.

    Used by the ``set_of`` selection grouping: a contiguous run of attributes on one owner folds
    into a shared genitive (*"the department and salary of an Employee"*), while a run of one, or a
    relational terminal, is said on its own.

    :param items: The sibling items, in order.
    :param classify: Maps an item to ``(owner, payload)`` or ``None`` (not foldable).
    :return: The items with each consecutive same-owner run of length ≥ 2 replaced by an
        ``OwnerGroup``; everything else passes through unchanged.

    >>> employee = variable(Employee, [])
    >>> folded = group_consecutive_by_owner(["department", "salary"], lambda item: (employee, item))
    >>> folded[0].items
    ['department', 'salary']
    """
    result: List[Union[OwnerGroup, _Item]] = []
    run_owner: Optional[SymbolicExpression] = None
    run_items: List[_Item] = []
    run_payloads: List[object] = []

    def flush() -> None:
        if len(run_payloads) > 1:
            result.append(OwnerGroup(owner=run_owner, items=list(run_payloads)))
        else:
            result.extend(run_items)

    for item in items:
        decomposed = classify(item)
        if decomposed is not None and (
            run_owner is None or decomposed[0]._id_ == run_owner._id_
        ):
            run_owner, payload = decomposed
            run_items.append(item)
            run_payloads.append(payload)
            continue
        flush()
        if decomposed is None:
            result.append(item)
            run_owner, run_items, run_payloads = None, [], []
        else:
            run_owner = decomposed[0]
            run_items = [item]
            run_payloads = [decomposed[1]]
    flush()
    return result


def coindexed_natural_parts(fold: CoindexedFold) -> Optional[CoindexedNaturalParts]:
    """
    The pieces for the natural *"the <a> and <b> of <shared> have the same …"* rendering, or
    ``None`` when the fold should use the faithful *"… are <op> those of …"* form instead.

    The natural form applies only to an equality fold whose two prefixes are *siblings* — rooted at
    the same variable and identical in every hop but the last (``p.begin`` vs ``p.end``), so the
    shared structure (``p``) factors out and the differing final hops coordinate.

    :param fold: The co-indexed fold.
    :return: The natural-form pieces, or ``None`` for the faithful fallback.

    >>> import dataclasses
    >>> Span = dataclasses.make_dataclass("Span", [("low", int), ("high", int)])
    >>> Pair = dataclasses.make_dataclass("Pair", [("left", Span), ("right", Span)])
    >>> pair = variable(Pair, [])
    >>> [fold] = fold_coindexed_groups([pair.left.low == pair.right.low, pair.left.high == pair.right.high])
    >>> coindexed_natural_parts(fold).left_hop[0], coindexed_natural_parts(fold).right_hop[0]
    ('left', 'right')
    """
    if fold.operation is not operator.eq:
        return None
    left_chain, left_root = walk_chain(fold.left_prefix_expression)
    right_chain, right_root = walk_chain(fold.right_prefix_expression)
    if not (isinstance(left_root, Variable) and isinstance(right_root, Variable)):
        return None
    if left_root._id_ != right_root._id_:
        return None
    if not left_chain or len(left_chain) != len(right_chain):
        return None
    left_hops = [_attribute_pair(node) for node in left_chain]
    right_hops = [_attribute_pair(node) for node in right_chain]
    if None in left_hops or None in right_hops:
        return None
    if left_hops[:-1] != right_hops[:-1]:
        return None  # prefixes must share everything but their final hop
    return CoindexedNaturalParts(
        shared_prefix_expression=fold.left_prefix_expression._child_,
        left_hop=left_hops[-1],
        right_hop=right_hops[-1],
    )


# %% fragment-level coordination builders


def build_between(
    left_fragment: VerbalizationFragment,
    lower_fragment: VerbalizationFragment,
    upper_fragment: VerbalizationFragment,
    *,
    compact: bool,
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
) -> VerbalizationFragment:
    """
    Build *"<left> is between <low> and <high>"* (or copula-less *"<left> between …"* when *compact*).

    :param left_fragment: The already-rendered left side (a full chain, or a bare attribute).
    :param lower_fragment: Rendered lower-bound value.
    :param upper_fragment: Rendered upper-bound value.
    :param compact: Drop the copula (for HAVING / post-nominal contexts).
    :param number: The number the copula agrees with — *"are between"* for a plural subject.
    :return: The range phrase fragment.

    >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text, WordFragment
    >>> flatten_fragment_to_plain_text(build_between(WordFragment("x"), WordFragment("1"), WordFragment("10"), compact=False))
    'x is between 1 and 10'
    """
    # Spread the range phrase's parts so the clause stays flat ([left, operator, bounds]); the
    # coreference agreement pass re-tags the operator leaf in place for a plural subject ("are
    # between"), which a nested phrase would hide.
    return PhraseFragment(
        parts=[
            left_fragment,
            *between_phrase(
                lower_fragment, upper_fragment, compact=compact, number=number
            ).parts,
        ]
    )


def between_phrase(
    lower_fragment: VerbalizationFragment,
    upper_fragment: VerbalizationFragment,
    *,
    compact: bool,
    number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
) -> VerbalizationFragment:
    """
    Build the subject-less range predicate *"is between <low> and <high>"* (or the copula-less
    *"between <low> and <high>"* when *compact*) — the part of a *between* clause after the subject,
    shared by :func:`build_between` and the shared-subject conjunction's range tail.

    :param lower_fragment: Rendered lower-bound value.
    :param upper_fragment: Rendered upper-bound value.
    :param compact: Drop the copula (a coordinated / post-nominal tail).
    :param number: The number the copula agrees with — *"are between"* for a plural subject.
    :return: The range-predicate fragment.

    >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text, WordFragment
    >>> flatten_fragment_to_plain_text(between_phrase(WordFragment("1"), WordFragment("10"), compact=False))
    'is between 1 and 10'
    """
    return PhraseFragment(
        parts=[
            _between_operator(compact, number),
            oxford_comma(
                [lower_fragment, upper_fragment], Conjunctions.AND.as_fragment()
            ),
        ]
    )


def _between_operator(
    compact: bool, number: GrammaticalNumber
) -> VerbalizationFragment:
    """:return: the *between* operator fragment — the copula-less core when *compact*, else an
    agreeing copula plus *"between"* (*"is between"* / *"are between"*).

    >>> from krrood.entity_query_language.verbalization.fragments.base import flatten_fragment_to_plain_text
    >>> flatten_fragment_to_plain_text(_between_operator(True, GrammaticalNumber.SINGULAR))
    'between'
    >>> flatten_fragment_to_plain_text(_between_operator(False, GrammaticalNumber.SINGULAR))
    'is between'
    """
    if compact:
        return RangePhrases.BETWEEN.as_fragment()
    return copula_with(RangePhrases.BETWEEN.text, number)


#: The largest candidate set spelled out as *"one of A, B, …, or F"*; an empty or larger set
#: falls back to a non-enumerated surface (a type name, a count) the caller chooses.
MAX_SET_MEMBERS = 6


def one_of(candidates: List[VerbalizationFragment]) -> Optional[VerbalizationFragment]:
    """
    Render a small, bounded candidate set as a membership phrase — the shared *"one of A, B, or C"*
    surface used both for a domain-constrained variable's candidate values and for a tuple of
    admissible types.

    The candidates are already-rendered fragments, so each caller decides how a candidate is
    lexicalised (a value via :meth:`RoleFragment.for_literal`, a type via
    :meth:`RoleFragment.for_type`) and only the coordination is shared here.

    :param candidates: The rendered candidate fragments (the caller materialises at most
        :data:`MAX_SET_MEMBERS` ``+ 1`` so an over-cap set is detectable).
    :return: the lone candidate for a singleton, *"one of …, or …"* for two-to-:data:`MAX_SET_MEMBERS`
        candidates, or ``None`` when the set is empty or exceeds the cap (the caller then falls back).

    >>> from krrood.entity_query_language.verbalization.fragments.base import (
    ...     flatten_fragment_to_plain_text, RoleFragment,
    ... )
    >>> options = [RoleFragment.for_literal(value) for value in ("a", "b", "c")]
    >>> flatten_fragment_to_plain_text(one_of(options))
    "one of 'a', 'b', or 'c'"
    >>> one_of([]) is None
    True
    """
    if not candidates or len(candidates) > MAX_SET_MEMBERS:
        return None
    if len(candidates) == 1:
        return candidates[0]
    return PhraseFragment(
        parts=[
            SetMembership.ONE_OF.as_fragment(),
            oxford_comma(candidates, Conjunctions.OR.as_fragment()),
        ]
    )
