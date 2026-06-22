from __future__ import annotations

import operator
from dataclasses import dataclass
from enum import Enum, auto
from typing_extensions import (
    TYPE_CHECKING,
    Callable,
    Dict,
    List,
    Optional,
    Tuple,
    Union,
)

from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.verbalization.fragments.base import (
    oxford_comma,
    PhraseFragment,
    Fragment,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    RangePhrases,
)

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression

#: Hashable identity of a pure attribute chain: ``(root variable id, ((name, owner), …))``.
ChainKey = Tuple


@dataclass
class RangeFold:
    """A folded lower/upper bound pair on one attribute chain."""

    chain_expression: SymbolicExpression
    """The shared attribute chain (e.g. ``t.booking_date``)."""

    lower_expression: SymbolicExpression
    """The lower-bound value expression (the ``>=`` / ``>`` right operand)."""

    upper_expression: SymbolicExpression
    """The upper-bound value expression (the ``<=`` / ``<`` right operand)."""


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
    """The co-indexed terminal attributes ``(name, owner)``, in source order — the *"month and
    year"* the comparators range over."""

    left_prefix_expression: SymbolicExpression
    """An exemplar of the left chain minus its terminal hop (e.g. ``p.begin``)."""

    right_prefix_expression: SymbolicExpression
    """An exemplar of the right chain minus its terminal hop (e.g. ``p.end``)."""


@dataclass(frozen=True)
class CoindexedNaturalParts:
    """The pieces of the natural *"the <a> and <b> of <shared> have the same …"* rendering."""

    shared_prefix_expression: SymbolicExpression
    """The chain the two prefixes share (e.g. ``p.period``), rendered via the normal recursion."""

    left_hop: Tuple[str, type]
    """The left prefix's distinguishing final hop ``(name, owner)`` (e.g. ``begin``)."""

    right_hop: Tuple[str, type]
    """The right prefix's distinguishing final hop ``(name, owner)`` (e.g. ``end``)."""


class _Bound(Enum):
    """Internal marker for the direction of a bound comparison in range folding."""

    LOWER = auto()
    UPPER = auto()


def _chain_key(expression: SymbolicExpression) -> Optional[ChainKey]:
    """:return: The hashable identity of a pure attribute chain — ``(root_id, ((name, owner),
    …))`` — or ``None``."""
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


def _classify(conjunct: SymbolicExpression) -> Optional[Tuple[ChainKey, _Bound]]:
    """
    :param conjunct: A candidate conjunct.
    :return: ``(chain_key, _Bound)`` when *conjunct* is a bound comparison, else ``None``.
    """
    if not isinstance(conjunct, Comparator):
        return None
    key = _chain_key(conjunct.left)
    if key is None:
        return None
    if conjunct.operation in (operator.gt, operator.ge):
        return key, _Bound.LOWER
    if conjunct.operation in (operator.lt, operator.le):
        return key, _Bound.UPPER
    return None


def fold_range_pairs(
    conjuncts: List[SymbolicExpression],
) -> List[Union[SymbolicExpression, RangeFold]]:
    """
    Fold complementary lower/upper bound comparisons on the same chain into range items,
    preserving the order of everything else.

    This is the coordination (aggregation) microplanning task — conjunction reduction folding
    ``x >= low`` and ``x <= high`` into ``x is between low and high``. Direction (not position) decides
    which operand is the lower vs upper bound, so ``t.x <= high`` written before ``t.x >= low`` still
    yields ``between low and high``.

    References:

    * Reiter, E. & Dale, R. (2000), "Building Natural Language Generation Systems", CUP —
      aggregation as a microplanning task.
    * Dalianis, H. (1999), "Aggregation in Natural Language Generation", *Computational
      Intelligence* 15(4) — aggregation realised via coordination / conjunction reduction.

    :param conjuncts: A flat list of conjuncts (e.g. the operands of an ``AND``).
    :return: A list whose items are either the original expressions or range folds.
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
                if bound is _Bound.UPPER
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
    """
    return any(isinstance(item, RangeFold) for item in fold_range_pairs(conjuncts))


def _attribute_pair(node: SymbolicExpression) -> Optional[Tuple[str, type]]:
    """:return: ``(name, owner)`` when *node* is an ``Attribute`` hop, else ``None``."""
    if isinstance(node, Attribute):
        return (node._attribute_name_, node._owner_class_)
    return None


def _terminal_attribute(expression: SymbolicExpression) -> Optional[Attribute]:
    """:return: The leaf ``Attribute`` of a ``MappedVariable`` chain (e.g. ``month`` of
    ``p.begin.month``), or ``None`` when the chain does not end in an attribute."""
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
    left_prefix_key = _chain_key(conjunct.left._child_)
    right_prefix_key = _chain_key(conjunct.right._child_)
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
    comparators is never folded.

    :param items: A list of conjuncts, possibly already containing :class:`RangeFold` items.
    :return: The list with each co-indexed group reduced to a single fold at the group's first
        position.
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
            terminals=[signatures[index][1] for index in indices],
            left_prefix_expression=exemplar.left._child_,
            right_prefix_expression=exemplar.right._child_,
        )
        for index in indices[1:]:
            dropped[index] = True
    return [slot for index, slot in enumerate(slots) if not dropped[index]]


def reduce_conjuncts(
    conjuncts: List[SymbolicExpression],
) -> List[Union[SymbolicExpression, RangeFold, CoindexedFold]]:
    """
    Reduce a flat conjunct list by both coordination folds — range pairs then co-indexed groups —
    the single entry every conjunct-rendering caller uses so neither has to know a fold exists.

    :param conjuncts: A flat list of conjuncts (e.g. the operands of an ``AND``).
    :return: The reduced list (raw expressions interleaved with range / co-indexed folds).
    """
    return fold_coindexed_groups(fold_range_pairs(list(conjuncts)))


def coindexed_natural_parts(fold: CoindexedFold) -> Optional[CoindexedNaturalParts]:
    """
    The pieces for the natural *"the <a> and <b> of <shared> have the same …"* rendering, or
    ``None`` when the fold should use the faithful *"… are <op> those of …"* form instead.

    The natural form applies only to an equality fold whose two prefixes are *siblings* — rooted at
    the same variable and identical in every hop but the last (``p.begin`` vs ``p.end``), so the
    shared structure (``p``) factors out and the differing final hops coordinate.

    :param fold: The co-indexed fold.
    :return: The natural-form pieces, or ``None`` for the faithful fallback.
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


def fragment_for_folded_conjunct(
    item: Union[SymbolicExpression, RangeFold],
    child: Callable[[SymbolicExpression], Fragment],
    *,
    compact: bool,
) -> Fragment:
    """
    Render one folded conjunct: a range fold becomes a *between* phrase; any other conjunct is
    rendered via *child*.

    :param item: A folded conjunct (a range fold or a raw expression).
    :param child: The fold continuation rendering a raw expression.
    :param compact: Drop the copula in the *between* phrase (HAVING / post-nominal contexts).
    :return: The fragment for *item*.
    """
    if isinstance(item, RangeFold):
        return build_between(
            child(item.chain_expression),
            child(item.lower_expression),
            child(item.upper_expression),
            compact=compact,
        )
    return child(item)


def build_between(
    left_fragment: Fragment,
    lower_fragment: Fragment,
    upper_fragment: Fragment,
    *,
    compact: bool,
) -> Fragment:
    """
    Build *"<left> is between <low> and <high>"* (or copula-less *"<left> between …"* when *compact*).

    :param left_fragment: The already-rendered left side (a full chain, or a bare attribute).
    :param lower_fragment: Rendered lower-bound value.
    :param upper_fragment: Rendered upper-bound value.
    :param compact: Drop the copula (for HAVING / post-nominal contexts).
    :return: The range phrase fragment.
    """
    op = (RangePhrases.BETWEEN if compact else RangePhrases.IS_BETWEEN).as_fragment()
    bounds = oxford_comma(
        [lower_fragment, upper_fragment], Conjunctions.AND.as_fragment()
    )
    return PhraseFragment(parts=[left_fragment, op, bounds])
