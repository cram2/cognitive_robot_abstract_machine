"""
Subject-aware grouping of WHERE conjuncts into relational-binding scopes.

A query subject can *introduce* a secondary entity through a relational attribute
(``robot.assigned_to == mission``) and separately *restrict* that entity
(``mission.priority > 2``). This module collapses such a binding together with the
entity's own conjuncts into a single :class:`RelationalBindingFold`, so the pair reads
as one nested relative clause on the subject noun (*"a Robot that is assigned to a
Mission with priority greater than 2"*) rather than a binding clause plus an orphaned
*"such that the priority of the Mission …"* residual.

It runs inside :func:`~…placement.as_subject_restrictions` right after
:func:`reduce_conjuncts` — where the subject is in hand — so it sits beside the
(subject-agnostic) coordination reducer rather than inside it. Gathering is transitive:
an entity bound within the subtree pulls its own restrictions in too, so nesting
generalises to arbitrary depth through the placement layer's recursion.
"""

from __future__ import annotations

import operator
import uuid
from dataclasses import dataclass

from typing_extensions import Dict, List, Optional, Set, Tuple, Union

from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
from krrood.entity_query_language.core.variable import Literal, Variable
from krrood.entity_query_language.operators.comparator import Comparator
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_none_literal,
    references,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    ConjunctList,
    FoldNode,
    RangeFold,
)
from krrood.entity_query_language.verbalization.relational_attributes import (
    relational_verb,
)


@dataclass
class RelationalBindingFold:
    """
    A subject→entity relational binding collapsed with the entity's own restrictions.

    Rendered as a subject-relative clause naming the entity with its restrictions nested
    onto it (*"that is assigned to a Mission with priority greater than 2"*).
    """

    relation_hop: Attribute
    """The subject's single relational-attribute hop (``robot.assigned_to``) whose verb names the
    clause."""

    entity: Variable
    """
    The secondary variable the relation introduces (the clause's object).
    """

    nested: List[FoldNode]
    """The entity's own conjuncts, transitively (removed from the subject's list); empty for a bare
    binding that only names the entity."""


BoundConjunct = Union[FoldNode, RelationalBindingFold]
"""
A conjunct after binding grouping: an ordinary fold node or a collapsed relational
binding.
"""


def bind_relational_entities(
    items: ConjunctList, subject: Variable
) -> List[BoundConjunct]:
    """:return: *items* with each relational binding between *subject* and another entity replaced by
    a :class:`RelationalBindingFold` that also absorbs that entity's own (transitively reachable)
    conjuncts; order preserved, everything else untouched. A no-op when no such binding folds.

    A binding is recognised in either direction: ``subject.<relational_attr> == entity`` (the relation
    on the subject) and ``entity.<relational_attr> == subject`` (the relation on the other entity,
    pointing back — the referent-unification shape). A *reverse* binding only folds when the entity
    actually carries a restriction to nest; a bare one keeps its existing *"it is <participle> …"*
    rendering.
    """
    candidates: Dict[int, Tuple[Attribute, Variable, bool]] = {}
    for index, item in enumerate(items):
        binding = _relational_binding(item)
        if binding is None:
            continue
        relation_hop, owner, other = binding
        if owner._id_ == subject._id_:
            candidates[index] = (relation_hop, other, False)
        elif other._id_ == subject._id_:
            candidates[index] = (relation_hop, owner, True)
    if not candidates:
        return list(items)
    claimed: Set[int] = set(candidates)
    folds: Dict[int, RelationalBindingFold] = {}
    for index, (relation_hop, entity, is_reverse) in candidates.items():
        nested_indices = _subtree_indices(items, entity, subject, claimed)
        if is_reverse and not nested_indices:
            # A bare reverse binding reads fine as "it is assigned to a Mission" already; only fold it
            # when there is a restriction to nest onto the entity noun.
            claimed.discard(index)
            continue
        claimed.update(nested_indices)
        folds[index] = RelationalBindingFold(
            relation_hop=relation_hop,
            entity=entity,
            nested=[items[nested] for nested in sorted(nested_indices)],
        )
    return [
        folds[index] if index in folds else item
        for index, item in enumerate(items)
        if index in folds or index not in claimed
    ]


def _subtree_indices(
    items: ConjunctList, entity: Variable, subject: Variable, claimed: Set[int]
) -> Set[int]:
    """:return: the indices of *items* that restrict *entity* or an entity bound within its subtree —
    the transitive closure over relational bindings — excluding already-*claimed* indices and any
    conjunct whose value side references the outer *subject* (a cross-correlation stays in the subject
    residual rather than being mis-attributed to the entity noun)."""
    reachable: Set[uuid.UUID] = {entity._id_}
    collected: Set[int] = set()
    changed = True
    while changed:
        changed = False
        for index, item in enumerate(items):
            if index in claimed or index in collected:
                continue
            root = _item_root(item)
            if root not in reachable or not _is_clean_restriction(item, subject):
                continue
            collected.add(index)
            introduced = _relational_binding(item)
            if introduced is not None:
                reachable.add(introduced[2]._id_)
            changed = True
    return collected


def _relational_binding(
    item: BoundConjunct,
) -> Optional[Tuple[Attribute, Variable, Variable]]:
    """:return: ``(relation_hop, owner, entity)`` when *item* is ``owner.<relational_attr> == entity``
    (either operand order, *entity* a bare variable distinct from *owner*), else ``None``. A
    plain-attribute equality or an ``== None`` absence is not a binding."""
    if not isinstance(item, Comparator) or item.operation is not operator.eq:
        return None
    for attribute_side, entity_side in (
        (item.left, item.right),
        (item.right, item.left),
    ):
        if not isinstance(entity_side, Variable) or isinstance(entity_side, Literal):
            continue
        relation_hop = _single_relational_hop(attribute_side)
        if relation_hop is None:
            continue
        _, owner = walk_chain(attribute_side)
        if not isinstance(owner, Variable) or owner._id_ == entity_side._id_:
            continue
        return relation_hop, owner, entity_side
    return None


def _single_relational_hop(expression: object) -> Optional[Attribute]:
    """:return: the single relational-attribute hop when *expression* is ``<variable>.<relational_attr>``
    (one hop whose name is a relation verb), else ``None``."""
    if not isinstance(expression, MappedVariable):
        return None
    chain, _ = walk_chain(expression)
    if len(chain) != 1 or not isinstance(chain[0], Attribute):
        return None
    if relational_verb(chain[0]._attribute_name_) is None:
        return None
    return chain[0]


def _item_root(item: BoundConjunct) -> Optional[uuid.UUID]:
    """:return: the id of the variable *item* restricts — the owner of a binding, else the root of a
    comparator's/range-fold's chain, else ``None``."""
    binding = _relational_binding(item)
    if binding is not None:
        return binding[1]._id_
    if isinstance(item, Comparator):
        _, root = walk_chain(item.left)
        return root._id_ if isinstance(root, Variable) else None
    if isinstance(item, RangeFold):
        _, root = walk_chain(item.chain_expression)
        return root._id_ if isinstance(root, Variable) else None
    return None


def _is_clean_restriction(item: BoundConjunct, subject: Variable) -> bool:
    """:return: whether *item* is safe to nest — a relational binding (which defines deeper
    structure) always is; a comparator/range-fold is only when its value side does not reference the
    outer *subject*, so a cross-correlation is left in the subject residual. Anything else is not
    nested (conservative)."""
    if _relational_binding(item) is not None:
        return True
    if isinstance(item, Comparator):
        return not references(item.right, subject)
    if isinstance(item, RangeFold):
        return not (
            references(item.lower_expression, subject)
            or references(item.upper_expression, subject)
        )
    return False
