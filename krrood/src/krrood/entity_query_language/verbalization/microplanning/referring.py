"""
Referring-expression generation — choosing how to *name* an entity each time it
is mentioned.

This is the **referring-expression generation** subtask of microplanning
(Reiter & Dale 2000): deciding between an indefinite first mention (*"a Robot"*),
a definite subsequent mention (*"the Robot"*), a numbered form when one type
occurs several times (*"Robot 1"* / *"Robot 2"*), and a pronoun (*"its …"*) when a
chain is rooted at the current discourse subject.  All of these facets share one
piece of state — what has been mentioned so far — and change for one reason
(referring-expression policy), so they form a single cohesive responsibility.

References:

* Reiter, E. & Dale, R. (2000), "Building Natural Language Generation Systems",
  CUP — referring-expression generation as a microplanning subtask.
* Dale, R. & Reiter, E. (1995), "Computational interpretations of the Gricean
  maxims in the generation of referring expressions", *Cognitive Science* 19(2).
"""

from __future__ import annotations

import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING
from typing_extensions import Dict, List, Optional

from krrood.entity_query_language.core.variable import Variable, Literal
from krrood.entity_query_language.query.query import Entity, Query
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import Definiteness
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.subquery import (
    aggregation_source_root,
    selected_aggregator,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Pronouns

if TYPE_CHECKING:
    from krrood.entity_query_language.core.base_expressions import SymbolicExpression


def _aggregation_source_ids(expression) -> set:
    """
    Return the ``_id_`` of every variable that serves as the source *population*
    of an aggregation sub-query (e.g. the ``BankTransaction`` behind
    ``max(t.amount_details.amount)``).

    Such a variable denotes a population to aggregate over, not a specific
    entity, so it must not consume an entity-disambiguation number — otherwise
    the outer subject would pick up a spurious *"1"* with no matching *"2"*, and
    a constrained aggregation scope would read *"among BankTransaction 2"* rather
    than *"among BankTransactions"*.

    :param expression: Root expression to scan.
    :return: Set of variable ids to exclude from numbering.
    :rtype: set
    """
    ids: set = set()
    for node in expression._all_expressions_:
        if isinstance(node, Entity) and selected_aggregator(node) is not None:
            root = aggregation_source_root(node)
            if root is not None:
                ids.add(root._id_)
    return ids


def _build_disambiguation_map(expression) -> Dict[uuid.UUID, str]:
    """
    Pre-scan *expression* and return a mapping of variable._id_ → display label.

    Types appearing once keep the plain type name; types appearing two or more
    times get "TypeName 1", "TypeName 2", … labels in encounter order.
    Literal nodes are excluded, as are variables that only serve as the source
    population of an aggregation sub-query (see :func:`_aggregation_source_ids`).
    """
    if isinstance(expression, Query):
        expression.build()

    suppressed = _aggregation_source_ids(expression)

    type_to_ids: Dict[str, List[uuid.UUID]] = defaultdict(list)
    seen_ids: set = set()

    for node in expression._all_expressions_:
        if isinstance(node, Variable) and not isinstance(node, Literal):
            if node._id_ in suppressed:
                continue
            type_name = (
                node._type_.__name__
                if getattr(node, "_type_", None)
                else node.__class__.__name__
            )
            if node._id_ not in seen_ids:
                seen_ids.add(node._id_)
                type_to_ids[type_name].append(node._id_)

    result: Dict[uuid.UUID, str] = {}
    for type_name, ids in type_to_ids.items():
        if len(ids) == 1:
            result[ids[0]] = type_name
        else:
            for n, vid in enumerate(ids, 1):
                result[vid] = f"{type_name} {n}"
    return result


@dataclass
class ReferringExpressions:
    """
    Tracks discourse state and chooses the referring expression for each mention.

    Owns the coreference state for a single verbalization pass: which variables
    have been mentioned (:attr:`seen`), the pre-computed disambiguation labels
    (:attr:`disambiguation_map`), and the stack of pronoun-eligible subjects
    (:attr:`coref_subjects`).
    """

    seen: "Dict[uuid.UUID, VerbFragment]" = field(default_factory=dict)
    """Maps expression ``_id_`` → its **label fragment** for every expression already
    verbalized in this pass (a subsequent mention renders as *"the <label>"*).  Storing a
    fragment (not a flattened string) keeps discourse state structural — surface inflection
    is applied once, later, by the morphology pass."""

    disambiguation_map: Dict[uuid.UUID, str] = field(default_factory=dict)
    """Maps variable ``_id_`` → display label, pre-computed before verbalization
    begins.  Single-type variables keep the plain type name; colliding types get
    ``"TypeName 1"``, ``"TypeName 2"`` labels."""

    coref_subjects: List[uuid.UUID] = field(default_factory=list)
    """Stack of subject variable ``_id_`` s (or ``None`` when the enclosing clause
    has no single coreference subject, e.g. ``SetOf``).  A chain rooted at the
    top-of-stack subject is eligible for pronominalisation — see :meth:`pronoun_for`."""

    @classmethod
    def from_expression(cls, expression) -> ReferringExpressions:
        """Create an instance with the disambiguation map pre-built for *expression*."""
        return cls(disambiguation_map=_build_disambiguation_map(expression))

    def push_subject(self, var) -> None:
        """
        Push *var* as the current coreference subject.

        Stores the variable's ``_id_`` when *var* is a single
        :class:`~krrood.entity_query_language.core.variable.Variable`; otherwise
        stores ``None`` so no pronoun fires (e.g. a ``SetOf`` with several subjects).
        Always pushes exactly one frame so callers can pair it with
        :meth:`pop_subject` unconditionally.

        :param var: The subject variable being described, or any non-Variable.
        """
        self.coref_subjects.append(var._id_ if isinstance(var, Variable) else None)

    def pop_subject(self) -> None:
        """Pop the current coreference subject pushed by :meth:`push_subject`."""
        if self.coref_subjects:
            self.coref_subjects.pop()

    @property
    def current_subject_id(self):
        """``_id_`` of the current coreference subject, or ``None`` when there is none."""
        return self.coref_subjects[-1] if self.coref_subjects else None

    def register(self, expression, label: VerbFragment) -> None:
        """Record *expression*'s label **fragment** (reused for a definite later mention)."""
        self.seen[expression._id_] = label

    def register_label(self, expression, text: str) -> None:
        """Record *expression*'s label as a plain ``VARIABLE``-role noun (the common case)."""
        self.seen[expression._id_] = RoleFragment(text=text, role=SemanticRole.VARIABLE)

    def alias(self, target, source) -> None:
        """Give *target* the label already registered for *source* (no-op if *source* unseen)."""
        label = self.seen.get(source._id_)
        if label is not None:
            self.seen[target._id_] = label

    def label_of(self, expression) -> Optional[VerbFragment]:
        """The registered label fragment for *expression*, or ``None``."""
        return self.seen.get(getattr(expression, "_id_", None))

    def seen_reference(self, expression) -> Optional[VerbFragment]:
        """
        Return *"the <label>"* when *expression* has already been verbalized in this pass,
        else ``None``.

        :param expression: Any expression carrying an ``_id_``.
        :return: The definite-reference phrase, or ``None`` when *expression* is unseen.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment or None
        """
        variable_id = getattr(expression, "_id_", None)
        if variable_id is None or variable_id not in self.seen:
            return None
        return NounPhrase(
            head=self.seen[variable_id], definiteness=Definiteness.DEFINITE
        )

    def pronoun_for(self, root) -> Optional[VerbFragment]:
        """
        Return the possessive-pronoun fragment (*"its"*) for *root* when it is the
        current, unambiguous, already-introduced coreference subject; else ``None``.

        Eligibility (all required): *root* is a Variable; ``root._id_`` is the top of
        :attr:`coref_subjects`; *root* is not numbered in :attr:`disambiguation_map`;
        *root* has already been mentioned (is in :attr:`seen`).

        :param root: Candidate chain-root expression.
        :return: The *"its"* fragment, or ``None`` when pronominalisation is unsafe.
        :rtype: ~krrood.entity_query_language.verbalization.fragments.base.VerbFragment or None
        """
        if not isinstance(root, Variable):
            return None
        if root._id_ != self.current_subject_id or root._id_ not in self.seen:
            return None
        type_name = root._type_.__name__ if getattr(root, "_type_", None) else None
        label = self.disambiguation_map.get(root._id_, type_name)
        if type_name is not None and label != type_name:
            return None
        return Pronouns.ITS.as_fragment()

    def noun_for_parts(self, var) -> tuple[Definiteness, str]:
        """
        Return ``(Definiteness, label)`` for *var* — the coreference decision, expressed
        directly as the realisation feature the determiner phase consumes.

        Consults :attr:`disambiguation_map` for the display label, then :attr:`seen`
        for first vs. subsequent mention, recording the mention as a side effect: a
        numbered variable (*"Robot 2"*) is ``BARE``, a first mention ``INDEFINITE``
        (*"a Robot"*), a subsequent mention ``DEFINITE`` (*"the Robot"*).

        :param var: A :class:`~krrood.entity_query_language.core.variable.Variable` instance.
        :return: Tuple of ``(Definiteness, display_label)``.
        :rtype: tuple
        """
        type_name = (
            var._type_.__name__
            if getattr(var, "_type_", None)
            else var.__class__.__name__
        )
        label = self.disambiguation_map.get(var._id_, type_name)
        is_numbered = label != type_name
        if var._id_ in self.seen:
            return (Definiteness.BARE if is_numbered else Definiteness.DEFINITE), label
        self.seen[var._id_] = RoleFragment(text=label, role=SemanticRole.VARIABLE)
        return (Definiteness.BARE if is_numbered else Definiteness.INDEFINITE), label
