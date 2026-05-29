"""
EQL-native Single-Class Ripple Down Rules.

The rule tree is a live EQL query DAG over a shared case variable. Classification is
plain EQL evaluation (via :func:`classify_case`); fitting grows the DAG in place using
the observed firing rule as the anchor:

* wrong conclusion  -> add a **refinement** at the firing rule (it overrides)
* no rule fired     -> add an **alternative** at the conditions root

Single-class means conclusions are mutually exclusive: each case resolves to one value.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from functools import cached_property

from typing_extensions import Any, List, Optional, Type

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import CanBehaveLikeAVariable
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.factories import add, entity, variable
from krrood.entity_query_language.query.query import Query
from krrood.entity_query_language.rdr.conclusion_domain import (
    ConclusionDomain,
    resolve_conclusion_domain,
)
from krrood.entity_query_language.rdr.expert import Expert
from krrood.entity_query_language.rdr.utils import UNSET
from krrood.entity_query_language.rdr.observer import (
    ClassificationTrace,
    ConclusionObserver,
    classify_case,
    trace_case,
)
from krrood.entity_query_language.rdr.rule_tree import (
    insert_alternative,
    insert_refinement,
)
from krrood.entity_query_language.rdr.rule_tree_view import (
    DEFAULT_HEAD,
    DEFAULT_TAIL,
    render_rule_tree,
)
from krrood.entity_query_language.scope import (
    attach_definition_scope,
    capture_caller_scope,
)


@dataclass
class EQLSingleClassRDR:
    """A single-class RDR whose rule tree is a live EQL expression DAG."""

    case_type: Type
    """The type of case the RDR classifies (e.g. ``Animal``)."""
    conclusion_attribute_name: str
    """The underspecified attribute the RDR predicts (e.g. ``"species"``)."""

    case_variable: Variable = field(init=False)
    """The shared EQL variable the whole rule tree ranges over."""
    conclusion_variable: CanBehaveLikeAVariable = field(init=False)
    """The attribute expression the rules conclude on (``case_variable.<attr>``)."""
    query: Optional[Query] = field(init=False, default=None)
    """The root rule-tree query; ``None`` until the first rule is added."""

    def __post_init__(self) -> None:
        self.case_variable = variable(self.case_type, domain=[])
        self.conclusion_variable = getattr(
            self.case_variable, self.conclusion_attribute_name
        )
        # Snapshot the caller's namespace (where the RDR was created) so an interactive
        # expert can be driven with the same scope, plus the EQL factories on top.
        attach_definition_scope(self.case_variable, capture_caller_scope())

    @classmethod
    def from_underspecified(cls, template: Any) -> "EQLSingleClassRDR":
        """
        Build an RDR from an underspecified ``Match`` template: the lone ``...`` attribute
        defines what the RDR predicts.

        :param template: e.g. ``underspecified(Animal)(species=...)``.
        :return: An RDR with ``case_type`` and ``conclusion_attribute_name`` taken from the
            template's single underspecified slot.
        """
        from krrood.entity_query_language.rdr.underspecified import UnderspecifiedMatch

        statement = UnderspecifiedMatch(template)
        return cls(statement.case_type, statement.target_attribute_name)

    def classify(self, case: Any) -> Optional[Any]:
        """:return: The inferred conclusion for ``case``, or ``None`` if no rule fires."""
        if self.query is None:
            return None
        return self._observe(case).conclusion

    def _observe(self, case: Any) -> ConclusionObserver:
        return classify_case(
            self.query, self.case_variable, self.conclusion_variable, case
        )

    def _trace(self, case: Any) -> ClassificationTrace:
        return trace_case(
            self.query,
            self.case_variable,
            self.conclusion_variable,
            case,
            self.conditions_root,
        )

    def render_tree(
        self,
        case: Any,
        *,
        head: int = DEFAULT_HEAD,
        tail: int = DEFAULT_TAIL,
        use_color: bool = True,
    ) -> str:
        """
        Render this RDR's rule tree for ``case`` as coloured text (fired/evaluated/skipped).

        :param case: The case to classify and explain.
        :param head: Rules to show before the elision marker.
        :param tail: Rules to show after it (ending on the firing rule).
        :param use_color: Whether to colour rows by status.
        :return: The rendered tree, or ``""`` when the RDR has no rules yet.
        """
        if self.query is None:
            return ""
        return render_rule_tree(
            self._trace(case), head=head, tail=tail, use_color=use_color
        )

    def fit_case(
        self, case: Any, target: Any = UNSET, expert: Optional[Expert] = None
    ) -> Any:
        """
        Ensure the RDR classifies ``case`` as ``target``, growing the rule tree when it does
        not.

        When ``target`` is ``UNSET`` (no ground truth) the expert supplies **both** the
        conclusion and its conditions via :meth:`Expert.ask_for_rule`; otherwise only the
        conditions are requested (the conclusion is the known ``target``).

        :return: The conclusion now associated with ``case`` (``target`` when given, else the
            expert's conclusion).
        """
        if expert is None:
            raise ValueError("fit_case requires an expert.")

        trace = None if self.query is None else self._trace(case)
        current = trace.conclusion if trace is not None else UNSET

        if target is not UNSET and current == target:
            return target

        if target is UNSET:
            target, condition = expert.ask_for_rule(
                case, self.case_variable, self.conclusion_domain, current, trace
            )
            if condition is None:
                # The expert kept the current conclusion; nothing to insert.
                return target
        else:
            condition = expert.ask_for_conditions(
                case, self.case_variable, target, current, trace
            )

        self._insert_rule(trace, current, condition, target)
        return target

    def _insert_rule(
        self,
        trace: Optional[ClassificationTrace],
        current: Optional[Any],
        condition: SymbolicExpression,
        target: Any,
    ) -> None:
        """Splice a new rule into the tree, choosing first-rule / alternative / refinement."""
        if self.query is None:
            # First rule: seed the tree.
            self.query = entity(self.case_variable).where(condition)
            with self.query:
                add(self.conclusion_variable, target)
            self.query.build()
        elif current is UNSET:
            # Nothing fired: attach an alternative at the conditions root.
            insert_alternative(
                self.query._conditions_root_,
                condition,
                self.conclusion_variable,
                target,
            )
        else:
            # A rule fired with the wrong value: refine it so the new condition overrides.
            insert_refinement(
                trace.firing_anchor,
                condition,
                self.conclusion_variable,
                target,
            )

    def fit(
        self,
        cases: List[Any],
        targets: Optional[List[Any]] = None,
        expert: Optional[Expert] = None,
    ) -> "EQLSingleClassRDR":
        """
        Fit the RDR over ``cases``. When ``targets`` is given it is paired with ``cases``
        (ground-truth fitting); when ``None`` the expert labels each case (the no-target
        ``ask_for_rule`` path), so each case is paired with the ``UNSET`` sentinel rather than
        a literal ``None`` target.
        """
        paired_targets = targets if targets is not None else [UNSET] * len(cases)
        for case, target in zip(cases, paired_targets):
            self.fit_case(case, target, expert)
        return self

    @property
    def conditions_root(self) -> Optional[SymbolicExpression]:
        """The root of the rule tree's condition DAG, or ``None`` if empty."""
        return self.query._conditions_root_ if self.query is not None else None

    @cached_property
    def conclusion_domain(self) -> ConclusionDomain:
        """The allowable-value domain of the predicted attribute, resolved from its type."""
        return resolve_conclusion_domain(self.case_type, self.conclusion_attribute_name)
