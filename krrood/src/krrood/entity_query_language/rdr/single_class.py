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

from typing_extensions import Any, List, Optional, TYPE_CHECKING, Type, Self

if TYPE_CHECKING:
    from krrood.entity_query_language.rdr.progress import ProgressReporter

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
from krrood.entity_query_language.rdr.backward_inference import (
    BackwardInferenceIndex,
    ConclusionKnowledge,
)
from krrood.entity_query_language.rdr.condition_resolver import ConditionResolver
from krrood.entity_query_language.rdr.corner_case import CornerCaseStore

_FITTING_DESCRIPTION = "Fitting RDR"
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
from krrood.entity_query_language.rdr.serialization import save_rdr_with_case
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
    save_path: Optional[str] = field(default=None)
    """When set, the RDR is automatically saved to this path after every rule insertion."""
    corner_cases: CornerCaseStore = field(default_factory=CornerCaseStore)
    """Maps each rule's condition-node id to the corner case that triggered its creation."""
    _backward_index: BackwardInferenceIndex = field(
        default_factory=BackwardInferenceIndex, repr=False
    )
    """Lazy cache for backward-inference queries. Invalidated on every rule insertion."""
    condition_resolver: Optional[ConditionResolver] = field(default=None)
    """Optional resolver for automatic condition derivation using backward inference.

    When set, :meth:`fit_case` attempts to derive a differentiating condition automatically
    before asking the expert. Only applies to the refinement branch (wrong rule fired).
    Use :class:`~krrood.entity_query_language.rdr.condition_resolver.ChainConditionResolver`
    ``.backward_inference_default()`` for the standard two-phase resolution strategy.
    """

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

        :param case: The case to classify.
        :param target: The known correct conclusion, or ``UNSET`` when no ground truth is available.
        :param expert: The expert that supplies rule conditions (and conclusion, when ``target`` is ``UNSET``).
        :return: The conclusion now associated with ``case`` (``target`` when given, else the
            expert's conclusion).
        """

        trace = None if self.query is None else self._trace(case)
        current = trace.conclusion if trace is not None else UNSET

        if target is not UNSET and current == target:
            return target

        if expert is None:
            raise ValueError("Expert must be supplied to fit_case")

        corner_case = self.corner_cases.get(
            trace.firing_anchor_id if trace is not None else None
        )

        if target is UNSET:
            target, condition = expert.ask_for_rule(
                case,
                self.case_variable,
                self.conclusion_domain,
                current,
                trace,
                corner_case,
            )
            if condition is None:
                # The expert kept the current conclusion; nothing to insert.
                return target
        else:
            resolved = self._try_auto_resolve(case, target, current, corner_case)
            condition = (
                resolved
                if resolved is not None
                else expert.ask_for_conditions(
                    case, self.case_variable, target, current, trace, corner_case
                )
            )

        self._insert_rule(trace, current, condition, target, case)
        self._backward_index.invalidate()
        return target

    def _try_auto_resolve(
        self,
        case: Any,
        target: Any,
        current: Any,
        corner_case: Optional[Any],
    ) -> Optional[SymbolicExpression]:
        """Attempt to derive a differentiating condition without asking the expert.

        Only active for the refinement branch: returns ``None`` immediately when
        :attr:`condition_resolver` is unset, ``corner_case`` is ``None``, or ``current``
        is ``UNSET``. When active, queries the backward-inference index for both
        ``target`` and ``current`` knowledge and delegates to :attr:`condition_resolver`.

        :param case: The new case being fit.
        :param target: The correct conclusion.
        :param current: The wrong conclusion currently returned by the firing rule.
        :param corner_case: The case that triggered the currently-firing rule's creation.
        :return: An auto-derived EQL condition expression, or ``None`` to fall back to the expert.
        """
        if self.condition_resolver is None or corner_case is None or current is UNSET:
            return None
        target_knowledge = self._backward_index.query(self.conditions_root, target)
        current_knowledge = self._backward_index.query(self.conditions_root, current)
        result = self.condition_resolver.resolve(
            case,
            self.case_variable,
            target,
            current,
            corner_case,
            target_knowledge,
            current_knowledge,
        )
        return result.expression if result is not None else None

    def _insert_rule(
        self,
        trace: Optional[ClassificationTrace],
        current: Optional[Any],
        condition: SymbolicExpression,
        target: Any,
        case: Any,
    ) -> None:
        """Splice a new rule into the tree, choosing first-rule / alternative / refinement."""
        if self.query is None:
            # First rule: seed the tree.
            self.query = entity(self.case_variable).where(condition)
            with self.query:
                add(self.conclusion_variable, target)
            self.query.build()
            new_node = self.query._conditions_root_
        elif current is UNSET:
            # Nothing fired: attach an alternative at the conditions root.
            new_node = insert_alternative(
                self.query._conditions_root_,
                condition,
                self.conclusion_variable,
                target,
            )
        else:
            # A rule fired with the wrong value: refine it so the new condition overrides.
            new_node = insert_refinement(
                trace.firing_anchor,
                condition,
                self.conclusion_variable,
                target,
            )

        self.corner_cases.record(new_node, case)

        if self.save_path is not None:
            save_rdr_with_case(self, self.save_path)

    def fit(
        self,
        cases: List[Any],
        targets: Optional[List[Any]] = None,
        expert: Optional[Expert] = None,
        max_passes: int = 10,
    ) -> Self:
        """
        Fit the RDR over ``cases``. When ``targets`` is given it is paired with ``cases``
        (ground-truth fitting); when ``None`` the expert labels each case (the no-target
        ``ask_for_rule`` path), so each case is paired with the ``UNSET`` sentinel rather than
        a literal ``None`` target.

        When ground-truth ``targets`` are provided, the fit is *convergent*: after each
        pass the model is rechecked and any cases that are now misclassified (because a
        later rule retroactively intercepted them) are re-fitted.  Convergence stops when
        every case is correct or ``max_passes`` is exhausted — whichever comes first.

        Correctly-classified cases on re-passes are idempotent (the expert is never called
        for them), so the overhead is one :meth:`classify` call per case per pass —
        negligible compared to expert interaction.

        When ``targets`` is ``None`` the no-target path has no ground truth to converge
        against, so the method is a single pass (unchanged from previous behaviour).

        :param cases: The case instances to fit.
        :param targets: Optional ground-truth conclusions paired with ``cases``.
            ``None`` triggers the expert-labeling (``ask_for_rule``) path.
        :param expert: The expert that supplies rule conditions.
        :param max_passes: Maximum number of convergent passes.  Defaults to 10.
        :return: This RDR, for chaining.
        """
        paired_targets = targets if targets is not None else [UNSET] * len(cases)
        pending = list(range(len(cases)))

        progress: Optional[ProgressReporter] = None
        if expert is not None:
            progress = expert.interface.make_progress_reporter()
        if progress is not None:
            progress.start(len(pending), _FITTING_DESCRIPTION)

        for pass_num in range(max_passes):
            if pass_num > 0 and progress is not None:
                progress.reset(len(pending))

            for i in pending:
                self.fit_case(cases[i], paired_targets[i], expert)
                if progress is not None:
                    progress.update()

            if targets is None:
                if progress is not None:
                    progress.finish()
                return self

            pending = [
                i
                for i in range(len(cases))
                if self.classify(cases[i]) != paired_targets[i]
            ]
            if not pending:
                break

        if progress is not None:
            progress.finish()
        return self

    @property
    def conditions_root(self) -> Optional[SymbolicExpression]:
        """The root of the rule tree's condition DAG, or ``None`` if empty."""
        return self.query._conditions_root_ if self.query is not None else None

    @cached_property
    def conclusion_domain(self) -> ConclusionDomain:
        """The allowable-value domain of the predicted attribute, resolved from its type."""
        return resolve_conclusion_domain(self.case_type, self.conclusion_attribute_name)

    def what_do_we_know_about(self, conclusion_value: Any) -> ConclusionKnowledge:
        """Return the rule-tree conditions that would produce *conclusion_value*.

        This inspects the rule tree from the perspective of *conclusion_value*,
        walking the conclusion-selector DAG backwards to enumerate every rule path
        that could produce it. Each path yields one
        :class:`~krrood.entity_query_language.rdr.backward_inference.SufficientConditionSet`;
        the full result is a disjunction of all such sets (DNF).

        The result is lazily cached in :attr:`_backward_index` and invalidated on
        every tree mutation, so repeated queries after fitting are O(1).

        :param conclusion_value: The conclusion value to query (e.g. ``Species.molusc``).
        :return: The backward-inference knowledge for *conclusion_value*.
        """
        return self._backward_index.query(self.conditions_root, conclusion_value)
