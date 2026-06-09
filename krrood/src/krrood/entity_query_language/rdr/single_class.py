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

import warnings
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
from krrood.entity_query_language.rdr.condition_resolver import (
    ConditionResolver,
    ResolvedCondition,
    ResolutionMode,
)
from krrood.entity_query_language.rdr.corner_case import CornerCaseStore
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

_FITTING_DESCRIPTION = "Fitting RDR"
"""
The description of the fitting RDR situation, used in error messages and documentation.
"""


class RDRConvergenceWarning(UserWarning):
    """Emitted when the RDR fitting loop detects oscillation and terminates early.

    The pending set of misclassified cases repeated a previously seen signature,
    meaning the tree is oscillating rather than converging. Inspect the warning
    message for the clashing case reprs; if :attr:`EQLSingleClassRDR.save_path`
    is set the partially fitted model is saved there for inspection.
    """


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
    ``.backward_inference_default()`` for the standard target-knowledge resolution strategy.
    """
    resolution_mode: ResolutionMode = field(default=ResolutionMode.SILENT)
    """Controls whether an auto-resolved condition is silently inserted or shown as a hint.

    :attr:`~krrood.entity_query_language.rdr.condition_resolver.ResolutionMode.SILENT`
    (default) inserts the condition directly without asking the expert.
    :attr:`~krrood.entity_query_language.rdr.condition_resolver.ResolutionMode.HINT`
    passes the resolved condition to the expert as a pre-seeded suggestion; the expert
    may accept it unchanged or overwrite it.  Has no effect when :attr:`condition_resolver`
    is ``None``.
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
    def from_underspecified(cls, template: Any) -> EQLSingleClassRDR:
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
        """
        Observe a case and return a conclusion observer for it. An observer is a tool for
        tracking the progress of classification and can be used to debug or explain the
        classification process.

        :param case: The case to observe.
        :return: A conclusion observer for the case.
        """
        return classify_case(
            self.query, self.case_variable, self.conclusion_variable, case
        )

    def _trace(self, case: Any) -> ClassificationTrace:
        """
        Trace the classification process for a given case, returning a detailed trace of
        the classification steps.

        :param case: The case to trace.
        :return: A classification trace object.
        """
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
            resolved = self._try_auto_resolve(
                case,
                target,
                current,
                corner_case,
                trace.firing_anchor if trace is not None else None,
            )
            condition = self._apply_resolution(
                resolved, case, target, current, trace, corner_case, expert
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
        firing_anchor: Optional[SymbolicExpression] = None,
    ) -> Optional[ResolvedCondition]:
        """Attempt to derive a differentiating condition without asking the expert.

        Only active for the refinement branch: returns ``None`` immediately when
        :attr:`condition_resolver` is unset, ``corner_case`` is ``None``, or ``current``
        is ``UNSET``. When active, queries the backward-inference index for both
        ``target`` and ``current`` knowledge and delegates to :attr:`condition_resolver`.

        :param case: The new case being fit.
        :param target: The correct conclusion.
        :param current: The wrong conclusion currently returned by the firing rule.
        :param corner_case: The case that triggered the currently-firing rule's creation.
        :param firing_anchor: The condition expression of the rule that fired; forwarded
            to the resolver for efficient active-path identification.
        :return: The full :class:`~krrood.entity_query_language.rdr.condition_resolver.ResolvedCondition`
            (expression + resolver provenance), or ``None`` to fall back to the expert.
        """
        if self.condition_resolver is None or corner_case is None or current is UNSET:
            return None
        target_knowledge = self._backward_index.query(self.conditions_root, target)
        current_knowledge = self._backward_index.query(self.conditions_root, current)
        return self.condition_resolver.resolve(
            case,
            self.case_variable,
            target,
            current,
            corner_case,
            target_knowledge,
            current_knowledge,
            firing_anchor,
        )

    def _apply_resolution(
        self,
        resolved: Optional[ResolvedCondition],
        case: Any,
        target: Any,
        current: Any,
        trace: Optional[ClassificationTrace],
        corner_case: Optional[Any],
        expert: Expert,
    ) -> SymbolicExpression:
        """Apply the auto-resolver outcome according to :attr:`resolution_mode`.

        In :attr:`~krrood.entity_query_language.rdr.condition_resolver.ResolutionMode.SILENT`
        mode (default), a resolved condition is inserted directly without prompting.  In
        :attr:`~krrood.entity_query_language.rdr.condition_resolver.ResolutionMode.HINT`
        mode, the full :class:`~krrood.entity_query_language.rdr.condition_resolver.ResolvedCondition`
        is passed as a suggestion so the expert can accept or overwrite it.  When
        ``resolved`` is ``None`` the expert is always consulted regardless of mode.

        :param resolved: The full auto-resolved condition (expression + resolver provenance),
            or ``None`` if the resolver found nothing.
        :param case: The case being fit.
        :param target: The correct conclusion.
        :param current: The wrong conclusion currently returned.
        :param trace: Classification trace.
        :param corner_case: Corner case of the firing rule.
        :param expert: The expert that supplies conditions.
        :return: The EQL condition expression to insert.
        """
        if resolved is not None and self.resolution_mode is ResolutionMode.SILENT:
            return resolved.expression
        suggestion = resolved if self.resolution_mode is ResolutionMode.HINT else None
        return expert.ask_for_conditions(
            case,
            self.case_variable,
            target,
            current,
            trace,
            corner_case,
            suggestion=suggestion,
        )

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
        every case is correct, ``max_passes`` is exhausted, or oscillation is detected —
        whichever comes first.  Oscillation is signalled with a :class:`RDRConvergenceWarning`.

        Correctly-classified cases on re-passes are idempotent (the expert is never called
        for them), so the overhead is one :meth:`classify` call per case per pass —
        negligible compared to expert interaction.

        When ``targets`` is ``None`` the no-target path has no ground truth to converge
        against, so each case is fitted exactly once with no cycle detection.

        :param cases: The case instances to fit.
        :param targets: Optional ground-truth conclusions paired with ``cases``.
            ``None`` triggers the expert-labeling (``ask_for_rule``) path.
        :param expert: The expert that supplies rule conditions.
        :param max_passes: Maximum number of convergent passes.  Defaults to 10.
        :return: This RDR, for chaining.
        """
        paired_targets = targets if targets is not None else [UNSET] * len(cases)
        pending = list(range(len(cases)))

        if expert is not None and self.save_path is not None and expert.interface.on_save is None:
            expert.interface.on_save = lambda: save_rdr_with_case(self, self.save_path)

        progress: Optional[ProgressReporter] = None
        if expert is not None:
            progress = expert.interface.make_progress_reporter()
        if progress is not None:
            progress.start(len(pending), _FITTING_DESCRIPTION)

        try:
            if targets is None:
                for i in pending:
                    self.fit_case(cases[i], paired_targets[i], expert)
                    if progress is not None:
                        progress.update()
            else:
                self._run_convergence(
                    cases, paired_targets, pending, expert, progress, max_passes
                )
        finally:
            if progress is not None:
                progress.finish()

        return self

    def _run_convergence(
        self,
        cases: List[Any],
        paired_targets: List[Any],
        pending: List[int],
        expert: Optional[Expert],
        progress: Optional[ProgressReporter],
        max_passes: int,
    ) -> None:
        """Run the convergent fitting loop until all cases are correct or a cycle is detected.

        After each pass, recomputes the misclassified-case set and checks whether its
        index signature (a ``frozenset``) has appeared in any previous pass.  A repeated
        signature means the tree is oscillating rather than converging; the loop stops
        early, saves the model if :attr:`save_path` is set, and emits a
        :class:`RDRConvergenceWarning` naming the clashing cases.

        :param cases: All case instances (full list, not just pending).
        :param paired_targets: Target conclusions paired with ``cases``.
        :param pending: Indices of currently misclassified cases for the first pass.
        :param expert: The expert supplying conditions.
        :param progress: Optional progress reporter (already started by the caller).
        :param max_passes: Maximum number of convergence passes.
        """
        seen_signatures: List[frozenset] = []

        for pass_num in range(max_passes):
            if pass_num > 0 and progress is not None:
                progress.reset(len(pending))

            for i in pending:
                self.fit_case(cases[i], paired_targets[i], expert)
                if progress is not None:
                    progress.update()

            pending = [
                i
                for i in range(len(cases))
                if paired_targets[i] is not UNSET
                and self.classify(cases[i]) != paired_targets[i]
            ]

            if not pending:
                return

            signature = frozenset(pending)
            if signature in seen_signatures:
                if self.save_path is not None:
                    save_rdr_with_case(self, self.save_path)
                clashing = ", ".join(repr(cases[i]) for i in pending)
                save_hint = (
                    f" The partially fitted model has been saved to {self.save_path!r}."
                    if self.save_path is not None
                    else ""
                )
                warnings.warn(
                    f"RDR fitting detected oscillation and stopped after {pass_num + 1} "
                    f"pass(es). Clashing cases: {clashing}.{save_hint}",
                    RDRConvergenceWarning,
                    stacklevel=3,
                )
                return

            seen_signatures.append(signature)

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
