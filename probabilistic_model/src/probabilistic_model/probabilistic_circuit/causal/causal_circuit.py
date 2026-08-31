from __future__ import annotations

import copy
import enum
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple, Union

from anytree import NodeMixin, PreOrderIter, findall
from scipy.special import logsumexp
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent, Event
from random_events.sigma_algebra import AbstractCompositeSet, AbstractSimpleSet
from random_events.variable import Variable
from tabulate import tabulate


from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)


from probabilistic_model.probabilistic_circuit.causal.exceptions import (
    SupportDeterminismViolation,
    MissingQueryVariableViolation,
    UnnormalizedSumUnitViolation,
    OverlappingChildSupportsViolation,
    SupportDeterminismVerificationResult,
    UnregisteredVariableError,
    EmptyInterventionalCircuitError,
    NoCauseVariablesError,
)


@dataclass
class MarginalDeterminismTreeNode(NodeMixin):
    """
    One node of a Marginal Determinism Variable Tree.

    Each node carries the Variable objects in its subtree and a query_set specifying
    which Variables SumUnits at this level must be support-deterministic over. Support
    determinism enables polytime backdoor adjustment.

    Inherits from anytree.NodeMixin, which provides parent / children management,
    is_leaf and is_root properties, and PreOrderIter / findall traversal utilities.

    Build using MarginalDeterminismTreeNode.from_causal_graph() rather than constructing
    nodes manually.

    :param variables: All Variables in this node's subtree.
    :param query_set: Variables for which SumUnits at this level must be support-
        deterministic. Defaults to an empty set.
    :param parent_node: Parent node in the tree. None for the root.
    """

    variables: Set[Variable]
    """All Variables in this node's subtree."""

    query_set: Set[Variable] = field(default=None)
    """
    Variables for which SumUnits at this level must be support-deterministic.

    Defaults to an empty set.
    """

    parent_node: Optional[MarginalDeterminismTreeNode] = field(default=None)
    """
    Parent node supplied at construction.

    Consumed by __post_init__ to wire the tree. Named parent_node so it does not shadow
    NodeMixin's parent descriptor.
    """

    def __post_init__(self) -> None:
        NodeMixin.__init__(self)
        if self.query_set is None:
            self.query_set = set()
        self.parent = self.parent_node

    def find_node_for_variable(
        self, variable: Variable
    ) -> Optional[MarginalDeterminismTreeNode]:
        """
        Return the shallowest node whose query_set contains variable, or None.

        :param variable: The Variable to search for.
        :returns: The shallowest matching node, or None if not found.
        """
        results = findall(
            self,
            filter_=lambda node: variable in node.query_set,
            maxlevel=None,
        )
        return results[0] if results else None

    def all_query_sets(self) -> List[Set[Variable]]:
        """
        Return all non-empty query_sets in pre-order (depth-first) order.

        :returns: List of non-empty query_sets in pre-order traversal sequence.
        """
        return [node.query_set for node in PreOrderIter(self) if node.query_set]

    @classmethod
    def from_causal_graph(
        cls,
        causal_variables: List[Variable],
        effect_variables: List[Variable],
        causal_priority_order: List[Variable] = None,
    ) -> MarginalDeterminismTreeNode:
        """
        Build a Marginal Determinism Variable Tree from a causal graph specification.

        :param causal_variables: All input Variables that causally affect the outcome.
        :param effect_variables: All outcome Variables.
        :param causal_priority_order: Ordering of cause Variables from most to least
            important. Defaults to causal_variables order if None.
        """
        ordered = (
            causal_priority_order
            if causal_priority_order is not None
            else causal_variables
        )
        return cls._build_subtree(ordered, parent=None)

    @classmethod
    def _build_subtree(
        cls,
        ordered: List[Variable],
        parent: Optional[MarginalDeterminismTreeNode],
    ) -> MarginalDeterminismTreeNode:
        """
        Recursively build a subtree from an ordered list of Variables.

        The first Variable becomes the query_set at this level; the remainder are split
        evenly between left and right children, with the primary Variable repeated in
        the left child to propagate its determinism constraint.

        :param ordered: Variables in priority order, highest priority first.
        :param parent: Parent node to attach this subtree to. None for the root.
        :returns: The root node of the constructed subtree.
        """
        if len(ordered) == 0:
            return cls(variables=set(), query_set=set(), parent_node=parent)
        if len(ordered) == 1:
            return cls(
                variables={ordered[0]}, query_set={ordered[0]}, parent_node=parent
            )

        primary = ordered[0]
        remaining = ordered[1:]
        split = len(remaining) // 2
        left_vars = [primary] + remaining[:split]
        right_vars = remaining[split:]

        node = cls(variables=set(ordered), query_set={primary}, parent_node=parent)
        cls._build_subtree(left_vars, parent=node)
        if right_vars:
            cls._build_subtree(right_vars, parent=node)

        return node


@dataclass
class FailureDiagnosisResult:
    """
    Result of identifying which cause Variable is most responsible for an observed
    outcome falling outside the training distribution.

    interventional_probability_at_failure is P(cause in training support at the observed
    value), evaluated in the joint (cause, effect) interventional circuit. Zero means
    the observed value lies entirely outside the training distribution — the most
    unambiguous signal that this Variable is the primary cause of the unexpected
    outcome.
    """

    primary_cause_variable: Variable
    """
    The cause Variable with the lowest interventional probability at its observed value.
    """

    actual_value: float
    """
    The observed value of the primary cause Variable.
    """

    interventional_probability_at_failure: float
    """
    P(cause in training support at observed value) from the interventional circuit.

    Zero indicates the observed value lies entirely outside the training distribution.
    """

    recommended_region: Optional[Event]
    """
    The cause support region with the highest interventional probability, as a composite
    set event.

    None if no regions were found. The caller can derive a point recommendation from
    region.simple_sets[0][cause_variable].
    """

    interventional_probability_at_recommendation: float
    """
    Interventional probability of the recommended region.
    """

    all_variable_results: Dict[Variable, Dict[str, Any]]
    """
    Per-variable diagnosis results keyed by Variable.

    Each entry contains actual_value, interventional_probability, and
    recommended_region.
    """

    def __str__(self) -> str:
        """
        Format the diagnosis as a two-section plain-text table.

        The header section shows the primary cause variable, its observed value, the
        interventional probability at that value, the recommended region as a [lower,
        upper] interval, and the interventional probability at the recommendation. The
        all-variables section lists every diagnosed cause variable with its observed
        value and interventional probability, marking the primary cause.

        :returns: Multi-line tabulated string suitable for console output.
        """
        if self.recommended_region is not None:
            simple_set = self.recommended_region.simple_sets[0]
            interval_set = simple_set[self.primary_cause_variable]
            interval = (
                interval_set.simple_sets[0]
                if hasattr(interval_set, "simple_sets")
                else interval_set
            )
            region_str = f"[{interval.lower:.4f}, {interval.upper:.4f}]"
        else:
            region_str = "None"
        header = [
            ["Primary cause", self.primary_cause_variable.name],
            ["Actual value", f"{self.actual_value:.4f}"],
            ["P(outcome | do)", f"{self.interventional_probability_at_failure:.4f}"],
            ["Recommended region", region_str],
            [
                "P(outcome | rec)",
                f"{self.interventional_probability_at_recommendation:.4f}",
            ],
        ]

        rows = [
            [
                variable.name
                + (" <- PRIMARY" if variable == self.primary_cause_variable else ""),
                f"{result['actual_value']:.4f}",
                f"{result['interventional_probability']:.4f}",
            ]
            for variable, result in self.all_variable_results.items()
        ]

        return (
            "Failure Diagnosis\n"
            + tabulate(header, tablefmt="simple")
            + "\n\n"
            + "All variables:\n"
            + tabulate(rows, headers=["Variable", "Actual", "P(do)"], tablefmt="simple")
        )


@dataclass
class SupportRegion:
    """
    One support region of a variable (or several jointly), with its probability under
    the circuit it was extracted from.
    """

    event: Event
    """
    The region, as a composite-set event over the variable(s) it was extracted for.
    """

    probability: float
    """
    The region's probability under the circuit it was extracted from.
    """


@dataclass
class CausalCircuit:
    """
    A ProbabilisticCircuit extended with exact, tractable causal inference using the
    marginal determinism framework. The Marginal Determinism Variable Tree structure
    encodes the causal graph and enables polytime backdoor adjustment for any valid
    adjustment set Z.

    Wraps a fitted ProbabilisticCircuit and adds:
      - backdoor_adjustment()        — P(effect | do(cause)) as a new circuit
      - verify_support_determinism() — structural validity check against the tree
      - diagnose_failure()           — identify the most anomalous cause Variable

    Use empty adjustment sets for independent randomised training data.
    Supply confounder Variables for correlated deployment data.
    """

    probabilistic_circuit: ProbabilisticCircuit
    """
    The fitted circuit that encodes the joint distribution over all variables.
    """

    marginal_determinism_tree: MarginalDeterminismTreeNode
    """
    Tree encoding which Variables SumUnits must be support-deterministic over.
    """

    causal_variables: List[Variable]
    """
    Ordered list of cause Variables registered at construction time.
    """

    effect_variables: List[Variable]
    """
    Ordered list of effect Variables registered at construction time.
    """

    def __post_init__(self) -> None:
        self.causal_variables = list(self.causal_variables)
        self.effect_variables = list(self.effect_variables)

    @classmethod
    def from_probabilistic_circuit(
        cls,
        circuit: ProbabilisticCircuit,
        marginal_determinism_tree: MarginalDeterminismTreeNode,
        causal_variables: List[Variable],
        effect_variables: List[Variable],
    ) -> CausalCircuit:
        """
        Construct a CausalCircuit from an existing ProbabilisticCircuit without
        retraining.

        :param circuit: A fitted ProbabilisticCircuit over all causal and effect
            variables.
        :param marginal_determinism_tree: Tree encoding support-determinism constraints.
        :param causal_variables: Ordered list of cause Variables.
        :param effect_variables: Ordered list of effect Variables.
        :returns: A new CausalCircuit wrapping the provided circuit and tree.
        """
        return cls(
            circuit, marginal_determinism_tree, causal_variables, effect_variables
        )

    def _check_query_variables_exist(
        self, all_query_variables: Set[Variable]
    ) -> List[MissingQueryVariableViolation]:
        """
        Check that every Variable in every query_set exists in the circuit.

        :param all_query_variables: Union of all Variables across all query_sets.
        :returns: List of violations, empty if all Variables are present.
        """
        circuit_variables: Set[Variable] = set(self.probabilistic_circuit.variables)
        missing_variables = [
            v for v in all_query_variables if v not in circuit_variables
        ]
        if not missing_variables:
            return []
        return [
            MissingQueryVariableViolation(
                missing_variables=missing_variables,
                available_variables=list(self.probabilistic_circuit.variables),
            )
        ]

    def _check_sum_units_normalized(self) -> List[UnnormalizedSumUnitViolation]:
        """
        Check that every SumUnit's log-weights sum to log(1).

        :returns: List of violations, empty if all SumUnits are normalised.
        """
        violations: List[UnnormalizedSumUnitViolation] = []
        for node in self.probabilistic_circuit.nodes():
            if isinstance(node, SumUnit) and not node.is_normalized():
                violations.append(
                    UnnormalizedSumUnitViolation(
                        sum_unit_index=node.index,
                        actual_log_weight_sum=float(logsumexp(node.log_weights)),
                    )
                )
        return violations

    @staticmethod
    def _child_marginals_split_on_variable(
        child_marginals: List[Any],
    ) -> bool:
        """
        Return True if at least one pair of child marginals is disjoint.

        A SumUnit is treated as a split node for a query Variable only when at least one
        pair of its children has disjoint support on that Variable. SumUnits whose
        children all share the same marginal (e.g. a sibling SumUnit in a ProductUnit
        that has no relationship to this variable) are not split nodes and must be
        skipped to avoid false positives.

        :param child_marginals: Marginal support events, one per SumUnit child.
        :returns: True if any pair of marginals is disjoint.
        """
        return any(
            child_marginals[i].intersection_with(child_marginals[j]).is_empty()
            for i, j in itertools.combinations(range(len(child_marginals)), 2)
        )

    @staticmethod
    def _overlapping_pair_exists(child_marginals: List[Any]) -> bool:
        """
        Return True if any pair of child marginals has non-empty intersection.

        :param child_marginals: Marginal support events, one per SumUnit child.
        :returns: True if any pair overlaps.
        """
        return any(
            not child_marginals[i].intersection_with(child_marginals[j]).is_empty()
            for i, j in itertools.combinations(range(len(child_marginals)), 2)
        )

    def _check_sum_unit_for_variable(
        self,
        node: SumUnit,
        child_support_events: List[Any],
        query_variable: Variable,
    ) -> Optional[OverlappingChildSupportsViolation]:
        """
        Check a single SumUnit against a single query Variable.

        Returns a violation if the SumUnit splits on the variable but has at least one
        overlapping child pair. Returns None if the SumUnit does not split on the
        variable, or if all splitting children are pairwise disjoint.

        :param node: The SumUnit to inspect.
        :param child_support_events: result_of_current_query from each child.
        :param query_variable: The query Variable to check marginal disjointness on.
        :returns: A violation if overlapping children are detected, else None.
        """
        if not all(query_variable in event.variables for event in child_support_events):
            return None
        child_marginals = [
            event.marginal([query_variable]) for event in child_support_events
        ]
        if not self._child_marginals_split_on_variable(child_marginals):
            return None
        if self._overlapping_pair_exists(child_marginals):
            return OverlappingChildSupportsViolation(
                sum_unit_index=node.index,
                query_variable=query_variable,
            )
        return None

    def _check_support_disjointness(
        self, all_query_variables: Set[Variable]
    ) -> List[OverlappingChildSupportsViolation]:
        """
        Check that for each declared query Variable, no SumUnit that splits on that
        Variable has children with overlapping marginal support.

        Calls self.probabilistic_circuit.support once as a side-effecting traversal that
        populates result_of_current_query on every node bottom-up. The returned event is
        discarded; only the per-node side effect matters. Delegates per-node, per-
        variable inspection to _check_sum_unit_for_variable.

        :param all_query_variables: Union of all Variables across all query_sets.
        :returns: List of violations, empty if all split nodes are support-disjoint.
        """
        violations: List[OverlappingChildSupportsViolation] = []
        _ = self.probabilistic_circuit.support

        for layer in self.probabilistic_circuit.layers:
            for node in layer:
                if (
                    not isinstance(node, SumUnit)
                    or len(node.subcircuits) < 2
                    or len(node.variables) == 1
                ):
                    continue

                child_support_events = [
                    child.result_of_current_query for child in node.subcircuits
                ]
                if any(event is None for event in child_support_events):
                    continue

                for query_variable in all_query_variables:
                    violation = self._check_sum_unit_for_variable(
                        node, child_support_events, query_variable
                    )
                    if violation is not None:
                        violations.append(violation)

        return violations

    def verify_support_determinism(self) -> SupportDeterminismVerificationResult:
        """
        Verify support determinism of the circuit against the Marginal Determinism
        Variable Tree. Runs three checks in order — variable existence, SumUnit
        normalization, and pairwise support disjointness for all declared query
        Variables. Returns early after Check 1 if any Variables are missing, since
        Variable objects are required for marginalisation in Check 3.

        :returns: SupportDeterminismVerificationResult when all checks pass.
        :raises SupportDeterminismVerificationResult: Containing all collected
            violations when any check fails. Raising the result rather than a separate
            exception type keeps all diagnostic information — violation list, checked
            query sets, circuit variables — in one object.
        """
        checked_query_sets = self.marginal_determinism_tree.all_query_sets()
        all_query_variables: Set[Variable] = {
            v for qs in checked_query_sets for v in qs
        }

        violations: List[SupportDeterminismViolation] = (
            self._check_query_variables_exist(all_query_variables)
        )
        if violations:
            raise SupportDeterminismVerificationResult(
                passed=False,
                violations=violations,
                checked_query_sets=checked_query_sets,
                circuit_variables=list(self.probabilistic_circuit.variables),
            )

        violations += self._check_sum_units_normalized()
        violations += self._check_support_disjointness(all_query_variables)

        result = SupportDeterminismVerificationResult(
            passed=len(violations) == 0,
            violations=violations,
            checked_query_sets=checked_query_sets,
            circuit_variables=list(self.probabilistic_circuit.variables),
        )
        if violations:
            raise result
        return result

    def backdoor_adjustment(
        self,
        cause_variable: Variable,
        effect_variable: Variable,
        adjustment_variables: List[Variable] = None,
        query_resolution: float = 0.005,
    ) -> ProbabilisticCircuit:
        """
        Compute P(effect | do(cause)) as a new ProbabilisticCircuit.

        With empty adjustment set:
            P(effect | do(cause=v)) = P(effect | cause=v)

        With non-empty adjustment set Z:
            P(effect | do(cause=v)) = sum_z P(effect | cause=v, Z=z) * P(Z=z)

        The output encodes the joint P(effect, cause) — probability(),
        marginal(), and sample() all work on the returned circuit.

        :param cause_variable: A registered cause Variable.
        :param effect_variable: A registered effect Variable.
        :param adjustment_variables: Variables to adjust for. Empty list performs
            no adjustment. Defaults to None, treated as empty.
        :param query_resolution: Half-width of the interval used when querying
            point probabilities. Defaults to 0.005.
        :returns: Joint interventional circuit over (cause, effect).
        """
        if cause_variable not in self.causal_variables:
            raise UnregisteredVariableError(
                variable_name=cause_variable.name,
                registered_names=[v.name for v in self.causal_variables],
                role="cause",
            )
        if effect_variable not in self.effect_variables:
            raise UnregisteredVariableError(
                variable_name=effect_variable.name,
                registered_names=[v.name for v in self.effect_variables],
                role="effect",
            )
        if adjustment_variables is None:
            adjustment_variables = []

        if not adjustment_variables:
            return self._compute_interventional_circuit_without_adjustment(
                cause_variable, effect_variable, query_resolution
            )
        return self._compute_interventional_circuit_with_adjustment(
            cause_variable,
            effect_variable,
            adjustment_variables,
            query_resolution,
        )

    def _build_product_unit_for_region(
        self,
        cause_event: Any,
        cause_weight: float,
        effect_variable: Variable,
        cause_marginal_circuit: ProbabilisticCircuit,
        conditioned_circuit: ProbabilisticCircuit,
        output_circuit: ProbabilisticCircuit,
        root_sum_unit: SumUnit,
    ) -> bool:
        """
        Build one ProductUnit for a single cause region and attach it to root_sum_unit.

        Truncates conditioned_circuit to the cause region, extracts the effect marginal,
        truncates cause_marginal_circuit to the cause region, then assembles a
        ProductUnit of (cause branch, effect branch) weighted by cause_weight.

        :param cause_event: Composite event defining the cause support region.
        :param cause_weight: Probability mass of this cause region.
        :param effect_variable: The effect Variable to marginalise onto.
        :param cause_marginal_circuit: Marginal circuit over the cause Variable.
        :param conditioned_circuit: Full circuit to truncate to the cause region.
        :param output_circuit: Circuit to attach the new ProductUnit to.
        :param root_sum_unit: SumUnit to attach the weighted ProductUnit to.
        :returns: True if the ProductUnit was successfully added, False if cause
            truncation produced an empty circuit.
        """
        truncated_circuit, _ = copy.deepcopy(
            conditioned_circuit
        ).log_truncated_in_place(
            cause_event.fill_missing_variables_pure(conditioned_circuit.variables)
        )
        if truncated_circuit is None:
            return False

        effect_marginal_circuit = truncated_circuit.marginal([effect_variable])
        cause_region_circuit, _ = copy.deepcopy(
            cause_marginal_circuit
        ).log_truncated_in_place(
            cause_event.fill_missing_variables_pure(cause_marginal_circuit.variables)
        )
        if cause_region_circuit is None:
            return False

        product_unit = ProductUnit(probabilistic_circuit=output_circuit)
        product_unit.attach_marginal_circuit(cause_region_circuit, output_circuit)
        product_unit.attach_marginal_circuit(effect_marginal_circuit, output_circuit)
        root_sum_unit.add_subcircuit(product_unit, math.log(cause_weight))
        return True

    def _compute_interventional_circuit_without_adjustment(
        self,
        cause_variable: Variable,
        effect_variable: Variable,
        query_resolution: float,
    ) -> ProbabilisticCircuit:
        """
        Compute P(cause, effect | do(cause)) with an empty adjustment set.

        Returns a joint circuit over (cause, effect) as a SumUnit of
        ProductUnits, one per disjoint cause support region.

        Structure::

            SumUnit [weight = P(cause in region_i)]
                ProductUnit
                    cause branch  (UniformDistribution over region_i)
                    effect branch (P(effect | cause in region_i))

        :param cause_variable: The cause Variable to intervene on.
        :param effect_variable: The effect Variable to compute the distribution over.
        :param query_resolution: Half-width passed through to region extraction.
        :returns: Joint interventional circuit over (cause, effect).
        """
        cause_marginal_circuit = copy.deepcopy(self.probabilistic_circuit).marginal(
            [cause_variable]
        )
        output_circuit = ProbabilisticCircuit()
        root_sum_unit = SumUnit(probabilistic_circuit=output_circuit)
        regions_added = sum(
            self._build_product_unit_for_region(
                cause_event=region.event,
                cause_weight=region.probability,
                effect_variable=effect_variable,
                cause_marginal_circuit=cause_marginal_circuit,
                conditioned_circuit=copy.deepcopy(self.probabilistic_circuit),
                output_circuit=output_circuit,
                root_sum_unit=root_sum_unit,
            )
            for region in self._extract_disjoint_regions_for_variable(cause_variable)
            if region.probability > 0.0
        )

        if regions_added == 0:
            raise EmptyInterventionalCircuitError(
                cause_variable_name=cause_variable.name,
                adjustment_variable_names=[],
            )
        return output_circuit

    @staticmethod
    def _attach_weighted_marginal(
        sum_unit: SumUnit,
        marginal_circuit: ProbabilisticCircuit,
        log_weight: float,
        target_circuit: ProbabilisticCircuit,
    ) -> None:
        """
        Attach marginal_circuit's root as a weighted child of sum_unit, constructing
        fresh nodes owned by target_circuit.

        The :class:`ProductUnit` counterpart of this,
        ``ProductUnit.attach_marginal_circuit``, attaches its child unweighted (a
        product's factors carry no individual weight); this instead gives the new child
        *log_weight*, since a SumUnit's children each need one.

        marginal() and log_truncated_in_place() return flat circuits (SumUnit -> leaves,
        or a single leaf), so one level of recursion suffices to copy all nodes into
        target_circuit.

        :param sum_unit: The SumUnit to attach the new weighted child to.
        :param marginal_circuit: The marginal or truncated circuit whose root to attach.
        :param log_weight: The log-weight the new child is attached with.
        :param target_circuit: The owning circuit for all newly created nodes.
        """
        root = marginal_circuit.root
        if isinstance(root, SumUnit):
            new_sum_unit = SumUnit(probabilistic_circuit=target_circuit)
            for child_log_weight, child_subcircuit in root.log_weighted_subcircuits:
                new_sum_unit.add_subcircuit(
                    leaf(copy.deepcopy(child_subcircuit.distribution), target_circuit),
                    child_log_weight,
                )
            sum_unit.add_subcircuit(new_sum_unit, log_weight)
        else:
            sum_unit.add_subcircuit(
                leaf(copy.deepcopy(root.distribution), target_circuit), log_weight
            )

    def _add_region_for_cause_value(
        self,
        cause_region: SupportRegion,
        adjustment_partitions: List[SupportRegion],
        effect_variable: Variable,
        cause_marginal_circuit: ProbabilisticCircuit,
        output_circuit: ProbabilisticCircuit,
        root_sum_unit: SumUnit,
    ) -> bool:
        """
        Build one ProductUnit for cause_region and attach it to root_sum_unit, weighted
        by cause_region's own marginal probability.

        The ProductUnit's effect side is ``sum_z P(effect | cause=v, Z=z) * P(Z=z)``: a
        SumUnit over each adjustment partition's own conditional effect distribution,
        weighted by that partition's probability and renormalized (some adjustment
        partitions may have zero joint probability with this cause region and be
        skipped, so the weights that remain do not already sum to 1). Weighting the
        ProductUnit itself by P(cause=v) -- not by any per-adjustment-partition weight
        -- is what keeps the circuit a valid joint distribution overall (weights across
        every cause region still sum to 1) while truncating to this one region recovers
        exactly the deconfounded mixture, regardless of that region's own absolute
        weight.

        :param cause_region: The cause value/region to build this ProductUnit for.
        :param adjustment_partitions: One :class:`SupportRegion` per distinct value
            combination of the adjustment variables (``adjustment_variables`` in
            :meth:`_compute_interventional_circuit_with_adjustment`) -- together they
            partition the population into disjoint, exhaustive groups.
        :param effect_variable: The effect Variable to marginalise onto.
        :param cause_marginal_circuit: Marginal circuit over the cause Variable.
        :param output_circuit: Circuit to attach the new ProductUnit to.
        :param root_sum_unit: SumUnit to attach the weighted ProductUnit to.
        :returns: True if the ProductUnit was successfully added, False if every
            adjustment partition's joint truncation with this cause region was empty.
        """
        effect_mixture = SumUnit(probabilistic_circuit=output_circuit)
        for adjustment_partition in adjustment_partitions:
            joint_event = adjustment_partition.event.intersection_with(
                cause_region.event
            )
            joint_conditioned_circuit, _ = copy.deepcopy(
                self.probabilistic_circuit
            ).log_truncated_in_place(
                joint_event.fill_missing_variables_pure(
                    self.probabilistic_circuit.variables
                )
            )
            if joint_conditioned_circuit is None:
                continue
            self._attach_weighted_marginal(
                effect_mixture,
                joint_conditioned_circuit.marginal([effect_variable]),
                math.log(adjustment_partition.probability),
                output_circuit,
            )

        if len(effect_mixture.log_weights) == 0:
            return False
        effect_mixture.normalize()

        cause_region_circuit, _ = copy.deepcopy(
            cause_marginal_circuit
        ).log_truncated_in_place(
            cause_region.event.fill_missing_variables_pure(
                cause_marginal_circuit.variables
            )
        )
        if cause_region_circuit is None:
            return False

        product_unit = ProductUnit(probabilistic_circuit=output_circuit)
        product_unit.attach_marginal_circuit(cause_region_circuit, output_circuit)
        # effect_mixture is already a node of output_circuit (built directly above,
        # not copied in from an external circuit), so it attaches as a plain child
        # rather than through attach_marginal_circuit.
        product_unit.add_subcircuit(effect_mixture)
        root_sum_unit.add_subcircuit(product_unit, math.log(cause_region.probability))
        return True

    def _compute_interventional_circuit_with_adjustment(
        self,
        cause_variable: Variable,
        effect_variable: Variable,
        adjustment_variables: List[Variable],
        query_resolution: float,
    ) -> ProbabilisticCircuit:
        """
        Compute P(cause, effect | do(cause)) with a non-empty adjustment set Z.

        Implements:
            P(effect | do(cause=v)) = sum_z P(effect | cause=v, Z=z) * P(Z=z)

        :param cause_variable: The cause Variable to intervene on.
        :param effect_variable: The effect Variable to compute the distribution over.
        :param adjustment_variables: Variables to condition on and marginalise over.
        :param query_resolution: Half-width passed through to region extraction.
        :returns: Joint interventional circuit over (cause, effect).
        """
        cause_marginal_circuit = copy.deepcopy(self.probabilistic_circuit).marginal(
            [cause_variable]
        )
        adjustment_partitions = [
            adjustment_partition
            for adjustment_partition in self._extract_leaf_regions_for_variables(
                adjustment_variables
            )
            if adjustment_partition.probability > 0.0
        ]
        output_circuit = ProbabilisticCircuit()
        root_sum_unit = SumUnit(probabilistic_circuit=output_circuit)
        regions_added = sum(
            self._add_region_for_cause_value(
                cause_region=cause_region,
                adjustment_partitions=adjustment_partitions,
                effect_variable=effect_variable,
                cause_marginal_circuit=cause_marginal_circuit,
                output_circuit=output_circuit,
                root_sum_unit=root_sum_unit,
            )
            for cause_region in self._extract_disjoint_regions_for_variable(
                cause_variable
            )
            if cause_region.probability > 0.0
        )

        if regions_added == 0:
            raise EmptyInterventionalCircuitError(
                cause_variable_name=cause_variable.name,
                adjustment_variable_names=[v.name for v in adjustment_variables],
            )
        return output_circuit

    def _extract_leaf_regions_for_variable(
        self,
        variable: Variable,
        base_circuit: ProbabilisticCircuit = None,
    ) -> List[SupportRegion]:
        """
        Return a :class:`SupportRegion` for each support region of variable.

        :param variable: The Variable whose support regions to extract.
        :param base_circuit: Circuit to query. Defaults to self.probabilistic_circuit.
        :returns: List of regions, one per region.
        """
        circuit = (
            base_circuit if base_circuit is not None else self.probabilistic_circuit
        )
        regions: List[SupportRegion] = []
        variable_support = circuit.support.marginal([variable])
        for simple_region in variable_support.simple_sets:
            region_event = SimpleEvent.from_data(
                {variable: simple_region[variable]}
            ).as_composite_set()
            probability = circuit.probability(
                region_event.fill_missing_variables_pure(circuit.variables)
            )
            if probability > 0.0:
                regions.append(SupportRegion(region_event, float(probability)))
        return regions

    def _extract_disjoint_regions_for_variable(
        self,
        variable: Variable,
        base_circuit: ProbabilisticCircuit = None,
    ) -> List[SupportRegion]:
        """
        Return a :class:`SupportRegion` for each structurally disjoint support region of
        variable, read from the circuit's unmarginalized joint support.

        `_extract_leaf_regions_for_variable` marginalizes the joint support down to
        variable alone first, which coalesces every disjoint per-branch range into one
        Interval/Set-valued region -- correct as a description of variable's own
        support, but unable to tell one SumUnit branch's region apart from another's.
        This matters beyond region-search cosmetics: `backdoor_adjustment` builds its
        output as one ProductUnit per region returned here, pairing each region's own
        cause branch with its own effect branch: with the coalesced regions, a
        multi-branch cause variable collapsed to a single ProductUnit spanning its
        whole domain, silently discarding the cause-effect correlation each branch
        encodes (marginal probabilities stayed correct either way, which is why this
        went unnoticed by tests that only checked marginals). This method instead reads
        variable's value directly off each of the joint support's own disjoint simple
        sets, before marginalization discards which branch it came from, merging
        branches that happen to share the same value by keeping a single entry
        (`circuit.probability` already aggregates every contributing branch for that
        value in one call, so no separate summation is needed). Query-set variables are
        exactly where this separation is meaningful: support determinism (see
        `verify_support_determinism`) guarantees SumUnit children have disjoint support
        on them, so the regions returned here correspond to actual circuit branches
        rather than an arbitrary decomposition.

        Query variables with a discrete (:class:`~random_events.set.Set`) domain need
        one further step: a single SumUnit branch can itself be a mixture over several
        of the variable's values (unlike a continuous branch, whose own support is
        already the atomic range a leaf distribution occupies), so
        ``simple_region[variable]`` there is the *union* of every value with positive
        probability in that branch, not one value. `_split_into_atomic_values` breaks
        that union apart before grouping, so e.g. a branch giving 80% probability to
        ``HIGH`` and 20% to ``LOW`` contributes two regions, not one spanning both --
        the same distinction this method already makes between branches applies within
        one branch's own mixture.

        :param variable: The Variable whose disjoint support regions to extract.
        :param base_circuit: Circuit to query. Defaults to self.probabilistic_circuit.
        :returns: List of regions, one per disjoint region, each with positive
            probability.
        """
        circuit = (
            base_circuit if base_circuit is not None else self.probabilistic_circuit
        )
        regions_by_value: Dict[Any, SupportRegion] = {}
        for simple_region in circuit.support.simple_sets:
            for value in self._split_into_atomic_values(simple_region[variable]):
                if value in regions_by_value:
                    continue
                region_event = SimpleEvent.from_data(
                    {variable: value}
                ).as_composite_set()
                probability = circuit.probability(
                    region_event.fill_missing_variables_pure(circuit.variables)
                )
                if probability > 0.0:
                    regions_by_value[value] = SupportRegion(
                        region_event, float(probability)
                    )
        return list(regions_by_value.values())

    @staticmethod
    def _split_into_atomic_values(
        value: Union[AbstractCompositeSet, int, float, bool, enum.Enum],
    ) -> List[
        Union[AbstractCompositeSet, AbstractSimpleSet, int, float, bool, enum.Enum]
    ]:
        """
        Split *value* into its individual elements if it is a union of more than one
        (several disjoint ranges for a :class:`~random_events.interval.Interval` --
        which both :class:`~random_events.variable.Continuous` and
        :class:`~random_events.variable.Integer` use as their domain -- or several
        values for a discrete :class:`~random_events.set.Set`), otherwise return it
        unchanged.

        *value* is only ever composite (and thus splittable) when the branch it came
        from mixes several ranges or values with positive probability; a branch whose
        leaf is a single deterministic point instead yields one of
        :data:`~random_events.variable.compatible_types` directly, which has no
        `simple_sets` to split.

        :param value: A single support value read off a joint support's simple set.
        :returns: The union's elements, or ``[value]`` if *value* is not a multi-element
            union.
        """
        if isinstance(value, AbstractCompositeSet) and len(value.simple_sets) > 1:
            return list(value.simple_sets)
        return [value]

    def _best_disjoint_region(
        self,
        variable: Variable,
        interventional_circuit: ProbabilisticCircuit,
    ) -> Optional[Event]:
        """
        Return the structurally disjoint support region of variable (see
        `_extract_disjoint_regions_for_variable`) with the highest interventional
        probability, or None if no regions are found.

        The disjoint counterpart of `_best_region`: use this when telling one SumUnit
        branch's region apart from another's matters (for example, ranking several
        candidate cause Variables against each other), since `_best_region` cannot --
        its regions come from `_extract_leaf_regions_for_variable`, which always
        collapses to variable's whole support as a single region.

        :param variable: The Variable whose disjoint support regions to search.
        :param interventional_circuit: Joint circuit used to score each region.
        :returns: Composite set event of the highest-probability disjoint region, or
            None.
        """
        best_probability = -1.0
        best_region: Optional[Event] = None
        for region in self._extract_disjoint_regions_for_variable(variable):
            region_probability = float(
                interventional_circuit.probability(
                    region.event.fill_missing_variables_pure(
                        interventional_circuit.variables
                    )
                )
            )
            if region_probability > best_probability:
                best_probability = region_probability
                best_region = region.event
        return best_region

    def _extract_leaf_regions_for_variables(
        self,
        variables: List[Variable],
    ) -> List[SupportRegion]:
        """
        Return a :class:`SupportRegion` for the joint support of variables.

        Decomposes each variable's value the same way
        :meth:`_extract_disjoint_regions_for_variable` does (see
        `_split_into_atomic_values`) before combining them, and takes the Cartesian
        product across *variables* to enumerate every joint combination -- needed
        because a discrete adjustment variable's own marginal support already unions
        every value it takes with positive probability into one entry, the same
        coalescing that method exists to undo for a single variable.

        :param variables: Variables whose joint support regions to extract.
        :returns: List of regions, one per joint region, each with positive probability.
        """
        regions_by_values: Dict[Any, SupportRegion] = {}
        joint_support = self.probabilistic_circuit.support.marginal(variables)
        for simple_region in joint_support.simple_sets:
            atomic_values_per_variable = [
                self._split_into_atomic_values(simple_region[variable])
                for variable in variables
            ]
            for combination in itertools.product(*atomic_values_per_variable):
                if combination in regions_by_values:
                    continue
                region_event = SimpleEvent.from_data(
                    dict(zip(variables, combination))
                ).as_composite_set()
                probability = self.probabilistic_circuit.probability(
                    region_event.fill_missing_variables_pure(
                        self.probabilistic_circuit.variables
                    )
                )
                if probability > 0.0:
                    regions_by_values[combination] = SupportRegion(
                        region_event, float(probability)
                    )
        return list(regions_by_values.values())

    def _query_probability_at_value(
        self,
        interventional_circuit: ProbabilisticCircuit,
        cause_variable: Variable,
        value: float,
        query_resolution: float,
    ) -> float:
        """
        Query P(cause in [value - eps, value + eps]) on the interventional circuit.

        :param interventional_circuit: Joint circuit over (cause, effect).
        :param cause_variable: The Variable whose value to query.
        :param value: The point value to query around.
        :param query_resolution: Half-width of the query interval.
        :returns: Probability mass within the query interval.
        """
        event = SimpleEvent.from_data(
            {cause_variable: closed(value - query_resolution, value + query_resolution)}
        ).as_composite_set()
        return float(
            interventional_circuit.probability(
                event.fill_missing_variables_pure(interventional_circuit.variables)
            )
        )

    def _best_region(
        self,
        cause_variable: Variable,
        interventional_circuit: ProbabilisticCircuit,
    ) -> Optional[Event]:
        """
        Return the cause support region with the highest interventional probability, or
        None if no regions are found.

        The full composite set event is returned so callers retain the interval bounds
        rather than a lossy midpoint summary.

        :param cause_variable: The Variable whose support regions to search.
        :param interventional_circuit: Joint circuit used to score each region.
        :returns: Composite set event of the highest-probability region, or None.
        """
        best_probability = -1.0
        best_region: Optional[Event] = None
        for region in self._extract_leaf_regions_for_variable(cause_variable):
            region_probability = float(
                interventional_circuit.probability(
                    region.event.fill_missing_variables_pure(
                        interventional_circuit.variables
                    )
                )
            )
            if region_probability > best_probability:
                best_probability = region_probability
                best_region = region.event
        return best_region

    def _diagnose_single_cause_variable(
        self,
        cause_variable: Variable,
        observed_value: float,
        effect_variable: Variable,
        adjustment_variables: List[Variable],
        query_resolution: float,
    ) -> Tuple[Dict[str, Any], ProbabilisticCircuit]:
        """
        Run diagnosis for a single cause Variable at its observed value.

        :param cause_variable: The cause Variable to diagnose.
        :param observed_value: The observed value of the cause Variable.
        :param effect_variable: The effect Variable.
        :param adjustment_variables: Variables to adjust for.
        :param query_resolution: Half-width of the query interval.
        :returns: Tuple of (result_dict, interventional_circuit).
        """
        interventional_circuit = self.backdoor_adjustment(
            cause_variable,
            effect_variable,
            adjustment_variables,
            query_resolution,
        )
        probability_at_observed = self._query_probability_at_value(
            interventional_circuit,
            cause_variable,
            observed_value,
            query_resolution,
        )
        recommended_region = self._best_region(cause_variable, interventional_circuit)
        result = {
            "actual_value": observed_value,
            "interventional_probability": round(probability_at_observed, 6),
            "recommended_region": recommended_region,
        }
        return result, interventional_circuit

    def diagnose_failure(
        self,
        observed_values: Dict[Variable, float],
        effect_variable: Variable,
        query_resolution: float = 0.005,
        adjustment_variables: List[Variable] = None,
    ) -> FailureDiagnosisResult:
        """
        Identify the primary cause Variable whose observed value is most anomalous under
        the interventional distribution.

        For each cause Variable, queries the interventional circuit at the observed
        value. The Variable with the lowest interventional probability at its observed
        value is identified as the primary cause. The recommendation is the cause region
        with the highest interventional probability, returned as a full composite set
        event.

        The interventional_probability recorded per Variable is P(cause in [observed-
        eps, observed+eps]) in the joint (cause, effect) interventional circuit. Zero
        means the value is outside training support.

        :param observed_values: Mapping from each cause Variable to its observed value.
        :param effect_variable: The effect Variable to evaluate the interventional
            distribution over.
        :param query_resolution: Half-width of the interval used when evaluating point
            probabilities. Defaults to 0.005.
        :param adjustment_variables: Variables to adjust for. Defaults to None, treated
            as empty.
        :returns: FailureDiagnosisResult identifying the primary cause and recommended
            corrective region.
        """
        if adjustment_variables is None:
            adjustment_variables = []

        all_variable_results: Dict[Variable, Dict[str, Any]] = {}
        interventional_circuits_by_cause: Dict[Variable, ProbabilisticCircuit] = {}

        for cause_variable in self.causal_variables:
            if cause_variable not in observed_values or not cause_variable.is_numeric:
                continue
            result, interventional_circuit = self._diagnose_single_cause_variable(
                cause_variable=cause_variable,
                observed_value=observed_values[cause_variable],
                effect_variable=effect_variable,
                adjustment_variables=adjustment_variables,
                query_resolution=query_resolution,
            )
            all_variable_results[cause_variable] = result
            interventional_circuits_by_cause[cause_variable] = interventional_circuit

        if not all_variable_results:
            raise NoCauseVariablesError(
                registered_cause_names=[v.name for v in self.causal_variables],
            )

        primary_cause_variable = min(
            all_variable_results,
            key=lambda v: all_variable_results[v]["interventional_probability"],
        )
        primary_result = all_variable_results[primary_cause_variable]
        recommended_region = primary_result["recommended_region"]

        if recommended_region is not None:
            interval_set = recommended_region.simple_sets[0][primary_cause_variable]
            interval = (
                interval_set.simple_sets[0]
                if hasattr(interval_set, "simple_sets")
                else interval_set
            )
            region_midpoint = (float(interval.lower) + float(interval.upper)) / 2.0
            probability_at_recommendation = self._query_probability_at_value(
                interventional_circuits_by_cause[primary_cause_variable],
                primary_cause_variable,
                region_midpoint,
                query_resolution,
            )
        else:
            probability_at_recommendation = 0.0

        return FailureDiagnosisResult(
            primary_cause_variable=primary_cause_variable,
            actual_value=primary_result["actual_value"],
            interventional_probability_at_failure=primary_result[
                "interventional_probability"
            ],
            recommended_region=recommended_region,
            interventional_probability_at_recommendation=round(
                probability_at_recommendation, 6
            ),
            all_variable_results=all_variable_results,
        )
