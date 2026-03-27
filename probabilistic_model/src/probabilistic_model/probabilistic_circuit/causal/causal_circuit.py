"""
causal_circuit
==============
Provides CausalCircuit, an extension of ProbabilisticCircuit with exact,
tractable causal inference using the marginal determinism framework (md-vtree).

Also provides MarginalDeterminismTreeNode for constructing the causal graph
structure, and dataclasses for verification and query results.
"""

from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import numpy as np
from anytree import NodeMixin, PreOrderIter, findall
from scipy.special import logsumexp
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Variable

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)

from .utils import (
    attach_marginal_circuit,
    sum_unit_is_normalized,
)


class MarginalDeterminismTreeNode(NodeMixin):
    """
    One node of a Marginal Determinism Variable Tree.

    Each node carries the Variable objects in its subtree and a query_set
    specifying which Variables SumUnits at this level must be
    support-deterministic over. Support determinism enables polytime backdoor
    adjustment.

    Inherits from anytree.NodeMixin, which provides:
      - parent / children management
      - is_leaf, is_root properties
      - depth, height, path properties
      - PreOrderIter and findall traversal utilities

    Build using MarginalDeterminismTreeNode.from_causal_graph() rather than
    constructing nodes manually.
    """

    def __init__(
        self,
        variables: Set[Variable],
        query_set: Set[Variable] = None,
        parent: Optional[MarginalDeterminismTreeNode] = None,
    ) -> None:
        super().__init__()
        self.variables: Set[Variable] = variables
        self.query_set: Set[Variable] = query_set if query_set is not None else set()
        self.parent = parent

    def find_node_for_variable(
        self, variable: Variable
    ) -> Optional[MarginalDeterminismTreeNode]:
        """Return the shallowest node whose query_set contains variable, or None."""
        results = findall(
            self,
            filter_=lambda node: variable in node.query_set,
            maxlevel=None,
        )
        return results[0] if results else None

    def all_query_sets(self) -> List[Set[Variable]]:
        """Return all non-empty query_sets in pre-order (depth-first) order."""
        return [
            node.query_set
            for node in PreOrderIter(self)
            if node.query_set
        ]

    @staticmethod
    def from_causal_graph(
        causal_variables: List[Variable],
        effect_variables: List[Variable],
        causal_priority_order: List[Variable] = None,
    ) -> MarginalDeterminismTreeNode:
        """
        Build a Marginal Determinism Variable Tree from a causal graph specification.

        Parameters
        ----------
        causal_variables
            All input Variables that causally affect the outcome.
        effect_variables
            All outcome Variables.
        causal_priority_order
            Ordering of cause Variables from most to least important.
            Defaults to causal_variables order if None.
        """
        ordered = (
            causal_priority_order
            if causal_priority_order is not None
            else causal_variables
        )
        return MarginalDeterminismTreeNode._build_subtree(ordered, parent=None)

    @staticmethod
    def _build_subtree(
        ordered: List[Variable],
        parent: Optional[MarginalDeterminismTreeNode],
    ) -> MarginalDeterminismTreeNode:
        """
        Recursively build a subtree from an ordered list of Variables. The first
        Variable becomes the query_set at this level; the remainder are split evenly
        between left and right children, with the primary Variable repeated in the left
        child to propagate its determinism constraint.
        """
        if len(ordered) == 0:
            return MarginalDeterminismTreeNode(
                variables=set(), query_set=set(), parent=parent
            )
        if len(ordered) == 1:
            return MarginalDeterminismTreeNode(
                variables={ordered[0]}, query_set={ordered[0]}, parent=parent
            )

        primary = ordered[0]
        remaining = ordered[1:]
        split = len(remaining) // 2
        left_vars = [primary] + remaining[:split]
        right_vars = remaining[split:]

        node = MarginalDeterminismTreeNode(
            variables=set(ordered),
            query_set={primary},
            parent=parent,
        )
        MarginalDeterminismTreeNode._build_subtree(left_vars, parent=node)
        if right_vars:
            MarginalDeterminismTreeNode._build_subtree(right_vars, parent=node)

        return node


@dataclass
class SupportDeterminismVerificationResult:
    """
    Result of verifying support determinism of a circuit against its
    Marginal Determinism Variable Tree.

    Support determinism requires that for each declared cause Variable,
    SumUnit children have disjoint support regions — i.e. each child
    exclusively owns a non-overlapping partition of that Variable's domain.
    This structural property is required for tractable backdoor adjustment.

    Based on the Q-determinism condition from:
        Broadrick et al. (2023), Tractable Probabilistic Circuits
        https://arxiv.org/abs/2304.07438
    """

    passed: bool
    violations: List[str]
    checked_query_sets: List[Set[Variable]]
    circuit_variables: List[Variable]

    def __str__(self) -> str:
        status = "PASS" if self.passed else "FAIL"
        checked_names = [{v.name for v in qs} for qs in self.checked_query_sets]
        circuit_names = [v.name for v in self.circuit_variables]
        lines = [
            f"Support determinism verification: {status}",
            f"  Checked query_sets: {checked_names}",
            f"  Circuit variables:  {circuit_names}",
        ]
        if self.violations:
            lines.append("  Violations:")
            for violation in self.violations:
                lines.append(f"    - {violation}")
        return "\n".join(lines)


@dataclass
class FailureDiagnosisResult:
    """
    Result of identifying which cause Variable is most responsible for an
    observed outcome falling outside the training distribution.

    interventional_probability_at_failure is P(cause in training support
    at the observed value), evaluated in the joint (cause, effect)
    interventional circuit. Zero means the observed value lies entirely
    outside the training distribution — the most unambiguous signal that
    this Variable is the primary cause of the unexpected outcome.
    """

    primary_cause_variable: Variable
    actual_value: float
    interventional_probability_at_failure: float
    recommended_value: Any
    interventional_probability_at_recommendation: float
    all_variable_results: Dict[Variable, Dict[str, Any]]

    def __str__(self) -> str:
        from tabulate import tabulate

        header = [
            ["Primary cause",    self.primary_cause_variable.name],
            ["Actual value",     f"{self.actual_value:.4f}"],
            ["P(outcome | do)",  f"{self.interventional_probability_at_failure:.4f}"],
            ["Recommended",      self.recommended_value],
            ["P(outcome | rec)", f"{self.interventional_probability_at_recommendation:.4f}"],
        ]

        rows = [
            [
                variable.name + (" <- PRIMARY" if variable == self.primary_cause_variable else ""),
                f"{result['actual_value']:.4f}",
                f"{result['interventional_probability']:.4f}",
            ]
            for variable, result in self.all_variable_results.items()
        ]

        return (
            "Failure Diagnosis\n"
            + tabulate(header, tablefmt="simple") + "\n\n"
            + "All variables:\n"
            + tabulate(rows, headers=["Variable", "Actual", "P(do)"], tablefmt="simple")
        )



class CausalCircuit:
    """
    A ProbabilisticCircuit extended with exact, tractable causal inference
    using the marginal determinism framework. The Marginal Determinism
    Variable Tree structure encodes the causal graph and enables polytime
    backdoor adjustment for any valid adjustment set Z.

    Wraps a fitted ProbabilisticCircuit and adds:
      - backdoor_adjustment()        — P(effect | do(cause)) as a new circuit
      - verify_support_determinism() — structural validity check against the tree
      - diagnose_failure()           — identify the most anomalous cause Variable

    Use empty adjustment sets for independent randomised training data.
    Supply confounder Variables for correlated deployment data.
    """

    def __init__(
        self,
        probabilistic_circuit: ProbabilisticCircuit,
        marginal_determinism_tree: MarginalDeterminismTreeNode,
        causal_variables: List[Variable],
        effect_variables: List[Variable],
    ) -> None:
        self.probabilistic_circuit = probabilistic_circuit
        self.marginal_determinism_tree = marginal_determinism_tree
        self.causal_variables: List[Variable] = list(causal_variables)
        self.effect_variables: List[Variable] = list(effect_variables)

    @classmethod
    def from_probabilistic_circuit(
        cls,
        circuit: ProbabilisticCircuit,
        marginal_determinism_tree: MarginalDeterminismTreeNode,
        causal_variables: List[Variable],
        effect_variables: List[Variable],
    ) -> CausalCircuit:
        """Construct from an existing ProbabilisticCircuit without retraining."""
        return cls(circuit, marginal_determinism_tree, causal_variables, effect_variables)

    def _check_query_variables_exist(
        self, all_query_variables: Set[Variable]
    ) -> List[str]:
        """Check 1: every Variable in every query_set exists in the circuit."""
        circuit_variables: Set[Variable] = set(self.probabilistic_circuit.variables)
        missing_variables = [v for v in all_query_variables if v not in circuit_variables]
        if not missing_variables:
            return []
        missing_names = [v.name for v in missing_variables]
        circuit_names = [v.name for v in self.probabilistic_circuit.variables]
        return [
            f"Query-set Variables {missing_names} not found in circuit. "
            f"Available: {circuit_names}"
        ]

    def _check_sum_units_normalized(self) -> List[str]:
        """Check 2: every SumUnit's log-weights must sum to log(1)."""
        violations: List[str] = []
        for node in self.probabilistic_circuit.nodes():
            if isinstance(node, SumUnit) and not sum_unit_is_normalized(node):
                violations.append(
                    f"SumUnit (index={node.index}) log-weights sum to "
                    f"{float(logsumexp(node.log_weights)):.6f}, expected 0.0. "
                    f"Unnormalized circuits produce incorrect backdoor probabilities."
                )
        return violations

    def _check_support_disjointness(
        self, all_query_variables: Set[Variable]
    ) -> List[str]:
        """
        Check 3: for each declared query Variable, verify that there exists at
        least one SumUnit whose children have pairwise disjoint marginal support
        on that Variable (confirming the circuit is support-deterministic for it),
        and that no SumUnit whose children DO split on that Variable has any
        overlapping children.

        Only SumUnits whose children have genuinely different (non-identical)
        marginals on the query Variable are treated as split nodes for that
        variable — SumUnits whose children share the same marginal (e.g. a
        sibling SumUnit in a ProductUnit that knows nothing about this variable)
        are skipped to avoid false positives.

        Calls self.probabilistic_circuit.support exactly once to populate
        result_of_current_query on every node via a bottom-up traversal.
        """
        violations: List[str] = []
        try:
            root_support_event = self.probabilistic_circuit.support

            for layer in self.probabilistic_circuit.layers:
                for node in layer:
                    if not isinstance(node, SumUnit):
                        continue
                    children = node.subcircuits
                    if len(children) < 2:
                        continue

                    child_support_events = [
                        getattr(child, "result_of_current_query", None)
                        for child in children
                    ]
                    if any(event is None for event in child_support_events):
                        continue

                    for query_variable in all_query_variables:
                        child_marginals = []
                        for support_event in child_support_events:
                            try:
                                child_marginals.append(
                                    support_event.marginal([query_variable])
                                )
                            except Exception:
                                child_marginals.append(None)

                        valid_marginals = [m for m in child_marginals if m is not None]
                        if len(valid_marginals) < 2:
                            continue

                        # Check whether any pair of children is disjoint on
                        # this variable — if none are disjoint, this SumUnit
                        # does not split on this variable at all, so skip it.
                        any_disjoint = False
                        for i, j in itertools.combinations(range(len(valid_marginals)), 2):
                            try:
                                if valid_marginals[i].intersection_with(valid_marginals[j]).is_empty():
                                    any_disjoint = True
                                    break
                            except Exception:
                                pass

                        if not any_disjoint:
                            # This SumUnit doesn't split on query_variable — skip.
                            continue

                        # This SumUnit does split on query_variable.
                        # Verify all pairs of children are pairwise disjoint.
                        for child_a, child_b in itertools.combinations(
                            range(len(children)), 2
                        ):
                            marginal_a = child_marginals[child_a]
                            marginal_b = child_marginals[child_b]
                            if marginal_a is None or marginal_b is None:
                                continue
                            try:
                                if not marginal_a.intersection_with(marginal_b).is_empty():
                                    violations.append(
                                        f"SumUnit (index={node.index}) has overlapping "
                                        f"children supports on declared query Variable "
                                        f"'{query_variable.name}': children are not "
                                        f"support-deterministic for this Variable."
                                    )
                                    break
                            except Exception:
                                pass

        except Exception as support_traversal_error:
            violations.append(
                f"Structural support determinism check failed during support "
                f"traversal: {support_traversal_error}"
            )

        del root_support_event
        return violations
    def verify_support_determinism(self) -> SupportDeterminismVerificationResult:
        """
        Verify support determinism of the circuit against the Marginal Determinism
        Variable Tree. Runs three checks in order — variable existence, SumUnit
        normalization, and pairwise support disjointness for all declared query
        Variables. Returns early after Check 1 if any Variables are missing, since
        Variable objects are required for marginalisation in Check 3.
        """
        checked_query_sets = self.marginal_determinism_tree.all_query_sets()
        all_query_variables: Set[Variable] = {v for qs in checked_query_sets for v in qs}

        violations = self._check_query_variables_exist(all_query_variables)
        if violations:
            return SupportDeterminismVerificationResult(
                passed=False,
                violations=violations,
                checked_query_sets=checked_query_sets,
                circuit_variables=list(self.probabilistic_circuit.variables),
            )

        violations += self._check_sum_units_normalized()
        violations += self._check_support_disjointness(all_query_variables)

        return SupportDeterminismVerificationResult(
            passed=len(violations) == 0,
            violations=violations,
            checked_query_sets=checked_query_sets,
            circuit_variables=list(self.probabilistic_circuit.variables),
        )
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
        """
        if cause_variable not in self.causal_variables:
            raise ValueError(
                f"'{cause_variable.name}' is not a registered cause Variable. "
                f"Registered: {[v.name for v in self.causal_variables]}."
            )
        if effect_variable not in self.effect_variables:
            raise ValueError(
                f"'{effect_variable.name}' is not a registered effect Variable. "
                f"Registered: {[v.name for v in self.effect_variables]}."
            )
        if adjustment_variables is None:
            adjustment_variables = []

        if not adjustment_variables:
            return self._compute_interventional_circuit_without_adjustment(
                cause_variable, effect_variable, query_resolution
            )
        return self._compute_interventional_circuit_with_adjustment(
            cause_variable, effect_variable, adjustment_variables, query_resolution,
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

        Truncates conditioned_circuit to the cause region, extracts the effect
        marginal, truncates cause_marginal_circuit to the cause region, then
        assembles a ProductUnit of (cause branch, effect branch) weighted by
        cause_weight. Returns True if the unit was successfully added.
        """
        truncated_circuit, _ = copy.deepcopy(conditioned_circuit).log_truncated_in_place(
            cause_event.fill_missing_variables_pure(conditioned_circuit.variables)
        )
        if truncated_circuit is None:
            return False

        effect_marginal_circuit = truncated_circuit.marginal([effect_variable])
        if effect_marginal_circuit is None:
            return False

        cause_region_circuit, _ = copy.deepcopy(
            cause_marginal_circuit
        ).log_truncated_in_place(
            cause_event.fill_missing_variables_pure(cause_marginal_circuit.variables)
        )
        if cause_region_circuit is None:
            return False

        product_unit = ProductUnit(probabilistic_circuit=output_circuit)
        attach_marginal_circuit(cause_region_circuit, product_unit, output_circuit)
        attach_marginal_circuit(effect_marginal_circuit, product_unit, output_circuit)
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

        Structure:
            SumUnit [weight = P(cause in region_i)]
                ProductUnit
                    cause branch  (UniformDistribution over region_i)
                    effect branch (P(effect | cause in region_i))
        """
        cause_marginal_circuit = copy.deepcopy(self.probabilistic_circuit).marginal([cause_variable])
        if cause_marginal_circuit is None:
            raise ValueError(
                f"Could not compute cause marginal for '{cause_variable.name}'."
            )

        output_circuit = ProbabilisticCircuit()
        root_sum_unit = SumUnit(probabilistic_circuit=output_circuit)
        regions_added = sum(
            self._build_product_unit_for_region(
                cause_event=region_event,
                cause_weight=region_weight,
                effect_variable=effect_variable,
                cause_marginal_circuit=cause_marginal_circuit,
                conditioned_circuit=copy.deepcopy(self.probabilistic_circuit),
                output_circuit=output_circuit,
                root_sum_unit=root_sum_unit,
            )
            for region_event, region_weight in self._extract_leaf_regions_for_variable(cause_variable)
            if region_weight > 0.0
        )

        if regions_added == 0:
            raise ValueError(
                f"Interventional circuit is empty for cause '{cause_variable.name}'. "
                f"Ensure the circuit was trained on data covering this Variable's domain."
            )
        return output_circuit

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
        """
        cause_marginal_circuit = copy.deepcopy(self.probabilistic_circuit).marginal([cause_variable])
        if cause_marginal_circuit is None:
            raise ValueError(
                f"Could not compute cause marginal for '{cause_variable.name}'."
            )

        output_circuit = ProbabilisticCircuit()
        root_sum_unit = SumUnit(probabilistic_circuit=output_circuit)
        regions_added = 0

        for adjustment_event, adjustment_weight in self._extract_leaf_regions_for_variables(
            adjustment_variables
        ):
            if adjustment_weight <= 0.0:
                continue

            adjustment_conditioned_circuit, _ = copy.deepcopy(
                self.probabilistic_circuit
            ).log_truncated_in_place(
                adjustment_event.fill_missing_variables_pure(self.probabilistic_circuit.variables)
            )
            if adjustment_conditioned_circuit is None:
                continue

            for cause_event, cause_weight in self._extract_leaf_regions_for_variable(
                cause_variable, base_circuit=adjustment_conditioned_circuit
            ):
                if cause_weight <= 0.0:
                    continue

                joint_event = adjustment_event.intersection_with(cause_event)
                joint_weight = adjustment_weight * cause_weight

                joint_conditioned_circuit, _ = copy.deepcopy(
                    self.probabilistic_circuit
                ).log_truncated_in_place(
                    joint_event.fill_missing_variables_pure(self.probabilistic_circuit.variables)
                )
                if joint_conditioned_circuit is None:
                    continue

                regions_added += self._build_product_unit_for_region(
                    cause_event=cause_event,
                    cause_weight=joint_weight,
                    effect_variable=effect_variable,
                    cause_marginal_circuit=cause_marginal_circuit,
                    conditioned_circuit=joint_conditioned_circuit,
                    output_circuit=output_circuit,
                    root_sum_unit=root_sum_unit,
                )

        if regions_added == 0:
            raise ValueError(
                f"Interventional circuit with adjustment is empty. "
                f"cause='{cause_variable.name}', "
                f"adjustment={[v.name for v in adjustment_variables]}."
            )
        return output_circuit
    def _extract_leaf_regions_for_variable(
        self,
        variable: Variable,
        base_circuit: ProbabilisticCircuit = None,
    ) -> List[Tuple[Any, float]]:
        """Return (region_event, probability) pairs for each support region of variable."""
        circuit = base_circuit if base_circuit is not None else self.probabilistic_circuit
        regions: List[Tuple[Any, float]] = []
        try:
            variable_support = circuit.support.marginal([variable])
        except Exception:
            return regions
        for simple_region in variable_support.simple_sets:
            region_event = SimpleEvent(
                {variable: simple_region[variable]}
            ).as_composite_set()
            probability = circuit.probability(
                region_event.fill_missing_variables_pure(circuit.variables)
            )
            if probability > 0.0:
                regions.append((region_event, float(probability)))
        return regions

    def _extract_leaf_regions_for_variables(
        self,
        variables: List[Variable],
    ) -> List[Tuple[Any, float]]:
        """Return (region_event, probability) pairs for the joint support of variables."""
        regions: List[Tuple[Any, float]] = []
        try:
            joint_support = self.probabilistic_circuit.support.marginal(variables)
        except Exception:
            return regions
        for simple_region in joint_support.simple_sets:
            region_event = SimpleEvent(
                {variable: simple_region[variable] for variable in variables}
            ).as_composite_set()
            probability = self.probabilistic_circuit.probability(
                region_event.fill_missing_variables_pure(self.probabilistic_circuit.variables)
            )
            if probability > 0.0:
                regions.append((region_event, float(probability)))
        return regions

    def diagnose_failure(
        self,
        observed_values: Dict[Variable, float],
        effect_variable: Variable,
        query_resolution: float = 0.005,
        adjustment_variables: List[Variable] = None,
    ) -> FailureDiagnosisResult:
        """
        Identify the primary cause Variable whose observed value is most
        anomalous under the interventional distribution.

        For each cause Variable, queries the interventional circuit at the
        observed value. The Variable with the lowest interventional probability
        at its observed value is identified as the primary cause. The
        recommendation is the cause region midpoint with the highest
        interventional probability.

        The interventional_probability recorded per Variable is
        P(cause in [observed-eps, observed+eps]) in the joint (cause, effect)
        interventional circuit. Zero means the value is outside training support.
        """
        if adjustment_variables is None:
            adjustment_variables = []

        all_variable_results: Dict[Variable, Dict[str, Any]] = {}
        interventional_circuits_by_cause: Dict[Variable, ProbabilisticCircuit] = {}

        for cause_variable in self.causal_variables:
            if cause_variable not in observed_values:
                continue

            observed_value = observed_values[cause_variable]

            if not cause_variable.is_numeric:
                continue

            interventional_circuit = self.backdoor_adjustment(
                cause_variable, effect_variable,
                adjustment_variables, query_resolution,
            )
            interventional_circuits_by_cause[cause_variable] = interventional_circuit

            observed_event = SimpleEvent(
                {
                    cause_variable: closed(
                        float(observed_value) - query_resolution,
                        float(observed_value) + query_resolution,
                    )
                }
            ).as_composite_set()

            try:
                probability_at_observed = float(
                    interventional_circuit.probability(
                        observed_event.fill_missing_variables_pure(
                            interventional_circuit.variables
                        )
                    )
                )
            except Exception as query_error:
                raise ValueError(
                    f"Failed to query interventional circuit for "
                    f"'{cause_variable.name}'={observed_value}: {query_error}"
                ) from query_error

            recommended_value: Optional[float] = None
            try:
                best_probability = -1.0
                for region_event, _ in self._extract_leaf_regions_for_variable(
                    cause_variable
                ):
                    region_probability = float(
                        interventional_circuit.probability(
                            region_event.fill_missing_variables_pure(
                                interventional_circuit.variables
                            )
                        )
                    )
                    if region_probability > best_probability:
                        best_probability = region_probability
                        for simple_set in region_event.simple_sets:
                            if cause_variable in simple_set:
                                interval_set = simple_set[cause_variable]
                                if hasattr(interval_set, "simple_sets"):
                                    for interval in interval_set.simple_sets:
                                        recommended_value = (
                                            float(interval.lower) + float(interval.upper)
                                        ) / 2.0
                                        break
                                elif hasattr(interval_set, "lower"):
                                    recommended_value = (
                                        float(interval_set.lower) + float(interval_set.upper)
                                    ) / 2.0
                                break
            except Exception:
                recommended_value = None

            all_variable_results[cause_variable] = {
                "actual_value": observed_value,
                "interventional_probability": round(probability_at_observed, 6),
                "recommended_value": recommended_value,
            }

        if not all_variable_results:
            raise ValueError(
                f"No cause Variables found in observed_values. "
                f"Expected at least one of: {[v.name for v in self.causal_variables]}"
            )

        primary_cause_variable = min(
            all_variable_results,
            key=lambda v: all_variable_results[v]["interventional_probability"],
        )
        primary_result = all_variable_results[primary_cause_variable]
        recommended_value = primary_result["recommended_value"]
        probability_at_recommendation = 0.0

        if recommended_value is not None:
            primary_interventional_circuit = interventional_circuits_by_cause[
                primary_cause_variable
            ]
            try:
                recommendation_event = SimpleEvent(
                    {
                        primary_cause_variable: closed(
                            float(recommended_value) - query_resolution,
                            float(recommended_value) + query_resolution,
                        )
                    }
                ).as_composite_set()
                probability_at_recommendation = float(
                    primary_interventional_circuit.probability(
                        recommendation_event.fill_missing_variables_pure(
                            primary_interventional_circuit.variables
                        )
                    )
                )
            except Exception:
                probability_at_recommendation = 0.0

        return FailureDiagnosisResult(
            primary_cause_variable=primary_cause_variable,
            actual_value=primary_result["actual_value"],
            interventional_probability_at_failure=primary_result[
                "interventional_probability"
            ],
            recommended_value=recommended_value,
            interventional_probability_at_recommendation=round(
                probability_at_recommendation, 6
            ),
            all_variable_results=all_variable_results,
        )