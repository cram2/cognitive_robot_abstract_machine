from __future__ import annotations

import copy
import itertools
import math
from dataclasses import dataclass, field
from typing import Any, Dict, Generic, List, Optional, Set, Tuple, Union

from anytree import NodeMixin, PreOrderIter, findall
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent, Event
from random_events.sigma_algebra import AbstractCompositeSet, AbstractSimpleSet
from random_events.variable import Variable, compatible_types
from sortedcontainers import SortedSet
from tabulate import tabulate


from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

from probabilistic_model.probabilistic_circuit.probabilistic_circuit_representation import (
    MixtureOverVariable,
    RepresentationType,
)


from probabilistic_model.probabilistic_circuit.causal.exceptions import (
    SupportDeterminismViolation,
    MissingQueryVariableViolation,
    UnnormalizedMixtureViolation,
    OverlappingComponentsViolation,
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
    which Variables the mixtures at this level must be support-deterministic over.
    Support determinism enables polytime backdoor adjustment.

    Inherits from anytree.NodeMixin, which provides parent / children management,
    is_leaf and is_root properties, and PreOrderIter / findall traversal utilities.

    Build using MarginalDeterminismTreeNode.from_causal_graph() rather than constructing
    nodes manually.

    :param variables: All Variables in this node's subtree.
    :param query_set: Variables for which the mixtures at this level must be support-
        deterministic. Defaults to an empty set.
    :param parent_node: Parent node in the tree. None for the root.
    """

    variables: Set[Variable]
    """All Variables in this node's subtree."""

    query_set: Set[Variable] = field(default=None)
    """
    Variables for which the mixtures at this level must be support-deterministic.

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
class CausalCircuit(Generic[RepresentationType], SubClassSafeGeneric):
    """
    A circuit that answers what the effect's distribution becomes when the cause is
    intervened on.

    The interventional circuit comes back in the representation the circuit was read
    from.
    """

    probabilistic_circuit: RepresentationType
    """
    The fitted circuit that encodes the joint distribution over all variables.
    """

    marginal_determinism_tree: MarginalDeterminismTreeNode
    """
    Tree encoding which Variables the mixtures must be support-deterministic over.
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
        circuit: RepresentationType,
        marginal_determinism_tree: MarginalDeterminismTreeNode,
        causal_variables: List[Variable],
        effect_variables: List[Variable],
    ) -> CausalCircuit[RepresentationType]:
        """
        Construct a CausalCircuit from an existing circuit without retraining.

        :param circuit: A fitted circuit over all causal and effect variables.
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

    def _check_mixtures_normalized(self) -> List[UnnormalizedMixtureViolation]:
        """
        Check that the log-weights of every mixture sum to log(1).

        :returns: List of violations, empty if all mixtures are normalised.
        """
        return [
            UnnormalizedMixtureViolation(
                mixture=unnormalized.mixture,
                actual_log_weight_sum=unnormalized.log_weight_sum,
            )
            for unnormalized in self.probabilistic_circuit.unnormalized_mixtures()
        ]

    @staticmethod
    def _components_split_on_variable(
        component_supports: List[Event],
    ) -> bool:
        """
        Whether the mixture splits on the query Variable, which it does as soon as its
        components differ in their support there, disjointly or not.

        :param component_supports: Support events, one per component of the mixture.
        :returns: True if any two supports differ.
        """
        if len(component_supports) < 2:
            return False
        first, *others = component_supports
        return any(other != first for other in others)

    @staticmethod
    def _overlapping_pair_exists(component_supports: List[Event]) -> bool:
        """
        Return True if any pair of component supports intersects in more than a
        boundary.

        Two continuous regions that only touch at an endpoint, as neighbouring leaves of
        a fitted tree do, intersect in a set of size zero and do not overlap.

        :param component_supports: Support events, one per component of the mixture.
        :returns: True if any pair overlaps.
        """
        return any(
            first.intersection_with(second).size > 0
            for first, second in itertools.combinations(component_supports, 2)
        )

    def _check_mixture_for_variable(
        self, mixture: MixtureOverVariable
    ) -> Optional[OverlappingComponentsViolation]:
        """
        Check a single mixture for disjoint components, given each component's support
        already restricted to one query Variable.

        Returns a violation if the mixture splits on the variable but has at least one
        overlapping pair of components. Returns None if the mixture does not split on
        the variable, or if all splitting components are pairwise disjoint.

        :param mixture: The mixture, seen through the query Variable being checked.
        :returns: A violation if overlapping components are detected, else None.
        """
        if not self._components_split_on_variable(mixture.component_supports):
            return None
        if self._overlapping_pair_exists(mixture.component_supports):
            return OverlappingComponentsViolation(
                mixture=mixture.mixture,
                query_variable=mixture.variable,
            )
        return None

    def _check_support_disjointness(
        self, all_query_variables: Set[Variable]
    ) -> List[OverlappingComponentsViolation]:
        """
        Check that for each declared query Variable, no mixture that splits on that
        Variable has components with overlapping support.

        :param all_query_variables: Union of all Variables across all query_sets.
        :returns: List of violations, empty if all split nodes are support-disjoint.
        """
        violations = [
            self._check_mixture_for_variable(mixture)
            for mixture in self.probabilistic_circuit.mixtures_over(
                SortedSet(all_query_variables)
            )
        ]
        return [violation for violation in violations if violation is not None]

    def verify_support_determinism(self) -> SupportDeterminismVerificationResult:
        """
        Verify support determinism of the circuit against the Marginal Determinism
        Variable Tree. Runs three checks in order — variable existence, mixture
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

        violations += self._check_mixtures_normalized()
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
    ) -> RepresentationType:
        """
        Compute P(effect | do(cause)) as a new circuit of the same type.

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
        :returns: Joint interventional circuit over (cause, effect).
        :raises UnregisteredVariableError: If the cause or the effect Variable was not
            registered at construction time.
        :raises EmptyInterventionalCircuitError: If no cause region has an effect
            distribution to pair with.
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
        return self._interventional_circuit(
            cause_variable, effect_variable, adjustment_variables or []
        )

    def _circuit_truncated_to(self, event: Event) -> Optional[RepresentationType]:
        """
        Truncate a copy of the wrapped circuit to an event.

        :param event: The event to truncate to, over some of the circuit's variables.
        :returns: The truncated copy, or None if the event is impossible.
        """
        truncated, _ = copy.deepcopy(self.probabilistic_circuit).log_truncated_in_place(
            event.fill_missing_variables_pure(self.probabilistic_circuit.variables)
        )
        return truncated

    def _product_for_cause_region(
        self,
        cause_region: SupportRegion,
        effect_branch: RepresentationType,
        cause_marginal_circuit: RepresentationType,
    ) -> Optional[RepresentationType]:
        """
        Build one branch of the interventional circuit: the cause restricted to its own
        region, multiplied by the effect distribution that region implies.

        :param cause_region: The cause region this branch stands for.
        :param effect_branch: The effect distribution within the region.
        :param cause_marginal_circuit: Marginal circuit over the cause Variable.
        :returns: The product, or None if the region is empty in the cause marginal.
        """
        cause_branch, _ = copy.deepcopy(cause_marginal_circuit).log_truncated_in_place(
            cause_region.event.fill_missing_variables_pure(
                cause_marginal_circuit.variables
            )
        )
        if cause_branch is None:
            return None
        return type(self.probabilistic_circuit).product_of(
            [cause_branch, effect_branch]
        )

    def _interventional_circuit(
        self,
        cause_variable: Variable,
        effect_variable: Variable,
        adjustment_variables: List[Variable],
    ) -> RepresentationType:
        """
        Compute P(cause, effect | do(cause)): a mixture over the disjoint cause regions,
        each a product of the cause marginal restricted to that region with the effect
        distribution that region implies.

        :param cause_variable: The cause Variable to intervene on.
        :param effect_variable: The effect Variable to compute the distribution over.
        :param adjustment_variables: Variables to adjust for, possibly none.
        :returns: Joint interventional circuit over (cause, effect).
        :raises EmptyInterventionalCircuitError: If no cause region has an effect
            distribution to pair with.
        """
        cause_marginal_circuit = copy.deepcopy(self.probabilistic_circuit).marginal(
            [cause_variable]
        )
        adjustment_partitions = self._adjustment_partitions(adjustment_variables)
        products: List[RepresentationType] = []
        log_weights: List[float] = []
        for cause_region in self._extract_disjoint_regions_for_variable(cause_variable):
            effect_branch = self._effect_branch_for_cause_region(
                cause_region, effect_variable, adjustment_partitions
            )
            if effect_branch is None:
                continue
            product = self._product_for_cause_region(
                cause_region, effect_branch, cause_marginal_circuit
            )
            if product is None:
                continue
            products.append(product)
            log_weights.append(math.log(cause_region.probability))

        if not products:
            raise EmptyInterventionalCircuitError(
                cause_variable_name=cause_variable.name,
                adjustment_variable_names=[
                    variable.name for variable in adjustment_variables
                ],
            )
        return type(self.probabilistic_circuit).mixture_of(products, log_weights)

    def _adjustment_partitions(
        self, adjustment_variables: List[Variable]
    ) -> List[SupportRegion]:
        """
        :param adjustment_variables: Variables to adjust for, possibly none.
        :returns: One region per distinct value combination of the adjustment variables
            with positive probability -- together they partition the population into
            disjoint, exhaustive groups. Empty if there is nothing to adjust for.
        """
        if not adjustment_variables:
            return []
        return [
            partition
            for partition in self._extract_leaf_regions_for_variables(
                adjustment_variables
            )
            if partition.probability > 0.0
        ]

    def _effect_branch_for_cause_region(
        self,
        cause_region: SupportRegion,
        effect_variable: Variable,
        adjustment_partitions: List[SupportRegion],
    ) -> Optional[RepresentationType]:
        """
        Build the effect side of one branch of the interventional circuit.

        :param cause_region: The cause region the branch stands for.
        :param effect_variable: The effect Variable to marginalise onto.
        :param adjustment_partitions: The adjustment partitions, empty when there is
            nothing to adjust for.
        :returns: The effect distribution within the region, or None if the region is
            impossible.
        """
        if not adjustment_partitions:
            truncated = self._circuit_truncated_to(cause_region.event)
            if truncated is None:
                return None
            return truncated.marginal([effect_variable])
        return self._effect_branch_adjusted_over(
            cause_region, effect_variable, adjustment_partitions
        )

    def _effect_branch_adjusted_over(
        self,
        cause_region: SupportRegion,
        effect_variable: Variable,
        adjustment_partitions: List[SupportRegion],
    ) -> Optional[RepresentationType]:
        """
        Build ``sum_z P(effect | cause in region, Z=z) * P(Z=z)`` as a mixture over the
        adjustment partitions, each weighted by its own probability.

        Partitions that are empty within this cause region are left out, so the mixture
        renormalizes the weights that remain.

        :param cause_region: The cause region the branch stands for.
        :param effect_variable: The effect Variable to marginalise onto.
        :param adjustment_partitions: One region per distinct value combination of the
            adjustment variables.
        :returns: The adjusted effect distribution, or None if every partition's joint
            truncation with this cause region was empty.
        """
        components: List[RepresentationType] = []
        log_weights: List[float] = []
        for adjustment_partition in adjustment_partitions:
            # fill first: intersection_with keeps only the left operand's variables.
            joint_event = adjustment_partition.event.fill_missing_variables_pure(
                self.probabilistic_circuit.variables
            ).intersection_with(
                cause_region.event.fill_missing_variables_pure(
                    self.probabilistic_circuit.variables
                )
            )
            truncated = self._circuit_truncated_to(joint_event)
            if truncated is None:
                continue
            components.append(truncated.marginal([effect_variable]))
            log_weights.append(math.log(adjustment_partition.probability))

        if not components:
            return None
        return type(self.probabilistic_circuit).mixture_of(components, log_weights)

    def _extract_leaf_regions_for_variable(
        self,
        variable: Variable,
        base_circuit: RepresentationType = None,
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

    def number_of_regions_of(self, variable: Variable) -> int:
        """
        :param variable: The Variable to count the regions of.
        :returns: How many disjoint support regions the circuit splits the Variable
            into, which is how many branches :meth:`backdoor_adjustment` builds for it
            as a cause, and how many values it enumerates for it as an adjustment
            variable.
        """
        return len(self._extract_disjoint_regions_for_variable(variable))

    def _extract_disjoint_regions_for_variable(
        self,
        variable: Variable,
        base_circuit: RepresentationType = None,
    ) -> List[SupportRegion]:
        """
        The disjoint support regions a variable takes, one per distinct value the
        circuit's branches hold, each carrying the probability of every branch holding
        it.

        :param variable: The Variable whose disjoint support regions to extract.
        :param base_circuit: Circuit to query. Defaults to self.probabilistic_circuit.
        :returns: List of regions, one per disjoint region, each with positive
            probability, in the order of their values.
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
        return [regions_by_value[value] for value in sorted(regions_by_value)]

    @staticmethod
    def _split_into_atomic_values(
        value: Union[AbstractCompositeSet, *compatible_types],
    ) -> List[Union[AbstractSimpleSet, *compatible_types]]:
        """
        Split *value* into the elements it holds: the disjoint ranges of an
        :class:`~random_events.interval.Interval` or the values of a
        :class:`~random_events.set.Set`.

        A composite set holding a single element is split the same way, so a range
        written on its own and the same range written inside a union come back as one
        element and count as one region. A value read off a deterministic leaf, one of
        :data:`~random_events.variable.compatible_types`, is returned as it is.

        :param value: A single support value read off a joint support's simple set.
        :returns: The elements of *value*.
        """
        if isinstance(value, AbstractCompositeSet):
            return list(value.simple_sets)
        return [value]

    def _best_disjoint_region(
        self,
        variable: Variable,
        interventional_circuit: RepresentationType,
    ) -> Optional[Event]:
        """
        The disjoint support region of a variable the interventional circuit gives the
        highest probability, or None where it has none.

        Regions of equal probability are decided by the order of their values.

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
        :returns: List of regions, one per joint region, each with positive probability,
            in the order of their values.
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
        return [regions_by_values[values] for values in sorted(regions_by_values)]

    def _query_probability_at_value(
        self,
        interventional_circuit: RepresentationType,
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
        interventional_circuit: RepresentationType,
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
    ) -> Tuple[Dict[str, Any], RepresentationType]:
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
        interventional_circuits_by_cause: Dict[Variable, RepresentationType] = {}

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
