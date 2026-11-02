from __future__ import annotations

from enum import Enum

import pytest
from random_events.interval import closed
from random_events.product_algebra import Event, SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Symbolic, Variable

from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MarginalDeterminismTreeNode,
)
from probabilistic_model.probabilistic_circuit.causal.exceptions import (
    NoCauseVariablesError,
    OverlappingComponentsViolation,
    SupportDeterminismVerificationResult,
    UnregisteredVariableError,
)
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
)

from .causal_circuits import (
    Outcome,
    Treatment,
    cause_specific_effect_circuit,
    circuit_with_a_rounded_cause_region,
    confounded_circuit,
    correlated_circuit,
    discrete_confounded_circuit,
    independent_circuit,
    nested_overlap_circuit,
    shared_cause_subcircuit_circuit,
    three_variable_circuit,
    unnormalized_circuit,
)


def causal_circuit_of(
    circuit: ProbabilisticCircuit, causes: list[Variable], effects: list[Variable]
) -> CausalCircuit:
    """
    :param circuit: The circuit to register.
    :param causes: The cause variables, in causal priority order.
    :param effects: The effect variables.
    :return: The causal circuit over the circuit, with the marginal determinism tree
        of the causes.
    """
    return CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode.from_variable_order(causes),
        causes,
        effects,
    )


def interval_probability(
    circuit: ProbabilisticCircuit, variable: Continuous, lower: float, upper: float
) -> float:
    """
    :return: P(variable ∈ [lower, upper]) from circuit.
    """
    event = (
        SimpleEvent.from_data({variable: closed(lower, upper)})
        .as_composite_set()
        .fill_missing_variables_pure(circuit.variables)
    )
    return float(circuit.probability(event))


def value_probability(
    circuit: ProbabilisticCircuit, variable: Symbolic, value: Enum
) -> float:
    """
    :return: P(variable == value) from circuit.
    """
    return float(circuit.probability(value_event(circuit, variable, value)))


def value_event(
    circuit: ProbabilisticCircuit, variable: Symbolic, value: Enum
) -> Event:
    """
    :return: The event that the variable takes the value, over every variable of the
        circuit.
    """
    return (
        SimpleEvent.from_data({variable: Set.from_iterable([value])})
        .as_composite_set()
        .fill_missing_variables_pure(circuit.variables)
    )


# %% marginal determinism trees


@pytest.fixture
def two_level_tree(x, y) -> MarginalDeterminismTreeNode:
    """
    A root that queries *x* with a child per variable.
    """
    root = MarginalDeterminismTreeNode(variables={x, y}, query_set={x})
    MarginalDeterminismTreeNode(variables={x}, query_set={x}, parent_node=root)
    MarginalDeterminismTreeNode(variables={y}, query_set={y}, parent_node=root)
    return root


@pytest.mark.parametrize("query_set_is_empty", [False, True])
def test_a_node_without_children_is_a_leaf(x, query_set_is_empty):
    query_set = set() if query_set_is_empty else {x}
    assert MarginalDeterminismTreeNode(variables={x}, query_set=query_set).is_leaf


def test_a_node_with_children_is_no_leaf(two_level_tree):
    assert not two_level_tree.is_leaf


def test_the_shallowest_node_querying_a_variable_is_found(two_level_tree, x, y):
    assert two_level_tree.find_node_for_variable(x) is two_level_tree
    found = two_level_tree.find_node_for_variable(y)
    assert y in found.query_set
    assert two_level_tree.find_node_for_variable(Continuous("z")) is None


def test_all_query_sets_are_listed_in_preorder(two_level_tree, x, y):
    assert two_level_tree.all_query_sets() == [{x}, {x}, {y}]


def test_an_empty_query_set_is_not_listed(x):
    node = MarginalDeterminismTreeNode(variables={x}, query_set=set())
    assert node.all_query_sets() == []


def test_the_root_of_a_variable_order_queries_its_first_variable(x, y, z):
    assert MarginalDeterminismTreeNode.from_variable_order(
        [x, y, z], causal_priority_order=[z, x, y]
    ).query_set == {z}
    assert MarginalDeterminismTreeNode.from_variable_order([x, y]).query_set == {x}


@pytest.mark.parametrize("number_of_causes", [3, 5])
def test_every_cause_and_no_effect_is_queried(number_of_causes):
    causes = [Continuous(f"x{index}") for index in range(number_of_causes)]
    root = MarginalDeterminismTreeNode.from_variable_order(causes)
    queried = set().union(*root.all_query_sets())
    assert queried == set(causes)
    assert not root.is_leaf


# %% results


def test_a_passing_verification_reads_as_passed(x):
    result = SupportDeterminismVerificationResult(
        passed=True, violations=[], checked_query_sets=[{x}], circuit_variables=[x]
    )
    assert "PASS" in str(result)
    assert "FAIL" not in str(result)
    assert x.name in str(result)


def test_a_failing_verification_reads_as_failed_with_its_violations(x):
    result = SupportDeterminismVerificationResult(
        passed=False,
        violations=[OverlappingComponentsViolation(mixture=0, query_variable=x)],
        checked_query_sets=[{x}],
        circuit_variables=[x],
    )
    assert "FAIL" in str(result)
    assert "overlapping" in str(result)


def test_a_failure_diagnosis_reads_with_its_primary_cause_and_recommendation(x, y):
    result = FailureDiagnosisResult(
        primary_cause_variable=x,
        actual_value=1.3,
        interventional_probability_at_failure=0.0,
        recommended_region=None,
        interventional_probability_at_recommendation=0.149,
        all_variable_results={
            x: {
                "actual_value": 1.3,
                "interventional_probability": 0.0,
                "recommended_region": None,
            },
            y: {
                "actual_value": 0.1,
                "interventional_probability": 0.089,
                "recommended_region": None,
            },
        },
    )
    assert "PRIMARY" in str(result)
    assert "Recommended region" in str(result)


# %% support determinism


@pytest.mark.parametrize(
    "build, number_of_causes",
    [
        (independent_circuit, 1),
        (three_variable_circuit, 2),
        (correlated_circuit, 2),
        (shared_cause_subcircuit_circuit, 1),
    ],
    ids=["independent", "three variables", "correlated", "shared cause subcircuit"],
)
def test_a_support_deterministic_circuit_passes(build, number_of_causes):
    circuit, *variables = build()
    causes, effects = variables[:number_of_causes], variables[number_of_causes:]
    result = causal_circuit_of(circuit, causes, effects).verify_support_determinism()
    assert result.passed, f"Violations: {result.violations}"
    assert result.violations == []
    assert set(result.circuit_variables) == set(variables)
    assert len(result.checked_query_sets) > 0


def test_a_query_variable_absent_from_the_circuit_is_a_named_violation():
    circuit, x, y = independent_circuit()
    ghost = Continuous("ghost")
    causal_circuit = CausalCircuit.from_probabilistic_circuit(
        circuit,
        MarginalDeterminismTreeNode(variables={x, ghost}, query_set={ghost}),
        [x],
        [y],
    )
    with pytest.raises(SupportDeterminismVerificationResult) as raised:
        causal_circuit.verify_support_determinism()
    assert not raised.value.passed
    assert any(ghost.name in str(violation) for violation in raised.value.violations)


def test_an_unnormalized_sum_unit_is_a_log_weight_violation():
    circuit, x, y = unnormalized_circuit()
    with pytest.raises(SupportDeterminismVerificationResult) as raised:
        causal_circuit_of(circuit, [x], [y]).verify_support_determinism()
    assert not raised.value.passed
    assert any("log-weights" in str(violation) for violation in raised.value.violations)


def test_components_overlapping_without_a_disjoint_pair_fail():
    """
    A sum unit splits on a variable as soon as its children's supports on it differ, not
    only once two of them are disjoint; overlapping children that differ still leave no
    disjoint regions to intervene on.
    """
    circuit, x, y = nested_overlap_circuit()
    with pytest.raises(SupportDeterminismVerificationResult) as raised:
        causal_circuit_of(circuit, [x], [y]).verify_support_determinism()
    assert [violation.query_variable for violation in raised.value.violations] == [x]


@pytest.mark.parametrize("number_of_children", [0, 1])
def test_fewer_than_two_children_do_not_split(x, number_of_children):
    """
    A sum unit with no children, or one, has nothing to split on.
    """
    children = [
        SimpleEvent.from_data({x: closed(0, 1)}).as_composite_set()
    ] * number_of_children
    assert not CausalCircuit._components_split_on_variable(children)


# %% backdoor adjustment


@pytest.fixture
def independent_causal_circuit() -> tuple[CausalCircuit, Continuous, Continuous]:
    """
    The causal circuit of :func:`independent_circuit` with *x* causing *y*.
    """
    circuit, x, y = independent_circuit()
    return causal_circuit_of(circuit, [x], [y]), x, y


@pytest.fixture
def correlated_causal_circuit() -> (
    tuple[CausalCircuit, Continuous, Continuous, Continuous]
):
    """
    The causal circuit of :func:`correlated_circuit` with *x* and *w* causing *y*.
    """
    circuit, x, w, y = correlated_circuit()
    return causal_circuit_of(circuit, [x, w], [y]), x, w, y


@pytest.mark.parametrize(
    "variable_index", [1, 0], ids=["effect as cause", "cause as effect"]
)
def test_a_variable_in_an_unregistered_role_is_refused(
    independent_causal_circuit, variable_index
):
    causal_circuit, *variables = independent_causal_circuit
    variable = variables[variable_index]
    with pytest.raises(UnregisteredVariableError):
        causal_circuit.backdoor_adjustment(variable, variable, [])


def test_the_default_adjustment_is_empty(independent_causal_circuit):
    causal_circuit, x, y = independent_causal_circuit
    explicit = causal_circuit.backdoor_adjustment(x, y, [])
    default = causal_circuit.backdoor_adjustment(x, y)
    assert interval_probability(explicit, y, 0, 1) == pytest.approx(
        interval_probability(default, y, 0, 1), abs=1e-6
    )


@pytest.mark.parametrize(
    "build, number_of_causes, adjust, upper",
    [
        (independent_circuit, 1, False, 2),
        (three_variable_circuit, 2, False, 2),
        (correlated_circuit, 2, False, 10),
        (confounded_circuit, 1, True, 2),
    ],
    ids=["independent", "three variables", "correlated", "confounded adjusted"],
)
def test_the_interventional_distribution_integrates_to_one(
    build, number_of_causes, adjust, upper
):
    circuit, *variables = build()
    causes, effect = variables[:number_of_causes], variables[number_of_causes]
    adjustment = variables[number_of_causes + 1 :] if adjust else []
    interventional = causal_circuit_of(circuit, causes, [effect]).backdoor_adjustment(
        causes[0], effect, adjustment
    )
    assert interval_probability(interventional, effect, 0, upper) == pytest.approx(
        1.0, abs=0.01
    )


def test_intervening_on_an_independent_cause_keeps_the_marginal_of_the_effect(
    independent_causal_circuit,
):
    causal_circuit, x, y = independent_causal_circuit
    interventional = causal_circuit.backdoor_adjustment(x, y, [])
    assert interval_probability(
        causal_circuit.probabilistic_circuit, y, 0, 1
    ) == pytest.approx(0.5, abs=0.01)
    assert interval_probability(interventional, y, 0, 1) == pytest.approx(0.5, abs=0.05)


@pytest.mark.parametrize("variable_index", [0, 1], ids=["cause", "effect"])
def test_a_value_outside_the_support_has_no_interventional_probability(
    independent_causal_circuit, variable_index
):
    causal_circuit, x, y = independent_causal_circuit
    interventional = causal_circuit.backdoor_adjustment(x, y, [])
    variable = (x, y)[variable_index]
    assert interval_probability(interventional, variable, 5, 6) == pytest.approx(
        0.0, abs=1e-6
    )


def test_the_effect_regions_of_a_correlated_circuit_hold_all_mass(
    correlated_causal_circuit,
):
    causal_circuit, x, _, y = correlated_causal_circuit
    interventional = causal_circuit.backdoor_adjustment(x, y, [])
    low = interval_probability(interventional, y, 0, 0.4)
    high = interval_probability(interventional, y, 9.6, 10)
    assert low > 0.0
    assert high > 0.0
    assert low + high == pytest.approx(1.0, abs=0.05)


def test_adjusting_for_a_confounder_recovers_the_causal_probability():
    circuit, x, y, z = confounded_circuit()
    adjusted = causal_circuit_of(circuit, [x], [y]).backdoor_adjustment(x, y, [z])
    assert interval_probability(adjusted, y, 0, 1) == pytest.approx(0.5, abs=0.1)


@pytest.mark.parametrize("adjust", [False, True], ids=["unadjusted", "adjusted"])
def test_adjusting_on_independent_data_does_not_change_the_result(adjust):
    circuit, x, y, z = three_variable_circuit()
    interventional = causal_circuit_of(circuit, [x, y], [z]).backdoor_adjustment(
        x, z, [y] if adjust else []
    )
    assert interval_probability(interventional, z, 0, 1) == pytest.approx(0.8, abs=0.05)


def test_backdoor_adjustment_of_a_shared_cause_subcircuit_is_valid():
    circuit, x, y = shared_cause_subcircuit_circuit()
    interventional = causal_circuit_of(circuit, [x], [y]).backdoor_adjustment(x, y, [])
    assert interventional.is_valid()


def test_a_rounded_cause_region_is_intervened_on_with_a_branch_per_region():
    """
    A cause region whose value an interval rounds is still intervened on, and the
    interventional circuit has one root with one branch per cause region.
    """
    circuit, x, y, z = circuit_with_a_rounded_cause_region()
    causal_circuit = causal_circuit_of(circuit, [x], [y])
    adjusted = causal_circuit.backdoor_adjustment(x, y, [z])
    assert isinstance(adjusted.root, SumUnit)
    assert len(adjusted.root.subcircuits) == len(
        causal_circuit._extract_disjoint_regions_for_variable(x)
    )


# %% backdoor adjustment of a discrete confounder


@pytest.fixture
def discrete_confounded() -> tuple[CausalCircuit, Symbolic, Symbolic, Symbolic]:
    """
    The causal circuit of :func:`discrete_confounded_circuit`, whose treatment has no
    causal effect on the outcome, only a spurious correlation through the season.
    """
    circuit, treatment, season, outcome = discrete_confounded_circuit()
    return (
        causal_circuit_of(circuit, [treatment, season], [outcome]),
        treatment,
        season,
        outcome,
    )


def test_conditioning_on_a_confounded_cause_is_spurious(discrete_confounded):
    causal_circuit, treatment, _, outcome = discrete_confounded
    circuit = causal_circuit.probabilistic_circuit
    conditioned = circuit.truncated(
        value_event(circuit, treatment, Treatment.HIGH)
    ).distribution
    assert value_probability(conditioned, outcome, Outcome.GOOD) == pytest.approx(
        0.8, abs=0.01
    )


def test_intervention_without_adjustment_matches_conditioning(discrete_confounded):
    # an empty adjustment set cannot separate the confound from the effect, so it
    # reproduces the spurious 0.8 rather than the causal 0.6
    causal_circuit, treatment, _, outcome = discrete_confounded
    naive = causal_circuit.backdoor_adjustment(treatment, outcome)
    narrowed = naive.truncated(
        value_event(naive, treatment, Treatment.HIGH)
    ).distribution
    assert value_probability(narrowed, outcome, Outcome.GOOD) == pytest.approx(
        0.8, abs=0.01
    )


@pytest.mark.parametrize(
    "build, treatment_value, expected",
    [
        (discrete_confounded_circuit, Treatment.HIGH, 0.6),
        (discrete_confounded_circuit, Treatment.LOW, 0.6),
        (cause_specific_effect_circuit, Treatment.HIGH, 1.0),
        (cause_specific_effect_circuit, Treatment.LOW, 0.0),
    ],
    ids=[
        "no effect, high",
        "no effect, low",
        "cause specific effect, high",
        "cause specific effect, low",
    ],
)
def test_adjustment_recovers_the_causal_probability_of_each_cause_value(
    build, treatment_value, expected
):
    circuit, treatment, season, outcome = build()
    adjusted = causal_circuit_of(
        circuit, [treatment, season], [outcome]
    ).backdoor_adjustment(treatment, outcome, [season])
    region_truncation = adjusted.truncated(
        value_event(adjusted, treatment, treatment_value)
    )
    assert region_truncation.probability > 0.0
    assert value_probability(
        region_truncation.distribution, outcome, Outcome.GOOD
    ) == pytest.approx(expected, abs=0.01)


def test_adjusted_distribution_of_a_discrete_effect_sums_to_one(discrete_confounded):
    causal_circuit, treatment, season, outcome = discrete_confounded
    adjusted = causal_circuit.backdoor_adjustment(treatment, outcome, [season])
    total = sum(value_probability(adjusted, outcome, value) for value in Outcome)
    assert total == pytest.approx(1.0, abs=0.01)


def test_adjusted_region_mass_is_the_marginal_cause_probability(discrete_confounded):
    causal_circuit, treatment, season, outcome = discrete_confounded
    adjusted = causal_circuit.backdoor_adjustment(treatment, outcome, [season])
    high_mass = adjusted.truncated(
        value_event(adjusted, treatment, Treatment.HIGH)
    ).probability
    # P(treatment=HIGH) = P(WARM)*0.8 + P(COLD)*0.3 = 0.6*0.8 + 0.4*0.3 = 0.6
    assert high_mass == pytest.approx(0.6, abs=0.01)


# %% regions of a variable


@pytest.mark.parametrize("value", [Treatment.HIGH, 0.5])
def test_an_atomic_value_is_not_split(value):
    assert CausalCircuit._split_into_atomic_values(value) == [value]


def test_a_union_of_values_is_split_into_its_elements():
    circuit, treatment, _, _ = discrete_confounded_circuit()
    union_value = next(
        simple_region[treatment] for simple_region in circuit.support.simple_sets
    )
    split_values = CausalCircuit._split_into_atomic_values(union_value)
    assert {value.element for value in split_values} == {Treatment.HIGH, Treatment.LOW}
    assert len(split_values) == 2


@pytest.mark.parametrize("variable_index", [0, 1], ids=["cause", "effect"])
def test_leaf_region_probabilities_are_positive_and_sum_to_one(
    independent_causal_circuit, variable_index
):
    causal_circuit, *variables = independent_causal_circuit
    regions = causal_circuit._extract_leaf_regions_for_variable(
        variables[variable_index]
    )
    assert all(region.probability > 0.0 for region in regions)
    assert sum(region.probability for region in regions) == pytest.approx(1.0, abs=0.01)


def test_disjoint_regions_keep_the_branches_the_leaf_regions_merge(
    correlated_causal_circuit,
):
    causal_circuit, x, _, _ = correlated_causal_circuit
    disjoint_regions = causal_circuit._extract_disjoint_regions_for_variable(x)
    assert len(causal_circuit._extract_leaf_regions_for_variable(x)) == 1
    # the two branches of x are equally weighted and disjoint
    assert sorted(region.probability for region in disjoint_regions) == pytest.approx(
        [0.5, 0.5], abs=0.05
    )


def test_disjoint_and_leaf_regions_agree_on_one_contiguous_region(
    independent_causal_circuit,
):
    causal_circuit, _, y = independent_causal_circuit
    assert len(causal_circuit._extract_disjoint_regions_for_variable(y)) == len(
        causal_circuit._extract_leaf_regions_for_variable(y)
    )


@pytest.fixture
def interventional_correlated(correlated_causal_circuit) -> ProbabilisticCircuit:
    """
    The interventional distribution of *y* under *x* of :func:`correlated_circuit`.
    """
    causal_circuit, x, _, y = correlated_causal_circuit
    return causal_circuit.backdoor_adjustment(x, y, [])


def truncated_to(
    circuit: ProbabilisticCircuit, variable: Continuous, lower: float, upper: float
) -> ProbabilisticCircuit:
    """
    :return: The circuit truncated to the interval of the variable.
    """
    event = SimpleEvent.from_data({variable: closed(lower, upper)}).as_composite_set()
    truncated = circuit.truncated(
        event.fill_missing_variables_pure(circuit.variables)
    ).distribution
    return truncated


def test_the_best_region_is_near_certain_once_truncated_to_one_branch(
    correlated_causal_circuit, interventional_correlated
):
    causal_circuit, x, _, y = correlated_causal_circuit
    truncated = truncated_to(interventional_correlated, y, 9.6, 10)
    best_region = causal_circuit._best_disjoint_region(x, truncated)
    assert truncated.probability(
        best_region.fill_missing_variables_pure(truncated.variables)
    ) == pytest.approx(1.0, abs=0.01)


def test_the_best_region_differs_for_different_effect_conditions(
    correlated_causal_circuit, interventional_correlated
):
    causal_circuit, x, _, y = correlated_causal_circuit
    low_best = causal_circuit._best_disjoint_region(
        x, truncated_to(interventional_correlated, y, 0, 0.4)
    )
    high_best = causal_circuit._best_disjoint_region(
        x, truncated_to(interventional_correlated, y, 9.6, 10)
    )
    assert low_best != high_best


# %% failure diagnosis


def test_a_diagnosis_reports_every_observed_cause(independent_causal_circuit):
    causal_circuit, x, y = independent_causal_circuit
    result = causal_circuit.diagnose_failure({x: 0.5}, y)
    assert result.actual_value == pytest.approx(0.5, abs=1e-6)
    assert set(result.all_variable_results) == {x}
    for variable_result in result.all_variable_results.values():
        assert set(variable_result) >= {
            "actual_value",
            "interventional_probability",
            "recommended_region",
        }


def test_a_diagnosis_without_observations_is_refused(independent_causal_circuit):
    causal_circuit, _, y = independent_causal_circuit
    with pytest.raises(NoCauseVariablesError):
        causal_circuit.diagnose_failure({}, y)


def test_the_primary_cause_has_the_lowest_interventional_probability(
    independent_causal_circuit,
):
    causal_circuit, x, y = independent_causal_circuit
    result = causal_circuit.diagnose_failure({x: 0.5}, y)
    lowest = min(
        variable_result["interventional_probability"]
        for variable_result in result.all_variable_results.values()
    )
    assert result.interventional_probability_at_failure == pytest.approx(
        lowest, abs=1e-6
    )
    assert result.interventional_probability_at_recommendation > 0.0


@pytest.mark.parametrize(
    "build, number_of_causes",
    [(three_variable_circuit, 2), (correlated_circuit, 2)],
    ids=["three variables", "correlated"],
)
def test_an_out_of_domain_cause_is_the_primary_cause(build, number_of_causes):
    circuit, *variables = build()
    causes, effect = variables[:number_of_causes], variables[number_of_causes]
    first_cause, second_cause = causes
    result = causal_circuit_of(circuit, causes, [effect]).diagnose_failure(
        {first_cause: 5.0, second_cause: 0.5}, effect
    )
    assert result.primary_cause_variable == first_cause
    assert result.interventional_probability_at_failure == pytest.approx(0.0, abs=1e-6)
    assert result.all_variable_results[second_cause]["interventional_probability"] > 0
    assert result.recommended_region is not None
    assert (
        result.interventional_probability_at_recommendation
        > result.interventional_probability_at_failure
    )


def test_in_domain_causes_have_positive_interventional_probability(
    correlated_causal_circuit,
):
    causal_circuit, x, w, y = correlated_causal_circuit
    result = causal_circuit.diagnose_failure({x: 0.5, w: 0.5}, y)
    for variable_result in result.all_variable_results.values():
        assert variable_result["interventional_probability"] > 0.0
