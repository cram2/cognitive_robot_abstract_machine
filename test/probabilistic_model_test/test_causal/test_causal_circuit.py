import math
import unittest
from enum import Enum, auto

import numpy as np
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.set import Set
from random_events.variable import Continuous, Symbolic

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.distributions.distributions import SymbolicDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.utils import MissingDict

from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MarginalDeterminismTreeNode,
)

from probabilistic_model.probabilistic_circuit.causal.exceptions import (
    OverlappingChildSupportsViolation,
    SupportDeterminismVerificationResult,
    UnregisteredVariableError,
    NoCauseVariablesError,
)


def _build_independent_circuit() -> tuple:
    """
    ProductUnit(SumUnit_x, SumUnit_y) — x and y are independent.

        SumUnit_x  [x∈[0,1] w=0.6,  x∈[1,2] w=0.4]
        SumUnit_y  [y∈[0,1] w=0.5,  y∈[1,2] w=0.5]

    Ground truth:
        P(y∈[0,1]) = 0.5 regardless of x.
        P(y | do(x)) = P(y) for all x in the support.
    """
    x = Continuous("x")
    y = Continuous("y")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)

    sum_x = SumUnit(probabilistic_circuit=circuit)
    sum_x.add_subcircuit(
        leaf(
            UniformDistribution(variable=x, interval=closed(0, 1).simple_sets[0]),
            circuit,
        ),
        math.log(0.6),
    )
    sum_x.add_subcircuit(
        leaf(
            UniformDistribution(variable=x, interval=closed(1, 2).simple_sets[0]),
            circuit,
        ),
        math.log(0.4),
    )

    sum_y = SumUnit(probabilistic_circuit=circuit)
    sum_y.add_subcircuit(
        leaf(
            UniformDistribution(variable=y, interval=closed(0, 1).simple_sets[0]),
            circuit,
        ),
        math.log(0.5),
    )
    sum_y.add_subcircuit(
        leaf(
            UniformDistribution(variable=y, interval=closed(1, 2).simple_sets[0]),
            circuit,
        ),
        math.log(0.5),
    )

    root.add_subcircuit(sum_x)
    root.add_subcircuit(sum_y)
    return circuit, x, y


def _build_three_variable_circuit() -> tuple:
    """
    ProductUnit(SumUnit_x, SumUnit_y, SumUnit_z) — all three independent.

        SumUnit_x [x∈[0,1] w=0.7, x∈[1,2] w=0.3]
        SumUnit_y [y∈[0,1] w=0.4, y∈[1,2] w=0.6]
        SumUnit_z [z∈[0,1] w=0.8, z∈[1,2] w=0.2]

    Ground truth:
        P(z∈[0,1]) = 0.8 regardless of x or y.
        P(z | do(x)) = P(z) with or without adjustment on y.
    """
    x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)

    sum_x = SumUnit(probabilistic_circuit=circuit)
    sum_x.add_subcircuit(
        leaf(
            UniformDistribution(variable=x, interval=closed(0, 1).simple_sets[0]),
            circuit,
        ),
        math.log(0.7),
    )
    sum_x.add_subcircuit(
        leaf(
            UniformDistribution(variable=x, interval=closed(1, 2).simple_sets[0]),
            circuit,
        ),
        math.log(0.3),
    )

    sum_y = SumUnit(probabilistic_circuit=circuit)
    sum_y.add_subcircuit(
        leaf(
            UniformDistribution(variable=y, interval=closed(0, 1).simple_sets[0]),
            circuit,
        ),
        math.log(0.4),
    )
    sum_y.add_subcircuit(
        leaf(
            UniformDistribution(variable=y, interval=closed(1, 2).simple_sets[0]),
            circuit,
        ),
        math.log(0.6),
    )

    sum_z = SumUnit(probabilistic_circuit=circuit)
    sum_z.add_subcircuit(
        leaf(
            UniformDistribution(variable=z, interval=closed(0, 1).simple_sets[0]),
            circuit,
        ),
        math.log(0.8),
    )
    sum_z.add_subcircuit(
        leaf(
            UniformDistribution(variable=z, interval=closed(1, 2).simple_sets[0]),
            circuit,
        ),
        math.log(0.2),
    )

    root.add_subcircuit(sum_x)
    root.add_subcircuit(sum_y)
    root.add_subcircuit(sum_z)

    return circuit, x, y, z


def _build_correlated_circuit() -> tuple:
    """
    SumUnit-rooted mixture where x fully determines which y region is active.

    Two equal-weight components:
        Low:  x∈[0,1], w∈[0,1], y∈[0,0.4]
        High: x∈[1,2], w∈[0,1], y∈[9.6,10]

    Ground truth:
        y∈[0,0.4] and y∈[9.6,10] each occur with marginal probability 0.5.
        w has the same marginal in both components — it carries no information about y.
    """
    x, w, y = Continuous("x"), Continuous("w"), Continuous("y")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)

    for x_range, y_range in [((0, 1), (0, 0.4)), ((1, 2), (9.6, 10))]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=x, interval=closed(*x_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                UniformDistribution(variable=w, interval=closed(0, 1).simple_sets[0]),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=y, interval=closed(*y_range).simple_sets[0]
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(0.5))

    return circuit, x, w, y


def _build_confounded_circuit() -> tuple:
    """
    Circuit with confounder z that drives both x and y.

    Two equal-weight strata:
        Low stratum:  x∈[0,1], y∈[0,1], z∈[0,1]
        High stratum: x∈[1,2], y∈[1,2], z∈[1,2]

    Ground truth:
        P(y∈[0,1] | x∈[0,1]) = 1.0  — spurious, driven by z.
        P(y∈[0,1] | do(x))    = 0.5  — causal truth after adjusting for z.
    """
    x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)

    for x_range, y_range, z_range in [
        ((0, 1), (0, 1), (0, 1)),
        ((1, 2), (1, 2), (1, 2)),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=x, interval=closed(*x_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=y, interval=closed(*y_range).simple_sets[0]
                ),
                circuit,
            )
        )
        component.add_subcircuit(
            leaf(
                UniformDistribution(
                    variable=z, interval=closed(*z_range).simple_sets[0]
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(0.5))

    return circuit, x, y, z


def _build_unnormalized_circuit() -> tuple:
    """
    ProductUnit circuit whose SumUnit over x has weights summing to 1.6.

    verify_support_determinism should report a normalization violation.
    """
    x = Continuous("x")
    y = Continuous("y")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)

    sum_x = SumUnit(probabilistic_circuit=circuit)
    sum_x.add_subcircuit(
        leaf(
            UniformDistribution(variable=x, interval=closed(0, 1).simple_sets[0]),
            circuit,
        ),
        math.log(0.8),
    )
    sum_x.add_subcircuit(
        leaf(
            UniformDistribution(variable=x, interval=closed(1, 2).simple_sets[0]),
            circuit,
        ),
        math.log(0.8),
    )

    sum_y = SumUnit(probabilistic_circuit=circuit)
    sum_y.add_subcircuit(
        leaf(
            UniformDistribution(variable=y, interval=closed(0, 1).simple_sets[0]),
            circuit,
        ),
        math.log(0.5),
    )
    sum_y.add_subcircuit(
        leaf(
            UniformDistribution(variable=y, interval=closed(1, 2).simple_sets[0]),
            circuit,
        ),
        math.log(0.5),
    )

    root.add_subcircuit(sum_x)
    root.add_subcircuit(sum_y)
    return circuit, x, y


def _marginal_probability(
    circuit: ProbabilisticCircuit,
    variable: Continuous,
    lower: float,
    upper: float,
) -> float:
    """
    Return P(variable ∈ [lower, upper]) from circuit.
    """
    event = (
        SimpleEvent.from_data({variable: closed(lower, upper)})
        .as_composite_set()
        .fill_missing_variables_pure(circuit.variables)
    )
    return float(circuit.probability(event))


class MarginalDeterminismTreeNodeLeafTestCase(unittest.TestCase):

    def test_single_variable_node_is_leaf(self):
        x = Continuous("x")
        self.assertTrue(
            MarginalDeterminismTreeNode(variables={x}, query_set={x}).is_leaf
        )

    def test_node_with_children_is_not_leaf(self):
        x, y = Continuous("x"), Continuous("y")
        parent = MarginalDeterminismTreeNode(variables={x, y}, query_set={x})
        MarginalDeterminismTreeNode(variables={x}, query_set={x}, parent_node=parent)
        MarginalDeterminismTreeNode(variables={y}, query_set={y}, parent_node=parent)
        self.assertFalse(parent.is_leaf)

    def test_node_with_empty_query_set_is_leaf(self):
        x = Continuous("x")
        self.assertTrue(
            MarginalDeterminismTreeNode(variables={x}, query_set=set()).is_leaf
        )


class MarginalDeterminismTreeNodeFindVariableTestCase(unittest.TestCase):

    def setUp(self):
        self.x, self.y = Continuous("x"), Continuous("y")
        self.root = MarginalDeterminismTreeNode(
            variables={self.x, self.y}, query_set={self.x}
        )
        MarginalDeterminismTreeNode(
            variables={self.x}, query_set={self.x}, parent_node=self.root
        )
        MarginalDeterminismTreeNode(
            variables={self.y}, query_set={self.y}, parent_node=self.root
        )

    def test_finds_variable_in_root_query_set(self):
        found = self.root.find_node_for_variable(self.x)
        self.assertIsNotNone(found)
        self.assertIn(self.x, found.query_set)

    def test_finds_variable_in_child_query_set(self):
        found = self.root.find_node_for_variable(self.y)
        self.assertIsNotNone(found)
        self.assertIn(self.y, found.query_set)

    def test_returns_none_for_absent_variable(self):
        self.assertIsNone(self.root.find_node_for_variable(Continuous("z")))

    def test_returns_shallowest_matching_node(self):
        found = self.root.find_node_for_variable(self.x)
        self.assertTrue(found.is_root)


class MarginalDeterminismTreeNodeAllQuerySetsTestCase(unittest.TestCase):

    def test_single_node_returns_one_query_set(self):
        x = Continuous("x")
        node = MarginalDeterminismTreeNode(variables={x}, query_set={x})
        all_sets = node.all_query_sets()
        self.assertEqual(len(all_sets), 1)
        self.assertIn(x, all_sets[0])

    def test_three_node_tree_returns_three_query_sets(self):
        x, y = Continuous("x"), Continuous("y")
        root = MarginalDeterminismTreeNode(variables={x, y}, query_set={x})
        MarginalDeterminismTreeNode(variables={x}, query_set={x}, parent_node=root)
        MarginalDeterminismTreeNode(variables={y}, query_set={y}, parent_node=root)
        self.assertEqual(len(root.all_query_sets()), 3)

    def test_empty_query_set_not_included(self):
        x = Continuous("x")
        self.assertEqual(
            len(
                MarginalDeterminismTreeNode(
                    variables={x}, query_set=set()
                ).all_query_sets()
            ),
            0,
        )

    def test_query_sets_returned_in_preorder(self):
        x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
        root = MarginalDeterminismTreeNode(variables={x, y, z}, query_set={x})
        MarginalDeterminismTreeNode(variables={y}, query_set={y}, parent_node=root)
        MarginalDeterminismTreeNode(variables={z}, query_set={z}, parent_node=root)
        self.assertIn(x, root.all_query_sets()[0])


class MarginalDeterminismTreeNodeFromCausalGraphTestCase(unittest.TestCase):

    def setUp(self):
        self.x = Continuous("x")
        self.y = Continuous("y")
        self.z = Continuous("z")
        self.effect = Continuous("effect")

    def test_root_query_set_is_first_in_priority_order(self):
        root = MarginalDeterminismTreeNode.from_causal_graph(
            [self.x, self.y, self.z],
            [self.effect],
            causal_priority_order=[self.z, self.x, self.y],
        )
        self.assertEqual(root.query_set, {self.z})

    def test_default_order_uses_causal_variables_order(self):
        root = MarginalDeterminismTreeNode.from_causal_graph(
            [self.x, self.y], [self.effect]
        )
        self.assertEqual(root.query_set, {self.x})

    def test_every_causal_variable_appears_in_some_query_set(self):
        root = MarginalDeterminismTreeNode.from_causal_graph(
            [self.x, self.y, self.z], [self.effect]
        )
        all_variables = set().union(*root.all_query_sets())
        for variable in [self.x, self.y, self.z]:
            self.assertIn(variable, all_variables)

    def test_five_variable_tree_assigns_all_variables(self):
        causal_variables = [Continuous(f"x{i}") for i in range(1, 6)]
        root = MarginalDeterminismTreeNode.from_causal_graph(
            causal_variables, [self.effect]
        )
        self.assertFalse(root.is_leaf)
        self.assertEqual(root.query_set, {causal_variables[0]})
        all_variables = set().union(*root.all_query_sets())
        for variable in causal_variables:
            self.assertIn(variable, all_variables)

    def test_effect_variables_not_assigned_to_any_query_set(self):
        root = MarginalDeterminismTreeNode.from_causal_graph([self.x, self.y], [self.z])
        all_variables = set().union(*root.all_query_sets())
        self.assertNotIn(self.z, all_variables)


class SupportDeterminismVerificationResultTestCase(unittest.TestCase):

    def test_passing_result_contains_pass_and_not_fail(self):
        x = Continuous("x")
        result = SupportDeterminismVerificationResult(
            passed=True, violations=[], checked_query_sets=[{x}], circuit_variables=[x]
        )
        self.assertIn("PASS", str(result))
        self.assertNotIn("FAIL", str(result))

    def test_failing_result_contains_fail_and_violation_text(self):
        x = Continuous("x")
        violation = OverlappingChildSupportsViolation(
            sum_unit_index=0, query_variable=x
        )
        result = SupportDeterminismVerificationResult(
            passed=False,
            violations=[violation],
            checked_query_sets=[{x}],
            circuit_variables=[x],
        )
        self.assertIn("FAIL", str(result))
        self.assertIn("overlapping", str(result))

    def test_string_contains_variable_name(self):
        x = Continuous("x")
        result = SupportDeterminismVerificationResult(
            passed=True, violations=[], checked_query_sets=[{x}], circuit_variables=[x]
        )
        self.assertIn("x", str(result))


class FailureDiagnosisResultTestCase(unittest.TestCase):

    def setUp(self):
        self.x = Continuous("x")
        self.y = Continuous("y")
        self.result = FailureDiagnosisResult(
            primary_cause_variable=self.x,
            actual_value=1.3,
            interventional_probability_at_failure=0.0,
            recommended_region=None,
            interventional_probability_at_recommendation=0.149,
            all_variable_results={
                self.x: {
                    "actual_value": 1.3,
                    "interventional_probability": 0.0,
                    "recommended_region": None,
                },
                self.y: {
                    "actual_value": 0.1,
                    "interventional_probability": 0.089,
                    "recommended_region": None,
                },
            },
        )

    def test_string_contains_primary_cause_marker(self):
        self.assertIn("PRIMARY", str(self.result))

    def test_string_contains_recommended_region(self):
        self.assertIn("Recommended region", str(self.result))

    def test_primary_cause_has_lowest_interventional_probability(self):
        lowest_probability = min(
            vr["interventional_probability"]
            for vr in self.result.all_variable_results.values()
        )
        self.assertEqual(
            self.result.interventional_probability_at_failure, lowest_probability
        )


class CausalCircuitConstructionTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_circuit()
        self.tree = MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y])

    def _make_causal_circuit(self) -> CausalCircuit:
        return CausalCircuit.from_probabilistic_circuit(
            self.circuit, self.tree, [self.x], [self.y]
        )

    def test_from_probabilistic_circuit_returns_causal_circuit(self):
        self.assertIsInstance(self._make_causal_circuit(), CausalCircuit)

    def test_causal_and_effect_variables_are_partitioned_correctly(self):
        cc = self._make_causal_circuit()
        self.assertIn(self.x, cc.causal_variables)
        self.assertIn(self.y, cc.effect_variables)
        self.assertNotIn(self.y, cc.causal_variables)
        self.assertNotIn(self.x, cc.effect_variables)

    def test_causal_variables_preserves_provided_list(self):
        self.assertEqual(self._make_causal_circuit().causal_variables, [self.x])

    def test_probabilistic_circuit_is_stored_by_identity(self):
        self.assertIs(self._make_causal_circuit().probabilistic_circuit, self.circuit)

    def test_marginal_determinism_tree_is_stored_by_identity(self):
        self.assertIs(self._make_causal_circuit().marginal_determinism_tree, self.tree)


class VerifySupportDeterminismVariableExistenceTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_circuit()

    def _make_causal_circuit(self, tree: MarginalDeterminismTreeNode) -> CausalCircuit:
        return CausalCircuit.from_probabilistic_circuit(
            self.circuit, tree, [self.x], [self.y]
        )

    def test_correct_circuit_passes_with_no_violations(self):
        tree = MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y])
        result = self._make_causal_circuit(tree).verify_support_determinism()
        self.assertTrue(result.passed)
        self.assertEqual(len(result.violations), 0)

    def test_query_variable_absent_from_circuit_fails_with_named_violation(self):
        ghost = Continuous("ghost")
        tree = MarginalDeterminismTreeNode(variables={self.x, ghost}, query_set={ghost})
        with self.assertRaises(SupportDeterminismVerificationResult) as ctx:
            self._make_causal_circuit(tree).verify_support_determinism()
        self.assertFalse(ctx.exception.passed)
        self.assertTrue(any("ghost" in str(v) for v in ctx.exception.violations))

    def test_result_lists_all_circuit_variables(self):
        tree = MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y])
        result = self._make_causal_circuit(tree).verify_support_determinism()
        self.assertIn(self.x, result.circuit_variables)
        self.assertIn(self.y, result.circuit_variables)

    def test_result_lists_at_least_one_checked_query_set(self):
        tree = MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y])
        result = self._make_causal_circuit(tree).verify_support_determinism()
        self.assertGreater(len(result.checked_query_sets), 0)


class VerifySupportDeterminismNormalizationTestCase(unittest.TestCase):

    def test_unnormalized_sum_unit_fails_with_log_weight_violation(self):
        circuit, x, y = _build_unnormalized_circuit()
        tree = MarginalDeterminismTreeNode.from_causal_graph([x], [y])
        cc = CausalCircuit.from_probabilistic_circuit(circuit, tree, [x], [y])
        with self.assertRaises(SupportDeterminismVerificationResult) as ctx:
            cc.verify_support_determinism()
        self.assertFalse(ctx.exception.passed)
        self.assertTrue(any("log-weights" in str(v) for v in ctx.exception.violations))


class VerifySupportDeterminismDisjointnessTestCase(unittest.TestCase):

    def test_correlated_circuit_with_clean_split_passes(self):
        circuit, x, w, y = _build_correlated_circuit()
        tree = MarginalDeterminismTreeNode.from_causal_graph([x, w], [y])
        cc = CausalCircuit.from_probabilistic_circuit(circuit, tree, [x, w], [y])
        result = cc.verify_support_determinism()
        self.assertTrue(result.passed, msg=f"Violations: {result.violations}")

    def test_independent_productunit_circuit_passes(self):
        circuit, x, y = _build_independent_circuit()
        tree = MarginalDeterminismTreeNode.from_causal_graph([x], [y])
        cc = CausalCircuit.from_probabilistic_circuit(circuit, tree, [x], [y])
        result = cc.verify_support_determinism()
        self.assertTrue(result.passed, msg=f"Violations: {result.violations}")


class BackdoorAdjustmentStructuralTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y]),
            [self.x],
            [self.y],
        )

    def test_returns_probabilistic_circuit(self):
        self.assertIsInstance(
            self.cc.backdoor_adjustment(self.x, self.y, []), ProbabilisticCircuit
        )

    def test_returned_circuit_contains_variables(self):
        self.assertGreater(
            len(self.cc.backdoor_adjustment(self.x, self.y, []).variables), 0
        )

    def test_unregistered_cause_variable_raises_error(self):
        with self.assertRaises(UnregisteredVariableError):
            self.cc.backdoor_adjustment(self.y, self.y, [])

    def test_unregistered_effect_variable_raises_error(self):
        with self.assertRaises(UnregisteredVariableError):
            self.cc.backdoor_adjustment(self.x, self.x, [])

    def test_explicit_empty_and_default_adjustment_produce_identical_result(self):
        explicit = self.cc.backdoor_adjustment(self.x, self.y, [])
        default = self.cc.backdoor_adjustment(self.x, self.y)
        self.assertAlmostEqual(
            _marginal_probability(explicit, self.y, 0, 1),
            _marginal_probability(default, self.y, 0, 1),
            delta=1e-6,
        )


class BackdoorAdjustmentIndependentCorrectnessTestCase(unittest.TestCase):
    """
    When P(x, y) = P(x) P(y), the backdoor criterion gives
    P(y | do(x)) = P(y) for all x in the support. The interventional
    distribution must equal the observational marginal of y.
    """

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y]),
            [self.x],
            [self.y],
        )
        self.interventional_circuit = self.cc.backdoor_adjustment(self.x, self.y, [])

    def test_interventional_distribution_integrates_to_one(self):
        self.assertAlmostEqual(
            _marginal_probability(self.interventional_circuit, self.y, 0, 2),
            1.0,
            delta=0.01,
        )

    def test_interventional_matches_observational_marginal(self):
        observational = _marginal_probability(self.circuit, self.y, 0, 1)
        interventional = _marginal_probability(
            self.interventional_circuit, self.y, 0, 1
        )
        self.assertAlmostEqual(observational, 0.5, delta=0.01)
        self.assertAlmostEqual(interventional, 0.5, delta=0.05)

    def test_out_of_support_cause_value_has_zero_probability(self):
        self.assertAlmostEqual(
            _marginal_probability(self.interventional_circuit, self.x, 5, 6),
            0.0,
            delta=1e-6,
        )

    def test_out_of_support_effect_value_has_zero_probability(self):
        self.assertAlmostEqual(
            _marginal_probability(self.interventional_circuit, self.y, 5, 6),
            0.0,
            delta=1e-6,
        )


class BackdoorAdjustmentCorrelatedCorrectnessTestCase(unittest.TestCase):
    """
    In the correlated circuit, x and y are grouped into two equal-weight components.

    The interventional distribution marginalised over y should place all probability on
    y∈[0,0.4] ∪ y∈[9.6,10] and sum to 1.
    """

    @classmethod
    def setUpClass(cls):
        cls.circuit, cls.x, cls.w, cls.y = _build_correlated_circuit()
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([cls.x, cls.w], [cls.y]),
            [cls.x, cls.w],
            [cls.y],
        )
        cls.interventional_circuit = cls.cc.backdoor_adjustment(cls.x, cls.y, [])

    def test_both_effect_regions_have_positive_probability(self):
        probability_low = _marginal_probability(
            self.interventional_circuit, self.y, 0, 0.4
        )
        probability_high = _marginal_probability(
            self.interventional_circuit, self.y, 9.6, 10
        )
        self.assertGreater(probability_low, 0.0)
        self.assertGreater(probability_high, 0.0)

    def test_effect_regions_account_for_all_probability_mass(self):
        probability_low = _marginal_probability(
            self.interventional_circuit, self.y, 0, 0.4
        )
        probability_high = _marginal_probability(
            self.interventional_circuit, self.y, 9.6, 10
        )
        self.assertAlmostEqual(probability_low + probability_high, 1.0, delta=0.05)

    def test_interventional_distribution_integrates_to_one(self):
        self.assertAlmostEqual(
            _marginal_probability(self.interventional_circuit, self.y, 0, 10),
            1.0,
            delta=0.01,
        )


class BackdoorAdjustmentWithAdjustmentTestCase(unittest.TestCase):
    """
    In the confounded circuit, P(y∈[0,1] | x∈[0,1]) = 1.0 spuriously.
    Adjusting for z recovers the causal truth P(y∈[0,1] | do(x)) = 0.5.
    On independent data, adjusting must not change the result.
    """

    @classmethod
    def setUpClass(cls):
        cls.circuit, cls.x, cls.y, cls.z = _build_confounded_circuit()
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([cls.x], [cls.y]),
            [cls.x],
            [cls.y],
        )

    def test_adjusted_distribution_integrates_to_one(self):
        adjusted = self.cc.backdoor_adjustment(self.x, self.y, [self.z])
        self.assertAlmostEqual(
            _marginal_probability(adjusted, self.y, 0, 2), 1.0, delta=0.01
        )

    def test_adjustment_recovers_causal_probability(self):
        adjusted = self.cc.backdoor_adjustment(self.x, self.y, [self.z])
        self.assertAlmostEqual(
            _marginal_probability(adjusted, self.y, 0, 1), 0.5, delta=0.1
        )

    def test_adjustment_on_independent_data_does_not_change_result(self):
        circuit, x, y, z = _build_three_variable_circuit()
        cc = CausalCircuit.from_probabilistic_circuit(
            circuit,
            MarginalDeterminismTreeNode.from_causal_graph([x, y], [z]),
            [x, y],
            [z],
        )
        without_adjustment = cc.backdoor_adjustment(x, z, [])
        with_adjustment = cc.backdoor_adjustment(x, z, [y])
        self.assertAlmostEqual(
            _marginal_probability(without_adjustment, z, 0, 1), 0.8, delta=0.05
        )
        self.assertAlmostEqual(
            _marginal_probability(with_adjustment, z, 0, 1), 0.8, delta=0.05
        )


class Season(Enum):
    WARM = auto()
    COLD = auto()


class Treatment(Enum):
    LOW = auto()
    HIGH = auto()


class Outcome(Enum):
    GOOD = auto()
    BAD = auto()


def _build_discrete_confounded_circuit() -> tuple:
    """
    Circuit with a discrete confounder (season) driving both a discrete cause
    (treatment) and the effect (outcome), where treatment has no causal effect of
    its own -- unlike `_build_confounded_circuit`'s continuous, deterministically
    branch-tied cause, treatment's own distribution *overlaps* across confounder
    states (both values have positive probability in both seasons), the shape that
    exposes the union-coalescing `_split_into_atomic_values` exists to undo.

    Two equal-weight strata:
        Warm stratum (p=0.6): P(treatment=HIGH)=0.8, outcome=GOOD (deterministic)
        Cold stratum (p=0.4): P(treatment=HIGH)=0.3, outcome=BAD (deterministic)

    Ground truth:
        P(outcome=GOOD | treatment=HIGH) = 0.8 -- spurious, driven by season.
        P(outcome=GOOD | do(treatment=HIGH)) = 0.6 -- causal truth after adjusting
            for season; equal to P(outcome=GOOD | do(treatment=LOW)), since
            treatment has no real effect on outcome.
    """
    season = Symbolic("season", domain=Set.from_iterable(Season))
    treatment = Symbolic("treatment", domain=Set.from_iterable(Treatment))
    outcome = Symbolic("outcome", domain=Set.from_iterable(Outcome))

    circuit = ProbabilisticCircuit()
    root = SumUnit(probabilistic_circuit=circuit)
    for season_value, treatment_high_probability, stratum_weight, outcome_value in [
        (Season.WARM, 0.8, 0.6, Outcome.GOOD),
        (Season.COLD, 0.3, 0.4, Outcome.BAD),
    ]:
        component = ProductUnit(probabilistic_circuit=circuit)
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=season,
                    probabilities=MissingDict(float, {hash(season_value): 1.0}),
                ),
                circuit,
            )
        )
        treatment_distribution = SumUnit(probabilistic_circuit=circuit)
        treatment_distribution.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=treatment,
                    probabilities=MissingDict(float, {hash(Treatment.HIGH): 1.0}),
                ),
                circuit,
            ),
            math.log(treatment_high_probability),
        )
        treatment_distribution.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=treatment,
                    probabilities=MissingDict(float, {hash(Treatment.LOW): 1.0}),
                ),
                circuit,
            ),
            math.log(1 - treatment_high_probability),
        )
        component.add_subcircuit(treatment_distribution)
        component.add_subcircuit(
            leaf(
                SymbolicDistribution(
                    variable=outcome,
                    probabilities=MissingDict(float, {hash(outcome_value): 1.0}),
                ),
                circuit,
            )
        )
        root.add_subcircuit(component, math.log(stratum_weight))

    return circuit, treatment, season, outcome


def _symbolic_probability(circuit: ProbabilisticCircuit, variable, value) -> float:
    """
    :return: P(variable == value) from circuit.
    """
    event = (
        SimpleEvent.from_data({variable: Set.from_iterable([value])})
        .as_composite_set()
        .fill_missing_variables_pure(circuit.variables)
    )
    return float(circuit.probability(event))


class DiscreteConfounderAdjustmentTestCase(unittest.TestCase):
    """
    `_build_discrete_confounded_circuit`'s treatment has no causal effect on outcome,
    only a spurious correlation through the discrete confounder season. Regression
    coverage for the union-coalescing bug `_split_into_atomic_values` fixes in
    `_extract_disjoint_regions_for_variable` and `_extract_leaf_regions_for_variables`,
    and for the cause-region weighting fix in.

    `_add_region_for_cause_value` (weighting by P(cause=v), not P(cause=v | stratum)
    * P(stratum), so a discrete adjustment set is not just a structural no-op).
    """

    @classmethod
    def setUpClass(cls):
        cls.circuit, cls.treatment, cls.season, cls.outcome = (
            _build_discrete_confounded_circuit()
        )
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MarginalDeterminismTreeNode.from_causal_graph(
                [cls.treatment, cls.season], [cls.outcome]
            ),
            [cls.treatment, cls.season],
            [cls.outcome],
        )

    def test_conditioning_is_confounded(self):
        conditioned, _ = self.circuit.truncated(
            SimpleEvent.from_data({self.treatment: Set.from_iterable([Treatment.HIGH])})
            .as_composite_set()
            .fill_missing_variables_pure(self.circuit.variables)
        )
        self.assertAlmostEqual(
            _symbolic_probability(conditioned, self.outcome, Outcome.GOOD),
            0.8,
            delta=0.01,
        )

    def test_intervention_without_adjustment_matches_conditioning(self):
        # An empty adjustment set cannot separate the confound from the effect --
        # it should reproduce the same spurious 0.8, not the causal 0.6.
        naive = self.cc.backdoor_adjustment(self.treatment, self.outcome)
        narrowed, _ = naive.truncated(
            SimpleEvent.from_data({self.treatment: Set.from_iterable([Treatment.HIGH])})
            .as_composite_set()
            .fill_missing_variables_pure(naive.variables)
        )
        self.assertAlmostEqual(
            _symbolic_probability(narrowed, self.outcome, Outcome.GOOD),
            0.8,
            delta=0.01,
        )

    def test_adjustment_recovers_causal_probability_for_each_cause_value(self):
        adjusted = self.cc.backdoor_adjustment(
            self.treatment, self.outcome, [self.season]
        )
        for treatment_value in (Treatment.HIGH, Treatment.LOW):
            with self.subTest(treatment=treatment_value):
                narrowed, region_probability = adjusted.truncated(
                    SimpleEvent.from_data(
                        {self.treatment: Set.from_iterable([treatment_value])}
                    )
                    .as_composite_set()
                    .fill_missing_variables_pure(adjusted.variables)
                )
                self.assertGreater(region_probability, 0.0)
                self.assertAlmostEqual(
                    _symbolic_probability(narrowed, self.outcome, Outcome.GOOD),
                    0.6,
                    delta=0.01,
                )

    def test_adjusted_distribution_integrates_to_one(self):
        adjusted = self.cc.backdoor_adjustment(
            self.treatment, self.outcome, [self.season]
        )
        total = _symbolic_probability(
            adjusted, self.outcome, Outcome.GOOD
        ) + _symbolic_probability(adjusted, self.outcome, Outcome.BAD)
        self.assertAlmostEqual(total, 1.0, delta=0.01)

    def test_adjusted_region_mass_matches_the_marginal_cause_probability(self):
        adjusted = self.cc.backdoor_adjustment(
            self.treatment, self.outcome, [self.season]
        )
        _, high_mass = adjusted.truncated(
            SimpleEvent.from_data({self.treatment: Set.from_iterable([Treatment.HIGH])})
            .as_composite_set()
            .fill_missing_variables_pure(adjusted.variables)
        )
        # P(treatment=HIGH) = P(WARM)*0.8 + P(COLD)*0.3 = 0.6*0.8 + 0.4*0.3 = 0.6
        self.assertAlmostEqual(high_mass, 0.6, delta=0.01)


class SplitIntoAtomicValuesTestCase(unittest.TestCase):
    """
    `_extract_disjoint_regions_for_variable` and `_extract_leaf_regions_for_variables`
    both rely on `_split_into_atomic_values` to break a discrete union value (several of
    a Symbolic variable's values, all with positive probability in one SumUnit branch)
    into its individual elements.
    """

    def test_single_value_is_returned_unchanged(self):
        self.assertEqual(
            CausalCircuit._split_into_atomic_values(Treatment.HIGH), [Treatment.HIGH]
        )

    def test_union_of_values_is_split_into_elements(self):
        circuit, treatment, _, _ = _build_discrete_confounded_circuit()
        union_value = next(
            simple_region[treatment] for simple_region in circuit.support.simple_sets
        )
        split_values = CausalCircuit._split_into_atomic_values(union_value)
        self.assertEqual(len(split_values), 2)
        self.assertEqual(
            {value.element for value in split_values}, {Treatment.HIGH, Treatment.LOW}
        )

    def test_non_set_value_without_simple_sets_is_returned_unchanged(self):
        self.assertEqual(CausalCircuit._split_into_atomic_values(0.5), [0.5])


class ExtractLeafRegionsTestCase(unittest.TestCase):
    """
    _extract_leaf_regions_for_variable underpins both backdoor_adjustment and
    diagnose_failure.

    Testing it directly ensures the probability decomposition is sound before higher-
    level logic runs.
    """

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y]),
            [self.x],
            [self.y],
        )

    def test_at_least_one_region_returned_for_cause_variable(self):
        self.assertGreaterEqual(
            len(self.cc._extract_leaf_regions_for_variable(self.x)), 1
        )

    def test_at_least_one_region_returned_for_effect_variable(self):
        self.assertGreaterEqual(
            len(self.cc._extract_leaf_regions_for_variable(self.y)), 1
        )

    def test_cause_region_probabilities_sum_to_one(self):
        regions = self.cc._extract_leaf_regions_for_variable(self.x)
        self.assertAlmostEqual(
            sum(region.probability for region in regions), 1.0, delta=0.01
        )

    def test_effect_region_probabilities_sum_to_one(self):
        regions = self.cc._extract_leaf_regions_for_variable(self.y)
        self.assertAlmostEqual(
            sum(region.probability for region in regions), 1.0, delta=0.01
        )

    def test_all_region_probabilities_are_positive(self):
        for variable in [self.x, self.y]:
            for region in self.cc._extract_leaf_regions_for_variable(variable):
                self.assertGreater(region.probability, 0.0)

    def test_regions_are_returned_as_event_probability_pairs(self):
        for region in self.cc._extract_leaf_regions_for_variable(self.x):
            self.assertIsInstance(region.probability, float)
            self.assertTrue(hasattr(region.event, "simple_sets"))


class ExtractDisjointRegionsTestCase(unittest.TestCase):
    """
    _extract_disjoint_regions_for_variable is the disjoint counterpart of
    _extract_leaf_regions_for_variable: it separates support regions coming from
    different SumUnit branches instead of collapsing them into the variable's whole
    marginal support.
    """

    def setUp(self):
        self.circuit, self.x, self.w, self.y = _build_correlated_circuit()
        self.causal_circuit = CausalCircuit.from_probabilistic_circuit(
            self.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x, self.w], [self.y]),
            [self.x, self.w],
            [self.y],
        )

    def test_finds_both_branches_separately_unlike_the_marginalized_version(self):
        disjoint_regions = self.causal_circuit._extract_disjoint_regions_for_variable(
            self.x
        )
        coarsened_regions = self.causal_circuit._extract_leaf_regions_for_variable(
            self.x
        )
        self.assertEqual(len(disjoint_regions), 2)
        self.assertEqual(len(coarsened_regions), 1)

    def test_region_probabilities_sum_to_one(self):
        regions = self.causal_circuit._extract_disjoint_regions_for_variable(self.x)
        self.assertAlmostEqual(
            sum(region.probability for region in regions), 1.0, delta=0.01
        )

    def test_all_region_probabilities_are_positive(self):
        for region in self.causal_circuit._extract_disjoint_regions_for_variable(
            self.x
        ):
            self.assertGreater(region.probability, 0.0)

    def test_regions_are_returned_as_event_probability_pairs(self):
        for region in self.causal_circuit._extract_disjoint_regions_for_variable(
            self.x
        ):
            self.assertIsInstance(region.probability, float)
            self.assertTrue(hasattr(region.event, "simple_sets"))

    def test_regions_are_each_roughly_equal_weight(self):
        # x's two branches are equal-weight (0.5 each) and non-overlapping.
        probabilities = sorted(
            region.probability
            for region in self.causal_circuit._extract_disjoint_regions_for_variable(
                self.x
            )
        )
        self.assertAlmostEqual(probabilities[0], 0.5, delta=0.05)
        self.assertAlmostEqual(probabilities[1], 0.5, delta=0.05)

    def test_matches_the_marginalized_version_on_a_circuit_with_one_true_region(self):
        # on a circuit whose cause variable genuinely has one contiguous support
        # region (no branches to keep separate), both methods must agree.
        circuit, x, y = _build_independent_circuit()
        causal_circuit = CausalCircuit.from_probabilistic_circuit(
            circuit, MarginalDeterminismTreeNode.from_causal_graph([x], [y]), [x], [y]
        )
        disjoint_regions = causal_circuit._extract_disjoint_regions_for_variable(y)
        coarsened_regions = causal_circuit._extract_leaf_regions_for_variable(y)
        self.assertEqual(len(disjoint_regions), len(coarsened_regions))


class BestDisjointRegionTestCase(unittest.TestCase):
    """
    _best_disjoint_region can distinguish which SumUnit branch's region best explains an
    interventional distribution -- unlike _best_region, whose regions always collapse to
    the variable's whole domain (see ExtractDisjointRegionsTestCase).
    """

    def setUp(self):
        self.circuit, self.x, self.w, self.y = _build_correlated_circuit()
        self.causal_circuit = CausalCircuit.from_probabilistic_circuit(
            self.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x, self.w], [self.y]),
            [self.x, self.w],
            [self.y],
        )
        self.interventional = self.causal_circuit.backdoor_adjustment(
            self.x, self.y, []
        )

    def _truncated_to_y(self, lower: float, upper: float):
        event = SimpleEvent.from_data({self.y: closed(lower, upper)}).as_composite_set()
        truncated, _ = self.interventional.truncated(
            event.fill_missing_variables_pure(self.interventional.variables)
        )
        return truncated

    def test_best_region_is_near_certain_once_truncated_to_one_branchs_effect(self):
        truncated = self._truncated_to_y(9.6, 10)
        best_region = self.causal_circuit._best_disjoint_region(self.x, truncated)
        self.assertAlmostEqual(
            truncated.probability(
                best_region.fill_missing_variables_pure(truncated.variables)
            ),
            1.0,
            delta=0.01,
        )

    def test_best_region_differs_for_different_effect_conditions(self):
        low_best = self.causal_circuit._best_disjoint_region(
            self.x, self._truncated_to_y(0, 0.4)
        )
        high_best = self.causal_circuit._best_disjoint_region(
            self.x, self._truncated_to_y(9.6, 10)
        )
        self.assertNotEqual(low_best, high_best)


class DiagnoseFailureStructuralTestCase(unittest.TestCase):

    def setUp(self):
        circuit, self.x, self.y = _build_independent_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y]),
            [self.x],
            [self.y],
        )

    def test_returns_failure_diagnosis_result(self):
        self.assertIsInstance(
            self.cc.diagnose_failure({self.x: 0.5}, self.y), FailureDiagnosisResult
        )

    def test_primary_cause_variable_is_a_continuous_variable(self):
        self.assertIsInstance(
            self.cc.diagnose_failure({self.x: 0.5}, self.y).primary_cause_variable,
            Continuous,
        )

    def test_actual_value_matches_observed_input(self):
        self.assertAlmostEqual(
            self.cc.diagnose_failure({self.x: 0.5}, self.y).actual_value,
            0.5,
            delta=1e-6,
        )

    def test_all_variable_results_are_keyed_by_variable_objects(self):
        result = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        self.assertIn(self.x, result.all_variable_results)
        for key in result.all_variable_results:
            self.assertIsInstance(key, Continuous)

    def test_all_variable_results_contain_required_fields(self):
        result = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        for variable_result in result.all_variable_results.values():
            for field in (
                "actual_value",
                "interventional_probability",
                "recommended_region",
            ):
                self.assertIn(field, variable_result)

    def test_empty_observed_values_raises_error(self):
        with self.assertRaises(NoCauseVariablesError):
            self.cc.diagnose_failure({}, self.y)

    def test_primary_cause_has_the_minimum_interventional_probability(self):
        result = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        minimum_probability = min(
            vr["interventional_probability"]
            for vr in result.all_variable_results.values()
        )
        self.assertAlmostEqual(
            result.interventional_probability_at_failure,
            minimum_probability,
            delta=1e-6,
        )

    def test_recommendation_probability_is_non_negative(self):
        self.assertGreaterEqual(
            self.cc.diagnose_failure(
                {self.x: 0.5}, self.y
            ).interventional_probability_at_recommendation,
            0.0,
        )

    def test_in_domain_observation_yields_positive_recommendation_probability(self):
        self.assertGreater(
            self.cc.diagnose_failure(
                {self.x: 0.5}, self.y
            ).interventional_probability_at_recommendation,
            0.0,
        )


class DiagnoseFailureCorrectnessTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.circuit, cls.x, cls.w, cls.y = _build_correlated_circuit()
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([cls.x, cls.w], [cls.y]),
            [cls.x, cls.w],
            [cls.y],
        )

    def test_out_of_domain_cause_is_identified_as_primary(self):
        self.assertEqual(
            self.cc.diagnose_failure(
                {self.x: 5.0, self.w: 0.5}, self.y
            ).primary_cause_variable,
            self.x,
        )

    def test_out_of_domain_cause_has_zero_interventional_probability(self):
        self.assertAlmostEqual(
            self.cc.diagnose_failure(
                {self.x: 5.0, self.w: 0.5}, self.y
            ).interventional_probability_at_failure,
            0.0,
            delta=1e-6,
        )

    def test_in_domain_variable_has_positive_interventional_probability(self):
        result = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertGreater(
            result.all_variable_results[self.w]["interventional_probability"], 0.0
        )

    def test_all_in_domain_causes_have_positive_interventional_probability(self):
        result = self.cc.diagnose_failure({self.x: 0.5, self.w: 0.5}, self.y)
        for variable, variable_result in result.all_variable_results.items():
            self.assertGreater(
                variable_result["interventional_probability"],
                0.0,
                msg=f"Expected positive probability for {variable.name}",
            )

    def test_recommended_value_improves_over_observed_failure(self):
        result = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertGreater(
            result.interventional_probability_at_recommendation,
            result.interventional_probability_at_failure,
        )

    def test_recommended_value_is_within_training_support(self):
        result = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertGreater(result.interventional_probability_at_recommendation, 0.0)

    def test_recommended_region_has_simple_sets(self):
        result = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertIsNotNone(result.recommended_region)
        self.assertTrue(hasattr(result.recommended_region, "simple_sets"))

    def test_string_output_identifies_primary_cause(self):
        result = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertIn("PRIMARY", str(result))


class EndToEndIntegrationTestCase(unittest.TestCase):
    """
    Exercises the full pipeline — tree construction, support determinism verification,
    backdoor adjustment, and failure diagnosis — on the three-variable independent
    circuit and the correlated circuit.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)

        (
            cls.circuit_independent,
            cls.x_independent,
            cls.y_independent,
            cls.z_independent,
        ) = _build_three_variable_circuit()

        cls.cc_independent = CausalCircuit.from_probabilistic_circuit(
            cls.circuit_independent,
            MarginalDeterminismTreeNode.from_causal_graph(
                [cls.x_independent, cls.y_independent],
                [cls.z_independent],
                causal_priority_order=[cls.x_independent, cls.y_independent],
            ),
            [cls.x_independent, cls.y_independent],
            [cls.z_independent],
        )

        (
            cls.circuit_correlated,
            cls.x_correlated,
            cls.w_correlated,
            cls.y_correlated,
        ) = _build_correlated_circuit()

        cls.cc_correlated = CausalCircuit.from_probabilistic_circuit(
            cls.circuit_correlated,
            MarginalDeterminismTreeNode.from_causal_graph(
                [cls.x_correlated, cls.w_correlated], [cls.y_correlated]
            ),
            [cls.x_correlated, cls.w_correlated],
            [cls.y_correlated],
        )

    def test_support_determinism_passes_for_independent_circuit(self):
        self.assertTrue(self.cc_independent.verify_support_determinism().passed)

    def test_support_determinism_passes_for_correlated_circuit(self):
        self.assertTrue(self.cc_correlated.verify_support_determinism().passed)

    def test_interventional_circuit_integrates_to_one(self):
        interventional_circuit = self.cc_independent.backdoor_adjustment(
            self.x_independent, self.z_independent, []
        )
        self.assertAlmostEqual(
            _marginal_probability(interventional_circuit, self.z_independent, 0, 2),
            1.0,
            delta=0.01,
        )

    def test_interventional_distribution_equals_observational_on_independent_data(self):
        interventional_circuit = self.cc_independent.backdoor_adjustment(
            self.x_independent, self.z_independent, []
        )
        self.assertAlmostEqual(
            _marginal_probability(interventional_circuit, self.z_independent, 0, 1),
            0.8,
            delta=0.05,
        )

    def test_failure_diagnosis_identifies_out_of_domain_primary_cause(self):
        result = self.cc_independent.diagnose_failure(
            {self.x_independent: 5.0, self.y_independent: 0.5}, self.z_independent
        )
        self.assertEqual(result.primary_cause_variable, self.x_independent)
        self.assertAlmostEqual(
            result.interventional_probability_at_failure, 0.0, delta=1e-6
        )

    def test_failure_diagnosis_in_domain_gives_non_negative_probabilities(self):
        result = self.cc_independent.diagnose_failure(
            {self.x_independent: 0.5, self.y_independent: 0.5}, self.z_independent
        )
        for variable_result in result.all_variable_results.values():
            self.assertGreaterEqual(variable_result["interventional_probability"], 0.0)

    def test_failure_diagnosis_string_identifies_primary_cause(self):
        result = self.cc_independent.diagnose_failure(
            {self.x_independent: 5.0, self.y_independent: 0.5}, self.z_independent
        )
        self.assertIn("PRIMARY", str(result))

    def test_failure_diagnosis_recommendation_improves_on_correlated_circuit(self):
        result = self.cc_correlated.diagnose_failure(
            {self.x_correlated: 5.0, self.w_correlated: 0.5}, self.y_correlated
        )
        self.assertGreater(
            result.interventional_probability_at_recommendation,
            result.interventional_probability_at_failure,
        )


if __name__ == "__main__":
    unittest.main()
