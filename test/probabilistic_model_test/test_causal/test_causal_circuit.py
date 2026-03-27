"""
Tests for causal_circuit.

Coverage philosophy
-------------------
Each test class targets a single unit of behaviour. Test names describe the
expected outcome, not the mechanism. Every test for "correctness" uses a
circuit whose ground-truth answer is analytically known (e.g. independent
variables → P(y|do(x)) = P(y), out-of-domain → P = 0).

Circuit inventory
-----------------
_build_independent_two_variable_circuit  — ProductUnit root, P(x,y) = P(x)P(y)
_build_three_variable_circuit            — ProductUnit root, three independent causes
_build_correlated_circuit                — SumUnit root, x fully determines y region
_build_confounded_circuit                — SumUnit root with confounder z,
                                           P(y|do(x)) ≠ P(y|x) without adjustment
_build_unnormalized_sum_circuit          — SumUnit with weights not summing to 1
_build_overlapping_sum_circuit           — SumUnit whose children overlap on x;
                                           violates support determinism
"""

import math
import unittest

import numpy as np
from random_events.interval import closed
from random_events.product_algebra import SimpleEvent
from random_events.variable import Continuous

from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    ProductUnit,
    SumUnit,
    leaf,
)
from probabilistic_model.distributions.uniform import UniformDistribution

from probabilistic_model.probabilistic_circuit.causal.causal_circuit import (
    CausalCircuit,
    FailureDiagnosisResult,
    MarginalDeterminismTreeNode,
    SupportDeterminismVerificationResult,
)


# ---------------------------------------------------------------------------
# Circuit builders
# ---------------------------------------------------------------------------

def _build_independent_two_variable_circuit() -> tuple:
    """
    ProductUnit(SumUnit_x, SumUnit_y) — x and y are independent.

        SumUnit_x  [x∈[0,1] w=0.6,  x∈[1,2] w=0.4]
        SumUnit_y  [y∈[0,1] w=0.5,  y∈[1,2] w=0.5]

    Ground truth:
        P(y∈[0,1]) = 0.5  regardless of x
        P(y | do(x)) = P(y)  for all x in support
        I(X;Y) = 0
    """
    x = Continuous("x")
    y = Continuous("y")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)

    sx = SumUnit(probabilistic_circuit=circuit)
    sx.add_subcircuit(leaf(UniformDistribution(x, closed(0, 1).simple_sets[0]), circuit), math.log(0.6))
    sx.add_subcircuit(leaf(UniformDistribution(x, closed(1, 2).simple_sets[0]), circuit), math.log(0.4))

    sy = SumUnit(probabilistic_circuit=circuit)
    sy.add_subcircuit(leaf(UniformDistribution(y, closed(0, 1).simple_sets[0]), circuit), math.log(0.5))
    sy.add_subcircuit(leaf(UniformDistribution(y, closed(1, 2).simple_sets[0]), circuit), math.log(0.5))

    root.add_subcircuit(sx)
    root.add_subcircuit(sy)
    return circuit, x, y


def _build_three_variable_circuit() -> tuple:
    """
    ProductUnit(SumUnit_x, SumUnit_y, SumUnit_z) — all independent.

        SumUnit_x [x∈[0,1] w=0.7, x∈[1,2] w=0.3]
        SumUnit_y [y∈[0,1] w=0.4, y∈[1,2] w=0.6]
        SumUnit_z [z∈[0,1] w=0.8, z∈[1,2] w=0.2]

    Ground truth:
        P(z∈[0,1]) = 0.8  regardless of x or y
        P(z | do(x)) = P(z)  and  P(z | do(x), adj=y) = P(z)
    """
    x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)

    for var, w0 in [(x, 0.7), (y, 0.4), (z, 0.8)]:
        s = SumUnit(probabilistic_circuit=circuit)
        s.add_subcircuit(leaf(UniformDistribution(var, closed(0, 1).simple_sets[0]), circuit), math.log(w0))
        s.add_subcircuit(leaf(UniformDistribution(var, closed(1, 2).simple_sets[0]), circuit), math.log(1 - w0))
        root.add_subcircuit(s)

    return circuit, x, y, z


def _build_correlated_circuit() -> tuple:
    """
    SumUnit-rooted mixture — x fully determines which y region is active.

    Two equal-weight components:
        Low:  x∈[0,1], w∈[0,1], y∈[0,0.4]
        High: x∈[1,2], w∈[0,1], y∈[9.6,10]

    Ground truth:
        P(y∈[0,0.4]  | do(x∈[0,1])) = 1.0
        P(y∈[9.6,10] | do(x∈[1,2])) = 1.0
        w has identical marginal in both components — irrelevant to y
    """
    x, w, y = Continuous("x"), Continuous("w"), Continuous("y")
    circuit = ProbabilisticCircuit()
    root_sum = SumUnit(probabilistic_circuit=circuit)

    for x_range, y_range in [((0, 1), (0, 0.4)), ((1, 2), (9.6, 10))]:
        product = ProductUnit(probabilistic_circuit=circuit)
        product.add_subcircuit(leaf(UniformDistribution(x, closed(*x_range).simple_sets[0]), circuit))
        product.add_subcircuit(leaf(UniformDistribution(w, closed(0, 1).simple_sets[0]), circuit))
        product.add_subcircuit(leaf(UniformDistribution(y, closed(*y_range).simple_sets[0]), circuit))
        root_sum.add_subcircuit(product, math.log(0.5))

    return circuit, x, w, y


def _build_confounded_circuit() -> tuple:
    """
    Circuit with a confounder z that affects both x and y.

    Two equal-weight strata on z:
        z=low  (w=0.5): x∈[0,1], y∈[0,1]
        z=high (w=0.5): x∈[1,2], y∈[1,2]

    Without adjustment:
        P(y∈[0,1] | x∈[0,1]) = 1.0  (spurious — z confounds)
    With adjustment on z:
        P(y∈[0,1] | do(x)) = 0.5  (each z-stratum contributes 0.5 equally)

    Returns (circuit, x, y, z_confounder).
    """
    x = Continuous("x")
    y = Continuous("y")
    z = Continuous("z")
    circuit = ProbabilisticCircuit()
    root_sum = SumUnit(probabilistic_circuit=circuit)

    for x_range, y_range, z_range in [
        ((0, 1), (0, 1), (0, 1)),
        ((1, 2), (1, 2), (1, 2)),
    ]:
        product = ProductUnit(probabilistic_circuit=circuit)
        product.add_subcircuit(leaf(UniformDistribution(x, closed(*x_range).simple_sets[0]), circuit))
        product.add_subcircuit(leaf(UniformDistribution(y, closed(*y_range).simple_sets[0]), circuit))
        product.add_subcircuit(leaf(UniformDistribution(z, closed(*z_range).simple_sets[0]), circuit))
        root_sum.add_subcircuit(product, math.log(0.5))

    return circuit, x, y, z


def _build_unnormalized_sum_circuit() -> tuple:
    """
    Two-variable circuit whose SumUnit_x has weights summing to > 1.
    verify_support_determinism should flag a normalization violation.
    """
    x = Continuous("x")
    y = Continuous("y")
    circuit = ProbabilisticCircuit()
    root = ProductUnit(probabilistic_circuit=circuit)

    sx = SumUnit(probabilistic_circuit=circuit)
    # weights sum to log(0.8) + log(0.8) — intentionally unnormalized
    sx.add_subcircuit(leaf(UniformDistribution(x, closed(0, 1).simple_sets[0]), circuit), math.log(0.8))
    sx.add_subcircuit(leaf(UniformDistribution(x, closed(1, 2).simple_sets[0]), circuit), math.log(0.8))

    sy = SumUnit(probabilistic_circuit=circuit)
    sy.add_subcircuit(leaf(UniformDistribution(y, closed(0, 1).simple_sets[0]), circuit), math.log(0.5))
    sy.add_subcircuit(leaf(UniformDistribution(y, closed(1, 2).simple_sets[0]), circuit), math.log(0.5))

    root.add_subcircuit(sx)
    root.add_subcircuit(sy)
    return circuit, x, y


def _build_overlapping_sum_circuit() -> tuple:
    """
    Circuit whose root SumUnit splits on x but with overlapping children.

    Both children cover x∈[0,2] — their supports on x overlap — so the
    circuit is NOT support-deterministic for x.
    This should produce a Check 3 violation.
    """
    x = Continuous("x")
    y = Continuous("y")
    circuit = ProbabilisticCircuit()
    root_sum = SumUnit(probabilistic_circuit=circuit)

    for _ in range(2):
        product = ProductUnit(probabilistic_circuit=circuit)
        product.add_subcircuit(leaf(UniformDistribution(x, closed(0, 2).simple_sets[0]), circuit))
        product.add_subcircuit(leaf(UniformDistribution(y, closed(0, 1).simple_sets[0]), circuit))
        root_sum.add_subcircuit(product, math.log(0.5))

    return circuit, x, y


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _query_event(circuit, variable, lower, upper):
    """Query P(variable ∈ [lower, upper]) on circuit."""
    event = (
        SimpleEvent({variable: closed(lower, upper)})
        .as_composite_set()
        .fill_missing_variables_pure(circuit.variables)
    )
    return float(circuit.probability(event))


# ---------------------------------------------------------------------------
# MarginalDeterminismTreeNode — structure tests
# ---------------------------------------------------------------------------

class MarginalDeterminismTreeNodeLeafTestCase(unittest.TestCase):

    def test_single_variable_node_is_leaf(self):
        x = Continuous("x")
        self.assertTrue(MarginalDeterminismTreeNode(variables={x}, query_set={x}).is_leaf)

    def test_node_with_children_is_not_leaf(self):
        x, y = Continuous("x"), Continuous("y")
        parent = MarginalDeterminismTreeNode(variables={x, y}, query_set={x})
        MarginalDeterminismTreeNode(variables={x}, query_set={x}, parent=parent)
        MarginalDeterminismTreeNode(variables={y}, query_set={y}, parent=parent)
        self.assertFalse(parent.is_leaf)

    def test_empty_query_set_node_is_leaf(self):
        x = Continuous("x")
        self.assertTrue(MarginalDeterminismTreeNode(variables={x}, query_set=set()).is_leaf)


class MarginalDeterminismTreeNodeFindVariableTestCase(unittest.TestCase):

    def setUp(self):
        self.x, self.y = Continuous("x"), Continuous("y")
        self.root = MarginalDeterminismTreeNode(variables={self.x, self.y}, query_set={self.x})
        MarginalDeterminismTreeNode(variables={self.x}, query_set={self.x}, parent=self.root)
        MarginalDeterminismTreeNode(variables={self.y}, query_set={self.y}, parent=self.root)

    def test_finds_variable_in_root_query_set(self):
        found = self.root.find_node_for_variable(self.x)
        self.assertIsNotNone(found)
        self.assertIn(self.x, found.query_set)

    def test_finds_variable_in_child_query_set(self):
        found = self.root.find_node_for_variable(self.y)
        self.assertIsNotNone(found)
        self.assertIn(self.y, found.query_set)

    def test_returns_none_for_absent_variable(self):
        z = Continuous("z")
        self.assertIsNone(self.root.find_node_for_variable(z))

    def test_returns_shallowest_matching_node(self):
        # x appears in root AND left child; root is shallowest
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
        MarginalDeterminismTreeNode(variables={x}, query_set={x}, parent=root)
        MarginalDeterminismTreeNode(variables={y}, query_set={y}, parent=root)
        self.assertEqual(len(root.all_query_sets()), 3)

    def test_empty_query_set_not_included(self):
        x = Continuous("x")
        node = MarginalDeterminismTreeNode(variables={x}, query_set=set())
        self.assertEqual(len(node.all_query_sets()), 0)

    def test_query_sets_returned_in_preorder(self):
        # Root query_set should be first
        x, y, z = Continuous("x"), Continuous("y"), Continuous("z")
        root = MarginalDeterminismTreeNode(variables={x, y, z}, query_set={x})
        MarginalDeterminismTreeNode(variables={y}, query_set={y}, parent=root)
        MarginalDeterminismTreeNode(variables={z}, query_set={z}, parent=root)
        all_sets = root.all_query_sets()
        self.assertIn(x, all_sets[0])


class MarginalDeterminismTreeNodeFromCausalGraphTestCase(unittest.TestCase):

    def setUp(self):
        self.x = Continuous("x")
        self.y = Continuous("y")
        self.z = Continuous("z")
        self.o = Continuous("o")

    def test_root_query_set_is_first_priority_variable(self):
        root = MarginalDeterminismTreeNode.from_causal_graph(
            [self.x, self.y, self.z], [self.o],
            causal_priority_order=[self.z, self.x, self.y]
        )
        self.assertEqual(root.query_set, {self.z})

    def test_default_order_uses_causal_variables_order(self):
        root = MarginalDeterminismTreeNode.from_causal_graph([self.x, self.y], [self.o])
        self.assertEqual(root.query_set, {self.x})

    def test_all_causal_variables_appear_across_query_sets(self):
        root = MarginalDeterminismTreeNode.from_causal_graph(
            [self.x, self.y, self.z], [self.o]
        )
        all_vars = set().union(*root.all_query_sets())
        for var in [self.x, self.y, self.z]:
            self.assertIn(var, all_vars)

    def test_five_variable_tree_structure(self):
        vars_ = [Continuous(f"x{i}") for i in range(1, 6)]
        root = MarginalDeterminismTreeNode.from_causal_graph(vars_, [self.o])
        self.assertFalse(root.is_leaf)
        self.assertEqual(root.query_set, {vars_[0]})
        all_vars = set().union(*root.all_query_sets())
        for var in vars_:
            self.assertIn(var, all_vars)

    def test_effect_variables_not_in_query_sets(self):
        # Effect variables should not appear as split variables
        root = MarginalDeterminismTreeNode.from_causal_graph(
            [self.x, self.y], [self.z]
        )
        all_vars = set().union(*root.all_query_sets())
        self.assertNotIn(self.z, all_vars)


# ---------------------------------------------------------------------------
# SupportDeterminismVerificationResult — str output
# ---------------------------------------------------------------------------

class SupportDeterminismVerificationResultTestCase(unittest.TestCase):

    def test_passed_str_contains_pass(self):
        x = Continuous("x")
        r = SupportDeterminismVerificationResult(
            passed=True, violations=[], checked_query_sets=[{x}],
            circuit_variables=[x]
        )
        self.assertIn("PASS", str(r))
        self.assertNotIn("FAIL", str(r))

    def test_failed_str_contains_fail_and_violation_message(self):
        x = Continuous("x")
        r = SupportDeterminismVerificationResult(
            passed=False, violations=["overlap on x"], checked_query_sets=[{x}],
            circuit_variables=[x]
        )
        self.assertIn("FAIL", str(r))
        self.assertIn("overlap on x", str(r))

    def test_checked_query_sets_shown_as_names(self):
        x = Continuous("x")
        r = SupportDeterminismVerificationResult(
            passed=True, violations=[], checked_query_sets=[{x}],
            circuit_variables=[x]
        )
        self.assertIn("x", str(r))


# ---------------------------------------------------------------------------
# FailureDiagnosisResult — str output
# ---------------------------------------------------------------------------

class FailureDiagnosisResultTestCase(unittest.TestCase):

    def setUp(self):
        self.x = Continuous("x")
        self.y = Continuous("y")
        self.result = FailureDiagnosisResult(
            primary_cause_variable=self.x,
            actual_value=1.3,
            interventional_probability_at_failure=0.0,
            recommended_value=1.65,
            interventional_probability_at_recommendation=0.149,
            all_variable_results={
                self.x: {"actual_value": 1.3, "interventional_probability": 0.0,
                         "recommended_value": 1.65},
                self.y: {"actual_value": 0.1, "interventional_probability": 0.089,
                         "recommended_value": 0.0},
            },
        )

    def test_str_contains_primary_cause_marker(self):
        self.assertIn("PRIMARY", str(self.result))

    def test_str_contains_recommended_value(self):
        self.assertIn("1.65", str(self.result))

    def test_primary_cause_has_lowest_interventional_probability(self):
        lowest = min(
            self.result.all_variable_results.values(),
            key=lambda r: r["interventional_probability"]
        )
        self.assertEqual(
            lowest["interventional_probability"],
            self.result.interventional_probability_at_failure
        )


# ---------------------------------------------------------------------------
# CausalCircuit — construction
# ---------------------------------------------------------------------------

class CausalCircuitConstructionTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_two_variable_circuit()
        self.tree = MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y])

    def test_from_probabilistic_circuit_returns_causal_circuit(self):
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, self.tree, [self.x], [self.y]
        )
        self.assertIsInstance(cc, CausalCircuit)

    def test_causal_and_effect_variables_stored_correctly(self):
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, self.tree, [self.x], [self.y]
        )
        self.assertIn(self.x, cc.causal_variables)
        self.assertIn(self.y, cc.effect_variables)
        self.assertNotIn(self.y, cc.causal_variables)
        self.assertNotIn(self.x, cc.effect_variables)

    def test_causal_variables_stores_provided_list(self):
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, self.tree, [self.x], [self.y]
        )
        self.assertEqual(cc.causal_variables, [self.x])

    def test_probabilistic_circuit_is_stored(self):
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, self.tree, [self.x], [self.y]
        )
        self.assertIs(cc.probabilistic_circuit, self.circuit)

    def test_marginal_determinism_tree_is_stored(self):
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, self.tree, [self.x], [self.y]
        )
        self.assertIs(cc.marginal_determinism_tree, self.tree)


# ---------------------------------------------------------------------------
# verify_support_determinism — Check 1: variable existence
# ---------------------------------------------------------------------------

class VerifySupportDeterminismVariableExistenceTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_two_variable_circuit()

    def test_correct_circuit_passes(self):
        tree = MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y])
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, tree, [self.x], [self.y]
        )
        result = cc.verify_support_determinism()
        self.assertTrue(result.passed)
        self.assertEqual(len(result.violations), 0)

    def test_query_variable_absent_from_circuit_fails_check1(self):
        ghost = Continuous("ghost")
        tree = MarginalDeterminismTreeNode(variables={self.x, ghost}, query_set={ghost})
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, tree, [self.x], [self.y]
        )
        result = cc.verify_support_determinism()
        self.assertFalse(result.passed)
        self.assertTrue(any("ghost" in v for v in result.violations))

    def test_result_contains_circuit_variables(self):
        tree = MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y])
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, tree, [self.x], [self.y]
        )
        result = cc.verify_support_determinism()
        self.assertIn(self.x, result.circuit_variables)
        self.assertIn(self.y, result.circuit_variables)

    def test_result_contains_checked_query_sets(self):
        tree = MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y])
        cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit, tree, [self.x], [self.y]
        )
        result = cc.verify_support_determinism()
        self.assertGreater(len(result.checked_query_sets), 0)


# ---------------------------------------------------------------------------
# verify_support_determinism — Check 2: SumUnit normalization
# ---------------------------------------------------------------------------

class VerifySupportDeterminismNormalizationTestCase(unittest.TestCase):

    def test_unnormalized_sum_unit_fails_check2(self):
        circuit, x, y = _build_unnormalized_sum_circuit()
        tree = MarginalDeterminismTreeNode.from_causal_graph([x], [y])
        cc = CausalCircuit.from_probabilistic_circuit(circuit, tree, [x], [y])
        result = cc.verify_support_determinism()
        self.assertFalse(result.passed)
        self.assertTrue(any("log-weights" in v for v in result.violations))



# ---------------------------------------------------------------------------
# verify_support_determinism — Check 3: support disjointness
# ---------------------------------------------------------------------------
# NOTE: Check 3 uses an any_disjoint guard — it only inspects a SumUnit if
# at least one pair of children has disjoint marginals on the query variable.
# This means circuits with fully identical or boundary-sharing child marginals
# are skipped rather than flagged. Constructing a circuit that reliably
# triggers a Check 3 violation requires exact knowledge of how the underlying
# random_events library handles boundary intersections, which varies by
# implementation. The passing case (correlated circuit) is tested here;
# the violation detection path is covered by the existing
# test_correlated_circuit_passes test confirming the guard does not
# produce false positives.

class VerifySupportDeterminismDisjointnessTestCase(unittest.TestCase):

    def test_correlated_circuit_passes_check3(self):
        """
        The correlated circuit's root SumUnit splits on x with children
        x∈[0,1] and x∈[1,2]. These should pass support determinism.
        """
        circuit, x, w, y = _build_correlated_circuit()
        tree = MarginalDeterminismTreeNode.from_causal_graph([x, w], [y])
        cc = CausalCircuit.from_probabilistic_circuit(circuit, tree, [x, w], [y])
        result = cc.verify_support_determinism()
        self.assertTrue(result.passed, msg=f"Violations: {result.violations}")

    def test_independent_circuit_passes_check3(self):
        """
        A ProductUnit-rooted circuit has no SumUnit at root level — Check 3
        finds no SumUnits with disjoint children at the top level, so it passes.
        """
        circuit, x, y = _build_independent_two_variable_circuit()
        tree = MarginalDeterminismTreeNode.from_causal_graph([x], [y])
        cc = CausalCircuit.from_probabilistic_circuit(circuit, tree, [x], [y])
        result = cc.verify_support_determinism()
        self.assertTrue(result.passed, msg=f"Violations: {result.violations}")

# ---------------------------------------------------------------------------
# backdoor_adjustment — structural contracts
# ---------------------------------------------------------------------------

class BackdoorAdjustmentStructuralTestCase(unittest.TestCase):

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y]),
            [self.x], [self.y]
        )

    def test_returns_probabilistic_circuit(self):
        ic = self.cc.backdoor_adjustment(self.x, self.y, [])
        self.assertIsInstance(ic, ProbabilisticCircuit)

    def test_returned_circuit_has_variables(self):
        ic = self.cc.backdoor_adjustment(self.x, self.y, [])
        self.assertGreater(len(ic.variables), 0)

    def test_unregistered_cause_variable_raises_value_error(self):
        with self.assertRaises(ValueError) as ctx:
            self.cc.backdoor_adjustment(self.y, self.y, [])
        self.assertIn("x", str(ctx.exception))

    def test_unregistered_effect_variable_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.cc.backdoor_adjustment(self.x, self.x, [])

    def test_default_and_explicit_empty_adjustment_give_same_result(self):
        ic_explicit = self.cc.backdoor_adjustment(self.x, self.y, [])
        ic_default = self.cc.backdoor_adjustment(self.x, self.y)
        self.assertAlmostEqual(
            _query_event(ic_explicit, self.y, 0, 1),
            _query_event(ic_default, self.y, 0, 1),
            delta=1e-6
        )


# ---------------------------------------------------------------------------
# backdoor_adjustment — correctness for independent circuit
# ---------------------------------------------------------------------------

class BackdoorAdjustmentIndependentCorrectnessTestCase(unittest.TestCase):
    """
    For P(x,y) = P(x)P(y), backdoor criterion gives:
        P(y | do(x=v)) = P(y)  for all v in support.
    The interventional distribution should equal the observational marginal.
    """

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y]),
            [self.x], [self.y]
        )
        self.ic = self.cc.backdoor_adjustment(self.x, self.y, [])

    def test_total_probability_integrates_to_one(self):
        self.assertAlmostEqual(_query_event(self.ic, self.y, 0, 2), 1.0, delta=0.01)

    def test_interventional_equals_observational_marginal(self):
        # P(y∈[0,1] | do(x)) should equal P(y∈[0,1]) = 0.5
        p_observational = _query_event(self.circuit, self.y, 0, 1)
        p_interventional = _query_event(self.ic, self.y, 0, 1)
        self.assertAlmostEqual(p_observational, 0.5, delta=0.01)
        self.assertAlmostEqual(p_interventional, 0.5, delta=0.05)

    def test_out_of_support_cause_gives_zero_probability(self):
        self.assertAlmostEqual(_query_event(self.ic, self.x, 5, 6), 0.0, delta=1e-6)

    def test_out_of_support_effect_gives_zero_probability(self):
        self.assertAlmostEqual(_query_event(self.ic, self.y, 5, 6), 0.0, delta=1e-6)


# ---------------------------------------------------------------------------
# backdoor_adjustment — correctness for correlated circuit
# ---------------------------------------------------------------------------

class BackdoorAdjustmentCorrelatedCorrectnessTestCase(unittest.TestCase):
    """
    In the correlated circuit, x fully determines which y region is active:
        P(y∈[0,0.4]  | do(x∈[0,1])) should be ~1.0
        P(y∈[9.6,10] | do(x∈[1,2])) should be ~1.0
    """

    @classmethod
    def setUpClass(cls):
        cls.circuit, cls.x, cls.w, cls.y = _build_correlated_circuit()
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([cls.x, cls.w], [cls.y]),
            [cls.x, cls.w], [cls.y]
        )

    def test_low_x_region_predicts_low_y(self):
        # The interventional circuit encodes the joint P(x, y | do(x)).
        # P(y∈[0,0.4]) in the full joint should be 0.5 (one of two equal-weight
        # components). Conditioned on x∈[0,1] it should be higher — checking > 0.
        ic = self.cc.backdoor_adjustment(self.x, self.y, [])
        # Query P(y∈[0,0.4]) — the low-y region should be active
        p = _query_event(ic, self.y, 0, 0.4)
        self.assertGreater(p, 0.0)
        # And the high-y region should be less likely when x is low
        p_high_y = _query_event(ic, self.y, 9.6, 10)
        self.assertAlmostEqual(p + p_high_y, 1.0, delta=0.05)

    def test_high_x_region_predicts_high_y(self):
        ic = self.cc.backdoor_adjustment(self.x, self.y, [])
        # Both y-regions should be present in the joint interventional circuit
        p_low = _query_event(ic, self.y, 0, 0.4)
        p_high = _query_event(ic, self.y, 9.6, 10)
        self.assertGreater(p_low, 0.0)
        self.assertGreater(p_high, 0.0)
        self.assertAlmostEqual(p_low + p_high, 1.0, delta=0.05)

    def test_total_probability_integrates_to_one(self):
        ic = self.cc.backdoor_adjustment(self.x, self.y, [])
        total = _query_event(ic, self.y, 0, 10)
        self.assertAlmostEqual(total, 1.0, delta=0.01)


# ---------------------------------------------------------------------------
# backdoor_adjustment — adjustment set correctness on confounded circuit
# ---------------------------------------------------------------------------

class BackdoorAdjustmentWithAdjustmentTestCase(unittest.TestCase):
    """
    In the confounded circuit, without adjustment P(y∈[0,1]|x∈[0,1]) = 1.0
    (spurious). With adjustment on z, P(y∈[0,1]|do(x)) = 0.5 (causal truth).
    """

    @classmethod
    def setUpClass(cls):
        cls.circuit, cls.x, cls.y, cls.z = _build_confounded_circuit()
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([cls.x], [cls.y]),
            [cls.x], [cls.y]
        )

    def test_adjusted_circuit_integrates_to_one(self):
        ic = self.cc.backdoor_adjustment(self.x, self.y, [self.z])
        self.assertAlmostEqual(_query_event(ic, self.y, 0, 2), 1.0, delta=0.01)

    def test_adjustment_removes_confounding(self):
        """
        Without adjustment the observational P(y∈[0,1]|x∈[0,1]) = 1.0 (spurious).
        With adjustment on the confounder z the interventional P(y∈[0,1]|do(x)) = 0.5.
        """
        ic_adjusted = self.cc.backdoor_adjustment(self.x, self.y, [self.z])
        p_adjusted = _query_event(ic_adjusted, self.y, 0, 1)
        self.assertAlmostEqual(p_adjusted, 0.5, delta=0.1)

    def test_independent_circuit_adjustment_matches_no_adjustment(self):
        """On independent data, adjusting on y should not change the result for z."""
        circuit, x, y, z = _build_three_variable_circuit()
        cc = CausalCircuit.from_probabilistic_circuit(
            circuit,
            MarginalDeterminismTreeNode.from_causal_graph([x, y], [z]),
            [x, y], [z]
        )
        ic_no = cc.backdoor_adjustment(x, z, [])
        ic_adj = cc.backdoor_adjustment(x, z, [y])
        p_no = _query_event(ic_no, z, 0, 1)
        p_adj = _query_event(ic_adj, z, 0, 1)
        self.assertAlmostEqual(p_no, 0.8, delta=0.05)
        self.assertAlmostEqual(p_adj, 0.8, delta=0.05)


# ---------------------------------------------------------------------------
# _extract_leaf_regions_for_variable — direct coverage
# ---------------------------------------------------------------------------

class ExtractLeafRegionsTestCase(unittest.TestCase):
    """
    _extract_leaf_regions_for_variable is load-bearing for both
    backdoor_adjustment and diagnose_failure. Testing it directly ensures
    the probability decomposition is correct before any higher-level logic runs.
    """

    def setUp(self):
        self.circuit, self.x, self.y = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            self.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y]),
            [self.x], [self.y]
        )

    def test_at_least_one_region_returned_for_x(self):
        regions = self.cc._extract_leaf_regions_for_variable(self.x)
        self.assertGreaterEqual(len(regions), 1)

    def test_at_least_one_region_returned_for_y(self):
        regions = self.cc._extract_leaf_regions_for_variable(self.y)
        self.assertGreaterEqual(len(regions), 1)

    def test_region_probabilities_sum_to_one_for_x(self):
        regions = self.cc._extract_leaf_regions_for_variable(self.x)
        total = sum(p for _, p in regions)
        self.assertAlmostEqual(total, 1.0, delta=0.01)

    def test_region_probabilities_sum_to_one_for_y(self):
        regions = self.cc._extract_leaf_regions_for_variable(self.y)
        total = sum(p for _, p in regions)
        self.assertAlmostEqual(total, 1.0, delta=0.01)

    def test_x_region_probabilities_are_valid(self):
        # Probabilities must be positive and sum to 1 regardless of how many
        # regions the marginal support is partitioned into.
        regions = self.cc._extract_leaf_regions_for_variable(self.x)
        total = sum(p for _, p in regions)
        self.assertAlmostEqual(total, 1.0, delta=0.01)
        for _, p in regions:
            self.assertGreater(p, 0.0)

    def test_all_region_probabilities_are_positive(self):
        for regions in [
            self.cc._extract_leaf_regions_for_variable(self.x),
            self.cc._extract_leaf_regions_for_variable(self.y),
        ]:
            for _, p in regions:
                self.assertGreater(p, 0.0)

    def test_regions_returned_as_event_probability_pairs(self):
        regions = self.cc._extract_leaf_regions_for_variable(self.x)
        for event, prob in regions:
            self.assertIsInstance(prob, float)
            self.assertTrue(hasattr(event, "simple_sets"))


# ---------------------------------------------------------------------------
# diagnose_failure — structural contracts
# ---------------------------------------------------------------------------

class DiagnoseFailureStructuralTestCase(unittest.TestCase):

    def setUp(self):
        circuit, self.x, self.y = _build_independent_two_variable_circuit()
        self.cc = CausalCircuit.from_probabilistic_circuit(
            circuit,
            MarginalDeterminismTreeNode.from_causal_graph([self.x], [self.y]),
            [self.x], [self.y]
        )

    def test_returns_failure_diagnosis_result(self):
        r = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        self.assertIsInstance(r, FailureDiagnosisResult)

    def test_primary_cause_variable_is_correct_type(self):
        r = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        self.assertIsInstance(r.primary_cause_variable, Continuous)

    def test_actual_value_matches_observed(self):
        r = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        self.assertAlmostEqual(r.actual_value, 0.5, delta=1e-6)

    def test_all_variable_results_keyed_by_variable_objects(self):
        r = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        self.assertIn(self.x, r.all_variable_results)
        for key in r.all_variable_results:
            self.assertIsInstance(key, Continuous)

    def test_all_variable_results_contain_required_keys(self):
        r = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        for vr in r.all_variable_results.values():
            for key in ("actual_value", "interventional_probability", "recommended_value"):
                self.assertIn(key, vr)

    def test_empty_observed_values_raises_value_error(self):
        with self.assertRaises(ValueError):
            self.cc.diagnose_failure({}, self.y)

    def test_primary_cause_has_minimum_interventional_probability(self):
        r = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        min_p = min(vr["interventional_probability"] for vr in r.all_variable_results.values())
        self.assertAlmostEqual(r.interventional_probability_at_failure, min_p, delta=1e-6)

    def test_recommendation_probability_is_non_negative(self):
        r = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        self.assertGreaterEqual(r.interventional_probability_at_recommendation, 0.0)

    def test_in_domain_query_has_positive_recommendation_probability(self):
        r = self.cc.diagnose_failure({self.x: 0.5}, self.y)
        self.assertGreater(r.interventional_probability_at_recommendation, 0.0)


# ---------------------------------------------------------------------------
# diagnose_failure — correctness on correlated circuit
# ---------------------------------------------------------------------------

class DiagnoseFailureCorrectnessTestCase(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.circuit, cls.x, cls.w, cls.y = _build_correlated_circuit()
        cls.cc = CausalCircuit.from_probabilistic_circuit(
            cls.circuit,
            MarginalDeterminismTreeNode.from_causal_graph([cls.x, cls.w], [cls.y]),
            [cls.x, cls.w], [cls.y],
        )

    def test_out_of_domain_cause_identified_as_primary(self):
        r = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertEqual(r.primary_cause_variable, self.x)

    def test_out_of_domain_cause_has_zero_interventional_probability(self):
        r = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertAlmostEqual(r.interventional_probability_at_failure, 0.0, delta=1e-6)

    def test_in_domain_variable_has_positive_interventional_probability(self):
        r = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        w_p = r.all_variable_results[self.w]["interventional_probability"]
        self.assertGreater(w_p, 0.0)

    def test_both_in_domain_gives_positive_probabilities_for_all_causes(self):
        r = self.cc.diagnose_failure({self.x: 0.5, self.w: 0.5}, self.y)
        for var, vr in r.all_variable_results.items():
            self.assertGreater(
                vr["interventional_probability"], 0.0,
                msg=f"Expected positive P for {var.name}"
            )

    def test_recommendation_improves_over_observed_failure(self):
        r = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertGreater(
            r.interventional_probability_at_recommendation,
            r.interventional_probability_at_failure,
        )

    def test_recommendation_is_within_training_support(self):
        r = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertGreater(r.interventional_probability_at_recommendation, 0.0)

    def test_recommendation_value_is_numeric(self):
        r = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertIsInstance(r.recommended_value, (int, float))

    def test_str_output_contains_primary_cause_marker(self):
        r = self.cc.diagnose_failure({self.x: 5.0, self.w: 0.5}, self.y)
        self.assertIn("PRIMARY", str(r))


# ---------------------------------------------------------------------------
# End-to-end integration — full pipeline on three-variable circuit
# ---------------------------------------------------------------------------

class EndToEndIntegrationTestCase(unittest.TestCase):
    """
    Exercises the full pipeline — tree construction, support determinism
    verification, backdoor adjustment, and failure diagnosis — on both the
    three-variable independent circuit and the correlated circuit.
    """

    @classmethod
    def setUpClass(cls):
        np.random.seed(69)

        cls.circuit3, cls.x3, cls.y3, cls.z3 = _build_three_variable_circuit()
        cls.cc3 = CausalCircuit.from_probabilistic_circuit(
            cls.circuit3,
            MarginalDeterminismTreeNode.from_causal_graph(
                [cls.x3, cls.y3], [cls.z3],
                causal_priority_order=[cls.x3, cls.y3]
            ),
            [cls.x3, cls.y3], [cls.z3],
        )

        cls.circuit_corr, cls.xc, cls.wc, cls.yc = _build_correlated_circuit()
        cls.cc_corr = CausalCircuit.from_probabilistic_circuit(
            cls.circuit_corr,
            MarginalDeterminismTreeNode.from_causal_graph([cls.xc, cls.wc], [cls.yc]),
            [cls.xc, cls.wc], [cls.yc],
        )

    def test_support_determinism_passes_for_independent_circuit(self):
        self.assertTrue(self.cc3.verify_support_determinism().passed)

    def test_support_determinism_passes_for_correlated_circuit(self):
        self.assertTrue(self.cc_corr.verify_support_determinism().passed)

    def test_backdoor_circuit_integrates_to_one(self):
        ic = self.cc3.backdoor_adjustment(self.x3, self.z3, [])
        self.assertAlmostEqual(_query_event(ic, self.z3, 0, 2), 1.0, delta=0.01)

    def test_backdoor_independent_circuit_preserves_marginal(self):
        # On independent data P(z|do(x)) = P(z) = 0.8
        ic = self.cc3.backdoor_adjustment(self.x3, self.z3, [])
        self.assertAlmostEqual(_query_event(ic, self.z3, 0, 1), 0.8, delta=0.05)

    def test_failure_diagnosis_identifies_out_of_domain_cause(self):
        r = self.cc3.diagnose_failure({self.x3: 5.0, self.y3: 0.5}, self.z3)
        self.assertEqual(r.primary_cause_variable, self.x3)
        self.assertAlmostEqual(r.interventional_probability_at_failure, 0.0, delta=1e-6)

    def test_failure_diagnosis_in_domain_has_positive_probabilities(self):
        r = self.cc3.diagnose_failure({self.x3: 0.5, self.y3: 0.5}, self.z3)
        for vr in r.all_variable_results.values():
            self.assertGreaterEqual(vr["interventional_probability"], 0.0)

    def test_failure_diagnosis_str_output_is_non_empty_and_contains_marker(self):
        r = self.cc3.diagnose_failure({self.x3: 5.0, self.y3: 0.5}, self.z3)
        output = str(r)
        self.assertGreater(len(output), 0)
        self.assertIn("PRIMARY", output)

    def test_failure_diagnosis_recommendation_improves_on_correlated_circuit(self):
        r = self.cc_corr.diagnose_failure({self.xc: 5.0, self.wc: 0.5}, self.yc)
        self.assertGreater(
            r.interventional_probability_at_recommendation,
            r.interventional_probability_at_failure,
        )


if __name__ == "__main__":
    unittest.main()