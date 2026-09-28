"""
Tests for the ways a circuit is fitted on a dataframe.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest
from random_events.product_algebra import SimpleEvent

from probabilistic_model.learning.jpt.variables import infer_variables_from_dataframe
from probabilistic_model.learning.jpt.jpt import JointProbabilityTree
from probabilistic_model.learning.learning_method import StratifiedLearning
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    SumUnit,
)


@pytest.fixture
def two_groups() -> pd.DataFrame:
    """
    Forty rows in two groups of a discrete column, each group with its own continuous
    value range.
    """
    generator = np.random.default_rng(0)
    return pd.DataFrame(
        {
            "group": [1] * 30 + [3] * 10,
            "value": np.concatenate(
                [generator.normal(0.0, 1.0, 30), generator.normal(10.0, 1.0, 10)]
            ),
        }
    )


def _tree_leaf_count(tree_root: SumUnit) -> int:
    """
    :param tree_root: The root sum a joint probability tree fit produced.
    :return: How many leaves the tree grew: the product units its root sum mixes.
    """
    return len(tree_root.subcircuits)


def _branch_circuit(branch: SumUnit) -> ProbabilisticCircuit:
    """
    :param branch: A unit of another circuit.
    :return: A circuit of its own holding a copy of the branch.
    """
    circuit = ProbabilisticCircuit()
    circuit.mount(branch)
    return circuit


def test_a_joint_probability_tree_fitted_with_given_variables_honours_the_leaf_minimum(
    two_groups,
):
    variables = infer_variables_from_dataframe(two_groups)

    circuit = JointProbabilityTree(min_samples_per_leaf=len(two_groups)).fit(
        two_groups, variables
    )

    assert _tree_leaf_count(circuit.root) == 1


def test_a_tree_without_variables_infers_them_from_the_data(two_groups):
    circuit = JointProbabilityTree().fit(two_groups)

    assert {variable.name for variable in circuit.variables} == set(two_groups.columns)


def test_stratified_learning_infers_the_variables_when_none_are_given(two_groups):
    circuit = StratifiedLearning(
        variables=["group"], method=JointProbabilityTree()
    ).fit(two_groups)

    assert {variable.name for variable in circuit.variables} == set(two_groups.columns)
    assert len(circuit.root.subcircuits) == 2


def test_a_joint_probability_tree_fits_each_call_into_a_circuit_of_its_own(two_groups):
    """
    A stratified fit reuses one tree for every partition, so fitting again must not
    pile a second tree into the first fit's circuit.
    """
    variables = infer_variables_from_dataframe(two_groups)
    tree = JointProbabilityTree()

    first = tree.fit(two_groups, variables)
    second = tree.fit(two_groups, variables)

    assert first is not second
    assert len(first.nodes()) == len(second.nodes())


def test_stratified_learning_fits_one_branch_per_value_weighted_by_frequency(
    two_groups,
):
    variables = infer_variables_from_dataframe(two_groups)

    circuit = StratifiedLearning(variables=["group"], method=JointProbabilityTree()).fit(two_groups, variables)

    root = circuit.root
    assert isinstance(root, SumUnit)
    assert len(root.subcircuits) == 2
    assert sorted(math.exp(weight) for weight in root.log_weights) == pytest.approx(
        [0.25, 0.75]
    )


def test_stratified_learning_keeps_each_value_under_exactly_one_branch(two_groups):
    variables = infer_variables_from_dataframe(two_groups)
    circuit = StratifiedLearning(variables=["group"], method=JointProbabilityTree()).fit(two_groups, variables)
    group = next(variable for variable in circuit.variables if variable.name == "group")

    for value in (1, 3):
        event = SimpleEvent.from_data({group: value}).as_composite_set()
        branches_with_value = [
            branch
            for branch in circuit.root.subcircuits
            if _branch_circuit(branch).probability(event) > 0
        ]
        assert len(branches_with_value) == 1


def test_stratified_learning_fits_every_partition_with_the_wrapped_method(two_groups):
    """
    With the wrapped tree's minimum set to a partition's size, each partition is one
    leaf.
    """
    variables = infer_variables_from_dataframe(two_groups)
    method = StratifiedLearning(
        variables=["group"],
        method=JointProbabilityTree(min_samples_per_leaf=30),
    )

    circuit = method.fit(two_groups, variables)

    assert [
        _tree_leaf_count(partition) for partition in circuit.root.subcircuits
    ] == [1, 1]
