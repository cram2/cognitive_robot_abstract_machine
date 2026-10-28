"""
Both ways to store the probability table of a discrete layer answer every operation of
the layer the same way as the dense numpy array they stand for.
"""

from __future__ import annotations

import dataclasses

import numpy as np
import pytest

from krrood.adapters import json_serializer
from krrood.adapters.json_field import JSONField
from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
    ProbabilityTable,
    SparseProbabilityTable,
)

# %% fixtures


@pytest.fixture
def probabilities() -> np.ndarray:
    """
    A node with mass on a few states, a node with all of its mass on one state and a
    node without any mass, as truncation leaves it.
    """
    return np.array(
        [
            [0.1, 0.0, 0.6, 0.0, 0.3],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )


@pytest.fixture(
    params=[DenseProbabilityTable, SparseProbabilityTable],
    ids=lambda table_type: table_type.__name__,
)
def table_type(request) -> type[ProbabilityTable]:
    """
    Every way to store a probability table.
    """
    return request.param


def table_of(
    table_type: type[ProbabilityTable], probabilities: np.ndarray
) -> ProbabilityTable:
    """
    :param table_type: The way to store the table.
    :param probabilities: The dense probabilities, one row per node.
    :return: The table of the probabilities.
    """
    rows, columns = np.nonzero(probabilities)
    return table_type.from_entries(
        rows, columns, probabilities[rows, columns], probabilities.shape
    )


@pytest.fixture
def table(table_type, probabilities) -> ProbabilityTable:
    """
    The table of :func:`probabilities`, stored as :func:`table_type`.
    """
    return table_of(table_type, probabilities)


def log_of(probabilities: np.ndarray) -> np.ndarray:
    """
    :return: The logarithm of the probabilities, without warning about zeros.
    """
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


# %% every way to store a table


def test_shape(table, probabilities):
    assert table.number_of_nodes == probabilities.shape[0]
    assert table.number_of_states == probabilities.shape[1]


def test_log_probabilities_of_states(table, probabilities):
    indices = np.array([2, 3, -1, 0, 2, 4])
    expected = log_of(probabilities)[:, np.where(indices >= 0, indices, 0)].T
    expected[indices < 0] = -np.inf
    np.testing.assert_allclose(table.log_probabilities_of_states(indices), expected)


def test_probability_of_states(table, probabilities):
    selected = np.array([True, False, True, True, False])
    np.testing.assert_allclose(
        table.probability_of_states(selected), probabilities[:, selected].sum(axis=1)
    )


def test_restricted_and_normalized(table, table_type, probabilities):
    selected = np.array([True, False, False, True, True])
    restricted = np.where(selected, probabilities, 0.0)
    totals = restricted.sum(axis=1, keepdims=True)
    expected = np.divide(
        restricted, totals, out=np.zeros_like(restricted), where=totals > 0
    )
    normalized = table.restricted_to(selected).normalized()
    assert isinstance(normalized, table_type)
    np.testing.assert_allclose(normalized.dense_probabilities(), expected)


def test_a_node_without_mass_stays_without_mass(table):
    normalized = table.normalized().dense_probabilities()
    assert np.isfinite(normalized).all()
    np.testing.assert_array_equal(normalized[2], 0.0)


def test_select_nodes_and_concatenate(table, table_type, probabilities):
    mask = np.array([True, False, True])
    selected = table.select_nodes(mask)
    np.testing.assert_allclose(selected.dense_probabilities(), probabilities[mask])
    joined = table_type.concatenate([selected, table])
    np.testing.assert_allclose(
        joined.dense_probabilities(),
        np.concatenate([probabilities[mask], probabilities]),
    )


def test_product_with_values_of_the_states(table, probabilities):
    values = np.arange(10, dtype=float).reshape(5, 2)
    np.testing.assert_allclose(table.dot(values), probabilities @ values)


def test_probabilities_of_node(table):
    states, probabilities = table.probabilities_of_node(0)
    np.testing.assert_array_equal(states, [0, 2, 4])
    np.testing.assert_allclose(probabilities, [0.1, 0.6, 0.3])


def test_copy_shares_nothing(table, probabilities):
    copy = table.copy()
    table.normalized()
    assert copy is not table
    np.testing.assert_allclose(copy.dense_probabilities(), probabilities)


def test_samples_follow_the_probabilities_of_their_node(table, probabilities):
    amount = 20000
    nodes = np.repeat(np.arange(3), amount)
    states = table.sample_state_indices(nodes)
    for node in range(2):
        frequencies = (
            np.bincount(states[nodes == node], minlength=probabilities.shape[1])
            / amount
        )
        np.testing.assert_allclose(frequencies, probabilities[node], atol=0.015)
    # a node without mass has no state to draw
    np.testing.assert_array_equal(states[nodes == 2], -1)


def test_cumulative_distribution_of_entries(table_type, probabilities):
    rows, columns, cumulative = table_of(
        table_type, probabilities * 7.0
    ).cumulative_distribution_of_entries()
    np.testing.assert_array_equal(rows, [0, 0, 0, 1])
    np.testing.assert_array_equal(columns, [0, 2, 4, 3])
    np.testing.assert_allclose(cumulative, [0.1, 0.7, 1.0, 1.0])


def test_samples_of_unnormalized_rows(table_type, probabilities):
    table = table_of(table_type, probabilities * 7.0)
    states = table.sample_state_indices(np.zeros(5000, dtype=np.int64))
    assert set(states.tolist()) <= {0, 2, 4}


def test_json_round_trip(table, table_type, probabilities):
    serialized = json_serializer.to_json(table)
    # every field is written by the generic dataclass serializer
    assert set(serialized) - {JSONField.TYPE} == {
        field_.name for field_ in dataclasses.fields(table_type)
    }
    restored = json_serializer.from_json(serialized)
    assert isinstance(restored, table_type)
    np.testing.assert_allclose(restored.dense_probabilities(), probabilities)


@pytest.mark.parametrize(
    "target", [DenseProbabilityTable, SparseProbabilityTable], ids=lambda t: t.__name__
)
def test_conversion_between_the_types(table, target, probabilities):
    converted = target.of(table)
    assert isinstance(converted, target)
    np.testing.assert_allclose(converted.dense_probabilities(), probabilities)


# %% storage of each type


def test_dense_table_stores_the_log_probability_of_every_state(probabilities):
    np.testing.assert_allclose(
        table_of(DenseProbabilityTable, probabilities).log_probabilities,
        log_of(probabilities),
    )


def test_sparse_table_stores_the_log_probabilities_of_the_non_zero_entries(
    probabilities,
):
    stored = table_of(SparseProbabilityTable, probabilities).log_probabilities
    rows, columns = np.nonzero(probabilities)
    np.testing.assert_array_equal(stored.indices, columns)
    np.testing.assert_array_equal(
        np.diff(stored.indptr), np.bincount(rows, minlength=3)
    )
    np.testing.assert_allclose(stored.data, np.log(probabilities[rows, columns]))


@pytest.mark.parametrize(
    "derive",
    [
        lambda table: table,
        lambda table: SparseProbabilityTable.concatenate([table]),
        lambda table: table.copy(),
        lambda table: json_serializer.from_json(json_serializer.to_json(table)),
    ],
    ids=["selected", "concatenated", "copied", "restored"],
)
def test_sparse_table_keeps_a_state_with_probability_one(derive, probabilities):
    """
    The log-probability of such a state is a stored zero, which is not taken for a state
    that is not stored.
    """
    only_certain_node = np.array([False, True, False])
    selected = table_of(SparseProbabilityTable, probabilities).select_nodes(
        only_certain_node
    )
    derived = derive(selected)
    assert derived.number_of_stored_entries == 1
    np.testing.assert_array_equal(
        derived.dense_probabilities(), probabilities[only_certain_node]
    )


def test_number_of_stored_entries(probabilities):
    assert table_of(
        SparseProbabilityTable, probabilities
    ).number_of_stored_entries == int(np.count_nonzero(probabilities))
    assert (
        table_of(DenseProbabilityTable, probabilities).number_of_stored_entries
        == probabilities.size
    )
