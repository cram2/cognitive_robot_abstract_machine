import dataclasses
import unittest

import numpy as np
from krrood.adapters import json_serializer
from krrood.adapters.json_field import JSONField

from probabilistic_model.probabilistic_circuit.tensorized.input_layer.probability_table import (
    DenseProbabilityTable,
    ProbabilityTable,
    SparseProbabilityTable,
)

TABLE_TYPES = (DenseProbabilityTable, SparseProbabilityTable)

# a node with mass on a few states, a node with all of its mass on one state and a node
# without any mass, as truncation leaves it
PROBABILITIES = np.array(
    [
        [0.1, 0.0, 0.6, 0.0, 0.3],
        [0.0, 0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 0.0, 0.0],
    ]
)


def table_of(table_type, probabilities: np.ndarray = PROBABILITIES) -> ProbabilityTable:
    rows, columns = np.nonzero(probabilities)
    return table_type.from_entries(
        rows, columns, probabilities[rows, columns], probabilities.shape
    )


def log_of(probabilities: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(probabilities)


class ProbabilityTableTestCase(unittest.TestCase):
    """
    Both ways to store the probability table of a discrete layer answer every operation
    of the layer the same way as the dense numpy array they stand for.
    """

    def setUp(self):
        np.random.seed(69)

    def test_shape(self):
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                table = table_of(table_type)
                self.assertEqual(table.number_of_nodes, PROBABILITIES.shape[0])
                self.assertEqual(table.number_of_states, PROBABILITIES.shape[1])

    def test_log_probabilities_of_states(self):
        indices = np.array([2, 3, -1, 0, 2, 4])
        expected = log_of(PROBABILITIES)[:, np.where(indices >= 0, indices, 0)].T
        expected[indices < 0] = -np.inf
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                np.testing.assert_allclose(
                    table_of(table_type).log_probabilities_of_states(indices), expected
                )

    def test_probability_of_states(self):
        selected = np.array([True, False, True, True, False])
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                np.testing.assert_allclose(
                    table_of(table_type).probability_of_states(selected),
                    PROBABILITIES[:, selected].sum(axis=1),
                )

    def test_restricted_and_normalized(self):
        selected = np.array([True, False, False, True, True])
        restricted = np.where(selected, PROBABILITIES, 0.0)
        totals = restricted.sum(axis=1, keepdims=True)
        expected = np.divide(
            restricted, totals, out=np.zeros_like(restricted), where=totals > 0
        )
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                table = table_of(table_type).restricted_to(selected).normalized()
                self.assertIsInstance(table, table_type)
                np.testing.assert_allclose(table.dense_probabilities(), expected)

    def test_a_node_without_mass_stays_without_mass(self):
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                normalized = table_of(table_type).normalized().dense_probabilities()
                self.assertTrue(np.isfinite(normalized).all())
                np.testing.assert_array_equal(normalized[2], 0.0)

    def test_select_nodes_and_concatenate(self):
        mask = np.array([True, False, True])
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                table = table_of(table_type)
                selected = table.select_nodes(mask)
                np.testing.assert_allclose(
                    selected.dense_probabilities(), PROBABILITIES[mask]
                )
                joined = table_type.concatenate([selected, table])
                np.testing.assert_allclose(
                    joined.dense_probabilities(),
                    np.concatenate([PROBABILITIES[mask], PROBABILITIES]),
                )

    def test_product_with_values_of_the_states(self):
        values = np.arange(10, dtype=float).reshape(5, 2)
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                np.testing.assert_allclose(
                    table_of(table_type).dot(values), PROBABILITIES @ values
                )

    def test_probabilities_of_node(self):
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                states, probabilities = table_of(table_type).probabilities_of_node(0)
                np.testing.assert_array_equal(states, [0, 2, 4])
                np.testing.assert_allclose(probabilities, [0.1, 0.6, 0.3])

    def test_copy_shares_nothing(self):
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                table = table_of(table_type)
                copy = table.copy()
                table.normalized()
                np.testing.assert_allclose(copy.dense_probabilities(), PROBABILITIES)
                self.assertIsNot(copy, table)

    def test_samples_follow_the_probabilities_of_their_node(self):
        amount = 20000
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                nodes = np.repeat(np.arange(3), amount)
                states = table_of(table_type).sample_state_indices(nodes)
                for node in range(2):
                    frequencies = (
                        np.bincount(
                            states[nodes == node], minlength=PROBABILITIES.shape[1]
                        )
                        / amount
                    )
                    np.testing.assert_allclose(
                        frequencies, PROBABILITIES[node], atol=0.015
                    )
                # a node without mass has no state to draw
                np.testing.assert_array_equal(states[nodes == 2], -1)

    def test_cumulative_distribution_of_entries(self):
        unnormalized = PROBABILITIES * 7.0
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                rows, columns, cumulative = table_of(
                    table_type, unnormalized
                ).cumulative_distribution_of_entries()
                np.testing.assert_array_equal(rows, [0, 0, 0, 1])
                np.testing.assert_array_equal(columns, [0, 2, 4, 3])
                np.testing.assert_allclose(cumulative, [0.1, 0.7, 1.0, 1.0])

    def test_samples_of_unnormalized_rows(self):
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                table = table_of(table_type, PROBABILITIES * 7.0)
                states = table.sample_state_indices(np.zeros(5000, dtype=np.int64))
                self.assertTrue(set(states.tolist()) <= {0, 2, 4})

    def test_json_round_trip(self):
        for table_type in TABLE_TYPES:
            with self.subTest(table_type.__name__):
                table = table_of(table_type)
                serialized = json_serializer.to_json(table)
                # every field is written by the generic dataclass serializer
                self.assertEqual(
                    set(serialized) - {JSONField.TYPE},
                    {field_.name for field_ in dataclasses.fields(table_type)},
                )
                restored = json_serializer.from_json(serialized)
                self.assertIsInstance(restored, table_type)
                np.testing.assert_allclose(
                    restored.dense_probabilities(), PROBABILITIES
                )

    def test_conversion_between_the_types(self):
        for source in TABLE_TYPES:
            for target in TABLE_TYPES:
                with self.subTest(source=source.__name__, target=target.__name__):
                    converted = target.of(table_of(source))
                    self.assertIsInstance(converted, target)
                    np.testing.assert_allclose(
                        converted.dense_probabilities(), PROBABILITIES
                    )

    def test_sparse_table_stores_only_the_non_zero_entries(self):
        self.assertEqual(
            table_of(SparseProbabilityTable).number_of_stored_entries,
            int(np.count_nonzero(PROBABILITIES)),
        )
        self.assertEqual(
            table_of(DenseProbabilityTable).number_of_stored_entries,
            PROBABILITIES.size,
        )


if __name__ == "__main__":
    unittest.main()
