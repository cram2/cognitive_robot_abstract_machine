import unittest

import jax.numpy as jnp
import jax.random
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from random_events.interval import SimpleInterval
from scipy.sparse import coo_array

from probabilistic_model.probabilistic_circuit.jax import (
    create_bcsr_indices_from_row_lengths,
    shrink_index_array,
    sparse_remove_rows_and_cols_where_all,
)
from probabilistic_model.probabilistic_circuit.jax.utils import (
    copy_bcoo,
    simple_interval_to_open_array,
    create_bcoo_indices_from_row_lengths,
    sample_from_sparse_probabilities_csc,
)


class BCOOTestCase(unittest.TestCase):

    def test_copy(self):
        x = BCOO.fromdense(jnp.array([[0, 1], [2, 3]]))
        y = copy_bcoo(x)
        self.assertTrue(jnp.allclose(x.todense(), y.todense()))
        x.data += 1
        self.assertFalse(jnp.allclose(x.todense(), y.todense()))
        y.data += 1
        self.assertTrue(jnp.allclose(x.todense(), y.todense()))
        y.data += 1
        self.assertFalse(jnp.allclose(x.todense(), y.todense()))

    def test_create_sparse_array_indices_from_row_lengths(self):
        row_lengths = jnp.array([2, 3])
        indices = create_bcoo_indices_from_row_lengths(row_lengths)
        result = jnp.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]])
        self.assertTrue(jnp.allclose(indices, result))

    def test_create_bcsr_indices_from_row_lengths(self):
        row_lengths = jnp.array([2, 3])
        column_indices, indent_pointer = create_bcsr_indices_from_row_lengths(
            row_lengths
        )

        bcsr = BCSR.fromdense(jnp.array([[1, 1, 0], [1, 1, 1]]))

        self.assertTrue(jnp.allclose(bcsr.indices, column_indices))
        self.assertTrue(jnp.allclose(bcsr.indptr, indent_pointer))

    def test_sample_from_sparse_probabilities_csc(self):
        probs = coo_array(
            np.array([[0.1, 0.2, 0.0, 0.7], [0.4, 0.0, 0.6, 0.0]])
        ).tocsr()
        amount = jnp.array([2, 3])

        samples = sample_from_sparse_probabilities_csc(probs, amount)

        amounts = samples.sum(axis=1)
        self.assertTrue(np.all(amounts == amount))
        self.assertTrue(np.all(samples.data <= 3))

    def test_shrink_index_array(self):
        index_array = jnp.array([[0, 3], [1, 0], [4, 1]])
        new_index_tensor = shrink_index_array(index_array)
        result = jnp.array([[0, 2], [1, 0], [2, 1]])
        self.assertTrue(jnp.allclose(new_index_tensor, result))

    def test_sparse_remove_rows_and_cols_where_all(self):
        array = BCOO.fromdense(jnp.array([[1, 0, 3], [0, 0, 0], [7, 0, 9]]))
        result = jnp.array([[1, 3], [7, 9]])
        new_array = sparse_remove_rows_and_cols_where_all(array, 0)
        self.assertTrue(jnp.allclose(new_array.todense(), result))


class IntervalConversionTestCase(unittest.TestCase):

    def simple_interval_to_open_array(self):
        simple_interval = SimpleInterval(0, 1)
        array = simple_interval_to_open_array(simple_interval)
        self.assertTrue(jnp.allclose(array, jnp.array([0, 1])))


if __name__ == "__main__":
    unittest.main()
