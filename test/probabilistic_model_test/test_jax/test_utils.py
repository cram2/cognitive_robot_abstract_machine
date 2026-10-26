from __future__ import annotations

import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR
from random_events.interval import SimpleInterval
from scipy.sparse import coo_array

from probabilistic_model.probabilistic_circuit.jax.utils import (
    copy_bcoo,
    create_bcoo_indices_from_row_lengths,
    create_bcsr_indices_from_row_lengths,
    sample_from_sparse_probabilities_csc,
    shrink_index_array,
    simple_interval_to_open_array,
    sparse_remove_rows_and_columns_where_all,
)

# %% sparse arrays


def test_copy_of_a_sparse_array_is_independent():
    original = BCOO.fromdense(jnp.array([[0, 1], [2, 3]]))
    copy = copy_bcoo(original)
    assert jnp.allclose(original.todense(), copy.todense())
    original.data += 1
    assert not jnp.allclose(original.todense(), copy.todense())


def test_sparse_indices_from_row_lengths():
    indices = create_bcoo_indices_from_row_lengths(jnp.array([2, 3]))
    assert jnp.allclose(indices, jnp.array([[0, 0], [0, 1], [1, 0], [1, 1], [1, 2]]))


def test_compressed_sparse_row_indices_from_row_lengths():
    column_indices, index_pointer = create_bcsr_indices_from_row_lengths(
        jnp.array([2, 3])
    )
    expected = BCSR.fromdense(jnp.array([[1, 1, 0], [1, 1, 1]]))
    assert jnp.allclose(expected.indices, column_indices)
    assert jnp.allclose(expected.indptr, index_pointer)


def test_sampling_from_sparse_probabilities_draws_the_amount_per_row():
    probabilities = coo_array(
        np.array([[0.1, 0.2, 0.0, 0.7], [0.4, 0.0, 0.6, 0.0]])
    ).tocsr()
    amount = jnp.array([2, 3])
    samples = sample_from_sparse_probabilities_csc(probabilities, amount)
    assert np.all(samples.sum(axis=1) == amount)


def test_shrink_index_array():
    shrunk = shrink_index_array(jnp.array([[0, 3], [1, 0], [4, 1]]))
    assert jnp.allclose(shrunk, jnp.array([[0, 2], [1, 0], [2, 1]]))


def test_sparse_remove_rows_and_columns_where_all():
    array = BCOO.fromdense(jnp.array([[1, 0, 3], [0, 0, 0], [7, 0, 9]]))
    removed = sparse_remove_rows_and_columns_where_all(array, 0)
    assert jnp.allclose(removed.todense(), jnp.array([[1, 3], [7, 9]]))


# %% intervals


def test_open_interval_keeps_its_bounds():
    array = simple_interval_to_open_array(SimpleInterval.from_data(0, 1))
    assert jnp.array_equal(array, jnp.array([0.0, 1.0]))
