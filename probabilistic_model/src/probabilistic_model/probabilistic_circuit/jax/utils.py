import jax.numpy as jnp
import numpy as np
from jax.experimental.sparse import BCOO, BCSR, CSC, CSR
from random_events.interval import SimpleInterval, Bound
import jax
from scipy.sparse import csr_matrix, csr_array, csc_array
from typing_extensions import Tuple

from probabilistic_model.utils import timeit_print


def copy_bcoo(x: BCOO) -> BCOO:
    return x.__class__((x.data.copy(), x.indices.copy()), shape=x.shape, indices_sorted=x.indices_sorted,
                unique_indices=x.unique_indices)

def copy_bcsr(x: BCSR) -> BCSR:
    return x.__class__((x.data.copy(), x.indices.copy(), x.indptr.copy()), shape=x.shape, indices_sorted=x.indices_sorted,
                unique_indices=x.unique_indices)

def simple_interval_to_open_array(interval: SimpleInterval) -> jnp.array:
    lower = jnp.array(interval.lower)
    if interval.left == Bound.CLOSED:
        lower = jnp.nextafter(lower, lower - 1)
    upper = jnp.array(interval.upper)
    if interval.right == Bound.CLOSED:
        upper = jnp.nextafter(upper, upper + 1)
    return jnp.array([lower, upper])


def create_bcoo_indices_from_row_lengths(row_lengths: np.array) -> np.array:
    """
    Create the indices of a BCOO array with the given row lengths.

    The shape of the indices is (2, sum(row_lengths)).
    The shape of the sparse tensor that the indices describe should be (len(row_lengths), max(row_lengths)).

    Example::

        >>> row_lengths = jnp.array([2, 3])
        >>> create_bcoo_indices_from_row_lengths(row_lengths)
            [[0 0]
             [0 1]
             [1 0]
             [1 1]
             [1 2]]

    :param row_lengths: The row lengths.
    :return: The indices of the sparse tensor
    """

    # create row indices
    row_indices = np.repeat(np.arange(len(row_lengths)), row_lengths)

    # offset the row lengths by the one element
    offset_row_lengths = np.concatenate([jnp.array([0]), row_lengths[:-1]])

    # create a cumulative sum of the offset row lengths and offset it by the first row length
    cum_sum = np.repeat(offset_row_lengths, row_lengths)

    # arrange column indices
    summed_row_lengths = np.arange(row_lengths.sum())

    # create the column indices
    col_indices = summed_row_lengths - cum_sum

    return np.vstack((row_indices, col_indices)).T

def create_bcoo_indices_from_row_lengths_np(row_lengths: np.array) -> np.array:
    """
    Create the indices of a BCOO array with the given row lengths.

    The shape of the indices is (2, sum(row_lengths)).
    The shape of the sparse tensor that the indices describe should be (len(row_lengths), max(row_lengths)).

    Example::

        >>> row_lengths = jnp.array([2, 3])
        >>> create_bcoo_indices_from_row_lengths(row_lengths)
            [[0 0]
             [0 1]
             [1 0]
             [1 1]
             [1 2]]

    :param row_lengths: The row lengths.
    :return: The indices of the sparse tensor
    """

    # create row indices
    row_indices = np.repeat(jnp.arange(len(row_lengths)), row_lengths)

    # offset the row lengths by the one element
    offset_row_lengths = np.concatenate([jnp.array([0]), row_lengths[:-1]])

    # create a cumulative sum of the offset row lengths and offset it by the first row length
    cum_sum = np.repeat(offset_row_lengths, row_lengths)

    # arrange column indices
    summed_row_lengths = np.arange(row_lengths.sum())

    # create the column indices
    col_indices = summed_row_lengths - cum_sum

    return np.vstack((row_indices, col_indices))


def create_bcsr_indices_from_row_lengths(row_lengths: jax.Array) -> Tuple[jax.Array, jax.Array]:
    """
    Create the column indices and indent pointer of bcsr array with the given row lengths.

    The shape of the sparse tensor that the indices describe should be (len(row_lengths), max(row_lengths)).

    Example::

        >>> row_lengths = jnp.array([2, 3])
        >>> create_bcsr_indices_from_row_lengths(row_lengths)
        (Array([0, 1, 0, 1, 2], dtype=int32), Array([0, 2, 5], dtype=int32))

    :param row_lengths: The row lengths.
    :return: The indices of the sparse tensor
    """

    # offset the row lengths by the one element
    offset_row_lengths = jnp.concatenate([jnp.array([0]), row_lengths[:-1]])

    # create a cumulative sum of the offset row lengths and offset it by the first row length
    cum_sum = jnp.repeat(offset_row_lengths, row_lengths)

    # arrange column indices
    summed_row_lengths = jnp.arange(row_lengths.sum())

    # create the column indices
    col_indices = summed_row_lengths - cum_sum

    indent_pointer = jnp.concatenate([jnp.array([0]), jnp.cumsum(row_lengths)])

    return col_indices, indent_pointer


def embed_sparse_array_in_nan_array(sparse_array: BCOO) -> jax.Array:
    result = jnp.full(sparse_array.shape, jnp.nan, dtype=jnp.float32)
    result = result.at[sparse_array.indices[:, 0], sparse_array.indices[:, 1]].set(sparse_array.data)
    return result


def sample_from_sparse_probabilities_csc(probabilities: csr_array, amount: np.array) -> csc_array:
    """
    Sample from a sparse array of probabilities.
    Each row in the sparse array encodes a categorical probability distribution.

    :param probabilities: The sparse array of probabilities.
    :param amount: The amount of samples to draw from each row.
    :return: The samples that are drawn for each state in the probabilities indicies.
    """
    all_samples = np.concatenate([np.random.multinomial(amount_.item(), pvals=probability_row.data) for amount_, probability_row in zip(amount, probabilities)], axis=0)
    result = csr_array((all_samples, probabilities.indices, probabilities.indptr),
                       shape=probabilities.shape).tocsc(copy=False)
    return result


def remove_rows_and_cols_where_all(array: jax.Array, value: float) -> jax.Array:
    """
    Remove rows and columns from an array where all elements are equal to a given value.

    :param array: The tensor to remove rows and columns from.
    :param value: The value to remove.
    :return: The tensor without the rows and columns.

    Example::


        >>> a = jnp.array([[1, 0, 3], [0, 0, 0], [7, 0, 9]])
        >>> remove_rows_and_cols_where_all(a, 0)
        array([[1, 3], [7, 9]])
    """

    # get the rows and columns that are not entirely -inf
    valid_rows = (array != value).any(axis=1)
    valid_cols = (array != value).any(axis=0)

    # remove rows and cols that are entirely -inf
    valid = array[valid_rows][:, valid_cols]
    return valid

def shrink_index_array(index_array: jax.Array) -> jax.Array:
    """
    Shrink an index array to only contain successive indices.

    Example::

        >>> shrink_index_array(jnp.array([[0, 3], [1, 0], [4, 1]]))
            [[0 2]
             [1 0]
             [2 1]]
    :param index_array: The index tensor to shrink.
    :return: The shrunken index tensor.
    """
    result = index_array.copy()

    for dim in range(index_array.shape[1]):
        unique_indices = jnp.unique(index_array[:, dim])

        # map the old indices to the new indices
        for new_index, unique_index in zip(range(len(unique_indices)), unique_indices):
            result = result.at[result[:, dim] == unique_index, dim].set(new_index)


    return result


def sparse_remove_rows_and_cols_where_all(array: BCOO, value: float) -> BCOO:
    """
    Remove rows and columns from a sparse tensor where all elements are equal to a given value.

    Example::
        >>> array = BCOO.fromdense(jnp.array([[1, 0, 3], [0, 0, 0], [7, 0, 9]]))
        >>> sparse_remove_rows_and_cols_where_all(array, 0).todense()
            [[1 3]
             [7 9]]

    :param array: The sparse tensor to remove rows and columns from.
    :param value: The value to remove.
    :return: The tensor without the unnecessary rows and columns.
    """
    # get indices of values where all elements are equal to a given value
    values = array.data
    valid_elements = (values != value)

    # filter indices by valid elements
    valid_indices = array.indices[valid_elements]

    # shrink indices
    valid_indices = shrink_index_array(valid_indices)

    new_shape = jnp.max(valid_indices, axis=0) + 1

    # construct result tensor
    result = BCOO((values[valid_elements], valid_indices), shape=new_shape, indices_sorted=array.indices_sorted,
                  unique_indices=array.unique_indices)
    return result
