"""
The JSON serializers of scipy sparse arrays keep every stored entry.
"""

from __future__ import annotations

import numpy as np
from scipy.sparse import coo_array, csr_array

from krrood.adapters.json_serializer import (
    JSONSerializableTypeRegistry,
    from_json,
    to_json,
)
from krrood.adapters.sparse_array_json_serializer import (
    CompressedSparseRowArrayJSONSerializer,
    CoordinateSparseArrayJSONSerializer,
)

# %% registration


def test_importing_the_module_registers_the_serializer_of_each_sparse_format():
    """
    The registry finds a serializer for each sparse format once the module is imported.
    """
    registry = JSONSerializableTypeRegistry()

    assert registry.get_external_serializer(clazz=coo_array) is (
        CoordinateSparseArrayJSONSerializer
    )
    assert registry.get_external_serializer(clazz=csr_array) is (
        CompressedSparseRowArrayJSONSerializer
    )


# %% round trips


def test_coordinate_sparse_array_keeps_its_stored_entries():
    """
    A coordinate-format array round-trips with its stored zero.
    """
    # the first entry stores a zero, which is a value and not a missing entry
    obj = coo_array(
        (np.array([0, 3]), (np.array([0, 1]), np.array([1, 0]))), shape=(2, 3)
    )
    result = from_json(to_json(obj))
    assert isinstance(result, coo_array)
    assert result.shape == obj.shape
    np.testing.assert_array_equal(result.data, obj.data)
    np.testing.assert_array_equal(result.row, obj.row)
    np.testing.assert_array_equal(result.col, obj.col)


def test_compressed_sparse_row_array_keeps_its_stored_entries():
    """
    A compressed-sparse-row array round-trips with its stored zero.
    """
    # the first entry stores a zero, which is a value and not a missing entry
    obj = csr_array(
        (np.array([0, 3]), (np.array([0, 1]), np.array([1, 0]))), shape=(2, 3)
    )
    result = from_json(to_json(obj))
    assert isinstance(result, csr_array)
    assert result.shape == obj.shape
    np.testing.assert_array_equal(result.data, obj.data)
    np.testing.assert_array_equal(result.indices, obj.indices)
    np.testing.assert_array_equal(result.indptr, obj.indptr)
