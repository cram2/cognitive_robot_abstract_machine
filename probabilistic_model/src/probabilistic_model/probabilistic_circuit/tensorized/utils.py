from __future__ import annotations

import numpy as np
import numpy.typing as npt
from typing_extensions import Tuple

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeMask,
)


def embedded_logsumexp(values: npt.NDArray, axis: int) -> npt.NDArray:
    """
    Numerically stable ``log(sum(exp(values)))`` that maps an all ``-inf`` reduction to
    ``-inf`` instead of ``nan``.

    :param values: The values in log space.
    :param axis: The axis to reduce.
    :return: The reduced array.
    """
    values = np.asarray(values, dtype=float)
    maximum = np.max(values, axis=axis, keepdims=True)
    maximum = np.where(np.isfinite(maximum), maximum, 0.0)

    # the subtraction and the exponentiation are done into the same buffer: these arrays
    # hold one entry per edge per event, so every avoided temporary matters
    shifted = values - maximum
    with np.errstate(over="ignore"):
        np.exp(shifted, out=shifted)
    summed = np.sum(shifted, axis=axis, keepdims=True)

    with np.errstate(divide="ignore"):
        result = np.where(summed > 0, np.log(summed) + maximum, -np.inf)
    return np.squeeze(result, axis=axis)


def remap_indices(
    keep_mask: NodeMask,
) -> Tuple[NodeIndices, int]:
    """
    Create an index remapping for a prune operation.

    :param keep_mask: A boolean mask of the entries that survive.
    :return: An array that maps old indices to new indices (``-1`` for removed entries)
        and the number of surviving entries.
    """
    remap = np.full(len(keep_mask), -1, dtype=np.int64)
    number_of_kept = int(keep_mask.sum())
    remap[keep_mask] = np.arange(number_of_kept, dtype=np.int64)
    return remap, number_of_kept
