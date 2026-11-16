from __future__ import annotations

import numpy as np
import numpy.typing as npt

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    NodeIndices,
    NodeMask,
)


def remap_indices(
    keep_mask: NodeMask,
) -> tuple[NodeIndices, int]:
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
