"""
Copies of the numpy arrays of the structure of a circuit on the devices of the torch
tensors they are used with, so that a step of training copies nothing from the host, as
recording it as a CUDA graph requires.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

import numpy as np
from array_api_compat import array_namespace, is_torch_array

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# %% structure array copies


@dataclass(kw_only=True, eq=False)
class StructureArrayCopies:
    """
    The copies of numpy arrays of a structure on the devices of the torch tensors they
    were used with, made the first time an array is used on a device.
    """

    copies: dict[tuple[int, str], tuple[np.ndarray, BackendArray]] = field(
        default_factory=dict, repr=False
    )
    """
    The numpy array, kept so that its identity stays unique, and its copy, by the
    identity of the numpy array and the device of the copy.
    """

    def placed(self, *, array: np.ndarray, values: BackendArray) -> BackendArray:
        """
        :param array: An array of the structure, such as positions that index
            ``values``.
        :param values: The array ``array`` is used with.
        :return: ``array`` as a tensor on the device of ``values`` if that is a torch
            tensor, ``array`` itself otherwise.
        """
        if not is_torch_array(values):
            return array
        key = (id(array), str(values.device))
        if key not in self.copies:
            self.copies[key] = (
                array,
                array_namespace(values).asarray(array, device=values.device),
            )
        return self.copies[key][1]
