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


@dataclass(kw_only=True, frozen=True)
class DevicePlacement:
    """
    A numpy array of a structure and the device it is copied to.
    """

    array_identity: int
    """
    The identity of the numpy array.
    """

    device: str
    """
    The device of the copy.
    """


@dataclass(kw_only=True, eq=False)
class StructureArrayCopy:
    """
    A numpy array of a structure and its copy on a device.
    """

    original: np.ndarray
    """
    The numpy array, kept so that its identity stays unique while the copy is used.
    """

    device_array: BackendArray
    """
    The copy of :attr:`original` on the device.
    """


@dataclass(kw_only=True, eq=False)
class StructureArrayCopies:
    """
    The copies of numpy arrays of a structure on the devices of the torch tensors they
    were used with, made the first time an array is used on a device.
    """

    copies: dict[DevicePlacement, StructureArrayCopy] = field(
        default_factory=dict, repr=False
    )
    """
    The copies made so far, by the array they copy and their device.
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
        placement = DevicePlacement(array_identity=id(array), device=str(values.device))
        if placement not in self.copies:
            self.copies[placement] = StructureArrayCopy(
                original=array,
                device_array=array_namespace(values).asarray(
                    array, device=values.device
                ),
            )
        return self.copies[placement].device_array
