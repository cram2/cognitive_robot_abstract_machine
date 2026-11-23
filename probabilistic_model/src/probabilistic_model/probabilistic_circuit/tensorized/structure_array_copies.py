"""
Copies of the numpy arrays of the structure of a circuit on the devices of the torch
tensors they are used with, so that a step of training copies nothing from the host, as
recording it as a CUDA graph requires, and the values at the positions they hold.
"""

from __future__ import annotations

# %% imports
from dataclasses import dataclass, field

from array_api_compat import array_namespace, is_torch_array
from typing_extensions import Any, Self

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    ArraySelection,
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

    original: ArraySelection
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

    def __deepcopy__(self, memo: dict[int, Any]) -> Self:
        """
        :return: Copies of no arrays, since a deep copy of a structure has arrays of its
            own, which no copy made so far is of.
        """
        return type(self)()

    def placed(self, *, array: ArraySelection, values: BackendArray) -> BackendArray:
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

    def values_at_positions(
        self, *, values: BackendArray, positions: np.ndarray
    ) -> BackendArray:
        """
        :param values: Values along the last axis.
        :param positions: Positions in the last axis, an array of the structure.
        :return: The values at the positions, flattened in their order, along the last
            axis.
        """
        if not is_torch_array(values):
            return array_namespace(values).take(values, positions.reshape(-1), axis=-1)
        flat_positions = self.placed(array=positions, values=values).reshape(-1)
        return values.gather(
            -1, flat_positions.expand(*values.shape[:-1], flat_positions.shape[0])
        )
