"""
The parameter arrays of a part of a layered circuit and the array backend they belong
to.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType

import numpy as np
from array_api_compat import array_namespace, is_cupy_array, is_torch_array
from krrood.patterns.dataclass_deep_copy import DataclassDeepcopyMixin
from typing_extensions import Any, Self

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# %% parameter arrays of one backend


@dataclass(eq=False, kw_only=True)
class HasParameterArrays(DataclassDeepcopyMixin, ABC):
    """
    Base class for the parts of a layered circuit whose numeric queries compute with the
    array backend of their parameters.

    Everything else it stores is structure, numpy on the host. A deep copy copies only
    the constructor fields, so it looks up the backend anew.
    """

    @property
    @abstractmethod
    def parameters(self) -> list[BackendArray]:
        """
        :return: The parameter arrays, the only arrays an array backend may replace. All
            of them belong to one backend.
        """
        raise NotImplementedError

    @abstractmethod
    def copy_with_parameters(self, parameters: list[BackendArray]) -> Self:
        """
        :param parameters: Arrays of one backend in the order and with the shapes of
            :attr:`parameters`.
        :return: A copy of this part with these parameter arrays, sharing its structure.
        """
        raise NotImplementedError

    @functools.cached_property
    def dtype(self) -> Any:
        """
        :return: The floating point type of the parameters, which the numeric queries
            compute in.
        """
        return self.namespace.result_type(*self.parameters)

    @staticmethod
    def host_copy(array: BackendArray) -> np.ndarray:
        """
        :param array: An array of any backend, on any device.
        :return: The values of the array as a numpy array.
        """
        if is_torch_array(array):
            return array.detach().cpu().numpy()
        if is_cupy_array(array):
            return array.get()
        return np.asarray(array)

    @functools.cached_property
    def namespace(self) -> ModuleType:
        """
        :return: The Array API namespace of the backend of the parameters, which the
            numeric queries compute with.

        .. note::
            It is looked up once per object. Parameters replaced by arrays of another
            backend need a new object, such as one :func:`dataclasses.replace` makes.
        """
        return array_namespace(*self.parameters)
