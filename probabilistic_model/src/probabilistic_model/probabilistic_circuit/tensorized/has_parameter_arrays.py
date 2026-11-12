"""
The parameter arrays of a part of a layered circuit and the array backend they belong
to.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType

from array_api_compat import array_namespace

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# %% parameter arrays of one backend


@dataclass(eq=False, kw_only=True)
class HasParameterArrays(ABC):
    """
    Base class for the parts of a layered circuit whose numeric queries compute with the
    array backend of their parameters.

    Everything else such a part stores is structure: numpy arrays on the host, fixed
    once constructed.
    """

    @property
    @abstractmethod
    def parameters(self) -> list[BackendArray]:
        """
        :return: The parameter arrays, the only arrays an array backend may replace. All
            of them belong to one backend.
        """
        raise NotImplementedError

    @property
    def namespace(self) -> ModuleType:
        """
        :return: The Array API namespace of the backend of the parameters, which the
            numeric queries compute with.
        """
        return array_namespace(*self.parameters)
