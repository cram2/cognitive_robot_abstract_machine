"""
The parameter arrays of a part of a layered circuit and the array backend they belong
to.
"""

from __future__ import annotations

import functools
from abc import ABC, abstractmethod
from dataclasses import dataclass
from types import ModuleType

from array_api_compat import array_namespace
from krrood.patterns.dataclass_deep_copy import DataclassDeepcopyMixin

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)

# %% parameter arrays of one backend


@dataclass(eq=False, kw_only=True)
class HasParameterArrays(DataclassDeepcopyMixin, ABC):
    """
    Base class for the parts of a layered circuit whose numeric queries compute with the
    array backend of their parameters.

    Everything else such a part stores is structure: numpy arrays on the host, fixed
    once constructed. A deep copy copies the constructor fields only, so it looks up
    the backend anew.
    """

    @property
    @abstractmethod
    def parameters(self) -> list[BackendArray]:
        """
        :return: The parameter arrays, the only arrays an array backend may replace. All
            of them belong to one backend.
        """
        raise NotImplementedError

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
