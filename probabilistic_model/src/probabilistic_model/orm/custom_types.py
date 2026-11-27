"""
Column types for the values of probabilistic models that SQLAlchemy cannot store by
itself.
"""

from __future__ import annotations

from krrood.ormatic.custom_types import NumpyArrayType
from sqlalchemy.engine import Dialect

from probabilistic_model.probabilistic_circuit.tensorized.array_types import (
    BackendArray,
)
from probabilistic_model.probabilistic_circuit.tensorized.has_parameter_arrays import (
    HasParameterArrays,
)


# %% arrays of any backend


class BackendArrayType(NumpyArrayType):
    """
    Column type for arrays of any backend and device, stored as their values on the host
    in the ``.npy`` format and loaded as numpy arrays.
    """

    cache_ok = True

    def process_bind_param(
        self, value: BackendArray | None, dialect: Dialect
    ) -> bytes | None:
        """
        :meth:`NumpyArrayType.process_bind_param` of the host copy of the value, which
        may be an array of any backend and device.
        """
        if value is None:
            return None
        return super().process_bind_param(HasParameterArrays.host_copy(value), dialect)
