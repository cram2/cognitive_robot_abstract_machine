"""
Storing learned circuits in files, so that their queries can be measured in processes of
their own.
"""

from __future__ import annotations

# %% imports
import io
import pickle
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from krrood.adapters.json_serializer import SubclassJSONSerializer
from krrood.patterns.called_positionally import called_positionally
from random_events.sigma_algebra import AbstractCompositeSet, AbstractSimpleSet
from random_events.variable import Variable
from typing_extensions import Any

from probabilistic_model.probabilistic_circuit.tensorized.layered_probabilistic_circuit import (
    LayeredProbabilisticCircuit,
)

# %% pickling


def rebuilt_variable(
    variable_type: type[Variable], name: str, domain: AbstractCompositeSet
) -> Variable:
    """
    :param variable_type: The type of the variable.
    :param name: Its name.
    :param domain: Its domain.
    :return: The variable.
    """
    return variable_type(name=name, domain=domain)


def rebuilt_set(json_data: dict[str, Any]) -> Any:
    """
    :param json_data: The JSON form of a set of random events.
    :return: The set.
    """
    return SubclassJSONSerializer.from_json(data=json_data)


class RandomEventsPickler(pickle.Pickler):
    """
    A pickler that stores the objects of random events, which wrap C++ objects, by their
    Python state.
    """

    @called_positionally
    def reducer_override(self, obj: Any) -> Any:
        """
        :param obj: An object to pickle.
        :return: How to rebuild a variable or set of random events from its Python
            state, :data:`NotImplemented` for every other object.
        """
        if isinstance(obj, Variable):
            return rebuilt_variable, (type(obj), obj.name, obj.domain)
        if isinstance(obj, (AbstractSimpleSet, AbstractCompositeSet)):
            return rebuilt_set, (obj.to_json(),)
        return NotImplemented


# %% storage


@dataclass(kw_only=True)
class PickledFile:
    """
    A file an object holding random events is pickled to.
    """

    path: Path
    """
    The path of the file.
    """

    def save_object(self, *, value: Any) -> None:
        """
        :param value: The object, written atomically.
        """
        buffer = io.BytesIO()
        RandomEventsPickler(buffer).dump(value)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_suffix(".partial")
        temporary.write_bytes(buffer.getvalue())
        temporary.replace(self.path)

    def load(self) -> Any:
        """
        :return: The object.
        """
        return pickle.loads(self.path.read_bytes())


@dataclass(kw_only=True)
class CircuitFile(PickledFile):
    """
    The file a learned circuit is stored in, with host parameters.
    """

    def save(self, *, circuit: LayeredProbabilisticCircuit) -> None:
        """
        :param circuit: The circuit, of any backend.
        """
        self.save_object(value=circuit.to_namespace(np))
