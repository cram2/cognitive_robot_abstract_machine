"""
The base class of the conversions between the circuit representations.

The representations do not know about each other; every conversion lives in a converter
class in a subpackage of this package. A converter handles one input type and is found
by the base class it derives from, so adding a conversion is subclassing a base class,
without any registration.
"""

from __future__ import annotations

from abc import ABC
from dataclasses import dataclass, field

from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from krrood.utils import recursive_subclasses
from sortedcontainers import SortedSet
from typing_extensions import Any, Dict, Generic, Type, TypeVar

from probabilistic_model.adapters.exceptions import (
    CannotConvertError,
)

InputType = TypeVar("InputType")
OutputType = TypeVar("OutputType")


class Converter(Generic[InputType, OutputType], SubClassSafeGeneric, ABC):
    """
    Base class for converters from one representation to another.

    A concrete converter binds the input and output type and overrides :meth:`convert`.
    """

    @classmethod
    def input_type(cls) -> Type[InputType]:
        """
        :return: The type this converter converts.
        """
        return cls.get_generic_type_parameters()[0]

    @classmethod
    def output_type(cls) -> Type[OutputType]:
        """
        :return: The type this converter converts into.
        """
        return cls.get_generic_type_parameters()[1]

    @classmethod
    def can_convert(cls, data: Any) -> bool:
        """
        Override this to customize which objects this converter handles.

        :param data: The object to convert.
        :return: Whether this converter handles the object.
        """
        return type(data) is cls.input_type()

    @classmethod
    def converter_for(cls, data: Any) -> Type[Converter]:
        """
        :param data: The object to convert.
        :return: The subclass of this class that handles the object.
        :raises CannotConvertError: If no subclass handles it.
        """
        for converter in recursive_subclasses(cls):
            if isinstance(converter.input_type(), TypeVar):
                continue
            if converter.can_convert(data):
                return converter
        raise CannotConvertError(data_type=type(data))

    @classmethod
    def convert(cls, data: InputType, *context: Any) -> OutputType:
        """
        Convert an object with the converter that handles it.

        :param data: The object to convert.
        :param context: The state of the conversion the object is part of.
        :return: The converted object.
        """
        return cls.converter_for(data).convert(data, *context)


@dataclass
class LayerConversion(Generic[InputType, OutputType]):
    """
    The conversion of the layers of one circuit into the layers of another
    representation, which converts every layer once, so that a layer shared by several
    parents stays shared.
    """

    converter: Type[Converter[InputType, OutputType]]
    """
    The base class of the converters of the layers, which finds the converter of every
    layer.
    """

    variables: SortedSet
    """
    The variables of the circuit, in the order the layers index them.
    """

    converted_layers: Dict[int, OutputType] = field(default_factory=dict)
    """
    The result of every layer converted so far, keyed by the id of the layer.
    """

    def convert(self, layer: InputType) -> OutputType:
        """
        :param layer: A layer of the circuit.
        :return: The layer of the other representation with the same nodes.
        """
        if id(layer) not in self.converted_layers:
            self.converted_layers[id(layer)] = self.converter.convert(layer, self)
        return self.converted_layers[id(layer)]
