"""
The base classes of the conversion between the circuits of the ``rx`` package and the
layered circuits of the ``tensorized`` package.
"""

from __future__ import annotations

from abc import ABC

from probabilistic_model.adapters.converter import Converter, InputType, OutputType


class RustworkxToTensorizedConverter(Converter[InputType, OutputType], ABC):
    """
    Base class for converters from the ``rx`` package to the ``tensorized`` package.
    """


class TensorizedToRustworkxConverter(Converter[InputType, OutputType], ABC):
    """
    Base class for converters from the ``tensorized`` package to the ``rx`` package.
    """
