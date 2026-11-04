"""
The base classes of the conversion between the circuits of the ``jax`` package and the
layered circuits of the ``tensorized`` package.

The ``jax`` package only learns circuits. Every other query is answered by the
``tensorized`` package, so a learned circuit is converted into it for inference, and the
``tensorized`` package is the only bridge between the ``jax`` package and the circuits
of the ``rx`` package.
"""

from __future__ import annotations

from abc import ABC

from probabilistic_model.adapters.converter import Converter, InputType, OutputType


class JaxToTensorizedConverter(Converter[InputType, OutputType], ABC):
    """
    Base class for converters from the ``jax`` package to the ``tensorized`` package.
    """


class TensorizedToJaxConverter(Converter[InputType, OutputType], ABC):
    """
    Base class for converters from the ``tensorized`` package to the ``jax`` package.
    """
