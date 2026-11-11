"""Importable dataclasses describing annotator configuration and capabilities."""

from __future__ import annotations

# %% Imports
from dataclasses import dataclass, field

from krrood.ormatic.utils import classproperty
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric
from typing_extensions import Generic, TypeVar

from robokudo.datastructures.annotator_parameters import AnnotatorParameters


# %% Capability declarations
@dataclass(kw_only=True)
class AnnotatorCapabilities:
    """Declared annotation types accepted and produced by an annotator."""

    inputs: list[type] | None = None
    """Accepted input types, or None when no declaration is provided."""

    outputs: list[type] | None = None
    """Produced output types, or None when no declaration is provided."""


# %% Typed descriptors
ParametersType = TypeVar("ParametersType", bound=AnnotatorParameters)
"""Concrete parameter type carried by a descriptor family member."""


@dataclass(kw_only=True)
class AnnotatorDescriptor(Generic[ParametersType], SubClassSafeGeneric):
    """Declarative settings independent of an annotator's runtime resources."""

    parameters: ParametersType
    """Effective processing settings for a single annotator."""

    capabilities: AnnotatorCapabilities = field(default_factory=AnnotatorCapabilities)
    """Declarations of accepted and produced annotation types."""

    @classproperty
    def Parameters(cls) -> type[ParametersType]:
        """Parameter constructor exposed through the descriptor discovery interface."""
        return cls.get_type_of_generic_parameter(parameter=ParametersType)

    @classproperty
    def Capabilities(cls) -> type[AnnotatorCapabilities]:
        """Capability constructor shared by descriptor types."""
        return AnnotatorCapabilities


@dataclass(kw_only=True)
class BaseAnnotatorDescriptor(AnnotatorDescriptor[AnnotatorParameters]):
    """Configuration for annotators without algorithm-specific settings."""

    parameters: AnnotatorParameters = field(
        default_factory=AnnotatorParameters.from_defaults
    )
    """Common settings initialized from the application's configured defaults."""
