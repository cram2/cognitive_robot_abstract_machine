"""Common parameter fields and construction-time application defaults."""

from __future__ import annotations

# %% Imports
from dataclasses import asdict, dataclass, fields

from krrood.singleton import SingletonMeta
from typing_extensions import Any, Self


# %% Global configuration
@dataclass(kw_only=True)
class GlobalParameters:
    """Processing defaults declared once for every annotator type."""

    global_with_depth: bool = True
    """Whether depth images and their derived data are processed."""

    global_with_visualization: bool = True
    """Whether annotators prepare visualizations of their results."""


@dataclass(kw_only=True)
class AnnotatorPredefinedParameters(GlobalParameters, metaclass=SingletonMeta):
    """Application defaults configured before constructing annotator descriptors."""


# %% Effective annotator parameters
@dataclass(kw_only=True)
class AnnotatorParameters(GlobalParameters):
    """Effective settings belonging exclusively to one annotator instance."""

    @classmethod
    def from_defaults(cls, **overrides: Any) -> Self:
        """Snapshot application defaults, retaining explicit subclass and caller choices."""
        configured_defaults = asdict(AnnotatorPredefinedParameters())
        inherited_fields = fields(cls)
        arguments = {
            parameter.name: configured_defaults[parameter.name]
            for parameter in fields(GlobalParameters)
            if parameter in inherited_fields
        }
        return cls(**(arguments | overrides))
