"""
Conversion of dataclass instances to eval-able Python constructor source.

:class:`ObjectSourceSerializer` is the pluggable strategy; the default,
:class:`DataclassConstructorSerializer`, handles flat and nested dataclasses whose
fields are scalars, enum members or further dataclasses.
"""

from __future__ import annotations

import dataclasses
from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Any

from krrood.code_generation.ast_helpers import ast_unparse_expression
from krrood.code_generation.exceptions import ObjectNotSerializableError
from krrood.code_generation.source_emission import (
    SourceEmissionContext,
    ValueSourceEmitter,
)
from krrood.ormatic.utils import classproperty

# %% ObjectSource


@dataclass
class ObjectSource:
    """
    The eval-able Python constructor source for an object, plus its type dependencies.
    """

    source: str
    """
    Eval-able Python constructor expression, e.g. ``Animal(name='Rex')``.
    """

    referenced_types: set[type]
    """
    Every type referenced in :attr:`source` that must be imported for it to evaluate.
    """


# %% ObjectSourceSerializer


@dataclass
class ObjectSourceSerializer(ABC):
    """
    Converts an object to eval-able Python constructor source.
    """

    @abstractmethod
    def to_source(self, instance: Any) -> ObjectSource:
        """
        Return the constructor source and referenced types for ``instance``.

        :param instance: The object to serialize.
        :return: The eval-able constructor expression and the types it needs imported.
        """


# %% DataclassConstructorSerializer


@dataclass
class DataclassConstructorSerializer(ObjectSourceSerializer):
    """
    Emits ``DataclassType(field=value, ...)`` source for dataclass instances whose
    fields are ``None``, a :attr:`supported_scalar_types` value or a nested dataclass,
    and raises :class:`ObjectNotSerializableError` for any other field value.

    .. warning::
        Instances must be ``dataclasses.asdict``-safe: no circular references, and every
        field settable through the constructor.
    """

    @classproperty
    def supported_scalar_types(cls) -> tuple[type, ...]:
        """
        The scalar types emitted directly as field values.
        """
        return ValueSourceEmitter.supported_scalar_types

    def to_source(self, instance: Any) -> ObjectSource:
        if not dataclasses.is_dataclass(instance) or isinstance(instance, type):
            raise ObjectNotSerializableError(instance, self.supported_scalar_types)
        context = SourceEmissionContext()
        expression = ValueSourceEmitter.emit_source(subject=instance, context=context)
        return ObjectSource(
            ast_unparse_expression(expression), context.referenced_types
        )
