"""Families of emitters that build the :mod:`ast` expression rebuilding an
object: each concrete emitter binds the type it handles, and its family picks
the one with the most specific type that applies to a subject."""

from __future__ import annotations

import ast
import dataclasses
import enum
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from types import NoneType

from typing_extensions import Any, Callable, Generic, TypeVar

from krrood.code_generation.ast_helpers import (
    ast_call,
    ast_enum_member,
)
from krrood.code_generation.exceptions import (
    NoApplicableSourceEmitterError,
    ObjectNotSerializableError,
)
from krrood.ormatic.utils import classproperty
from krrood.patterns.specificity_ranking import (
    SpecificityRankedAlternative,
    concrete_subclasses,
    mro_depth,
)
from krrood.patterns.subclass_safe_generic import SubClassSafeGeneric

EmittedType = TypeVar("EmittedType")

# %% emission context


@dataclass(kw_only=True)
class SourceEmissionContext:
    """What the emitters of one generated source share."""

    referenced_types: set[type] = field(default_factory=set)
    """The types the emitted source names, which the generated module has to
    import."""


# %% emitter family base


@dataclass(kw_only=True)
class SourceEmitter(
    Generic[EmittedType], SubClassSafeGeneric, SpecificityRankedAlternative, ABC
):
    """Builds the expression that rebuilds a subject of its bound type, within
    a family of emitters that :meth:`emit_source` chooses from."""

    context: SourceEmissionContext
    """The state shared by every emitter of the generated source."""

    @classproperty
    def emitted_type(cls) -> type:
        """The type of the subjects this emitter handles."""
        return cls.get_type_of_generic_parameter(parameter=EmittedType)

    @classproperty
    def family(cls) -> type[SourceEmitter]:
        """The direct subclass of :class:`SourceEmitter` this emitter descends
        from, whose concrete subclasses :meth:`emit_source` chooses from."""
        return next(
            ancestor for ancestor in cls.__mro__ if SourceEmitter in ancestor.__bases__
        )

    @classmethod
    def alternatives(cls) -> list[type[SourceEmitter]]:
        """Like :meth:`SpecificityRankedAlternative.alternatives`, but the
        concrete subclasses of :attr:`family` that bind the emitted type."""
        return [
            emitter
            for emitter in concrete_subclasses(cls.family)
            if not isinstance(emitter.get_generic_type_parameters()[0], TypeVar)
        ]

    @classproperty
    def guard(cls) -> Callable[..., bool]:
        """:meth:`can_emit_source_for_object`."""
        return cls.can_emit_source_for_object

    @classproperty
    def specificity(cls) -> int:
        """Like :attr:`SpecificityRankedAlternative.specificity`, but the class
        depth of :attr:`emitted_type`, so the emitter of the most specific type
        that applies wins."""
        return mro_depth(cls.emitted_type)

    @classmethod
    def unsupported_subject_error(cls, *, subject: Any) -> Exception:
        """The error raised when no emitter of the family applies to a subject.

        :param subject: The subject no emitter applies to.
        :return: The error to raise.
        """
        return NoApplicableSourceEmitterError(subject=subject, family=cls.family)

    @classmethod
    def can_emit_source_for_object(
        cls, *, subject: Any, context: SourceEmissionContext
    ) -> bool:
        """Whether this emitter can rebuild ``subject``.

        :param subject: The object to rebuild.
        :param context: The shared emission state.
        :return: Whether ``subject`` is of :attr:`emitted_type`.
        """
        return isinstance(subject, cls.emitted_type)

    @classmethod
    def emit_source(cls, *, subject: Any, context: SourceEmissionContext) -> ast.expr:
        """Build the expression that rebuilds ``subject`` with the emitter of
        :attr:`family` whose type is the most specific among those that apply.

        :param subject: The object to rebuild.
        :param context: The shared emission state.
        :return: The expression that rebuilds ``subject``.
        """
        emitter = cls.most_applicable(subject=subject, context=context)
        if emitter is None:
            raise cls.unsupported_subject_error(subject=subject)
        return emitter(context=context).emit(subject=subject)

    @abstractmethod
    def emit(self, *, subject: EmittedType) -> ast.expr:
        """Build the expression that rebuilds ``subject``.

        :param subject: An object of :attr:`emitted_type`.
        :return: The expression that rebuilds ``subject``.
        """


# %% values


@dataclass(kw_only=True)
class ValueSourceEmitter(SourceEmitter[EmittedType], ABC):
    """The family of emitters for plain values: ``None``, scalars, enum members
    and dataclass instances made of these."""

    @classproperty
    def supported_scalar_types(cls) -> tuple[type, ...]:
        """The scalar types the family emits, besides ``None`` and dataclass
        instances."""
        return tuple(
            emitter.emitted_type
            for emitter in ValueSourceEmitter.alternatives()
            if issubclass(emitter, (ConstantSourceEmitter, EnumSourceEmitter))
            and emitter is not NoneSourceEmitter
        )

    @classmethod
    def unsupported_subject_error(cls, *, subject: Any) -> Exception:
        """Like :meth:`SourceEmitter.unsupported_subject_error`, but the
        :class:`ObjectNotSerializableError` that names the supported scalar
        types."""
        return ObjectNotSerializableError(subject, cls.supported_scalar_types)


@dataclass(kw_only=True)
class ConstantSourceEmitter(ValueSourceEmitter[EmittedType], ABC):
    """Emits a value that is written as a Python constant."""

    @classmethod
    def can_emit_source_for_object(
        cls, *, subject: Any, context: SourceEmissionContext
    ) -> bool:
        """Like :meth:`SourceEmitter.can_emit_source_for_object`, but never for an enum member,
        which :class:`EnumSourceEmitter` writes by its name even when it is
        also a constant, as an ``IntEnum`` member is."""
        return super().can_emit_source_for_object(
            subject=subject, context=context
        ) and not isinstance(subject, enum.Enum)

    def emit(self, *, subject: EmittedType) -> ast.expr:
        return ast.Constant(value=subject)


@dataclass(kw_only=True)
class NoneSourceEmitter(ConstantSourceEmitter[NoneType]):
    """Emits ``None``."""


@dataclass(kw_only=True)
class BoolSourceEmitter(ConstantSourceEmitter[bool]):
    """Emits a boolean."""


@dataclass(kw_only=True)
class IntSourceEmitter(ConstantSourceEmitter[int]):
    """Emits an integer."""


@dataclass(kw_only=True)
class FloatSourceEmitter(ConstantSourceEmitter[float]):
    """Emits a floating-point number."""


@dataclass(kw_only=True)
class StrSourceEmitter(ConstantSourceEmitter[str]):
    """Emits a string."""


@dataclass(kw_only=True)
class EnumSourceEmitter(ValueSourceEmitter[enum.Enum]):
    """Emits an enum member as ``EnumType.MEMBER``, which needs its enum type
    imported."""

    def emit(self, *, subject: enum.Enum) -> ast.expr:
        self.context.referenced_types.add(type(subject))
        return ast_enum_member(subject)


@dataclass(kw_only=True)
class DataclassSourceEmitter(ValueSourceEmitter[object]):
    """Emits a dataclass instance as a call of its constructor with every field
    passed by keyword.

    .. warning::
        Every field has to be settable through the constructor.
    """

    @classmethod
    def can_emit_source_for_object(
        cls, *, subject: Any, context: SourceEmissionContext
    ) -> bool:
        """Whether ``subject`` is a dataclass instance, not a dataclass
        type."""
        return dataclasses.is_dataclass(subject) and not isinstance(subject, type)

    def emit(self, *, subject: Any) -> ast.expr:
        self.context.referenced_types.add(type(subject))
        return ast_call(
            type(subject).__name__,
            [],
            {
                instance_field.name: self.emit_source(
                    subject=getattr(subject, instance_field.name),
                    context=self.context,
                )
                for instance_field in dataclasses.fields(subject)
            },
        )
