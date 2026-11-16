"""
Dataclass fields whose value is kept somewhere other than the instance's own attribute.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import MISSING, Field, dataclass, field
from itertools import repeat

from typing_extensions import Any

from krrood.patterns.mixin import Mixin


# %% dataclass field descriptor


@dataclass(kw_only=True, eq=False)
class DataclassFieldDescriptor(ABC):
    """
    A data descriptor that manages a field of a dataclass and keeps the field's value
    where :meth:`read` and :meth:`write` say.

    The field stays an ordinary dataclass field, declared with any default and
    :func:`dataclasses.field` arguments. Assign the descriptor to the field's name on
    the class once the class is created, e.g. ``Owner.value = KeptInStorage()``. The
    class inherits :class:`DataclassFieldDescriptorOwner`, so its subclasses keep the
    descriptor.
    """

    @abstractmethod
    def read(self, *, instance: Any) -> Any:
        """
        :param instance: The instance that owns the field.
        :return: The value of the field.
        """

    @abstractmethod
    def write(self, *, instance: Any, value: Any) -> None:
        """
        Keep the value of the field.

        :param instance: The instance that owns the field.
        :param value: The new value of the field.
        """

    def __get__(self, instance: Any | None, owner: type | None = None) -> Any:
        """
        :return: The value of the field, or the descriptor itself when read on the class.
        """
        if instance is None:
            return self
        return self.read(instance=instance)

    def __set__(self, instance: Any, value: Any) -> None:
        """
        Write the value of the field.
        """
        self.write(instance=instance, value=value)


# %% owner of descriptor managed fields


class DataclassFieldDescriptorOwner(Mixin):
    """
    A dataclass whose fields :class:`DataclassFieldDescriptor` instances may manage.

    A subclass may declare such a field again, with a default of its own, without one,
    or with a narrower type, and the field stays managed by the inherited descriptor.
    """

    def __init_subclass__(cls, **kwargs: Any) -> None:
        """
        Turn every field this subclass declares over an inherited descriptor into a
        :class:`~dataclasses.Field` that leaves no class attribute in its place.
        """
        super().__init_subclass__(**kwargs)
        for name in vars(cls).get("__annotations__", {}):
            if not isinstance(
                cls._inherited_attribute(name=name), DataclassFieldDescriptor
            ) or isinstance(vars(cls).get(name), DataclassFieldDescriptor):
                continue
            setattr(cls, name, cls._field_without_class_attribute(name=name))

    @classmethod
    def _inherited_attribute(cls, *, name: str) -> Any:
        """
        :param name: The name of an attribute.
        :return: The attribute the nearest base class defines under the name, or
            ``None`` if no base class does.
        """
        return next(
            (vars(base)[name] for base in cls.__mro__[1:] if name in vars(base)),
            None,
        )

    @classmethod
    def _field_without_class_attribute(cls, *, name: str) -> Field:
        """
        :param name: The name of a field this class declares.
        :return: The field's declaration with its default given by a factory, which
            :func:`dataclasses.dataclass` does not set as a class attribute.
        """
        declaration = vars(cls).get(name, MISSING)
        if not isinstance(declaration, Field):
            declaration = field(default=declaration)
        if declaration.default is MISSING:
            return declaration
        declaration.default_factory = repeat(declaration.default).__next__
        declaration.default = MISSING
        return declaration
