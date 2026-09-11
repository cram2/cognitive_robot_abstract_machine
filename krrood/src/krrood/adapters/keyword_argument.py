from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Dict, Self

from krrood.utils import get_full_class_name


@dataclass
class SerializationKeywordArgument:
    """
    An object that travels through nested ``to_json`` or ``from_json`` calls in their
    keyword arguments.

    The top-level caller passes :meth:`create_kwargs`, and every call hands its keyword
    arguments on to the calls it makes.
    """

    @classmethod
    def _keyword(cls) -> str:
        """
        :return: The keyword argument this type of object is passed as, distinct for
            every type.
        """
        return get_full_class_name(cls)

    def create_kwargs(self) -> Dict[str, Self]:
        """
        :return: Keyword arguments carrying this object, to pass to the top-level call.
        """
        return {self._keyword(): self}
