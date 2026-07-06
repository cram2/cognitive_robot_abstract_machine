from __future__ import annotations

from dataclasses import dataclass
from typing_extensions import Optional, Any


@dataclass(frozen=True)
class SourceReference:
    """
    A reference to the Python source entity — a class, or an attribute of a class — that a
    fragment names.

    Frozen so instances can be safely shared and hashed.
    """

    owner_type: type
    """The Python class this fragment refers to (always set)."""

    attribute: Optional[str] = None
    """Attribute name within *owner_type*; ``None`` means the fragment refers
    to the class itself (e.g. for type-name labels like *"Robot"*)."""

    @classmethod
    def for_type(cls, type_: Any) -> Optional[SourceReference]:
        """
        :param type_: Candidate type (any value accepted; non-types return ``None``).
        :return: A source reference for the class when *t* is a real ``type``, else ``None``.
        """
        return cls(owner_type=type_) if isinstance(type_, type) else None

    @classmethod
    def for_attribute(
        cls, owner: Any, attribute_name: str
    ) -> Optional[SourceReference]:
        """
        :param owner: Candidate owner class (any value; non-types return ``None``).
        :param attribute_name: Canonical attribute name on *owner*.
        :return: A source reference for the attribute when *owner* is a real ``type``, else
            ``None``.
        """
        return (
            cls(owner_type=owner, attribute=attribute_name)
            if isinstance(owner, type)
            else None
        )
