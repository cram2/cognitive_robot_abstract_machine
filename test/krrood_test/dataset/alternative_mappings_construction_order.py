from __future__ import annotations

from dataclasses import dataclass
from typing import Self, List, Type

from typing_extensions import Optional

from krrood.ormatic.data_access_objects.alternative_mappings import (
    AlternativeMapping,
    T,
)


@dataclass
class BuildFirst:

    value: str
    backreference_to_entrypoint: Optional[Entrypoint] = None


@dataclass
class BuildFirstAssociation:

    build_first: BuildFirst


@dataclass
class Entrypoint:

    build_first: BuildFirst

    build_first_association: BuildFirstAssociation


@dataclass(eq=False)
class BuildFirstMapping(AlternativeMapping[BuildFirst]):

    value: str
    backreference_to_entrypoint: Optional[Entrypoint] = None

    @classmethod
    def from_domain_object(cls, obj: T) -> Self:
        return cls(obj.value, obj.backreference_to_entrypoint)

    def to_domain_object(self) -> T:
        return BuildFirst(self.value, self.backreference_to_entrypoint)


@dataclass(eq=False)
class EntryPointMapping(AlternativeMapping[Entrypoint]):
    build_first: BuildFirst
    build_first_association: BuildFirstAssociation

    @classmethod
    def from_domain_object(cls, obj: T) -> Self:
        return cls(obj.build_first, obj.build_first_association)

    def to_domain_object(self) -> T:
        return Entrypoint(self.build_first, self.build_first_association)

    @classmethod
    def required_pre_build_classes(cls) -> List[Type]:
        return [BuildFirst, BuildFirstAssociation]


@dataclass
class HoldsAnEntrypoint:

    entrypoint: Entrypoint


@dataclass(eq=False)
class HoldsAnEntrypointMapping(AlternativeMapping[HoldsAnEntrypoint]):
    """
    A mapping that states no classes to wait for, holding one that does.
    """

    entrypoint: Entrypoint

    @classmethod
    def from_domain_object(cls, obj: T) -> Self:
        return cls(obj.entrypoint)

    def to_domain_object(self) -> T:
        return HoldsAnEntrypoint(self.entrypoint)


@dataclass
class OwnsAHolder:
    """
    What refers to a mapping holding another, so the holder is converted along with it.
    """

    holder: HoldsAnEntrypoint


@dataclass
class OneSideOfAHoldingCycle:
    """
    One of two classes whose mappings hold each other.
    """

    other_side: Optional[OtherSideOfAHoldingCycle] = None
    """
    The other side, whose mapping holds this one back.
    """


@dataclass
class OtherSideOfAHoldingCycle:
    """
    The other of two classes whose mappings hold each other.
    """

    one_side: Optional[OneSideOfAHoldingCycle] = None
    """
    The side holding this one.
    """


@dataclass(eq=False)
class OneSideOfAHoldingCycleMapping(AlternativeMapping[OneSideOfAHoldingCycle]):
    """
    A mapping stating no classes to wait for, holding the mapping that holds it.
    """

    other_side: Optional[OtherSideOfAHoldingCycle] = None
    """
    What this mapping holds, and what holds it.
    """

    @classmethod
    def from_domain_object(cls, obj: T) -> Self:
        return cls(obj.other_side)

    def to_domain_object(self) -> T:
        return OneSideOfAHoldingCycle(self.other_side)


@dataclass(eq=False)
class OtherSideOfAHoldingCycleMapping(AlternativeMapping[OtherSideOfAHoldingCycle]):
    """
    The mapping on the other side of the same holding cycle.
    """

    one_side: Optional[OneSideOfAHoldingCycle] = None
    """
    What this mapping holds, and what holds it.
    """

    @classmethod
    def from_domain_object(cls, obj: T) -> Self:
        return cls(obj.one_side)

    def to_domain_object(self) -> T:
        return OtherSideOfAHoldingCycle(self.one_side)
