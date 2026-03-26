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
