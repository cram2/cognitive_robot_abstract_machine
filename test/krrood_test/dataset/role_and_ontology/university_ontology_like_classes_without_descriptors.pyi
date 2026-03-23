from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set, List, TypeVar

from krrood.entity_query_language.predicate import (
    Symbol,  # type: ignore
)
from krrood.patterns.role.role import Role

@dataclass(eq=False)
class HasName:
    name: str

    def __hash__(self): ...
    def __eq__(self, other): ...

@dataclass(eq=False)
class RecognizedGroup(HasName, Symbol):
    members: Set[Person] = field(default_factory=set)
    sub_organization_of: List[RecognizedGroup] = field(default_factory=list)

@dataclass(eq=False)
class Company(RecognizedGroup): ...

@dataclass(eq=False)
class Country(RecognizedGroup): ...

@dataclass(unsafe_hash=True)
class Course(HasName, Symbol): ...

@dataclass(eq=False)
class PersonMixin(HasName, Symbol):
    name: str = field(init=False)
    works_for: RecognizedGroup = field(init=False)
    member_of: List[RecognizedGroup] = field(init=False)
    teacher_of: List[Course] = field(init=False)
    members: Set[Person] = field(init=False)
    sub_organization_of: List[RecognizedGroup] = field(init=False)
    head_of: RecognizedGroup = field(init=False)
    representative_of: RecognizedGroup = field(init=False)
    delegate_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class Person(PersonMixin):
    works_for: RecognizedGroup = None
    member_of: List[RecognizedGroup] = field(default_factory=list)

@dataclass(eq=False)
class SubclassOfARoleTakerMixin(PersonMixin):
    introduced_attribute: str = field(init=False)
    head_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class SubclassOfARoleTaker(SubclassOfARoleTakerMixin):
    introduced_attribute: str = field(default="", kw_only=True)

TPerson = TypeVar("TPerson", bound=Person)

@dataclass(eq=False)
class CEOAsFirstRoleMixin(PersonMixin, Role[TPerson], Symbol):
    person: TPerson = field(init=False)
    head_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class CEOAsFirstRole(CEOAsFirstRoleMixin[TPerson]):
    person: TPerson = field(kw_only=True)
    head_of: RecognizedGroup = None

    @classmethod
    def role_taker_attribute(cls) -> TPerson: ...

TSubclassOfARoleTaker = TypeVar("TSubclassOfARoleTaker", bound=SubclassOfARoleTaker)

@dataclass(eq=False)
class SubclassOfRoleThatUpdatesRoleTakerType(CEOAsFirstRole[TSubclassOfARoleTaker]): ...

@dataclass(eq=False)
class DirectDiamondShapedInheritanceWhereOneIsRole(
    PersonMixin,
    Role[TPerson],
):
    person: TPerson = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TPerson: ...

@dataclass(eq=False)
class InDirectDiamondShapedInheritanceWhereOneIsRole(
    RecognizedGroup, PersonMixin, Role[TPerson]
):
    person: TPerson = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TPerson: ...

@dataclass(eq=False)
class ProfessorAsFirstRole(
    PersonMixin,
    Role[TPerson],
):
    person: TPerson = field(kw_only=True)
    teacher_of: List[Course] = field(default_factory=list, kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TPerson: ...

@dataclass(eq=False)
class AssociateProfessorAsSubClassOfARoleInSameModule(
    ProfessorAsFirstRole[TPerson]
): ...

TCEOAsFirstRole = TypeVar("TCEOAsFirstRole", bound=CEOAsFirstRole)

@dataclass(eq=False)
class RepresentativeAsSecondRoleMixin(
    CEOAsFirstRoleMixin, Role[TCEOAsFirstRole], Symbol
):
    ceo: TCEOAsFirstRole = field(init=False)
    representative_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class RepresentativeAsSecondRole(RepresentativeAsSecondRoleMixin[TCEOAsFirstRole]):
    ceo: TCEOAsFirstRole = field(kw_only=True)
    representative_of: RecognizedGroup = field(default=None, kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TCEOAsFirstRole: ...

TRepresentativeAsSecondRole = TypeVar(
    "TRepresentativeAsSecondRole", bound=RepresentativeAsSecondRole
)

@dataclass(eq=False)
class DelegateAsThirdRole(
    RepresentativeAsSecondRoleMixin,
    Role[TRepresentativeAsSecondRole],
):
    representative: TRepresentativeAsSecondRole = field(kw_only=True)

    delegate_of: RecognizedGroup = field(kw_only=True, default=None)

    @classmethod
    def role_taker_attribute(cls) -> TRepresentativeAsSecondRole: ...
