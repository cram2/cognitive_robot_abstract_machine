from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set, List, TypeVar

from .role_takers_in_another_module import (
    RoleTakerInAnotherModule,
    RoleTakerInAnotherModuleMixin,
)
from krrood.patterns.role.role import Role
from krrood.symbol_graph.symbol_graph import Symbol

@dataclass(eq=False)
class HasName:
    name: str

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
class PersonRoleAttributes:
    head_of: RecognizedGroup = field(init=False)
    delegate_of: RecognizedGroup = field(init=False)
    members: Set[Person] = field(init=False)
    sub_organization_of: List[RecognizedGroup] = field(init=False)
    teacher_of: List[Course] = field(init=False)
    representative_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class PersonMixin(PersonRoleAttributes, HasName, Symbol):
    name: str = field(init=False)
    works_for: RecognizedGroup = field(init=False)
    member_of: List[RecognizedGroup] = field(init=False)

@dataclass(eq=False)
class Person(PersonRoleAttributes, HasName, Symbol):
    works_for: RecognizedGroup = None
    member_of: List[RecognizedGroup] = field(default_factory=list)

@dataclass(eq=False)
class SubclassOfARoleTakerMixin(PersonMixin):
    introduced_attribute: str = field(init=False)

@dataclass(eq=False)
class SubclassOfARoleTaker(Person):
    introduced_attribute: str = field(default="", kw_only=True)

TPerson = TypeVar("TPerson", bound=Person)

@dataclass(eq=False)
class DirectDiamondShapedInheritanceWhereOneIsRole(PersonMixin, Role[TPerson]):
    person: TPerson = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TPerson: ...

@dataclass(eq=False)
class InDirectDiamondShapedInheritanceWhereOneIsRole(
    RecognizedGroup,
    PersonMixin,
    Role[TPerson],
):
    person: TPerson = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TPerson: ...

@dataclass(eq=False)
class CEOAsFirstRoleMixin(PersonMixin, Role[TPerson], Symbol):
    person: TPerson = field(init=False)
    head_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class CEOAsFirstRole(PersonMixin, Role[TPerson], Symbol):
    person: TPerson = field(kw_only=True)
    # Original Owner of the head_of field
    head_of: RecognizedGroup = None

    @classmethod
    def role_taker_attribute(cls) -> TPerson: ...

TSubclassOfARoleTaker = TypeVar("TSubclassOfARoleTaker", bound=SubclassOfARoleTaker)

@dataclass(eq=False)
class SubclassOfRoleThatUpdatesRoleTakerType(
    SubclassOfARoleTakerMixin, CEOAsFirstRole[TSubclassOfARoleTaker]
): ...

@dataclass(eq=False)
class ProfessorAsFirstRole(PersonMixin, Role[TPerson]):
    person: TPerson = field(kw_only=True)
    # Original Owner of the teacher_of field
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
    # Original Owner of the representative_of field
    representative_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class RepresentativeAsSecondRole(CEOAsFirstRoleMixin, Role[TCEOAsFirstRole], Symbol):
    ceo: TCEOAsFirstRole = field(kw_only=True)
    # Original Owner of the representative_of field
    representative_of: RecognizedGroup = field(default=None, kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TCEOAsFirstRole: ...

TRepresentativeAsSecondRole = TypeVar(
    "TRepresentativeAsSecondRole", bound=RepresentativeAsSecondRole
)

@dataclass(eq=False)
class DelegateAsThirdRole(
    RepresentativeAsSecondRoleMixin, Role[TRepresentativeAsSecondRole]
):
    representative: TRepresentativeAsSecondRole = field(kw_only=True)
    # Original Owner of the delegate_of field
    delegate_of: RecognizedGroup = field(default=None, kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TRepresentativeAsSecondRole: ...

@dataclass(eq=False)
class RoleForTakerInAnotherModule(
    RoleTakerInAnotherModuleMixin, Role[RoleTakerInAnotherModule]
):
    taker: RoleTakerInAnotherModule = field(kw_only=True)
    introduced_attribute: str = field(default="", kw_only=True)
    same_module_annotated_introduced_attribute: DelegateAsThirdRole = field(
        default=None, kw_only=True
    )

    @classmethod
    def role_taker_attribute(cls) -> RoleTakerInAnotherModule: ...
