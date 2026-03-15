from __future__ import annotations

from dataclasses import dataclass, field, Field

from typing_extensions import Set, List, TypeVar

from krrood.patterns.role import Role
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
class PersonMixin(HasName, Symbol):
    works_for: RecognizedGroup = field(default=None, kw_only=True)
    member_of: List[RecognizedGroup] = field(default_factory=list, kw_only=True)
    head_of: RecognizedGroup = field(init=False)
    delegate_of: RecognizedGroup = field(init=False)
    members: Set[Person] = field(init=False)
    sub_organization_of: List[RecognizedGroup] = field(init=False)
    teacher_of: List[Course] = field(init=False)
    representative_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class Person(PersonMixin): ...

@dataclass(eq=False)
class SubclassOfARoleTakerMixin(Person):
    introduced_attribute: str = field(default="", kw_only=True)

@dataclass(eq=False)
class SubclassOfARoleTaker(SubclassOfARoleTakerMixin): ...

TPerson = TypeVar("TPerson", bound=Person)

@dataclass
class RoleForPerson(Role[TPerson], PersonMixin):
    person: TPerson = field(kw_only=True)
    name: str = field(init=False)
    works_for: RecognizedGroup = field(init=False)
    member_of: List[RecognizedGroup] = field(init=False)

    @classmethod
    def role_taker_attribute(cls) -> Field: ...

@dataclass(eq=False)
class DirectDiamondShapedInheritanceWhereOneIsRole(RoleForPerson[TPerson]): ...

@dataclass(eq=False)
class InDirectDiamondShapedInheritanceWhereOneIsRole(
    RoleForPerson[TPerson],
    RecognizedGroup,
): ...

@dataclass(eq=False)
class CEOAsFirstRole(RoleForPerson[TPerson]):
    # Original Owner of the head_of field
    head_of: RecognizedGroup = field(default=None, kw_only=True)

TSubclassOfARoleTaker = TypeVar("TSubclassOfARoleTaker", bound=SubclassOfARoleTaker)

@dataclass
class CEOAsFirstRoleAsRoleForSubClassOfARoleTaker(
    CEOAsFirstRole[TSubclassOfARoleTaker], SubclassOfARoleTakerMixin
):
    introduced_attribute: str = field(init=False)

@dataclass(eq=False)
class SubclassOfRoleThatUpdatesRoleTakerType(
    CEOAsFirstRoleAsRoleForSubClassOfARoleTaker
): ...

@dataclass(eq=False)
class ProfessorAsFirstRole(RoleForPerson):
    # Original Owner of the teacher_of field
    teacher_of: List[Course] = field(default_factory=list, kw_only=True)

@dataclass(eq=False)
class AssociateProfessorAsSubClassOfARoleInSameModule(ProfessorAsFirstRole): ...

@dataclass
class RoleForCEOAsFirstRole(CEOAsFirstRole):
    ceo: CEOAsFirstRole = field(kw_only=True)
    person: Person = field(init=False)
    head_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class RepresentativeAsSecondRole(RoleForCEOAsFirstRole):
    # Original Owner of the representative_of field
    representative_of: RecognizedGroup = field(default=None, kw_only=True)

@dataclass
class RoleForRepresentativeAsSecondRole(RepresentativeAsSecondRole):
    representative: RepresentativeAsSecondRole = field(kw_only=True)
    ceo: CEOAsFirstRole = field(init=False)
    representative_of: RecognizedGroup = field(init=False)

@dataclass(eq=False)
class DelegateAsThirdRole(RoleForRepresentativeAsSecondRole):
    # Original Owner of the delegate_of field
    delegate_of: RecognizedGroup = field(default=None, kw_only=True)
