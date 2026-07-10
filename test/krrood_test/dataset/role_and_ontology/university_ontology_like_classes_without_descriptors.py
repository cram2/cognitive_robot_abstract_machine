from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Self, Set, List, TypeVar

from test.krrood_test.dataset.role_and_ontology.role_takers_in_another_module import (
    RoleTakerInAnotherModule,
)
from krrood.entity_query_language.predicate import Symbol
from krrood.patterns.role import Role, factory_method


@dataclass(eq=False)
class HasName:
    name: str
    default_name: str = field(default="", kw_only=True)

    def __hash__(self):
        return hash(self.name)

    def __eq__(self, other):
        return self.name == other.name


@dataclass(eq=False)
class RecognizedGroup(HasName, Symbol):
    members: Set[PersonInRoleAndOntology] = field(default_factory=set)
    sub_organization_of: List[RecognizedGroup] = field(default_factory=list)


@dataclass(eq=False)
class Company(RecognizedGroup): ...


@dataclass(eq=False)
class Country(RecognizedGroup): ...


@dataclass(unsafe_hash=True)
class Course(HasName, Symbol): ...


@dataclass(eq=False)
class PersonInRoleAndOntology(HasName, Symbol):
    works_for: RecognizedGroup = None
    member_of: List[RecognizedGroup] = field(default_factory=list)

    def method_in_person(self) -> RecognizedGroup:
        return self.works_for

    def method_2_in_person(self) -> List[RecognizedGroup]:
        if self.works_for:
            return [self.works_for]
        return self.member_of

    @classmethod
    def from_name(cls, name: str) -> Self:
        """A factory classmethod detected via its ``-> Self`` return annotation."""
        return cls(name=name)

    @factory_method
    @classmethod
    def spawn(cls):
        """A factory classmethod detected only via the ``@factory_method`` marker."""
        return cls(name="spawned")

    @classmethod
    def describe(cls) -> str:
        """An ordinary classmethod that is not a factory (its return type is not the class)."""
        return cls.__name__


@dataclass(eq=False)
class SubclassOfARoleTaker(PersonInRoleAndOntology):
    introduced_attribute: str = field(default="", kw_only=True)


TPersonInRoleAndOntology = TypeVar(
    "TPersonInRoleAndOntology", bound=PersonInRoleAndOntology
)


@dataclass(eq=False)
class CEOAsFirstRole(Role[TPersonInRoleAndOntology], Symbol):
    head_of: RecognizedGroup = None


@dataclass(eq=False)
class CEOThatOverridesFactory(Role[TPersonInRoleAndOntology], Symbol):

    @classmethod
    def from_name(cls, name: str) -> Self:
        """Overrides the taker factory so the role is preserved instead of being dropped."""
        return cls(role_taker=PersonInRoleAndOntology(name=name))


TSubclassOfARoleTaker = TypeVar("TSubclassOfARoleTaker", bound=SubclassOfARoleTaker)


@dataclass(eq=False)
class SubclassOfRoleThatUpdatesRoleTakerType(CEOAsFirstRole[TSubclassOfARoleTaker]): ...


@dataclass(eq=False)
class ProfessorAsFirstRole(Role[TPersonInRoleAndOntology], Symbol):
    teacher_of: List[Course] = field(default_factory=list, kw_only=True)


@dataclass(eq=False)
class AssociateProfessorAsSubClassOfARoleInSameModule(
    ProfessorAsFirstRole[TPersonInRoleAndOntology]
): ...


TCEOAsFirstRole = TypeVar("TCEOAsFirstRole", bound=CEOAsFirstRole)


@dataclass(eq=False)
class RepresentativeAsSecondRole(Role[TCEOAsFirstRole], Symbol):
    representative_of: RecognizedGroup = field(default=None, kw_only=True)


TRepresentativeAsSecondRole = TypeVar(
    "TRepresentativeAsSecondRole", bound=RepresentativeAsSecondRole
)


@dataclass(eq=False)
class DelegateAsThirdRole(Role[TRepresentativeAsSecondRole], Symbol):
    delegate_of: RecognizedGroup = field(kw_only=True, default=None)


@dataclass(eq=False)
class RoleForTakerInAnotherModule(Role[RoleTakerInAnotherModule]):
    introduced_attribute: str = field(default="", kw_only=True)
    same_module_annotated_introduced_attribute: DelegateAsThirdRole = field(
        default=None, kw_only=True
    )
