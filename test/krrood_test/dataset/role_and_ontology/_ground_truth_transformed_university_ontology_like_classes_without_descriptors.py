from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set, List, TypeVar

from test.krrood_test.dataset.role_and_ontology.role_mixins._ground_truth_role_takers_in_another_module_role_mixins import (
    RoleForRoleTakerInAnotherModule,
)
from krrood.patterns.role import HasRoles
from test.krrood_test.dataset.role_and_ontology.role_mixins._ground_truth_university_ontology_like_classes_without_descriptors_role_mixins import (
    RoleForPersonInRoleAndOntology,
    RoleForSubclassOfARoleTaker,
    RoleForCEOAsFirstRole,
    RoleForRepresentativeAsSecondRole,
)
from test.krrood_test.dataset.role_and_ontology._ground_truth_transformed_role_takers_in_another_module import (
    RoleTakerInAnotherModule,
)
from krrood.entity_query_language.factories import variable_from
from krrood.entity_query_language.predicate import (
    Symbol,
    Predicate,  # type: ignore
    HasType,  # type: ignore
    HasTypes,  # type: ignore
    length,  # type: ignore
)
from krrood.patterns.role.role import Role


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
class PersonInRoleAndOntology(HasRoles, HasName, Symbol):
    works_for: RecognizedGroup = None
    member_of: List[RecognizedGroup] = field(default_factory=list)

    def method_in_person(self) -> RecognizedGroup:
        return self.works_for

    def method_2_in_person(self) -> List[RecognizedGroup]:
        if self.works_for:
            return [self.works_for]
        return self.member_of


@dataclass(eq=False)
class SubclassOfARoleTaker(PersonInRoleAndOntology):
    introduced_attribute: str = field(default="", kw_only=True)


TPersonInRoleAndOntology = TypeVar(
    "TPersonInRoleAndOntology", bound=PersonInRoleAndOntology
)


@dataclass(eq=False)
class CEOAsFirstRole(
    Role[TPersonInRoleAndOntology], RoleForPersonInRoleAndOntology, Symbol
):
    person: TPersonInRoleAndOntology = field(kw_only=True)
    head_of: RecognizedGroup = None

    @classmethod
    def role_taker_attribute(cls) -> TPersonInRoleAndOntology:
        return variable_from(cls).person


TSubclassOfARoleTaker = TypeVar("TSubclassOfARoleTaker", bound=SubclassOfARoleTaker)


@dataclass(eq=False)
class SubclassOfRoleThatUpdatesRoleTakerType(
    CEOAsFirstRole[TSubclassOfARoleTaker], RoleForSubclassOfARoleTaker
): ...


@dataclass(eq=False)
class DirectDiamondShapedInheritanceWhereOneIsRole(
    Role[TPersonInRoleAndOntology], RoleForPersonInRoleAndOntology, HasName
):
    person: TPersonInRoleAndOntology = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TPersonInRoleAndOntology:
        return variable_from(cls).person


@dataclass(eq=False)
class InDirectDiamondShapedInheritanceWhereOneIsRole(
    RecognizedGroup, Role[TPersonInRoleAndOntology], RoleForPersonInRoleAndOntology
):
    person: TPersonInRoleAndOntology = field(kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TPersonInRoleAndOntology:
        return variable_from(cls).person


@dataclass(eq=False)
class ProfessorAsFirstRole(
    Role[TPersonInRoleAndOntology], RoleForPersonInRoleAndOntology, Symbol
):
    person: TPersonInRoleAndOntology = field(kw_only=True)
    teacher_of: List[Course] = field(default_factory=list, kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TPersonInRoleAndOntology:
        return variable_from(cls).person


@dataclass(eq=False)
class AssociateProfessorAsSubClassOfARoleInSameModule(
    ProfessorAsFirstRole[TPersonInRoleAndOntology], RoleForPersonInRoleAndOntology
): ...


TCEOAsFirstRole = TypeVar("TCEOAsFirstRole", bound=CEOAsFirstRole)


@dataclass(eq=False)
class RepresentativeAsSecondRole(Role[TCEOAsFirstRole], RoleForCEOAsFirstRole, Symbol):
    ceo: TCEOAsFirstRole = field(kw_only=True)
    representative_of: RecognizedGroup = field(default=None, kw_only=True)

    @classmethod
    def role_taker_attribute(cls) -> TCEOAsFirstRole:
        return variable_from(cls).ceo


TRepresentativeAsSecondRole = TypeVar(
    "TRepresentativeAsSecondRole", bound=RepresentativeAsSecondRole
)


@dataclass(eq=False)
class DelegateAsThirdRole(
    Role[TRepresentativeAsSecondRole], RoleForRepresentativeAsSecondRole, Symbol
):
    representative: TRepresentativeAsSecondRole = field(kw_only=True)

    delegate_of: RecognizedGroup = field(kw_only=True, default=None)

    @classmethod
    def role_taker_attribute(cls) -> TRepresentativeAsSecondRole:
        return variable_from(cls).representative


@dataclass(eq=False)
class RoleForTakerInAnotherModule(
    Role[RoleTakerInAnotherModule], RoleForRoleTakerInAnotherModule
):
    taker: RoleTakerInAnotherModule = field(kw_only=True)
    introduced_attribute: str = field(default="", kw_only=True)
    same_module_annotated_introduced_attribute: DelegateAsThirdRole = field(
        default=None, kw_only=True
    )

    @classmethod
    def role_taker_attribute(cls) -> RoleTakerInAnotherModule:
        return variable_from(cls).taker
