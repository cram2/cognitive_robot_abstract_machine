from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import List, TYPE_CHECKING

if TYPE_CHECKING:
    from test.krrood_test.dataset.role_and_ontology.university_ontology_like_classes_without_descriptors import (
        HasName,
        RecognizedGroup,
        TPersonInRoleAndOntology,
        TSubclassOfARoleTaker,
        TCEOAsFirstRole,
        TRepresentativeAsSecondRole,
    )


@dataclass(eq=False)
class RoleForHasName(ABC):

    @property
    @abstractmethod
    def role_taker(self) -> HasName: ...

    @property
    def name(self) -> str:
        return self.role_taker.name

    @name.setter
    def name(self, value: str):
        self.role_taker.name = value

    @property
    def default_name(self) -> str:
        return self.role_taker.default_name

    @default_name.setter
    def default_name(self, value: str):
        self.role_taker.default_name = value

    def __eq__(self, other):
        return self.role_taker.__eq__(other)

    def __hash__(self):
        return self.role_taker.__hash__()


@dataclass(eq=False)
class RoleForPersonInRoleAndOntology(RoleForHasName, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> TPersonInRoleAndOntology: ...

    @property
    def works_for(self) -> RecognizedGroup:
        return self.role_taker.works_for

    @works_for.setter
    def works_for(self, value: RecognizedGroup):
        self.role_taker.works_for = value

    @property
    def member_of(self) -> List[RecognizedGroup]:
        return self.role_taker.member_of

    @member_of.setter
    def member_of(self, value: List[RecognizedGroup]):
        self.role_taker.member_of = value

    def method_in_person(self) -> RecognizedGroup:
        return self.role_taker.method_in_person()

    def method_2_in_person(self) -> List[RecognizedGroup]:
        return self.role_taker.method_2_in_person()


@dataclass(eq=False)
class RoleForSubclassOfARoleTaker(RoleForPersonInRoleAndOntology, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> TSubclassOfARoleTaker: ...

    @property
    def introduced_attribute(self) -> str:
        return self.role_taker.introduced_attribute

    @introduced_attribute.setter
    def introduced_attribute(self, value: str):
        self.role_taker.introduced_attribute = value


@dataclass(eq=False)
class RoleForCEOAsFirstRole(RoleForPersonInRoleAndOntology, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> TCEOAsFirstRole: ...

    @property
    def person(self) -> TPersonInRoleAndOntology:
        return self.role_taker.person

    @person.setter
    def person(self, value: TPersonInRoleAndOntology):
        self.role_taker.person = value

    @property
    def head_of(self) -> RecognizedGroup:
        return self.role_taker.head_of

    @head_of.setter
    def head_of(self, value: RecognizedGroup):
        self.role_taker.head_of = value


@dataclass(eq=False)
class RoleForRepresentativeAsSecondRole(RoleForCEOAsFirstRole, ABC):

    @property
    @abstractmethod
    def role_taker(self) -> TRepresentativeAsSecondRole: ...

    @property
    def ceo(self) -> TCEOAsFirstRole:
        return self.role_taker.ceo

    @ceo.setter
    def ceo(self, value: TCEOAsFirstRole):
        self.role_taker.ceo = value

    @property
    def representative_of(self) -> RecognizedGroup:
        return self.role_taker.representative_of

    @representative_of.setter
    def representative_of(self, value: RecognizedGroup):
        self.role_taker.representative_of = value
