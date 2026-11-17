from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set, List, Type

from krrood.patterns.role import Role
from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.mixins import (
    HasInverseProperty,
    TransitiveProperty,
)
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)


@dataclass(kw_only=True, eq=False)
class RecognizedGroup(Symbol):
    name: str

    members: Set[Person] = field(default_factory=set)
    sub_organization_of: List[RecognizedGroup] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)


@dataclass(kw_only=True, eq=False)
class Company(RecognizedGroup): ...


@dataclass(kw_only=True, eq=False)
class Country(RecognizedGroup): ...


@dataclass(kw_only=True, eq=False)
class Person(Symbol):
    name: str
    works_for: RecognizedGroup = None
    member_of: List[RecognizedGroup] = field(default_factory=list)

    def __hash__(self):
        return hash(self.name)


@dataclass(kw_only=True, eq=False)
class CEOAsFirstRole(Role[Person], Symbol):
    head_of: RecognizedGroup = None


@dataclass(kw_only=True, eq=False)
class RepresentativeAsSecondRole(Role[CEOAsFirstRole], Symbol):
    representative_of: RecognizedGroup = None


@dataclass(kw_only=True, eq=False)
class DelegateAsThirdRole(Role[RepresentativeAsSecondRole], Symbol):
    delegate_of: RecognizedGroup = field(kw_only=True, default=None)


@dataclass(kw_only=True)
class Member(PropertyDescriptor, HasInverseProperty):
    @classmethod
    def get_inverse(cls) -> Type[MemberOf]:
        return MemberOf


@dataclass(kw_only=True)
class MemberOf(PropertyDescriptor, HasInverseProperty):
    @classmethod
    def get_inverse(cls) -> Type[Member]:
        return Member


@dataclass(kw_only=True)
class WorksFor(MemberOf):
    pass


@dataclass(kw_only=True)
class HeadOf(WorksFor):
    pass


@dataclass(kw_only=True)
class RepresentativeOf(HeadOf):
    pass


@dataclass(kw_only=True)
class DelegateOf(RepresentativeOf):
    pass


@dataclass(kw_only=True)
class SubOrganizationOf(PropertyDescriptor, TransitiveProperty): ...


# Person fields' descriptors
Person.works_for = WorksFor(domain=Person, field_name="works_for")
Person.member_of = MemberOf(domain=Person, field_name="member_of")

# CEO fields' descriptors
CEOAsFirstRole.head_of = HeadOf(domain=CEOAsFirstRole, field_name="head_of")

# Representative fields' descriptors
RepresentativeAsSecondRole.representative_of = RepresentativeOf(
    domain=RepresentativeAsSecondRole, field_name="representative_of"
)

# Delegate fields' descriptors
DelegateAsThirdRole.delegate_of = DelegateOf(
    domain=DelegateAsThirdRole, field_name="delegate_of"
)

# RecognizedGroup fields' descriptors
RecognizedGroup.members = Member(domain=RecognizedGroup, field_name="members")
RecognizedGroup.sub_organization_of = SubOrganizationOf(
    domain=RecognizedGroup, field_name="sub_organization_of"
)
