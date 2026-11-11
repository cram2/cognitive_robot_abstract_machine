"""
A person with roles that declare the same property with narrower ranges: a sports lover loves sports, a basketball
lover loves basketball.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Set

from krrood.entity_query_language.predicate import Symbol
from krrood.ontomatic.property_descriptor.property_descriptor import (
    PropertyDescriptor,
)
from krrood.patterns.role import Role


@dataclass(kw_only=True, eq=False)
class PersonWithHobbies(Symbol):
    """
    A person, the role taker of its hobbies.
    """

    name: str
    """
    The name of the person.
    """


@dataclass(kw_only=True, eq=False)
class Interest(Symbol):
    """
    Something a person can like.
    """

    name: str
    """
    The name of the interest.
    """


@dataclass(kw_only=True, eq=False)
class Sports(Interest):
    """
    A sport.
    """


@dataclass(kw_only=True, eq=False)
class BasketBall(Sports):
    """
    Basketball.
    """


@dataclass(kw_only=True, eq=False)
class VideoGames(Interest):
    """
    Video games.
    """


@dataclass(kw_only=True, eq=False)
class PeopleWithHobby(Role[PersonWithHobbies], Symbol):
    """
    A person in the role of someone with a hobby.
    """

    likes: Set[Interest] = field(default_factory=set)
    """
    What the person likes.
    """


@dataclass(kw_only=True, eq=False)
class SportsLover(PeopleWithHobby):
    """
    A person in the role of a lover of sports.
    """

    loves: Set[Sports] = field(default_factory=set)
    """
    The sports the person loves.
    """


@dataclass(kw_only=True, eq=False)
class BasketBallLover(SportsLover):
    """
    A person in the role of a lover of basketball.
    """

    loves: Set[BasketBall] = field(default_factory=set)
    """
    The basketball the person loves.
    """


@dataclass(kw_only=True)
class Likes(PropertyDescriptor):
    """
    Relates a person to what it likes.
    """


@dataclass(kw_only=True)
class Loves(Likes):
    """
    Relates a person to what it loves.
    """


PeopleWithHobby.likes = Likes(domain=PeopleWithHobby, field_name="likes")
SportsLover.loves = Loves(domain=SportsLover, field_name="loves")
BasketBallLover.loves = Loves(domain=BasketBallLover, field_name="loves")


@dataclass(kw_only=True)
class PersonWithTwoRoles:
    """
    A person with a basketball lover role and a sports lover role, created in that order.
    """

    person: PersonWithHobbies
    """
    The root role taker.
    """
    basketball_lover: BasketBallLover
    """
    The role created first.
    """
    sports_lover: SportsLover
    """
    The role created second.
    """

    @classmethod
    def named(cls, *, name: str) -> PersonWithTwoRoles:
        """
        :param name: The name of the person.
        :return: A new person with both roles.
        """
        person = PersonWithHobbies(name=name)
        return cls(
            person=person,
            basketball_lover=BasketBallLover(role_taker=person),
            sports_lover=SportsLover(role_taker=person),
        )

    @property
    def objects(self) -> list[Symbol]:
        """
        The person followed by its roles, in the order they were created.
        """
        return [self.person, self.basketball_lover, self.sports_lover]
