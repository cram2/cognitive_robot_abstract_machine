"""
Builds a verbalization clause through every vocabulary path that decides an inflected
word, without realizing it, so a fresh interpreter can report what building loads.
"""

from __future__ import annotations

from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.factories import variable
from krrood.entity_query_language.predicate import HasType
from krrood.entity_query_language.verbalization._example_domain import Robot, Worker
from krrood.entity_query_language.verbalization.fragments.base import WordFragment
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    possessive_path,
)
from krrood.entity_query_language.verbalization.navigation_path import (
    build_path_parts,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    SingularExistential,
)
from krrood.entity_query_language.verbalization.vocabulary.parts_of_speech import (
    Noun,
    OneOf,
    predicate_clause,
)


# %% predicates read off a class name


class ConnectsTo:
    """
    A predicate whose name reads as a lexical verb phrase.
    """


class IsReachable:
    """
    A predicate whose name reads as a copular phrase.
    """


predicate_clause(ConnectsTo, Noun(content="body"), Noun(content="gripper"))
predicate_clause(IsReachable, Noun(content="Robot"))
HasType._verbalization_fragment_(
    fields={
        HasType.subject_field_name(): WordFragment(text="body"),
        HasType.object_field_name(): WordFragment(text="Robot"),
    }
)

# %% a set summarised by its count

OneOf(members=tuple(range(100))).as_fragment()

# %% an existential with its indefinite article

SingularExistential(text="there's").build_phrase(type_name="Apple")

# %% an indexed collection hop

possessive_path(
    build_path_parts(walk_chain(variable(Worker, []).tasks[-2])[0]),
    WordFragment(text="Worker"),
)
possessive_path(
    build_path_parts(walk_chain(variable(Robot, []).battery)[0]),
    WordFragment(text="Robot"),
)
