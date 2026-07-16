"""
Recognition of *relational* attribute names — a past participle plus a preposition
(``assigned_to``, ``owned_by``, ``cross_referenced_with``) — shared by every surface
that renders a relation.

A relational hop reads as a relative clause (*"the Robot to which a Mission is
assigned"*) or, in absence, as a passive verb (*"has not been assigned to any Robot"*),
rather than the genitive *"the assigned_to of …"*. Keeping the recognizer here — above
the lexicon, below the grammar — lets both navigation (``chain``) and absence
(``conditions``) depend on it without depending on each other.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional

from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Prepositions,
)

#: The prepositions that, as the final token of a snake_case attribute name, mark a *relational*
#: field (*assigned_to*, *owned_by*, *located_in*): the English preposition inventory minus ``of`` —
#: ``<noun>_of`` is a genitive (``number_of``, ``type_of``), never a relation.
_RELATIONAL_PREPOSITIONS = {preposition.text for preposition in Prepositions} - {"of"}


@dataclass(frozen=True)
class RelationVerb:
    """
    A relational field's verb, split into its parts — so callers can either strand the
    preposition (*"assigned to"*, used by absence) or pied-pipe it (*"to which … is
    assigned"*, used by relational navigation).
    """

    participle: str
    """
    The past-participle verb word(s) (*"assigned"*, *"cross referenced"*).
    """

    preposition: str
    """
    The trailing preposition (*"to"*, *"by"*, *"with"*).
    """

    @property
    def phrase(self) -> str:
        """:return: The stranded phrase *"<participle> <preposition>"* (*"assigned to"*).

        >>> RelationVerb(participle="assigned", preposition="to").phrase
        'assigned to'
        """
        return f"{self.participle} {self.preposition}"

    @property
    def is_agentive(self) -> bool:
        """:return: ``True`` for an agentive relation — one whose preposition is *"by"*
        (*"owned by"*, *"written by"*), whose owner is the agent of the action. Such a relation reads
        naturally in the active voice (*"who owns"*) rather than the passive (*"by which … is
        owned"*).

        >>> RelationVerb(participle="owned", preposition="by").is_agentive
        True
        >>> RelationVerb(participle="assigned", preposition="to").is_agentive
        False
        """
        return self.preposition == Prepositions.BY.text

    @property
    def active_verb(self) -> str:
        """:return: The active present third-person-singular form of the relation's verb
        (*"owned"* → *"owns"*, *"written"* → *"writes"*), for the active-voice reading of an agentive
        relation (*"the Person who owns a Book"*).

        >>> RelationVerb(participle="owned", preposition="by").active_verb
        'owns'
        >>> RelationVerb(participle="written", preposition="by").active_verb
        'writes'
        """
        return morphology.third_person_singular(morphology.verb_lemma(self.participle))


def relational_verb(attribute_name: str) -> Optional[RelationVerb]:
    """
    :param attribute_name: An attribute identifier (snake_case).
    :return: The relation's verb, split into participle + preposition, when *attribute_name* names a
        relation as *past-participle + preposition* (*assigned_to* → *"assigned"* + *"to"*); else
        ``None`` for a plain noun attribute.

    The participle check (:func:`morphology.is_past_participle`) is what distinguishes a relation
    from a noun that merely ends in a preposition: *color_in* / *price_at* / *name_to* are rejected
    because *color* / *price* / *name* are not participles.

    >>> relational_verb("assigned_to").phrase
    'assigned to'
    >>> relational_verb("color_in") is None
    True
    """
    tokens = attribute_name.split("_")
    if len(tokens) < 2 or tokens[-1] not in _RELATIONAL_PREPOSITIONS:
        return None
    if not morphology.is_past_participle(tokens[-2]):
        return None
    return RelationVerb(participle=" ".join(tokens[:-1]), preposition=tokens[-1])


def relational_verb_phrase(attribute_name: str) -> Optional[str]:
    """
    :param attribute_name: An attribute identifier (snake_case).
    :return: The humanized stranded verb phrase (*"assigned to"*) for a relational field, else
        ``None`` — the form absence uses (*"has not been assigned to any Robot"*). Thin wrapper over
        :func:`relational_verb` so the recognition lives in one place.

    >>> relational_verb_phrase("written_by")
    'written by'
    >>> relational_verb_phrase("number_of") is None
    True
    """
    verb = relational_verb(attribute_name)
    return verb.phrase if verb is not None else None
