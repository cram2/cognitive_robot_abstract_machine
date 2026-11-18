"""
Python names for the IRIs and anonymous class expressions of an OWL ontology, and the
RDF graph that describes them.
"""

from __future__ import annotations

import re
from abc import ABC, abstractmethod
from dataclasses import dataclass
from functools import lru_cache

import rdflib
from rdflib import OWL, RDFS
from rdflib.collection import Collection
from typing_extensions import Any, Optional, Union

from krrood.ontomatic.exceptions import UnsupportedClassExpression
from krrood.patterns.classproperty import classproperty
from krrood.utils import recursive_subclasses

# %% what an RDF graph says about its nodes


@dataclass(kw_only=True)
class HasRDFGraph:
    """
    Reads what an RDF graph states about its nodes: their names, labels and comments,
    and the members of its lists.
    """

    graph: rdflib.Graph
    """
    The graph that describes the nodes.
    """

    @staticmethod
    def uri_to_python_name(*, uri: Any) -> str:
        """
        :param uri: A named class, individual or property.
        :return: A valid Python identifier for it, or the text of anything else, such as
            a literal or a blank node.
        """
        if isinstance(uri, rdflib.URIRef):
            return re.sub(r"[^a-zA-Z0-9_]", "_", HasRDFGraph.local_name(uri=uri))
        return str(uri)

    @staticmethod
    @lru_cache(maxsize=None)
    def local_name(*, uri: Union[str, rdflib.URIRef]) -> str:
        """
        :param uri: An IRI.
        :return: The part of the IRI after the last ``#``, or else after the last ``/``.
        """
        text = str(uri)
        if "#" in text:
            return text.rsplit("#", 1)[1]
        return text.rstrip("/").rsplit("/", 1)[-1]

    def python_name_of_node(self, *, node: rdflib.term.Node) -> str:
        """
        :param node: A class, individual, literal or anonymous class expression.
        :return: A valid Python identifier for a named node, the name built from its kind
            and what it refers to for an anonymous class expression, and the text of a
            literal.
        :raises UnsupportedClassExpression: When a blank node is no known kind of
            anonymous class expression.
        """
        if isinstance(node, rdflib.BNode):
            return ClassExpression.from_node(graph=self.graph, node=node).python_name
        return self.uri_to_python_name(uri=node)

    def list_members(self, *, head: rdflib.term.Node) -> list[rdflib.term.Node]:
        """
        :param head: The first cell of an RDF list.
        :return: The members of the list, in order.
        """
        return list(Collection(self.graph, head))

    def label(self, *, uri: rdflib.term.Node) -> Optional[str]:
        """
        :param uri: A class, property or ontology.
        :return: Its first ``rdfs:label``, if any.
        """
        for label in self.graph.objects(uri, RDFS.label):
            return str(label)
        return None

    def comment(self, *, uri: rdflib.term.Node) -> Optional[str]:
        """
        :param uri: A class, property or ontology.
        :return: Its first ``rdfs:comment``, if any.
        """
        for comment in self.graph.objects(uri, RDFS.comment):
            return str(comment)
        return None


# %% anonymous class expressions


@dataclass(kw_only=True)
class ClassExpression(HasRDFGraph, ABC):
    """
    An anonymous OWL class expression, which names itself for what it describes.

    Each kind of expression is a subclass without subclasses of its own, which
    recognises its own blank nodes.
    """

    node: rdflib.BNode
    """
    The blank node of the expression.
    """

    @classmethod
    def from_node(cls, *, graph: rdflib.Graph, node: rdflib.BNode) -> ClassExpression:
        """
        :param graph: The graph that describes the expression.
        :param node: The blank node of the expression.
        :return: The expression, of the kind the node is.
        :raises UnsupportedClassExpression: When the node is no known kind.
        """
        for kind in recursive_subclasses(cls):
            if not kind.__subclasses__() and kind.is_kind_of(graph=graph, node=node):
                return kind(graph=graph, node=node)
        raise UnsupportedClassExpression(
            node=node, predicates=sorted(set(graph.predicates(node)))
        )

    @classproperty
    @abstractmethod
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        The predicate that makes a blank node an expression of this kind.
        """

    @classmethod
    def is_kind_of(cls, *, graph: rdflib.Graph, node: rdflib.BNode) -> bool:
        """
        :param graph: The graph that describes the node.
        :param node: A blank node.
        :return: Whether the node is an expression of this kind.
        """
        return (node, cls.defining_predicate, None) in graph

    @property
    @abstractmethod
    def python_name(self) -> str:
        """
        The name of the expression, built from its kind and what it refers to.
        """


# %% expressions over a list of members


@dataclass(kw_only=True)
class ListOfMembers(ClassExpression, ABC):
    """
    A class expression over an RDF list of members, named by joining their names.
    """

    @classproperty
    @abstractmethod
    def name_prefix(cls) -> str:
        """
        The text the name starts with.
        """

    @classproperty
    @abstractmethod
    def member_separator(cls) -> str:
        """
        The text between the names of the members.
        """

    @classmethod
    def join_member_names(cls, *, member_names: list[str]) -> str:
        """
        :param member_names: The names of the members.
        :return: The prefix and the member names, joined by the separator.
        """
        return cls.name_prefix + cls.member_separator.join(member_names)

    @property
    def python_name(self) -> str:
        """
        The prefix and the names of the members, joined by the separator.
        """
        members = self.list_members(
            head=self.graph.value(self.node, self.defining_predicate)
        )
        return self.join_member_names(
            member_names=[self.python_name_of_node(node=member) for member in members]
        )


@dataclass(kw_only=True)
class IntersectionOf(ListOfMembers):
    """
    The individuals of every one of the member classes (``owl:intersectionOf``).
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:intersectionOf``, which states the list of member classes.
        """
        return OWL.intersectionOf

    @classproperty
    def name_prefix(cls) -> str:
        """
        Nothing: the name is the members' names alone.
        """
        return ""

    @classproperty
    def member_separator(cls) -> str:
        """
        ``AND``, as an individual is of every member.
        """
        return "AND"


@dataclass(kw_only=True)
class UnionOf(ListOfMembers):
    """
    The individuals of any of the member classes (``owl:unionOf``).
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:unionOf``, which states the list of member classes.
        """
        return OWL.unionOf

    @classproperty
    def name_prefix(cls) -> str:
        """
        Nothing: the name is the members' names alone.
        """
        return ""

    @classproperty
    def member_separator(cls) -> str:
        """
        ``OR``, as an individual is of any member.
        """
        return "OR"


@dataclass(kw_only=True)
class OneOf(ListOfMembers):
    """
    Exactly the member individuals (``owl:oneOf``).
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:oneOf``, which states the list of member individuals.
        """
        return OWL.oneOf

    @classproperty
    def name_prefix(cls) -> str:
        """
        ``OneOf``, which marks an enumeration of individuals.
        """
        return "OneOf"

    @classproperty
    def member_separator(cls) -> str:
        """
        Nothing: the individuals' names follow each other.
        """
        return ""


# %% complement


@dataclass(kw_only=True)
class ComplementOf(ClassExpression):
    """
    The individuals not of a class (``owl:complementOf``), named for the classes that
    class is declared disjoint with, or else as its negation.
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:complementOf``, which states the complemented class.
        """
        return OWL.complementOf

    @property
    def python_name(self) -> str:
        """
        The names of the classes the complemented class is disjoint with, joined as
        their :class:`UnionOf`, or else ``Not`` and its name.
        """
        complemented = self.graph.value(self.node, OWL.complementOf)
        disjoint_names = [
            self.uri_to_python_name(uri=disjoint)
            for disjoint in self.graph.objects(complemented, OWL.disjointWith)
        ]
        return UnionOf.join_member_names(
            member_names=disjoint_names
        ) or "Not" + self.python_name_of_node(node=complemented)


# %% restrictions on a property


@dataclass(kw_only=True)
class Restriction(ClassExpression, ABC):
    """
    The individuals whose values of a property (``owl:onProperty``) satisfy a condition.
    """

    @property
    def property_name(self) -> str:
        """
        The name of the restricted property.
        """
        return self.uri_to_python_name(uri=self.graph.value(self.node, OWL.onProperty))

    @classproperty
    @abstractmethod
    def condition_word(cls) -> str:
        """
        The word between the property and the condition in the name.
        """


@dataclass(kw_only=True)
class FillerRestriction(Restriction, ABC):
    """
    A restriction of the values of a property to a class, an individual or a literal,
    the filler stated under the defining predicate.
    """

    @property
    def python_name(self) -> str:
        """
        The property's name, the condition word and the filler's name.
        """
        filler = self.graph.value(self.node, self.defining_predicate)
        return f"{self.property_name}{self.condition_word}{self.python_name_of_node(node=filler)}"


@dataclass(kw_only=True)
class SomeValuesFrom(FillerRestriction):
    """
    The individuals with at least one value of the property in the filler class.
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:someValuesFrom``, which states the filler class.
        """
        return OWL.someValuesFrom

    @classproperty
    def condition_word(cls) -> str:
        """
        ``SOME``, as some value is in the filler class.
        """
        return "SOME"


@dataclass(kw_only=True)
class AllValuesFrom(FillerRestriction):
    """
    The individuals whose values of the property are all in the filler class.
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:allValuesFrom``, which states the filler class.
        """
        return OWL.allValuesFrom

    @classproperty
    def condition_word(cls) -> str:
        """
        ``ALL``, as all values are in the filler class.
        """
        return "ALL"


@dataclass(kw_only=True)
class HasValue(FillerRestriction):
    """
    The individuals with the filler individual or literal as a value of the property.
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:hasValue``, which states the filler individual or literal.
        """
        return OWL.hasValue

    @classproperty
    def condition_word(cls) -> str:
        """
        ``HAS``, as the filler is one of the values.
        """
        return "HAS"


@dataclass(kw_only=True)
class HasSelf(Restriction):
    """
    The individuals that are a value of the property for themselves.
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:hasSelf``, which states that the individual relates to itself.
        """
        return OWL.hasSelf

    @classproperty
    def condition_word(cls) -> str:
        """
        ``SELF``, as the individual is its own value.
        """
        return "SELF"

    @property
    def python_name(self) -> str:
        """
        The property's name and the condition word.
        """
        return f"{self.property_name}{self.condition_word}"


# %% cardinality restrictions


@dataclass(kw_only=True)
class CardinalityRestriction(Restriction, ABC):
    """
    A bound on the number of values of a property, of any class or, when qualified, of
    one class (``owl:onClass``) or data range (``owl:onDataRange``).
    """

    @classproperty
    @abstractmethod
    def qualified_predicate(cls) -> rdflib.URIRef:
        """
        The predicate that states the bound on the values of one class or data range.
        """

    @classmethod
    def is_kind_of(cls, *, graph: rdflib.Graph, node: rdflib.BNode) -> bool:
        """
        Whether the node states the bound qualified or not, extending
        :meth:`ClassExpression.is_kind_of`.
        """
        return super().is_kind_of(graph=graph, node=node) or (
            (node, cls.qualified_predicate, None) in graph
        )

    @property
    def python_name(self) -> str:
        """
        The property's name, the condition word, the bound and the name of the
        qualifying class or data range, if any.
        """
        bound = self.graph.value(self.node, self.defining_predicate)
        if bound is None:
            bound = self.graph.value(self.node, self.qualified_predicate)
        qualifier = self.graph.value(self.node, OWL.onClass) or self.graph.value(
            self.node, OWL.onDataRange
        )
        qualifier_name = (
            "" if qualifier is None else self.python_name_of_node(node=qualifier)
        )
        return f"{self.property_name}{self.condition_word}{bound}{qualifier_name}"


@dataclass(kw_only=True)
class MinCardinality(CardinalityRestriction):
    """
    The individuals with at least the bound number of values of the property.
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:minCardinality``, which states the lower bound.
        """
        return OWL.minCardinality

    @classproperty
    def qualified_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:minQualifiedCardinality``, which states the lower bound on the values of
        one class.
        """
        return OWL.minQualifiedCardinality

    @classproperty
    def condition_word(cls) -> str:
        """
        ``MIN``, as the bound is the least number of values.
        """
        return "MIN"


@dataclass(kw_only=True)
class MaxCardinality(CardinalityRestriction):
    """
    The individuals with at most the bound number of values of the property.
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:maxCardinality``, which states the upper bound.
        """
        return OWL.maxCardinality

    @classproperty
    def qualified_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:maxQualifiedCardinality``, which states the upper bound on the values of
        one class.
        """
        return OWL.maxQualifiedCardinality

    @classproperty
    def condition_word(cls) -> str:
        """
        ``MAX``, as the bound is the greatest number of values.
        """
        return "MAX"


@dataclass(kw_only=True)
class ExactCardinality(CardinalityRestriction):
    """
    The individuals with exactly the bound number of values of the property.
    """

    @classproperty
    def defining_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:cardinality``, which states the exact number.
        """
        return OWL.cardinality

    @classproperty
    def qualified_predicate(cls) -> rdflib.URIRef:
        """
        ``owl:qualifiedCardinality``, which states the exact number of values of one
        class.
        """
        return OWL.qualifiedCardinality

    @classproperty
    def condition_word(cls) -> str:
        """
        ``EXACTLY``, as the bound is the exact number of values.
        """
        return "EXACTLY"
