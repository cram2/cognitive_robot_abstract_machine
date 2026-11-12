"""
Python names for the IRIs and anonymous class expressions of an OWL ontology.
"""

from __future__ import annotations

import re
from functools import lru_cache

import rdflib
from rdflib import OWL
from typing_extensions import Any, Optional, Union


class NamingRegistry:
    """
    Registry for converting OWL URIs and names to Python-compatible identifiers.
    """

    @staticmethod
    def uri_to_python_name(*, uri: Any, graph: Optional[rdflib.Graph] = None) -> str:
        """
        :param uri: A named or anonymous class expression, or a property.
        :param graph: The graph that describes an anonymous class expression.
        :return: A valid Python identifier for it.
        """
        if isinstance(uri, rdflib.URIRef):
            return re.sub(r"[^a-zA-Z0-9_]", "_", NamingRegistry.local_name(uri=uri))
        elif isinstance(uri, rdflib.BNode) and graph is not None:
            return NamingRegistry.bnode_to_name(graph=graph, bnode=uri)
        return str(uri)

    @staticmethod
    def bnode_to_name(*, graph: rdflib.Graph, bnode: rdflib.BNode) -> Optional[str]:
        """
        Generate a meaningful name for a BNode based on its description.
        """
        # Intersection
        intersections = list(graph.objects(bnode, OWL.intersectionOf))
        if intersections:
            items = NamingRegistry._get_rdf_list(graph=graph, head=intersections[0])
            return "AND".join(
                [NamingRegistry._get_node_name(graph=graph, node=i) for i in items]
            )

        # Union
        unions = list(graph.objects(bnode, OWL.unionOf))
        if unions:
            items = NamingRegistry._get_rdf_list(graph=graph, head=unions[0])
            return "OR".join(
                [NamingRegistry._get_node_name(graph=graph, node=i) for i in items]
            )

        # Enumeration of individuals
        enumerations = list(graph.objects(bnode, OWL.oneOf))
        if enumerations:
            items = NamingRegistry._get_rdf_list(graph=graph, head=enumerations[0])
            return "OneOf" + "".join(
                NamingRegistry._get_node_name(graph=graph, node=i) for i in items
            )

        # Restrictions
        on_prop = graph.value(bnode, OWL.onProperty)
        if on_prop:
            prop_name = NamingRegistry.uri_to_python_name(uri=on_prop)
            if graph.value(bnode, OWL.someValuesFrom):
                target = graph.value(bnode, OWL.someValuesFrom)
                return f"{prop_name}SOME{NamingRegistry._get_node_name(graph=graph, node=target)}"
            if graph.value(bnode, OWL.allValuesFrom):
                target = graph.value(bnode, OWL.allValuesFrom)
                target_name = NamingRegistry._get_node_name(graph=graph, node=target)
                return f"{prop_name}ALL{target_name}"
            if graph.value(bnode, OWL.hasValue):
                target = graph.value(bnode, OWL.hasValue)
                return f"{prop_name}HAS{NamingRegistry._get_node_name(graph=graph, node=target)}"
            if graph.value(bnode, OWL.maxQualifiedCardinality):
                target = NamingRegistry.uri_to_python_name(
                    uri=graph.value(bnode, OWL.maxQualifiedCardinality)
                )
                on_class = NamingRegistry.uri_to_python_name(
                    uri=graph.value(bnode, OWL.onClass)
                )
                return f"{prop_name}MAX{target}{on_class}"
            for cardinality, word in (
                (OWL.minCardinality, "MIN"),
                (OWL.maxCardinality, "MAX"),
                (OWL.cardinality, "EXACTLY"),
            ):
                if graph.value(bnode, cardinality) is not None:
                    return f"{prop_name}{word}{graph.value(bnode, cardinality)}"
            if graph.value(bnode, OWL.complementOf):
                target = NamingRegistry.uri_to_python_name(
                    uri=graph.value(bnode, OWL.complementOf)
                )
                return f"Not{target}"
        elif graph.value(bnode, OWL.complementOf):
            complement_of = graph.value(bnode, OWL.complementOf)
            possible_complements = map(
                NamingRegistry.uri_to_python_name,
                list(graph.objects(complement_of, OWL.disjointWith)),
            )
            return "OR".join(possible_complements) or (
                "Not" + NamingRegistry._get_node_name(graph=graph, node=complement_of)
            )
        return None

    @staticmethod
    def _get_node_name(*, graph, node):
        """
        :return: The Python name of a named class, of an anonymous class expression, or of a literal.
        """
        if isinstance(node, rdflib.URIRef):
            return NamingRegistry.uri_to_python_name(uri=node)
        if isinstance(node, rdflib.BNode):
            return NamingRegistry.bnode_to_name(graph=graph, bnode=node)
        return str(node)

    @staticmethod
    def _get_rdf_list(*, graph, head):
        """
        :return: The members of the RDF list starting at the head.
        """
        items = []
        while head != rdflib.RDF.nil:
            items.append(graph.value(head, rdflib.RDF.first))
            head = graph.value(head, rdflib.RDF.rest)
        return [i for i in items if i is not None]

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

    @staticmethod
    def to_snake_case(*, name: str) -> str:
        """
        Convert a name like 'worksFor' or 'WorksFor' to 'works_for'.
        """
        s1 = re.sub(r"(.)([A-Z][a-z]+)", r"\1_\2", name)
        s2 = re.sub(r"([a-z0-9])([A-Z])", r"\1_\2", s1)
        return s2.lower()

    @staticmethod
    def to_pascal_case(*, name: str) -> str:
        """
        Convert a name like 'worksFor' or 'works_for' to 'WorksFor'.
        """
        # If it contains underscores or hyphens, split and capitalize parts
        parts = re.split(r"[_\-\s]+", name)
        if len(parts) > 1:
            return "".join(p.capitalize() for p in parts if p)
        # Otherwise just capitalize first char
        return name[:1].upper() + name[1:]
