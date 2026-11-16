"""
The information about an OWL ontology the converter works on: its classes, its properties and the ontology as a whole.
"""

from __future__ import annotations

import re
from copy import deepcopy
from dataclasses import dataclass, field
from functools import cached_property

import rdflib
from typing_extensions import Any, Dict, List, Optional, Set

from krrood.ontomatic.naming import NamingRegistry
from krrood.ontomatic.ontology_to_python.property_type import PropertyType
from krrood.patterns.dataclass_deep_copy import DataclassDeepcopyMixin, DeepCopyOfField
from krrood.patterns.hashed_by_identity import HashedByIdentity
from krrood.patterns.role import Role


@dataclass(kw_only=True)
class RoleTakerInfo:
    """
    Information about a class that acts as a role taker.
    Used when a class is determined to be a 'role' of another class.
    """

    class_name: str
    """
    The name of the role-taker class.
    """
    field_name: str
    """
    The name of the field holding the role taker.
    """


@dataclass(kw_only=True, eq=False)
class ClassInfo(DataclassDeepcopyMixin, HashedByIdentity):
    """
    Maintains all metadata, inheritance, and property associations for an OWL class.
    Used during the code generation process to represent a Python class.
    """

    name: str
    """
    The name of the generated class.
    """
    uri: str
    """
    The IRI of the OWL class.
    """
    superclasses: List[str] = field(default_factory=list)
    """
    The names of the classes the class is declared a subclass of.
    """
    base_classes: List[str] = field(default_factory=list)
    """
    The names of the classes the generated class inherits from.
    """
    all_base_classes: List[str] = field(default_factory=list)
    """
    The names of all classes the class inherits from, directly or not.
    """
    all_base_classes_including_role_takers: List[str] = field(default_factory=list)
    """
    The names of all classes the class inherits from or is a role of, directly or not.
    """
    base_classes_for_topological_sort: List[str] = field(default_factory=list)
    """
    The names of the classes that must be generated before the class.
    """
    disjoint_with: List[str] = field(default_factory=list)
    """
    The names of the classes declared disjoint with the class.
    """
    equivalent_classes: List[str] = field(default_factory=list)
    """
    The names of the named classes declared equivalent to the class.
    """
    is_description_for: Optional[str] = None
    """
    The name of the named class this anonymous class expression describes, if it describes one.
    """
    has_descriptions: List[str] = field(default_factory=list)
    """
    The names of the anonymous class expressions describing the class.
    """
    label: Optional[str] = None
    """
    The label of the class, rendered into its docstring.
    """
    comment: Optional[str] = None
    """
    The comment of the class, rendered into its docstring.
    """
    add_role_taker: bool = True
    """
    Whether the generated class is a role and gets its role-taker type as a generic parameter.
    """
    role_taker: Optional[RoleTakerInfo] = None
    """
    The class the class is a role of, if it is a role.
    """
    declared_properties: List[str] = field(default_factory=list)
    """
    The names of the properties whose fields the generated class declares.
    """
    axioms: List[str] = field(default_factory=list)
    """
    EQL conditions of the sufficient conditions (owl:equivalentClass definitions and general class axioms with the
    class as superclass). Only these are used to classify individuals.
    """
    axioms_python: List[str] = field(default_factory=list)
    """
    Python conditions of the sufficient conditions, see :attr:`axioms`.
    """
    axioms_setup: List[str] = field(default_factory=list)
    """
    The statements declaring the variables the entity-query-language conditions of :attr:`axioms` quantify over.
    """
    necessary_axioms_python: List[str] = field(default_factory=list)
    """
    Python conditions of the necessary conditions (restrictions in superclass position, i.e. rdfs:subClassOf of the
    class). They hold for every member of the class but do not make an individual a member, so they are not used for
    classification.
    """
    ontology: OntologyInfo | None = field(
        default=None, repr=False, metadata={DeepCopyOfField: DeepCopyOfField.SHARED}
    )
    """
    The ontology the class belongs to, which a copy of the class keeps.
    """


@dataclass(kw_only=True, eq=False)
class PropertyInfo(DataclassDeepcopyMixin, HashedByIdentity):
    """
    Maintains metadata, domains, ranges, and inheritance for an OWL property.
    Used during the code generation process to represent a Python property or descriptor.
    """

    name: str
    """
    The name of the property in the ontology.
    """
    uri: str
    """
    The IRI of the property.
    """
    type: PropertyType
    """
    Whether the property relates individuals to individuals or to literals.
    """
    domains: List[str] = field(default_factory=list)
    """
    The names of the classes whose individuals the property relates.
    """
    ranges: List[str] = field(default_factory=list)
    """
    The names of the classes, or of the Python types of the literals, the property relates individuals to.
    """
    range_uris: List[Any] = field(default_factory=list)
    """
    The IRIs of the ranges.
    """
    label: Optional[str] = None
    """
    The label of the property, rendered as a comment of its fields.
    """
    comment: Optional[str] = None
    """
    The comment of the property, rendered as a comment of its fields.
    """
    field_name: str = ""
    """
    The name of the field holding the values of the property.
    """
    descriptor_name: str = ""
    """
    The name of the descriptor class of the property.
    """
    equivalent_properties: List[str] = field(default_factory=list)
    """
    The names of the properties declared equivalent to the property.
    """
    disjoint_properties: List[str] = field(default_factory=list)
    """
    The names of the properties declared disjoint with the property.
    """
    is_symmetric: bool = False
    """
    Whether the property is symmetric.
    """
    is_reflexive: bool = False
    """
    Whether the property is reflexive.
    """
    is_asymmetric: bool = False
    """
    Whether the property is asymmetric.
    """
    is_irreflexive: bool = False
    """
    Whether the property is irreflexive.
    """
    superproperties: List[str] = field(default_factory=list)
    """
    The names of the properties the property is declared a subproperty of.
    """
    all_superproperties: List[str] = field(default_factory=list)
    """
    The names of all superproperties of the property, directly or not.
    """
    inverses: List[str] = field(default_factory=list)
    """
    The names of the properties declared inverse to the property.
    """
    inverse_of: Optional[str] = None
    """
    The name of the property the property is the inverse of, if any.
    """
    inverse_target_is_prior: bool = False
    """
    Whether the descriptor class of the inverse is rendered before the descriptor class of the property.
    """
    is_transitive: bool = False
    """
    Whether the property is transitive.
    """
    is_functional: bool = False
    """
    Whether the property is functional, so that its field holds one value.
    """
    is_specialized: bool = False
    """
    Whether the property is narrowed for one class, with a range of its own.
    """
    declared_domains: List[str] = field(default_factory=list)
    """
    The names of the classes that declare a field for the property.
    """
    rdfs_domains: List[str] = field(default_factory=list)
    """
    Names of the named classes declared with rdfs:domain for this property in the ontology.
    """
    rdfs_ranges: List[str] = field(default_factory=list)
    """
    Names of the named classes declared with rdfs:range for this property in the ontology.
    """
    _overrides_for: List[str] = field(default_factory=list)
    """
    The names of the classes whose field type of the property is predefined.
    """
    _predefined_data_type: bool = False
    """
    Whether the type of the property's values is predefined instead of inferred.
    """
    data_type_hint_inner: Optional[str] = None
    """
    The Python type of a value of a data property.
    """
    object_range_hint: Optional[str] = None
    """
    The Python type annotation of a value of an object property.
    """
    base_descriptors: List[str] = field(default_factory=list)
    """
    The names of the classes the descriptor class of the property inherits from.
    """
    equivalent_properties_descriptor_names: List[str] = field(default_factory=list)
    """
    The names of the descriptor classes of the equivalent properties.
    """
    disjoint_properties_descriptor_names: List[str] = field(default_factory=list)
    """
    The names of the descriptor classes of the disjoint properties.
    """
    chain_axioms: List[List[str]] = field(default_factory=list)
    """
    The names of the descriptor classes of every property chain implying the property.
    """
    ontology: OntologyInfo | None = field(
        default=None, repr=False, metadata={DeepCopyOfField: DeepCopyOfField.SHARED}
    )
    """
    The ontology the property belongs to, which a copy of the property keeps.
    """


@dataclass(kw_only=True, eq=False)
class OntologyInfo(HashedByIdentity):
    """Information about the ontology."""

    graph: rdflib.Graph
    """
    The RDF graph of the ontology.
    """
    classes: Dict[str, ClassInfo] = field(default_factory=dict)
    """
    The classes to generate, by name.
    """
    class_descriptions: Dict[str, ClassInfo] = field(default_factory=dict)
    """
    The anonymous class expressions describing a named class, by the name of that class.
    """
    original_properties: Dict[str, PropertyInfo] = field(default_factory=dict)
    """
    The properties as the ontology declares them, by name.
    """
    predefined_data_types: Optional[Dict[str, Dict[str, str]]] = None
    """
    The predefined Python types of the fields of classes, by class name and field name.
    """
    ontology_label: str = "Thing"
    """
    The label of the ontology, naming its base class.
    """
    role_class_name: str = Role.__name__
    """
    The name of the role class generated classes take roles from.
    """
    _properties: Optional[Dict[str, PropertyInfo]] = None
    """
    The properties as inferred, computed once from :attr:`original_properties`.
    """
    property_restrictions: Dict[str, Dict[str, set]] = field(default_factory=dict)
    """
    The classes each property is restricted to, by the name of the restricted class and of the property.
    """

    def __post_init__(self):
        """
        Link the classes and properties to the ontology.
        """
        for property_information in self.original_properties.values():
            property_information.ontology = self
        for class_information in self.classes.values():
            class_information.ontology = self

    def description_for(self, *, description_name: str) -> Optional[str]:
        """
        :return: The name of the named class the anonymous class expression of this name describes, if any.
        """
        for class_name, class_information in self.class_descriptions.items():
            if class_information.name == description_name:
                return class_name
        return None

    @property
    def properties(self):
        """
        The properties as inferred, copies of :attr:`original_properties` the inference completes.
        """
        if not self._properties:
            self._properties = {
                name: deepcopy(property_information)
                for name, property_information in self.original_properties.items()
            }
        return self._properties

    @cached_property
    def base_class_name(self):
        """
        The name of the base class every generated class inherits from.
        """
        base_class_name = NamingRegistry.to_pascal_case(
            name=re.sub(r"\W+", " ", self.ontology_label).strip()
        )
        if not base_class_name.endswith("Thing"):
            base_class_name += "Thing"
        return base_class_name

    @cached_property
    def class_ancestors(self) -> Dict[str, Set[str]]:
        """
        The class ancestors map as a dictionary mapping class names to their ancestors.
        """
        return {
            name: set(class_information.all_base_classes)
            for name, class_information in self.classes.items()
        }
