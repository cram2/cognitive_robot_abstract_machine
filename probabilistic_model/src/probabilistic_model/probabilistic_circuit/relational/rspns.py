from __future__ import annotations

import enum
from typing import Dict, Optional, Type, Iterable, Union
from typing import List
import copy
from dataclasses import dataclass, field

import sqlalchemy
from sqlalchemy.orm import ONETOMANY, MANYTOMANY, MANYTOONE

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.class_diagrams.wrapped_field import WrappedField
from krrood.ormatic.data_access_objects.dao import DataAccessObject
from krrood.ormatic.utils import is_data_column
from krrood.symbol_graph.symbol_graph import SymbolGraph
from probabilistic_model.distributions.distributions import UnivariateDistribution
from random_events.variable import Continuous, Integer, Symbolic, compatible_types
from typing_extensions import Any

from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.distributions.uniform import UniformDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import (
    ProbabilisticCircuit,
    leaf,
    ProductUnit,
    Unit,
    SumUnit,
)


def aggregation_statistic(name: str):
    """
    Decorator to mark a method as an aggregate statistic.
    :param name: The name of the relation/part this statistic belongs to.
    """

    def decorator(func):
        func._statistic_name = name
        return func

    return decorator


@dataclass
class RSPNPredicate:
    """
    Abstract class to declare a predicate that can be used in an RSPN.
    This is just for type clarity and does not have any functionality on its own.
    """

    pass


@dataclass
class ExchangeableDistributionTemplate:
    """
    EDT is a function that returns a joint probability distribution over a set of variables.
    """

    variables: List[Continuous] = field(default_factory=list)
    """
    Random Variables that the EDT is defined over.
    """

    base_distribution: Type[UnivariateDistribution] = field(
        default_factory=GaussianDistribution
    )
    """
    The base distribution for the EDT
    """

    def __init__(
        self,
        variables: List[Continuous],
        base_distribution: Type[UnivariateDistribution],
        **kwargs,
    ):
        self.variables = variables
        self.base_distribution = base_distribution
        self.kwargs = kwargs

    def __call__(self, *args, **kwargs):
        if len(self.variables) == 0:
            raise ValueError(
                "ExchangeableDistributionTemplate requires at least one variable."
            )

        circuit = ProbabilisticCircuit()
        product = ProductUnit(probabilistic_circuit=circuit)

        for v in self.variables:
            dist = self.base_distribution(variable=v, **kwargs)
            unit = leaf(dist, probabilistic_circuit=circuit)
            product.add_subcircuit(unit)

        return product


@dataclass
class RSPNSpecification:
    """
    Specification for an RSPN class template.
    """

    spec: Type[DataAccessObject] = field(init=True)
    """
    The wrapped class that is supposed to be an RSPN.
    """

    def __post_init__(self):
        self.attributes = []
        self.unique_parts = []
        self.exchangeable_parts = []
        self.relations = []

        mapper: sqlalchemy.orm.Mapper = sqlalchemy.inspection.inspect(self.spec)

        for relationship in mapper.relationships:
            if relationship.direction == MANYTOONE:
                self.unique_parts.append(relationship.key)
            # not many to many since we have the association table
            elif relationship.direction == ONETOMANY:
                self.exchangeable_parts.append(relationship.key)
        for column in mapper.columns:
            if is_data_column(column) and column not in mapper.relationships:
                self.attributes.append(column)


@dataclass
class RSPNTemplate:
    """
    Class "C" in an RSPN consists of
    1. A set "A" of unary predicates applicable to individuals of C
    2. A vector "U_C"= (P_1,...,P_n) of unique parts.
    3. A vector "E_C"= (P_1,...,P_n) of exchangeable parts.
    4. A set of predicates (relations) of form R_1(P_1, P_2).
    5. A tractable univariate class distribution where leaves fall into one of three categories:
        5.1 L^C_A: univariate distribution over A of C
        5.2 L^C_R: EDT over predicate R involving C, E_C or U_C,
        5.3 L^C_P: Sub-SPN for part class P
    """

    class_spec: RSPNSpecification = field(init=True)

    probabilistic_circuit: Optional[ProbabilisticCircuit] = field(default=None)
    """
    The circuit this component is part of. 
    """

    edt_over_relations: Optional[List[ExchangeableDistributionTemplate]] = field(
        default_factory=list, init=False
    )
    """
    L^C_R: List of EDTs over predicates R involving C, E_C or U_C
    """

    def __post_init__(self):
        self.probabilistic_circuit = (
            self.probabilistic_circuit or ProbabilisticCircuit()
        )
        if len(self.probabilistic_circuit) == 0:
            self.edt_over_relations = []
            self.unique_parts = [
                RSPNTemplate(RSPNSpecification(WrappedClass(part.resolved_type)))
                for part in self.class_spec.unique_parts
            ]
            self.exchangeable_parts = [
                RSPNTemplate(RSPNSpecification(WrappedClass(part.contained_type)))
                for part in self.class_spec.exchangeable_parts
            ]

            self._prepare_structure()

    def _prepare_structure(self):
        root = ProductUnit(probabilistic_circuit=self.probabilistic_circuit)
        self.add_attributes(root)
        # self.fix_edt_over_relations(root)
        self.add_parts(root)

    def add_attributes(self, product):
        for attribute in self.class_spec.attributes:
            # TODO Flexible Distributions
            product.add_subcircuit(
                leaf(
                    GaussianDistribution(
                        variable=Continuous(attribute.name), location=0, scale=1
                    ),
                    self.probabilistic_circuit,
                )
            )

    # def fix_edt_over_relations(self, product):
    #     for relation in self.relations:
    #         fields = relation_mapping[relation].__dataclass_fields__
    #         if len(fields) != 2:
    #             raise ValueError(
    #                 f"Relation {relation} must be of the form R(P1, P2) or R(C, P1) where P1, P2 are part classes of class C."
    #             )
    #         first = Continuous(list(fields.keys())[0])
    #         second = Continuous(list(fields.keys())[1])
    #
    #         self.edt_over_relations.append(
    #             ExchangeableDistributionTemplate([first, second], GaussianDistribution)
    #         )
    #
    #     for relation_template in self.edt_over_relations:
    #         # TODO dont assume gaussian
    #         edt_product = relation_template(location=0, scale=1)
    #         self.edt_products.append(edt_product)
    #         index_remap = product.probabilistic_circuit.mount(edt_product)
    #         root_index = edt_product.index
    #         product.add_subcircuit(index_remap[root_index])

    def add_parts(self, product):
        sub_rspns = self.unique_parts + self.exchangeable_parts

        for part_class in sub_rspns:
            index_remap = product.probabilistic_circuit.mount(
                part_class.probabilistic_circuit.root
            )
            root_index = part_class.probabilistic_circuit.root.index
            product.add_subcircuit(index_remap[root_index])

    def ground_attributes(self, grounded_root: Unit, instance_repr: str):
        # Use distributions from the template's circuit if it was learned
        learned_distributions = {}
        if (
            len(self.probabilistic_circuit) > 1
        ):  # if it has more than just the root product
            for leaf_node in self.probabilistic_circuit.leaves:
                dist = leaf_node.distribution
                if dist.variable.name in [
                    attribute.name for attribute in self.class_spec.attributes
                ]:
                    learned_distributions[dist.variable.name] = dist

        for attribute in self.class_spec.attributes:
            # If we have a learned distribution for this attribute, use it
            if attribute in learned_distributions:
                distribution = learned_distributions[attribute]
            else:
                distribution = GaussianDistribution(Continuous(attribute.name), 0, 1)

            # Create a unique grounded variable name
            grounded_var_name = f"{attribute.name}({instance_repr})"

            original_var = distribution.variable
            grounded_var = original_var.__class__(grounded_var_name)

            # Create a grounded distribution
            grounded_dist = copy.deepcopy(distribution)
            grounded_dist.variable = grounded_var

            grounded_root.add_subcircuit(
                leaf(grounded_dist, grounded_root.probabilistic_circuit)
            )
        return grounded_root

    # def ground_relations(self, grounded_root: Unit, instance_repr: str):
    #     # Find learned EDTs if any
    #     learned_relation_distributions = {}
    #     if len(self.probabilistic_circuit) > 1:
    #         for leaf_node in self.probabilistic_circuit.leaves:
    #             dist = leaf_node.distribution
    #             if dist.variable.name in self.relations:
    #                 learned_relation_distributions[dist.variable.name] = dist
    #
    #     for relation, relation_template in zip(self.relations, self.edt_over_relations):
    #         # Instantiate the EDT for this instance.
    #         if relation in learned_relation_distributions:
    #             dist = learned_relation_distributions[relation]
    #             if hasattr(dist, "probabilities"):
    #                 grounded_edt = relation_template(probabilities=dist.probabilities)
    #             elif hasattr(dist, "location") and hasattr(dist, "scale"):
    #                 grounded_edt = relation_template(
    #                     location=dist.location, scale=dist.scale
    #                 )
    #             else:
    #                 grounded_edt = relation_template()
    #         else:
    #             grounded_edt = relation_template(location=0, scale=1)
    #
    #         # Ground the variables in the EDT product
    #         for l in grounded_edt.leaves:
    #             orig_var = l.distribution.variable
    #             new_var_name = f"{orig_var.name}({instance_repr})"
    #             l.distribution.variable = orig_var.__class__(new_var_name)
    #
    #         # Mount the grounded_edt into our current circuit
    #         # We use the fact that grounded_edt already has a circuit from relation_template call.
    #         index_remap = grounded_root.probabilistic_circuit.mount(grounded_edt)
    #         grounded_root.add_subcircuit(index_remap[grounded_edt.index])
    #
    #     return grounded_root

    def ground_unique_parts(self, grounded_root: Unit, instance):
        for idx, part_template in enumerate(self.unique_parts):
            x = self.class_spec.unique_parts[idx]
            part_instance = getattr(
                instance, self.class_spec.unique_parts[idx].name, None
            )
            if part_instance is not None:
                grounded_part = part_template.ground(part_instance)
                # Mount the grounded part into our current circuit
                index_remap = grounded_root.probabilistic_circuit.mount(grounded_part)
                grounded_root.add_subcircuit(index_remap[grounded_part.index])

        return grounded_root

    # def ground_exchangeable_parts(self, grounded_root: Unit, instance):
    #     for idx, part_template in enumerate(self.exchangeable_parts):
    #         part_instances = getattr(
    #             instance, self.class_spec["exchangeable_parts"][idx], []
    #         )
    #         for p_inst in part_instances:
    #             grounded_part = part_template.ground(p_inst)
    #             # Mount the grounded part into our current circuit
    #             index_remap = grounded_root.probabilistic_circuit.mount(grounded_part)
    #             grounded_root.add_subcircuit(index_remap[grounded_part.index])
    #
    #     return grounded_root

    def ground(self, instance: Any) -> Unit:
        """
        Ground the RSPN template for a specific instance.
        :param instance: The object to ground the template for.
        :return: A grounded SPN as a Unit.
        """
        # Create a product unit for this grounding
        grounded_root = type(self.probabilistic_circuit.root)(
            probabilistic_circuit=ProbabilisticCircuit()
        )

        if hasattr(instance, "name"):
            instance_repr = instance.name
        else:
            wrapped = SymbolGraph().get_wrapped_instance(instance)
            if wrapped is not None and wrapped.index is not None:
                instance_repr = f"{type(instance).__name__}{wrapped.index}"
            else:
                instance_repr = str(id(instance))

        grounded_root = self.ground_attributes(grounded_root, instance_repr)
        # grounded_root = self.ground_relations(grounded_root, instance_repr)
        grounded_root = self.ground_unique_parts(grounded_root, instance)
        # grounded_root = self.ground_exchangeable_parts(grounded_root, instance)

        return grounded_root
