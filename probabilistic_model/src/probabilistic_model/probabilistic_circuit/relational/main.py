from dataclasses import dataclass, field
from typing import List

from krrood.class_diagrams.class_diagram import WrappedClass
from krrood.entity_query_language.factories import variable, underspecified, variable_from
from krrood.entity_query_language.query.match import Match
from krrood.parametrization.model_registries import DictRegistry
from krrood.parametrization.parameterizer import UnderspecifiedParameters
from krrood.symbol_graph.symbol_graph import Symbol
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.relational.learn_rspn import (
    LearnRSPN,
)

from matplotlib import pyplot as plt

from probabilistic_model.probabilistic_circuit.relational.rspns import (
    RSPNTemplate,
    RSPNSpecification, aggregation_statistic, RSPNPredicate,
)
from probabilistic_model.probabilistic_circuit.rx.helper import fully_factorized
from pycram.datastructures.grasp import GraspDescription
from pycram_test.conftest import mutable_model_world
from random_events.variable import Continuous
from semantic_digital_twin.robots.abstract_robot import Manipulator


@dataclass
class Government(Symbol):
    funny: bool


@dataclass
class Person(Symbol):
    name: str
    age: float = field(default=0)


@dataclass
class Supports(RSPNPredicate):
    supporting_person: Person
    government: Government


@dataclass
class Nation(Symbol):
    government: Government = None
    persons: List[Person] = field(default_factory=list)
    supporters: List[Supports] = field(default_factory=list)
    gdp: float = 500

    @aggregation_statistic("supporters")
    def mean_age_of_supporters(self):
        return sum(s.supporting_person.age for s in self.supporters) / len(
            self.supporters
        )

    @aggregation_statistic("persons")
    def mean_age_of_persons(self):
        return sum(p.age for p in self.persons) / len(self.persons)

    @aggregation_statistic("government")
    def government_type(self):
        return float(self.government.funny)


@dataclass
class Adjacent(RSPNPredicate):
    nation1: Nation
    nation2: Nation


@dataclass
class Conflict(RSPNPredicate):
    nation1: Nation
    nation2: Nation


@dataclass
class Region(Symbol):
    nations: List[Nation]
    adjacency: List[Adjacent]
    conflicts: List[Conflict]

    @aggregation_statistic("adjacency")
    def adjacency_density(self):
        return len(self.adjacency) / len(self.nations)

    @aggregation_statistic("conflicts")
    def conflict_density(self):
        return len(self.conflicts) / len(self.nations)

    @aggregation_statistic("nations")
    def average_nation_age(self):
        total_age = 0
        total_persons = 0
        for nation in self.nations:
            for person in nation.persons:
                total_age += person.age
                total_persons += 1
        if total_persons == 0:
            return 0
        return total_age / total_persons


univariate_attribute_distributions = {
    "age": GaussianDistribution(variable=Continuous("age"), location=0, scale=1),
    "funny": GaussianDistribution(variable=Continuous("funny"), location=0, scale=1),
    "gdp": GaussianDistribution(variable=Continuous("gdp"), location=0, scale=1),
    # remove name later
    "name": GaussianDistribution(variable=Continuous("name"), location=0, scale=1),
}


def example():
    david = Person("David", 25)
    tom = Person("Tom", 27)
    checker_chan = Person("Simon Wallukat", 28)

    cdu = Government(funny=True)
    persons = [david, tom, checker_chan]
    supporters = [Supports(checker_chan, cdu)]
    n1 = Nation(government=cdu, supporters=supporters, persons=persons)

    daniel = Person("Daniel", 50)
    tede = Person("Tede", 25)
    daniel_union = Government(funny=False)

    knowrob_supporters = [daniel, tede]

    knowrob_nation = Nation(
        government=daniel_union,
        persons=[daniel, tede],
        supporters=[Supports(daniel, daniel_union), Supports(tede, daniel_union)],
        gdp=-10,
    )

    region = Region(
        nations=[n1, knowrob_nation],
        adjacency=[Adjacent(n1, knowrob_nation)],
        conflicts=[Conflict(n1, knowrob_nation)],
    )

    # n2 = let(Nation, [])
    # p3 = let(Person, [])
    # q = an(
    #     entity(
    #         n2,
    #         n2.government.funny == True,
    #         n2.persons == [david, tom, p3],
    #     )
    # )

    spec = RSPNSpecification()

    class_spec_nation = {
        "exchangeable_parts": ["persons"],
        "unique_parts": ["government"],
        "attributes": ["gdp"],
        "relations": ["supporters"],
    }
    class_spec_gov = {
        "attributes": ["funny"],
        "relations": [],
        "exchangeable_parts": [],
        "unique_parts": [],
    }
    class_spec_region = {
        "exchangeable_parts": ["nations"],
        "unique_parts": [],
        "relations": ["adjacency", "conflicts"],
        "attributes": [],
    }
    class_spec_person = {
        "attributes": ["age", "name"],
        "relations": [],
        "exchangeable_parts": [],
        "unique_parts": [],
    }

    classes = {
        "nations": class_spec_nation,
        "government": class_spec_gov,
        "region": class_spec_region,
        "persons": class_spec_person,
    }

    relation_mapping = {
        "supporters": Supports,
        "adjacency": Adjacent,
        "conflicts": Conflict,
    }

    region_template = RSPNTemplate(class_spec=class_spec_region)
    region_template.probabilistic_circuit.plot_structure()
    plt.show()

    nation_template = RSPNTemplate(class_spec=class_spec_nation)
    nation_template.probabilistic_circuit.plot_structure()
    plt.show()

    grounded = region_template.ground(region)
    grounded.probabilistic_circuit.plot_structure()
    plt.show()

    grounded_nation = nation_template.ground(n1)
    grounded_nation.probabilistic_circuit.plot_structure()
    plt.show()

    # learned_nation = LearnRSPN(Region, region, class_spec_region)
    # learned_nation.probabilistic_circuit.plot_structure()
    # plt.show()

    learned_nation = LearnRSPN(Nation, [n1, knowrob_nation], class_spec_nation)
    learned_nation.probabilistic_circuit.plot_structure()
    plt.show()

    grounded_learned_nation = learned_nation.ground(n1)
    grounded_learned_nation.probabilistic_circuit.plot_structure()
    plt.show()


# def real_example():
#     world, robot_view, context = mutable_model_world
#
#     milk = world.get_body_by_name("milk.stl")
#
#     milk_variable = variable_from([milk])
#
#     move_and_pick_up_description = underspecified(MoveAndPickUpAction)(
#         standing_position=underspecified(PoseStamped)(
#             pose=underspecified(PyCramPose)(
#                 position=underspecified(Position)(x=..., y=..., z=...),
#                 orientation=underspecified(Orientation)(x=..., y=..., z=..., w=...),
#             ),
#             header=underspecified(Header)(frame_id=variable_from([robot_view.root])),
#         ),
#         object_designator=milk_variable,
#         arm=...,
#         grasp_description=underspecified(GraspDescription)(
#             approach_direction=...,
#             vertical_alignment=...,
#             rotate_gripper=...,
#             manipulation_offset=0.05,
#             manipulator=variable(Manipulator, world.semantic_annotations),
#         ),
#     )
#
#     move_and_pick_up_description: Match = move_and_pick_up_description
#
#     parameters = UnderspecifiedParameters(move_and_pick_up_description)
#
#
#
#     move_and_pick_up_distribution = fully_factorized(parameters.variables.values())
#
#     probabilistic_registry = DictRegistry({MoveAndPickUpAction: move_and_pick_up_distribution})
#
#     sample = move_and_pick_up_distribution.sample(3)
#
#
#     #----------------- database stuff
#
#     wrapped_class = WrappedClass(MoveAndPickUpAction)
#
#     rspn_spec = RSPNSpecification(spec=wrapped_class)
#     template = RSPNTemplate(class_spec=rspn_spec)
#     template.probabilistic_circuit.plot_structure()
#     plt.show()
#
# if __name__ == "__main__":
#     real_example()
