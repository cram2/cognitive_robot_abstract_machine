"""
The world reasoner's rules, each on the smallest world that shows what it decides.
"""

from dataclasses import dataclass, field

import pytest
from typing_extensions import Dict, List, Self, Set

from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.reasoning.world_rdr.rules import (
    ContainerKind,
    NamedKind,
    asserted_kind,
    cabinets,
    container_kind_of,
    dishwashers,
    doors_with_a_handle,
    doors_without_a_handle,
    drawers_with_a_handle,
    drawers_without_a_handle,
    handles,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    SideTable,
    Spoon,
)
from semantic_digital_twin.spatial_types import Vector3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import (
    ActiveConnection,
    FixedConnection,
    PrismaticConnection,
    RevoluteConnection,
)
from semantic_digital_twin.world_description.geometry import Box, Scale
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.world_entity import Body

# %% building small worlds


@dataclass
class FurnitureUnderTest:
    """
    A world holding just enough furniture to exercise one rule, and the bodies of it by
    the name each was built under.
    """

    world: World
    """
    The world the bodies live in.
    """

    bodies: Dict[str, Body] = field(default_factory=dict)
    """
    Each body of the furniture, by the name it was built under.
    """

    @classmethod
    def build(cls) -> Self:
        """
        Start a world holding nothing but a root body.
        """
        world = World()
        root = Body(name=PrefixedName("root", prefix="rules"))
        with world.modify_world():
            world.add_kinematic_structure_entity(root)
        return cls(world, {"root": root})

    def add(self, name: str, parent: str, connection_type, **connection_kwargs) -> Self:
        """
        Hang a new body carrying geometry off an existing one.

        :param connection_type: The type of connection to hang it by.
        :param connection_kwargs: Passed on to the connection's factory, so that a
            ``multiplier`` can make the new body only follow its parent.
        """
        return self._add(name, parent, connection_type, True, **connection_kwargs)

    def add_without_geometry(
        self, name: str, parent: str, connection_type, **connection_kwargs
    ) -> Self:
        """
        Hang a new shapeless body off an existing one, as world models do to build up a
        compound motion.
        """
        return self._add(name, parent, connection_type, False, **connection_kwargs)

    def _add(
        self,
        name: str,
        parent: str,
        connection_type,
        with_geometry: bool,
        **connection_kwargs,
    ) -> Self:
        body = Body(name=PrefixedName(name, prefix="rules"))
        if with_geometry:
            body.collision = ShapeCollection(
                [Box(scale=Scale(0.1, 0.1, 0.1))], reference_frame=body
            )
        parent_body = self.bodies[parent]
        if issubclass(connection_type, ActiveConnection):
            connection_kwargs.setdefault("axis", Vector3.X(reference_frame=parent_body))
        with self.world.modify_world():
            self.world.add_kinematic_structure_entity(body)
            self.world.add_connection(
                connection_type.create_with_dofs(
                    world=self.world,
                    parent=parent_body,
                    child=body,
                    parent_T_connection_expression=HomogeneousTransformationMatrix(
                        reference_frame=parent_body
                    ),
                    **connection_kwargs,
                )
            )
        self.bodies[name] = body
        return self

    def annotate(self, annotation) -> Self:
        """
        Give the world an annotation it already holds before any rule runs.
        """
        with self.world.modify_world():
            self.world.add_semantic_annotation_recursively(annotation)
        return self

    def apply(self, *rules) -> Self:
        """
        Run rules in order, adding what each concludes before the next one runs, the way
        the classifier does.
        """
        for rule in rules:
            for annotation in rule(self.world):
                with self.world.modify_world():
                    self.world.add_semantic_annotation_recursively(annotation)
        return self

    def root_names(self, *rules) -> Set[str]:
        """
        The names of the bodies the given rules conclude something about.
        """
        return {
            annotation.root.name.name
            for rule in rules
            for annotation in rule(self.world)
        }

    def name_of(self, key: str) -> str:
        """
        The full name of the body built under ``key``.
        """
        return self.bodies[key].name.name


@pytest.fixture
def drawer_with_a_grip() -> FurnitureUnderTest:
    """
    A case that slides out of a cabinet, with a grip fixed to its front.
    """
    return (
        FurnitureUnderTest.build()
        .add("cabinet", "root", FixedConnection)
        .add("case", "cabinet", PrismaticConnection)
        .add("grip", "case", FixedConnection)
    )


# %% what a body's name asserts


def test_a_compound_name_settles_on_the_kind_it_ends_on():
    """
    A name says where the body is before it says what it is, so the kind it is still
    talking about at the end is the one the body is.
    """
    assert (
        asserted_kind(["sink", "area", "left", "drawer"], NamedKind) is NamedKind.DRAWER
    )


def test_a_name_spelling_out_a_specific_kind_settles_on_it_over_its_base():
    """
    ``coffee_table`` names both a table and a coffee table, and means the latter.
    """
    assert asserted_kind(["coffee", "table"], NamedKind) is NamedKind.COFFEE_TABLE


def test_a_name_using_a_synonym_settles_on_the_kind_declaring_it():
    """
    A kind is recognised through its own ``_synonyms``, which is how a word the class
    name never spells still reaches it.
    """
    [synonym] = SideTable._synonyms

    assert asserted_kind([synonym, "table"], NamedKind) is NamedKind.SIDE_TABLE


def test_a_name_mentioning_no_kind_settles_on_nothing():
    """
    Most bodies are named after neither their kind nor anything else recognised.
    """
    assert asserted_kind(["cabinet"], NamedKind) is None


# %% recognising a handle


def test_a_grip_fixed_to_a_moving_part_is_a_handle(drawer_with_a_grip):
    """
    Nothing but the joints is needed: the grip cannot move by itself and travels with
    the case, while the case moves by its own joint and the cabinet does not move at
    all.
    """
    assert drawer_with_a_grip.root_names(handles) == {
        drawer_with_a_grip.name_of("grip")
    }


def test_a_body_fixed_to_something_that_cannot_move_is_not_a_handle():
    """
    A body fixed to a body that is itself fixed travels with nothing that opens.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("worktop", "root", FixedConnection)
        .add("appliance", "worktop", FixedConnection)
    )

    assert furniture.root_names(handles) == set()


def test_a_body_the_world_already_identifies_is_not_a_handle(drawer_with_a_grip):
    """
    An object put away in a drawer is jointed exactly as a handle is, and keeps the
    identity the world already holds for it.
    """
    drawer_with_a_grip.annotate(Spoon(root=drawer_with_a_grip.bodies["grip"]))

    assert drawer_with_a_grip.root_names(handles) == set()


def test_a_body_named_after_furniture_is_not_a_handle():
    """
    A shelf board mounted inside a drawer is jointed exactly as a handle is, and is kept
    apart by being named a board.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("cabinet", "root", FixedConnection)
        .add("case", "cabinet", PrismaticConnection)
        .add("board", "case", FixedConnection)
    )

    assert furniture.root_names(handles) == set()


# %% things that open


def test_a_case_a_slider_pulls_out_is_a_drawer_with_its_grip(drawer_with_a_grip):
    """
    The drawer is found by its joint, and given the handle mounted on it.
    """
    drawer_with_a_grip.apply(handles)

    [drawer] = drawers_with_a_handle(drawer_with_a_grip.world)
    assert drawer.root is drawer_with_a_grip.bodies["case"]
    assert drawer.handle.root is drawer_with_a_grip.bodies["grip"]


def test_a_case_with_nothing_to_pull_it_by_is_still_a_drawer():
    """
    A drawer is what a slider pulls out, whether or not the model gives it a grip.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("cabinet", "root", FixedConnection)
        .add("case", "cabinet", PrismaticConnection)
    )

    [drawer] = drawers_without_a_handle(furniture.world)
    assert drawer.root is furniture.bodies["case"]
    assert drawer.handle is None


def test_a_drawer_offering_a_handle_is_claimed_by_only_one_rule(drawer_with_a_grip):
    """
    The two drawer rules are complements, so a drawer is never inferred twice - which
    would leave the world holding it both with and without its handle.
    """
    drawer_with_a_grip.apply(handles)

    assert len(drawers_with_a_handle(drawer_with_a_grip.world)) == 1
    assert drawers_without_a_handle(drawer_with_a_grip.world) == []


def test_a_leaf_a_hinge_swings_is_a_door_with_its_grip():
    """
    A door is found by its hinge, and given the handle mounted on it.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("cabinet", "root", FixedConnection)
        .add("leaf", "cabinet", RevoluteConnection)
        .add("grip", "leaf", FixedConnection)
        .apply(handles)
    )

    [door] = doors_with_a_handle(furniture.world)
    assert door.root is furniture.bodies["leaf"]
    assert door.handle.root is furniture.bodies["grip"]


def test_a_leaf_another_follows_is_opened_by_that_leafs_handle():
    """
    A front of two leaves is jointed so the lower only repeats the upper's motion and
    carries the only grip, so the upper leaf is a door too.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("cabinet", "root", FixedConnection)
        .add("upper_leaf", "cabinet", RevoluteConnection)
        .add("lower_leaf", "upper_leaf", RevoluteConnection, multiplier=1.2)
        .add("grip", "lower_leaf", FixedConnection)
        .apply(handles)
    )

    assert furniture.root_names(doors_without_a_handle) == {
        furniture.name_of("upper_leaf")
    }


def test_a_container_is_not_opened_by_the_handle_on_its_own_door():
    """
    A door opens by its own joint rather than by following the appliance it is mounted
    on, so the appliance is not a door as well.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("cabinet", "root", FixedConnection)
        .add("appliance", "cabinet", RevoluteConnection)
        .add("leaf", "appliance", RevoluteConnection)
        .add("grip", "leaf", FixedConnection)
        .apply(handles)
    )

    assert furniture.root_names(doors_with_a_handle, doors_without_a_handle) == {
        furniture.name_of("leaf")
    }


def test_a_linkage_with_no_grip_anywhere_holds_no_door():
    """
    A tap is a chain of hinges in which nothing is a grip for opening anything, so none
    of its parts is a door.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("basin", "root", FixedConnection)
        .add("spout", "basin", RevoluteConnection)
        .add("lever", "spout", RevoluteConnection)
        .apply(handles)
    )

    assert furniture.root_names(doors_with_a_handle, doors_without_a_handle) == set()


# %% what the rules leave alone


def test_a_robots_own_parts_are_not_furniture(drawer_with_a_grip):
    """
    A robot is jointed exactly like a drawer with a handle on it, so only knowing that
    the branch is a robot keeps the rules off it.
    """
    MinimalRobot.from_branch_in_world(drawer_with_a_grip.bodies["cabinet"])

    assert drawer_with_a_grip.root_names(handles) == set()
    assert (
        drawer_with_a_grip.root_names(drawers_with_a_handle, drawers_without_a_handle)
        == set()
    )
    assert drawer_with_a_grip.root_names(cabinets) == set()


# %% containers


def test_a_body_things_open_out_of_is_a_container(drawer_with_a_grip):
    """
    A container is recognised by what opens out of it, and is given those parts.
    """
    drawer_with_a_grip.apply(handles, drawers_with_a_handle)

    [cabinet] = cabinets(drawer_with_a_grip.world)
    assert cabinet.root is drawer_with_a_grip.bodies["cabinet"]
    assert [drawer.root for drawer in cabinet.drawers] == [
        drawer_with_a_grip.bodies["case"]
    ]


def test_a_container_holds_a_part_reached_through_a_shapeless_helper():
    """
    A door that pops out before it swings hangs off a shapeless helper body, and belongs
    to the container beyond it rather than to the helper.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("cabinet", "root", FixedConnection)
        .add_without_geometry("popout", "cabinet", PrismaticConnection)
        .add("leaf", "popout", RevoluteConnection)
        .add("grip", "leaf", FixedConnection)
        .apply(handles, doors_with_a_handle)
    )

    [cabinet] = cabinets(furniture.world)
    assert cabinet.root is furniture.bodies["cabinet"]
    assert [door.root for door in cabinet.doors] == [furniture.bodies["leaf"]]


def test_a_container_takes_its_kind_from_the_part_that_names_it():
    """
    Only the front of an appliance is named after it, so the kind is read from the whole
    branch rather than from the container's own name.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("unit", "root", FixedConnection)
        .add("dishwasher_front", "unit", PrismaticConnection)
        .apply(handles, drawers_without_a_handle)
    )

    assert container_kind_of(furniture.bodies["unit"]) is ContainerKind.DISHWASHER
    assert furniture.root_names(dishwashers) == {furniture.name_of("unit")}
    assert furniture.root_names(cabinets) == set()


def test_a_container_naming_no_kind_is_a_plain_cabinet():
    """
    A container whose parts name no kind is left to the plain cabinet rule.
    """
    furniture = (
        FurnitureUnderTest.build()
        .add("unit", "root", FixedConnection)
        .add("case", "unit", PrismaticConnection)
        .apply(handles, drawers_without_a_handle)
    )

    assert container_kind_of(furniture.bodies["unit"]) is None
    assert furniture.root_names(cabinets) == {furniture.name_of("unit")}
