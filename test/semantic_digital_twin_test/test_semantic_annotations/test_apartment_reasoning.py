"""
What the world reasoner must infer from the ``iai_apartment`` world model.

The expected bodies are named here rather than counted, so that a rule finding the right
number of the wrong things fails. Every name is checked against the world itself by
:func:`test_every_expected_body_exists`, so a typo here fails as a typo rather than as a
missing annotation.
"""

from enum import StrEnum

import pytest
from typing_extensions import List, Set, Type

from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.semantic_annotations.mixins import HasRootBody
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Cabinet,
    CoffeeMachine,
    CoffeeTable,
    Cooktop,
    CounterTop,
    Dishwasher,
    Door,
    Drawer,
    Fridge,
    Handle,
    Hinge,
    Oven,
    ShelfLayer,
    SideTable,
    Sink,
    Slider,
    Sofa,
    Table,
    Wall,
    Wardrobe,
)
from semantic_digital_twin.world import World

# %% the bodies each annotation is expected on


class ApartmentCabinet(StrEnum):
    """
    The bodies of the apartment that are a Cabinet.
    """

    CABINET1 = "cabinet1"
    CABINET10 = "cabinet10"
    CABINET11 = "cabinet11"
    CABINET2 = "cabinet2"
    CABINET3 = "cabinet3"
    CABINET4 = "cabinet4"
    CABINET5 = "cabinet5"
    CABINET6 = "cabinet6"
    CABINET8 = "cabinet8"
    CABINET9 = "cabinet9"


class ApartmentCoffeeMachine(StrEnum):
    """
    The bodies of the apartment that are a CoffeeMachine.
    """

    COFFE_MACHINE = "coffe_machine"


class ApartmentCoffeeTable(StrEnum):
    """
    The bodies of the apartment that are a CoffeeTable.
    """

    COFFEE_TABLE = "coffee_table"


class ApartmentCooktop(StrEnum):
    """
    The bodies of the apartment that are a Cooktop.
    """

    COOKTOP = "cooktop"


class ApartmentCounterTop(StrEnum):
    """
    The bodies of the apartment that are a CounterTop.
    """

    COUNTERTOP = "countertop"
    ISLAND_COUNTERTOP = "island_countertop"


class ApartmentDishwasher(StrEnum):
    """
    The bodies of the apartment that are a Dishwasher.
    """

    CABINET7 = "cabinet7"


class ApartmentDoor(StrEnum):
    """
    The bodies of the apartment that are a Door.
    """

    CABINET1_DOOR_TOP_LEFT = "cabinet1_door_top_left"
    CABINET2_DOOR_LEFT = "cabinet2_door_left"
    CABINET3_DOOR_BOTTOM_LEFT = "cabinet3_door_bottom_left"
    CABINET3_DOOR_TOP_LEFT = "cabinet3_door_top_left"
    CABINET4_DOOR_BOTTOM = "cabinet4_door_bottom"
    CABINET4_DOOR_TOP = "cabinet4_door_top"
    CABINET7_DOOR_BOTTOM_LEFT = "cabinet7_door_bottom_left"
    WARDROBE_DOOR_LEFT = "wardrobe_door_left"
    WARDROBE_DOOR_RIGHT = "wardrobe_door_right"


class ApartmentDrawer(StrEnum):
    """
    The bodies of the apartment that are a Drawer.
    """

    CABINET10_DRAWER_BOTTOM = "cabinet10_drawer_bottom"
    CABINET10_DRAWER_MIDDLE = "cabinet10_drawer_middle"
    CABINET10_DRAWER_TOP = "cabinet10_drawer_top"
    CABINET11_DRAWER_BOTTOM = "cabinet11_drawer_bottom"
    CABINET11_DRAWER_MIDDLE = "cabinet11_drawer_middle"
    CABINET11_DRAWER_TOP = "cabinet11_drawer_top"
    CABINET1_DRAWER_BOTTOM = "cabinet1_drawer_bottom"
    CABINET1_DRAWER_MIDDLE = "cabinet1_drawer_middle"
    CABINET2_DRAWER_BIG = "cabinet2_drawer_big"
    CABINET2_DRAWER_SMALL = "cabinet2_drawer_small"
    CABINET5_DRAWER_BOTTOM = "cabinet5_drawer_bottom"
    CABINET5_DRAWER_MIDDLE = "cabinet5_drawer_middle"
    CABINET5_DRAWER_TOP = "cabinet5_drawer_top"
    CABINET6_DRAWER_BOTTOM = "cabinet6_drawer_bottom"
    CABINET6_DRAWER_MIDDLE = "cabinet6_drawer_middle"
    CABINET6_DRAWER_TOP = "cabinet6_drawer_top"
    CABINET8_DRAWER_BOTTOM = "cabinet8_drawer_bottom"
    CABINET8_DRAWER_MIDDLE = "cabinet8_drawer_middle"
    CABINET9_DRAWER_BOTTOM = "cabinet9_drawer_bottom"
    CABINET9_DRAWER_MIDDLE = "cabinet9_drawer_middle"
    CABINET9_DRAWER_TOP = "cabinet9_drawer_top"
    COFFEE_TABLE_DRAWER = "coffee_table_drawer"
    DISHWASHER_DRAWER_MIDDLE = "dishwasher_drawer_middle"


class ApartmentHandle(StrEnum):
    """
    The bodies of the apartment that are a Handle.
    """

    HANDLE_CAB10_B = "handle_cab10_b"
    HANDLE_CAB10_M = "handle_cab10_m"
    HANDLE_CAB10_T = "handle_cab10_t"
    HANDLE_CAB11_B = "handle_cab11_b"
    HANDLE_CAB11_M = "handle_cab11_m"
    HANDLE_CAB11_T = "handle_cab11_t"
    HANDLE_CAB1_DRAWER_BOTTOM = "handle_cab1_drawer_bottom"
    HANDLE_CAB1_DRAWER_MID = "handle_cab1_drawer_mid"
    HANDLE_CAB1_TOP_DOOR = "handle_cab1_top_door"
    HANDLE_CAB2_DOOR = "handle_cab2_door"
    HANDLE_CAB3_DOOR_BOTTOM = "handle_cab3_door_bottom"
    HANDLE_CAB3_DOOR_TOP = "handle_cab3_door_top"
    HANDLE_CAB4_DOOR_BOTTOM = "handle_cab4_door_bottom"
    HANDLE_CAB5_B = "handle_cab5_b"
    HANDLE_CAB5_M = "handle_cab5_m"
    HANDLE_CAB5_T = "handle_cab5_t"
    HANDLE_CAB6_B = "handle_cab6_b"
    HANDLE_CAB6_M = "handle_cab6_m"
    HANDLE_CAB6_T = "handle_cab6_t"
    HANDLE_CAB7 = "handle_cab7"
    HANDLE_CAB7_MIDDLE = "handle_cab7_middle"
    HANDLE_CAB8 = "handle_cab8"
    HANDLE_CAB9_B = "handle_cab9_b"
    HANDLE_CAB9_M = "handle_cab9_m"
    HANDLE_CAB9_T = "handle_cab9_t"
    WARDROBE_DOOR_LEFT_HANDLE = "wardrobe_door_left_handle"
    WARDROBE_DOOR_RIGHT_HANDLE = "wardrobe_door_right_handle"


class ApartmentOven(StrEnum):
    """
    The bodies of the apartment that are a Oven.
    """

    OVEN = "oven"


class ApartmentShelfLayer(StrEnum):
    """
    The bodies of the apartment that are a ShelfLayer.
    """

    CABINET1_COLOKSU_LEVEL4 = "cabinet1_coloksu_level4"
    CABINET1_COLOKSU_LEVEL5 = "cabinet1_coloksu_level5"


class ApartmentSideTable(StrEnum):
    """
    The bodies of the apartment that are a SideTable.
    """

    BEDSIDE_TABLE = "bedside_table"


class ApartmentSink(StrEnum):
    """
    The bodies of the apartment that are a Sink.
    """

    SINK = "sink"


class ApartmentSofa(StrEnum):
    """
    The bodies of the apartment that are a Sofa.
    """

    SOFA = "sofa"


class ApartmentTable(StrEnum):
    """
    The bodies of the apartment that are a Table.
    """

    TABLE_AREA_MAIN = "table_area_main"


class ApartmentWall(StrEnum):
    """
    The bodies of the apartment that are a Wall.
    """

    WALL_COLOKSU_WALL1 = "wall_coloksu_wall1"
    WALL_COLOKSU_WALL2 = "wall_coloksu_wall2"
    WALL_COLOKSU_WALL3 = "wall_coloksu_wall3"
    WALL_COLOKSU_WALL4 = "wall_coloksu_wall4"
    WALLS = "walls"


class ApartmentWardrobe(StrEnum):
    """
    The bodies of the apartment that are a Wardrobe.
    """

    WARDROBE = "wardrobe"


# %% pairing each annotation type with what it is expected on

EXPECTED_BODIES: List[tuple] = [
    (Handle, ApartmentHandle),
    (Drawer, ApartmentDrawer),
    (Door, ApartmentDoor),
    (Cabinet, ApartmentCabinet),
    (Wardrobe, ApartmentWardrobe),
    (Dishwasher, ApartmentDishwasher),
    (Oven, ApartmentOven),
    (Sink, ApartmentSink),
    (Cooktop, ApartmentCooktop),
    (CounterTop, ApartmentCounterTop),
    (Sofa, ApartmentSofa),
    (Wall, ApartmentWall),
    (ShelfLayer, ApartmentShelfLayer),
    (CoffeeMachine, ApartmentCoffeeMachine),
    (Table, ApartmentTable),
    (CoffeeTable, ApartmentCoffeeTable),
    (SideTable, ApartmentSideTable),
]
"""
Each annotation type together with the bodies the reasoner must infer it on.
"""


def expected_body_names(expected: Type[StrEnum]) -> Set[str]:
    """
    The body names one of the expectations above holds.
    """
    return {member.value for member in expected}


def inferred_body_names(world: World, annotation_type: Type[HasRootBody]) -> Set[str]:
    """
    The names of the bodies the world's annotations of ``annotation_type`` sit on.

    Subclasses are excluded, so that each expectation is checked against the type the
    reasoner actually chose rather than against a broader family.
    """
    return {
        annotation.root.name.name
        for annotation in world.semantic_annotations
        if type(annotation) is annotation_type
    }


# %% the expectations themselves


ROBOT_JOINT_NAME_ONLY_A_ROBOT_HAS = "l_upper_arm_roll_joint"
"""
A joint of the robot, which reasoning over a world it stands in must leave in place.
"""

PREANNOTATED_DRAWER_BODY_NAME = "cabinet10_drawer_top"
"""
The drawer the apartment fixture annotates by hand before any reasoning runs.
"""


@pytest.fixture
def reasoned_apartment(apartment_world_copy) -> World:
    """
    The apartment with everything the reasoner can infer about it already inferred.
    """
    WorldReasoner(apartment_world_copy).infer_semantic_annotations()
    return apartment_world_copy


@pytest.mark.parametrize(
    "annotation_type, expected",
    EXPECTED_BODIES,
    ids=[annotation_type.__name__ for annotation_type, _ in EXPECTED_BODIES],
)
def test_every_expected_body_exists(
    annotation_type: Type[HasRootBody],
    expected: Type[StrEnum],
    apartment_world_copy: World,
):
    """
    Every body this module expects an annotation on is really in the apartment.
    """
    for name in expected_body_names(expected):
        assert apartment_world_copy.get_body_by_name(name).name.name == name


@pytest.mark.parametrize(
    "annotation_type, expected",
    EXPECTED_BODIES,
    ids=[annotation_type.__name__ for annotation_type, _ in EXPECTED_BODIES],
)
def test_annotations_sit_on_the_expected_bodies(
    annotation_type: Type[HasRootBody],
    expected: Type[StrEnum],
    reasoned_apartment: World,
):
    """
    The reasoner infers each annotation on exactly the bodies that are it, so that both
    a missed body and a wrongly claimed one fail.
    """
    assert inferred_body_names(reasoned_apartment, annotation_type) == (
        expected_body_names(expected)
    )


def test_the_apartment_holds_no_fridge(reasoned_apartment: World):
    """
    The apartment has no fridge, so the rule that recognises one must stay quiet rather
    than mistake a cabinet for it.
    """
    assert inferred_body_names(reasoned_apartment, Fridge) == set()


# %% the parts each annotation is wired to


def test_a_drawer_is_wired_to_the_handle_mounted_on_it(
    reasoned_apartment: World,
):
    """
    A drawer carrying a handle is given that handle, and one without stays without.
    """
    drawers_by_body = {
        drawer.root.name.name: drawer
        for drawer in reasoned_apartment.get_semantic_annotations_by_type(Drawer)
    }

    handled = drawers_by_body[ApartmentDrawer.CABINET5_DRAWER_TOP]
    assert handled.handle.root.name.name == ApartmentHandle.HANDLE_CAB5_T

    assert drawers_by_body[ApartmentDrawer.COFFEE_TABLE_DRAWER].handle is None


def test_a_door_is_wired_to_the_handle_mounted_on_it(
    reasoned_apartment: World,
):
    """
    A door carrying a handle is given that handle, and one without stays without.
    """
    doors_by_body = {
        door.root.name.name: door
        for door in reasoned_apartment.get_semantic_annotations_by_type(Door)
    }

    handled = doors_by_body[ApartmentDoor.WARDROBE_DOOR_LEFT]
    assert handled.handle.root.name.name == ApartmentHandle.WARDROBE_DOOR_LEFT_HANDLE

    assert doors_by_body[ApartmentDoor.CABINET4_DOOR_TOP].handle is None


def test_a_container_holds_the_parts_that_open_out_of_it(
    reasoned_apartment: World,
):
    """
    A cabinet is given every drawer and door that opens out of it, including one reached
    through the shapeless helper body of a pop-out mechanism.
    """
    cabinets_by_body = {
        cabinet.root.name.name: cabinet
        for cabinet in reasoned_apartment.get_semantic_annotations_by_type(Cabinet)
    }

    both = cabinets_by_body[ApartmentCabinet.CABINET1]
    assert {drawer.root.name.name for drawer in both.drawers} == {
        ApartmentDrawer.CABINET1_DRAWER_MIDDLE.value,
        ApartmentDrawer.CABINET1_DRAWER_BOTTOM.value,
    }
    assert {door.root.name.name for door in both.doors} == {
        ApartmentDoor.CABINET1_DOOR_TOP_LEFT.value
    }

    behind_a_helper_body = cabinets_by_body[ApartmentCabinet.CABINET2]
    assert {door.root.name.name for door in behind_a_helper_body.doors} == {
        ApartmentDoor.CABINET2_DOOR_LEFT.value
    }


def test_the_dishwasher_is_named_by_the_part_that_opens_out_of_it(
    reasoned_apartment: World,
):
    """
    A container takes its kind from its parts, so the cabinet holding the dishwasher
    front is a dishwasher rather than a plain cabinet.
    """
    [dishwasher] = reasoned_apartment.get_semantic_annotations_by_type(Dishwasher)

    assert {drawer.root.name.name for drawer in dishwasher.drawers} == {
        ApartmentDrawer.DISHWASHER_DRAWER_MIDDLE.value
    }


# %% the mechanical joints the reasoner inserts


def test_no_body_carries_the_same_kind_of_annotation_twice(
    reasoned_apartment: World,
):
    """
    Reasoning over a world that already holds some of the annotations must recognise
    them rather than store a second copy of each.
    """
    for annotation_type, _ in EXPECTED_BODIES:
        inferred = [
            annotation.root.name.name
            for annotation in reasoned_apartment.semantic_annotations
            if type(annotation) is annotation_type
        ]
        assert len(inferred) == len(set(inferred)), annotation_type.__name__


def test_an_annotation_the_world_already_held_is_wired_to_its_mechanical_joint(
    reasoned_apartment: World,
):
    """
    The world is given one drawer by hand before reasoning runs.

    Reasoning must wire the joint onto that very drawer, since that is the one
    everything else refers to.
    """
    drawers = [
        drawer
        for drawer in reasoned_apartment.get_semantic_annotations_by_type(Drawer)
        if drawer.root.name.name == PREANNOTATED_DRAWER_BODY_NAME
    ]

    [drawer] = drawers
    assert isinstance(drawer.mechanical_joint, Slider)


def test_every_openable_part_is_carried_by_a_mechanical_joint(
    reasoned_apartment: World,
):
    """
    Every drawer and door the world ends up holding is given the joint that already
    moves it, so the world stays valid after the reasoner rewires it.
    """
    drawers = reasoned_apartment.get_semantic_annotations_by_type(Drawer)
    doors = reasoned_apartment.get_semantic_annotations_by_type(Door)
    assert drawers and doors

    for drawer in drawers:
        assert isinstance(drawer.mechanical_joint, Slider)
    for door in doors:
        assert isinstance(door.mechanical_joint, Hinge)

    assert reasoned_apartment.validate()


# %% what the reasoner leaves alone


def test_a_robot_in_the_apartment_is_left_alone(pr2_apartment_world: World):
    """
    A robot standing in the apartment keeps every one of its joints.

    Its links are jointed like furniture - the torso slides as a drawer does and an arm
    link swings as a door does - so annotating them would insert mechanical joints into
    the robot and rewire the chain its motions are addressed by.
    """
    robot_body_names = {
        body.name.name for body in pr2_apartment_world.robot_body_to_robot_mapping
    }
    assert ROBOT_JOINT_NAME_ONLY_A_ROBOT_HAS in {
        connection.name.name for connection in pr2_apartment_world.connections
    }

    inferred = WorldReasoner(pr2_apartment_world).infer_semantic_annotations()

    claimed = {
        annotation.root.name.name
        for annotation in inferred
        if isinstance(annotation, HasRootBody)
    } & robot_body_names
    assert claimed == set()
    assert ROBOT_JOINT_NAME_ONLY_A_ROBOT_HAS in {
        connection.name.name for connection in pr2_apartment_world.connections
    }
