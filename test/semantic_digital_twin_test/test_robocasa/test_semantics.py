from pathlib import Path
from xml.etree import ElementTree as ET

from semantic_digital_twin.adapters.mjcf import MJCFParser
from semantic_digital_twin.adapters.robocasa_dataset.loader import (
    RoboCasaDatasetLoader,
    _category_from_class_name,
    _mjcf_document_from_element_copy,
)
from semantic_digital_twin.adapters.robocasa_dataset.semantics import (
    RoboCasaKitchenApplianceCategory,
    RoboCasaKitchenApplianceResolver,
    RoboCasaObjectCategory,
    RoboCasaObjectResolver,
)
from semantic_digital_twin.semantic_annotations.natural_language import (
    NaturalLanguageWithTypeDescription,
)
from semantic_digital_twin.semantic_annotations.semantic_annotations import (
    Apple,
    Cabinet,
    Door,
    Handle,
    Microwave,
)
from semantic_digital_twin.world_description.connections import RevoluteConnection

ROBOCASA_RESOURCES_DIR = (
    Path(__file__).resolve().parents[3]
    / "semantic_digital_twin"
    / "resources"
    / "robocasa"
)


def test_kitchen_appliance_resolver_resolves_known_category():
    resolver = RoboCasaKitchenApplianceResolver()
    assert resolver.resolve("cabinet") is Cabinet
    assert resolver.resolve("MICROWAVE") is Microwave


def test_kitchen_appliance_resolver_returns_none_for_unknown_category():
    resolver = RoboCasaKitchenApplianceResolver()
    assert resolver.resolve("paper_towel_holder") is None


def test_kitchen_appliance_resolver_resolves_compound_class_name():
    resolver = RoboCasaKitchenApplianceResolver()
    assert resolver.resolve("HingeCabinet") is Cabinet
    assert resolver.resolve("hinge_cabinet") is Cabinet


def test_kitchen_appliance_resolver_resolves_enum_category():
    resolver = RoboCasaKitchenApplianceResolver()
    assert resolver.resolve(RoboCasaKitchenApplianceCategory.CABINET) is Cabinet


def test_object_resolver_resolves_known_category():
    resolver = RoboCasaObjectResolver()
    assert resolver.resolve("apple") is Apple


def test_object_resolver_returns_none_for_unknown_category():
    resolver = RoboCasaObjectResolver()
    assert resolver.resolve("blender") is None


def test_object_resolver_resolves_enum_category():
    resolver = RoboCasaObjectResolver()
    assert resolver.resolve(RoboCasaObjectCategory.APPLE) is Apple


def test_category_from_class_name_converts_upper_camel_case():
    assert _category_from_class_name("HingeCabinet") == "hinge_cabinet"
    assert _category_from_class_name("Sink") == "sink"


def test_mjcf_document_from_element_copy_wraps_element_in_worldbody():
    fixture_body = ET.fromstring(
        '<body name="hinge_cabinet_main">'
        '<geom type="box" size="0.3 0.3 0.5"/>'
        "</body>"
    )

    document = _mjcf_document_from_element_copy(fixture_body)

    root = ET.fromstring(document)
    assert root.tag == "mujoco"
    assert root.find("worldbody/body").attrib["name"] == "hinge_cabinet_main"


def test_mjcf_document_from_element_copy_does_not_mutate_original_element():
    fixture_body = ET.fromstring(
        '<body name="hinge_cabinet_main">'
        '<geom type="box" size="0.3 0.3 0.5"/>'
        "</body>"
    )

    _mjcf_document_from_element_copy(fixture_body)

    # the original element must be unchanged, i.e. a copy (not the element itself) was reparented
    # into the new document
    assert [child.tag for child in fixture_body] == ["geom"]
    assert fixture_body.attrib["name"] == "hinge_cabinet_main"


def test_mjcf_document_from_element_copy_is_parseable_by_mjcf_parser():
    fixture_body = ET.fromstring(
        '<body name="hinge_cabinet_main">'
        '<inertial mass="5.0" pos="0 0 0" diaginertia="1 1 1"/>'
        '<geom type="box" size="0.3 0.3 0.5" contype="1" conaffinity="1" group="3"/>'
        "</body>"
    )

    world = MJCFParser.from_xml_string(
        _mjcf_document_from_element_copy(fixture_body)
    ).parse()

    assert world.get_body_by_name("hinge_cabinet_main") is not None


def test_attach_semantic_annotation_uses_kitchen_appliance_resolver():
    world = MJCFParser(str(ROBOCASA_RESOURCES_DIR / "cabinet_fixture.xml")).parse()
    body = world.get_body_by_name("hinge_cabinet_main")

    loader = RoboCasaDatasetLoader()
    loader._attach_semantic_annotation(world, body, "cabinet")

    annotations = world.get_semantic_annotations_by_type(Cabinet)
    assert len(annotations) == 1
    assert annotations[0].root == body


def test_attach_semantic_annotation_falls_back_to_natural_language():
    world = MJCFParser(str(ROBOCASA_RESOURCES_DIR / "cabinet_fixture.xml")).parse()
    body = world.get_body_by_name("hinge_cabinet_main")

    loader = RoboCasaDatasetLoader()
    loader._attach_semantic_annotation(world, body, "paper_towel_holder")

    annotations = world.get_semantic_annotations_by_type(
        NaturalLanguageWithTypeDescription
    )
    assert len(annotations) == 1
    assert annotations[0].type_description == "paper_towel_holder"


def test_attach_semantic_annotation_attaches_door_and_handle_sub_parts():
    world = MJCFParser(str(ROBOCASA_RESOURCES_DIR / "cabinet_fixture.xml")).parse()
    body = world.get_body_by_name("hinge_cabinet_main")
    door_body = world.get_body_by_name("hinge_cabinet_door")

    loader = RoboCasaDatasetLoader()
    loader._attach_semantic_annotation(world, body, "cabinet")

    [cabinet] = world.get_semantic_annotations_by_type(Cabinet)
    [door] = world.get_semantic_annotations_by_type(Door)
    [handle] = world.get_semantic_annotations_by_type(Handle)

    assert door in cabinet.doors
    assert door.root.name.name == "hinge_cabinet_door"
    # the handle is nested inside the door in the fixture, so it belongs to the door's annotation,
    # not directly to the cabinet's
    assert cabinet.handle is None
    assert door.handle == handle
    assert handle.root.name.name == "hinge_cabinet_handle"

    # the door's real hinge joint (parsed from the fixture's MJCF) must not have been disturbed by
    # recording the part-whole relationship
    assert isinstance(door_body.parent_connection, RevoluteConnection)
    assert door_body.parent_kinematic_structure_entity == body


def test_find_body_returns_none_for_missing_body():
    world = MJCFParser(str(ROBOCASA_RESOURCES_DIR / "cabinet_fixture.xml")).parse()
    assert RoboCasaDatasetLoader._find_body(world, "does_not_exist") is None


def test_find_body_falls_back_to_prefix_match():
    world = MJCFParser(str(ROBOCASA_RESOURCES_DIR / "cabinet_fixture.xml")).parse()

    body = RoboCasaDatasetLoader._find_body(world, "hinge_cabinet")

    assert body is not None
    assert body.name.name == "hinge_cabinet_main"


def test_apply_object_semantics_annotates_body_with_collision_not_root():
    world = MJCFParser(str(ROBOCASA_RESOURCES_DIR / "cabinet_fixture.xml")).parse()

    loader = RoboCasaDatasetLoader()
    loader._apply_object_semantics(world, "cabinet")

    annotations = world.get_semantic_annotations_by_type(Cabinet)
    assert len(annotations) == 1
    assert annotations[0].root.name.name == "hinge_cabinet_main"
    assert annotations[0].root != world.root
