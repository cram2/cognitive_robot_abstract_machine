import logging

from krrood.entity_query_language.factories import entity, variable, in_, inference, an
from semantic_digital_twin.adapters.world_entity_kwargs_tracker import (
    WorldEntityWithIDKwargsTracker,
)
from semantic_digital_twin.reasoning.world_reasoner import WorldReasoner
from semantic_digital_twin.robots.abstract_robot import AbstractRobot
from semantic_digital_twin.robots.minimal_robot import MinimalRobot
from semantic_digital_twin.semantic_annotations.semantic_annotations import *
from semantic_digital_twin.testing import *
from semantic_digital_twin.world_description.world_entity import (
    KinematicStructureEntity,
)

try:
    from krrood.ripple_down_rules.user_interface.gui import RDRCaseViewer
    from PyQt6.QtWidgets import QApplication
except ImportError as e:
    logging.debug(e)
    QApplication = None
    RDRCaseViewer = None

try:
    from semantic_digital_twin.reasoning.world_rdr import world_rdr
except ImportError:
    world_rdr = None


@dataclass(eq=False)
class TestSemanticAnnotation(SemanticAnnotation):
    """
    A Generic semantic annotation for multiple bodies.
    """

    _private_entity: Optional[KinematicStructureEntity] = None
    public_entity: Optional[KinematicStructureEntity] = None
    _private_entity_list: List[KinematicStructureEntity] = field(default_factory=list)
    public_entity_list: List[KinematicStructureEntity] = field(default_factory=list)
    _private_annotation: Optional[SemanticAnnotation] = None
    public_annotation: Optional[SemanticAnnotation] = None
    _private_annotation_list: List[SemanticAnnotation] = field(default_factory=list)
    public_annotation_list: List[SemanticAnnotation] = field(default_factory=list)


def test_aggregate_bodies(kitchen_world):
    """
    Tests that SemanticAnnotation.kinematic_structure_entities aggregates:
    -  public direct kinematic structure entity fields
    - public list fields containing kinematic structure entities
    - public nested semantic annotations' kinematic structure entities
    but nothing from private fields
    The exact order is not specified by the contract, so we check set membership.
    """

    # Arrange: pick some existing bodies from the world fixture
    b0, b1, b2, b3 = kitchen_world.bodies[:4]

    # Private values (should NOT appear)
    private_entity = b0
    private_entity_list = [b0]
    private_annotation = TestSemanticAnnotation(public_entity=b0)
    private_annotation_list = [TestSemanticAnnotation(public_entity=b0)]

    # Public values (should appear)
    public_entity = b1
    public_entity_list = [b2]
    public_annotation = TestSemanticAnnotation(public_entity=b3)
    public_annotation_list = [TestSemanticAnnotation(public_entity=b1)]

    ann = TestSemanticAnnotation(
        _private_entity=private_entity,
        public_entity=public_entity,
        _private_entity_list=private_entity_list,
        public_entity_list=public_entity_list,
        _private_annotation=private_annotation,
        public_annotation=public_annotation,
        _private_annotation_list=private_annotation_list,
        public_annotation_list=public_annotation_list,
    )

    # Act
    aggregated_list = ann.kinematic_structure_entities

    # Expected to include public direct entities and entities from public lists
    expected_present = [public_entity, *public_entity_list]

    # Nested public annotations contribute their own public_entity
    if isinstance(public_annotation, SemanticAnnotation):
        expected_present.extend(public_annotation.kinematic_structure_entities)
    for x in public_annotation_list:
        expected_present.extend(x.kinematic_structure_entities)

    # Expected to exclude any private contributions
    unexpected = [private_entity, *private_entity_list]
    if isinstance(private_annotation, SemanticAnnotation):
        unexpected.extend(private_annotation.kinematic_structure_entities)
    for x in private_annotation_list:
        unexpected.extend(x.kinematic_structure_entities)

    sorting_key = lambda kse: kse.id
    # Check all expected are present
    assert sorted(aggregated_list, key=sorting_key) == sorted(
        expected_present, key=sorting_key
    )
    # no unexpected item
    assert set(aggregated_list) == set(expected_present) - set(unexpected)


def test_has_root_kinematic_structure_entity_aggregate_bodies(kitchen_world):
    annotation = SemanticEnvironmentAnnotation(root=kitchen_world.root)
    with kitchen_world.modify_world():
        kitchen_world.add_semantic_annotation(annotation)

    assert (
        annotation.kinematic_structure_entities
        == kitchen_world.kinematic_structure_entities
    )


def test_has_hinge_has_slider_aggregate_bodies():
    world = World()
    root = Body(name=PrefixedName("root"))
    with world.modify_world():
        world.add_kinematic_structure_entity(root)

    door_body = Body(name=PrefixedName("door_body"))
    drawer_body = Body(name=PrefixedName("drawer_body"))
    handle1_body = Body(name=PrefixedName("handle1_body"))
    handle2_body = Body(name=PrefixedName("handle2_body"))
    hinge_body = Body(name=PrefixedName("hinge_body"))
    slider_body = Body(name=PrefixedName("slider_body"))
    handle1 = Handle(root=handle1_body)
    handle2 = Handle(root=handle2_body)
    hinge = Hinge(root=hinge_body)
    slider = Slider(root=slider_body)
    drawer = Drawer(root=drawer_body)
    door = Door(root=door_body)
    with world.modify_world():
        world.add_connection(Connection(parent=root, child=door_body))
        world.add_connection(Connection(parent=root, child=drawer_body))
        world.add_connection(Connection(parent=root, child=handle2_body))
        world.add_connection(Connection(parent=root, child=handle1_body))
        world.add_connection(Connection(parent=root, child=hinge_body))
        world.add_connection(Connection(parent=root, child=slider_body))
        world.add_semantic_annotation(handle1)
        world.add_semantic_annotation(handle2)
        world.add_semantic_annotation(hinge)
        world.add_semantic_annotation(slider)
        world.add_semantic_annotation(drawer)
        world.add_semantic_annotation(door)
        door.add_handle(handle2)
        door.add_hinge(hinge)
        drawer.add_handle(handle1)
        drawer.add_slider(slider)

    expected_door_bodies = {door_body, handle2_body, hinge_body}
    expected_drawer_bodies = {drawer_body, handle1_body, slider_body}
    assert set(door.kinematic_structure_entities) == expected_door_bodies
    assert set(drawer.kinematic_structure_entities) == expected_drawer_bodies


def test_semantic_annotation_hash(apartment_world_setup):
    semantic_annotation1 = Handle(root=apartment_world_setup.bodies[0])
    semantic_annotation2 = Handle(root=apartment_world_setup.bodies[0])
    with apartment_world_setup.modify_world():
        apartment_world_setup.add_semantic_annotation(semantic_annotation1)
        apartment_world_setup.add_semantic_annotation(semantic_annotation2)

    # hash of semantic annotations should be based on their properties, not ids
    assert id(semantic_annotation1) != id(semantic_annotation2)
    assert hash(semantic_annotation1) == hash(semantic_annotation2)
    assert semantic_annotation1 == semantic_annotation2


def test_handle_semantic_annotation_eql(apartment_world_setup):
    body = variable(type_=Body, domain=apartment_world_setup.bodies)
    query = an(
        entity(inference(Handle)(root=body)).where(
            in_("handle", body.name.name.lower())
        )
    )
    handles = list(query.evaluate())
    assert len(handles) > 0


@pytest.mark.parametrize(
    "semantic_annotation_type, update_existing_semantic_annotations, scenario",
    [
        (Handle, False, None),
        (Drawer, False, None),
        (Wardrobe, False, None),
        (Door, False, None),
    ],
)
def test_infer_apartment_semantic_annotation(
    semantic_annotation_type,
    update_existing_semantic_annotations,
    scenario,
    apartment_world_setup,
):
    fit_rules_and_assert_semantic_annotations(
        apartment_world_setup,
        semantic_annotation_type,
        update_existing_semantic_annotations,
        scenario,
    )


@pytest.mark.skipif(world_rdr is None, reason="requires world_rdr")
def test_generated_semantic_annotations(kitchen_world):
    found_semantic_annotations = world_rdr.classify(kitchen_world)[
        "semantic_annotations"
    ]
    drawer_container_names = [
        v.root.name.name
        for v in found_semantic_annotations
        if isinstance(v, HasCaseAsRootBody)
    ]
    assert len(drawer_container_names) == 19


@pytest.mark.order("second_to_last")
def test_apartment_semantic_annotations(apartment_world_setup):
    world_reasoner = WorldReasoner(apartment_world_setup)
    world_reasoner.fit_semantic_annotations(
        [Handle, Drawer, Wardrobe],
        world_factory=lambda: apartment_world_setup,
        scenario=None,
    )

    found_semantic_annotations = world_reasoner.infer_semantic_annotations()
    drawer_container_names = [
        v.root.name.name
        for v in found_semantic_annotations
        if isinstance(v, HasCaseAsRootBody)
    ]
    assert len(drawer_container_names) == 27


def fit_rules_and_assert_semantic_annotations(
    world, semantic_annotation_type, update_existing_semantic_annotations, scenario
):
    world_reasoner = WorldReasoner(world)
    world_reasoner.fit_semantic_annotations(
        [semantic_annotation_type],
        update_existing_semantic_annotations=update_existing_semantic_annotations,
        world_factory=lambda: world,
        scenario=scenario,
    )

    found_semantic_annotations = world_reasoner.infer_semantic_annotations()
    assert any(
        isinstance(v, semantic_annotation_type) for v in found_semantic_annotations
    )


def test_semantic_annotation_serialization_deserialization_once(apartment_world_setup):
    handle_body = apartment_world_setup.bodies[0]
    door_body = apartment_world_setup.bodies[1]

    handle = Handle(root=handle_body)
    door = Door(root=door_body, handle=handle)
    with apartment_world_setup.modify_world():
        apartment_world_setup.add_semantic_annotation(handle)
        apartment_world_setup.add_semantic_annotation(door)

    door_se = door.to_json()

    with apartment_world_setup.modify_world():
        apartment_world_setup.remove_semantic_annotation(door)

    tracker = WorldEntityWithIDKwargsTracker.from_world(apartment_world_setup)
    kwargs = tracker.create_kwargs()

    door_de = Door.from_json(door_se, **kwargs)

    assert type(door.handle) == type(door_de.handle)
    assert type(door.root) == type(door_de.root)


def test_minimal_robot_annotation(pr2_world_state_reset):
    urdf_path = "package://iai_pr2_description/robots/pr2_with_ft2_cableguide.xacro"
    world_copy = URDFParser.from_xacro(urdf_path).parse()
    with world_copy.modify_world():
        MinimalRobot.from_world(world_copy)
        pr2_root = world_copy.root
        localization_body = Body(name=PrefixedName("odom_combined"))
        world_copy.add_kinematic_structure_entity(localization_body)
        c_root_bf = OmniDrive.create_with_dofs(
            parent=localization_body, child=pr2_root, world=world_copy
        )
        world_copy.add_connection(c_root_bf)

    robot = world_copy.get_semantic_annotations_by_type(AbstractRobot)[0]
    pr2 = pr2_world_state_reset.get_semantic_annotations_by_type(AbstractRobot)[0]
    assert len(robot.bodies) == len(pr2.bodies)
    assert len(robot.connections) == len(pr2.connections)
