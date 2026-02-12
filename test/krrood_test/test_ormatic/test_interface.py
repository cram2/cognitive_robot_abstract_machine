import pytest
from sqlalchemy import select, inspect

from krrood.ormatic.alternative_mappings import FunctionMapping, UncallableFunction
from krrood.ormatic.dao import (
    to_dao,
    is_data_column,
    ToDataAccessObjectState,
    get_dao_class,
)
from krrood.ormatic.exceptions import NoDAOFoundError
from ..dataset.example_classes import *
from ..dataset.ormatic_interface import *


def test_position(session, database):
    p1 = Position(1, 2, 3)

    p1dao: PositionDAO = PositionDAO.to_dao(p1)
    assert p1.x == p1dao.x
    assert p1.y == p1dao.y
    assert p1.z == p1dao.z

    session.add(p1dao)
    session.commit()

    # krrood_test the content of the database
    queried_p1 = session.scalars(select(PositionDAO)).one()

    assert p1.x == queried_p1.x
    assert p1.y == queried_p1.y
    assert p1.z == queried_p1.z

    p1_reconstructed = queried_p1.from_dao()
    assert p1 == p1_reconstructed


def test_position4d(session, database):
    p4d = Position4D(1.0, 2.0, 3.0, 4.0)

    p4d_dao = Position4DDAO.to_dao(p4d)
    assert p4d.x == p4d_dao.x
    assert p4d.y == p4d_dao.y
    assert p4d.z == p4d_dao.z
    assert p4d.w == p4d_dao.w

    session.add(p4d_dao)
    session.commit()

    # krrood_test the content of the database
    # Note: Polymorphic queries don't work correctly yet, so we query directly for Position4DDAO objects
    queried_p4d = session.scalars(select(PositionDAO)).one()

    assert p4d.x == queried_p4d.x
    assert p4d.y == queried_p4d.y
    assert p4d.z == queried_p4d.z
    assert p4d.w == queried_p4d.w

    p4d_reconstructed = queried_p4d.from_dao()
    assert p4d == p4d_reconstructed


def test_orientation(session, database):
    o1 = Orientation(1.0, 2.0, 3.0, None)

    o1dao = OrientationDAO.to_dao(o1)
    assert o1.x == o1dao.x
    assert o1.y == o1dao.y
    assert o1.z == o1dao.z
    assert o1.w == o1dao.w

    session.add(o1dao)
    session.commit()

    # krrood_test the content of the database
    queried_o1 = session.scalars(select(OrientationDAO)).one()

    assert o1.x == queried_o1.x
    assert o1.y == queried_o1.y
    assert o1.z == queried_o1.z
    assert o1.w == queried_o1.w

    o1_reconstructed = queried_o1.from_dao()
    assert o1 == o1_reconstructed


def test_pose(session, database):
    p1 = Position(1, 2, 3)
    o1 = Orientation(1.0, 2.0, 3.0, None)
    pose = Pose(p1, o1)

    posedao = PoseDAO.to_dao(pose)
    assert isinstance(posedao.position, PositionDAO)
    assert isinstance(posedao.orientation, OrientationDAO)

    session.add(posedao)
    session.commit()

    queried = session.scalars(select(PoseDAO)).one()
    assert queried.position is not None
    assert queried.orientation is not None
    assert queried == posedao
    queried = queried.from_dao()
    assert pose == queried


def test_atom(session, database):
    atom = Atom(Element.C, 1, 2.0)
    atomdao = AtomDAO.to_dao(atom)
    assert atomdao.element == Element.C

    session.add(atomdao)
    session.commit()

    queried = session.scalars(select(AtomDAO)).one()
    assert isinstance(queried.element, Element)

    atom_from_session = queried.from_dao()
    assert atom == atom_from_session


def test_entity_and_derived(session, database):
    entity = Entity("TestEntity")
    derived = DerivedEntity("DerivedEntity")

    entity_dao = to_dao(entity)
    derived_dao = to_dao(derived)

    session.add(entity_dao)
    session.add(derived_dao)
    session.commit()

    # krrood_test the content of the database
    queried_entity = session.scalars(select(CustomEntityDAO)).first()
    queried_derived = session.scalars(select(DerivedEntityDAO)).first()

    assert entity.name == queried_entity.overwritten_name
    assert derived.name == queried_derived.overwritten_name
    assert derived.description == queried_derived.description

    entity_reconstructed = queried_entity.from_dao()
    derived_reconstructed = queried_derived.from_dao()

    assert entity.name == entity_reconstructed.name
    assert derived.name == derived_reconstructed.name
    assert derived.description == derived_reconstructed.description


def test_parent_and_child(session, database):
    parent = Parent("TestParent")
    child_mapped = ChildMapped("ChildMapped", 42)
    child_not_mapped = ChildNotMapped("a", 2, {})

    parent_dao = to_dao(parent)
    child_dao = to_dao(child_mapped)

    assert parent.name == parent_dao.name
    assert child_mapped.name == child_dao.name
    assert child_mapped.attribute1 == child_dao.attribute1

    session.add(parent_dao)
    session.add(child_dao)
    session.commit()

    # krrood_test the content of the database
    queried_parent = session.scalars(select(ParentDAO)).all()
    queried_child = session.scalars(select(ChildMappedDAO)).all()

    assert child_dao in queried_parent
    assert queried_child[0] in queried_parent

    assert parent.name == queried_parent[0].name
    assert child_mapped.name == queried_child[0].name
    assert child_mapped.attribute1 == queried_child[0].attribute1

    parent_reconstructed = queried_parent[0].from_dao()
    child_reconstructed = queried_child[0].from_dao()

    assert parent.name == parent_reconstructed.name
    assert child_mapped.name == child_reconstructed.name
    assert child_mapped.attribute1 == child_reconstructed.attribute1


def test_node(session, database):
    n1 = Node()
    n2 = Node(parent=n1)
    n3 = Node(parent=n1)

    n2dao = NodeDAO.to_dao(n2)

    session.add(n2dao)
    session.commit()

    results = session.scalars(select(NodeDAO)).all()
    assert len(results) == 2


def test_position_type_wrapper(session, database):
    wrapper = PositionTypeWrapper(Position)
    dao = PositionTypeWrapperDAO.to_dao(wrapper)
    assert dao.position_type == wrapper.position_type
    session.add(dao)
    session.commit()

    result = session.scalars(select(PositionTypeWrapperDAO)).one()
    assert result == dao


def test_positions(session, database):
    p1 = Position(1, 2, 3)
    p2 = Position(2, 3, 4)
    positions = Positions([p1, p2], ["a", "b", "c"])
    dao = PositionsDAO.to_dao(positions)
    assert len(dao.positions) == 2

    session.add(dao)
    session.commit()

    positions_results = session.scalars(select(PositionDAO)).all()
    assert len(positions_results) == 2

    result = session.scalars(select(PositionsDAO)).one()
    assert result.some_strings == positions.some_strings

    assert len(result.positions) == 2


def test_positions_with_duplicated_entry_in_list(session, database):
    p1 = Position(1, 2, 3)
    positions = Positions([p1, p1], ["a", "b", "c"])
    dao: PositionsDAO = to_dao(positions)
    assert len(dao.positions) == 2
    session.add(dao)
    session.commit()

    associations_in_db = session.execute(
        select(PositionsDAO_positions_association)
    ).all()
    assert len(associations_in_db) == 2

    queried = session.scalars(select(PositionsDAO)).one()
    assert len(queried.positions) == 2


def test_double_position_aggregator(session, database):
    p1, p2, p3 = Position(1, 2, 3), Position(2, 3, 4), Position(3, 4, 5)
    dpa = DoublePositionAggregator([p1, p2], [p1, p3])
    dpa_dao = DoublePositionAggregatorDAO.to_dao(dpa)
    session.add(dpa_dao)
    session.commit()

    queried_positions = session.scalars(select(PositionDAO)).all()
    assert len(queried_positions) == 3

    queried = session.scalars(select(DoublePositionAggregatorDAO)).one()
    assert queried == dpa_dao
    assert queried.positions1[0].target in queried_positions


def test_kinematic_chain_and_torso(session, database):
    k1 = KinematicChain("a")
    k2 = KinematicChain("b")
    torso = Torso("t", [k1, k2])
    torso_dao = TorsoDAO.to_dao(torso)

    session.add(torso_dao)
    session.commit()

    queried_torso = session.scalars(select(TorsoDAO)).one()
    assert queried_torso == torso_dao


def test_custom_types(session, database):
    ogs = OriginalSimulatedObject(Bowl(), 1)
    ogs_dao = OriginalSimulatedObjectDAO.to_dao(ogs)
    assert ogs.concept == ogs_dao.concept

    session.add(ogs_dao)
    session.commit()

    queried = session.scalars(select(OriginalSimulatedObjectDAO)).one()
    assert ogs_dao == queried
    assert isinstance(queried.concept, Bowl)


def test_inheriting_from_explicit_mapping(session, database):
    entity: DerivedEntity = DerivedEntity(name="TestEntity")

    entity_dao = DerivedEntityDAO.to_dao(entity)
    assert isinstance(entity_dao, DerivedEntityDAO)
    session.add(entity_dao)
    session.commit()

    queried_entities_og = session.scalars(select(CustomEntityDAO)).all()
    queried_entity = session.scalars(select(DerivedEntityDAO)).one()
    assert queried_entity.description is not None
    assert queried_entity.overwritten_name is not None
    assert queried_entity in queried_entities_og

    reconstructed = queried_entity.from_dao()
    assert reconstructed == entity


def test_entity_association(session, database):
    entity = Entity("TestEntity")
    association = EntityAssociation(entity=entity, a=["a"])

    association_dao = to_dao(association)

    assert isinstance(association_dao, EntityAssociationDAO)
    assert isinstance(association_dao.entity, CustomEntityDAO)

    session.add(association_dao)
    session.commit()

    queried_association = session.scalars(select(EntityAssociationDAO)).one()
    assert queried_association.entity.overwritten_name == entity.name
    reconstructed = queried_association.from_dao()
    assert reconstructed == association


def test_assertion(session, database):
    p = NotMappedParent()
    with pytest.raises(NoDAOFoundError):
        to_dao(p)


def test_PositionsSubclassWithAnotherPosition(session, database):
    position = Position(1, 2, 3)
    obj = PositionsSubclassWithAnotherPosition([position], ["a", "b", "c"], position)
    dao: PositionsSubclassWithAnotherPositionDAO = to_dao(obj)

    session.add(dao)
    session.commit()


def test_inheriting_from_inherited_class(session, database):
    position_5d = Position5D(1, 2, 3, 4, 5)
    position_4d = Position4D(1, 2, 3, 4)

    position_4d_dao = to_dao(position_4d)
    position_5d_dao = to_dao(position_5d)

    session.add(position_4d_dao)
    session.commit()

    session.add(position_5d_dao)
    session.commit()

    queried_position_5d = session.scalars(select(Position5DDAO)).one()
    queried_position_4d = session.scalars(select(Position4DDAO)).all()
    queried_position = session.scalars(select(PositionDAO)).all()
    columns = [
        column
        for column in queried_position_5d.__table__.columns
        if is_data_column(column)
    ]
    assert queried_position_5d in queried_position_4d
    assert queried_position_5d in queried_position
    assert queried_position_4d[0] in queried_position
    assert len(columns) == 1  # w column


def test_backreference_with_mapping(session, database):
    back_ref = Backreference({1: 1})
    ref = Reference(0, back_ref)
    back_ref.reference = ref

    dao = to_dao(ref)
    session.add(dao)
    session.commit()
    reconstructed = dao.from_dao()

    # Check individual properties instead of comparing entire objects
    assert reconstructed.value == ref.value
    assert reconstructed.backreference is not None
    assert reconstructed.backreference.unmappable == back_ref.unmappable

    # Check that the circular reference is correctly reconstructed
    assert reconstructed.backreference.reference is reconstructed


def test_alternative_mapping_aggregator(session, database):
    e1 = Entity("E1")
    e2 = Entity("E2")
    e3 = Entity("E3")

    ama = AlternativeMappingAggregator([e1, e2], [e2, e3])
    dao = to_dao(ama)

    assert dao.entities1[1].target is dao.entities2[0].target

    session.add(dao)
    session.commit()

    queried = session.scalars(select(AlternativeMappingAggregatorDAO)).one()
    reconstructed = queried.from_dao()
    assert reconstructed.entities1[1] is reconstructed.entities2[0]


def test_container_item(session, database):
    i1 = ItemWithBackreference(0)
    i2 = ItemWithBackreference(1)
    container = ContainerGeneration([i1, i2])

    dao = to_dao(container)
    session.add(dao)
    session.commit()

    queried_items = session.scalars(select(ItemWithBackreferenceDAO)).all()
    assert len(queried_items) == 2

    queried_container = session.scalar(select(ContainerGenerationDAO))
    assert queried_container is queried_items[0].container

    reconstructed_item = queried_items[0].from_dao()
    assert len(reconstructed_item.container.items) == 2


def test_nested_mappings(session, database):
    shape_1 = Shape("rectangle", Transformation(Vector(1), Rotation(1)))
    shape_2 = Shape("circle", Transformation(Vector(2), Rotation(2)))
    shape_3 = Shape("rectangle", Transformation(Vector(3), Rotation(3)))
    shapes = Shapes([shape_1, shape_2, shape_3])
    more_shapes = MoreShapes([shapes, shapes])
    dao = to_dao(more_shapes)
    session.add(dao)
    session.commit()


def test_vector_mapped(session, database):
    vector = Vector(1.0)
    vector_mapped = VectorsWithProperty([vector])
    dao = to_dao(vector_mapped)

    session.add(dao)
    session.commit()

    queried = session.scalars(select(VectorsWithPropertyMappedDAO)).one()
    reconstructed = queried.from_dao()

    assert reconstructed.vectors[0].x == vector.x


def test_test_classes(session, database):
    parent = ParentBase("x", 0)
    child = ChildBase("y", 0)
    parent_mapping = ParentBaseMapping("x")
    child_mapping = ChildBaseMapping("y")

    parent_dao = to_dao(parent_mapping)
    child_dao = to_dao(child_mapping)

    session.add_all([parent_dao, child_dao])
    session.commit()

    child_result_dao = session.scalars(select(ChildBaseMappingDAO)).one()
    parent_result_dao = session.scalars(select(ParentBaseMappingDAO)).first()

    child_from_dao = child_result_dao.from_dao()
    parent_from_dao = parent_result_dao.from_dao()

    assert child_result_dao == child_dao
    assert parent_result_dao == parent_dao
    assert child_from_dao == child
    assert parent_from_dao == parent


def test_private_factories(session, database):
    obj = PrivateDefaultFactory()
    dao = to_dao(obj)
    reconstructed: PrivateDefaultFactory = dao.from_dao()
    assert reconstructed._private_list == []


def test_relationship_overloading(session, database):
    obj = RelationshipChild(Position(1, 2, 3))
    dao = to_dao(obj)
    session.add(dao)
    session.commit()

    queried = session.scalars(select(RelationshipParentDAO)).one()
    reconstructed = queried.from_dao()
    assert reconstructed.positions == Position(1, 2, 3)


def test_alternative_mapping_inheritance(session, database):
    assert issubclass(ChildBaseMappingDAO, ParentBaseMappingDAO)


def test_inheritance_mapper_args(session, database):
    assert InheritanceBaseWithoutSymbolButAlternativelyMappedMappingDAO.__mapper_args__


def test_to_dao_alternatively_mapped_parent(session, database):
    ch2 = ChildLevel2NormallyMapped(1, [Entity("a")], 2, 3)
    ch2_dao: ChildLevel2NormallyMappedDAO = to_dao(ch2)

    result_by_hand = ChildLevel2NormallyMappedDAO(
        derived_attribute="1",
        entities=[
            ParentAlternativelyMappedMappingDAO_entities_association(
                target=CustomEntityDAO(overwritten_name="a")
            )
        ],
        level_one_attribute=2,
        level_two_attribute=3,
    )

    assert isinstance(ch2_dao.entities[0].target, CustomEntityDAO)
    assert (
        ch2_dao.entities[0].target.overwritten_name
        == result_by_hand.entities[0].target.overwritten_name
    )
    assert len(ch2_dao.entities) == len(result_by_hand.entities)
    assert ch2_dao.level_one_attribute == result_by_hand.level_one_attribute
    assert ch2_dao.level_two_attribute == result_by_hand.level_two_attribute


def test_callable_alternative_mapping():
    callable_mapping = FunctionMapping.from_domain_object(module_level_function)
    reconstructed = callable_mapping.to_domain_object()
    assert reconstructed() == 1


def test_callable_alternative_mapping_instance_method():
    callable_mapping = FunctionMapping.from_domain_object(
        CallableWrapper.custom_instance_method
    )
    reconstructed = callable_mapping.to_domain_object()
    assert reconstructed is CallableWrapper.custom_instance_method


def test_callable_alternative_mapping_class_method():
    callable_mapping = FunctionMapping.from_domain_object(
        CallableWrapper.custom_class_method
    )
    reconstructed = callable_mapping.to_domain_object()
    assert reconstructed == CallableWrapper.custom_class_method


def test_callable_alternative_mapping_static_method():
    callable_mapping = FunctionMapping.from_domain_object(
        CallableWrapper.custom_static_method
    )
    reconstructed = callable_mapping.to_domain_object()
    assert reconstructed is CallableWrapper.custom_static_method
    assert reconstructed() == 4


def test_anonymous_function_mapping():
    func = lambda: 0
    callable_mapping = FunctionMapping.from_domain_object(func)
    reconstructed = callable_mapping.to_domain_object()
    with pytest.raises(UncallableFunction):
        reconstructed()


def test_callable_mapping(session, database):

    obj = CallableWrapper(module_level_function)
    assert obj.func() == 1

    dao = to_dao(obj)

    from_dao = dao.from_dao()
    assert from_dao.func() == 1


def test_uuid(session, database):
    obj = UUIDWrapper(uuid.uuid4())
    dao = to_dao(obj)
    session.add(dao)
    session.commit()

    queried = session.scalars(select(UUIDWrapperDAO)).one()
    assert queried.identification == obj.identification


def test_list_of_custom_type(session, database):
    obj = UUIDWrapper(uuid.uuid4(), [uuid.uuid4(), uuid.uuid4()])
    dao = to_dao(obj)

    session.add(dao)
    session.commit()

    queried = session.scalars(select(UUIDWrapperDAO)).one()
    assert queried.identification == obj.identification
    assert queried.other_identifications == obj.other_identifications


def test_json_integration(session, database):
    obj = JSONWrapper(JSONSerializableClass(1, 2), [JSONSerializableClass(3, 4)])
    dao = to_dao(obj)
    session.add(dao)
    session.commit()

    queried = session.scalars(select(JSONWrapperDAO)).one()
    reconstructed = queried.from_dao()
    assert reconstructed == obj


def test_many_to_many_with_same_type(session, database):

    state = ToDataAccessObjectState()
    position = Position(1, 2, 3)
    ps1 = Positions([position], ["a"])
    ps2 = Positions([position], ["a"])

    ps1_dao = to_dao(ps1, state)
    ps2_dao = to_dao(ps2, state)

    session.add_all([ps1_dao, ps2_dao])
    session.commit()
    session.expunge_all()

    q1 = select(PositionDAO)
    r = session.scalars(q1).all()
    assert len(r) == 1

    q = select(PositionsDAO)
    r_ps1, r_ps2 = session.scalars(q).all()

    assert r_ps1.positions[0].target is r_ps2.positions[0].target


def test_multiple_inheritance(session, database):
    assert issubclass(MultipleInheritanceDAO, (PrimaryBaseDAO, MixinDAO))
    obj = MultipleInheritance(
        primary_attribute="p", mixin_attribute="m", extra_attribute="e"
    )
    dao = to_dao(obj)
    assert hasattr(dao, "extra_attribute")
    assert hasattr(dao, "mixin_attribute")
    assert hasattr(dao, "primary_attribute")
    session.add(dao)
    session.commit()

    queried = session.scalars(select(MultipleInheritanceDAO)).one()
    reconstructed = queried.from_dao()
    assert reconstructed == obj


def test_list_of_enum(session, database):
    obj = ListOfEnum([TestEnum.OPTION_A, TestEnum.OPTION_B, TestEnum.OPTION_C])
    dao = to_dao(obj)

    session.add(dao)
    session.commit()

    queried = session.scalars(select(ListOfEnumDAO)).one()
    reconstructed = queried.from_dao()
    assert reconstructed == obj
    assert reconstructed.list_of_enum == [
        TestEnum.OPTION_A,
        TestEnum.OPTION_B,
        TestEnum.OPTION_C,
    ]


def test_persons(session, database):
    p1 = Person(name="Alice")
    p2 = Person(name="Bob")
    p1.knows.append(p2)

    dao = to_dao(p1)
    session.add(dao)
    session.commit()

    q = session.scalar(select(PersonDAO).where(PersonDAO.name == "Alice"))
    assert q.name == "Alice"
    assert q.knows[0].target.name == "Bob"


def test_underspecified_types():
    dao_class = get_dao_class(UnderspecifiedTypesContainer)
    assert dao_class is not None
    inst = inspect(dao_class)
    column_names = [c_attr.key for c_attr in inst.mapper.column_attrs]
    assert "any_list" not in column_names
    assert "any_field" not in column_names


def test_position_set(session, database):
    p1, p2 = Position(1, 2, 3), Position(2, 3, 4)
    obj = TestPositionSet({p1, p2})
    dao = to_dao(obj)
    session.add(dao)
    session.commit()

    r = session.scalars(select(TestPositionSetDAO)).one()
    reconstructed = r.from_dao()
    assert reconstructed == obj


def test_post_init_and_circular_reference(session, database):
    """
    Test the 4-phase from_dao logic with __post_init__ and circular references.
    """
    c1_dao = ContainerGenerationDAO()
    i1_dao = ItemWithBackreferenceDAO(value=10)
    i2_dao = ItemWithBackreferenceDAO(value=20)

    c1_dao.items = [
        ContainerGenerationDAO_items_association(target=i1_dao),
        ContainerGenerationDAO_items_association(target=i2_dao),
    ]
    i1_dao.container = c1_dao
    i2_dao.container = c1_dao

    session.add(c1_dao)
    session.commit()

    # Clear session to ensure we are loading from DB
    session.expunge_all()

    queried_c1_dao = session.scalars(select(ContainerGenerationDAO)).one()

    # Reconstruct domain object
    c1 = queried_c1_dao.from_dao()

    assert isinstance(c1, ContainerGeneration)
    assert len(c1.items) == 2
    assert c1.items[0].value == 10
    assert c1.items[1].value == 20

    # Check if __post_init__ was called and backreferences are set
    assert c1.items[0].container is c1
    assert c1.items[1].container is c1


def test_polymorphic_enum(session, database):
    v1 = PolymorphicEnumAssociation(ChildEnum1.A)
    v2 = PolymorphicEnumAssociation(ChildEnum1.B)
    v3 = PolymorphicEnumAssociation(ChildEnum2.B)
    v4 = PolymorphicEnumAssociation(ChildEnum2.C)

    dao_1, dao_2, dao_3, dao_4 = to_dao(v1), to_dao(v2), to_dao(v3), to_dao(v4)

    session.add_all([dao_1, dao_2, dao_3, dao_4])
    session.commit()

    statement = select(PolymorphicEnumAssociationDAO)
    r1, r2, r3, r4 = session.scalars(statement).all()

    assert r1.from_dao() == v1
    assert r2.from_dao() == v2
    assert r3.from_dao() == v3
    assert r4.from_dao() == v4

    statement = select(PolymorphicEnumAssociationDAO).where(
        PolymorphicEnumAssociationDAO.value == ChildEnum1.B
    )
    r = session.scalars(statement).all()
    assert len(r) == 1


def test_generic_class(session, database):

    assert issubclass(GenericClass_floatDAO, GenericClassDAO)
    assert issubclass(GenericClass_PositionDAO, GenericClassDAO)
    assert hasattr(GenericClass_PositionDAO, "value")
    assert hasattr(GenericClass_PositionDAO, "optional_value")
    assert hasattr(GenericClass_PositionDAO, "container")

    assert hasattr(GenericClassAssociationDAO, "associated_value")
    generic_position = GenericClass[Position](Position(1.0, 2.0, 3.0))
    obj = GenericClassAssociation(
        associated_value=GenericClass[float](1.0),
        associated_value_list=[
            generic_position,
            generic_position,
        ],
        associated_value_not_parametrized=generic_position,
        associated_value_not_parametrized_list=[GenericClass(2.0)],
    )
    dao: GenericClassAssociationDAO = to_dao(obj)
    assert isinstance(dao.associated_value, GenericClass_floatDAO)
    assert dao.associated_value.value == 1

    session.add(dao)
    session.commit()

    # check that there exist two relations in the generic position relations table.
    q = session.execute(
        select(GenericClassAssociationDAO_associated_value_list_association)
    ).all()
    assert len(q) == 2

    q = session.scalar(select(GenericClassAssociationDAO))
    assert q.associated_value.value == 1

    reconstructed: GenericClassAssociation = q.from_dao()
    assert reconstructed.associated_value == obj.associated_value
    assert len(reconstructed.associated_value_list) == 2
    assert reconstructed.associated_value_list == obj.associated_value_list
    assert reconstructed.associated_value_not_parametrized is None
    assert reconstructed.associated_value_not_parametrized_list == []
