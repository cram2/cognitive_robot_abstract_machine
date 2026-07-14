from dataclasses import dataclass
from typing import Self

import pytest
from typing_extensions import List

from krrood.entity_query_language.factories import variable
from krrood.ormatic.data_access_objects.helper import to_dao
from krrood.parametrization.feature_extraction.aggregations import (
    AggregationStatistic,
    aggregation_statistic,
    get_aggregation_class,
)
from krrood.parametrization.feature_extraction.feature_extractor import (
    FeatureExtractor,
)
from krrood.entity_query_language.core.mapped_variable import Call
from random_events.interval import SimpleInterval, Bound
from ..dataset.ormatic_interface import *  # type: ignore
from ..dataset.example_classes import (
    SceneObject,
    SceneRoom,
    KRROODPosition,
    KRROODOrientation,
    SceneObjectType,
    TestExParts,
    SceneRoomAggregations,
    TestExPartsAggregations,
)


@pytest.fixture
def example_scenario():
    obj1 = SceneObject(type=SceneObjectType.TABLE)
    obj2 = SceneObject(type=SceneObjectType.CHAIR)
    obj3 = SceneObject(type=...)
    obj4 = SceneObject(type=...)
    obj5 = SceneObject(type=...)
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1, obj2, obj3, obj4, obj5],
    )
    return room


def test_single_aggregation(example_scenario):
    room = example_scenario
    aggregation_cls = get_aggregation_class(type(room))
    aggregation_instance = aggregation_cls(instance=room, field_name="objects")
    aggregation_features = aggregation_instance.aggregation_features
    aggregations = aggregation_instance.symbolic_aggregation_features()
    assert aggregation_features == aggregations
    values = aggregation_instance.apply_mapping()
    assert len(aggregations) == 3
    closed = Bound.CLOSED
    assert values == {
        "chair_count": SimpleInterval.from_data(1, 4, closed, closed),
        "table_count": SimpleInterval.from_data(1, 4, closed, closed),
        "total_count": 5,
    }


def test_feature_extraction_with_aggregation_statistics(example_scenario):
    room = example_scenario
    extractor = FeatureExtractor.from_instances([to_dao(room)])

    agg_features = [f for f in extractor.features if isinstance(f, Call)]
    assert len(agg_features) == 3

    names = {f._name_ for f in agg_features}
    assert any("table" in n for n in names)
    assert any("chair" in n for n in names)

    values = extractor.apply_mapping(to_dao(room))
    assert 1 in values


def test_multiple_exchangeable_parts():
    obj1 = SceneObject(type=SceneObjectType.TABLE)
    obj2 = SceneObject(type=SceneObjectType.CHAIR)
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1, obj2],
    )
    room2 = SceneRoom(
        position=KRROODPosition(1, 1, 1),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[obj1],
    )
    test_ex_parts = TestExParts(objects=[obj1, obj2], rooms=[room, room2])

    extractor = FeatureExtractor.from_instances([to_dao(test_ex_parts)])
    assert len([f for f in extractor.features if isinstance(f, Call)]) == 4
    assert extractor.apply_mapping(to_dao(test_ex_parts)) == [1, 1, 2, 2]


def test_aggregation_count_values(example_scenario):
    room = example_scenario
    aggregation_cls = get_aggregation_class(type(room))
    aggregation_instance = aggregation_cls(instance=room, field_name="objects")
    values = aggregation_instance.apply_mapping()
    assert values["chair_count"] == SimpleInterval.from_data(
        1, 4, Bound.CLOSED, Bound.CLOSED
    )


def test_only_marked_methods_are_statistics():
    @dataclass
    class Owner:
        items: List[SceneObject]

    @dataclass
    class PartiallyMarkedAggregations(AggregationStatistic[Owner]):
        @aggregation_statistic("items")
        def marked_statistic(self) -> int:
            return 1

        def unmarked_helper(self) -> int:
            return 2

    owner = Owner(items=[SceneObject(type=SceneObjectType.TABLE)])
    instance = PartiallyMarkedAggregations(instance=owner, field_name="items")
    statistic_names = {function.__name__ for function in instance.aggregation_features}
    print(PartiallyMarkedAggregations.aggregation_registry)
    assert statistic_names == {"marked_statistic"}


def test_registry_is_isolated_between_unrelated_subclasses():
    @dataclass
    class OwnerA:
        items: List[SceneObject]

    @dataclass
    class OwnerB:
        items: List[SceneObject]

    @dataclass
    class AggregationsA(AggregationStatistic[OwnerA]):
        @aggregation_statistic("items")
        def count_a(self) -> int:
            return len(self.instance.items)

    @dataclass
    class AggregationsB(AggregationStatistic[OwnerB]):
        @aggregation_statistic("items")
        def count_b(self) -> int:
            return len(self.instance.items)

    owner_a = OwnerA(items=[SceneObject(type=SceneObjectType.TABLE)])
    owner_b = OwnerB(items=[SceneObject(type=SceneObjectType.CHAIR)])
    instance_a = AggregationsA(instance=owner_a, field_name="items")
    instance_b = AggregationsB(instance=owner_b, field_name="items")

    assert {f.__name__ for f in instance_a.aggregation_features} == {"count_a"}
    assert {f.__name__ for f in instance_b.aggregation_features} == {"count_b"}


def test_inherited_statistics_are_visible_in_subclass():
    @dataclass
    class Owner:
        items: List[SceneObject]

    @dataclass
    class BaseAggregations(AggregationStatistic[Owner]):
        @aggregation_statistic("items")
        def base_stat(self) -> int:
            return len(self.instance.items)

    @dataclass
    class DerivedAggregations(BaseAggregations):
        pass

    instance = DerivedAggregations(instance=Owner(items=[]), field_name="items")
    assert {f.__name__ for f in instance.aggregation_features} == {"base_stat"}


def test_own_registry_contains_only_directly_defined_methods():
    @dataclass
    class Owner:
        items: List[SceneObject]

    @dataclass
    class BaseAggregations(AggregationStatistic[Owner]):
        @aggregation_statistic("items")
        def base_stat(self) -> int:
            return len(self.instance.items)

    @dataclass
    class DerivedAggregations(BaseAggregations):
        @aggregation_statistic("items")
        def derived_stat(self) -> int:
            return 0

    own_names = {f.__name__ for f in DerivedAggregations.aggregation_registry["items"]}
    assert own_names == {"derived_stat"}

    instance = DerivedAggregations(instance=Owner(items=[]), field_name="items")
    assert {f.__name__ for f in instance.aggregation_features} == {
        "base_stat",
        "derived_stat",
    }


def test_aggregation_class_discovered_for_concrete_subclasses():
    assert get_aggregation_class(SceneRoom) is not None
    assert get_aggregation_class(TestExParts) is not None


def test_feature_extraction_over_empty_exchangeable_part_does_not_raise():
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[],
    )
    extractor = FeatureExtractor.from_instances([to_dao(room)])
    assert extractor is not None
    assert all(not isinstance(feature, Call) for feature in extractor.features)


# --- SceneObjectAggregationBase shared-base-class tests ---


def test_scene_room_aggregations_exposes_correct_feature_names_for_objects_field():
    """
    SceneRoomAggregations must inherit all three statistics from
    SceneObjectAggregationBase for the 'objects' field.
    """
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[],
    )
    instance = SceneRoomAggregations(instance=room, field_name="objects")
    feature_names = {feature.__name__ for feature in instance.aggregation_features}
    assert feature_names == {"chair_count", "table_count", "total_count"}


def test_test_ex_parts_aggregations_exposes_correct_feature_names_for_objects_field():
    """
    TestExPartsAggregations must inherit all three statistics from
    SceneObjectAggregationBase for the 'objects' field.
    """
    test_ex_parts = TestExParts(objects=[], rooms=[])
    instance = TestExPartsAggregations(instance=test_ex_parts, field_name="objects")
    feature_names = {feature.__name__ for feature in instance.aggregation_features}
    assert feature_names == {"chair_count", "table_count", "total_count"}


def test_base_class_statistics_are_not_duplicated_in_scene_room_aggregations():
    """
    Each statistic from SceneObjectAggregationBase should appear exactly once in
    SceneRoomAggregations.aggregation_features — no duplicates via multiple inheritance
    paths.
    """
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[],
    )
    instance = SceneRoomAggregations(instance=room, field_name="objects")
    feature_names = [feature.__name__ for feature in instance.aggregation_features]
    assert len(feature_names) == len(set(feature_names))


def test_base_class_statistics_are_not_duplicated_in_test_ex_parts_aggregations():
    """
    Each statistic from SceneObjectAggregationBase should appear exactly once in
    TestExPartsAggregations.aggregation_features — no duplicates via multiple
    inheritance paths.
    """
    test_ex_parts = TestExParts(objects=[], rooms=[])
    instance = TestExPartsAggregations(instance=test_ex_parts, field_name="objects")
    feature_names = [feature.__name__ for feature in instance.aggregation_features]
    assert len(feature_names) == len(set(feature_names))


def test_scene_room_aggregations_has_no_features_for_rooms_field():
    """
    SceneRoomAggregations owns no 'rooms' statistics — that field belongs only to
    TestExPartsAggregations.
    """
    room = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[],
    )
    instance = SceneRoomAggregations(instance=room, field_name="rooms")
    assert instance.aggregation_features == []


def test_test_ex_parts_aggregations_exposes_room_count_for_rooms_field():
    """
    TestExPartsAggregations must expose exactly the 'room_count' statistic for the
    'rooms' field.
    """
    test_ex_parts = TestExParts(objects=[], rooms=[])
    instance = TestExPartsAggregations(instance=test_ex_parts, field_name="rooms")
    feature_names = {feature.__name__ for feature in instance.aggregation_features}
    assert feature_names == {"room_count"}


def test_get_aggregation_class_returns_scene_room_aggregations_for_scene_room():
    """
    The registry must resolve SceneRoom to SceneRoomAggregations specifically, not just
    any AggregationStatistic subclass.
    """
    assert get_aggregation_class(SceneRoom) is SceneRoomAggregations


def test_get_aggregation_class_returns_test_ex_parts_aggregations_for_test_ex_parts():
    """
    The registry must resolve TestExParts to TestExPartsAggregations specifically.
    """
    assert get_aggregation_class(TestExParts) is TestExPartsAggregations


def test_base_class_objects_aggregation_produces_same_values_for_test_ex_parts_as_for_scene_room(
    example_scenario,
):
    """
    The inherited chair_count, table_count, and total_count methods must produce
    identical results regardless of which concrete subclass executes them, given the
    same 'objects' list.
    """
    room = example_scenario
    test_ex_parts = TestExParts(objects=room.objects, rooms=[])

    room_aggregation = SceneRoomAggregations(instance=room, field_name="objects")
    test_ex_parts_aggregation = TestExPartsAggregations(
        instance=test_ex_parts, field_name="objects"
    )

    assert room_aggregation.apply_mapping() == test_ex_parts_aggregation.apply_mapping()


def test_room_count_computes_correct_number_of_rooms():
    """
    The room_count statistic defined on TestExPartsAggregations must return the exact
    number of SceneRoom instances in the 'rooms' field.
    """
    room1 = SceneRoom(
        position=KRROODPosition(0, 0, 0),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[],
    )
    room2 = SceneRoom(
        position=KRROODPosition(1, 1, 1),
        orientation=KRROODOrientation(0, 0, 0, 1),
        objects=[],
    )
    test_ex_parts = TestExParts(objects=[], rooms=[room1, room2])
    instance = TestExPartsAggregations(instance=test_ex_parts, field_name="rooms")
    [room_count] = instance.apply_mapping().values()
    assert room_count == 2
