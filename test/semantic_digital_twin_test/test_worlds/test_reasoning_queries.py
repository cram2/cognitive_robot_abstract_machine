from semantic_digital_twin.predetermined_maps.kitchen_environment import KitchenEnvironment
from semantic_digital_twin.reasoning.queries import list_semantic_annotations_on_surfaces, \
    get_next_object_using_planar_distance, goal_surface_of_object, filter_annotations_by_color, \
    annotation_class_by_label, sort_annotations_by_volume
from semantic_digital_twin.semantic_annotations.semantic_annotations import *
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.geometry import Color


def test_load_environment_returns_world():
    """
    Tests that loading the environment returns a World object with the correct root name.
    """
    world = KitchenEnvironment().get_world()
    assert isinstance(world, World)
    assert world.root.name == PrefixedName("root")


def test_list_semantic_annotations_on_surfaces(kitchen_environment_fixture):
    """
    Tests that giving Table annotations gives a list of the correct annotation on top.
    """
    table1 = kitchen_environment_fixture.get_semantic_annotation_by_name("fruit_table")
    table2 = kitchen_environment_fixture.get_semantic_annotation_by_name("vegetable_table")
    table3 = kitchen_environment_fixture.get_semantic_annotation_by_name("empty_table")
    apple = kitchen_environment_fixture.get_semantic_annotation_by_name("apple")
    carrot = kitchen_environment_fixture.get_semantic_annotation_by_name("carrot")
    orange = kitchen_environment_fixture.get_semantic_annotation_by_name("orange")
    lettuce = kitchen_environment_fixture.get_semantic_annotation_by_name("lettuce")
    banana1 = kitchen_environment_fixture.get_semantic_annotation_by_name("banana1")

    assert list_semantic_annotations_on_surfaces([table1, table2, table3], kitchen_environment_fixture) == [
        apple,
        orange,
        banana1,
        carrot,
        lettuce
    ]
    assert list_semantic_annotations_on_surfaces([], kitchen_environment_fixture) == []


def test_get_next_object_using_planar_distance(kitchen_environment_fixture):
    """
    Tests the functionality of the `query_get_next_object_euclidean_x_y` function to verify that it accurately identifies
    the next objects based on their Euclidean proximity within a simulation world. The test involves setting up a virtual
    world, retrieving specific objects and annotations, and validating the results returned by the function against
    predetermined expectations.

    :raises AssertionError: If any of the function assertions fail during testing.
    """
    toya = kitchen_environment_fixture.get_body_by_name("base_link_body")
    table1 = kitchen_environment_fixture.get_semantic_annotation_by_name("fruit_table")
    table2 = kitchen_environment_fixture.get_semantic_annotation_by_name("vegetable_table")
    table3 = kitchen_environment_fixture.get_semantic_annotation_by_name("empty_table")
    apple = kitchen_environment_fixture.get_semantic_annotation_by_name("apple")
    carrot = kitchen_environment_fixture.get_semantic_annotation_by_name("carrot")
    orange = kitchen_environment_fixture.get_semantic_annotation_by_name("orange")
    lettuce = kitchen_environment_fixture.get_semantic_annotation_by_name("lettuce")
    banana1 = kitchen_environment_fixture.get_semantic_annotation_by_name("banana1")

    assert get_next_object_using_planar_distance(toya, table1, Vector3.Z()).tolist() == [orange, banana1, apple]
    assert get_next_object_using_planar_distance(toya, table2, Vector3.Z()).tolist() == [
        carrot,
        lettuce,
    ]
    assert get_next_object_using_planar_distance(apple.bodies[0], table1, Vector3.Z()).tolist() == [apple, banana1, orange]
    assert get_next_object_using_planar_distance(apple.bodies[0], table1, Vector3.Y()).tolist() == [apple, orange, banana1]

    assert get_next_object_using_planar_distance(toya, table3, Vector3.Z()).tolist() == []


def test_goal_surface_of_object(kitchen_environment_fixture):
    """
    Tests the `goal_surface_of_object` function for determining the surface of the most suitable
    semantic annotation object from a set of candidates based on similarity and
    other constraints. The function is evaluated under multiple scenarios to verify
    its logic in choosing the correct table, handling empty tables, and cases with
    no valid candidates.
    """
    table1 = kitchen_environment_fixture.get_semantic_annotation_by_name("fruit_table")
    table2 = kitchen_environment_fixture.get_semantic_annotation_by_name("vegetable_table")
    table3 = kitchen_environment_fixture.get_semantic_annotation_by_name("empty_table")
    table4 = kitchen_environment_fixture.get_semantic_annotation_by_name("empty_table2")

    banana = kitchen_environment_fixture.get_semantic_annotation_by_name("banana")
    apple = kitchen_environment_fixture.get_semantic_annotation_by_name("apple")
    carrot = kitchen_environment_fixture.get_semantic_annotation_by_name("carrot")
    orange = kitchen_environment_fixture.get_semantic_annotation_by_name("orange")
    lettuce = kitchen_environment_fixture.get_semantic_annotation_by_name("lettuce")

    # choosing the correct table
    assert goal_surface_of_object(banana, [table1, table2, table3]) == table1
    assert goal_surface_of_object(carrot, [table1, table2, table3]) == table2
    # choosing the empty table
    assert goal_surface_of_object(lettuce, [table1, table3]) == table3
    assert goal_surface_of_object(table1, [table1, table2, table3]) == table3
    # trying with a new threshold
    assert goal_surface_of_object(orange, [table2, table3], 2) == table2
    # returning None if there is no empty table or no tables
    assert goal_surface_of_object(apple, [table2]) == None
    assert goal_surface_of_object(orange, []) == None
    # trying with 2 empty tables
    assert goal_surface_of_object(apple, [table2, table3, table4]) == table3
    assert goal_surface_of_object(apple, [table2, table4, table3]) == table4

def test_filter_annotations_by_color(kitchen_environment_fixture):
    """
    Tests the filter_annotations_by_color function by verifying the retrieval of semantic
    annotations by their associated colors.

    The function validates that calling filter_annotations_by_color with different color
    parameters returns the expected list of annotations corresponding to that color
    within the given list of objects.
    """
    table1 = kitchen_environment_fixture.get_semantic_annotation_by_name("fruit_table")
    table2 = kitchen_environment_fixture.get_semantic_annotation_by_name("vegetable_table")
    apple = kitchen_environment_fixture.get_semantic_annotation_by_name("apple")
    orange = kitchen_environment_fixture.get_semantic_annotation_by_name("orange")
    carrot = kitchen_environment_fixture.get_semantic_annotation_by_name("carrot")

    assert filter_annotations_by_color(Color.RED(), [apple, orange]).tolist() == [apple]
    assert filter_annotations_by_color(Color.ORANGE(), [apple, orange]).tolist() == [orange]
    assert filter_annotations_by_color(Color.BLUE(), [apple, orange,carrot]).tolist() == []
    assert filter_annotations_by_color(Color.ORANGE(), (list_semantic_annotations_on_surfaces([table1, table2], kitchen_environment_fixture))).tolist() == [orange, carrot]
    assert filter_annotations_by_color(Color.YELLOW(), []).tolist() == []

def test_annotation_class_by_label():
    """
    Tests the annotation_class_by_label function by verifying the retrieval of the semantic class
    """
    assert annotation_class_by_label("salt_aquasale_mill_transparent") == Salt
    assert annotation_class_by_label("milk_jumbo_pack_voll") == Milk
    assert annotation_class_by_label("bowl_collapsable_yellowgrey") == Bowl

    assert annotation_class_by_label("unknown_object") is None


def test_sort_annotations_by_volume(kitchen_environment_fixture):
    """
    Tests the sort_annotations_by_volume function by verifying the order of the returned annotations.
    """
    table1 = kitchen_environment_fixture.get_semantic_annotation_by_name("fruit_table")
    table2 = kitchen_environment_fixture.get_semantic_annotation_by_name("vegetable_table")

    apple = kitchen_environment_fixture.get_semantic_annotation_by_name("apple")
    carrot = kitchen_environment_fixture.get_semantic_annotation_by_name("carrot")
    lettuce = kitchen_environment_fixture.get_semantic_annotation_by_name("lettuce")

    assert sort_annotations_by_volume([]) ==[]
    assert sort_annotations_by_volume([apple, carrot]) == [apple, carrot]
    assert sort_annotations_by_volume([table1, lettuce, apple]) == [table1, lettuce, apple]
    assert sort_annotations_by_volume([table1, lettuce, apple], False) == [apple, lettuce, table1]
    assert sort_annotations_by_volume(list_semantic_annotations_on_surfaces([table2], kitchen_environment_fixture)) == [lettuce, carrot]