import numpy as np
import plotly.graph_objects as go
import pytest
from dataclasses import dataclass

from numpy import allclose

from random_events.interval import SimpleInterval
from random_events.product_algebra import SimpleEvent
from semantic_digital_twin.adapters.ros.visualization.viz_marker import (
    VizMarkerPublisher,
)
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.exceptions import PointOccupiedError
from semantic_digital_twin.spatial_types import Point3, Pose
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.connections import FixedConnection
from semantic_digital_twin.world_description.geometry import BoundingBox, Box, Scale
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
    create_reference_frame_with_only_yaw_from_body,
)
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
)
from semantic_digital_twin.world_description.world_entity import Body


@dataclass
class GraphOfConvexSetsFixture:
    """
    Data class for Graph of Convex Sets test fixture.
    """

    world: World
    graph_of_convex_sets: GraphOfConvexSets


@pytest.fixture
def graph_of_convex_sets_unit_box() -> GraphOfConvexSetsFixture:
    """
    Create a GraphOfConvexSets for navigation around a unit box.
    """
    world = World()
    with world.modify_world():
        world.add_kinematic_structure_entity(Body())

    graph_of_convex_sets = GraphOfConvexSets(world)

    obstacle = BoundingBox(0, 0, 0, 1, 1, 1, world.root.global_pose)

    z_lim = SimpleInterval.from_data(0.45, 0.55)
    x_lim = SimpleInterval.from_data(-2, 3)
    y_lim = SimpleInterval.from_data(-2, 3)
    limiting_event = SimpleEvent.from_data(
        {
            SpatialVariables.x.value: x_lim,
            SpatialVariables.y.value: y_lim,
            SpatialVariables.z.value: z_lim,
        }
    )
    obstacles = BoundingBoxCollection.from_event(
        world.root,
        ~obstacle.simple_event.as_composite_set() & limiting_event.as_composite_set(),
    )
    for bounding_box in obstacles:
        graph_of_convex_sets.add_node(bounding_box)

    graph_of_convex_sets.calculate_connectivity()
    return GraphOfConvexSetsFixture(world, graph_of_convex_sets)


def test_reachability(graph_of_convex_sets_unit_box: GraphOfConvexSetsFixture):
    """
    Verify if a path can be found around the unit box.
    """
    start_point = Point3(
        -1, -1, 0.5, reference_frame=graph_of_convex_sets_unit_box.world.root
    )
    target_point = Point3(
        2, 2, 0.5, reference_frame=graph_of_convex_sets_unit_box.world.root
    )

    path = graph_of_convex_sets_unit_box.graph_of_convex_sets.path_from_to(
        start_point, target_point
    )
    assert len(path) == 4


def test_plot(graph_of_convex_sets_unit_box: GraphOfConvexSetsFixture):
    """
    Verify if the free and occupied space can be plotted.
    """
    free_space_plot = go.Figure(
        graph_of_convex_sets_unit_box.graph_of_convex_sets.plot_free_space()
    )
    assert free_space_plot is not None
    occupied_space_plot = go.Figure(
        graph_of_convex_sets_unit_box.graph_of_convex_sets.plot_occupied_space()
    )
    assert occupied_space_plot is not None


def test_from_world(table_world: World):
    """
    Verify the generation of a connectivity graph from a world.
    """
    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=-5,
                max_x=-2,
                min_y=-1,
                max_y=2,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=table_world.root
                ),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = GraphOfConvexSets.free_space_from_world(
        table_world, search_space=search_space
    )
    assert graph_of_convex_sets is not None
    assert len(graph_of_convex_sets.graph.nodes()) > 0
    assert len(graph_of_convex_sets.graph.edges()) > 0

    start = Point3(-4.5, -0.5, 0.4, reference_frame=table_world.root)
    target = Point3(-2.5, 1.5, 0.9, reference_frame=table_world.root)

    path = graph_of_convex_sets.path_from_to(start, target)

    assert path is not None
    assert len(path) > 1

    with pytest.raises(PointOccupiedError):
        start_occupied = Point3(-10, -10, -10, reference_frame=table_world.root)
        target_occupied = Point3(10, 10, 10, reference_frame=table_world.root)
        graph_of_convex_sets.path_from_to(start_occupied, target_occupied)


def test_navigation_map_from_world(table_world: World):
    """
    Verify the generation of a navigation map from a world.
    """
    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=-5,
                max_x=-2,
                min_y=-1,
                max_y=2,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=table_world.root
                ),
            )
        ],
        table_world.root,
    )
    graph_of_convex_sets = GraphOfConvexSets.navigation_map_from_world(
        table_world, search_space=search_space
    )
    assert len(graph_of_convex_sets.graph.nodes()) > 0
    assert len(graph_of_convex_sets.graph.edges()) > 0


def test_from_world_with_rotated_box():
    """
    Verify if a path can be found in a world with two boxes, where one is rotated and a
    gcs is calculated for the rotated box.
    """
    world = World()
    with world.modify_world():
        root_body = Body(name=PrefixedName("map"))
        world.add_kinematic_structure_entity(root_body)

        # Box 1: at origin
        axis_aligned_box_body = Body(name=PrefixedName("box_one"))
        world.add_connection(
            FixedConnection.create_with_dofs(world, root_body, axis_aligned_box_body)
        )
        axis_aligned_box = Box(scale=Scale(1, 1, 1))
        axis_aligned_box_body.collision.append(axis_aligned_box)

        # Box 2: at (2, 0, 0) rotated 45 deg around Z
        rotated_box_body = Body(name=PrefixedName("box_two"))
        rotated_box_body_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            2, 0, 0, 0, np.pi / 4, np.pi / 4, reference_frame=root_body
        )
        world.add_connection(
            FixedConnection.create_with_dofs(
                world,
                root_body,
                rotated_box_body,
                parent_T_connection_expression=rotated_box_body_pose,
            )
        )
        rotated_box = Box(scale=Scale(1, 1, 1))
        rotated_box_body.collision.append(rotated_box)

    vertical_stabilized_base = create_reference_frame_with_only_yaw_from_body(
        rotated_box_body
    )

    search_space = BoundingBoxCollection(
        [
            BoundingBox(
                min_x=-5,
                max_x=5,
                min_y=-5,
                max_y=5,
                min_z=0,
                max_z=2,
                origin=HomogeneousTransformationMatrix(
                    reference_frame=vertical_stabilized_base
                ),
            )
        ],
        reference_frame=vertical_stabilized_base,
    )

    graph_of_convex_sets = GraphOfConvexSets.free_space_from_world(
        world, search_space=search_space
    )

    assert len(graph_of_convex_sets.graph.nodes()) > 0

    start = Point3(-1, 0, 0.5, reference_frame=world.root)
    target = Point3(4, 0, 0.5, reference_frame=world.root)

    path = graph_of_convex_sets.path_from_to(start, target)
    assert path is not None

    for bounding_box in graph_of_convex_sets.graph.nodes():
        bounding_box_T_world: Pose = world.transform(
            bounding_box.as_shape().origin, world.root
        ).to_pose()

        assert bounding_box_T_world.roll == 0
        assert bounding_box_T_world.pitch == 0
        assert allclose(bounding_box_T_world.yaw, rotated_box_body_pose.to_pose().yaw)
