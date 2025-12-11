import os
import unittest

import plotly.graph_objects as go
from random_events.interval import SimpleInterval
from random_events.product_algebra import SimpleEvent

from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.world_description.geometry import BoundingBox
from semantic_digital_twin.world_description.shape_collection import (
    BoundingBoxCollection,
    ShapeCollection,
)
from semantic_digital_twin.world_description.graph_of_convex_sets import (
    GraphOfConvexSets,
    PoseOccupiedError,
)
from semantic_digital_twin.spatial_types import Point3
from semantic_digital_twin.spatial_types.spatial_types import (
    HomogeneousTransformationMatrix,
)
from semantic_digital_twin.datastructures.variables import SpatialVariables
from semantic_digital_twin.world import World
from semantic_digital_twin.world_description.world_entity import Body


class GCSTestCase(unittest.TestCase):
    """
    Testcase to krrood_test the navigation around a unit box.
    """

    gcs: GraphOfConvexSets

    @classmethod
    def setUpClass(cls):
        world = World()
        with world.modify_world():
            world.add_kinematic_structure_entity(Body())
        gcs = GraphOfConvexSets(world)

        obstacle = BoundingBox(0, 0, 0, 1, 1, 1, world.root)

        z_lim = SimpleInterval(0.45, 0.55)
        x_lim = SimpleInterval(-2, 3)
        y_lim = SimpleInterval(-2, 3)
        limiting_event = SimpleEvent(
            {
                SpatialVariables.x.value: x_lim,
                SpatialVariables.y.value: y_lim,
                SpatialVariables.z.value: z_lim,
            }
        )
        obstacles = BoundingBoxCollection.from_event(
            world.root,
            ~obstacle.simple_event.as_composite_set()
            & limiting_event.as_composite_set(),
        )
        [gcs.add_node(bb) for bb in obstacles]
        gcs.calculate_connectivity()
        cls.gcs = gcs

    def test_reachability(self):
        start_point = Point3(-1, -1, 0.5)
        target_point = Point3(2, 2, 0.5)

        path = self.gcs.path_from_to(start_point, target_point)
        self.assertEqual(len(path), 4)

    def test_plot(self):
        free_space_plot = go.Figure(self.gcs.plot_free_space())
        self.assertIsNotNone(free_space_plot)
        occupied_space_plot = go.Figure(self.gcs.plot_occupied_space())
        self.assertIsNotNone(occupied_space_plot)


class GCSFromWorldTestCase(unittest.TestCase):
    """
    Test the application of a connectivity graph to the belief state.
    """

    world: World

    @classmethod
    def setUpClass(cls):
        urdf_dir = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            "..",
            "semantic_digital_twin",
            "resources",
            "urdf",
        )
        apartment = os.path.join(urdf_dir, "table.urdf")
        apartment_parser = URDFParser.from_file(file_path=apartment)
        cls.world = apartment_parser.parse()

    def test_from_world(self):
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
                        reference_frame=self.world.root
                    ),
                )
            ],
            self.world.root,
        )
        gcs = GraphOfConvexSets.free_space_from_world(
            self.world, search_space=search_space
        )
        self.assertIsNotNone(gcs)
        self.assertGreater(len(gcs.graph.nodes()), 0)
        self.assertGreater(len(gcs.graph.edges()), 0)

        start = Point3(-4.5, -0.5, 0.4)
        target = Point3(-2.5, 1.5, 0.9)

        path = gcs.path_from_to(start, target)

        self.assertIsNotNone(path)
        self.assertGreater(len(path), 1)

        with self.assertRaises(PoseOccupiedError):
            start = Point3(-10, -10, -10)
            target = Point3(10, 10, 10)
            gcs.path_from_to(start, target)

    def test_navigation_map_from_world(self):
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
                        reference_frame=self.world.root
                    ),
                )
            ],
            self.world.root,
        )
        gcs = GraphOfConvexSets.navigation_map_from_world(
            self.world, search_space=search_space
        )
        self.assertGreater(len(gcs.graph.nodes()), 0)
        self.assertGreater(len(gcs.graph.edges()), 0)


if __name__ == "__main__":
    unittest.main()
