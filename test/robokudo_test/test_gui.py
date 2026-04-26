import py_trees
import pytest
import rclpy

import robokudo.annotators.testing
from robokudo.pipeline import Pipeline
from robokudo.vis.cv_visualizer import CVVisualizer

# from robokudo.vis.o3d_visualizer import O3DVisualizer # Leaving this out for now, since it introduces a SIGINT under test?
from robokudo.vis.ros_visualizer import AllAnnotatorROSVisualizer
from robokudo.vis.visualizer import Visualizer


@pytest.fixture()
def function_setup(node):
    """
    :rtype: WorldObject
    """
    Visualizer.clear_visualizer_instances()
    FakeVisualizer.test_number = (
        0  # Reset test number to count calls in static post tick test
    )
    yield


class FakeVisualizer(Visualizer):
    test_number = 0

    @staticmethod
    def static_post_tick():
        FakeVisualizer.test_number += 1
        return FakeVisualizer.test_number


class TestVisualizerObject(object):
    def test_visualizer_instantiation_unique_objects(self, function_setup):
        root = Pipeline("TestPipeline")
        root.add_child(robokudo.annotators.testing.SlowAnnotator())
        vis1 = Visualizer.new_visualizer_instance(pipeline=root)
        vis2 = Visualizer.new_visualizer_instance(pipeline=root)

        assert id(vis1) != id(vis2)

    def test_visualizer_instantiation_bookkeeping(self, function_setup):
        root = Pipeline("TestPipeline")
        root.add_child(robokudo.annotators.testing.SlowAnnotator())
        Visualizer.new_visualizer_instance(pipeline=root)
        Visualizer.new_visualizer_instance(pipeline=root)

        assert len(Visualizer.instances) == 2

    def test_visualizer_instantiation_subtypes(self, function_setup):
        root = Pipeline("TestPipeline")
        root.add_child(robokudo.annotators.testing.SlowAnnotator())
        shared_state = Visualizer.SharedState()
        CVVisualizer.new_visualizer_instance(
            pipeline=root, shared_visualizer_state=shared_state
        )
        AllAnnotatorROSVisualizer.new_visualizer_instance(pipeline=root)

        assert len(Visualizer.instances) == 2

    def test_visualizer_instantiation_get_unique_visualizer_types(self, function_setup):
        root = Pipeline("TestPipeline")
        shared_state = Visualizer.SharedState()
        root.add_child(robokudo.annotators.testing.SlowAnnotator())
        CVVisualizer.new_visualizer_instance(
            pipeline=root, shared_visualizer_state=shared_state
        )
        CVVisualizer.new_visualizer_instance(
            pipeline=root, shared_visualizer_state=shared_state
        )
        AllAnnotatorROSVisualizer.new_visualizer_instance(pipeline=root)
        AllAnnotatorROSVisualizer.new_visualizer_instance(pipeline=root)
        FakeVisualizer.new_visualizer_instance(pipeline=root)
        FakeVisualizer.new_visualizer_instance(pipeline=root)

        assert len(Visualizer.instances) == 6
        types_of_visualizers = Visualizer.get_unique_types_of_visualizer_instances()
        assert len(types_of_visualizers) == 3
        assert CVVisualizer in types_of_visualizers
        assert AllAnnotatorROSVisualizer in types_of_visualizers
        assert FakeVisualizer in types_of_visualizers

    def test_visualizer_instantiation_call_static_post_tick(self, function_setup):
        root = Pipeline("TestPipeline")
        root.add_child(robokudo.annotators.testing.SlowAnnotator())
        AllAnnotatorROSVisualizer.new_visualizer_instance(pipeline=root)
        AllAnnotatorROSVisualizer.new_visualizer_instance(pipeline=root)
        FakeVisualizer.new_visualizer_instance(pipeline=root)
        FakeVisualizer.new_visualizer_instance(pipeline=root)

        assert FakeVisualizer.static_post_tick() == 1
