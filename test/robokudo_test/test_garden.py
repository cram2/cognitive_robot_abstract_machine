"""
Tests for selecting which visualizer backends a grown tree's GUI uses.
"""

from py_trees.behaviours import Success

# robokudo.pipeline must be imported before robokudo.garden: robokudo.garden pulls in
# robokudo.annotators.outputs, which itself imports robokudo.pipeline, and the reverse
# order trips a circular import between robokudo.pipeline and robokudo.annotators.core.
import robokudo.pipeline
from robokudo.garden import grow_tree
from robokudo.vis.cv_visualizer import CVVisualizer
from robokudo.vis.o3d_visualizer import O3DVisualizer
from robokudo.vis.ros_visualizer import AllAnnotatorROSVisualizer, SharedROSVisualizer
from robokudo.vis.visualizer_manager import VisualizationManager


def test_visualization_manager_defaults_to_every_visualizer():
    manager = VisualizationManager("VisManager")

    assert set(manager.visualizer_types) == {
        CVVisualizer,
        O3DVisualizer,
        SharedROSVisualizer,
        AllAnnotatorROSVisualizer,
    }


def test_visualization_manager_respects_a_visualizer_type_override():
    manager = VisualizationManager("VisManager", visualizer_types=[CVVisualizer])

    assert manager.visualizer_types == [CVVisualizer]


def test_grow_tree_forwards_visualizer_types_to_the_visualization_manager(node):
    """
    ``_no3d`` (main.py) excludes the Open3D viewer this way, so it must actually reach
    the ``VisualizationManager`` the tree ends up with rather than being dropped along
    the way.
    """
    behavior_tree = grow_tree(
        Success("leaf"),
        node=node,
        visualizer_types=[CVVisualizer, SharedROSVisualizer, AllAnnotatorROSVisualizer],
    )

    visualization_manager = next(
        child
        for child in behavior_tree.root.children
        if isinstance(child, VisualizationManager)
    )
    assert visualization_manager.visualizer_types == [
        CVVisualizer,
        SharedROSVisualizer,
        AllAnnotatorROSVisualizer,
    ]
    assert O3DVisualizer not in visualization_manager.visualizer_types
