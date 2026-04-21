import os
import sys

import pytest

from krrood.rustworkx_utils.rxnode import RWXNode
from krrood.rustworkx_utils.utils import ColorLegend
import rustworkx as rx


@pytest.fixture
def missing_viz_libraries():
    modules_to_patch = ["matplotlib", "matplotlib.pyplot", "numpy"]
    original_modules = {mod: sys.modules.get(mod) for mod in modules_to_patch}
    original_gv = sys.modules.get("krrood.rustworkx_utils.graph_visualizer")

    for mod in modules_to_patch:
        sys.modules[mod] = None
    sys.modules.pop("krrood.rustworkx_utils.graph_visualizer", None)

    yield

    for mod, original in original_modules.items():
        if original is None:
            sys.modules.pop(mod, None)
        else:
            sys.modules[mod] = original
    if original_gv is not None:
        sys.modules["krrood.rustworkx_utils.graph_visualizer"] = original_gv


def test_raising_module_not_found_error_on_missing_libraries_1(missing_viz_libraries):
    import krrood.rustworkx_utils.graph_visualizer as gv_reloaded

    with pytest.raises(ModuleNotFoundError):
        gv_reloaded.GraphVisualizer(RWXNode("Root", rx.PyDAG())).render()


def test_raising_module_not_found_error_on_missing_libraries_2():
    # monkey patch matplotlib, and numpy to None for rustworkx_utils.graph_visualizer
    import krrood.rustworkx_utils.graph_visualizer as gv
    original_mpl = gv.mpl
    original_np = gv.np
    gv.mpl = None
    gv.np = None
    with pytest.raises(ModuleNotFoundError):
        gv.GraphVisualizer(RWXNode("Root", rx.PyDAG())).render()
    # restore original values
    gv.mpl = original_mpl
    gv.np = original_np


def test_create_and_visualize_graph(tmp_path):
    # Build a small DAG using RWXNode
    graph = rx.PyDAG()
    root = RWXNode("Root", graph, enclosed=True)
    a = RWXNode("A", graph, color=ColorLegend(name="A", color="red"))
    b = RWXNode("B", graph, color=ColorLegend(name="B", color="green"))
    c = RWXNode("C", graph, color=ColorLegend(name="C", color="blue"))

    # Establish primary parent relationships
    a.parent = root
    b.parent = root
    c.parent = a

    # Add an additional non-primary edge to test multi-parent
    c.add_parent(b)

    # Visualize (should save a pdf called pdf_graph.pdf in CWD)
    # Change CWD to tmp to avoid cluttering repo root
    cwd = os.getcwd()
    try:
        os.chdir(tmp_path)
        fig, ax = root.visualize(
            figsize=(10, 10),
            node_size=1500,
            font_size=15,
            spacing_x=2.0,
            spacing_y=2.0,
            layout="tidy",
            edge_style="orthogonal",
        )
        assert fig is not None and ax is not None
        out_file = tmp_path / "pdf_graph.pdf"
        assert out_file.exists(), "Visualization did not produce expected output file"
        # Basic sanity: file not empty
        assert out_file.stat().st_size > 1000
    finally:
        os.chdir(cwd)
