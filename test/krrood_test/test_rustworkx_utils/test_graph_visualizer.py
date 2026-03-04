import os

from krrood.rustworkx_utils.rxnode import RWXNode
from krrood.rustworkx_utils.utils import ColorLegend
import rustworkx as rx


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
