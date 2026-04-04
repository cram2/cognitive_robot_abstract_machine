---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Visualize rustworkx Graphs

Below is an example of how to visualize a rooted directed acyclic rustworkx graph using the `visualize` method.

```{code-cell} ipython3
from krrood.rustworkx_utils import RWXNode, ColorLegend
import rustworkx as rx

# Build a small DAG using RWXNode
graph = rx.PyDAG()
root = RWXNode("Root", graph=graph, enclosed=True)
a = RWXNode("A", graph=graph, color=ColorLegend(name="A", color="red"))
b = RWXNode("B", graph=graph, color=ColorLegend(name="B", color="green"))
c = RWXNode("C", graph=graph, color=ColorLegend(name="C", color="blue"))

# Establish primary parent relationships
a.parent = root
b.parent = root
c.parent = a

# Add an additional non-primary edge for multi-parent showcase
c.add_parent(b)

# Visualize (should save a pdf called pdf_graph.pdf in CWD)
fig, ax = root.visualize(figsize=(10, 10), node_size=1500, font_size=15,
                          spacing_x=2.0, spacing_y=2.0,
                          layout='tidy', edge_style='orthogonal')
```