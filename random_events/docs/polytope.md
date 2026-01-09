---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---


# Polytopes

The tutorial walks through the conversion from polytopes to events from the product algebra.

First, let's define a polytope in 2D.
We start by generating some random points and rotating them. Then we create a polytope over the set of points.

```{code-cell} ipython3
import numpy as np
from scipy.spatial import ConvexHull
from random_events.polytope import Polytope
import plotly
plotly.offline.init_notebook_mode()
import plotly.graph_objects as go

def rotation_matrix(theta: float):
    return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])

np.random.seed(69)
points = np.random.uniform(-1, 1, (100, 2))
points = points @ rotation_matrix(0.3)

polytope = Polytope.from_2d_points(points)
```

Let's have a look at the generated data and polytope.

```{code-cell} ipython3
fig = go.Figure()
fig.add_trace(go.Scatter(x=points[:, 0], y=points[:, 1], mode='markers', name='Points'))
convex_hull = ConvexHull(points)
hull_points = np.vstack([points[convex_hull.vertices], points[convex_hull.vertices[0]]])
fig.add_trace(go.Scatter(x=hull_points[:, 0], y=hull_points[:, 1], mode='lines', name='Convex Hull'))
fig.show()
```

Next, let's create the inner box approximation of the polytope.

```{code-cell} ipython3
inner_event = polytope.inner_box_approximation()
fig.add_traces(inner_event.plot("#2ca02c"))
fig.show()
```

Let's create the outer box approximation of the polytope.

```{code-cell} ipython3
outer_event = polytope.outer_box_approximation()
outer_event = outer_event - inner_event
fig.add_traces(outer_event.plot("#d62728"))
fig.show()
```
