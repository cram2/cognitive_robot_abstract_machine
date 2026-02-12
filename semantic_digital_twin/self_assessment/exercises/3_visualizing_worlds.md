---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.17.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

(visualizing-worlds-exercise)=
# Visualizing Worlds

This exercise demonstrates a lightweight way to visualize a world inside a notebook using the RayTracer.

You will:
- Load a simple world from URDF
- Create a VizMarkerPublisher and render the scene

## 0. Setup

```{code-cell} ipython3
:tags: [remove-input]
import os
import logging
from pkg_resources import resource_filename
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.spatial_computations.raytracer import RayTracer

logging.disable(logging.CRITICAL)
```

## 1. Visualize 
Your goal:
- Construct a `VizMarkerPublisher` for the loaded world and store it in a variable named `viz`

```{code-cell} ipython3
:tags: [exercise]
root = resource_filename("semantic_digital_twin", "../../")
table_urdf = os.path.join(root, "resources", "urdf", "table.urdf")
world = URDFParser.from_file(table_urdf).parse()

from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
import threading
import rclpy

# TODO: create a viz marker publisher and store it in a variable named `viz`
viz = ...
```

```{code-cell} ipython3
:tags: [example-solution]
root = resource_filename("semantic_digital_twin", "../../")
table_urdf = os.path.join(root, "resources", "urdf", "table.urdf")
world = URDFParser.from_file(table_urdf).parse()

from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
import threading
import rclpy
rclpy.init()

node = rclpy.create_node("semantic_digital_twin")
thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
thread.start()

viz = VizMarkerPublisher(world=world, node=node)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]

assert viz is not ..., "Instantiate a VizMarkerPublisher and assign it to `viz`."
assert isinstance(viz, VizMarkerPublisher), "Make sure you are using the VizMarkerPublisher"
```
