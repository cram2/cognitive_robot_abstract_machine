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
from importlib.resources import files
from pathlib import Path
from semantic_digital_twin.adapters.urdf import URDFParser
from semantic_digital_twin.spatial_computations.raytracer import RayTracer
from semantic_digital_twin.exceptions import ExerciseVerificationFailed

logging.disable(logging.CRITICAL)
```

## 1. Visualize 
Your goal:
- Construct a `TFPublisher` for the loaded world and store it in a variable named `tf_publisher`
- Construct a `VizMarkerPublisher` for the loaded world and store it in a variable named `viz`

```{code-cell} ipython3
:tags: [exercise]
root = Path(files("semantic_digital_twin")).parent.parent
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
root = Path(files("semantic_digital_twin")).parent.parent
table_urdf = os.path.join(root, "resources", "urdf", "table.urdf")
world = URDFParser.from_file(table_urdf).parse()

from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
import threading
import rclpy
rclpy.init()

node = rclpy.create_node("semantic_digital_twin")
thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
thread.start()

tf_publisher = TFPublisher(_world=world, node=node)
viz = VizMarkerPublisher(_world=world, node=node)
```

```{code-cell} ipython3
:tags: [verify-solution, remove-input]

if viz is ...: raise ExerciseVerificationFailed("Instantiate a VizMarkerPublisher and assign it to `viz`.")
if not isinstance(tf_publisher, TFPublisher): raise ExerciseVerificationFailed("Make sure you are using the TFPublisher")
if not isinstance(viz, VizMarkerPublisher): raise ExerciseVerificationFailed("Make sure you are using the VizMarkerPublisher")
```
