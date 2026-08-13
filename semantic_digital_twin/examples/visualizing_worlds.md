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

(visualizing-worlds)=
# Visualizing Worlds

This tutorial explains you how to visualize a world.
There are three recommended ways of doing it.
The Rerun viewer needs no ROS and is the easiest way to just look at a world, RVIZ2 is the light weight choice inside a ROS 2 environment, and simulation with multiverse is the heavy weight interactive one.
Let's load a world first to get started.

```{code-cell} ipython3
import logging
import os

from importlib.resources import files
from pathlib import Path

from semantic_digital_twin.adapters.urdf import URDFParser 

logging.disable(logging.CRITICAL)
apartment = os.path.join(Path(files("semantic_digital_twin")).parent.parent, "resources", "urdf", "apartment.urdf")
world = URDFParser.from_file(apartment).parse()

```

## Rerun

The [Rerun](https://rerun.io) viewer renders the world's real meshes without any ROS setup.
Creating a `RerunAdapter` registers callbacks on the world, so the viewer follows every model and state change on its own.
With `RerunMode.SPAWN` (the default) a local viewer window opens; here we use `RerunMode.NONE`, which only builds the recording, because these guides run without a display.

```{code-cell} ipython3
from semantic_digital_twin.adapters.rerun import RerunAdapter, RerunMode

adapter = RerunAdapter(_world=world, mode=RerunMode.NONE)
```

Setting `state_history=True` keeps a scrubbable timeline of every state change instead of only the latest one; `state_log_stride` thins high-frequency updates.
`RerunMode.SAVE` together with `target="my_world.rrd"` records to a file you can open later with `rerun my_world.rrd` or share.
When you are done, detach the adapter from the world.

```{code-cell} ipython3
adapter.stop()
```

## RVIZ2

For the RVIZ2 way, ROS2 and the TFPublisher is needed. A caveat of this approach is that you have to manage the lifecycle of a ROS2 node yourself.
We recommend to put the spinning into sperate threads and just shutdown the thread when exiting the system.

```{code-cell} ipython3
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

When you want to stop visualizing, you have to stop the visualizer and afterwards clean up ROS2.

```{code-cell} ipython3
node.destroy_node()
rclpy.shutdown()
```

## Multiverse

The world can also be visualized directly through a running simulation.
Although this approach is computationally heavier, it provides the important advantage of enabling interaction with the environment. 
In addition to visualization, the physics engine can be used to simulate dynamics, contacts, and collisions.
Further details are provided in the [physics simulators](physics-simulators) section.

If you have followed the guide until here, you have probably noticed that we have used the RayTracer to visualize the world 
a few times. This is a convenient way of visualizing a world inside a notebook, like in these guides, but it is not recommended 
for normal usage.
