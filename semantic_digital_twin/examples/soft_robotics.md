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

(soft-robotics)=
# Soft Robotics

This notebook demonstrates how to build and animate continuum robot models using Piecewise Constant Curvature (PCC) and Cosserat Rod Theory.

## 0. Setup and ROS 2 Initialization
We start by initializing a single ROS 2 node. We use a background thread to spin the node.

```{code-cell} ipython3
import numpy as np
import rclpy
import threading
import time

from semantic_digital_twin.world import World
from semantic_digital_twin.adapters.ros.tf_publisher import TFPublisher
from semantic_digital_twin.adapters.ros.visualization.viz_marker import VizMarkerPublisher
from semantic_digital_twin.robots.soft_trunk import SoftTrunk

if not rclpy.ok():
    rclpy.init()
node = rclpy.create_node("soft_robotics_examples")
thread = threading.Thread(target=rclpy.spin, args=(node,), daemon=True)
thread.start()

def update_visualization(world, tf_pub, viz_pub):
    """Push current world state to RViz."""
    world.notify_state_change()
    tf_pub.notify()
    viz_pub.notify()
```

## 1. Piecewise Constant Curvature (PCC)
PCC approximates the robot as a series of circular arcs. 

### 1.1 Uniform Snake
Build a standard uniform snake where every section has the same dimensions.

```{code-cell} ipython3
world_pcc = World()

trunk_pcc, kappas, phis = SoftTrunk.build_pcc(
    world_pcc, 
    num_sections=3, 
    segs_per_section=10, 
    total_length=1.0,
    radius=0.02
)

tf_pcc = TFPublisher(_world=world_pcc, node=node)
viz_pcc = VizMarkerPublisher(_world=world_pcc, node=node)

print("PCC Robot Ready. Set fixed frame to 'pcc/base' in RViz.")
```

```{code-cell} ipython3
# To see the robot model in RViz, we need to push the initial state to the visualization
update_visualization(world_pcc, tf_pcc, viz_pcc)
```

### 1.2 Custom Heterogeneous PCC
We can also build a custom robot where each section has different properties (lengths, radii, resolutions).

```{code-cell} ipython3
world_pcc = World()
lengths = [0.2, 0.3, 0.5]
radii = [0.08, 0.04, 0.02]
res = [5, 10, 15]

trunk_pcc, kappas, phis = SoftTrunk.build_custom_pcc(world_pcc, lengths, radii, res)

tf_pcc = TFPublisher(_world=world_pcc, node=node)
viz_pcc = VizMarkerPublisher(_world=world_pcc, node=node)

print("PCC Robot Ready. Set fixed frame to 'pcc/base' in RViz.")
update_visualization(world_pcc, tf_pcc, viz_pcc)
```

## 1.3 Animating PCC

```{code-cell} ipython3
print("Starting PCC Animation")
try:
    for t in np.linspace(0, 10, 200):
        for s in range(len(kappas)): # Update world state for each section
            # Curvature
            world_pcc.state[kappas[s].id].position = 1.5 * np.sin(t - s * 1.0) # Make it wave by phase-shifting the kappas
            # Plane
            world_pcc.state[phis[s].id].position = t * 0.5 # Make it spiral by rotating the phis
            
        update_visualization(world_pcc, tf_pcc, viz_pcc)
        time.sleep(0.05)
except KeyboardInterrupt:
    print("Stopped.")
```

## 2. Cosserat Rod Theory
Cosserat models allow for torsion (twist) and stretching, which PCC cannot represent.

### 2.1 Uniform Snake
Build a standard uniform snake where every section has the same dimensions.

```{code-cell} ipython3
world_cosserat = World()

trunk_cos, uxs, uys, uzs, vzs = SoftTrunk.build_cosserat(
    world_cosserat, 
    num_sections=3,
    segs_per_section=10,
    total_length=1.0,
    radius=0.02
)

tf_cos = TFPublisher(_world=world_cosserat, node=node)
viz_cos = VizMarkerPublisher(_world=world_cosserat, node=node)

print("Cosserat Robot Ready. Set fixed frame to 'cosserat/base' in RViz.")
```

```{code-cell} ipython3
# To see the robot model in RViz, we need to push the initial state to the visualization
update_visualization(world_cosserat, tf_cos, viz_cos)
```

### 2.2 Custom Heterogeneous Cosserat

We can also build a custom robot where each section has different properties (lengths, radii, resolutions).

```{code-cell} ipython3
world_cosserat = World()

# Custom Cosserat with different lengths, radii, and resolutions for each section
c_lengths = [0.4, 0.4, 0.4]
c_radii = [0.06, 0.04, 0.02]
c_res = [10, 10, 10]

trunk_cos, uxs, uys, uzs, vzs = SoftTrunk.build_custom_cosserat(world_cosserat, c_lengths, c_radii, c_res)

tf_cos = TFPublisher(_world=world_cosserat, node=node)
viz_cos = VizMarkerPublisher(_world=world_cosserat, node=node)

print("Cosserat Robot Ready. Set fixed frame to 'cosserat/base' in RViz.")
update_visualization(world_cosserat, tf_cos, viz_cos)
```

### 2.3 Animating Cosserat

```{code-cell} ipython3
try:
    print("Starting Cosserat Animation")
    for t in np.linspace(0, 10, 200):
        for s in range(len(uxs)): # Update world state for each section
            world_cosserat.state[uxs[s].id].position = 2.0 * np.sin(t) # Curvature (Bending X)
            world_cosserat.state[uys[s].id].position = 2.0 * np.cos(t) # Curvature (Bending Y)
            world_cosserat.state[uzs[s].id].position = 1.5 * np.sin(t * 0.5) # Torsion (Twisting)      
            world_cosserat.state[vzs[s].id].position = 1.0 + 0.5 * np.sin(t) # Strecthing
        update_visualization(world_cosserat, tf_cos, viz_cos)
        time.sleep(0.03)
except KeyboardInterrupt:
    print("Stopped.")
```

## 3. Cleanup

```{code-cell} ipython3
node.destroy_node()
rclpy.shutdown()
print("ROS Node shut down.")
```
