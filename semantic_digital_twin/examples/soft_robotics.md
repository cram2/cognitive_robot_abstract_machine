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

This notebook demonstrates how to build and animate continuum robot models using Piecewise Constant Curvature (PCC) and Cosserat Rod Theory. We also perform Inverse Kinematics to reach targets for both models. 

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

trunk_pcc, kappas, phis = SoftTrunk.build_piecewise_constant_curvature(
    world_pcc, 
    num_sections=3, 
    segments_per_section=10, 
    total_length=1.0,
    radius=0.02
)

tf_pcc = TFPublisher(_world=world_pcc, node=node)
viz_pcc = VizMarkerPublisher(_world=world_pcc, node=node)

print("PCC Robot Ready. Set fixed frame to 'pcc/base' in RViz.")

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

trunk_pcc, kappas, phis = SoftTrunk.build_custom_piecewise_constant_curvature(world_pcc, lengths, radii, res)

tf_pcc = TFPublisher(_world=world_pcc, node=node)
viz_pcc = VizMarkerPublisher(_world=world_pcc, node=node)

print("PCC Robot Ready. Set fixed frame to 'pcc/base' in RViz.")
update_visualization(world_pcc, tf_pcc, viz_pcc)
```

### 1.3 Animating PCC

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
        time.sleep(0.03)
except KeyboardInterrupt:
    print("Stopped.")
```

### 1.4 Inverse Kinematics PCC


Create a visualization for the target

```{code-cell} ipython3
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Sphere, Color
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

# Create a transparent red sphere for the target
goal_visual = Sphere(radius=0.03, color=Color(1.0, 0.0, 0.0, 0.5))
goal_body = Body(
    name=PrefixedName(name="goal_marker", prefix="ik"), 
    visual=ShapeCollection([goal_visual])
)

# Add it to the world and connect it to the root
with world_pcc.modify_world():
    world_pcc.add_body(goal_body)
    goal_connection = Connection6DoF.create_with_dofs(
        parent=world_pcc.root, 
        child=goal_body, 
        world=world_pcc
    )
    world_pcc.add_connection(goal_connection)

print("Goal marker added to world.")
```

Define a target pose and solve the Inverse Kinematics.

```{code-cell} ipython3
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
import numpy as np

# Initialize the solver
ik_solver = InverseKinematicsSolver(world=world_pcc)

# Give the robot a tiny nudge so it's not perfectly straight
# This helps the numerical solver find a direction to start bending
for k in kappas:
    world_pcc.state[k.id].position = 0.01 
world_pcc.notify_state_change()

# Define a random target pose within a boundary relative to the world root
# Try testing with different bounds!
x = np.random.uniform(-0.5, 0.5)
y = np.random.uniform(-0.5, 0.5)
z = np.random.uniform(0.5, 0.8)

target_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=x, y=y, z=z, reference_frame=world_pcc.root
)

# Update the marker in RViz
goal_connection.origin = target_pose
update_visualization(world_pcc, tf_pcc, viz_pcc)

# Solve the IK
try:
    print("Solving IK")
    ik_results = ik_solver.solve(
        root=world_pcc.root,
        tip=trunk_pcc.manipulator_chains[0].tip,
        target=target_pose,
        max_iterations=500,
        dt=0.1
    )
    
    # Apply results to robot
    for dof, position in ik_results.items():
        world_pcc.state[dof.id].position = position
    
    # Refresh RViz
    update_visualization(world_pcc, tf_pcc, viz_pcc)
    print("Robot reached the target marker!")

except Exception as e:
    print(f"Failed: {e}")
```

## 2. Cosserat Rod Theory
Cosserat models allow for torsion (twist) and stretching, which PCC cannot represent.

### 2.1 Uniform Snake
Build a standard uniform snake where every section has the same dimensions.

```{code-cell} ipython3
world_cosserat = World()

trunk_cos, all_bx, all_by, all_tor, all_ext = SoftTrunk.build_cosserat(
    world_cosserat, 
    num_sections=3,
    segments_per_section=10,
    total_length=1.0,
    radius=0.02
)

tf_cos = TFPublisher(_world=world_cosserat, node=node)
viz_cos = VizMarkerPublisher(_world=world_cosserat, node=node)

print("Cosserat Robot Ready. Set fixed frame to 'cosserat/base' in RViz.")

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

trunk_cos, all_bx, all_by, all_tor, all_ext = SoftTrunk.build_custom_cosserat(world_cosserat, c_lengths, c_radii, c_res)

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
        for s in range(len(all_bx)): # Update world state for each section
            world_cosserat.state[all_bx[s].id].position = 2.0 * np.sin(t) # Curvature (Bending X)
            world_cosserat.state[all_by[s].id].position = 2.0 * np.cos(t) # Curvature (Bending Y)
            world_cosserat.state[all_tor[s].id].position = 1.5 * np.sin(t * 0.5) # Torsion (Twisting)      
            world_cosserat.state[all_ext[s].id].position = 1.0 + 0.5 * np.sin(t) # Strecthing
        update_visualization(world_cosserat, tf_cos, viz_cos)
        time.sleep(0.03)
except KeyboardInterrupt:
    print("Stopped.")
```

### 2.4 Inverse Kinematics Cosserat


Create a visualization for the target

```{code-cell} ipython3
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Sphere, Color
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName

# Create a transparent green sphere for the target
cos_goal_visual = Sphere(radius=0.03, color=Color(0.0, 1.0, 0.0, 0.5)) 
cos_goal_body = Body(
    name=PrefixedName(name="cos_goal_marker", prefix="ik"), 
    visual=ShapeCollection([cos_goal_visual])
)

# Add it to world_cos and connect it to root
with world_cosserat.modify_world():
    world_cosserat.add_body(cos_goal_body)
    cos_goal_connection = Connection6DoF.create_with_dofs(
        parent=world_cosserat.root, 
        child=cos_goal_body, 
        world=world_cosserat
    )
    world_cosserat.add_connection(cos_goal_connection)

print("Goal marker added to Cosserat world.")
```

Define a target pose and solve the Inverse Kinematics.

```{code-cell} ipython3
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver

# Initialize solver for the Cosserat world
ik_solver_cos = InverseKinematicsSolver(world=world_cosserat)


# Define a random target pose within a boundary relative to the world root
# Try testing with different bounds!
x = np.random.uniform(-1.0, 1.0)
y = np.random.uniform(-1.0, 1.0)
z = np.random.uniform(0.5, 1.2)

# Define a target
target_pose_cos = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=x, y=y, z=z, reference_frame=world_cosserat.root
)

# Update marker position
cos_goal_connection.origin = target_pose_cos

# Solve IK
try:
    print("Solving Cosserat IK")
    ik_results_cos = ik_solver_cos.solve(
        root=world_cosserat.root,
        tip=trunk_cos.manipulator_chains[0].tip,
        target=target_pose_cos,
        max_iterations=500,
        dt=0.1,
        translation_velocity=1.0
    )
    
    # Apply and Notify
    for dof, position in ik_results_cos.items():
        world_cosserat.state[dof.id].position = position
    
    update_visualization(world_cosserat, tf_cos, viz_cos)
    print("Cosserat model successfully reached the target!")

except Exception as e:
    print(f"Cosserat IK Failed: {e}")
```

## 4. Workspace Reachability Analysis

```{code-cell} ipython3
import numpy as np
import time
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver

ik_solver = InverseKinematicsSolver(world=world_pcc)
ik_solver_cos = InverseKinematicsSolver(world=world_cosserat)

def run_reachability_study(world, robot, solver, num_trials=100):

    setup = {
        "x_range": (-0.8, 1.2),
        "y_range": (-0.8, 1.2),
        "z_range": (0.05, 2.0),
        "yaw_range": (-np.pi, np.pi),
    }

    print(f"\nEXPERIMENT SETUP: {robot.name.prefix.upper()}")
    print(f"Target Volume: X,Y in {setup['x_range']}, Z in {setup['z_range']}")
    print(f"Total Trials: {num_trials}")

    success_count = 0
    errors = []
    times = []
    
    # Get the primary manipulator chain
    chain = robot.manipulator_chains[0]
    tip_body = chain.tip

    # Collect all active DOFs to reset the robot state between trials
    active_dofs = []
    for conn in world.compute_chain_of_connections(chain.root, chain.tip):
        active_dofs.extend(conn.active_dofs)

    print(f"Starting study for robot: {robot.name.name} (Type: {robot.name.prefix})")

    for i in range(num_trials):
        # Generate a random target pose within the defined volume
        x = np.random.uniform(*setup['x_range'])
        y = np.random.uniform(*setup['y_range'])
        z = np.random.uniform(*setup['z_range'])
        yaw = np.random.uniform(*setup['yaw_range'])
        
        target_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x, y=y, z=z, reference_frame=world.root
        )

        # Reset robot to nudge for gradient descent start
        for dof in active_dofs:
            # Extension vz reset to 1.0, others to a small 0.01
            if "extension" in str(dof.name):
                world.state[dof.id].position = 1.0
            else:
                world.state[dof.id].position = 0.01
        world.notify_state_change()

        # Solve IK
        start_time = time.time()
        try:
            ik_results = solver.solve(
                root=world.root, 
                tip=tip_body, 
                target=target_pose,
                max_iterations=300, 
                dt=0.1, 
                translation_velocity=1.0, 
                rotation_velocity=1.0
            )
            
            # Apply results to world state
            for dof, pos in ik_results.items():
                world.state[dof.id].position = pos
            world.notify_state_change()
            
            # Check final positional error
            current_fk = world.compute_forward_kinematics_np(world.root, tip_body)
            current_xyz = current_fk[:3, 3]
            target_xyz = np.array([x, y, z])
            dist_error = np.linalg.norm(current_xyz - target_xyz)
            
            # Count success if within 3cm
            if dist_error < 0.03:
                success_count += 1
            
            errors.append(dist_error)
            
        except Exception:
            # If solver fails to converge or target is unreachable
            current_fk = world.compute_forward_kinematics_np(world.root, tip_body)
            dist_error = np.linalg.norm(current_fk[:3, 3] - np.array([x, y, z]))
            errors.append(dist_error)
            
        times.append(time.time() - start_time)
        
        if (i+1) % 25 == 0:
            print(f"  Trial {i+1}/{num_trials} complete...")

    return {
        "success_rate": (success_count / num_trials) * 100,
        "mean_error": np.mean(errors),
        "mean_time": np.mean(times)
    }

results_pcc = run_reachability_study(world_pcc, trunk_pcc, ik_solver, num_trials=100)
results_cos = run_reachability_study(world_cosserat, trunk_cos, ik_solver_cos, num_trials=100)

print("\nRESULTS")
print(30 * "-")
print(f"PCC Success Rate:       {results_pcc['success_rate']:.1f}%")
print(f"PCC Mean Error:         {results_pcc['mean_error']:.4f}m")
print(f"PCC Avg Solve Time:     {results_pcc['mean_time']:.4f}s")
print(30 * "-")
print(f"Cosserat Success Rate:  {results_cos['success_rate']:.1f}%")
print(f"Cosserat Mean Error:    {results_cos['mean_error']:.4f}m")
print(f"Cosserat Avg Solve Time:{results_cos['mean_time']:.4f}s")
```

## 5. Performance Benchmark

```{code-cell} ipython3
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver

ik_solver = InverseKinematicsSolver(world=world_pcc)
ik_solver_cos = InverseKinematicsSolver(world=world_cosserat)

def run_performance_benchmark(world, robot, solver, num_trials=50):
    times = []
    
    # Benchmark Zone: Easily reachable for both
    setup = {
        "x_range": (-0.2, 0.2),
        "y_range": (-0.2, 0.2),
        "z_range": (0.7, 0.9),
        "rest_length": 1.0
    }
    
    chain = robot.manipulator_chains[0]
    
    print(f"Benchmarking {robot.name.prefix.upper()}...")

    for i in range(num_trials):
        x = np.random.uniform(*setup['x_range'])
        y = np.random.uniform(*setup['y_range'])
        z = np.random.uniform(*setup['z_range'])
        target_pose = HomogeneousTransformationMatrix.from_xyz_rpy(x=x, y=y, z=z, reference_frame=world.root)

        # Solve and measure the solver time only for successful runs
        start_time = time.time()
        try:
            solver.solve(root=world.root, tip=chain.tip, target=target_pose, max_iterations=200, dt=0.1)
            times.append(time.time() - start_time)
        except:
            continue

    return {
        "avg_time_ms": np.mean(times) * 1000, 
        "std_dev_ms": np.std(times) * 1000,
        "sample_size": len(times)
    }

# Run benchmark
perf_pcc = run_performance_benchmark(world_pcc, trunk_pcc, ik_solver)
perf_cos = run_performance_benchmark(world_cosserat, trunk_cos, ik_solver_cos)

print("\nPERFORMANCE BENCHMARK (SUCCESSFUL RUNS ONLY)")
print(f"PCC Avg Time:      {perf_pcc['avg_time_ms']:.2f} ms (+/- {perf_pcc['std_dev_ms']:.2f})")
print(f"Cosserat Avg Time: {perf_cos['avg_time_ms']:.2f} ms (+/- {perf_cos['std_dev_ms']:.2f})")
```

## Cleanup

```{code-cell} ipython3
node.destroy_node()
rclpy.shutdown()
print("ROS Node shut down.")
```
