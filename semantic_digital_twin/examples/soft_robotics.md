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
from semantic_digital_twin.robots.soft_trunk import SoftTrunk, SoftTrunkSection

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

### 1.1 Build PCC Robot

```{code-cell} ipython3
world_pcc = World()

# Uniform sections for the trunk robot
sections = [SoftTrunkSection(0.3, 0.02, 10)] * 3

# Custom sections for the trunk robot (uncomment to use)
#sections = [
#    SoftTrunkSection(length=0.2, radius=0.08, resolution=5),
#    SoftTrunkSection(length=0.3, radius=0.04, resolution=10),
#    SoftTrunkSection(length=0.5, radius=0.02, resolution=15)
#]

# Build the robot and add it to the world
trunk_pcc = SoftTrunk.build_piecewise_constant_curvature(world_pcc, sections)

tf_pcc = TFPublisher(_world=world_pcc, node=node)
viz_pcc = VizMarkerPublisher(_world=world_pcc, node=node)

print("PCC Robot Ready. Set fixed frame to 'pcc/base' in RViz.")

# To see the robot model in RViz, we need to push the initial state to the visualization
update_visualization(world_pcc, tf_pcc, viz_pcc)
```

### 1.2 Animating PCC


This loop animates the Piecewise Constant Curvature (PCC) model by continuously updating its two control parameters for each section:

- **Curvature ($\kappa$):** We apply a sine wave to the curvature. By subtracting the section index (`t - section`), we create a phase shift. This results in a wave-like motion that propagates from the base to the tip.
- **Bending Plane ($\phi$):** We increase the bending plane angle linearly over time. This causes the entire robot to rotate or spiral its bending direction around the central axis.

```{code-cell} ipython3
print("Starting PCC Animation")
try:
    for t in np.linspace(0, 10, 200):
        for section, (k_dof, p_dof) in enumerate(trunk_pcc.pcc_sections):
            world_pcc.state[k_dof.id].position = 1.5 * np.sin(t - section * 1.0)    
            world_pcc.state[p_dof.id].position = t * 0.5
        update_visualization(world_pcc, tf_pcc, viz_pcc)
        time.sleep(0.03)
except KeyboardInterrupt:
    print("Stopped.")
```

### 1.3 Inverse Kinematics PCC

We use the built-in Quadratic Programming (QP) solver to find the robot configuration required to reach a specific target point in space.
- The solver attempts to minimize the distance between the robot tip and a target marker (red sphere).
- The solver automatically optimizes the **Curvature ($\kappa$)** and **Bending Plane ($\phi$)** for every section.
- Because PCC assumes a fixed arc length, the robot can only reach the target if it lies within the geometric shell formed by its constant total length.

```{code-cell} ipython3
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Sphere, Color
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
import numpy as np

# Setup goal marker (transparent red sphere)
goal_visual = Sphere(radius=0.03, color=Color(1.0, 0.0, 0.0, 0.5))
goal_body = Body(
    name=PrefixedName(name="goal_marker", prefix="ik"), 
    visual=ShapeCollection([goal_visual])
)

# Add marker to world and connect to root
with world_pcc.modify_world():
    world_pcc.add_body(goal_body)
    goal_connection = Connection6DoF.create_with_dofs(
        parent=world_pcc.root, 
        child=goal_body, 
        world=world_pcc
    )
    world_pcc.add_connection(goal_connection)

# Initialize the Solver
ik_solver = InverseKinematicsSolver(world=world_pcc)

# Give the robot a tiny nudge
for k_dof in trunk_pcc.kappa_dofs:
    world_pcc.state[k_dof.id].position = 0.01 
world_pcc.notify_state_change()

# Define a random target pose within reachable bounds (adjust as needed based on your robot's workspace)
x = np.random.uniform(-0.4, 0.4)
y = np.random.uniform(-0.4, 0.4)
z = np.random.uniform(0.5, 0.9)

target_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=x, y=y, z=z, reference_frame=world_pcc.root
)

# Update marker and visualize target
goal_connection.origin = target_pose
update_visualization(world_pcc, tf_pcc, viz_pcc)

# Solve the IK
try:
    print(f"Solving IK for target: [{x:.2f}, {y:.2f}, {z:.2f}]")
    
    # Pass the tip from the robot's manipulator chain
    ik_results = ik_solver.solve(
        root=world_pcc.root,
        tip=trunk_pcc.manipulator_chains[0].tip,
        target=target_pose,
        max_iterations=500,
        dt=0.1
    )
    
    # Apply results back to the world state
    for dof, position in ik_results.items():
        world_pcc.state[dof.id].position = position
    
    # Final Refresh
    update_visualization(world_pcc, tf_pcc, viz_pcc)
    print("Success: Robot reached the target marker!")

except Exception as e:
    print(f"IK Failed: {e}")
```

## 2. Cosserat Rod
Cosserat models allow for torsion (twist) and stretching, which PCC cannot represent.

### 2.1 Build Cosserat Robot

```{code-cell} ipython3
world_cosserat = World()

# Uniform sections for the trunk robot
sections = [SoftTrunkSection(length=0.3, radius=0.02, resolution=10)] * 3

#  Custom sections for the trunk robot (uncomment to use)
#sections = [
#    SoftTrunkSection(length=0.2, radius=0.08, resolution=5),
#    SoftTrunkSection(length=0.3, radius=0.04, resolution=10),
#    SoftTrunkSection(length=0.5, radius=0.02, resolution=15)
#]

trunk_cos = SoftTrunk.build_cosserat(world_cosserat, sections)

tf_cos = TFPublisher(_world=world_cosserat, node=node)
viz_cos = VizMarkerPublisher(_world=world_cosserat, node=node)

print("Cosserat Robot Ready. Set fixed frame to 'cosserat/base' in RViz.")

# To see the robot model in RViz, we need to push the initial state to the visualization
update_visualization(world_cosserat, tf_cos, viz_cos)
```

### 2.2 Animating Cosserat


This loop demonstrates the capabilities of the Cosserat Rod model by manipulating four strain parameters for each section:

-  **Bending ($u_x, u_y$):** By applying a sine wave to the X-axis and a cosine wave to the Y-axis, the robot performs a circular motion, sweeping around its central axis.
-  **Torsion ($u_z$):** We oscillate the torsion parameter to show the robot twisting back and forth around its own spine. This is a feature of the Cosserat model that is not possible with standard PCC.
-  **Extension ($v_z$):** We vary the extension strain between 0.5 and 1.5. This causes the robot to stretch and shrink in total length, also not possible with standard PCC.

```{code-cell} ipython3
try:
    print("Starting Cosserat Animation")
    for t in np.linspace(0, 10, 200):
        for section, (bx, by, tor, ext) in enumerate(trunk_cos.cosserat_sections): # Update world state for each section
            world_cosserat.state[bx.id].position = 2.0 * np.sin(t) # Curvature (Bending X)
            world_cosserat.state[by.id].position = 2.0 * np.cos(t) # Curvature (Bending Y)
            world_cosserat.state[tor.id].position = 1.5 * np.sin(t * 0.5) # Torsion (Twisting)      
            world_cosserat.state[ext.id].position = 1.0 + 0.5 * np.sin(t) # Strecthing
        update_visualization(world_cosserat, tf_cos, viz_cos)
        time.sleep(0.03)
except KeyboardInterrupt:
    print("Stopped.")
```

### 2.3 Inverse Kinematics Cosserat

Inverse Kinematics using the Cosserat Rod model. 

- The robot attempts to touch a target marker (green sphere).
- The solver optimizes four variables per section: **Bending rates ($u_x, u_y$)**, **Torsion ($u_z$)**, and **Extension ($v_z$)**.
- Unlike the PCC model, the Cosserat solver can reach targets that are further away by physically stretching the robot (increasing $v_z$). It can also twist the body (torsion) to match a specific target orientation.

```{code-cell} ipython3
from semantic_digital_twin.world_description.world_entity import Body
from semantic_digital_twin.world_description.shape_collection import ShapeCollection
from semantic_digital_twin.world_description.geometry import Sphere, Color
from semantic_digital_twin.world_description.connections import Connection6DoF
from semantic_digital_twin.datastructures.prefixed_name import PrefixedName
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
import numpy as np

# Setup goal marker (Transparent green sphere)
cos_goal_visual = Sphere(radius=0.03, color=Color(0.0, 1.0, 0.0, 0.5)) 
cos_goal_body = Body(
    name=PrefixedName(name="cos_goal_marker", prefix="ik"), 
    visual=ShapeCollection([cos_goal_visual])
)

# Add marker to world and connect to root
with world_cosserat.modify_world():
    world_cosserat.add_body(cos_goal_body)
    cos_goal_connection = Connection6DoF.create_with_dofs(
        parent=world_cosserat.root, 
        child=cos_goal_body, 
        world=world_cosserat
    )
    world_cosserat.add_connection(cos_goal_connection)

# Initialize the Solver for Cosserat world
ik_solver_cos = InverseKinematicsSolver(world=world_cosserat)

# Give the robot a tiny nudge to help numerical convergence
# nudge bending and torsion, but keep extension at rest length (1.0)
for bx_dof in trunk_cos.bending_x_dofs:
    world_cosserat.state[bx_dof.id].position = 0.01
for by_dof in trunk_cos.bending_y_dofs:
    world_cosserat.state[by_dof.id].position = 0.01
for tor_dof in trunk_cos.torsion_dofs:
    world_cosserat.state[tor_dof.id].position = 0.01
for ext_dof in trunk_cos.extension_dofs:
    world_cosserat.state[ext_dof.id].position = 1.0

world_cosserat.notify_state_change()

# Define a random target pose
# Cosserat can reach further due to extension, so we can use a wider Z bound
x = np.random.uniform(-0.6, 0.6)
y = np.random.uniform(-0.6, 0.6)
z = np.random.uniform(0.6, 1.5) # Targets beyond robot rest length (~1.0m in the model we defined) will trigger stretching
yaw = np.random.uniform(-np.pi, np.pi)

target_pose_cos = HomogeneousTransformationMatrix.from_xyz_rpy(
    x=x, y=y, z=z, yaw=yaw, reference_frame=world_cosserat.root
)

# Update marker and visualize target
cos_goal_connection.origin = target_pose_cos
update_visualization(world_cosserat, tf_cos, viz_cos)

# Solve the IK
try:
    print(f"Solving Cosserat IK for target: [{x:.2f}, {y:.2f}, {z:.2f}, {yaw:.2f}]")
    
    ik_results_cos = ik_solver_cos.solve(
        root=world_cosserat.root,
        tip=trunk_cos.manipulator_chains[0].tip,
        target=target_pose_cos,
        max_iterations=500,
        dt=0.1,
        translation_velocity=1.0
    )
    
    # Apply results back to the world state
    for dof, position in ik_results_cos.items():
        world_cosserat.state[dof.id].position = position
    
    # Final Refresh
    update_visualization(world_cosserat, tf_cos, viz_cos)
    print("Success: Cosserat model reached the target marker!")

except Exception as e:
    print(f"Cosserat IK Failed: {e}")
```

## 3. Workspace Reachability Analysis


This experiment performs a statistical evaluation of the reachable workspace for both the Piecewise Constant Curvature (PCC) and Cosserat Rod models. For each model, 100 random trials are performed. In each trial, a target pose is generated within a defined bounding box. The Inverse Kinematics (IK) solver attempts to reach the target within 300 iterations. A trial is marked as a Success if the final Euclidean distance between the robot tip and the target is less than 3 cm.

```{code-cell} ipython3
import numpy as np
import time
from semantic_digital_twin.spatial_types import HomogeneousTransformationMatrix
from semantic_digital_twin.spatial_computations.ik_solver import InverseKinematicsSolver

# Initialize solvers
ik_solver_pcc = InverseKinematicsSolver(world=world_pcc)
ik_solver_cos = InverseKinematicsSolver(world=world_cosserat)

def run_reachability_study(world, robot, solver, num_trials=100):
    # Targets sampled from a wide volume to test model limits
    setup = {
        "x_range": (-0.8, 1.2),
        "y_range": (-0.8, 1.2),
        "z_range": (0.05, 1.5),
        "yaw_range": (-np.pi, np.pi),
    }

    print(f"EXPERIMENT: {robot.name.prefix.upper()} REACHABILITY STUDY")
    print(f"Target Volume: X,Y {setup['x_range']}, Z {setup['z_range']}")
    print(f"Total Trials: {num_trials}")

    success_count = 0
    errors = []
    times = []
    
    # Identify the tip body from the primary manipulator chain
    tip_body = robot.manipulator_chains[0].tip

    def reset_robot_state():
        """Helper to reset robot to a neutral starting nudge using robot properties."""
        if robot.name.prefix == "pcc":
            for d in robot.kappa_dofs: world.state[d.id].position = 0.01
            for d in robot.phi_dofs: world.state[d.id].position = 0.01
        elif robot.name.prefix == "cosserat":
            for d in robot.bending_x_dofs: world.state[d.id].position = 0.01
            for d in robot.bending_y_dofs: world.state[d.id].position = 0.01
            for d in robot.torsion_dofs: world.state[d.id].position = 0.01
            for d in robot.extension_dofs: world.state[d.id].position = 1.0
        world.notify_state_change()

    print(f"Starting trials for {robot.name.name}...")

    for i in range(num_trials):
        # Generate a random target pose
        x = np.random.uniform(*setup['x_range'])
        y = np.random.uniform(*setup['y_range'])
        z = np.random.uniform(*setup['z_range'])
        yaw = np.random.uniform(*setup['yaw_range'])
        
        # we can include yaw in the target pose as well, but for error calculation we will focus on XYZ distance
        target_pose = HomogeneousTransformationMatrix.from_xyz_rpy(
            x=x, y=y, z=z, reference_frame=world.root
        )

        # Reset and Nudge
        reset_robot_state()

        # Solve Inverse Kinematics
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
            
            # Apply results to verify final error
            for dof, pos in ik_results.items():
                world.state[dof.id].position = pos
            world.notify_state_change()
            
            # Error Calculation
            current_fk = world.compute_forward_kinematics_np(world.root, tip_body)
            current_xyz = current_fk[:3, 3]
            target_xyz = np.array([x, y, z])
            dist_error = np.linalg.norm(current_xyz - target_xyz)
            
            # Threshold for success: 3cm
            if dist_error < 0.03:
                success_count += 1
            errors.append(dist_error)
            
        except Exception:
            # Solver failed to converge or target was mathematically unreachable
            current_fk = world.compute_forward_kinematics_np(world.root, tip_body)
            dist_error = np.linalg.norm(current_fk[:3, 3] - np.array([x, y, z]))
            errors.append(dist_error)
            
        times.append(time.time() - start_time)
        
        if (i+1) % 25 == 0:
            print(f"  Progress: {i+1}/{num_trials} trials complete...")

    return {
        "success_rate": (success_count / num_trials) * 100,
        "mean_error": np.mean(errors),
        "mean_time": np.mean(times)
    }

# Run studies
results_pcc = run_reachability_study(world_pcc, trunk_pcc, ik_solver_pcc, num_trials=100)
results_cos = run_reachability_study(world_cosserat, trunk_cos, ik_solver_cos, num_trials=100)

# Print Summary Table
print("\n" + 30 * "=")
print("RESULTS")
print(30 * "=")
print(f"PCC Success Rate:       {results_pcc['success_rate']:.1f}%")
print(f"PCC Mean Error:         {results_pcc['mean_error']:.4f}m")
print(f"PCC Avg Solve Time:     {results_pcc['mean_time']:.4f}s")
print(30 * "-")
print(f"Cosserat Success Rate:  {results_cos['success_rate']:.1f}%")
print(f"Cosserat Mean Error:    {results_cos['mean_error']:.4f}m")
print(f"Cosserat Avg Solve Time:{results_cos['mean_time']:.4f}s")
```

## Cleanup

```{code-cell} ipython3
node.destroy_node()
rclpy.shutdown()
print("ROS Node shut down.")
```
