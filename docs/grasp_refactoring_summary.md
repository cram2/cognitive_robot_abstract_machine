# GraspDescription Refactoring: Robot-Centric Coordinate Frames

## Overview

Refactored `GraspDescription` (`coraplex/src/coraplex/datastructures/grasp.py`) to transition grasp definitions from an **object-centric frame** to a **robot-centric frame**.

### 1. Problem (Object-Centric Grasps)
Previously, approach directions (`FRONT`, `LEFT`, `RIGHT`, `BACK`, `TOP`) were evaluated relative to the target object's coordinate frame:
- If a target object (e.g. cereal box or milk carton) was rotated on a table, the `FRONT` approach direction rotated with it.
- If rotated 180°, the robot attempted to reach behind the object through a wall or obstacle, ignoring the robot's physical base position.

### 2. Solution (Robot-Centric Grasps)
Approach directions are now defined strictly in the **robot's coordinate system**:
- `FRONT` consistently approaches from the front of the robot (straight out from the robot's chest toward the object).
- The object can be placed in any arbitrary orientation; `FRONT` always targets the surface facing the robot.

---

## Technical Implementation

### 1. `_world_frame_gripper_rotation()`
Added a central helper method computing the gripper orientation directly in the world frame from the robot's base pose:

```python
map_R_gripper = map_R_robot @ robot_R_side @ robot_R_vertical @ robot_R_horizontal @ ee_correction
```

This chains:
1. `map_R_robot`: The robot's current world orientation.
2. `robot_R_side`: The desired approach direction in the robot's frame.
3. `robot_R_vertical`: Vertical alignment (tilt).
4. `robot_R_horizontal`: Roll angle.
5. `ee_correction`: End-effector physical mounting rotation.

### 2. Frame Projection in `grasp_orientation()`
To preserve compatibility with the downstream `_pose_sequence` composition pipeline, `grasp_orientation(object_pose)` projects this world-frame result back into the reference frame of `object_pose`:

```python
pose_R_world = object_pose.to_rotation_matrix().to_np()[:3, :3].T
pose_R_gripper = pose_R_world @ map_R_gripper
```

When composed inside `_pose_sequence`, this yields the exact robot-centric target orientation in world coordinates:
$$\text{target\_R\_gripper} = \text{target\_R\_pose} \cdot \text{pose\_R\_world} \cdot \text{map\_R\_gripper} = \text{target\_R\_map} \cdot \text{map\_R\_gripper}$$

### 3. Math Redesign in `calculate_manipulator_axis()`
Renamed from `calculate_end_effector_axis` and simplified. The approach and lift axes are physical properties of the gripper hardware itself (not dynamic approach angles). The method was rewritten as a direct matrix projection into the tool mounting frame:

```python
def calculate_manipulator_axis(self, axis: AxisIdentifier) -> List[float]:
    world_axis = np.array(axis.value, dtype=float)
    ee_correction = quaternion_matrix(self.end_effector.front_facing_orientation.to_np())[:3, :3]
    return (ee_correction.T @ world_axis).round(4).tolist()
```

---

## Testing & Verification

1. **Automated Unit Tests:**
   All 24 unit tests in `test/coraplex_test/test_dataclasses/test_grasp.py` pass.

   Run locally from the repository root:
   ```bash
   export PYTHONPATH=$PWD/physics_simulators/src:$PWD/coraplex/src:$PWD/semantic_digital_twin/src:$PWD/krrood/src:$PYTHONPATH
   source /opt/ros/jazzy/setup.bash
   source install/setup.bash
   python -m pytest test/coraplex_test/test_dataclasses/test_grasp.py -v
   ```

2. **Standalone Invariant Script:**
   `scripts/verify_grasp_robot_frame.py` verifies coordinate transformations independently of ROS dependencies (7/7 checks pass):
   - Rotating the robot rotates the approach angle accordingly.
   - Rotating the target object on the table does not change the world-frame approach direction.

   Run via:
   ```bash
   bash run_demo.sh scripts/verify_grasp_robot_frame.py
   ```

---

## Summary of Changed Files

| File | Type | Description |
| :--- | :--- | :--- |
| `coraplex/src/coraplex/datastructures/grasp.py` | **Core** | Implemented `_world_frame_gripper_rotation()`, updated `grasp_orientation(object_pose)` with projection math, renamed and simplified `calculate_manipulator_axis()`. |
| `coraplex/src/coraplex/datastructures/rotations.py` | **Core** | Updated docstrings clarifying rotation tables encode offsets in the robot's coordinate frame. |
| `coraplex/src/coraplex/robot_plans/actions/core/misc.py` | **Plan** | Updated `MoveToReach` action caller to pass the target pose to the new `grasp_orientation` API. |
| `test/coraplex_test/test_dataclasses/test_grasp.py` | **Tests** | Updated expected quaternions across 24 unit tests to match robot-centric semantics; verified manipulation axis stability. |
| `test/coraplex_test/test_designator/test_tracy_action_designator.py` | **Tests** | Updated `grasp_orientation` calls with `Pose(reference_frame=...)`. |
| `test/coraplex_test/test_designator/test_multi_robot_action_designator.py` | **Tests** | Updated `grasp_orientation` calls with `Pose(reference_frame=...)`. |
| `scripts/verify_grasp_robot_frame.py` | **Tool** | Standalone mathematical verification script verifying all frame transformations and invariants without ROS dependencies. |
| `coraplex/demos/pr2/pr2_giskard_pick_place_demo.py` | **Demo** | Interactive pick-and-place simulation demo with Giskard collision avoidance and live RViz2 visualization. |

---

## Visual Verification via Demo (RViz2)

To visually observe the robot execute the robot-centric grasp sequence:

1. **Start RViz2 (Terminal 1):**
   ```bash
   source /opt/ros/jazzy/setup.bash
   rviz2
   ```
   * Set **Fixed Frame** to: `iai_kitchen/room_root`
   * Add **TF** display
   * Add **MarkerArray** display with Topic `/semworld/viz_marker` and Durability `Transient Local`

2. **Launch the Demo (Terminal 2):**
   ```bash
   bash run_demo.sh coraplex/demos/pr2/pr2_giskard_pick_place_demo.py
   ```
   * Select option `1` (Simulation mode).
   * Press **Enter** to watch the PR2 approach the object from its forward direction, grasp, lift, place, and park.

3. **Configuring Different Approach Directions:**
   In `coraplex/demos/pr2/pr2_giskard_pick_place_demo.py` at **line 105**:
   * **Front:** `GraspDescription(ApproachDirection.FRONT, VerticalAlignment.NoAlignment, manipulator)`
   * **Left:** `GraspDescription(ApproachDirection.LEFT, VerticalAlignment.NoAlignment, manipulator)`
   * **Right:** `GraspDescription(ApproachDirection.RIGHT, VerticalAlignment.NoAlignment, manipulator)`
   * **Top:** `GraspDescription(ApproachDirection.FRONT, VerticalAlignment.TOP, manipulator)`
