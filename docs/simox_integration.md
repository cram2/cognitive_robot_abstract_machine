# Simox Grasp Planner — Integration, Architecture & Demo Guide

> **Branch:** `feature/robot-centric-grasp`  
> **Related Documentation:** [Grasp Refactoring Summary](grasp_refactoring_summary.md)

---

## Table of Contents

1. [What is Simox?](#1-what-is-simox)
2. [How Simox Finds and Ranks Grasps](#2-how-simox-finds-and-ranks-grasps)
3. [File & Directory Overview](#3-file--directory-overview)
4. [How I Integrated Simox into the Mono-Repo](#4-how-i-integrated-simox-into-the-mono-repo)
5. [File Formats: STL, URDF, and ManipulationObject XML](#5-file-formats-stl-urdf-and-manipulationobject-xml)
6. [The .iv (Open Inventor) Format Problem and My Solution](#6-the-iv-open-inventor-format-problem-and-my-solution)
7. [The STL Unit Problem and My Solution](#7-the-stl-unit-problem-and-my-solution)
8. [The Simox ↔ CoraPlex Frame Conversion](#8-the-simox--coraplex-frame-conversion)
9. [How Simox Works With Giskard and CoraPlex](#9-how-simox-works-with-giskard-and-coraplex)
10. [The PR2 Pick-and-Place Demo](#10-the-pr2-pick-and-place-demo)
11. [Running the Demo](#11-running-the-demo)
12. [The Grasp Planner GUI](#12-the-grasp-planner-gui)
13. [Installation & Prerequisites](#13-installation--prerequisites)
14. [Grasp Refactoring Architecture Summary](grasp_refactoring_summary.md)

---

## 1. What is Simox?

**Simox** (Simulation of Motion and Grasping for Robots) is an open-source C++ robotics framework developed at KIT. Within this project, I use its sub-library **GraspStudio**, which implements:

- **`GraspPlanner::ApproachMovementSurfaceNormal`** — samples candidate approach points on the surface of an object's STL mesh by shooting rays along local surface normals.
- **`GraspQuality::GraspQualityMeasureWrenchSpace`** — scores each grasp by computing the *wrench resistance* (force-closure quality) in 6-DOF wrench space. A score close to `1.0` means the gripper can resist arbitrary forces and torques; close to `0.0` means the grasp would slip.
- Collision checking against the full PR2 kinematic model to discard physically infeasible grasps.

Simox is **not** embedded as a Python library. It runs as a **standalone ROS 2 service node** (`grasp_planner_service_node`) on the `/plan_grasp` service topic. CoraPlex sends it requests and receives ranked grasp poses back.

---

## 2. How Simox Finds and Ranks Grasps

```
┌─────────────────────────────────────────────────────────────┐
│  CoraPlex sends a PlanGrasp.Request to Simox:               │
│    - robot_model_path (pr2.xml)                             │
│    - object_model_path (e.g. breakfast_cereal.xml)          │
│    - object_pose (in robot base frame, meters)              │
│    - end_effector_name (r_gripper / l_gripper)              │
│    - num_grasps_to_plan, quality_threshold, timeout_ms      │
└────────────────────────────┬────────────────────────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │  Simox GraspStudio            │
             │  1. Load PR2 robot model      │
             │  2. Load object STL mesh      │
             │  3. Sample approach points    │
             │     on the object surface     │
             │     (ApproachMovement         │
             │      SurfaceNormal)           │
             │  4. For each approach:        │
             │     a. Position gripper       │
             │     b. Check collision with   │
             │        robot + object mesh    │
             │     c. Score via wrench space │
             │        (GraspQualityMeasure   │
             │         WrenchSpace)          │
             │  5. Return top N valid grasps │
             │     sorted by quality score   │
             └───────────────────────────────┘
                             │
                             ▼
             ┌───────────────────────────────┐
             │  PlanGrasp.Response           │
             │    - grasp_poses[] (geometry_ │
             │      msgs/Pose in robot frame)│
             │    - qualities[] (float[])    │
             │    - success, error_message   │
             └───────────────────────────────┘
```

**Quality score interpretation:**

| Score | Meaning |
|-------|---------|
| `> 0.15` | High quality — good force closure, preferred |
| `0.05 – 0.15` | Acceptable — use if higher scores not available |
| `< 0.05` | Poor quality — likely to slip; filtered by `quality_threshold` |
| `0.0` | Collision or fully infeasible |

**Why bottom grasps are skipped:** Simox sometimes returns grasps where the gripper approaches from below (e.g. through the table). My `_classify_approach()` function detects these (`fwd_z > 0.5`) and discards them automatically.

---

## 3. Architecture & Multi-Repository Overview (3 Repositories)

Running the complete autonomous grasp planning and execution pipeline on the PR2 (in simulation or on physical hardware) coordinates **three separate repositories**:

```
┌────────────────────────────────────────────────────────────────────────────────────────┐
│ 1. cognitive_robot_abstract_machine (Monorepo, branch: feature/robot-centric-grasp)    │
│    - High-level reasoning, CoraPlex plans, SimoxPickUpAction, semantic_digital_twin    │
│    - simox_grasp_planner.py (ROS 2 service client & frame transformations)             │
│    - PR2 Pick & Place Demo (coraplex/demos/pr2_simox_demo/demo.py)                     │
│    - PR2 ROS bridge motion mapping (translates joint trajectories for bridge)          │
└────────────────────────────────────────┬───────────────────────────────────────────────┘
                                         │
                 ┌───────────────────────┴───────────────────────┐
                 │                                               │
                 ▼                                               ▼
┌───────────────────────────────────────┐       ┌────────────────────────────────────────┐
│ 2. grasp_planner (Simox Service Repo) │       │ 3. ros1_2_bridge (Docker Bridge Repo)  │
│    - KIT Simox C++ library & models   │       │    - Bridges ROS 2 Jazzy ↔ ROS 1       │
│    - grasp_planner_msgs               │       │      Noetic without DDS clashes        │
│    - grasp_planner_service C++ node   │       │    - Forwards joint trajectory actions │
│      (/plan_grasp service)            │       │      directly to PR2 hardware          │
│    - scripts/install_simox.sh         │       │    - start_bridge_to_robot.sh          │
└───────────────────────────────────────┘       └────────────────────────────────────────┘
```

### Repository Structure & Responsibilities

#### Repository 1: `cognitive_robot_abstract_machine` (Monorepo)
```
cognitive_robot_abstract_machine/
│
├── coraplex/
│   ├── src/coraplex/external_interfaces/
│   │   └── simox_grasp_planner.py          ← Main Simox↔CoraPlex bridge
│   │
│   ├── src/coraplex/datastructures/
│   │   └── grasp.py                        ← GraspPose, translate_pose_along_local_axis
│   │
│   ├── src/coraplex/robot_plans/actions/core/
│   │   └── pick_up.py                      ← SimoxPickUpAction (high-level action plan)
│   │
│   ├── alternative_motion_mappings/
│   │   └── pr2_motion_mapping.py           ← PR2 trajectory mapping for ROS bridge
│   │
│   ├── resources/objects/
│   │   ├── breakfast_cereal.stl            ← Mesh in meters (used directly by Simox)
│   │   ├── milk.stl
│   │   ├── bowl.stl
│   │   └── apartment_bowl.stl
│   │
│   └── demos/
│       ├── pr2_simox_demo/
│       │   ├── demo.py                     ← End-to-end pick-and-place demo
│       │   └── README.md                   ← Quick-start guide
│       └── pr2_ros_bridge_demo/            ← Low-level bridge validation demo
│
└── docs/
    ├── simox_integration.md                ← This document
    └── grasp_refactoring_summary.md        ← GraspDescription refactoring summary
```

#### Repository 2: `grasp_planner` (Simox ROS 2 Workspace)
```
grasp_planner/ (<simox_ws>)
├── grasp_test_files/
│   └── resources/
│       ├── robots/
│       │   ├── pr2.xml                     ← Simox robot XML entry point
│       │   └── pr2_for_simox.urdf          ← PR2 URDF with Simox EEF definitions
│       └── objects/                        ← Auto-generated cache (not version-controlled)
│           ├── breakfast_cereal.stl        ← Copied/scaled STL in meters
│           ├── breakfast_cereal.xml        ← Auto-generated ManipulationObject XML
│           └── ...
├── grasp_planner_gui/                      ← Diagnostic GUI client
├── scripts/
│   └── install_simox.sh                    ← Automated Simox C++ build/install script
└── src/
    ├── grasp_planner_msgs/                 ← ROS 2 service definition (PlanGrasp.srv)
    └── grasp_planner_service/              ← C++ node (runs /plan_grasp)
```
> **Note:** `grasp_test_files/resources/objects/` is a **runtime cache**. STL files and XML wrappers are auto-generated by `simox_grasp_planner.py` on first use and do not need to be manually maintained.

#### Repository 3: `ros1_2_bridge` (ROS 1 ↔ ROS 2 Docker Bridge)
```
ros1_2_bridge/
├── Dockerfile                              ← Dual ROS 1 Noetic + ROS 2 Jazzy container
├── docker-compose.yml                      ← Network and container configuration
├── bridge_rules.yaml                       ← Topic/Action forwarding rules
└── start_bridge_to_robot.sh                ← Entrypoint to launch bridge against real PR2 / Gazebo
```
> **Purpose:** PR2 hardware controllers run exclusively on ROS 1 Noetic. The bridge forwards ROS 2 action goals (`FollowJointTrajectory`, `Pr2GripperCommand`) to ROS 1 action servers without DDS transport clashes. Required for Mode 2 (Real PR2) and Gazebo.

---

## 4. How I Integrated Simox into the Mono-Repo

### What existed before

The mono-repo previously had no Simox integration. Giskard was used directly with hand-coded grasp poses.

### What I added

| Component | Location in mono-repo | Purpose |
|-----------|----------------------|---------|
| `simox_grasp_planner.py` | `coraplex/src/coraplex/external_interfaces/` | Core bridge: calls the Simox ROS 2 service, converts results to CoraPlex types |
| `demo.py` | `coraplex/demos/pr2_simox_demo/` | Full pick-and-place pipeline for PR2 |
| `apartment_bowl.stl` | `coraplex/resources/objects/` | New conical bowl object mesh |
| `GraspPose.approach`, `.quality` fields | `coraplex/src/coraplex/datastructures/grasp.py` | Extended `GraspPose` to carry Simox metadata |
| `translate_pose_along_local_axis()` | `coraplex/src/coraplex/datastructures/grasp.py` | Utility for pre-grasp backoff along the tool axis |
| `ExecutionEnvironment.BRIDGE` | `coraplex/src/coraplex/datastructures/enums.py` | Execution mode for real PR2 via ROS 1/2 bridge |
| PR2 bridge motion mapping | `coraplex/alternative_motion_mappings/pr2_motion_mapping.py` | Translates Giskard trajectories → ROS 1 actionlib via docker bridge |

### Integration design pattern

My Simox integration follows the same **external-interface** pattern as `robokudo.py` in this repo:

1. **Lazy initialisation** — The ROS 2 service client is created only on first call, not at import time, so importing the module does not require a live ROS 2 environment.
2. **Module-level singleton** — A single `_SimoxClient` instance is reused across calls within one process.
3. **Minimal public surface** — Only `plan_grasps_for_body()` is public. All helpers are prefixed with `_`.
4. **Zero Simox source modifications** — Simox runs as a black-box ROS 2 service. All CoraPlex-specific logic lives entirely within `simox_grasp_planner.py`.

---

## 5. File Formats: STL, URDF, and ManipulationObject XML

### STL (object meshes)

Object meshes are stored as binary STL files in `coraplex/resources/objects/`. CoraPlex uses them for:
- Collision detection via convex decomposition (handled by `semantic_digital_twin`)
- Visual markers in RViz2

All STLs used with Simox **must be in meters** in the cache directory (see Section 6).

### pr2.xml (Simox robot entry point)

Located at `<simox_ws>/grasp_test_files/resources/robots/pr2.xml`, this is a **Simox-format XML** referencing:
- `pr2_for_simox.urdf` — A specially adapted URDF with Simox-compatible end-effector definitions (finger contacts, preshapes)
- End-effector groups: `r_gripper`, `l_gripper`
- Kinematic chains: `RightArm`, `LeftArm`

This file is **read exclusively by the Simox service**. CoraPlex only sends its path as a string in the service request.

### ManipulationObject XML (auto-generated per object)

For each object, `simox_grasp_planner.py` auto-generates a file like `breakfast_cereal.xml`:

```xml
<?xml version="1.0" encoding="UTF-8"?>
<!-- Auto-generated by simox_grasp_planner.py for body: breakfast_cereal.stl -->
<ManipulationObject name="breakfast_cereal.stl">
    <Visualization>
        <File type="stl">breakfast_cereal.stl</File>
    </Visualization>
    <CollisionModel>
        <File type="stl">breakfast_cereal.stl</File>
    </CollisionModel>
    <GraspSet name="Simox_r_gripper" RobotType="PR2" EndEffector="r_gripper"/>
</ManipulationObject>
```

This XML is placed in the cache directory alongside the STL and its path is sent in the `PlanGrasp.Request`.

---

## 6. The .iv (Open Inventor) Format Problem and My Solution

### What Simox Expects by Default

Simox was originally designed at KIT to work with **Open Inventor (`.iv`) files** — a legacy 3D scene graph format developed by SGI and read by the Coin3D library. The internal Simox mesh loaders (`VirtualRobot::CoinVisualizationNode`) natively load `.iv` files for both visualisation and collision geometry.

When you look at the robot XML files already present in the Simox workspace (e.g. `pr2.xml`, `boxy.urdf`, `hsrb.xml`), you will notice that their object meshes all reference `.iv` files:

```xml
<!-- Example of how Simox's native robot descriptions reference meshes -->
<File type="inventor">pr2/meshes/base.iv</File>
```

### The Problem I Faced

The CoraPlex objects (breakfast cereal, bowl, apartment bowl, milk, etc.) are exported from CAD tools as **binary STL files**. I had no `.iv` files and no straightforward way to batch-convert the entire object library to Open Inventor format.

The first approach I attempted was to **point the ManipulationObject XML at the `.stl` files but declare the type as `inventor`**:

```xml
<!-- WRONG — Simox tries to parse the STL as an .iv scene graph and crashes -->
<Visualization>
    <File type="inventor">breakfast_cereal.stl</File>
</Visualization>
```

This caused Simox's Coin3D parser to throw a parse error and refuse to load the object, returning an immediate failure on the `/plan_grasp` service with no grasps.

### My Solution: `type="stl"` in the ManipulationObject XML

After reading the Simox/VirtualRobot source code, I found that the `CoinVisualizationNode` also supports a `type="stl"` attribute which routes loading through a native STL reader — completely bypassing Coin3D's `.iv` parser. This is not prominently documented in the Simox docs but is present in the source.

The fix I implemented was to declare the correct type in the generated XML:

```xml
<!-- CORRECT — Simox loads the STL natively via its STL mesh reader -->
<Visualization>
    <File type="stl">breakfast_cereal.stl</File>
</Visualization>
<CollisionModel>
    <File type="stl">breakfast_cereal.stl</File>
</CollisionModel>
```

This is what `simox_grasp_planner.py` now auto-generates for every object in `_ensure_simox_object_xml()`. **No `.iv` conversion is ever needed.**

### Why This Matters

This was the **primary blocker** that prevented Simox from loading any of the objects. Without this fix, the service would always return `success=False` with a parse error, regardless of object quality or robot configuration. Once I used `type="stl"`, all objects loaded correctly and grasp planning worked end-to-end.

### Summary

| Approach | Result |
|----------|--------|
| `type="inventor"` + `.stl` file | ❌ Coin3D parse error — service fails |
| Convert `.stl` → `.iv` manually | ⚠️ Possible but requires Coin3D tools; not scalable |
| `type="stl"` + `.stl` file | ✅ Works natively — used in my integration |

---

## 7. The STL Unit Problem and My Solution

### The Problem

Simox's `CoinVisualizationNode` mesh loader **expects STL vertex coordinates in meters**. This means:

- STL in **meters** (e.g. `breakfast_cereal.stl`, bounding box ≈ `[-0.10, +0.12] m`) → pass directly; Simox treats those as meters and produces a correctly-sized ~20 cm object.
- STL in **millimeters** (e.g. `apartment_bowl.stl` as originally exported, `Z: [0, 100]`) → Simox would interpret those millimeter values as meters and produce a 100-meter-tall bowl!

### My Solution: Auto-Detection in `simox_grasp_planner.py`

I implemented the `_detect_stl_unit()` function to apply a heuristic:

```python
def _detect_stl_unit(path: Path) -> str:
    # If the maximum vertex coordinate < 5.0, assume meters.
    # PR2-graspable objects are 5-30 cm (0.05-0.30 m) or 50-300 mm.
    return 'meters' if max_coord < 5.0 else 'millimeters'
```

If detected as **millimeters**, `_scale_stl()` pre-scales by `× 0.001` before copying to the cache:

```python
# In _ensure_simox_object_xml():
if unit == 'millimeters':
    _scale_stl(source_stl, target_stl_path, scale=0.001)
else:
    shutil.copy2(source_stl, target_stl_path)
```

This is **fully automatic** for any future objects that follow the heuristic.

> **`apartment_bowl.stl` special case:**
> This mesh was originally exported in millimeters (`Z: [0, 100]`) and had its origin at the base (not center) of the bowl.
> I pre-processed it once to:
> 1. Scale to meters (× 0.001): `Z: [0.000, 0.100]`
> 2. Center in Z (subtract 0.050): `Z: [-0.050, +0.050]`
>
> This centering matches all other object STLs in the repo (which are centered around `z = 0`) and ensures Simox places the object correctly relative to its declared pose.
>
> The original backup is at `coraplex/resources/objects/apartment_bowl_mm.stl.bak`.

---

## 8. The Simox ↔ CoraPlex Frame Conversion

This is the most critical technical detail of the integration. If this conversion is wrong, the gripper will approach at the wrong angle or collide with the object.

### Frame Definitions

**Simox PR2 tool frame** (`r_gripper_tool_joint` in `pr2_for_simox.urdf`):
```
origin xyz="0.13 0 0"  rpy="0 1.5707963 0"   (90° pitch around Y)
  Local +Z → forward approach direction (into the object)
  Local +Y → finger opening/closing axis
  Local +X → pointing "down" relative to Simox's palm
```

**CoraPlex PR2 tool frame** (`r_gripper_tool_frame` in standard PR2 URDF):
```
origin xyz="0.18 0 0"  rpy="0 0 0"
  Local +X → forward approach direction (into the object)
  Local +Y → finger opening/closing axis
  Local +Z → orthogonal up
```

### The Conversion (implemented in `_simox_pose_to_coraplex_tool_pose()`)

The relative rotation is a **−90° pitch around Y** plus a **+5 cm position offset** along the Simox forward axis:

```python
R_simox_to_coraplex = np.array([
    [0, 0, -1],   # CoraPlex +X ← Simox +Z (forward)
    [0, 1,  0],   # CoraPlex +Y ← Simox +Y (finger axis, unchanged)
    [1, 0,  0],   # CoraPlex +Z ← Simox +X
])

R_coraplex = R_simox @ R_simox_to_coraplex
P_coraplex = P_simox + 0.05 * R_simox[:, 2]   # +5cm along Simox forward axis (+Z)
```

The `0.05 m` offset accounts for `0.18 m − 0.13 m = 0.05 m`: the difference between the two tool frame origins along the approach axis.

### Approach Direction Classification

After converting to the robot base frame, `_classify_approach()` labels each grasp by reading the **Simox forward vector** (column 2 of `R_simox`):

| Condition on `fwd = R_simox[:, 2]` | Classified as |
|------------------------------------|---------------|
| `fwd_z > 0.5` | `skipped` (bottom grasp — filtered out) |
| `fwd_z < -0.4` | `top` |
| `fwd_x ≥ 0` and `\|fwd_x\| ≥ \|fwd_y\|` | `front` |
| `fwd_x < 0` and `\|fwd_x\| ≥ \|fwd_y\|` | `back` |
| `fwd_y ≥ 0` and `\|fwd_y\| > \|fwd_x\|` | `right` |
| `fwd_y < 0` and `\|fwd_y\| > \|fwd_x\|` | `left` |

---

## 9. How Simox Works With Giskard and CoraPlex

The complete data flow in the pick-and-place pipeline:

```
 CoraPlex World (PR2 + apartment scene + object)
        │
        │  1. Resolve body.global_pose → transform to robot base frame
        ▼
 simox_grasp_planner.py  ─── plan_grasps_for_body()
        │
        │  PlanGrasp.Request (robot XML, object XML, object pose in robot frame)
        ▼
 Simox ROS 2 Service  (/plan_grasp)
        │  Surface normal sampling → collision check → wrench space scoring
        │
        │  PlanGrasp.Response (grasp_poses[], qualities[])
        ▼
 simox_grasp_planner.py
        │  · _simox_pose_to_coraplex_tool_pose() — frame conversion
        │  · _classify_approach() — label top/front/back/left/right
        │  · Sort by quality descending per direction
        │
        │  Returns: Dict[str, List[GraspPose]]
        ▼
 demo.py  (or SimoxPickUpAction in pick_up.py)
        │  · Select preferred approach direction
        │  · Compute pre-grasp pose (backoff along tool +X or fixed vertical offset)
        │  · MoveManipulatorAction → pre-grasp pose (Giskard IK, joint space)
        │  · MoveToolCenterPointMotion (TRANSLATION) → grasp pose (Cartesian straight line)
        ▼
 Giskard Motion Planner  (inside simulated_robot context)
        │  · Collision-aware IK solution for each motion command
        │  · Publishes joint states and TF to ROS 2
        ▼
 Execution backend:
   Mode 1: CoraPlex simulation + RViz2 visualization only
   Mode 2: Simulated PR2 in Gazebo via ROS 1/2 bridge docker container
   Mode 3: Real PR2 robot via ROS 1/2 bridge docker container
```

### Key Design Decisions

| Decision | Rationale |
|----------|-----------|
| Simox for grasp generation, Giskard for motion planning | Clear separation of concerns: Simox knows about object geometry and force-closure; Giskard knows about arm kinematics and collision avoidance. |
| `MoveToolCenterPointMotion(TRANSLATION)` for final approach | Guarantees a geometrically straight Cartesian path to the grasp pose, preventing the gripper from sweeping sideways into the object. |
| Object pose sent in robot frame | Simox loads the PR2 robot at the world origin, so the object position must be expressed relative to the robot base. |
| `simulated_robot` context manager | Ensures Giskard and the CoraPlex world state (object positions, joint angles, TF) stay in sync throughout execution. |

---

## 10. The PR2 Pick-and-Place Demo

**File:** `coraplex/demos/pr2_simox_demo/demo.py`

### Supported Objects

| `--object` value | Mesh file | Grasp strategy |
|--|--|--|
| `cereal` / `breakfast_cereal` | `breakfast_cereal.stl` | Simox side-grasp (`right` approach), 4 cm gripper gap |
| `milk` | `milk.stl` | Simox side-grasp, 4 cm gripper gap |
| `bowl` / `apartment_bowl` | `apartment_bowl.stl` | Fixed vertical/tilted rim grasp (overrides Simox orientation), full gripper closure |

### Demo Steps

```
Step 1  Park both arms + Raise torso to max + Open right gripper
Step 2  Navigate PR2 base forward to working distance from kitchen counter
        (+ BOWL_EXTRA_FORWARD_M if picking bowl)
Step 3  Query Simox /plan_grasp → receive ranked GraspPose candidates
        For each candidate, highest quality first:
          3a. MoveManipulatorAction → pre-grasp pose (clearance from object)
          3b. MoveToolCenterPointMotion TRANSLATION → grasp pose (straight approach)
          If Giskard fails IK, try the next Simox candidate
Step 4  Close gripper (fully for bowl, 4 cm gap for cereal/milk)
        Attach object to gripper tool frame in the world model
Step 5a Lift object straight up (MoveToolCenterPointMotion TRANSLATION)
Step 5b Move PR2 base sideways along counter to place location
Step 5c Lower arm to place height (MoveToolCenterPointMotion TRANSLATION)
Step 6  Open gripper, detach object from gripper, retract arm straight up
Step 7  Move base back to start + Park both arms
```

### Execution Modes

| Mode | Flag | Description | Extra requirements |
|------|------|-------------|-------------------|
| 1 | `--mode 1` | RViz2 simulation only | Simox service + RViz2 |
| 2 | `--mode 2` | Simulated PR2 in Gazebo | Mode 1 + docker bridge container (`start_bridge_to_robot.sh --sim`) |
| 3 | `--mode 3` | Real PR2 robot | Mode 1 + docker bridge container + PR2 powered on |

### CLI Arguments

| Argument | Default | Options | Description |
|----------|---------|---------|-------------|
| `--object` | `cereal` | `cereal`, `milk`, `bowl`, `apartment_bowl` | Object to pick |
| `--arm` | `right` | `right`, `left` | Which arm to use |
| `--approach` | `right` | `front`, `right`, `left`, `top`, `back`, `any` | Preferred Simox approach direction |
| `--mode` | *(interactive)* | `1`, `2`, `3` | Execution mode |
| `--num-grasps` | `15` | any int | Number of grasp candidates to request from Simox |
| `--quality-threshold` | `0.01` | `0.0`–`1.0` | Minimum acceptable Simox quality score |
| `--no-prompt` | off | flag | Skip interactive Enter prompts between steps |

---

## 11. Running the Demo

The following uses placeholder paths. Replace:
- `<mono_repo>` with the absolute path to `cognitive_robot_abstract_machine`
- `<simox_ws>` with the absolute path to the Simox ROS 2 workspace root
- `<venv>` with the absolute path to the Python virtual environment

### Step 1 — Start the Simox Service

```bash
cd <simox_ws>
source /opt/ros/jazzy/setup.bash
source install/setup.bash
ros2 run grasp_planner_service grasp_planner_service_node
```

Expected: `[INFO] [grasp_planner_service_node]: Simox service ready on /plan_grasp`

### Step 2 — Start RViz2 (optional, recommended)

```bash
source /opt/ros/jazzy/setup.bash
rviz2
```

In RViz2:
- **Fixed Frame:** `apartment/apartment_root` (or `world`)
- Add **TF** display
- Add **MarkerArray** display → Topic: `/semworld/viz_marker` → Durability: `Transient Local`

### Step 3 — Run the Demo

```bash
source /opt/ros/jazzy/setup.bash
source <simox_ws>/install/setup.bash

PYTHONPATH=<mono_repo>/coraplex/src:<mono_repo>/semantic_digital_twin/src:<mono_repo>/giskardpy/src:<mono_repo>/krrood/src:$PYTHONPATH \
  <venv>/bin/python \
  <mono_repo>/coraplex/demos/pr2_simox_demo/demo.py \
  --object cereal \
  --arm right \
  --mode 1
```

**Quick one-liner (simulation, no prompts):**

```bash
source /opt/ros/jazzy/setup.bash && source <simox_ws>/install/setup.bash && \
PYTHONPATH=<mono_repo>/coraplex/src:<mono_repo>/semantic_digital_twin/src:<mono_repo>/giskardpy/src:<mono_repo>/krrood/src:$PYTHONPATH \
  <venv>/bin/python <mono_repo>/coraplex/demos/pr2_simox_demo/demo.py \
  --object cereal --mode 1 --no-prompt
```

**Expected final output:**
```
✔ Demo Complete! breakfast_cereal.stl picked via Simox and placed on the left!
```



## 12. The Grasp Planner GUI

The Grasp Planner GUI is a standalone diagnostic tool for visualising Simox grasp results without running the full demo.

**Location:** `<simox_ws>/grasp_planner_gui/gui_client.py`

**Launch:**
```bash
source /opt/ros/jazzy/setup.bash
source <simox_ws>/install/setup.bash
python3 <simox_ws>/grasp_planner_gui/gui_client.py
```

**Objects available in the GUI dropdown:**
- `apartment_bowl (Conical Bowl)`
- `bowl (Dish)`
- `milk (Carton)`
- `breakfast_cereal (Box)`
- `spoon (Utensil)`
- `jeroen_cup (Cup)`

The GUI sends the same `PlanGrasp.Request` as the demo and displays each grasp candidate with its quality score and approach direction in the Simox 3D viewer.

---

## 13. Installation & Prerequisites

### System Requirements

| Requirement | Version |
|-------------|---------|
| OS | Ubuntu 24.04 (Noble) |
| ROS 2 | Jazzy |
| Python | 3.12 |
| Docker | Required for Mode 2 (Real robot bridge) only |

### 1. Install Simox C++ Library

Simox (Simulation of Motion and Grasping for Robots) must be compiled with `VirtualRobot` and `GraspStudio` components.

You can use the one-shot automated installation script provided in the **`grasp_planner`** repository (`<simox_ws>/scripts/install_simox.sh` or `<simox_ws>/install_simox.sh`):

```bash
# Automated install (installs apt dependencies, clones cram2/Simox, compiles and installs):
cd <simox_ws>
bash scripts/install_simox.sh
```

Or manually:

```bash
# 1. Install build dependencies
sudo apt-get update && sudo apt-get install -y \
    build-essential cmake git \
    libboost-all-dev libeigen3-dev \
    qtbase5-dev libqt5opengl5-dev \
    libcoin-dev libsoqt-dev-common \
    libnlopt-dev liburdfdom-dev

# 2. Clone cram2 Simox repository
git clone https://github.com/cram2/Simox.git /tmp/Simox
cd /tmp/Simox

# 3. Build & install
cmake -B build -S . \
    -DCMAKE_BUILD_TYPE=Release \
    -DCMAKE_INSTALL_PREFIX=/usr/local \
    -DSimox_BUILD_EXAMPLES=OFF \
    -DSimox_BUILD_VirtualRobot=ON \
    -DSimox_BUILD_GraspStudio=ON \
    -DBUILD_TESTING=OFF
cmake --build build -j$(nproc)
sudo cmake --install build
```

### 2. Clone and Configure the Mono-Repo

```bash
git clone <mono_repo_url> cognitive_robot_abstract_machine
cd cognitive_robot_abstract_machine
git checkout feature/robot-centric-grasp
```

### 3. Build the Simox ROS 2 Workspace

The Simox service packages (`grasp_planner_msgs` and `grasp_planner_service`) are built in your ROS 2 workspace:

```bash
cd <simox_ws>
source /opt/ros/jazzy/setup.bash
colcon build --symlink-install
source install/setup.bash
```

Verify:
```bash
ros2 pkg list | grep grasp_planner
# Expected output:
#   grasp_planner_msgs
#   grasp_planner_service
```

### 4. Install Python Dependencies

```bash
python3 -m venv <venv>
source <venv>/bin/activate

# Install mono-repo packages in editable mode:
pip install -e <mono_repo>/coraplex
pip install -e <mono_repo>/semantic_digital_twin
pip install -e <mono_repo>/giskardpy
pip install -e <mono_repo>/krrood
```

### 5. Verify the Integration End-to-End

```bash
# Terminal 1: Start the Simox service
source /opt/ros/jazzy/setup.bash
source <simox_ws>/install/setup.bash
ros2 run grasp_planner_service grasp_planner_service_node

# Terminal 2: Run demo in simulation mode
source /opt/ros/jazzy/setup.bash && source <simox_ws>/install/setup.bash

PYTHONPATH=<mono_repo>/coraplex/src:<mono_repo>/semantic_digital_twin/src:<mono_repo>/giskardpy/src:<mono_repo>/krrood/src:$PYTHONPATH \
  <venv>/bin/python <mono_repo>/coraplex/demos/pr2_simox_demo/demo.py \
  --object cereal --mode 1 --no-prompt
```

Expected:
```
✔ Demo Complete! breakfast_cereal.stl picked via Simox and placed on the left!
```
