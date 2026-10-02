# PR2 Simox Grasp Planner + CoraPlex & Giskard End-to-End Demo

This demonstration showcases physics-based grasp planning using Simox integrated seamlessly into the CoraPlex / Semantic Digital Twin cognitive architecture, executed with Giskard collision-aware motion planning.

---

## Architecture Overview

```
CoraPlex World (PR2 + apartment scene + cereal box)
       │
       ▼
SimoxPickUpAction (coraplex/robot_plans/actions/core/pick_up.py)
       │
       ├─► simox_grasp_planner.py (External Interface)
       │       │  Transforms object pose from world root to PR2 base frame
       │       ▼
       │   Simox ROS 2 Service (/plan_grasp)
       │       - Surface normal approach sampling
       │       - Collision checking against PR2 kinematic model
       │       - Force-closure wrench resistance scoring
       │       ▼
       │   Ranked Candidate Grasp Poses returned to CoraPlex
       │
       ├─► Giskard Motion Planner (simulated_robot)
       │       - Reaches top candidate grasp pose
       │       - Closes PR2 gripper
       │       - Attaches object to gripper tool frame
       │       - Lifts object (+15 cm)
       │
       └─► RViz2 Visualisation (/semworld/viz_marker & TF)
```

---

## How to Run

### Step 1: Start the Simox Grasp Planner Service
In **Terminal 1**:
```bash
cd /grasp_planner
bash scripts/launch_simox_service.sh
```

### Step 2: (Optional) Open RViz2 for 3D Visualization
In **Terminal 2**:
```bash
source /opt/ros/jazzy/setup.bash
rviz2
```
In RViz2:
- Set **Fixed Frame** to `world` (or `apartment/apartment_root`).
- Add display: **TF**.
- Add display: **MarkerArray**, set Topic to `/semworld/viz_marker`, Durability to `Transient Local`.

### Step 3: Run the End-to-End Demo
In **Terminal 3**:
```bash
cd /grasp_planner
source /opt/ros/jazzy/setup.bash
source install/setup.bash
PYTHONPATH=cognitive_robot_abstract_machine/coraplex/src:$PYTHONPATH     ~/.virtualenvs/cram-env/bin/python cognitive_robot_abstract_machine/coraplex/demos/pr2_simox_demo/demo.py --spin
```

*(Note: pass `--spin` to keep the RViz markers and robot state alive for inspection until you press `Ctrl+C`)*
