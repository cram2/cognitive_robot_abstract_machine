# PR2 ROS Bridge & Giskard Demos

This directory contains PR2 demonstration scripts for the CoraPlex manipulation pipeline integrated with Giskard motion planning and the ROS 1/2 bridge.

## Architecture

In the CoraPlex architecture:
- **CoraPlex** defines high-level symbolic actions and plans (`ParkArmsAction`, `MoveTorsoAction`, `MoveManipulatorAction`).
- **Giskard** acts as the whole-body controller and collision-aware motion planner, generating coordinated trajectories and joint commands.
- The **ROS 1/2 Bridge** (`ros-bridge-config`) is a passive communication layer forwarding standard ROS topics (e.g., `/joint_states`, `/tf`, controller commands) between ROS 2 Jazzy and ROS 1 Noetic (PR2 hardware or Gazebo simulation).
- Demos support both `--mode sim` (local digital twin simulation via QP solver) and `--mode real` (delegates execution to the Giskard action server).

## Available Demos

| Script | Description |
|---|---|
| `pr2_full_pycram_demo.py` | PyCRAM manipulation demo using the `RobotDemonstration` base class |
| `pr2_giskard_both_arms_demo.py` | Dual-arm coordinated parking and planning using semantic annotations |
| `pr2_giskard_pick_place_demo.py` | Full collision-aware pick-and-place manipulation (park → approach → grasp → lift → place → release → park) |
| `pr2_real_robot_demo.py` | Hardware inspection of `/joint_states` and safe torso motion execution |

## How to Run

### Simulation Mode (default, no robot or bridge required)

```bash
# Full PyCRAM demonstration
python coraplex/demos/pr2_ros_bridge_demo/pr2_full_pycram_demo.py --mode sim

# Dual-arm coordinated motion
python coraplex/demos/pr2_ros_bridge_demo/pr2_giskard_both_arms_demo.py --mode sim

# Pick-and-place manipulation (add --rviz to publish MarkerArray)
python coraplex/demos/pr2_ros_bridge_demo/pr2_giskard_pick_place_demo.py --mode sim

# State and torso demo
python coraplex/demos/pr2_ros_bridge_demo/pr2_real_robot_demo.py --mode sim
```

### Real Robot / Hardware Bridge Mode

1. Start the ROS 1/2 bridge container in `ros-bridge-config`:
   ```bash
   docker compose up -d bridge
   ```

2. Run the demo specifying `--mode real`:
   ```bash
   python coraplex/demos/pr2_ros_bridge_demo/pr2_giskard_pick_place_demo.py --mode real
   ```
