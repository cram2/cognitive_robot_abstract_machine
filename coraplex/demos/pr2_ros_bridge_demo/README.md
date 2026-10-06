# PR2 ROS Bridge Demos

This folder contains PR2-specific demonstration scripts for the CoraPlex manipulation pipeline running over the Docker-based ROS 1/2 bridge.

## Available Demos

| Script | Description |
|---|---|
| `pr2_giskard_pick_place_demo.py` | Full pick-and-place: park → navigate → grasp → carry → place → retract |
| `pr2_giskard_both_arms_demo.py` | Both arms coordinated movement demo |
| `pr2_torso_demo.py` | Torso lift and spine motion control |
| `pr2_full_pycram_demo.py` | Full manipulation demo with PyCRAM plan integration |
| `pr2_pycram_sim_demo.py` | PyCRAM simulation demo |
| `pr2_real_robot_demo.py` | Real robot execution demo via hardware bridge |

## How to Run

1. Ensure the ROS 1/2 bridge Docker container is running.
2. In your terminal:

```bash
source /opt/ros/jazzy/setup.bash
python coraplex/demos/pr2_ros_bridge_demo/pr2_giskard_pick_place_demo.py
```
