# PR2 Demos

This folder contains PR2-specific demo scripts for the CoraPlex pick-and-place pipeline.

## Available Demos

| Script | Description |
|---|---|
| `pr2_giskard_pick_place_demo.py` | Full pick-and-place: park → navigate → grasp → carry → place → retract |
| `pr2_giskard_arm_demo.py` | Arm parking demo (simpler, good for first tests) |
| `pr2_giskard_both_arms_demo.py` | Both arms movement demo |

## How to Run

```bash
source /opt/ros/jazzy/setup.bash
export ROS_DOMAIN_ID=1
workon cram-env
python coraplex/demos/pr2/pr2_giskard_pick_place_demo.py
```

See [docs/03_running.md](../../../docs/03_running.md) for full instructions.
