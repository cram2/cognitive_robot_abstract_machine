# PR2 CoraPlex demonstration

`pr2_demo.py` offers three plans through the shared `RobotDemonstration` framework:

| `--task` | Plan |
| --- | --- |
| `torso` (default) | Move the torso to the model’s HIGH and LOW configurations |
| `arms` | Park both arms using their semantic joint configurations |
| `pick-place` | Park, move the torso to MID, pick up the cereal with the right arm, place it 20 cm along the world’s y axis, and park again |

Pick-and-place uses CoraPlex `PickUpAction` and `PlaceAction`, including their grasp and attachment handling. The demo uses the existing PR2 navigation support.

The default apartment scene completes with the model’s MID torso configuration. HIGH causes the right elbow to reach its flexion limit during the pre-grasp approach from the default starting pose. When embedding the demo in Python, use `pick_and_place_torso_state` to select a configuration appropriate to the scene.

## Local simulation

Activate the workspace Python environment and source ROS 2 and the ROS workspace overlay. The PR2 model comes from `iai_pr2_description`; the pick-and-place scene uses the apartment and object resources supplied with CoraPlex.

From the repository root:

```bash
python coraplex/demos/pr2_ros_bridge_demo/pr2_demo.py --task torso --mode sim
python coraplex/demos/pr2_ros_bridge_demo/pr2_demo.py --task arms --mode sim
python coraplex/demos/pr2_ros_bridge_demo/pr2_demo.py --task pick-place --mode sim
```

`sim` selects `KinematicallySimulatedPlanExecutor`. It creates a local digital twin and uses the framework’s visualization lifecycle; it requires no robot or bridge. It does not simulate physical grasp contact. Collision avoidance is off by default; add `--collision-avoidance` to enable it.

## Controller execution through the bridge

`real` selects `RealRobotPlanExecutor`. The demo fetches the world from an already running ROS 2 Giskard server, synchronizes it, and sends the plan's statechart through the standard Giskard client. The shared demonstration framework passes its ROS node to the executor. The configured `ros_node_name` is consumed by `RobotDemonstration.acquire_world()` when starting that session.

### Where the ROS 1/2 bridge belongs

The bridge is external deployment infrastructure, configured in
[pr2-config-iai/ros-bridge-config](https://github.com/pr2-config-iai/ros-bridge-config).
This directory supplies the CoraPlex demo; it does not start bridge containers or
implement PR2 hardware controllers. The intended command path is:

```text
CoraPlex RealRobotPlanExecutor
    -> ROS 2 Giskard server
    -> configured PR2 controller interfaces
    -> ros1_bridge transport
    -> ROS 1 PR2 controllers
```

Joint-state and transform feedback returns through the bridge to the ROS 2
controller world. CoraPlex's Giskard client/server communication stays on ROS 2.
No bridge-specific execution enum or direct trajectory publisher is needed in
this demo. The same high-level plan can use a ROS 2 controller deployment without
the bridge.

The external repository includes experiments and older direct-command demos.
Those are not this demo's execution path. A topic appearing across the bridge
does not establish that Giskard can drive the hardware controller; its selected
command interface, message types, feedback and cancellation must be validated
together. Giskard-to-PR2 hardware integration remains a separate lab acceptance
step and is not established by the kinematic tests below.

For ROS 1 PR2 hardware or Gazebo, first configure the Giskard server’s PR2 controller interfaces and start the ROS 1/2 bridge with the required state and command topics. Starting the bridge alone does not provide a Giskard server or controller support. Hardware execution depends on those interfaces being operational.

```bash
python coraplex/demos/pr2_ros_bridge_demo/pr2_demo.py --task torso --mode real
```

This command executes the selected motion. For `--task pick-place`, the controller world must already contain the cereal body `breakfast_cereal.stl`, its measured pose, and the surrounding scene, with the robot positioned within reach. The demo annotates that existing body; it does not relocate the real robot or spawn an assumed object. Its body name and placement offset can be configured on `PR2Demonstration` when embedding it in Python.

## Validation

The focused tests check command-line import, the controller world/node connection, torso and arm plan construction, and object-relative grasp and placement handling. They also execute the full pick-and-place plan in kinematic simulation and check that the cereal is released at the requested pose:

```bash
python -m pytest test/coraplex_test/test_pr2_bridge_demo.py --orm-build never
```

The tests use the current executor/statechart APIs and retain the original
behavior checks. Command-line tests verify that `sim` and `real` select the
appropriate executor without executing hardware commands. Passing these focused
tests does not establish that the full CI matrix passes.

### Python API migration

The command-line options above are unchanged. Python callers now pass
`executor_type=KinematicallySimulatedPlanExecutor` or
`executor_type=RealRobotPlanExecutor`, rather than `execution_type=ExecutionType...`.
The demo implements `build_context_extensions()` and `build_statechart()` for the
shared framework, using `RobotAccess`, `StatementGrounding`, and a `Sequence` of
standard actions. World construction uses the framework's current specification
API. A controller pick-and-place still requires the measured cereal body before
scene population; it never inserts the simulated apartment into that world.

These checks do not establish successful motion or physical grasping on the PR2. Hardware and bridge integration require a separate run with the configured controller.
