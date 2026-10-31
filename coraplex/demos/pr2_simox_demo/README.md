# PR2 Simox demonstration

Simox runs as an external ROS 2 `/plan_grasp` service. This demo exports the cereal's
actual geometry, requests ranked grasps, converts them to upstream `GraspCandidate`
objects, and uses the standard `PickUpAction` and `PlaceAction` in simulation.
The upstream `PickAndPlaceAction` trials the complete pickup and placement before
selecting a grasp, rejecting grasps that cannot complete either step.

## Run

Start the separately built `grasp_planner_service` on the same machine. Source ROS 2,
the workspace containing CoraPlex, and the service workspace containing
`grasp_planner_msgs` in both terminals. With the appropriate Python environment active:

```bash
ros2 run grasp_planner_service grasp_planner_service_node
```

In another terminal, from the monorepo root:

```bash
python coraplex/demos/pr2_simox_demo/demo.py \
  --robot-xml /absolute/path/to/grasp_test_files/resources/robots/pr2.xml \
  --num-grasps 15 --timeout-ms 30000
```

The robot XML must be the PR2 Simox model whose right tool joint is at
`xyz="0.13 0 0"`, `rpy="0 1.5707963 0"`. The adapter's calibration maps this to
CoraPlex's PR2 tool frame. Other models require an explicit `SimoxConfiguration`
with their own calibration; do not reuse this default without checking the model.

The client and service must share a filesystem: the service reads temporary XML/STL
files by absolute path. Geometry and ROS positions are in metres. The Simox STL
loader converts mesh coordinates to its internal millimetres.

This replaces the previous demo CLI, custom Simox pickup action, and pose replay.
The command above runs simulation only. It is not a real-robot execution command.

## Motion and ROS ownership

The demo composes upstream `ParkArmsAction`, `MoveTorsoAction` and
`PickAndPlaceAction`. Arm joint targets come from the robot's semantic configurations.
Pickup and placement use the standard `MoveGripperMotion` with `GripperState`;
the demo has no joint lists, gripper gap overrides, bowl tilt tuning or Docker motion
commands. Grasp orientation comes from Simox and the configured tool calibration.

`RobotDemonstrationRosSession` owns the client node and its executor. The planner
borrows that node, and `RobotDemonstration` owns the visualization lifecycle. This
uses the shared demonstration framework rather than starting another Giskard `rospy`
lifecycle inside the demo. The transport can also borrow a node owned by another
application, provided its executor is already spinning.

This entry point demonstrates cereal manipulation in simulation. It does not preserve
the old bowl-specific sequence or Docker-based hardware replay. A physical robot run
needs the standard Giskard controller path and measured scene; that integration is
outside this demo and has not been validated here. The repository's existing bowl mesh
is unchanged by the Simox integration.

## Tests

```bash
python -m pytest test/coraplex_test/test_simox_candidates.py \
  test/coraplex_test/test_simox_grasp_planner.py \
  test/coraplex_test/test_simox_demo.py --orm-build never -q
```

These tests mock the external service and include a complete simulated pick-and-place.
They do not require a running Simox server. Cap memory and use the project's configured
interpreter as described in `AGENTS.md`.

## Limits

Simox quality and measured force-closure values are retained alongside each candidate.
They do not certify a collision-free trajectory. The default configuration leaves
Simox's kinematic chain empty because the service does not receive the current robot
joint state; CoraPlex tests reachability against its current simulated world.
The external service's existing collision checks do not contain the apartment scene,
and its legacy “collision-free” log message should not be treated as a guarantee.
Physical robot execution and contact behavior remain unverified.

## RViz

Start `rviz2` in the same ROS domain before running the demo. Set Fixed Frame to
`apartment/apartment_root` and add a MarkerArray display on `/semworld/viz_marker`.
The demo stops its visualization publisher and executor when it finishes or fails.
