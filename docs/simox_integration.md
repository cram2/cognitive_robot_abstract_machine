# External Simox grasp planning

The integration depends on the existing `grasp_planner_msgs/srv/PlanGrasp` ROS 2
interface. Simox libraries stay in the external service process; importing the
CoraPlex adapter does not require Simox or generated service messages.

## Data contract

`SimoxGraspPlanner` takes a `SimoxConfiguration` and a `SimoxService` implementation.
`RosSimoxService` borrows a node whose executor is already spinning; it does not
initialize or shut down ROS. Requests have bounded discovery and response waits.

The service fields are unstamped `geometry_msgs/Pose`. Transport uses the existing
`PoseToRos2Converter` and `PoseToSemDTConverter`, rather than duplicating their field
mapping or adding headers the service does not accept. Incoming translations must be
finite; quaternion values are validated and normalized on a copy before conversion.
Frame interpretation and tool calibration remain explicit in the planner.

`plan_grasps_for_body(graspable, arm, robot_frame)` snapshots the object's pose relative
to the explicit robot frame and exports `graspable.grasp_surface()` in the annotation
root frame. Temporary files remain available until the service call completes.
The server must share this filesystem.

ROS translations and exported STL coordinates are metres. The external Simox loader
converts STL coordinates to millimetres internally; multiplying the STL by 1000 first
would double-convert it. The external converter now preserves metre inputs and scales
millimetre inputs by 0.001. Its automatic unit guess is still a heuristic.

For every returned external tool pose, conversion follows:

```text
object_T_tool = inverse(base_T_object) * base_T_simox * simox_T_tool
```

The end effector's `tool_P_grasp_when_closed` and `tool_R_grasp` then provide the
standard grasp origin and axes. `GraspCandidate` stores that pose relative to the
object root, so it follows later object movement. `RankedGrasp` retains quality and
force-closure metadata; results are sorted by descending quality.

Malformed array lengths, non-finite poses/qualities and invalid quaternions are
rejected. No candidates, unavailable service and timeout raise explicit planning
errors. Quality is not a probability, and force closure is not a collision guarantee.

## PR2 demonstration

See [the demo instructions](../coraplex/demos/pr2_simox_demo/README.md).
The demo uses `RobotDemonstration`, standard pickup/place actions and deferred action
trials. There is no separate Simox manipulation action or Simox-specific grasp enum.
The PR2 default calibration is specific to the supplied external PR2 model.

The generic adapter has no default robot XML. Callers supply the robot model, external
end effector name, preshape, optional kinematic chain and tool calibration through
`SimoxConfiguration`, then pass the semantic arm and robot base frame to planning.
`SimoxConfiguration.for_pr2` is a convenience preset for this demo, not a requirement
of the service client. Other robot/model combinations require their own validation.

The demo uses the shared ROS-session lifecycle and standard motion actions; it has no
custom joint-action wrapper, direct gripper control or Docker-based hardware commands.
Its supported scenario is cereal pickup and placement in simulation. No bowl-specific
grasp correction or modified bowl resource is required. Real controller execution is
not demonstrated by this entry point.

The integration uses upstream `GraspCandidate`; it does not depend on the old
`feat/robot-centric-grasping` branch or its deleted `grasp.py` API.

## External service changes

The paired service checkout captures the force-closure measurement immediately after
each accepted grasp, preserves it through filtering and sorting, and returns it instead
of a hardcoded true value. The message schema is unchanged. The old service may still
return hardcoded flags, so deploy the paired service fix when relying on this metadata.
An explicitly requested but missing kinematic chain now returns an error.

The service does not receive a full environment or the robot's live joint state.
By default, CoraPlex handles reachability using its own world. Neither the metadata nor
successful simulated execution establishes physical robot safety or contact success.
