# Connecting Giskard to a Robot

This tutorial connects Giskard to a robot that is driven by
[ros2_control](https://control.ros.org), using a Universal Robots UR5e as the example.
No real arm is needed: the Universal Robots driver can run on mock hardware, which
integrates the commanded velocities and reports the result as joint states. The setup is
the same one the driver uses for a real arm.

## Prerequisites

- A ROS workspace as described in the
  [monorepo README](../../README.md#optional-setup-your-ros-workspace), sourced in every
  terminal used below.
- The Universal Robots driver: `sudo apt install ros-jazzy-ur-robot-driver`.

## What Giskard needs from a robot

Giskard is configured with four objects:

| Object | What it decides |
|---|---|
| `WorldConfig` | Which robot description is loaded and how the robot is attached to the world. |
| `RobotInterfaceConfig` | Whether a motion is only simulated (`StandAloneRobotInterfaceConfig`) or sent to the robot (a `ClosedLoopRobotInterfaceConfig`), and for a real robot where its state is read from and where commands are sent. |
| `GiskardServerConfig` | The goal lifecycle and the debug output of the motion server. |
| `QPControllerConfig` | The control frequency and the tuning of the controller. |

To command a robot, Giskard needs two things from it:

- a `joint_state_broadcaster/JointStateBroadcaster`, which publishes the joint states;
- a `velocity_controllers/JointGroupVelocityController`, which accepts joint velocities.

Both have to be *active*. Giskard asks the controller manager which controllers are
running and connects to these two kinds on its own, so the joints and topics do not have
to be listed by hand.

## The configuration

```{literalinclude} ../src/giskardpy/middleware/ros2/scripts/tutorial/universal_robot_velocity.py
:language: python
```

- `WorldWithFixedRobot` loads the robot description and fixes the robot to the world. It
  suits an arm that is bolted down. The description is read from the
  `/robot_description` topic, so Giskard uses the same model as the driver. The
  section "Where the robot description comes from" below says when to read it from a
  file instead.
- `ControllerManagerInterface` asks the controller manager for its active controllers.
  It reads the joint states from the broadcaster and sends velocities to the velocity
  controller.
  Because it is a `ClosedLoopRobotInterfaceConfig`, Giskard sends its commands to the
  robot in real time and controls only the joints a controller commands. Passing
  `StandAloneRobotInterfaceConfig()` instead simulates the motion without hardware.
- `target_frequency` is the control frequency in hertz.

Because nothing in this file names a joint or a topic, it works unchanged for any
fixed-base robot whose controller manager runs these two controllers.

## Robots with a robot class or several controller managers

A robot that has a robot class in `semantic_digital_twin.robots` already declares the
topics its parts publish their state on, and it may run more than one controller manager.
Tracy runs one per arm, each in its own namespace:

```{literalinclude} ../src/giskardpy/middleware/ros2/scripts/iai_robots/tracy/configs.py
:language: python
:pyobject: TracyVelocityInterface
```

- `sync_robot_parts()` reads the state from the topics the robot class declares.
- `add_velocity_controllers_from_controller_manager()` connects only to the velocity
  controllers, so the joint states are not read a second time. It takes the name of the
  controller manager including its namespace, for example `left_arm/controller_manager`.
- `whitelist` names the controllers Giskard may command. Without it Giskard commands
  every active velocity controller of the manager, including one that belongs to a
  gripper. With it, a listed controller that is not loaded or not active stops Giskard
  at startup instead of leaving its joints uncontrolled.

`discover_interfaces_from_controller_manager()`, used above, does both: it reads the
joint state broadcaster and connects to the velocity controllers. It takes the same
`whitelist`.

The world config of such a robot only names the robot class:

```{literalinclude} ../src/giskardpy/middleware/ros2/scripts/iai_robots/tracy/configs.py
:language: python
:pyobject: WorldWithTracyConfig
```

With a robot class Giskard avoids collisions and uses the velocity limits the class
declares.

## Robots with a mobile base

A robot that drives uses `WorldWithMobileRobot` instead of `WorldWithFixedRobot`. It adds
an odometry body below the root of the world and connects the robot to it with a drive.

| Field | What it decides |
|---|---|
| `drive_class` | `OmniDrive` (the default) or `DifferentialDrive`. |
| `root_name`, `odom_body_name` | The names of the root and the odometry body, `map` and `odom` by default. |
| `translation_velocity_limit`, `rotation_velocity_limit` | The velocity limits of the base. |

The interface then says where the pose of the base is read from and where its velocity is
sent. The HSR does it like this:

```{literalinclude} ../src/giskardpy/middleware/ros2/scripts/iai_robots/hsr/configs.py
:language: python
:pyobject: HSRVelocityInterface
```

- `sync_6dof_joint_with_tf_frame()` reads the localization, the pose of the odometry body
  in the root, from tf.
- `sync_robot_parts()` reads the joint states and the odometry from the topics the robot
  class declares. A robot without a robot class uses `sync_joint_state_topic()` and
  `sync_odometry_topic()` instead.
- `add_base_cmd_velocity()` sends the velocity of the drive as a `Twist`.
- `add_joint_velocity_group_controller()` sends joint velocities to a controller that is
  not found through a controller manager. The joints have to be listed in the order of
  the controller's command layout, because the command is an array without names.

Inputs are applied in the order `setup()` adds them, except the tf frames, which are
always applied first.

## Where the robot description comes from

- A script that commands a robot reads the description from the `/robot_description`
  topic with `get_robot_description()`. Giskard then uses the model the robot uses. It
  waits until the description is published.
- A script that simulates passes no `urdf` to its world config, which then loads the file
  its robot class declares, so nothing has to publish the description.

## Running it

Start the robot on mock hardware, with the velocity controller active:

```bash
ros2 launch ur_robot_driver ur_control.launch.py ur_type:=ur5e robot_ip:=0.0.0.0 \
    use_mock_hardware:=true initial_joint_controller:=forward_velocity_controller
```

Start Giskard in a second terminal. The script is an ordinary Python program:

```bash
python giskardpy/src/giskardpy/middleware/ros2/scripts/tutorial/universal_robot_velocity.py
```

Giskard reports the controller it connected to and then waits for goals:

```
Created publisher for /forward_velocity_controller/commands for ['shoulder_pan_joint', ...]
giskard is ready
```

If Giskard cannot set the robot up, for example because the controller manager is not
running, it exits with the error instead of waiting for goals.

Send a joint goal from a third terminal:

```bash
python giskardpy/src/giskardpy/middleware/ros2/scripts/tutorial/joint_goal_client.py
```

```{literalinclude} ../src/giskardpy/middleware/ros2/scripts/tutorial/joint_goal_client.py
:language: python
```

The arm moves in RViz, and `ros2 topic echo /joint_states` shows the two joints at their
goal positions.

## Starting Giskard from a launch file

Running the script with `python` is enough to get started. To start Giskard together with
the rest of a robot, register the `main` function of the script as a `console_scripts`
entry point of a ROS 2 package and start that executable from a launch file of the
package. The `giskardpy_ros` package in
[cram_ros2_packages](https://github.com/cram2/cram_ros2_packages) does this for the robots
it supports.

## Using a real arm

Start the driver with the address of the arm instead of `use_mock_hardware:=true`. The
Giskard configuration stays the same.

```{warning}
The robot in this tutorial is described only by its URDF. Giskard therefore does not avoid
collisions and limits every joint to a default velocity. Test motions on mock hardware
first.
```
