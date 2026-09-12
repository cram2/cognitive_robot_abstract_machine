# Tracy MuJoCo add-ons

The clutter-picking demo drives Tracy by commanding MuJoCo actuators directly rather than
through Giskard's live closed loop. Giskard's own QP control loop reads `world.state`
as its belief of the robot's current position, but for a physically simulated degree
of freedom that same state is also written by Giskard's own prior command. Giskard can
therefore be satisfied by its own prior write rather than by the robot actually having
moved (see the module docstring of `equipment.py`). Instead, every reach is planned by
Giskard against an isolated scratch copy of the world (`trajectory_planning.py`), and
the resulting trajectory is played back by commanding real MuJoCo actuators
(`real_time_simulation.py`).

## Layout

- `equipment.py` parses and mounts Tracy (`parse_tracy`, `mount_stationary_robot`,
  `tracy_table_mount_position`), equips it with position servos (`TracyServoTuning`,
  `ServoGains`, `equip_arms_with_servos`, `equip_grippers_with_servos`), adds gravity
  compensation and self-collision exclusion (`CollisionGroup`), and places loose boxes
  (`add_box`, `add_cube`).
- `grasp_contact.py` holds `ContactParameters`: MuJoCo friction and solver settings for
  a grasped object, a resting surface, or a plain cube, applied to bodies.
- `real_time_simulation.py` holds `RealTimeSimulation`, a MuJoCo mirror of a world
  stepped from the calling thread rather than MuJoCo's own background thread, whose
  reads would otherwise race a caller's own. It is paced to the wall clock or, with
  `real_time_factor=None`, runs as fast as the machine allows.
- `trajectory_planning.py` holds `TrajectoryPlanner`: plan a Cartesian or joint goal
  against a scratch copy of the world and play it back on the real, physically
  simulated one; park the arms, open or close a gripper, or close it around an object
  sized to the object's width. `RobotiqGripper` names the gripper's own bodies and
  joints and finds the knuckle angle for a given opening.
- `pick_and_place_action.py` holds `PickUpActionMujoco` and `PlaceActionMujoco`, which
  match the field interface of the real `PickUpAction` and `PlaceAction` but are driven
  by the planner above instead of a Giskard motion mapping. `TopDownGraspGeometry`
  places the tool frame so that the fingers, not the tool frame, meet the object.

A grasped object is held by real contact friction between the fingers throughout and is
never kinematically attached, so a poor grasp visibly fails instead of being rescued by
a weld.

The ten-carton clutter demo built on this is the enclosing
`tracy_clutter_picking` package.
