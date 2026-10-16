# Tracy montessori demo

Tracy picks the four Montessori pieces (cube, cylinder, rectangular prism, triangular
prism) off its own table and drops each through its hole in the shape-sorting board. The scene, the poses and the plan are the same however it runs;
only how the plan is carried out differs, by coraplex's `ExecutionType`: physically
simulated in MuJoCo, kinematically simulated in RViz, or on the real robot.

The pieces use the sizes and colours measured off the
real set. There is no perception: where the board and the pieces start is hardcoded.

In both simulations Tracy starts with both arms parked: `get_world()` applies the `PARK`
joint states of `TracyLeftArm` and `TracyRightArm`, and both grippers start open. The plan
still begins with `ParkArmsAction`, because the real robot starts wherever it was left.

Branch: `tracy-montessori-demo`, off `cram2/main` at `746302cd9b`.

## Running it

Choose the execution type at the bottom of `demo.py`:

```python
main(ExecutionType.PHYSICALLY_SIMULATED)  # or KINEMATICALLY_SIMULATED, or REAL
```

Then, **from inside this folder**:

```bash
python demo.py
```


| Execution type | What you need | What you see |
|---|---|---|
| `KINEMATICALLY_SIMULATED` | ROS 2, RViz2 | A `MarkerArray` display on `/semworld/viz_marker`, durability set to transient local. The robot is moved kinematically. |
| `PHYSICALLY_SIMULATED` | `mujoco` | A MuJoCo viewer. Pass `headless=True` to the `MujocoSim` in `mujoco_demo.py` to run without one. The viewer stays open after the plan ends until you close it. |
| `REAL` | The robot, its ROS stack | It launches `giskardpy_tracy_standalone.launch.py` itself and fetches the world from the running stack. |

## File layout

| File | Role |
|---|---|
| `demo.py` | The entry point. Holds `MontessoriEnvironment` — every hardcoded pose and size as a field, the board and pieces it stands in a world, and the plan that sorts them — and the dispatch on the execution type. |
| `montessori_board.py` | The board: its holes read off the mesh, and a body with the mesh to look at and boxes to collide with. |
| `resources/board.stl` | The board's mesh. |
| `rviz_demo.py` | Kinematic run, published for RViz. |
| `mujoco_demo.py` | Physics run, controller and physics stepped in lockstep. |
| `real_demo.py` | Real robot, through giskard. |

The runners do not import `demo`; `demo` builds the scene and hands it down. That is what
keeps the import from being circular. Each runner's `run()` takes the
`MontessoriEnvironment` and the world it stood the scene in, and asks the environment for
the plan; the runners import the type under a `TYPE_CHECKING` guard, which costs nothing
at runtime. The real robot's world only exists once its stack is up, so
`real_demo.running_robot()` brings the stack up first and the environment `populate`s the
world it fetched; the simulated runs let `get_world()` read Tracy from its own
description.

## What the plan does

```
ParkArmsAction(all arms)
for each piece, with the hole it was made for:
    PickUpAction(the piece's grasp, left arm)
    PlaceAction(piece, 1 cm above that hole, turned to match it)
ParkArmsAction(all arms)
```

Every piece is a `MontessoriPiece` (`semantic_digital_twin`), which offers one
`GraspCandidate` derived from its own geometry: approached from above, the fingers closing
along the piece's own x, `grasp_height` above its bottom face. The environment sets that
height from how far Tracy's Robotiq pads swing down as they close. Which hole each piece
goes into is a plain mapping the environment keeps as it makes the pieces. The left arm
sorts, as in every Tracy demo in this repository.

## Where the numbers come from

`table_top_z = 0.88` is measured, not guessed. Tracy's kinematic root *is* its table, and
that body carries two collision boxes:

- a `Scale(1.18, 1.6, 0.855)` slab centred at local `(0.59, 0, -0.4275)`, whose top lands
  at exactly `z = 0.88` in world coordinates — this is the work surface;
- a small `0.2 × 0.2 × 0.25` mount at local `(0.13, 0, 0.125)`, top at `z = 1.13`.

The usable surface runs `x` from 0 to 1.18 and `y` from -0.8 to 0.8, with that mount
occupying roughly `x` 0.03 to 0.23. The layout follows the ICRA branch's
`TracyMontessoriWorld`:

| | x | y | z |
|---|---|---|---|
| pieces (start) | 0.55 | 0.1 / 0.2 / 0.3 / 0.4 | on the table |
| board (centre) | 0.85 | 0.0 | on the table, 8 cm tall, drawers facing the robot |
| release poses | over each hole | | 1 cm above the lid |

Holes, as read off the mesh, in the board's frame:

| hole | centre (x, y) | size | piece |
|---|---|---|---|
| circle | -0.0215, -0.087 | Ø 32 mm | cylinder, Ø 28 mm |
| square | 0.0205, -0.087 | 32 × 32 mm | cube, 30 mm |
| triangle | 0.0125, 0.0 | side ≈ 42 mm, apex +x | triangular prism, side 37 mm |
| rectangle | -0.0265, 0.0 | 22 × 42 mm | rectangular prism, 20 × 40 mm |
| circle | -0.0175, 0.091 | Ø 40 mm | none |
| slot | 0.025, 0.091 | 5 × 48 mm | none (no disk in the set) |

The triangular prism is modelled with its apex along its own +y, so the fingers close on
one flat face and the opposite edge, and it is turned -90° when released. MuJoCo collides
with the convex hull of a mesh, which would close every hole, so the board's collision
geometry is boxes: the walls and the base are boxes in the mesh already, and the lid is
tiled around the holes' bounding boxes. The drawer handles are only drawn.

**These poses are a first pass.** They were measured and then confirmed by running, but
they have not been tuned for margin. Treat them as a starting point.

## Running under physics

`mujoco_demo.py` performs the plan inside coraplex's `PhysicallySimulatedRobot`
(`coraplex/src/coraplex/execution_environment.py`).

## What has actually been run

| | Result |
|---|---|
| Kinematically simulated | Runs clean. All four pieces reach their release poses over their holes. |
| Physically simulated (headless) | Runs clean. All four pieces drop through their holes into the drawers. |
| Real | **Never run.** No hardware was available. |

Final positions from the physically simulated run (the drawer floor is at z = 0.888, the
lid top at 0.96):

| piece | ended at | release pose | |
|---|---|---|---|
| square_piece | `0.873, -0.087, 0.908` | `0.870, -0.087, 0.985` | through its hole |
| circle_piece | `0.825, -0.088, 0.903` | `0.828, -0.087, 0.985` | through its hole |
| rectangle_piece | `0.824, 0.001, 0.903` | `0.823, 0.000, 0.985` | through its hole |
| triangle_piece | `0.867, 0.004, 0.903` | `0.869, 0.000, 0.985` | through its hole |

Every body is made solid where it is created (`MontessoriEnvironment.solid`): Tracy's in
`get_world()`, the board's and the pieces' in `populate()`. Their contacts resolve within
2 ms and barely give way, where MuJoCo's soft defaults let a held piece sink into the
pads. Making only the pieces and the board solid is not enough — a contact is as soft as
the softer shape in it, and three of the four pieces then catch on the lid. Of the settings
in `mujoco_demo.py`, removing the friction cone or the no-slip iterations alone costs the
square piece. Which pieces land is
sensitive to small changes — renaming the bodies alone moved one combination of settings
from four pieces in to two — so judge a change over more than one configuration. A run of
one configuration is deterministic: repeating it gives the same positions.