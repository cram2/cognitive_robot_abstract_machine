# Tracy screwing demo

Tracy demonstrates joint-driven screwing using `ScrewJointAction`, `ScrewThreadAction`, and `InsertScrewAction`.

Two scenes are included:

- **`airplane` (default):** Tracy picks a `Bolt` from the holder, inserts it into the airplane's `RearWing` (with an optional initial hand turn), picks up a `Screwdriver`, and drives the bolt along its `ScrewConnection` thread.
- **`nut`:** Tracy directly grasps a `Bolt` mounted on a stationary `Nut` plate and turns it by hand.

## Running it

Run either runner **from inside this folder**:

| Command | Mode | Description |
|---|---|---|
| `python demo.py` | Headless kinematic | Runs the `airplane` scene without RViz. |
| `python demo.py --scene nut` | Headless kinematic | Runs the `nut` scene without RViz. |
| `python rviz_demo.py --real-time` | ROS 2 + RViz2 | Publishes TF and a `MarkerArray` on `/semworld/viz_marker` (set durability to transient local) in wall-clock real time. |

Pass `--no-turn-by-hand` on the `airplane` scene to insert the bolt without the initial hand turn before tightening with the screwdriver.

## File layout

| File | Role |
|---|---|
| `demo.py` | `NutScene`, `AirplaneScene`, `ScrewingScene`, and `HeadlessRunner` (no ROS visualization publishers). |
| `rviz_demo.py` | `RvizRunner` publishing TF and markers for RViz2. |
| `resources/` | Battat toy airplane, nut, bolt, and holder STL meshes. |
