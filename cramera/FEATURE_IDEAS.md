# cramera feature ideas — "a cooler RViz"

Brainstorm, not a roadmap. Framing: cramera already has the three things RViz
does not — a semantic knowledge base with EQL, plan/statechart execution
status, and recorded playback. The interesting features are the ones that only
make sense *because* of those, not "RViz features we are missing".

## The differentiator: we know what things mean

RViz shows a `visualization_msgs/Marker` at a pose. We know it is `Milk`, that
it is `on` the table, and that a `Grasping` node is holding it.

- **Query-driven visibility.** The EQL console already selects entities — let a
  query *be* a display filter: `all bodies within 0.5m of the gripper` →
  those highlight, everything else dims. RViz has checkbox trees; this is a
  query language over the scene.
- **Symbolic overlays.** Draw the *relations*, not the geometry: `supported-by`
  as a line to the table, `reachable` as a shaded region, containment as a
  nested outline. This is what people fake today with a hundred hand-published
  markers.
- **Click anything → why.** Click a body, get the designator resolution that
  produced it, the plan node that touched it last, and the KB facts about it.
  All three sources are already wired into the event bus.

## Time is the second axis RViz throws away

RViz is a live firehose with no memory; `live/recording.py` already exists.

- **Scrub the world, not a bag.** The scrubber now marks every recorded step,
  pick and release, previews the frame under the pointer as a thumbnail, and
  jumps there on click (`core/timeline-events.js`). Still open: the plan tree
  itself as the ruler, so clicking `Transporting` seeks to its start.
- **Ghosting / trails.** An object's past poses as fading ghosts, TF frame
  trails, the gripper path. Cheap to render, large payoff for debugging
  manipulation.
- **Diff two runs.** Load run A and run B, overlay both robots, colour-code
  where the trajectories diverge and which plan node first differed. The
  killer feature for "why did it work yesterday".
- **Failure bookmarks.** Auto-mark every failed/interrupted node on the
  timeline so opening a recording drops you at the interesting moment instead
  of t=0.

## Things a browser can do that a Qt app cannot

- **Share a URL.** `?scene=x&t=12.4&focus=milk` reproduces exactly the current
  view — paste it into an issue or a paper. Should be cheap given the existing
  scene picker and panel config.
- **Multi-user.** Two people on one live bridge, each seeing the other's camera
  and selection. Remote debugging of a lab robot without a VPN and X
  forwarding.
- **Export.** Selected view → GLB / SVG / short video, for a thesis figure or a
  talk.

## Interaction beyond dragging poses

Dragged poses are already written back into the world; extend the same channel.

- **Pose goal by hand** — drag a 6-DOF handle, ship it to giskard, watch the
  plan react. RViz's interactive marker, but with the plan panel showing the
  consequence.
- **Joint sliders** bound to the live world (an in-scene
  `joint_state_publisher_gui`).
- **Poke the knowledge base** — assert a fact from the UI (`the drawer is
  open`) and watch the plan re-derive. Nothing in the ROS world does this.

## The boring-but-load-bearing gap list vs RViz

For someone to actually *replace* RViz with cramera, these have to exist. None
are exciting; all are the reason people keep RViz open in the other window.

- ~~TF tree view with staleness/age~~ — built as the Transforms tab
  (`live/transforms.py`): the world's connection graph, each frame ringed by how
  recently the connection carrying it moved, and labelled with who wrote it last,
  plus the in-scene axis triads of the *TF frames* layer (`core/frame-axes.js`)
- point cloud and depth image displays
- camera-image panels with overlays
- costmap / occupancy grid
- a measuring tool
- per-display transparency and wireframe

## If only three

1. Plan-tree-as-timeline scrubbing
2. Query-driven visibility
3. Shareable state URLs

They are the cheapest given the current architecture and the ones RViz
structurally cannot copy.
