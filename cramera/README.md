# cramera — CRAM Visualization

Browser-based visualization for the CRAM architecture — one tool for two modes:

- **Recorded**: run any coraplex demo once through the onboarder and get a
  lightweight, self-contained 3D scene (URDFs + meshes + the real recorded
  giskardpy trajectory) that plays in any browser, no ROS required.
- **Live** (RViz replacement): attach the viewer to a *running* demo — it
  renders the executing world in real time, and dragging objects in the viewer
  writes their pose back into the demo's world.

## Quick start

```bash
cramera                                  # serves http://localhost:8711
cramera-onboard path/to/demo.py --name my_scene    # record a demo once
cramera-live path/to/demo.py             # run a demo with the live bridge
```

Scene bundles are **generated artifacts** (tens of MB per scene) and are not
part of this repository — ready-made demo recordings live in
[cram2/cram-scenes](https://github.com/cram2/cram-scenes), wired as an
**optional** submodule:

```bash
git submodule update --init cramera/scenes    # ready-made demo scenes (optional)
```

The viewer looks for bundles in this order: `CRAMERA_SCENES=/path` (env
override) → the initialized submodule `cramera/scenes` → `~/.cramera/scenes`
(where the onboarder writes by default). Live visualization and freshly
onboarded scenes need none of the ready-made bundles. Select a scene with
`?scene=<name>` or `CRAMERA_SCENE=<name>`.

## Live mode

Two ways to attach the live bridge to a running coraplex demo:

1. As the run wrapper (e.g. a PyCharm run configuration):

   ```bash
   cramera-live path/to/demo.py
   ```

2. As a one-liner at the top of a demo file:

   ```python
   from cramera.live.runner import start; start()
   ```

Either way an HTTP bridge starts on port 8765 (`LIVE_VIZ_PORT` to change);
while it is reachable the viewer shows a *Live* button that renders the
running world instead of the recording, and dragging an object writes its
pose back into the demo's world.

## Panels — how the UI is composed

The frontend (`src/cramera/web/`) is a set of **panels** mounted into layout
slots. Which panel appears where is decided by **one file**:

```js
// web/config.js
window.CRAMERA_CONFIG = {
  layout: {
    left:  ['robot-scene'],
    right: ['eql', 'graph'],
  },
};
```

Removing a visualization = deleting its id here. Adding your own:

1. create `web/panels/<name>/panel.js`:

   ```js
   Panels.define('my-panel', function (root, bus) {
     root.innerHTML = '<div class="panel-head"><h2>My panel</h2></div>…';
     bus.on('entity:highlight', function (p) { /* react */ });
     bus.emit('entity:select', { id: 'x', detail: {…}, relations: [] });
   });
   ```

2. include the script in `web/index.html`,
3. add the id to `config.js`.

Panels **never call each other directly** — they publish/subscribe on the
event bus (`web/core/bus.js`), so any subset of panels works. The contract
between the built-in panels:

| event                | payload                          | emitted by → consumed by       |
| -------------------- | -------------------------------- | ------------------------------ |
| `scene:part-clicked` | `{id}`                           | robot-scene → eql              |
| `scene:step`         | `{step}` (`'__done__'` at end)   | robot-scene → eql, graph       |
| `live:changed`       | `{on, url}`                      | robot-scene → graph            |
| `entity:highlight`   | `{ids, focus?}`                  | eql → robot-scene, graph       |
| `entity:select`      | `{id, detail, relations}`        | graph → eql                    |
| `kb:ready`           | `{payload}`                      | eql → anyone                   |

### Built-in panels

| panel         | shows                                                        |
| ------------- | ------------------------------------------------------------ |
| `robot-scene` | three.js scene: environment, robot, draggable objects, TF frame axes, playback with key-event marks + live controls |
| `eql`         | EQL query console + entity answer panel                      |
| `graph`       | five tabs: Knowledge / Kinematics / Plan / Statechart / Transforms |

On the Plan and Statechart tabs the node border is its execution status —
running (amber), succeeded/done (green), failed (red), paused (blue),
interrupted (orange), not started (dim, dashed) — streamed live from the
bridge while attached.

Two things to know about those statuses: coraplex performs only the plan
**root** (`Plan.perform` → `root.perform`), while `ActionNode.notify` merely
expands its children into one merged motion statechart. So a *recorded* plan
tree has real status on the root only. Live, the bridge derives per-step
status from the statechart life cycle via `GiskardExecutable.motion_mappings`
(`{MotionNode: Task}`) and propagates it up the tree; those nodes are flagged
`derived`. Statecharts exist only during execution, so the Statechart tab is
live-only.

The replay scrubber marks the recording's key moments — every plan step, and the
frame each transported object was picked up and let go. Those are the bundle's
`segments`: an onboarded scene derives them from the recorded action list, a
live recording from what each tick captured (see `live/recording_segments.py`,
which reads the carry windows off the recorded poses and names each stretch
after the action the plan reported performing at the time). Hovering a mark previews what the scene looks like
there: the frame is rendered off-screen into a thumbnail, captioned with the
moment and its run time, and clicking the mark jumps the playhead to it.

The scene panel draws the same frames in 3D: the *TF frames* layer puts an axis
triad (red X, green Y, blue Z) on every URDF link and every loose object. A
triad is a child of the frame's own object, so it follows both recorded
playback and the live world without any pose plumbing. Its gear opens the size
slider, the frame-name toggle and the frame tree: a row per source (each loaded
model, plus the loose objects) that drops down into the frames under it, so a
whole model is ticked at once or single frames are picked out below it. A source
whose frames are partly on reads as partly ticked. The choices are remembered
between visits, and the arms ignore depth so a frame inside its own mesh stays
visible.

The Transforms tab is the world's connection graph — one node per frame, one
edge per connection — and is live-only for the same reason: which frame hangs
from which, and when each of those transforms was last written, only exists
while a demo runs. A frame's ring is the freshness of the connection carrying
it: moving now (amber), moved just now (green), not written for a while (dim,
dashed) or fixed and unable to move at all. Each frame also reports who wrote
it last, so a pose the demo drove is told apart from one dragged in the viewer.

## Layout

```
src/cramera/
  server.py        static frontend + JSON API (/api/knowledge, /api/eql, /scenes/)
  paths.py         all filesystem locations (env-overridable)
  knowledge/       the recorded scene as an EQL knowledge base
    knowledge_base.py  the entity lists one scene bundle yields
    eql_session.py     evaluating one EQL query string
    graph_payload.py   the knowledge graph the UI draws
    presets.py         the ready-made queries the panel offers
    views/             the graph-panel tabs and their drill-downs
  live/            stream a running coraplex demo into the viewer
    bridge.py      bridge state + serializers (runs on the sim thread)
    hooks.py       Executor/Plan/GiskardExecutable/mesh hooks
    http.py        the bridge's HTTP endpoints (port 8765)
    recording_segments.py  the steps and manipulations a recording went through
    transforms.py  the connection graph: what moved, when, and who wrote it
    __main__.py    cramera-live entry point
  onboard/         turn a demo run into a scene bundle
    demo.py        demo -> scene bundle (record + bundle, one command)
    bundle_urdf.py standalone URDF/xacro asset bundler
  web/
    index.html     shell: topbar + slots + script includes
    config.js      which panels are shown where  ← edit this to swap panels
    core/          bus, panel registry, split/resize helper, frame-axes display
                   state, timeline key events
    panels/        robot_scene/, eql/, graph/
    vendor/        three.js, vis-network, … (all local, no CDN)
```

## Tests

```bash
pytest test/cramera_test
```

The JS core (bus, registry, graph status rendering) is covered by node-based
tests invoked from pytest (skipped when node is unavailable).
