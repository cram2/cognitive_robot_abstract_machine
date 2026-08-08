# cram_viz — CRAM Visualization

Browser-based visualization for the CRAM architecture — one tool for two modes:

- **Recorded**: run any coraplex demo once through the onboarder and get a
  lightweight, self-contained 3D scene (URDFs + meshes + the real recorded
  giskardpy trajectory) that plays in any browser, no ROS required.
- **Live** (RViz replacement): attach the viewer to a *running* demo — it
  renders the executing world in real time, and dragging objects in the viewer
  writes their pose back into the demo's world.

## Quick start

```bash
cram-viz                                  # serves http://localhost:8711
cram-viz-onboard path/to/demo.py --name my_scene    # record a demo once
cram-viz-live path/to/demo.py             # run a demo with the live bridge
```

Scene bundles are **generated artifacts** (tens of MB per scene) and are not
part of this repository — ready-made demo recordings live in
[cram2/cram-scenes](https://github.com/cram2/cram-scenes), wired as an
**optional** submodule:

```bash
git submodule update --init cram_viz/scenes    # ready-made demo scenes (optional)
```

The viewer looks for bundles in this order: `CRAM_VIZ_SCENES=/path` (env
override) → the initialized submodule `cram_viz/scenes` → `~/.cram_viz/scenes`
(where the onboarder writes by default). Live visualization and freshly
onboarded scenes need none of the ready-made bundles. Select a scene with
`?scene=<name>` or `CRAM_VIZ_SCENE=<name>`.

## Live mode

Two ways to attach the live bridge to a running coraplex demo:

1. As the run wrapper (e.g. a PyCharm run configuration):

   ```bash
   cram-viz-live path/to/demo.py
   ```

2. As a one-liner at the top of a demo file:

   ```python
   from cram_viz.live.runner import start; start()
   ```

Either way an HTTP bridge starts on port 8765 (`LIVE_VIZ_PORT` to change);
while it is reachable the viewer shows a *Live* button that renders the
running world instead of the recording, and dragging an object writes its
pose back into the demo's world.

## Panels — how the UI is composed

The frontend (`src/cram_viz/web/`) is a set of **panels** mounted into layout
slots. Which panel appears where is decided by **one file**:

```js
// web/config.js
window.CRAM_VIZ_CONFIG = {
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
| `robot-scene` | three.js scene: environment, robot, draggable objects, playback + live controls |
| `eql`         | EQL query console + entity answer panel                      |
| `graph`       | four tabs: Knowledge / Kinematics / Plan / Statechart        |

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

## Layout

```
src/cram_viz/
  server.py        static frontend + JSON API (/api/kb, /api/eql, /scenes/)
  knowledge/       scene-driven EQL knowledge base + graph payloads (package)
  paths.py         all filesystem locations (env-overridable)
  live/
    bridge.py      bridge state + serializers (runs on the sim thread)
    hooks.py       Executor/Plan/GiskardExecutable/mesh hooks
    http.py        the bridge's HTTP endpoints (port 8765)
    __main__.py    cram-viz-live entry point
  onboard/
    demo.py        demo -> scene bundle (record + bundle, one command)
    bundle_urdf.py standalone URDF/xacro asset bundler
  web/
    index.html     shell: topbar + slots + script includes
    config.js      which panels are shown where  ← edit this to swap panels
    core/          bus, panel registry, split/resize helper
    panels/        robot_scene/, eql/, graph/
    vendor/        three.js, vis-network, … (all local, no CDN)
```

## Tests

```bash
pytest test/cram_viz_test
```

The JS core (bus, registry, graph status rendering) is covered by node-based
tests invoked from pytest (skipped when node is unavailable).
