# segmind-live-1-numeric-reads

Where this branch stands, written on 2026-09-17 at `78657a8f04` so the work can be picked
up again later. Delete this file before the branch is merged.

## What the branch is for

SegMind detecting what happens in a live world while a plan runs, made correct for **one
demo**: the PR2 bullet world demo, `coraplex/demos/coraplex_bullet_world_demo/demo.py`.
The robot transports three objects — milk, bowl, spoon — onto `table_area_main`; the
cereal box stands still.

Work for every other demo (Tracy, Stretch, the four tool demos) was deliberately
removed to keep this branch about one demo. See "Other demos" below for what was learned
there and has to be rebuilt.

The first three commits (`0123ed83fc`, `ce4777512b`, `a1d6d7a1af`) make SegMind read the
world as plain numbers instead of building CasADi expressions every tick, which is what
lets it run on a thread of its own beside a plan.

## Where it stands

- **Pushed** to `origin` up to `423ed2cb4a`. The two commits after it, `4eef76093d`
  and `78657a8f04`, are **local only**. This README itself is not committed.
- **Demo**, with the detectors it asks for (`PickUpDetector`, `PlacingDetector`,
  `ContainmentDetector`, `GraspDetector`): all 18 ground-truth events found, 54 events
  detected in all, 36 beyond the ground truth. The log is **not yet clean**: the bowl is
  picked up twice, and both the milk and the bowl get a placing on the countertop they
  were lifted from — see the first open question.
- **Tests** (last runs): SegMind 94 passed, 1 skipped. semantic_digital_twin
  `test_worlds` 257 passed, `test_semantic_annotations` 154 passed. coraplex 497
  passed, 7 skipped. semantic_digital_twin and coraplex were last run before the
  SegMind-only commits; nothing in either has changed since apart from the demo script.

## Commits, oldest first

| Commit | What it did |
|---|---|
| `0123ed83fc` | A pose and a patched method read without building a symbolic expression |
| `ce4777512b` | Support, containment and regions answered from numbers |
| `a1d6d7a1af` | SegMind's existing detectors read the world without CasADi |
| `cccaa36e7d` | **semantic_digital_twin:** a body rests on what touches it from below (see "How detection works") |
| `b86b8e22a7` | **coraplex:** the demo's objects stand where they rest instead of hovering or sunk in; the demo prints its event log |
| `09b2610b28` | **segmind:** grasp detection, pick-up and placing read from grasps, `LiveSegmenter`, `scripts/report_demo_events.py` |
| `e914e45026` | The live event dashboard, taken from `segmind-live-5-dashboard` |
| `450067f74e` | `exclude_robot`: the robot is left out of what a run checks objects against |
| `defd6e46b3` | `SHOW_LIVE_EVENTS` field in the demo switches the dashboard on and off |
| `cd135a7cae` | Pick-ups and placings read from grasps only where a grasp detector runs |
| `12748b1cf4` | The dashboard leaves contacts and motions off the page |
| `423ed2cb4a` | A grasp needs **both** fingers to touch — see the note below |
| `4eef76093d` | A run is built from the detectors it asks for, each bringing what it needs |
| `78657a8f04` | The bullet world demo asks for grasping |

### The message of `423ed2cb4a` does not match its contents

It is titled *"added Insertion imagination through Mujoco for Segmind to detect"*, but
it contains only the two-sided grasp rule
(`segmind/src/segmind/detectors/agent_event_detector_nodes.py`) and its test
(`test/segmind_test/test_detectors/test_grasps.py`). There is no insertion or MuJoCo code
on this branch, committed or not. If insertion imagination was meant to go in, it is
not here. The commit is already pushed, so correcting the message means amending it and
force-pushing.

## Running the demo

```bash
.venv/bin/python coraplex/demos/coraplex_bullet_world_demo/demo.py
```

- `DETECTORS` at the top of the demo says what SegMind is asked to detect. The demo
  prints the full list of detectors that actually run when it starts.
- `SHOW_LIVE_EVENTS = True` serves a live page at <http://127.0.0.1:5000> while the plan
  runs, showing the events as they are detected. Drawing the statechart on the page
  lives on its own branch, see "Related local branches". Set it to `False` and nothing
  imports flask.
- It prints the detected events at the end. `SEGMIND_EVENTS_FILE=/tmp/events.json` also
  writes them as JSON.
- To score a run against the ground truth:
  `.venv/bin/python segmind/scripts/report_demo_events.py bullet_world`

**Run the demo alone, in the foreground.** It needs a lot of memory: runs started in the
background were killed for low memory, and starting a pytest session beside a running
demo killed it too. pytest regenerates the ORM interfaces as it starts, which a demo
importing them at the same moment can also trip over.

## Running the tests

```bash
export PATH="$PWD/.venv/bin:$PATH"   # the conftest regenerates the ORM, which needs ruff
.venv/bin/python -m pytest test/segmind_test
.venv/bin/python -m pytest test/semantic_digital_twin_test/test_worlds test/semantic_digital_twin_test/test_semantic_annotations
.venv/bin/python -m pytest test/coraplex_test        # about 19 minutes
```

## How detection works now

These are the rules that took the most work to get right; each has tests.

- **Choosing the detectors** (`DetectorSelection`, `LiveSegmenter.watching(detectors=...)`):
  a run asks for what is to be detected, and everything that is read from comes along.
  Each detector declares what it `requires`, and the detector reporting that something
  ended names the one reporting that it began as its `counterpart`; a run using either of
  the two uses both. Pick-ups and placings require supports and translations; insertions
  require contacts and containments. Grasping is never brought along unless asked for.
  Asking for nothing uses every kind of detector. A kind that is read from others is
  run once over all bodies; every other kind watches each body.
- **Support** (`is_supported_by`, semantic_digital_twin): the two bodies touch within
  `RESTING_CONTACT_TOLERANCE` (5 mm), and the point where they touch lies below the
  supported body's middle. It used to compare bounding boxes and centres of mass, which
  failed in opposite directions: a wall's bounding box "supported" whatever passed
  through it, while nothing standing in a drawer was ever supported, because a drawer's
  own middle sits above its floor.
- **Contact** (`contact`): queries the collision detector at its own threshold, and reads
  the distance from the answer. `BulletCollisionDetector` returns the closest pair
  whatever range it is asked for, and which detector answers depends on what a session
  has loaded — so never rely on the query range to filter.
- **The robot is left out** (`AbstractDetector.exclude_robot`, default `True`): contacts,
  supports and containments are never checked against robot bodies. The grasp detector
  asks about end effectors directly and is not affected.
- **A grasp** (`GraspDetector`): both fingers of a two-fingered hand touch the object —
  the thumb side and the finger side. It is reported against the hand's tool frame, which
  the detector finds itself from each `EndEffector` annotation in the world; nothing is
  configured. A hand that is not two-fingered is read as a whole.
- **Pick-ups and placings** (`PickUpDetector`, `PlacingDetector`): read from grasps only
  where a `GraspDetector` / `LossOfGraspDetector` runs in the same statechart
  (`runs_beside`); otherwise from the object's own motion.
- **A held object gains no new support**, so a surface it brushes while carried is not
  somewhere it came to rest. This needs grasps to be detected.
- **An event is evidence for one interaction only**
  (`SegmindContext.spent_interaction_events`), so a hand that loses its grip and takes
  hold again is not a second pick-up.
- **The dashboard** hides Contact, LossOfContact, Translation, StopTranslation, Rotation
  and StopRotation (`LiveEventDashboard.hidden_event_types`). The event feed keeps every event, so the
  events file and the report script still see them all.

## Open questions and next steps

1. **Grasps drop out mid-carry, and a placing is read from a support the object had
   already lost.** With the demo's own detectors, the milk and the bowl each report a
   `LossOfGraspEvent` just after being picked up, while still in the hand. The placing
   detector then pairs that loss of grasp with the `SupportEvent` from the countertop the
   object stood on before it was lifted — a support already ended by a
   `LossOfSupportEvent` — and reports a placing on the countertop. Once the bowl counts
   as released, it gains that support again and loses it, which is its second pick-up.
   Two fixes, likely both wanted:
   - a placing may only be read from a support that still holds, not one ended before
     the object was let go;
   - a grasp is lost only once the hand has let go for several ticks, not on a single
     tick without contact from both fingers.
   With every kind of detector running (before `DETECTORS` existed) the same demo gave
   exactly one pick-up and one placing per object, so it is the timing of a smaller set
   of detectors that exposes this.
2. **SegMind reports a fault in the plan here, and that is correct.** Reaching for the
   milk, the gripper collides with the bowl and closes on it: a `GraspEvent` and
   `LossOfGraspEvent` for the bowl come just before the milk loses its support. The plan
   should not do that, so this is SegMind detecting an execution error, not a detection
   error to fix.
3. **The message of `423ed2cb4a`** (see above).
4. **`SHOW_LIVE_EVENTS = True` binds port 5000 on every run**, including runs launched by
   `report_demo_events.py`. Two demos at once will fail to start the page.
5. **`scripts/format_docstrings.py` does not run**: `docformatter` is not installed in the
   venv. `black` was used on every file touched instead.

## Other demos

Their wiring was removed from this branch and was never committed anywhere, so it has
to be rebuilt. What was learned:

- **Tracy** (`coraplex/demos/coraplex_real_tracy/demo.py`, stacks Box2 on Box1 and Box3
  on Box2): reached 12 of 12 with two changes.
  - The resting tolerance, set to 5 mm from a measured placement error of 1.9 mm (Box2
    above Box1). That tolerance is on this branch.
  - `PlaceAction` drives the gripper's *tool frame* to the target pose, not the object,
    so an object lands offset by however the grasp happened to hold it — measured at
    15.5 mm above the box's centre in one run and about 51 mm below it in another,
    which left Box3 floating 48 mm above Box2. The fix was to drive the *held body*
    itself to the target in the final descent: a `MoveHeldBodyMotion` subclassing
    `MoveToolCenterPointMotion` that names the body as the tip link. Giskard supports an
    attached body as a tip, and binds its kinematics when the motion starts, so it works
    even though the plan is built before the object is picked up. **This is not on the
    branch** and has to be rebuilt.
- **Stretch, and the cutting, mixing, pouring and wiping tool demos** were wired to
  SegMind and scored earlier, before most of the rules above existed. Their numbers
  would need measuring again, and each demo should name the detectors it asks for, as
  the bullet world demo does.

## Related local branches

`segmind-live-statechart-page` (local only, off this branch before the statechart was
taken out of it) draws the statechart on the live page: a tab of its own, one column per
watched object with arrows from what each detector is read from, and each detector's
live state. It also adds `reads_when_running` to the detectors. Its commits are
`081e3a7eea` and `5a6a8b3c90`, on top of `4d2244ebd5`, the older copy of this branch's
last commit.

`segmind-live-2` to `segmind-live-7` form a separate stack off the same base: SegMind's
port from the ICRA branch, `DetectorSet`, `SegmindMonitor` and the original dashboard.
The dashboard was taken from `segmind-live-5-dashboard` and adapted to `LiveSegmenter`.
`DetectorSelection` does the job of that stack's `DetectorSet` in a smaller way:
`DetectorSet` binds each detector's event type as a generic and derives what it needs
from that, which is the fuller design but did not port onto this branch's detectors.
Nothing else from that stack is on this branch.

## Working on this branch

- Commit as `sorinar329 <mrsoran2009@gmail.com>`, and push only to `origin`.
- Follow `AGENTS.md`: a failing test before every fix, and never change a pre-existing
  test to make it pass.
