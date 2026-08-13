# Bullet World Demo

A PR2 in the IAI apartment transports a milk carton, a bowl, and a spoon (which
starts inside a drawer) to the kitchen table: park arms, raise the torso, then
three `TransportAction`s.

## Running

```bash
uv run python demo.py
```

By default this spawns a [Rerun](https://rerun.io) viewer showing the apartment
and the robot while the plan executes. The timeline can be scrubbed, and the
text-log panel lists every action and motion as it starts and ends.

## Visualization options

The demo reads three environment variables:

| Variable | Values | Meaning |
| --- | --- | --- |
| `CORAPLEX_VISUALIZATION` | `rerun` (default), `rviz`, `none` | Which renderer to use. `rviz` needs a sourced ROS 2 environment. |
| `CORAPLEX_RERUN_MODE` | `spawn` (default), `connect`, `save`, `none` | Where the Rerun stream goes. |
| `CORAPLEX_RERUN_TARGET` | URL or file path | gRPC URL for `connect`, `.rrd` path for `save`. |

Record the run to a shareable file:

```bash
CORAPLEX_RERUN_MODE=save CORAPLEX_RERUN_TARGET=demo.rrd uv run python demo.py
rerun demo.rrd
```

Run headless (what CI does through `test_demo.py`):

```bash
CORAPLEX_VISUALIZATION=none uv run python demo.py
```
