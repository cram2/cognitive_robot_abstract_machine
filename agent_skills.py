"""
The skills of this repository, which ``krrood skills`` writes into
``.claude/skills/<name>/SKILL.md``.
"""

from __future__ import annotations

from krrood.development_tools.agents.rule import Section
from krrood.development_tools.agents.skill import AgentTool, Skill
from krrood.ormatic.utils import classproperty

# %% CRAM live session


class WorkingLoopRules(Section):
    """
    How each step of a live session goes.
    """

    ONE_STEP_PER_INSTRUCTION = (
        "**One step per instruction.** Do what the developer asked, then stop and "
        "wait. Never chain further robot steps on your own, even when the next one "
        'seems obvious. An instruction with several parts ("open the gripper, put a '
        'finger left of the lever, push right") is one step; perform its parts in '
        "one cell"
    )

    SHOW_THE_CODE_FIRST = (
        "**Show the code before you perform it.** Every motion cell goes into the "
        "chat as a code block first, then gets performed. Measuring code may be "
        "summarised in a sentence"
    )

    REPORT_FROM_THE_REPORTS = (
        "**Report from the reports.** Every cell ends with "
        "`session.report(label=..., outcome=...)`; quote the status, tool and base "
        "positions, and say in a sentence what that means for the instruction "
        "(reached, stopped short by how much, what failed)"
    )

    DIAGNOSE_BEFORE_RETRYING = (
        "**When a step fails, diagnose before retrying.** Measure with a raw run "
        "(what is near, which joint is at its limit, which body blocks) and offer "
        "the developer concrete options with a recommendation. Do not loop through "
        "blind retries"
    )

    UNDO_INSTEAD_OF_REBUILDING = (
        "**Undo instead of rebuilding.** To retry a step, "
        "`session.revert_last_cell()`; to show steps again, "
        "`session.replay(first_cell=i, namespace=globals())`. Name a checkpoint the "
        "developer will want to come back to with `session.take_snapshot(name=...)`"
    )

    KEEP_THE_DEVELOPERS_WORDS = (
        "**Keep the developer's words.** Cell titles say what the developer asked "
        "for, in their words; they are what the history shows"
    )


class SettingUpRules(Section):
    """
    How a live session is started.
    """

    WHERE_THE_KERNEL_RUNS = (
        "The kernel runs in a container with CRAM and MuJoCo on a server reachable by "
        "ssh. `scripts/live_kernel.py` drives it from this machine; run it with the "
        "project's Python interpreter. Every option of it defaults to the `LIVE_*` "
        "environment variable of the same name (`--server` to `LIVE_SERVER`), and "
        "the defaults fit the CRAM docker image (CRAM in `/opt/cram`, environment "
        "`/entrypoint.sh`)"
    )

    ASK_FOR_THE_SERVER = (
        "Ask the developer for `LIVE_SERVER`, `LIVE_CONTAINER` and "
        "`LIVE_EXCHANGE_ON_SERVER`, the server's side of a directory mounted into the "
        "container at `LIVE_EXCHANGE_IN_CONTAINER`"
    )

    START_THE_SESSION = (
        "`live_kernel.py start` starts the kernel in the container and runs "
        "`templates/robocasa_session.py` in it: that loads a RoboCasa kitchen with a "
        "PR2, starts the physics, defines `run(*actions) -> PlanOutcome` and creates "
        "`session` with a snapshot `composed`. For another scene or robot, copy the "
        "template to your scratchpad, change its first cell and pass it with "
        "`--template`"
    )

    START_AGAIN_AFTER_A_RESTART = (
        "If the kernel runs already, `start` leaves it alone; set the session up "
        "again with `live_kernel.py raw TEMPLATE`. After a kernel restart everything "
        "in it is gone, snapshots included"
    )


class PerformingStepsRules(Section):
    """
    How a cell is written and performed.
    """

    WRITE_THE_CELL_TO_A_FILE = (
        "Write the cell to a file in your scratchpad, starting with "
        "`# %% cell <title>` and ending with "
        '`session.report(label="...", outcome=outcome)`'
    )

    PERFORM_IT = (
        "Perform it with `live_kernel.py perform FILE`, several cells of a file with "
        "`--cells 2:4`. The output streams back as the kernel prints it. A run still "
        "going after `LIVE_WAIT_SECONDS` keeps running; follow it with "
        "`live_kernel.py wait NAME`"
    )

    MEASURE_WITH_RAW_RUNS = (
        "Use `live_kernel.py raw FILE` for measuring and for set up code: a raw run "
        "is not kept in the history, so replays do not repeat it"
    )


class DataStructuresRules(Section):
    """
    The classes a live session is made of.
    """

    THE_SESSION = (
        "`experiments.live_session.session.LiveSession(simulation=..., "
        "tool_frame=..., robot_base=...)` is `session` in the kernel:",
        (
            "`report(label=..., outcome=None) -> StepReport` logs one aligned line "
            "(label, status, tool and base position) and keeps it with the running "
            "cell",
            "`perform(cell=..., namespace=...)` takes a snapshot, then runs the cell "
            "in the kernel's `globals()`; `perform_cells.py` calls it",
            "`history: list[PerformedCell]` holds every performed cell with its "
            "`cell`, `snapshot_before`, `reports` and `finished` (false if it raised)",
            "`revert_last_cell() -> Cell` goes back to the moment before the last "
            "cell, which is dropped",
            "`replay(first_cell=..., namespace=...)` goes back to the moment before "
            "history entry `first_cell`, then performs it and every later cell again",
            "`take_snapshot(name=...)` and `restore_snapshot(name=...)` keep and "
            "restore named checkpoints in `snapshots`",
        ),
    )

    STEP_STATUS = (
        "`StepReport.status` is `StepStatus.SUCCEEDED` (`ok`), `StepStatus.NO_PLAN` "
        "(empty), or the failure's class name"
    )

    CELLS = (
        "`experiments.live_session.cells` holds `Cell(title=..., source=...)`, "
        "`CellScript.from_file(path=...)` and `CellMarker.START` (`# %% cell `)"
    )

    SNAPSHOTS = (
        "`semantic_digital_twin.adapters.multi_sim.MujocoSim.take_snapshot()` and "
        "`restore_snapshot(snapshot=...)` restore the physics exactly (positions, "
        "velocities, controls, time), the world state and the servo set points; the "
        "physics then continues as it did after the snapshot. A snapshot does not "
        "survive a change to the world's model (`SnapshotDoesNotFitSimulationError`)"
    )


class UsefulPiecesForCellsRules(Section):
    """
    Code that keeps coming up in cells.
    """

    BODY_BOX = (
        "World frame box of a body: "
        "`body.collision.as_bounding_box_collection_in_frame(world.root)"
        ".bounding_box().to_array_bounds()` (`.lower`, `.upper`); pass the robot's "
        "root as the frame for a footprint"
    )

    MUJOCO_JOINT = (
        "A MuJoCo joint's anchor and axis: "
        "`mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)`, then "
        "`data.xanchor[j]`, `data.xaxis[j]`, `data.qpos[m.jnt_qposadr[j]]`, "
        "`m.jnt_range[j]` (`m, data = simulation.simulator._mj_model, "
        "simulation.simulator._mj_data`)"
    )

    MOTIONS = (
        "Motions: `NavigateAction(target_location=pose)`, "
        "`MoveTorsoAction(torso_state=TorsoState.LOW)`, "
        "`ParkArmsAction(arms=robot.all_arms)`, `PickUpAction(grasp=..., arm=arm)`; "
        "perform them with the template's `run(...)`, which returns the "
        "`PlanOutcome` to report"
    )

    MEASURE_A_STALLED_MOTION = (
        "Joint limits and what is near each arm link are the first things to "
        "measure when a motion stops with `MotionMadeNoProgress`"
    )


class WhenSomethingGoesWrongRules(Section):
    """
    What to do when the session misbehaves.
    """

    NO_OUTPUT = (
        "**No output and the run does not end**: `live_kernel.py` warns when the "
        "kernel sleeps without output, most likely a deadlock. "
        "`live_kernel.py interrupt` raises `KeyboardInterrupt` in the kernel and the "
        "traceback shows where it waited; report it to the developer"
    )

    DOCKER_EXEC_HANGS = (
        "**`docker exec` hangs** on some servers after the command finished: the "
        "script always runs detached and reads a log instead. Do the same for "
        "anything you run by hand"
    )

    RESTORE_REFUSED = (
        "**A restore is refused** with `SnapshotDoesNotFitSimulationError`: the "
        "world's model changed (an object was spawned or removed); take new "
        "snapshots after such a change"
    )


class CRAMLiveSessionSkill(Skill):
    """
    Driving a simulated CRAM robot cell by cell from a persistent IPython kernel.
    """

    @classproperty
    def description(cls) -> str:
        return (
            "Drive a simulated CRAM robot step by step from a persistent IPython "
            "kernel while the developer says what the next step is. Use when the "
            'developer wants to "work in the IPython workflow", "tell you each '
            'step", replay or undo a step, or debug an action (e.g. in a RoboCasa '
            "kitchen) interactively."
        )

    @classproperty
    def allowed_tools(cls) -> list[AgentTool]:
        return [AgentTool.BASH, AgentTool.READ, AgentTool.WRITE, AgentTool.EDIT]

    @classproperty
    def introduction(cls) -> str:
        return (
            "The developer says what the robot does next. You write each step as a "
            "*cell*, show its code, perform it in a kernel that stays alive between "
            "steps, and report what happened. The kernel keeps the world, the robot, "
            "the simulation and every moment before a cell, so a step can be undone, "
            "retried or replayed in seconds instead of rebuilding the scene.\n\n"
            "The data structures live in CRAM (`experiments.live_session`, and the "
            "snapshot of a stepped MuJoCo simulation in "
            "`semantic_digital_twin.adapters.multi_sim`); the scripts next to this "
            "file drive the kernel. Code you write in cells follows `@AGENTS.md` "
            "only loosely: cells are throwaway, but anything that proves itself "
            "moves into the codebase under the full rules."
        )

    @classproperty
    def sections(cls) -> list[type[Section]]:
        return [
            WorkingLoopRules,
            SettingUpRules,
            PerformingStepsRules,
            DataStructuresRules,
            UsefulPiecesForCellsRules,
            WhenSomethingGoesWrongRules,
        ]
