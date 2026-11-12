"""
The skills of this repository, which ``krrood skills`` writes into
``.claude/skills/<name>/SKILL.md``.
"""

from __future__ import annotations

from pathlib import Path

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
        "`session.report(label=..., outcome=...)`; quote the status, measure where "
        "the robot got to when the step is about reaching somewhere, and say in a "
        "sentence what that means for the instruction "
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


class CRAMLiveSessionSkill(Skill):
    """
    Driving a simulated CRAM robot cell by cell from a persistent IPython kernel.
    """

    @classproperty
    def path(cls) -> Path:
        return Path(".claude/skills/cram-live-session/SKILL.md")

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
            "`semantic_digital_twin.adapters.multi_sim`); read them, and the "
            "actions and helpers a step needs, in the codebase. The scripts next to "
            "this file drive the kernel. Code you write in cells follows `@AGENTS.md` "
            "only loosely: cells are throwaway, but anything that proves itself "
            "moves into the codebase under the full rules."
        )

    @classproperty
    def sections(cls) -> list[type[Section]]:
        return [
            WorkingLoopRules,
            SettingUpRules,
            PerformingStepsRules,
        ]
