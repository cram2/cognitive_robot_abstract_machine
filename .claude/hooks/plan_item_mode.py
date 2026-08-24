#!/usr/bin/env python3
"""
Resolve how a plan-item skill starts an item: by asking, by planning, or on its own.

``plan-item-kickoff`` and ``plan-item-resolve`` both gather their context and then hand
the user a plan to approve. That gate is worth keeping where a plan genuinely needs a
decision; what is wrong is that it is unconditional, so it costs the same round trip on
an item whose notes and roadmap section already settle every design call.

Three modes, one of which only decides who picks:

``plan``
    Present the plan and stop until it is approved.

``auto``
    Draft the plan, record it, and implement it without asking. The planning phase still
    happens and is still written down; it stops blocking. The default.

``ask``
    Put the choice to the user, with the skill's own recommendation.

Precedence is invocation argument > personal setting > committed default, so one run can
depart from the setting without changing it. What each mode obliges a skill to do is
``plan-dashboard/execution-modes.md``'s to say; this module only answers which one is in
force.

Usage:
    python3 plan_item_mode.py resolve --skill <kickoff|resolve> [--requested <mode>]
    python3 plan_item_mode.py set --skill <kickoff|resolve> [--skill ...] --mode <mode>

Prints a one-line JSON report led by ``status`` and ``exit_code``, so a caller acting on
the document never has to decode an integer back into a meaning.

.. note::
   An absent personal settings file, or an unreachable notes branch, resolves to the
   committed default rather than failing: someone who has never set personal notes up
   still gets a working skill. A setting that is *present but unreadable* - a value
   outside the enum, or TOML that will not parse - is refused instead. A silent fall back
   there would be indistinguishable from the setting having worked, and the point of the
   setting is that the user does not have to watch a run to know what it will do.

.. note::
   ``resolve-personal-notes-config.sh`` is the one path spelled literally here. It is the
   bootstrap seam every script and skill in this system sources to find every other path,
   for the reason ``refresh_dashboard.sh``'s own header records: something has to name it
   before anything can ask it where things are.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import tempfile
import tomllib
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, ClassVar

# %% locations

HOOKS_DIRECTORY = Path(".claude/hooks")
"""
Where this script and the shell configuration it sources live, from the project root.
"""


class Location(StrEnum):
    """
    Every file this module reads, runs or writes, from the project root.

    A member is the path as text, since that is what most of them are handed to - a
    ``git`` reference, a ``bash -c`` line, a subprocess argument, the report a caller
    parses. :attr:`path` is the same location where a real :class:`~pathlib.Path` is
    wanted. ``HOOKS_DIRECTORY`` stays a constant above rather than a member, so the
    directory the first three share is still named once.
    """

    CONFIGURATION_SCRIPT = f"{HOOKS_DIRECTORY}/resolve-personal-notes-config.sh"
    """
    The shell configuration that resolves the personal-notes remote and branch.
    """

    NOTES_WRITER_SCRIPT = f"{HOOKS_DIRECTORY}/write-personal-notes-file.sh"
    """
    The generic commit-and-push-one-file helper ``set`` writes through.
    """

    COMMITTED_DEFAULTS = f"{HOOKS_DIRECTORY}/plan-item-modes.toml"
    """
    The shipped defaults, in the repository rather than on the personal-notes branch.
    """

    PERSONAL_SETTINGS = ".claude/personal/plan-item-modes.toml"
    """
    The per-user override, on the personal-notes branch.
    """

    @property
    def path(self) -> Path:
        """
        :return: This location as a path, for the callers that do path arithmetic on it.
        """
        return Path(self.value)


# %% vocabulary


class ExecutionMode(StrEnum):
    """
    How a plan-item skill proceeds once it has gathered the item's context.
    """

    ASK = "ask"
    """
    Put the choice between the other two to the user.
    """

    AUTO = "auto"
    """
    Plan implicitly and implement without waiting for approval.
    """

    PLAN = "plan"
    """
    Present the plan and stop until the user approves it.
    """


class ModeSource(StrEnum):
    """
    Where a resolved mode came from, in precedence order.
    """

    INVOCATION_ARGUMENT = "invocation_argument"
    """
    Asked for on the command line, for this run only.
    """

    PERSONAL_SETTING = "personal_setting"
    """
    Pinned in the personal settings file on the notes branch.
    """

    COMMITTED_DEFAULT = "committed_default"
    """
    Nothing was configured, so the shipped default applies.
    """


class PlanItemSkill(StrEnum):
    """
    The skills that resolve a mode, each owning the key its setting is stored under.
    """

    KICKOFF = "kickoff"
    """
    ``plan-item-kickoff``, which starts an item that has not been started.
    """

    RESOLVE = "resolve"
    """
    ``plan-item-resolve``, which unblocks or resumes one already underway.
    """

    @property
    def setting_key(self) -> str:
        """
        The key this skill's mode is stored under, in both settings files.
        """
        return f"{self.value}_mode"


class CommandLineOption(StrEnum):
    """
    The options this tool accepts, named once so a parser and a refusal cannot disagree.
    """

    SKILL = "--skill"
    """
    Which skill to act on; ``set`` accepts it once per skill it pins.
    """

    REQUESTED = "--requested"
    """
    ``resolve``'s mode override for one run.
    """

    MODE = "--mode"
    """
    The mode ``set`` pins.
    """


class ExitCode(IntEnum):
    """
    The process statuses this tool exits with.

    A distinct status per refusal lets a caller act on *which* failure happened without
    parsing stderr. The named refusals start at 3 because the two below them are already
    spoken for: 2 is the usage error ``argparse`` exits with, and 1 is what Python exits
    with on an uncaught exception, so leaving it free keeps a crash from reading as a
    refusal this tool chose to make.
    """

    SUCCESS = 0
    """
    The operation ran and printed its report.
    """

    USAGE = 2
    """
    No such subcommand, or the wrong arguments. Exited with by ``argparse`` itself, and
    named here so the value is reserved rather than merely unused.
    """

    UNKNOWN_MODE = 3
    """
    A mode was named that :class:`ExecutionMode` does not define.
    """

    MALFORMED_MODE_SETTINGS = 4
    """
    A settings file exists but cannot be read as the modes it is supposed to hold.
    """

    SETTINGS_WRITE_REFUSED = 5
    """
    The personal settings file could not be pushed to the notes branch.
    """

    @property
    def name_for_a_caller(self) -> str:
        """
        The status as the lowercase name a report and a message both use.
        """
        return self.name.lower()


# %% settings


@dataclass(frozen=True)
class ModeSetting:
    """
    One mode as it was given, together with what gave it.

    Pairing them is what lets a refusal name the setting or the option a bad value came
    from, without every reader of a value having to carry that name beside it.
    """

    key: str
    """
    The settings key the value was stored under, or the option that carried it.
    """

    value: Any
    """
    The value as it was read, which TOML and ``argparse`` are both free to make any type.
    """

    def parse(self) -> ExecutionMode:
        """
        Read this setting as a mode, refusing anything :class:`ExecutionMode` does not
        define.

        :raises UnknownModeError: If the value names no mode.
        :return: The mode.
        """
        try:
            return ExecutionMode(self.value)
        except ValueError:
            raise UnknownModeError(setting=self) from None


@dataclass(frozen=True)
class SettingsFile:
    """
    A modes file, together with the path a refusal about it should name.
    """

    path: Path
    """
    The file on disk to read.
    """

    origin: Location
    """
    Where the settings came from as the user knows it, which for the personal file is its
    location on the notes branch rather than the scratch copy it is read from.
    """

    def read(self) -> dict[str, Any]:
        """
        Parse the file.

        :raises MalformedModeSettingsError: If it will not parse as TOML.
        :return: The parsed settings.
        """
        try:
            return tomllib.loads(self.path.read_text())
        except tomllib.TOMLDecodeError as error:
            raise MalformedModeSettingsError(file=self, detail=str(error)) from None


# %% failures


@dataclass
class ModeError(Exception, ABC):
    """
    Base for every refusal this tool reports, each carrying its own exit status.

    Subclasses hold the context that explains the refusal as typed fields and compose it
    into the message at construction, so no call site formats one. Mirrors the
    ``plan_item_bootstrap.py`` idiom beside it.

    Where one of these translates a library exception - ``ValueError`` from an enum,
    ``TOMLDecodeError`` from a parse - the refusal is raised ``from None``: everything the
    original said is already folded into these fields, so chaining it would print a second
    traceback saying the same thing in the library's words instead of ours.
    """

    exit_code: ClassVar[ExitCode] = ExitCode.SUCCESS
    """
    The process status this refusal exits with.
    """

    def __post_init__(self) -> None:
        """
        Compose the message from the subclass's own description and advice.
        """
        correction = self.suggest_correction()
        message = self.error_message()
        super().__init__(f"{message}\n{correction}" if correction else message)

    def __str__(self) -> str:
        """
        The composed message, rather than a repr of the dataclass fields.
        """
        return Exception.__str__(self)

    @abstractmethod
    def error_message(self) -> str:
        """
        :return: What went wrong.
        """

    @abstractmethod
    def suggest_correction(self) -> str:
        """
        :return: What to do about it, or an empty string when there is nothing to add.
        """


@dataclass
class UnknownModeError(ModeError):
    """
    Raised when a mode is named that :class:`ExecutionMode` does not define.
    """

    exit_code: ClassVar[ExitCode] = ExitCode.UNKNOWN_MODE

    setting: ModeSetting
    """
    The setting whose value names no mode.
    """

    def error_message(self) -> str:
        return f"{self.setting.key}: {self.setting.value!r} is not an execution mode."

    def suggest_correction(self) -> str:
        return f"Use one of: {', '.join(ExecutionMode)}."


@dataclass
class MalformedModeSettingsError(ModeError):
    """
    Raised when a settings file exists but cannot be read as the modes it should hold.
    """

    exit_code: ClassVar[ExitCode] = ExitCode.MALFORMED_MODE_SETTINGS

    file: SettingsFile
    """
    The settings file that could not be read.
    """

    detail: str
    """
    What was wrong with it.
    """

    def error_message(self) -> str:
        return f"{self.file.origin} could not be read: {self.detail}"

    def suggest_correction(self) -> str:
        return (
            "Fix the file, or delete it to fall back to "
            f"{Location.COMMITTED_DEFAULTS}'s defaults."
        )


@dataclass
class SettingsWriteRefusedError(ModeError):
    """
    Raised when the personal settings file could not be pushed to the notes branch.
    """

    exit_code: ClassVar[ExitCode] = ExitCode.SETTINGS_WRITE_REFUSED

    detail: str
    """
    What the notes writer reported.
    """

    def error_message(self) -> str:
        return f"{Location.PERSONAL_SETTINGS} was not written: {self.detail}"

    def suggest_correction(self) -> str:
        return "Check the personal-notes branch is set up: run /setup-personal-notes."


# %% reading the settings


def committed_defaults(project_root: Path) -> dict[str, ExecutionMode]:
    """
    The shipped mode for every skill.

    :param project_root: The repository to read within.
    :raises MalformedModeSettingsError: If the file will not parse, or names no mode for
        some skill.
    :raises UnknownModeError: If a default names no mode.
    :return: Every skill's default, keyed by its setting key.
    """
    shipped = SettingsFile(
        path=project_root / Location.COMMITTED_DEFAULTS,
        origin=Location.COMMITTED_DEFAULTS,
    )
    raw = shipped.read()
    missing = [
        skill.setting_key for skill in PlanItemSkill if skill.setting_key not in raw
    ]
    if missing:
        raise MalformedModeSettingsError(
            file=shipped, detail=f"no default for {', '.join(missing)}"
        )
    return {
        skill.setting_key: ModeSetting(
            key=skill.setting_key, value=raw[skill.setting_key]
        ).parse()
        for skill in PlanItemSkill
    }


def personal_settings(project_root: Path) -> dict[str, Any]:
    """
    The per-user overrides as they stand on the notes branch.

    :param project_root: The repository to read within.
    :raises MalformedModeSettingsError: If the file exists but will not parse.
    :return: The parsed overrides, empty when there is no file or no notes branch.
    """
    if not fetch_notes_branch(project_root):
        return {}
    shown = subprocess.run(
        ["git", "show", f"FETCH_HEAD:{Location.PERSONAL_SETTINGS}"],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if shown.returncode != 0:
        return {}
    with tempfile.TemporaryDirectory() as scratch_directory:
        scratch_file = Path(scratch_directory) / "personal-plan-item-modes.toml"
        scratch_file.write_text(shown.stdout)
        return SettingsFile(path=scratch_file, origin=Location.PERSONAL_SETTINGS).read()


def fetch_notes_branch(project_root: Path) -> bool:
    """
    Fetch the personal-notes branch, leaving ``FETCH_HEAD`` pointing at it.

    Sources the shell configuration and calls its own fetch function rather than
    re-deriving the remote and branch precedence, so this and the hook scripts can never
    disagree about which branch a setting is on.

    :param project_root: The repository to fetch within.
    :return: Whether the branch was reachable.
    """
    probe = subprocess.run(
        [
            "bash",
            "-c",
            f'source "{Location.CONFIGURATION_SCRIPT}" && fetch_personal_notes_branch',
        ],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    return probe.returncode == 0


# %% reports


class ReportStatus(StrEnum):
    """
    What a printed report says happened, as the first thing a caller reads.
    """

    RESOLVED = "resolved"
    """
    A mode was worked out and is being reported.
    """

    SET = "set"
    """
    A mode was pinned in the personal settings file.
    """


class ReportKey(StrEnum):
    """
    The keys a printed report is read by.

    The document is parsed by callers outside this module, so its spelling is a contract
    rather than a diagnostic; naming the keys here is what gives a rename one place to
    happen.
    """

    STATUS = "status"
    """
    What happened, as a :class:`ReportStatus`.
    """

    EXIT_CODE = "exit_code"
    """
    The process status the run exits with, so acting on the document needs no decoding.
    """

    SKILL = "skill"
    """
    The skill a mode was resolved for.
    """

    SKILLS = "skills"
    """
    The skills a mode was pinned for.
    """

    MODE = "mode"
    """
    The mode in force, or the one pinned.
    """

    SOURCE = "source"
    """
    What decided the mode, as a :class:`ModeSource`.
    """

    SETTING_KEY = "setting_key"
    """
    The key the resolved skill's mode is stored under.
    """

    PERSONAL_SETTING_PATH = "personal_setting_path"
    """
    Where a per-user override goes, so a caller can say where to change the answer.
    """


class Report(ABC):
    """
    What an operation prints, and the status it exits with.

    Declared rather than left to a convention each report is trusted to have followed,
    since ``main`` prints and exits through this pair without knowing which operation
    produced it.

    ``to_json`` is the name and signature
    :class:`krrood.adapters.json_serializer.SubclassJSONSerializer` gives the same
    operation, so a report can inherit it wherever that dependency becomes available.
    """

    @property
    @abstractmethod
    def exit_code(self) -> ExitCode:
        """
        :return: The process status this report exits with.
        """

    @abstractmethod
    def to_json(self) -> dict[str, Any]:
        """
        :return: The report a caller reads, led by the status it can act on.
        """


# %% resolving


@dataclass(frozen=True)
class ModeResolution(Report):
    """
    Which mode a skill runs in, and what decided it.
    """

    skill: PlanItemSkill
    """
    The skill the mode was resolved for.
    """

    mode: ExecutionMode
    """
    The mode in force.
    """

    source: ModeSource
    """
    What decided it.
    """

    exit_code: ExitCode = ExitCode.SUCCESS
    """
    The process status this resolution exits with.
    """

    def to_json(self) -> dict[str, Any]:
        """
        :return: The mode in force, what decided it, and where to change it.
        """
        return {
            ReportKey.STATUS: ReportStatus.RESOLVED,
            ReportKey.EXIT_CODE: int(self.exit_code),
            ReportKey.SKILL: str(self.skill),
            ReportKey.MODE: str(self.mode),
            ReportKey.SOURCE: str(self.source),
            ReportKey.SETTING_KEY: self.skill.setting_key,
            ReportKey.PERSONAL_SETTING_PATH: Location.PERSONAL_SETTINGS,
        }


def resolve_mode(
    skill: PlanItemSkill, requested: str | None, project_root: Path
) -> ModeResolution:
    """
    Work out which mode *skill* runs in, in precedence order.

    :param skill: The skill to resolve for.
    :param requested: A mode asked for on the command line, if any.
    :param project_root: The repository to read within.
    :raises UnknownModeError: If a named mode is not one this tool defines.
    :raises MalformedModeSettingsError: If a settings file cannot be read.
    :return: The mode and what decided it.
    """
    if requested is not None:
        return ModeResolution(
            skill=skill,
            mode=ModeSetting(key=CommandLineOption.REQUESTED, value=requested).parse(),
            source=ModeSource.INVOCATION_ARGUMENT,
        )

    overrides = personal_settings(project_root)
    if skill.setting_key in overrides:
        return ModeResolution(
            skill=skill,
            mode=ModeSetting(
                key=skill.setting_key, value=overrides[skill.setting_key]
            ).parse(),
            source=ModeSource.PERSONAL_SETTING,
        )

    return ModeResolution(
        skill=skill,
        mode=committed_defaults(project_root)[skill.setting_key],
        source=ModeSource.COMMITTED_DEFAULT,
    )


# %% writing


@dataclass(frozen=True)
class SettingsWriteReport(Report):
    """
    What ``set`` pinned, and where.
    """

    skills: tuple[PlanItemSkill, ...]
    """
    The skills whose mode was pinned, in the order they were named.
    """

    mode: ExecutionMode
    """
    The mode they were pinned to.
    """

    exit_code: ExitCode = ExitCode.SUCCESS
    """
    The process status this write exits with.
    """

    def to_json(self) -> dict[str, Any]:
        """
        :return: What was pinned, to what, and in which file.
        """
        return {
            ReportKey.STATUS: ReportStatus.SET,
            ReportKey.EXIT_CODE: int(self.exit_code),
            ReportKey.SKILLS: [str(skill) for skill in self.skills],
            ReportKey.MODE: str(self.mode),
            ReportKey.PERSONAL_SETTING_PATH: Location.PERSONAL_SETTINGS,
        }


def render_settings(modes: dict[str, ExecutionMode]) -> str:
    """
    Render the personal settings file.

    :param modes: Every skill's mode, keyed by its setting key.
    :return: The file's text.
    """
    header = (
        "# Personal plan-item execution modes, layered over the committed defaults\n"
        f"# at {Location.COMMITTED_DEFAULTS}. Rewritten in full by plan_item_mode.py's\n"
        "# set command, so anything other than the keys below is not preserved.\n"
    )
    body = "".join(
        f'{skill.setting_key} = "{modes[skill.setting_key]}"\n'
        for skill in PlanItemSkill
    )
    return header + body


def write_mode(
    skills: Sequence[PlanItemSkill], mode: ExecutionMode, project_root: Path
) -> SettingsWriteReport:
    """
    Pin one or more skills' mode in the personal settings file on the notes branch.

    Every skill not named keeps whatever it already had, falling back to the committed
    default where the file said nothing - so pinning one mode can never clobber another.
    Naming several writes the file once, which matters because each write is a push.

    :param skills: The skills to pin.
    :param mode: The mode to pin them to.
    :param project_root: The repository to write from.
    :raises UnknownModeError: If the file already holds a value naming no mode.
    :raises MalformedModeSettingsError: If a settings file cannot be read.
    :raises SettingsWriteRefusedError: If the notes writer refused the push.
    :return: What was pinned.
    """
    overrides = personal_settings(project_root)
    modes = {
        other.setting_key: (
            ModeSetting(
                key=other.setting_key, value=overrides[other.setting_key]
            ).parse()
            if other.setting_key in overrides
            else committed_defaults(project_root)[other.setting_key]
        )
        for other in PlanItemSkill
    }
    # Naming a skill twice pins it once, so the report never claims more than happened.
    requested = tuple(dict.fromkeys(skills))
    for skill in requested:
        modes[skill.setting_key] = mode

    pinned = ", ".join(skill.setting_key for skill in requested)
    with tempfile.TemporaryDirectory() as scratch_directory:
        scratch_file = Path(scratch_directory) / "plan-item-modes.toml"
        scratch_file.write_text(render_settings(modes))
        written = subprocess.run(
            [
                "bash",
                Location.NOTES_WRITER_SCRIPT,
                "--source",
                str(scratch_file),
                "--destination",
                Location.PERSONAL_SETTINGS,
                "--message",
                f"Set {pinned} to {mode}",
            ],
            cwd=project_root,
            capture_output=True,
            text=True,
        )
    if written.returncode != 0:
        raise SettingsWriteRefusedError(
            detail=written.stderr.strip() or "the notes writer reported no detail"
        )
    return SettingsWriteReport(skills=requested, mode=mode)


# %% command line


class Subcommand(StrEnum):
    """
    The operations this tool offers.
    """

    RESOLVE = "resolve"
    """
    Report which mode a skill runs in.
    """

    SET = "set"
    """
    Pin a skill's mode in the personal settings file.
    """


def build_parser() -> argparse.ArgumentParser:
    """
    :return: The command line this tool accepts, per the module docstring.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    subcommands = parser.add_subparsers(dest="subcommand", required=True)

    for name in Subcommand:
        subcommands.add_parser(name)

    # `resolve` answers for one skill; `set` takes --skill as often as there are skills
    # to pin, because the settings file is rewritten whole and one push should not
    # become several.
    subcommands.choices[Subcommand.RESOLVE].add_argument(
        CommandLineOption.SKILL,
        dest="skill",
        type=PlanItemSkill,
        choices=list(PlanItemSkill),
        required=True,
    )
    subcommands.choices[Subcommand.SET].add_argument(
        CommandLineOption.SKILL,
        dest="skills",
        action="append",
        type=PlanItemSkill,
        choices=list(PlanItemSkill),
        required=True,
    )

    # Both mode options are plain strings, validated by ModeSetting rather than by
    # argparse's own choices, so a mode the enum does not name refuses identically
    # whether it came from the command line or from the settings file.
    subcommands.choices[Subcommand.RESOLVE].add_argument(
        CommandLineOption.REQUESTED, dest="requested", default=None
    )
    subcommands.choices[Subcommand.SET].add_argument(
        CommandLineOption.MODE, dest="mode", required=True
    )
    return parser


def main() -> int:
    """
    Parse arguments, run the requested operation, and print its report.

    See the module docstring for the command line contract.
    """
    arguments = build_parser().parse_args()
    project_root = Path(os.environ.get("CLAUDE_PROJECT_DIR", Path.cwd()))

    report: Report
    try:
        if arguments.subcommand == Subcommand.RESOLVE:
            report = resolve_mode(arguments.skill, arguments.requested, project_root)
        else:
            report = write_mode(
                arguments.skills,
                ModeSetting(key=CommandLineOption.MODE, value=arguments.mode).parse(),
                project_root,
            )
    except ModeError as error:
        print(f"{error.exit_code.name_for_a_caller}: {error}", file=sys.stderr)
        return int(error.exit_code)

    print(json.dumps(report.to_json()))
    return int(report.exit_code)


if __name__ == "__main__":
    sys.exit(main())
