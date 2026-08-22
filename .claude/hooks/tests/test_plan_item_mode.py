"""
Tests for plan_item_mode.py, which resolves how a plan-item skill starts an item.

Run against the local scratch repository fixture rather than a real notes branch, so
nothing here needs network access or credentials.

Every path and key asserted on comes from the module that owns it, so a test cannot pin
a second, independently-drifting copy of the settings file's own vocabulary.
"""

from __future__ import annotations

import json
import subprocess
import tomllib
from pathlib import Path

import pytest

import plan_item_mode
from plan_item_mode import (
    COMMITTED_DEFAULTS_PATH,
    PERSONAL_SETTINGS_PATH,
    CommandLineOption,
    ExecutionMode,
    ExitCode,
    MalformedModeSettingsError,
    ModeSetting,
    ModeSource,
    PlanItemSkill,
    Subcommand,
    UnknownModeError,
)
from scratch_repository import ScratchRepository

MODE_SCRIPT_FILENAME = Path(plan_item_mode.__file__).name
"""
The script under test, named from the module itself rather than retyped.
"""

CONFIGURATION_SCRIPT_FILENAME = "resolve-personal-notes-config.sh"
"""
The shell configuration the script sources to find the notes branch.
"""

NOTES_WRITER_SCRIPT_FILENAME = "write-personal-notes-file.sh"
"""
The helper ``set`` pushes the personal settings file through.
"""

PLACEHOLDER_NOTES_PATH = ".claude/personal/cram-notes.md"
"""
A file to give the scratch notes branch, which cannot be published empty.
"""

# %% fixtures


@pytest.fixture
def mode_repository(scratch_repository: ScratchRepository) -> ScratchRepository:
    """
    A scratch repository carrying the scripts this module drives and a published notes
    branch that has no personal settings file yet.

    :param scratch_repository: The initialized scratch repository and notes remote.
    :return: The same repository, ready to resolve a mode in.
    """
    scratch_repository.install_hook_scripts(
        CONFIGURATION_SCRIPT_FILENAME,
        NOTES_WRITER_SCRIPT_FILENAME,
        MODE_SCRIPT_FILENAME,
        COMMITTED_DEFAULTS_PATH.name,
    )
    scratch_repository.write("README.md", "scratch repo\n")
    scratch_repository.commit_everything("initial commit")
    scratch_repository.publish_notes_branch({PLACEHOLDER_NOTES_PATH: "notes\n"})
    scratch_repository.resolve_notes_remote_to()
    return scratch_repository


def run_mode(
    repository: ScratchRepository, *arguments: str
) -> subprocess.CompletedProcess[str]:
    """
    Run the scratch layout's plan_item_mode.py with *arguments*.

    :param repository: A fixture-built scratch repository.
    :param arguments: CLI arguments to pass.
    :return: The finished subprocess.
    """
    return subprocess.run(
        [
            "python3",
            str(repository.project_root / ".claude" / "hooks" / MODE_SCRIPT_FILENAME),
            *arguments,
        ],
        cwd=repository.project_root,
        capture_output=True,
        text=True,
    )


def resolve(
    repository: ScratchRepository, skill: PlanItemSkill, *arguments: str
) -> dict[str, object]:
    """
    Resolve *skill*'s mode and parse the report, failing the test if the run refused.

    :param repository: A fixture-built scratch repository.
    :param skill: The skill whose mode to resolve.
    :param arguments: Further CLI arguments, such as a requested mode.
    :return: The parsed report.
    """
    finished = run_mode(
        repository, Subcommand.RESOLVE, CommandLineOption.SKILL, skill, *arguments
    )
    assert finished.returncode == ExitCode.SUCCESS, finished.stderr
    return json.loads(finished.stdout)


def published_settings(repository: ScratchRepository, tmp_path: Path) -> dict[str, str]:
    """
    Read the personal settings file as it actually is on the notes branch, rather than
    what a run reported.

    :param repository: A fixture-built scratch repository.
    :param tmp_path: Somewhere to check the notes branch out into.
    :return: The parsed settings.
    """
    checkout = repository.clone_notes_branch(tmp_path / "published")
    return tomllib.loads((checkout / PERSONAL_SETTINGS_PATH).read_text())


# %% resolving with nothing configured


@pytest.mark.parametrize("skill", list(PlanItemSkill))
def test_a_clone_with_no_personal_settings_runs_on_its_own(
    mode_repository: ScratchRepository, skill: PlanItemSkill
):
    """
    The zero-configuration default is to implement without asking, for every skill.
    """
    report = resolve(mode_repository, skill)
    assert report["mode"] == ExecutionMode.AUTO
    assert report["source"] == ModeSource.COMMITTED_DEFAULT
    assert report["skill"] == skill
    assert report["setting_key"] == skill.setting_key


def test_the_report_names_where_a_personal_setting_would_go(
    mode_repository: ScratchRepository,
):
    """
    The path is reported even when the file is absent, so a session can tell the user
    where to put one without deriving the path itself.
    """
    report = resolve(mode_repository, PlanItemSkill.KICKOFF)
    assert report["personal_setting_path"] == str(PERSONAL_SETTINGS_PATH)


def test_an_unreachable_notes_branch_still_resolves_to_the_default(
    scratch_repository: ScratchRepository,
):
    """
    Someone who has never set personal notes up still gets a working skill: an absent
    setting is unset, not an error.
    """
    scratch_repository.install_hook_scripts(
        CONFIGURATION_SCRIPT_FILENAME,
        NOTES_WRITER_SCRIPT_FILENAME,
        MODE_SCRIPT_FILENAME,
        COMMITTED_DEFAULTS_PATH.name,
    )
    scratch_repository.write("README.md", "scratch repo\n")
    scratch_repository.commit_everything("initial commit")

    report = resolve(scratch_repository, PlanItemSkill.KICKOFF)
    assert report["mode"] == ExecutionMode.AUTO
    assert report["source"] == ModeSource.COMMITTED_DEFAULT


# %% resolving with a personal setting

# Every setting below pins a mode the committed default is not, so a resolution that
# silently fell through to the default would fail rather than coincide with the answer.


def test_a_personal_setting_overrides_the_committed_default(
    mode_repository: ScratchRepository,
):
    """
    Pinning one skill's mode changes that skill and leaves the other on the default.
    """
    mode_repository.update_notes_branch_file(
        str(PERSONAL_SETTINGS_PATH),
        f'{PlanItemSkill.KICKOFF.setting_key} = "{ExecutionMode.PLAN}"\n',
    )

    kickoff = resolve(mode_repository, PlanItemSkill.KICKOFF)
    assert kickoff["mode"] == ExecutionMode.PLAN
    assert kickoff["source"] == ModeSource.PERSONAL_SETTING

    resolve_skill = resolve(mode_repository, PlanItemSkill.RESOLVE)
    assert resolve_skill["mode"] == ExecutionMode.AUTO
    assert resolve_skill["source"] == ModeSource.COMMITTED_DEFAULT


def test_the_invocation_argument_beats_the_personal_setting(
    mode_repository: ScratchRepository,
):
    """
    A mode asked for on the command line wins, so one run can depart from the setting.
    """
    mode_repository.update_notes_branch_file(
        str(PERSONAL_SETTINGS_PATH),
        f'{PlanItemSkill.KICKOFF.setting_key} = "{ExecutionMode.PLAN}"\n',
    )

    report = resolve(
        mode_repository,
        PlanItemSkill.KICKOFF,
        CommandLineOption.REQUESTED,
        ExecutionMode.ASK,
    )
    assert report["mode"] == ExecutionMode.ASK
    assert report["source"] == ModeSource.INVOCATION_ARGUMENT


# %% refusals


def test_a_mode_the_enum_does_not_name_is_refused_in_the_personal_file(
    mode_repository: ScratchRepository,
):
    """
    A typo fails loudly: a silent fall back to the default is indistinguishable from the
    setting having worked.
    """
    mode_repository.update_notes_branch_file(
        str(PERSONAL_SETTINGS_PATH), f'{PlanItemSkill.KICKOFF.setting_key} = "atuo"\n'
    )

    finished = run_mode(
        mode_repository,
        Subcommand.RESOLVE,
        CommandLineOption.SKILL,
        PlanItemSkill.KICKOFF,
    )
    assert finished.returncode == ExitCode.UNKNOWN_MODE
    assert finished.stdout == ""
    assert ExitCode.UNKNOWN_MODE.name_for_a_caller in finished.stderr


def test_a_mode_the_enum_does_not_name_is_refused_on_the_command_line(
    mode_repository: ScratchRepository,
):
    """
    The command line is validated the same way the file is, so both refusals read alike.
    """
    finished = run_mode(
        mode_repository,
        Subcommand.RESOLVE,
        CommandLineOption.SKILL,
        PlanItemSkill.KICKOFF,
        CommandLineOption.REQUESTED,
        "go",
    )
    assert finished.returncode == ExitCode.UNKNOWN_MODE
    assert finished.stdout == ""


def test_a_personal_settings_file_that_will_not_parse_is_refused(
    mode_repository: ScratchRepository,
):
    """
    Broken syntax is reported rather than read as an absent setting.
    """
    mode_repository.update_notes_branch_file(
        str(PERSONAL_SETTINGS_PATH), "kickoff_mode = \n"
    )

    finished = run_mode(
        mode_repository,
        Subcommand.RESOLVE,
        CommandLineOption.SKILL,
        PlanItemSkill.KICKOFF,
    )
    assert finished.returncode == ExitCode.MALFORMED_MODE_SETTINGS
    assert finished.stdout == ""


def test_a_usage_error_exits_with_the_status_the_enum_reserves_for_it(
    mode_repository: ScratchRepository,
):
    """
    2 is left out of the refusals this tool raises because ``argparse`` already owns it.
    """
    finished = run_mode(
        mode_repository, Subcommand.RESOLVE, CommandLineOption.SKILL, "bogus"
    )

    assert finished.returncode == ExitCode.USAGE
    assert finished.stdout == ""


def test_each_refusal_carries_its_own_exit_code():
    codes = {
        UnknownModeError: ExitCode.UNKNOWN_MODE,
        MalformedModeSettingsError: ExitCode.MALFORMED_MODE_SETTINGS,
        plan_item_mode.SettingsWriteRefusedError: ExitCode.SETTINGS_WRITE_REFUSED,
    }
    assert {error: error.exit_code for error in codes} == codes


def test_a_refusal_composes_its_message_from_its_own_fields():
    refusal = UnknownModeError(
        setting=ModeSetting(key=PlanItemSkill.KICKOFF.setting_key, value="go")
    )
    assert refusal.error_message() in str(refusal)
    assert refusal.suggest_correction() in str(refusal)


def test_a_refusal_names_the_setting_the_bad_value_came_from():
    """
    The key is a field of the setting rather than an argument passed alongside its
    value, so a refusal can always say which setting or option it is about.
    """
    setting = ModeSetting(key=PlanItemSkill.RESOLVE.setting_key, value="atuo")

    assert setting.key in UnknownModeError(setting=setting).error_message()
    with pytest.raises(UnknownModeError):
        setting.parse()


# %% writing the setting


def test_setting_one_mode_preserves_the_other(
    mode_repository: ScratchRepository, tmp_path: Path
):
    """
    Writing one key must not clobber the key the user was not changing.
    """
    mode_repository.update_notes_branch_file(
        str(PERSONAL_SETTINGS_PATH),
        f'{PlanItemSkill.KICKOFF.setting_key} = "{ExecutionMode.PLAN}"\n',
    )

    finished = run_mode(
        mode_repository,
        Subcommand.SET,
        CommandLineOption.SKILL,
        PlanItemSkill.RESOLVE,
        CommandLineOption.MODE,
        ExecutionMode.ASK,
    )
    assert finished.returncode == ExitCode.SUCCESS, finished.stderr

    assert published_settings(mode_repository, tmp_path) == {
        PlanItemSkill.KICKOFF.setting_key: ExecutionMode.PLAN,
        PlanItemSkill.RESOLVE.setting_key: ExecutionMode.ASK,
    }


def test_setting_a_mode_with_no_personal_file_yet_seeds_it_from_the_defaults(
    mode_repository: ScratchRepository, tmp_path: Path
):
    """
    The first write records the committed default for every key it did not set, so the
    file always states the whole configuration rather than half of it.
    """
    finished = run_mode(
        mode_repository,
        Subcommand.SET,
        CommandLineOption.SKILL,
        PlanItemSkill.KICKOFF,
        CommandLineOption.MODE,
        ExecutionMode.ASK,
    )
    assert finished.returncode == ExitCode.SUCCESS, finished.stderr

    assert published_settings(mode_repository, tmp_path) == {
        PlanItemSkill.KICKOFF.setting_key: ExecutionMode.ASK,
        PlanItemSkill.RESOLVE.setting_key: ExecutionMode.AUTO,
    }


def notes_branch_commit_count(
    repository: ScratchRepository, tmp_path: Path, checkout_name: str
) -> int:
    """
    Count the commits on the published notes branch.

    :param repository: A fixture-built scratch repository.
    :param tmp_path: Somewhere to check the notes branch out into.
    :param checkout_name: A directory name not yet used in this test.
    :return: How many commits the branch carries.
    """
    checkout = repository.clone_notes_branch(tmp_path / checkout_name)
    counted = repository.run_git("rev-list", "--count", "HEAD", cwd=checkout)
    return int(counted.stdout.strip())


def test_one_run_can_set_every_skill_at_once(
    mode_repository: ScratchRepository, tmp_path: Path
):
    """
    Naming both skills writes once, so changing the whole configuration costs one push
    to the notes branch rather than one per skill.
    """
    before = notes_branch_commit_count(mode_repository, tmp_path, "before")

    finished = run_mode(
        mode_repository,
        Subcommand.SET,
        CommandLineOption.SKILL,
        PlanItemSkill.KICKOFF,
        CommandLineOption.SKILL,
        PlanItemSkill.RESOLVE,
        CommandLineOption.MODE,
        ExecutionMode.PLAN,
    )
    assert finished.returncode == ExitCode.SUCCESS, finished.stderr

    assert published_settings(mode_repository, tmp_path) == {
        skill.setting_key: ExecutionMode.PLAN for skill in PlanItemSkill
    }
    assert json.loads(finished.stdout)["skills"] == [
        PlanItemSkill.KICKOFF,
        PlanItemSkill.RESOLVE,
    ]
    assert notes_branch_commit_count(mode_repository, tmp_path, "after") == before + 1


def test_a_written_setting_is_what_the_next_resolve_reads(
    mode_repository: ScratchRepository,
):
    """
    The round trip closes: what ``set`` pushes is what ``resolve`` picks up.
    """
    run_mode(
        mode_repository,
        Subcommand.SET,
        CommandLineOption.SKILL,
        PlanItemSkill.KICKOFF,
        CommandLineOption.MODE,
        ExecutionMode.PLAN,
    )

    report = resolve(mode_repository, PlanItemSkill.KICKOFF)
    assert report["mode"] == ExecutionMode.PLAN
    assert report["source"] == ModeSource.PERSONAL_SETTING


# %% the committed defaults


def test_every_skill_names_a_key_the_committed_defaults_define():
    """
    The enum and the shipped TOML are edited in different files; this is what stops one
    gaining a skill the other has never heard of.
    """
    defaults = tomllib.loads(
        (
            Path(plan_item_mode.__file__).parent.parent.parent / COMMITTED_DEFAULTS_PATH
        ).read_text()
    )
    assert set(defaults) == {skill.setting_key for skill in PlanItemSkill}
