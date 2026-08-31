"""
Tests for setup_steps.py: which steps it prints, and the values it fills them with.

The subject is the list a first-time user is left holding, so the assertions are
about the steps' content - the repository they name, the commands they hand over,
and which variables they say to paste - rather than about the exact prose framing
them.
"""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import pytest

import setup_steps
from setup_steps import (
    COMMENT_MARKER,
    ForkLabels,
    git_value,
    PersistentVariables,
    PersonalNotesSetting,
    Host,
    Repository,
    RepositoryAccess,
    RepositoryLabel,
    SetupChecklist,
    SetupLink,
    resolve_repository,
)

from scratch_repository import SCRUBBED_ENVIRONMENT_PREFIXES, ScratchRepository

PLAN_DASHBOARD_DIRECTORY = (
    Path(setup_steps.__file__).parent.parent / "skills" / "plan-dashboard"
)

sys.path.insert(0, str(PLAN_DASHBOARD_DIRECTORY))

from build_dashboard import PullRequestLabel  # noqa: E402

FORK = Repository(owner="some-user", name="some-repository")

FORK_REMOTE_URL = f"https://github.com/{FORK.full_name}.git"

RESOLVE_CONFIG_SCRIPT = (
    Path(setup_steps.__file__).parent / "resolve-personal-notes-config.sh"
)

SETUP_STEPS_SCRIPT_NAME = "setup_steps.py"


@pytest.fixture
def clone_with_fork_remote(scratch_repository: ScratchRepository) -> ScratchRepository:
    """
    A scratch clone whose ``origin`` is a GitHub URL, the shape every step reads from.

    :param scratch_repository: The bare scratch clone to add the remote to.
    :return: The same clone.
    """
    scratch_repository.run_git("remote", "add", "origin", FORK_REMOTE_URL)
    return scratch_repository


# %% the pages the steps link to


def test_every_link_is_served_by_a_host_this_module_names() -> None:
    """
    A page is composed from the host serving it rather than spelling that host again, so
    the three services are named once between them.
    """
    for link in SetupLink:
        assert any(link.startswith(host.url) for host in Host), link


def test_the_repository_links_use_the_same_host_as_everything_else() -> None:
    """
    A link that varies with the repository is built rather than listed, and builds from
    the same place the fixed ones do.
    """
    assert FORK.labels_url.startswith(Host.GITHUB.url)


def test_the_module_writes_the_scheme_once() -> None:
    """
    Every link resolves through :attr:`Host.url`, so a second ``https://`` in the source
    is a page spelling out a host that is already named.
    """
    source = Path(setup_steps.__file__).read_text(encoding="utf-8")
    assert source.count('"https://') == 1


# %% reading the repository out of a remote


@pytest.mark.parametrize(
    "url",
    [
        FORK_REMOTE_URL,
        f"https://github.com/{FORK.full_name}",
        f"git@github.com:{FORK.full_name}.git",
        f"ssh://git@github.com/{FORK.full_name}.git",
    ],
)
def test_a_github_remote_url_names_its_repository(url: str) -> None:
    """
    Every URL form git writes a GitHub remote as resolves to the same repository.
    """
    assert Repository.from_remote_url(url) == FORK


@pytest.mark.parametrize(
    "url",
    ["https://gitlab.com/some-user/some-repository.git", "/srv/mirrors/bare.git"],
)
def test_a_remote_that_is_not_a_github_repository_names_none(url: str) -> None:
    """
    A remote pointing anywhere but a GitHub repository yields no repository at all,
    rather than a guess assembled from its path.
    """
    assert Repository.from_remote_url(url) is None


def test_the_repository_comes_from_the_notes_remotes_url(
    clone_with_fork_remote: ScratchRepository,
) -> None:
    """
    A remote *name* is resolved through git to the URL it stands for.
    """
    assert resolve_repository(clone_with_fork_remote.project_root, "origin") == FORK


def test_a_notes_remote_given_as_a_url_needs_no_remote_to_exist(
    scratch_repository: ScratchRepository,
) -> None:
    """
    The remote setting may hold a raw URL, in a clone that has never added that fork
    as a remote - the case the whole configuration exists to cover.
    """
    assert resolve_repository(scratch_repository.project_root, FORK_REMOTE_URL) == FORK


def test_a_clone_with_no_github_remote_resolves_no_repository(
    scratch_repository: ScratchRepository,
) -> None:
    """
    Nothing is invented when the clone names no GitHub repository.
    """
    assert resolve_repository(scratch_repository.project_root, "origin") is None


# %% reading a value out of the clone


def test_a_value_git_cannot_answer_is_reported_rather_than_raised(
    scratch_repository: ScratchRepository,
) -> None:
    """
    Both callers fall through to something else when git has no answer, so an unset key
    and an unknown remote have to come back as nothing at all.
    """
    root = scratch_repository.project_root
    assert git_value(root, "config", "--get", "claude.notSet") is None
    assert git_value(root, "remote", "get-url", "no-such-remote") is None


def test_a_value_git_does_answer_comes_back_trimmed(
    clone_with_fork_remote: ScratchRepository,
) -> None:
    """
    The answer is the value itself, without the newline git writes after it.
    """
    root = clone_with_fork_remote.project_root
    assert git_value(root, "remote", "get-url", "origin") == FORK_REMOTE_URL


# %% the labels step


def test_every_label_the_tooling_relies_on_is_offered() -> None:
    """
    This enum and the dashboard's own name the same labels, member for member.

    Asserted against ``build_dashboard.PullRequestLabel`` itself, by member name as well
    as by value: a label added there and not here would leave a fork missing one, with
    nothing else to catch it.
    """
    assert {label.name: label.value for label in RepositoryLabel} == {
        member.name: member.value for member in PullRequestLabel
    }


def test_each_label_carries_the_description_it_is_created_with() -> None:
    """
    Every label says what it means, and the creation command names the fork.
    """
    for label in RepositoryLabel:
        assert label.purpose
        assert label.creation_command(FORK) == (
            f"gh label create {label.value} --repo {FORK.full_name} "
            f'--description "{label.purpose}"'
        )


def test_the_labels_step_leads_with_the_page_that_creates_them_by_hand() -> None:
    """
    The ``gh`` CLI is not present everywhere, so the step offers the fork's own labels
    page before the commands that need it.
    """
    instructions = ForkLabels(FORK).instructions()
    assert FORK.labels_url in instructions[0]
    assert [line for line in instructions if line.startswith("gh label create")] == [
        label.creation_command(FORK) for label in RepositoryLabel
    ]


# %% the access step


def test_the_access_step_names_what_the_authorization_covers() -> None:
    """
    Pushing and opening pull requests come from this one authorization, so the step has
    to say so - a reader who reads it as read-only access has no other step to look for.
    """
    step = RepositoryAccess(FORK)
    covered = f"{step.reason} {' '.join(step.instructions())}"
    assert "push" in covered
    assert "pull request" in covered


def test_the_access_step_leads_with_the_authorization_a_user_performs() -> None:
    """
    The owner's connector toggle and the GitHub App are conditions around the
    authorization, not alternatives to it, so it comes first.
    """
    instructions = RepositoryAccess(FORK).instructions()
    assert SetupLink.GITHUB_AUTHORIZATION in instructions[0]
    assert any(SetupLink.ORGANIZATION_CONNECTORS in line for line in instructions[1:])


def test_the_access_step_says_the_github_app_is_not_the_grant() -> None:
    """
    Installing the app is skippable and grants nothing, so a reader must not take it for
    the access step and must not skip it expecting auto-fix anyway.
    """
    application = [
        line for line in RepositoryAccess(FORK).instructions() if "App" in line
    ]
    assert len(application) == 1
    assert "auto-fix" in application[0]
    assert "not what grants access" in application[0]


# %% the variables step


def test_a_clone_left_on_the_defaults_has_nothing_to_paste(
    scratch_repository: ScratchRepository,
) -> None:
    """
    The variables exist to move a setting off its default, so an untouched clone is told
    there is nothing to carry rather than handed the defaults back.
    """
    step = PersistentVariables.resolve(scratch_repository.project_root, {})
    assert step.variable_lines == ()


def test_only_the_settings_moved_off_their_defaults_are_listed(
    scratch_repository: ScratchRepository,
) -> None:
    """
    A setting changed in git config is printed as its environment variable; the two left
    alone are not.
    """
    scratch_repository.run_git(
        "config", PersonalNotesSetting.REMOTE.git_config_key, FORK_REMOTE_URL
    )
    step = PersistentVariables.resolve(scratch_repository.project_root, {})
    assert step.variable_lines == (
        f"{PersonalNotesSetting.REMOTE.environment_variable}={FORK_REMOTE_URL}",
    )


def test_a_setting_already_in_the_environment_is_listed_too(
    scratch_repository: ScratchRepository,
) -> None:
    """
    A value coming from the environment is still worth pasting into a *persistent*
    variable list, which is a different place from the current session's environment.
    """
    branch = "claude/my-own-notes"
    step = PersistentVariables.resolve(
        scratch_repository.project_root,
        {PersonalNotesSetting.BRANCH.environment_variable: branch},
    )
    assert step.variable_lines == (
        f"{PersonalNotesSetting.BRANCH.environment_variable}={branch}",
    )


def test_a_value_carrying_a_comment_marker_is_quoted(
    scratch_repository: ScratchRepository,
) -> None:
    """
    An environment's variable list reads an unquoted value only as far as its comment
    marker, so a branch or path carrying one has to be quoted or it is silently set to a
    prefix of itself.
    """
    branch = f"claude/notes{COMMENT_MARKER}2"
    step = PersistentVariables.resolve(
        scratch_repository.project_root,
        {PersonalNotesSetting.BRANCH.environment_variable: branch},
    )
    assert step.variable_lines == (
        f'{PersonalNotesSetting.BRANCH.environment_variable}="{branch}"',
    )


def test_a_value_with_nothing_to_escape_is_left_bare(
    scratch_repository: ScratchRepository,
) -> None:
    """
    Quoting is for the value that needs it, so an ordinary one is pasted as written.
    """
    branch = "claude/my-own-notes"
    step = PersistentVariables.resolve(
        scratch_repository.project_root,
        {PersonalNotesSetting.BRANCH.environment_variable: branch},
    )
    assert step.variable_lines == (
        f"{PersonalNotesSetting.BRANCH.environment_variable}={branch}",
    )


def test_git_config_outranks_the_environment(
    scratch_repository: ScratchRepository,
) -> None:
    """
    The precedence the hooks apply - git config, then environment, then default.
    """
    scratch_repository.run_git(
        "config", PersonalNotesSetting.PATH.git_config_key, "configured/notes.md"
    )
    resolved = PersonalNotesSetting.PATH.resolve(
        scratch_repository.project_root,
        {PersonalNotesSetting.PATH.environment_variable: "environment/notes.md"},
    )
    assert resolved == "configured/notes.md"


def test_the_settings_are_the_ones_the_hooks_resolve() -> None:
    """
    Every key, variable and default named here is named by resolve-personal-notes-
    config.sh, which is what the hooks themselves read.
    """
    configuration = RESOLVE_CONFIG_SCRIPT.read_text(encoding="utf-8")
    missing = [
        value
        for setting in PersonalNotesSetting
        for value in (
            setting.git_config_key,
            setting.environment_variable,
            setting.default,
        )
        if value not in configuration
    ]
    assert missing == []


# %% the list as a whole


def test_a_resolvable_fork_gets_every_step(
    clone_with_fork_remote: ScratchRepository,
) -> None:
    """
    All three steps are printed, with the two repository-specific ones first.
    """
    checklist = SetupChecklist.for_clone(clone_with_fork_remote.project_root, {})
    assert checklist.steps == (
        ForkLabels(FORK),
        RepositoryAccess(FORK),
        PersistentVariables(variable_lines=()),
    )


def test_the_repository_steps_are_dropped_when_no_fork_resolves(
    scratch_repository: ScratchRepository,
) -> None:
    """
    A clone naming no GitHub repository is given the step that still applies to it,
    rather than steps addressed to a repository nobody could name.
    """
    checklist = SetupChecklist.for_clone(scratch_repository.project_root, {})
    assert checklist.steps == (PersistentVariables(variable_lines=()),)


def test_running_the_script_prints_the_steps_for_its_own_clone(
    clone_with_fork_remote: ScratchRepository,
) -> None:
    """
    The script resolves the clone it lives in, not the caller's working directory.
    """
    clone_with_fork_remote.install_hook_scripts(SETUP_STEPS_SCRIPT_NAME)
    script = (
        clone_with_fork_remote.project_root
        / ".claude"
        / "hooks"
        / SETUP_STEPS_SCRIPT_NAME
    )
    result = subprocess.run(
        [sys.executable, str(script)],
        cwd=clone_with_fork_remote.project_root.parent,
        capture_output=True,
        text=True,
        env={
            name: value
            for name, value in os.environ.items()
            if not name.startswith(SCRUBBED_ENVIRONMENT_PREFIXES)
        },
    )
    assert result.returncode == 0, result.stderr
    expected = SetupChecklist.for_clone(clone_with_fork_remote.project_root, {})
    assert result.stdout == expected.render() + "\n"
