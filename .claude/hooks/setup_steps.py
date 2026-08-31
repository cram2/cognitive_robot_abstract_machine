#!/usr/bin/env python3
"""
Prints the setup steps that live outside this clone, filled in for the fork the
clone actually points at.

check-setup.sh reports on everything a shell can inspect - the notes branch, the
recorded git identity, the dashboard dependencies. What it cannot reach is
everything held by a service rather than a file: labels in a GitHub repository,
Claude's access to that repository, and the variables a fresh-clone environment
has to carry. Those are the steps a first-time user is left holding, so they are
printed here as a short list with the values already substituted, rather than
described in prose the reader has to translate.

Usage:
    python3 .claude/hooks/setup_steps.py
"""

from __future__ import annotations

import dataclasses
import os
import subprocess
from abc import ABC, abstractmethod
from collections.abc import Mapping
from dataclasses import dataclass
from enum import Enum, StrEnum
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
"""
The repository this script describes, resolved from the script's own location so the
answer does not depend on the caller's working directory.
"""


class Host(StrEnum):
    """
    The services the printed steps send a reader to, each named once.

    A member is the bare host, which is how a remote URL carries it; :attr:`url` is
    the same host addressed over HTTPS, which is how a link carries it.
    """

    CLAUDE = "claude.ai"
    """
    Where Claude's own settings and cloud sessions live.
    """

    CLAUDE_DOCUMENTATION = "code.claude.com"
    """
    Where Claude Code's documentation lives.
    """

    GITHUB = "github.com"
    """
    Where the fork, its labels and its pull requests live.
    """

    @property
    def url(self) -> str:
        """
        This host addressed the way every link in this module addresses it.
        """
        return f"https://{self}"


class SetupLink(StrEnum):
    """
    Every fixed page the printed steps link to, composed from the host serving it.

    A link that varies with the repository is built by that repository instead - see
    :attr:`Repository.labels_url`.
    """

    GITHUB_AUTHORIZATION = (
        f"{Host.CLAUDE.url}/customize/connectors"
        "?auth_start=github&auth_start_force=1"
    )
    """
    Where a user grants Claude access to their own GitHub repositories.

    The query opens the GitHub authorization directly rather than the connector list.
    """

    ORGANIZATION_CONNECTORS = f"{Host.CLAUDE.url}/admin-settings/connectors"
    """
    Where an owner enables the GitHub connector for a Team or Enterprise Claude plan.

    Until they do, the authorization above offers no sign-in button at all.
    """

    ENVIRONMENT_SELECTOR = f"{Host.CLAUDE.url}/code"
    """
    Where an environment is edited, which is the selector rather than a settings page.
    """

    ENVIRONMENT_VARIABLES = (
        f"{Host.CLAUDE_DOCUMENTATION.url}/docs/en/cloud-environments"
        "#set-environment-variables"
    )
    """
    Where the environment-level variable list for cloud sessions is documented.

    The section rather than the page: the overview only mentions that an environment
    carries variables, and points here for the list itself.
    """


# %% reading a value out of the clone


def git_value(project_root: Path, *arguments: str) -> str | None:
    """
    Read a value git can answer for a clone, or nothing when it has no answer.

    An absent answer is an ordinary outcome here - an unset configuration key, a remote
    the clone never added - so it is reported rather than raised, and every caller in
    this module falls through to the next thing it knows.

    :param project_root: The clone to run in.
    :param arguments: The git subcommand and its arguments.
    :return: The trimmed output, or ``None`` when the command failed or said nothing.
    """
    completed = subprocess.run(
        ["git", *arguments],
        cwd=project_root,
        capture_output=True,
        text=True,
    )
    if completed.returncode != 0:
        return None
    return completed.stdout.strip() or None


# %% the settings whose values the printed steps depend on


@dataclass(frozen=True)
class PersonalNotesSettingSpecification:
    """
    Where one personal-notes setting is read from, and what it falls back to.

    resolve-personal-notes-config.sh is the definition of both the precedence and the
    defaults; this mirrors them because the shell file exports nothing a child process
    could read them from.
    """

    git_config_key: str
    """
    The git config key that takes precedence over everything else.
    """

    environment_variable: str
    """
    The environment variable read when no git config value is set.
    """

    default: str
    """
    The value in force when neither of the two above is set.
    """

    def resolve(self, project_root: Path, environment: Mapping[str, str]) -> str:
        """
        Read this setting's value as the hooks themselves would.

        :param project_root: The clone whose git config to read.
        :param environment: The environment to read the variable from.
        :return: The value in force.
        """
        configured = git_value(project_root, "config", "--get", self.git_config_key)
        if configured is not None:
            return configured
        return environment.get(self.environment_variable) or self.default


class PersonalNotesSetting(PersonalNotesSettingSpecification, Enum):
    """
    The three settings that decide where personal notes live, in the order the printed
    variable list uses them.

    Each member *is* a specification, so a setting's own key, variable and default are
    reached directly and no caller holds a parallel list of them.

    .. note::
       Both :meth:`__new__` and :meth:`__init__` are needed: the enum machinery builds a
       member's value by calling the mixed-in type with the member's value as arguments,
       which a specification with no defaulted field cannot survive.
    """

    def __new__(
        cls, specification: PersonalNotesSettingSpecification
    ) -> PersonalNotesSetting:
        """
        Make the member without letting the enum rebuild the specification from it.

        :param specification: Where this setting is read from.
        :return: The member, carrying that specification as its value.
        """
        member = object.__new__(cls)
        member._value_ = specification
        return member

    def __init__(self, specification: PersonalNotesSettingSpecification) -> None:
        """
        Carry the specification's values on the member itself.

        :param specification: Where this setting is read from.
        """
        for field in dataclasses.fields(PersonalNotesSettingSpecification):
            object.__setattr__(self, field.name, getattr(specification, field.name))

    REMOTE = PersonalNotesSettingSpecification(
        git_config_key="claude.personalNotesRemote",
        environment_variable="CLAUDE_PERSONAL_NOTES_REMOTE",
        default="origin",
    )
    """
    The remote, or raw URL, the notes branch lives on.
    """

    BRANCH = PersonalNotesSettingSpecification(
        git_config_key="claude.personalNotesBranch",
        environment_variable="CLAUDE_PERSONAL_NOTES_BRANCH",
        default="claude/personal-notes",
    )
    """
    The branch the notes are stored on.
    """

    PATH = PersonalNotesSettingSpecification(
        git_config_key="claude.personalNotesPath",
        environment_variable="CLAUDE_PERSONAL_NOTES_PATH",
        default=".claude/personal/cram-notes.md",
    )
    """
    Where on that branch the notes file sits.
    """


# %% the repository the steps are about


@dataclass(frozen=True)
class Repository:
    """
    A GitHub repository, named the way its URLs and the ``gh`` CLI name it.
    """

    owner: str
    """
    The user or organization that owns it.
    """

    name: str
    """
    The repository's own name.
    """

    @classmethod
    def from_remote_url(cls, url: str) -> Repository | None:
        """
        Read a repository out of a git remote URL, in either the HTTPS or the SSH form.

        :param url: The remote URL.
        :return: The repository, or ``None`` if the URL names no GitHub repository.
        """
        if Host.GITHUB not in url:
            return None
        path = url.split(Host.GITHUB, 1)[1].lstrip(":/").removesuffix(".git")
        segments = [segment for segment in path.split("/") if segment]
        if len(segments) != 2:
            return None
        return cls(owner=segments[0], name=segments[1])

    @property
    def full_name(self) -> str:
        """
        The ``owner/name`` form the ``gh`` CLI and GitHub's own interface use.
        """
        return f"{self.owner}/{self.name}"

    @property
    def labels_url(self) -> str:
        """
        The page where labels are created by hand.
        """
        return f"{Host.GITHUB.url}/{self.full_name}/labels"


def resolve_repository(project_root: Path, notes_remote: str) -> Repository | None:
    """
    Find the repository a user's pull requests and notes go to.

    :param project_root: The clone to read remotes from.
    :param notes_remote: The resolved notes remote, either a remote name or a URL.
    :return: The repository, or ``None`` when no remote resolves to a GitHub URL.
    """
    remote_url = git_value(project_root, "remote", "get-url", notes_remote)
    for candidate in (remote_url, notes_remote):
        if candidate is None:
            continue
        repository = Repository.from_remote_url(candidate)
        if repository is not None:
            return repository
    return None


# %% the labels a fork has to carry


class RepositoryLabel(StrEnum):
    """
    Every label this tooling reads or applies, and therefore every one a fork must
    carry, each with the description it is created with.

    A member is its own label name, so the set mirrors build_dashboard.py's
    ``PullRequestLabel`` member for member and value for value - held equal by a test
    rather than by an import, because that module needs the dashboard's dependencies
    installed, which is one of the things this script exists to run before.
    """

    purpose: str
    """
    What the label means, used as its description when it is created.
    """

    def __new__(cls, label: str, purpose: str) -> RepositoryLabel:
        """
        Make a member that is its own label name and carries what the label means.

        :param label: The label's name, as GitHub stores it.
        :param purpose: What it means, used as the description when it is created.
        :return: The member.
        """
        member = str.__new__(cls, label)
        member._value_ = label
        member.purpose = purpose
        return member

    MERGED = ("merged", "The changes landed even though GitHub never recorded a merge")
    """
    Read by the dashboard, which treats the label exactly like a real merge.
    """

    IN_REVIEW = ("in-review", "Under review")
    """
    Recognized so it does not read as an unknown label; no script acts on it yet.
    """

    BUG = ("bug", "A bug fix")
    """
    Applied by a session opening a bug-fix pull request, and shown as a dashboard chip.
    """

    def creation_command(self, repository: Repository) -> str:
        """
        The ``gh`` command that creates this label.

        :param repository: The repository to create it in.
        :return: The command, ready to paste.
        """
        return (
            f"gh label create {self.value} --repo {repository.full_name} "
            f'--description "{self.purpose}"'
        )


# %% the steps themselves


@dataclass(frozen=True)
class SetupStep(ABC):
    """
    One setup step that no script can perform or verify, because it changes a setting
    held by GitHub or by Claude rather than a file in this clone.
    """

    @property
    @abstractmethod
    def title(self) -> str:
        """
        What the step changes, in a few words.
        """

    @property
    @abstractmethod
    def reason(self) -> str:
        """
        What stops working until it is done.
        """

    @abstractmethod
    def instructions(self) -> list[str]:
        """
        What to do, one line per action.

        :return: The lines, printed under the title.
        """

    def render(self, number: int) -> str:
        """
        Render this step as the numbered block the reader sees.

        :param number: The step's position in the printed list.
        :return: The block, without a trailing newline.
        """
        lines = [f"{number}. {self.title}", f"   {self.reason}", ""]
        lines.extend(f"   {instruction}" for instruction in self.instructions())
        return "\n".join(lines)


@dataclass(frozen=True)
class ForkLabels(SetupStep):
    """
    Creating the labels this tooling reads and applies, in the user's fork.
    """

    repository: Repository
    """
    The fork the labels belong in.
    """

    @property
    def title(self) -> str:
        """See :attr:`SetupStep.title`."""
        return f"Add three labels to {self.repository.full_name}"

    @property
    def reason(self) -> str:
        """See :attr:`SetupStep.reason`."""
        return (
            "A fresh fork has none of them: dashboards misread landed work, and "
            "applying a label a repository lacks fails mid-pull-request."
        )

    def instructions(self) -> list[str]:
        """See :meth:`SetupStep.instructions`."""
        commands = [
            label.creation_command(self.repository) for label in RepositoryLabel
        ]
        return [
            f"By hand: {self.repository.labels_url}",
            "Or, if you have the gh CLI:",
            *commands,
        ]


@dataclass(frozen=True)
class RepositoryAccess(SetupStep):
    """
    Giving Claude access to the repository sessions read and open pull requests
    against.
    """

    repository: Repository
    """
    The repository the access has to cover.
    """

    @property
    def title(self) -> str:
        """See :attr:`SetupStep.title`."""
        return f"Connect GitHub, so Claude can work in {self.repository.full_name}"

    @property
    def reason(self) -> str:
        """See :attr:`SetupStep.reason`."""
        return (
            "This is what lets a session clone the fork, push a branch, open a pull "
            "request and comment on one - and read the state a dashboard is built from."
        )

    def instructions(self) -> list[str]:
        """See :meth:`SetupStep.instructions`."""
        return [
            f"Authorize GitHub: {SetupLink.GITHUB_AUTHORIZATION}",
            "One authorization covers every repository your GitHub account can see.",
            "If your Claude plan is Team or Enterprise, an owner enables the GitHub "
            f"connector first or no sign-in appears: "
            f"{SetupLink.ORGANIZATION_CONNECTORS}",
            "The Claude GitHub App is separate and optional: it adds auto-fix on pull "
            "requests, and is not what grants access.",
        ]


COMMENT_MARKER = "#"
"""
What starts a comment in the ``.env`` format an environment's variable list uses.
"""


def quoted_if_needed(value: str) -> str:
    """
    Quote a variable's value when leaving it bare would truncate it.

    An unquoted value is read to the first comment marker and the rest of the line is
    dropped, which is silent - the variable is set, to a prefix of what was pasted.

    :param value: The value to write after the ``=``.
    :return: The value, quoted only when it carries a comment marker.
    """
    if COMMENT_MARKER not in value:
        return value
    return f'"{value}"'


@dataclass(frozen=True)
class PersistentVariables(SetupStep):
    """
    Carrying the personal-notes settings into environments that clone fresh every
    session.
    """

    variable_lines: tuple[str, ...]
    """
    The ``NAME=value`` lines to paste, empty when every setting is still its default.
    """

    @classmethod
    def resolve(
        cls, project_root: Path, environment: Mapping[str, str]
    ) -> PersistentVariables:
        """
        Work out which settings this clone has moved off their defaults.

        :param project_root: The clone to read settings from.
        :param environment: The environment to read them from.
        :return: The step, carrying only the lines that differ from the defaults.
        """
        lines = []
        for setting in PersonalNotesSetting:
            value = setting.resolve(project_root, environment)
            if value != setting.default:
                lines.append(
                    f"{setting.environment_variable}={quoted_if_needed(value)}"
                )
        return cls(variable_lines=tuple(lines))

    @property
    def title(self) -> str:
        """See :attr:`SetupStep.title`."""
        return "Carry your settings into environments that clone fresh"

    @property
    def reason(self) -> str:
        """See :attr:`SetupStep.reason`."""
        return (
            "Claude Code on the web starts from a new clone every session, so git "
            "config set inside one is gone by the next."
        )

    def instructions(self) -> list[str]:
        """See :meth:`SetupStep.instructions`."""
        if not self.variable_lines:
            return ["Nothing to paste: every setting is still its default."]
        return [
            f"Open your environment at {SetupLink.ENVIRONMENT_SELECTOR} and paste these into "
            f"its variable list ({SetupLink.ENVIRONMENT_VARIABLES}):",
            *self.variable_lines,
        ]


# %% the checklist they add up to


HEADING = "These steps are yours - no script can do them:"
"""
The line introducing the printed list.
"""


@dataclass(frozen=True)
class SetupChecklist:
    """
    The steps one clone leaves to its user, in the order they should be done.
    """

    steps: tuple[SetupStep, ...]
    """
    The steps that apply to this clone.
    """

    @classmethod
    def for_clone(
        cls, project_root: Path, environment: Mapping[str, str]
    ) -> SetupChecklist:
        """
        Work out which steps a clone leaves to its user, and fill them in from it.

        The repository-specific steps are omitted when no GitHub repository can be
        resolved from the clone's remotes, since neither can be acted on without one.

        :param project_root: The clone the steps are about.
        :param environment: The environment its settings resolve from.
        :return: The checklist.
        """
        notes_remote = PersonalNotesSetting.REMOTE.resolve(project_root, environment)
        repository = resolve_repository(project_root, notes_remote)
        steps: list[SetupStep] = []
        if repository is not None:
            steps.extend([ForkLabels(repository), RepositoryAccess(repository)])
        steps.append(PersistentVariables.resolve(project_root, environment))
        return cls(steps=tuple(steps))

    def render(self) -> str:
        """
        Render the whole checklist as it is printed.

        :return: The full output.
        """
        blocks = [
            step.render(number) for number, step in enumerate(self.steps, start=1)
        ]
        return "\n\n".join([HEADING, *blocks])


def main() -> None:
    """
    Print the checklist for this clone.
    """
    print(SetupChecklist.for_clone(PROJECT_ROOT, os.environ).render())


if __name__ == "__main__":
    main()
