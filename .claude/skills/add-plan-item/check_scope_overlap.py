#!/usr/bin/env python3
"""
Answer the two mechanical questions in ``scope-decision.md``: do the paths a piece of
work touches already exist on the base branch, and which unlanded branches already touch
them.

The judgement itself - fold or split - stays with the reader; this only gathers the
evidence, so the check is actually run rather than eyeballed. Pure git, so it needs no
network access and no GitHub API call.

Usage:
    python3 check_scope_overlap.py \\
        --base <base-branch> \\
        --path <path> [--path <path> ...] \\
        --candidate '<label>=<branch>' [--candidate '<label>=<branch>' ...]

Prints a one-line JSON object to stdout:
    {"paths_absent_from_base": ["<path>", ...],
     "candidates": [{"label": "<label>", "branch": "<branch>",
                     "shared_paths": ["<path>", ...],
                     "changed_paths": ["<path>", ...]}, ...]}

``paths_absent_from_base`` holds the requested paths the base branch doesn't have yet -
non-empty means whichever unlanded branch introduces them is a candidate owner of the
work. ``shared_paths`` is the intersection of a candidate's own changes with the
requested paths; ``changed_paths`` is everything that candidate changes, for comparing
by purpose when two branches build the same thing under different names.

A branch that doesn't resolve raises :class:`UnknownBranchError` rather than being
reported as no overlap.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

CANDIDATE_SEPARATOR = "="
"""
Separates a ``--candidate`` argument's label from its branch name.
"""


class ReportKey(StrEnum):
    """
    The keys of the JSON document this script prints.

    Named here so the reader that consumes the report and the code that builds it cannot
    drift apart, and so a caller never spells one out as a bare string.
    """

    PATHS_ABSENT_FROM_BASE = "paths_absent_from_base"
    """
    The requested paths the base branch doesn't contain yet.
    """

    CANDIDATES = "candidates"
    """
    One overlap entry per requested candidate branch.
    """

    LABEL = "label"
    """
    How a candidate was named on the command line.
    """

    BRANCH = "branch"
    """
    The candidate's branch name.
    """

    SHARED_PATHS = "shared_paths"
    """
    The requested paths a candidate already changes.
    """

    CHANGED_PATHS = "changed_paths"
    """
    Every path a candidate changes, for comparing by purpose.
    """


@dataclass
class UnknownBranchError(ValueError):
    """
    Raised when a requested base or candidate branch doesn't resolve in the repository.
    """

    git_command: str
    """
    The git command that failed.
    """
    result_stderr: str
    """
    The git error message.
    """

    def __str__(self) -> str:
        return f"git {self.git_command} failed: {self.result_stderr}"


@dataclass
class MalformedCandidateError(ValueError):
    """
    Raised when a ``--candidate`` argument isn't in ``<label>=<branch>`` form.
    """
    expected_candidate: str
    """
    The expected format.
    """
    wrong_candidate: str
    """
    The argument that failed to parse.
    """

    def __str__(self) -> str:
        return f"expected {self.expected_candidate!r}, got {self.wrong_candidate!r}"


# %% inputs and results


@dataclass(frozen=True)
class Candidate:
    """
    An unlanded branch the work might already belong to.
    """

    label: str
    """
    How the candidate is named back to the reader, typically a plan item id or pull
    request title.
    """

    branch: str
    """
    The branch to compare against.
    """

    @classmethod
    def parse(cls, argument: str) -> Candidate:
        """
        Build a candidate from a ``<label>=<branch>`` command line argument.

        :param argument: The raw argument value.
        :raises MalformedCandidateError: If the separator is missing or either side is
            empty.
        :return: The parsed candidate.
        """
        label, separator, branch = argument.partition(CANDIDATE_SEPARATOR)
        if not separator or not label.strip() or not branch.strip():
            raise MalformedCandidateError(
                f"'<label>{CANDIDATE_SEPARATOR}<branch>'", argument
            )
        return cls(label.strip(), branch.strip())


@dataclass(frozen=True)
class CandidateOverlap:
    """
    What one candidate branch changes, and how much of that the work would touch too.
    """

    candidate: Candidate
    """
    The branch this overlap was computed for.
    """

    shared_paths: list[str]
    """
    The requested paths this candidate already changes.
    """

    changed_paths: list[str]
    """
    Every path this candidate changes, relative to its merge base with the base branch.
    """

    def as_json(self) -> dict[str, Any]:
        """
        Render the overlap in the output shape documented in the module docstring.
        """
        return {
            ReportKey.LABEL: self.candidate.label,
            ReportKey.BRANCH: self.candidate.branch,
            ReportKey.SHARED_PATHS: self.shared_paths,
            ReportKey.CHANGED_PATHS: self.changed_paths,
        }


@dataclass(frozen=True)
class ScopeReport:
    """
    The full evidence for one scope decision.
    """

    paths_absent_from_base: list[str]
    """
    The requested paths the base branch doesn't contain yet.
    """

    candidates: list[CandidateOverlap]
    """
    One entry per requested candidate, in the order they were given.
    """

    def as_json(self) -> dict[str, Any]:
        """
        Render the report in the output shape documented in the module docstring.
        """
        return {
            ReportKey.PATHS_ABSENT_FROM_BASE: self.paths_absent_from_base,
            ReportKey.CANDIDATES: [
                overlap.as_json() for overlap in self.candidates
            ],
        }


# %% git plumbing


def run_git(repository_path: Path, *arguments: str) -> str:
    """
    Run git in *repository_path* and return its standard output.

    :param repository_path: The repository to run in.
    :param arguments: The arguments to pass to git.
    :raises UnknownBranchError: If git reports a revision it cannot resolve.
    :return: Git's standard output, stripped of the trailing newline.
    """
    result = subprocess.run(
        ["git", *arguments],
        cwd=repository_path,
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise UnknownBranchError(
            f"git {' '.join(arguments)}", result.stderr.strip()
        )
    return result.stdout.rstrip("\n")


def paths_present_on(
        repository_path: Path, branch: str, paths: Sequence[str]
) -> set[str]:
    """
    Report which of *paths* the given branch's tree already contains.

    :param repository_path: The repository to inspect.
    :param branch: The branch whose tree to look in.
    :param paths: The paths to look for.
    :raises UnknownBranchError: If *branch* doesn't resolve.
    :return: The subset of *paths* that exist there.
    """
    listing = run_git(
        repository_path, "ls-tree", "-r", "--name-only", branch, "--", *paths
    )
    found = {line for line in listing.splitlines() if line}
    return {path for path in paths if path.rstrip("/") in found or path in found}


def changed_paths_since_base(
        repository_path: Path, base_branch: str, branch: str
) -> list[str]:
    """
    List everything *branch* changes relative to its merge base with *base_branch*.

    Using the merge base rather than the base branch's tip keeps commits that landed on
    the base after *branch* forked out of the result.

    :param repository_path: The repository to inspect.
    :param base_branch: The branch the work ultimately targets.
    :param branch: The candidate branch.
    :raises UnknownBranchError: If either branch doesn't resolve.
    :return: The changed paths, in git's own order.
    """
    merge_base = run_git(repository_path, "merge-base", base_branch, branch)
    listing = run_git(repository_path, "diff", "--name-only", f"{merge_base}..{branch}")
    return [line for line in listing.splitlines() if line]


# %% the report


def build_scope_report(
        repository_path: Path,
        base_branch: str,
        paths: Sequence[str],
        candidates: Sequence[Candidate],
) -> ScopeReport:
    """
    Gather the evidence for one scope decision.

    :param repository_path: The repository to inspect.
    :param base_branch: The branch the work ultimately targets.
    :param paths: The paths the work would create or modify.
    :param candidates: The unlanded branches to compare against.
    :raises UnknownBranchError: If the base or any candidate branch doesn't resolve.
    :return: The assembled report.
    """
    present = paths_present_on(repository_path, base_branch, paths)
    requested = set(paths)
    overlaps = []
    for candidate in candidates:
        changed = changed_paths_since_base(
            repository_path, base_branch, candidate.branch
        )
        overlaps.append(
            CandidateOverlap(
                candidate,
                [path for path in changed if path in requested],
                changed,
            )
        )
    return ScopeReport([path for path in paths if path not in present], overlaps)


def main() -> int:
    """
    Parse the command line, build the report, and print it as one line of JSON.

    :return: The process exit code.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--repository",
        default=".",
        help="Repository to inspect (default: the current directory)",
    )
    parser.add_argument(
        "--base", required=True, help="The branch the work ultimately targets"
    )
    parser.add_argument(
        "--path",
        action="append",
        required=True,
        dest="paths",
        help="A path the work would create or modify; repeat for each one",
    )
    parser.add_argument(
        "--candidate",
        action="append",
        default=[],
        dest="candidates",
        help="An unlanded branch to compare against, as '<label>=<branch>'; repeat",
    )
    arguments = parser.parse_args()

    try:
        candidates = [Candidate.parse(argument) for argument in arguments.candidates]
        report = build_scope_report(
            Path(arguments.repository), arguments.base, arguments.paths, candidates
        )
    except (UnknownBranchError, MalformedCandidateError) as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps(report.as_json()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
