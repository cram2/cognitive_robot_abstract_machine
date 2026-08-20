"""
The one thing about the maintenance skill worth asserting from code.

The skill is instructions, so most of what it says can only be checked by reading it.
The exception is what it must *not* say: it runs on whichever fork invoked it, so a
repository named in it is an instruction to operate on somebody else's. That is an
absence, computed from this checkout's own remotes rather than from a string written
here, which is what makes it worth a test where a prose assertion would not be.
"""

from __future__ import annotations

import subprocess
from pathlib import Path

from stack import CONFIGURATION_PATH, Repository, _configuration_values

MAINTENANCE_SKILL_DOCUMENT = (
    Path(__file__).parents[3] / ".claude/skills/stacked-pr-maintenance/SKILL.md"
)
"""
The instructions a maintenance pass follows.
"""


def candidate_forks() -> set[Repository]:
    """
    Every repository this checkout could be operating on: the ones its remotes name,
    minus the upstream that is the same for everybody.

    Read from the remotes rather than from the resolved ``fork_repository`` so the check
    still has something to assert on a clone nobody has run setup on - which is every
    fresh CI checkout.

    :return: The candidate forks, empty if the checkout has no repository remote at all.
    """
    upstream = Repository.parse(
        _configuration_values(CONFIGURATION_PATH)["upstream_repository"]
    )
    listed = subprocess.run(
        ["git", "remote"], capture_output=True, text=True, check=True
    ).stdout.split()
    urls = (
        subprocess.run(
            ["git", "remote", "get-url", name],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
        for name in listed
    )
    named = {
        Repository.from_remote_url(url)
        for url in urls
        if Repository.names_a_repository(url)
    }
    return named - {upstream}


def test_the_skill_names_no_fork_of_its_own():
    """
    The fork is configuration, so the skill has to read it rather than spell it out.
    """
    skill = MAINTENANCE_SKILL_DOCUMENT.read_text()

    for fork in candidate_forks():
        assert fork.owner not in skill
        assert str(fork) not in skill


def test_the_skill_restores_the_tooling_without_writing_the_index():
    """
    ``git checkout <ref> -- <path>`` writes the index as well as the working tree, so on
    a branch that does not carry the tooling the files end up staged - and the next
    commit the pass makes on that branch is a restack merge, which would carry them into
    somebody's feature branch. Only the working-tree restore may be handed to a pass.
    """
    skill = MAINTENANCE_SKILL_DOCUMENT.read_text()

    assert "git restore --source=<ref> --worktree -- .claude/stack/" in skill
    assert "git checkout <ref> -- .claude/stack/" not in skill
