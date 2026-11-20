"""
Print the matrix of the CI test jobs a change needs, as JSON.

With ``--since``, the jobs run the tests the files changed since that revision can
affect; without it, every test.
"""

from __future__ import annotations

import argparse
import subprocess
from pathlib import Path

from cognitive_robot_abstract_machine.ci_jobs import CIJobLayout, CIJobMatrix, Workspace
from cognitive_robot_abstract_machine.repository_files import RepositoryFile


def main() -> None:
    """
    Print the matrix of the jobs the change since the given revision needs.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--since",
        metavar="REVISION",
        help="Run only the tests the files changed between REVISION and HEAD can affect.",
    )
    arguments = parser.parse_args()

    repository_root = Path(
        subprocess.run(
            ["git", "rev-parse", "--show-toplevel"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    )
    workspace = Workspace.from_repository_root(repository_root=repository_root)
    affected_tests = workspace.all_tests
    if arguments.since is not None:
        changed_files = subprocess.run(
            ["git", "diff", "--name-only", arguments.since, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repository_root,
        ).stdout.splitlines()
        affected_tests = workspace.affected_tests(
            changed_paths=[Path(changed_file) for changed_file in changed_files]
        )
    layout = CIJobLayout.from_toml(
        text=(repository_root / RepositoryFile.CI_JOBS).read_text()
    )
    jobs = layout.jobs_running(affected_tests=affected_tests, workspace=workspace)
    print(CIJobMatrix(jobs=jobs).to_json())


if __name__ == "__main__":
    main()
