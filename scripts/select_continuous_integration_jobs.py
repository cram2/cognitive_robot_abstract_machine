"""
Write the matrix of the CI test jobs a change needs, as JSON, to the step's output file
under GitHub Actions and to the standard output elsewhere.

With ``--since``, the jobs run the tests the files changed since that revision can
affect; without it, every test.
"""

from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from cognitive_robot_abstract_machine.continuous_integration_jobs import (
    ContinuousIntegrationJobLayout,
    ContinuousIntegrationJobMatrix,
    GitHubActionsName,
    RepositoryTestSuite,
    Workspace,
)
from cognitive_robot_abstract_machine.repository_files import RepositoryFile


def main() -> None:
    """
    Write the matrix of the jobs the change since the given revision needs.
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
    layout = ContinuousIntegrationJobLayout.from_toml(
        text=(repository_root / RepositoryFile.CONTINUOUS_INTEGRATION_JOBS).read_text()
    )
    test_suite = RepositoryTestSuite(
        repository_root=repository_root,
        workspace=Workspace.from_repository_root(repository_root=repository_root),
        assigned_tests=layout.assigned_tests,
    )
    affected_tests = test_suite.all_tests()
    if arguments.since is not None:
        changed_files = subprocess.run(
            ["git", "diff", "--name-only", arguments.since, "HEAD"],
            capture_output=True,
            text=True,
            check=True,
            cwd=repository_root,
        ).stdout.splitlines()
        affected_tests = test_suite.affected_tests(
            changed_paths=[Path(changed_file) for changed_file in changed_files]
        )
    jobs = layout.jobs_running(
        affected_tests=affected_tests,
        known_library_names=test_suite.library_names,
    )
    matrix = ContinuousIntegrationJobMatrix(jobs=jobs)
    output_file = os.environ.get(GitHubActionsName.OUTPUT_FILE_VARIABLE)
    if output_file is None:
        print(matrix.to_json())
        return
    matrix.write_github_output(output_file=Path(output_file))


if __name__ == "__main__":
    main()
