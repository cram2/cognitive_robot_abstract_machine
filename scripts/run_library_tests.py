"""
Run the tests of each library of a CI test job, each library in a pytest process of its
own, and exit non-zero when any of them failed.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from krrood.development_tools.continuous_integration_jobs import (
    ContinuousIntegrationJobLayout,
    LibraryTests,
)

from cognitive_robot_abstract_machine.repository_files import RepositoryFile


def main() -> None:
    """
    Run the tests the given matrix entry lists, every library even after another one
    failed.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--test-paths",
        required=True,
        metavar="JSON",
        help="The test paths of each library, as the job's matrix entry lists them.",
    )
    arguments = parser.parse_args()
    repository_root = Path(__file__).resolve().parents[1]
    layout = ContinuousIntegrationJobLayout.from_toml(
        text=(repository_root / RepositoryFile.CONTINUOUS_INTEGRATION_JOBS).read_text()
    )
    outcomes = [
        library_tests.run(
            repository_root=repository_root,
            environment=layout.test_environments.get(library_tests.library_name, {}),
        )
        for library_tests in LibraryTests.from_test_paths_json(
            text=arguments.test_paths
        )
    ]
    sys.exit(0 if all(outcomes) else 1)


if __name__ == "__main__":
    main()
