"""
The CI test jobs a change needs: those running the tests of every workspace member the
change touches and of every member depending on one of those, directly or through
others.
"""

from __future__ import annotations

import json
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from krrood.patterns.classproperty import classproperty
from krrood.patterns.path_enumeration import PathEnumeration
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from typing_extensions import Any, Iterable

from cognitive_robot_abstract_machine.exceptions import UnknownSharedJobLibraryError
from cognitive_robot_abstract_machine.repository_files import (
    RepositoryDirectory,
    RepositoryFile,
)

# %% keys of the files read and written


class PyprojectKey(StrEnum):
    """
    The keys of a ``pyproject.toml`` read to find the workspace's members and their
    dependencies.
    """

    PROJECT = "project"
    NAME = "name"
    DEPENDENCIES = "dependencies"
    TOOL = "tool"
    UV = "uv"
    WORKSPACE = "workspace"
    MEMBERS = "members"


class JobLayoutKey(StrEnum):
    """
    The keys of :attr:`RepositoryFile.CI_JOBS`.
    """

    SHARED_JOBS = "shared_jobs"
    NAME = "name"
    LIBRARIES = "libraries"
    CHECK_RULES = "check_rules"


class MatrixKey(StrEnum):
    """
    The keys of the matrix the CI workflow expands into its test jobs, which are the
    inputs of its reusable test workflow.
    """

    INCLUDE = "include"
    NAME = "name"
    LIBRARIES = "libs"
    CHECK_RULES = "check_rules"


# %% paths without tests


class DocumentationPath(PathEnumeration):
    """
    The paths of the repository's own documentation, whose change affects the tests of
    no workspace member.
    """

    DOCUMENTATION = Path("doc")
    """
    The documentation of the repository as a whole.
    """

    README = Path("README.md")
    """
    The introduction to the repository.
    """

    CHANGELOG = Path("CHANGELOG.md")
    """
    The changes of each release.
    """

    LICENSE = Path("LICENSE")
    """
    The license the repository is published under.
    """

    ISSUE_TEMPLATES = Path(".github") / "ISSUE_TEMPLATE"
    """
    The forms an issue is opened with.
    """


# %% tests of a library


@dataclass(kw_only=True, frozen=True)
class LibraryTests:
    """
    The tests of one library, in a directory of their own under
    :attr:`RepositoryDirectory.TESTS`.
    """

    library_name: str
    """
    The name the test directory is named after, which CI lists the library under.
    """

    @classproperty
    def directory_suffix(cls) -> str:
        """
        The suffix that follows the library's name in the name of its test directory.
        """
        return "_test"

    @property
    def directory(self) -> Path:
        """
        The directory holding the tests, relative to the repository root.
        """
        return RepositoryDirectory.TESTS / f"{self.library_name}{self.directory_suffix}"

    @classmethod
    def from_directory(cls, *, directory: Path) -> LibraryTests:
        """
        :param directory: A test directory, whose name ends in :attr:`directory_suffix`.
        :return: The tests it holds.
        """
        return cls(library_name=directory.name.removesuffix(cls.directory_suffix))


# %% workspace


@dataclass(kw_only=True, frozen=True)
class WorkspaceMember:
    """
    A package of the uv workspace, as its ``pyproject.toml`` declares it.
    """

    directory: Path
    """
    The directory of the member's ``pyproject.toml``, relative to the repository root.
    """

    distribution_name: str
    """
    The canonical name of the member's distribution.
    """

    dependency_names: frozenset[str]
    """
    The canonical names of every distribution the member declares a dependency on.
    """

    @property
    def tests(self) -> LibraryTests:
        """
        The tests of the member, in the test directory named after its directory.
        """
        return LibraryTests(library_name=self.directory.name)

    def owns(self, *, path: Path) -> bool:
        """
        :param path: A path relative to the repository root.
        :return: Whether the path lies in the member's directory or its test directory.
        """
        return path.is_relative_to(self.directory) or path.is_relative_to(
            self.tests.directory
        )

    @classmethod
    def from_directory(
        cls, *, repository_root: Path, directory: Path
    ) -> WorkspaceMember:
        """
        :param repository_root: The root of the repository.
        :param directory: The member's directory, relative to the repository root.
        :return: The member its ``pyproject.toml`` declares.
        """
        project = tomllib.loads(
            (repository_root / directory / RepositoryFile.PYPROJECT).read_text()
        )[PyprojectKey.PROJECT]
        return cls(
            directory=directory,
            distribution_name=canonicalize_name(project[PyprojectKey.NAME]),
            dependency_names=frozenset(
                canonicalize_name(Requirement(specifier).name)
                for specifier in project.get(PyprojectKey.DEPENDENCIES, [])
            ),
        )


@dataclass(kw_only=True, frozen=True)
class Workspace:
    """
    The members of the repository's uv workspace and the tests of the repository
    itself, which belong to no member.
    """

    members: tuple[WorkspaceMember, ...]
    """
    The members, in the order the workspace lists them.
    """

    repository_tests: tuple[LibraryTests, ...]
    """
    The tests of the repository as a whole, which a change to any file can affect and
    so run for every change.
    """

    @property
    def all_tests(self) -> tuple[LibraryTests, ...]:
        """
        The tests of every member, followed by those of the repository.
        """
        return tuple(member.tests for member in self.members) + self.repository_tests

    def members_depending_on(
        self, *, members: Iterable[WorkspaceMember]
    ) -> set[WorkspaceMember]:
        """
        :param members: Members of this workspace.
        :return: Those members and every member depending on one of them, directly or
            through other members.
        """
        reached = set(members)
        pending = list(reached)
        while pending:
            dependency_name = pending.pop().distribution_name
            for member in self.members:
                if member in reached or dependency_name not in member.dependency_names:
                    continue
                reached.add(member)
                pending.append(member)
        return reached

    def affected_tests(
        self, *, changed_paths: Iterable[Path]
    ) -> tuple[LibraryTests, ...]:
        """
        :param changed_paths: The paths a change touches, relative to the repository
            root.
        :return: The tests that change can affect, in the order of :attr:`all_tests`.
            A path outside every member, its tests and :class:`DocumentationPath`
            affects all of them.
        """
        changed_members = set()
        for path in changed_paths:
            if any(
                path.is_relative_to(documentation)
                for documentation in DocumentationPath
            ):
                continue
            if any(
                path.is_relative_to(tests.directory) for tests in self.repository_tests
            ):
                continue
            owner = next(
                (member for member in self.members if member.owns(path=path)), None
            )
            if owner is None:
                return self.all_tests
            changed_members.add(owner)
        affected_members = self.members_depending_on(members=changed_members)
        return (
            tuple(member.tests for member in self.members if member in affected_members)
            + self.repository_tests
        )

    @classmethod
    def from_repository_root(cls, *, repository_root: Path) -> Workspace:
        """
        :param repository_root: The root of the repository, holding the workspace's
            ``pyproject.toml``.
        :return: The workspace it declares.
        """
        member_directories = tomllib.loads(
            (repository_root / RepositoryFile.PYPROJECT).read_text()
        )[PyprojectKey.TOOL][PyprojectKey.UV][PyprojectKey.WORKSPACE][
            PyprojectKey.MEMBERS
        ]
        members = tuple(
            WorkspaceMember.from_directory(
                repository_root=repository_root, directory=Path(directory)
            )
            for directory in member_directories
        )
        member_library_names = {member.tests.library_name for member in members}
        test_directories = sorted(
            (repository_root / RepositoryDirectory.TESTS).glob(
                f"*{LibraryTests.directory_suffix}"
            )
        )
        return cls(
            members=members,
            repository_tests=tuple(
                tests
                for tests in (
                    LibraryTests.from_directory(directory=directory)
                    for directory in test_directories
                )
                if tests.library_name not in member_library_names
            ),
        )


# %% jobs


@dataclass(kw_only=True, frozen=True)
class CIJob:
    """
    A CI job running the tests of one or more libraries.
    """

    name: str
    """
    The name of the job, which its cache is saved under.
    """

    library_names: tuple[str, ...]
    """
    The libraries whose tests the job runs, by :attr:`LibraryTests.library_name`.
    """

    check_rules: bool = False
    """
    Whether the job first checks the code a pull request changes against the rules
    that have a checker.
    """

    def to_matrix_entry(self) -> dict[str, Any]:
        """
        :return: The job as an entry of the matrix the CI workflow expands.
        """
        return {
            MatrixKey.NAME: self.name,
            MatrixKey.LIBRARIES: json.dumps(list(self.library_names)),
            MatrixKey.CHECK_RULES: self.check_rules,
        }


@dataclass(kw_only=True, frozen=True)
class CIJobLayout:
    """
    How the tests of the libraries are divided into CI jobs: the libraries of a shared
    job run together, every other library in a job of its own.
    """

    shared_jobs: tuple[CIJob, ...]
    """
    The jobs running the tests of several libraries.
    """

    def jobs_running(
        self, *, affected_tests: Iterable[LibraryTests], workspace: Workspace
    ) -> tuple[CIJob, ...]:
        """
        :param affected_tests: The tests to run.
        :param workspace: The workspace the tests belong to.
        :return: The jobs running exactly those tests. A shared job that checks the
            rules is kept even when none of its libraries is affected.
        :raises UnknownSharedJobLibraryError: When a shared job names a library the
            workspace has no tests of.
        """
        known_library_names = [tests.library_name for tests in workspace.all_tests]
        affected_library_names = {tests.library_name for tests in affected_tests}
        shared_library_names = set()
        jobs = []
        for shared_job in self.shared_jobs:
            unknown_library_names = set(shared_job.library_names).difference(
                known_library_names
            )
            if unknown_library_names:
                raise UnknownSharedJobLibraryError(
                    job_name=shared_job.name,
                    library_names=tuple(sorted(unknown_library_names)),
                    known_library_names=tuple(known_library_names),
                )
            shared_library_names.update(shared_job.library_names)
            library_names = tuple(
                name
                for name in shared_job.library_names
                if name in affected_library_names
            )
            if library_names or shared_job.check_rules:
                jobs.append(
                    CIJob(
                        name=shared_job.name,
                        library_names=library_names,
                        check_rules=shared_job.check_rules,
                    )
                )
        own_jobs = tuple(
            CIJob(name=name, library_names=(name,))
            for name in known_library_names
            if name in affected_library_names and name not in shared_library_names
        )
        return own_jobs + tuple(jobs)

    @classmethod
    def from_toml(cls, *, text: str) -> CIJobLayout:
        """
        :param text: The content of :attr:`RepositoryFile.CI_JOBS`.
        :return: The layout it describes.
        """
        return cls(
            shared_jobs=tuple(
                CIJob(
                    name=job[JobLayoutKey.NAME],
                    library_names=tuple(job[JobLayoutKey.LIBRARIES]),
                    check_rules=job.get(JobLayoutKey.CHECK_RULES, False),
                )
                for job in tomllib.loads(text).get(JobLayoutKey.SHARED_JOBS, [])
            )
        )


@dataclass(kw_only=True, frozen=True)
class CIJobMatrix:
    """
    The test jobs of one CI run, as the matrix its workflow expands.
    """

    jobs: tuple[CIJob, ...]
    """
    The jobs, in the order the matrix lists them.
    """

    def to_json(self) -> str:
        """
        :return: The matrix in the JSON form GitHub Actions reads with ``fromJSON``.
        """
        return json.dumps(
            {MatrixKey.INCLUDE: [job.to_matrix_entry() for job in self.jobs]}
        )
