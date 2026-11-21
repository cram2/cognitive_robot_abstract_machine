"""
The CI test jobs a change needs. A workspace member is affected by a change to it or to
a member it depends on, and a test by a change to its library's member, to a member it
imports or to a test module it loads.
"""

from __future__ import annotations

import ast
import functools
import json
import os
import subprocess
import sys
import tomllib
from dataclasses import dataclass
from enum import StrEnum
from functools import cached_property
from pathlib import Path

from krrood.development_tools.exceptions import (
    UnknownContinuousIntegrationLibraryError,
)
from krrood.development_tools.pyproject import Pyproject
from krrood.patterns.classproperty import classproperty
from krrood.source_analysis.import_graph import ModuleGraph, PythonSourceFile
from packaging.requirements import Requirement
from packaging.utils import canonicalize_name
from typing_extensions import Any, Iterable

# %% keys and names of the files read and written


class JobLayoutKey(StrEnum):
    """
    The keys of the file declaring a :class:`ContinuousIntegrationJobLayout`.
    """

    TESTS_DIRECTORY = "tests_directory"
    DOCUMENTATION_PATHS = "documentation_paths"
    SHARED_JOBS = "shared_jobs"
    ASSIGNED_TESTS = "assigned_tests"
    TEST_ENVIRONMENTS = "test_environments"
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
    LIBRARIES = "libraries"
    TEST_PATHS = "test_paths"
    CHECK_RULES = "check_rules"


class PytestFileName(StrEnum):
    """
    The parts of file names pytest gives a meaning to.
    """

    TEST_PREFIX = "test_"
    """
    The prefix of a test file's name.
    """

    TEST_SUFFIX = "_test.py"
    """
    The suffix of a test file's name, as an alternative to the prefix.
    """

    CONFIGURATION = "conftest.py"
    """
    The file whose fixtures and hooks every test below its directory uses.
    """


class GitHubActionsName(StrEnum):
    """
    The names GitHub Actions gives a meaning to.
    """

    OUTPUT_FILE_VARIABLE = "GITHUB_OUTPUT"
    """
    The environment variable naming the file a step writes its outputs to.
    """

    MATRIX_OUTPUT = "matrix"
    """
    The output of the selection step the CI workflow expands into its test jobs.
    """


class PytestOption(StrEnum):
    """
    The command-line options and values of pytest that the test jobs pass.
    """

    WORKER_COUNT = "-n"
    """
    The option of pytest-xdist setting how many processes run the tests.
    """

    WORKER_PER_CORE = "auto"
    """
    The worker count starting one process per processor core.
    """


class MemberDirectoryName(StrEnum):
    """
    The names of directories inside a workspace member.
    """

    SOURCE = "src"
    """
    The directory holding the member's top-level packages.
    """


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
    def library_name(self) -> str:
        """
        The name CI lists the member's tests under.
        """
        return self.directory.name

    @classmethod
    def from_directory(
        cls, *, repository_root: Path, directory: Path
    ) -> WorkspaceMember:
        """
        :param repository_root: The root of the repository.
        :param directory: The member's directory, relative to the repository root.
        :return: The member its ``pyproject.toml`` declares.
        """
        pyproject = Pyproject.from_directory(directory=repository_root / directory)
        return cls(
            directory=directory,
            distribution_name=pyproject.distribution_name,
            dependency_names=frozenset(
                canonicalize_name(Requirement(specifier).name)
                for specifier in pyproject.dependencies
            ),
        )


@dataclass(kw_only=True, frozen=True)
class Workspace:
    """
    The members of the repository's uv workspace.
    """

    members: tuple[WorkspaceMember, ...]
    """
    The members, in the order the workspace lists them.
    """

    def owner(self, *, path: Path) -> WorkspaceMember | None:
        """
        :param path: A path relative to the repository root.
        :return: The member with the innermost directory holding the path, or none.
        """
        owners = [
            member for member in self.members if path.is_relative_to(member.directory)
        ]
        return max(owners, key=lambda member: len(member.directory.parts), default=None)

    def members_nested_in(
        self, *, member: WorkspaceMember
    ) -> tuple[WorkspaceMember, ...]:
        """
        :param member: A member of this workspace.
        :return: The other members whose directories lie inside the member's.
        """
        return tuple(
            other
            for other in self.members
            if other != member and other.directory.is_relative_to(member.directory)
        )

    def members_depending_on(
        self, *, members: Iterable[WorkspaceMember]
    ) -> frozenset[WorkspaceMember]:
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
        return frozenset(reached)

    @cached_property
    def dependency_closures(self) -> dict[WorkspaceMember, frozenset[WorkspaceMember]]:
        """
        Each member together with every member it depends on, directly or through
        other members.
        """
        dependents = {
            member: self.members_depending_on(members=[member])
            for member in self.members
        }
        return {
            member: frozenset(
                other for other in self.members if member in dependents[other]
            )
            for member in self.members
        }

    @classmethod
    def from_repository_root(cls, *, repository_root: Path) -> Workspace:
        """
        :param repository_root: The root of the repository, holding the workspace's
            ``pyproject.toml``.
        :return: The workspace it declares.
        """
        return cls(
            members=tuple(
                WorkspaceMember.from_directory(
                    repository_root=repository_root, directory=Path(directory)
                )
                for directory in Pyproject.from_directory(
                    directory=repository_root
                ).workspace_members
            )
        )


# %% tests of a library


@dataclass(kw_only=True, frozen=True)
class LibraryTests:
    """
    The tests of one library that a CI job runs.
    """

    library_name: str
    """
    The name CI lists the library under.
    """

    test_paths: tuple[Path, ...]
    """
    The test files and directories pytest runs, relative to the repository root.
    """

    @classproperty
    def directory_suffix(cls) -> str:
        """
        The suffix that follows a library's name in the name of its test directory.
        """
        return "_test"

    def run(self, *, repository_root: Path, environment: dict[str, str]) -> bool:
        """
        Run pytest over the tests in a process of its own, with one worker per core.

        :param repository_root: The root of the repository the test paths are in.
        :param environment: The variables added to the current environment for the
            run.
        :return: Whether every test passed.
        """
        command = [
            sys.executable,
            "-m",
            "pytest",
            PytestOption.WORKER_COUNT,
            PytestOption.WORKER_PER_CORE,
            *(str(repository_root / test_path) for test_path in self.test_paths),
        ]
        completed = subprocess.run(command, env=os.environ | environment)
        return completed.returncode == 0

    @classmethod
    def from_test_paths_json(cls, *, text: str) -> tuple[LibraryTests, ...]:
        """
        :param text: The :attr:`MatrixKey.TEST_PATHS` of a matrix entry.
        :return: The tests of each library it lists, in its order.
        """
        return tuple(
            cls(
                library_name=library_name,
                test_paths=tuple(Path(test_path) for test_path in test_paths),
            )
            for library_name, test_paths in json.loads(text).items()
        )


@dataclass(kw_only=True, eq=False)
class RepositoryTestSuite:
    """
    The tests of the repository, each assigned to the library CI lists it under, and the
    modules each one loads.
    """

    repository_root: Path
    """
    The root of the repository.
    """

    workspace: Workspace
    """
    The workspace whose members the tests test.
    """

    layout: ContinuousIntegrationJobLayout
    """
    The layout naming the directories of the tests and the tests assigned by hand.
    """

    @cached_property
    def module_graph(self) -> ModuleGraph:
        """
        The modules of the tests and of every workspace member.
        """
        package_directories = [self.repository_root / self.layout.tests_directory]
        for member in self.workspace.members:
            source = (
                self.repository_root / member.directory / MemberDirectoryName.SOURCE
            )
            if source.is_dir():
                package_directories.extend(
                    child
                    for child in sorted(source.iterdir())
                    if child.is_dir() and child.name.isidentifier()
                )
        return ModuleGraph.from_packages(package_directories=package_directories)

    @cached_property
    def module_names(self) -> dict[Path, str]:
        """
        The dotted name of every module, keyed by its file relative to the repository
        root.
        """
        return {
            path.relative_to(self.repository_root): name
            for name, path in self.module_graph.paths.items()
        }

    @cached_property
    def module_owners(self) -> dict[str, WorkspaceMember]:
        """
        The member shipping each of the workspace's modules, keyed by the module's
        dotted name.
        """
        owners = {}
        for path, name in self.module_names.items():
            owner = self.workspace.owner(path=path)
            if owner is not None:
                owners[name] = owner
        return owners

    @cached_property
    def test_files(self) -> dict[str, Path]:
        """
        The test files in the test directories of the libraries, keyed by their dotted
        module name.
        """
        return {
            name: path
            for path, name in self.module_names.items()
            if self.test_directory(path=path) is not None
            and (
                path.name.startswith(PytestFileName.TEST_PREFIX)
                or path.name.endswith(PytestFileName.TEST_SUFFIX)
            )
        }

    @cached_property
    def configuration_files(self) -> dict[str, Path]:
        """
        The ``conftest.py`` files of the tests, keyed by their dotted module name.
        """
        return {
            name: path
            for path, name in self.module_names.items()
            if path.is_relative_to(self.layout.tests_directory)
            and path.name == PytestFileName.CONFIGURATION
        }

    @cached_property
    def library_names(self) -> tuple[str, ...]:
        """
        Every library CI can list: one per member, then one per test directory named
        after no member, in name order.
        """
        member_library_names = tuple(
            member.library_name for member in self.workspace.members
        )
        directory_library_names = sorted(
            {
                directory.name.removesuffix(LibraryTests.directory_suffix)
                for directory in (
                    self.test_directory(path=path) for path in self.test_files.values()
                )
            }
        )
        return member_library_names + tuple(
            name for name in directory_library_names if name not in member_library_names
        )

    @cached_property
    def library_members(self) -> dict[str, WorkspaceMember]:
        """
        The member of each library named after one.
        """
        return {member.library_name: member for member in self.workspace.members}

    @cached_property
    def test_libraries(self) -> dict[str, str]:
        """
        The library of each test file, keyed by its dotted module name. A test belongs
        to its :attr:`ContinuousIntegrationJobLayout.assigned_tests` library, else to the member its test directory is
        named after or, where members nested in that one depend on every member the
        test uses, to the one of those depending on the fewest members.
        """
        libraries = {}
        for module, path in self.test_files.items():
            directory_library_name = self.test_directory(path=path).name.removesuffix(
                LibraryTests.directory_suffix
            )
            libraries[module] = directory_library_name
            member = self.library_members.get(directory_library_name)
            used = self.used_members(module=module)
            if member is None or not used:
                continue
            covering = [
                nested
                for nested in self.workspace.members_nested_in(member=member)
                if used <= self.workspace.dependency_closures[nested]
            ]
            if covering:
                libraries[module] = min(
                    covering,
                    key=lambda nested: len(self.workspace.dependency_closures[nested]),
                ).library_name
        for assigned in self.layout.assigned_tests:
            for module, path in self.test_files.items():
                if any(
                    path.is_relative_to(test_path) for test_path in assigned.test_paths
                ):
                    libraries[module] = assigned.library_name
        return libraries

    def test_directory(self, *, path: Path) -> Path | None:
        """
        :param path: A path relative to the repository root.
        :return: The test directory of a library that holds the path, or none.
        """
        if not path.is_relative_to(self.layout.tests_directory):
            return None
        parts = path.relative_to(self.layout.tests_directory).parts
        if len(parts) < 2 or not parts[0].endswith(LibraryTests.directory_suffix):
            return None
        return self.layout.tests_directory / parts[0]

    @functools.cache
    def imported_modules(self, *, module: str) -> tuple[str, ...]:
        """
        :param module: The dotted name of a module of the tests.
        :return: Every module it imports anywhere in its source, parent packages
            included.
        """
        path = self.module_graph.paths[module]
        package = (
            module
            if path.name == PythonSourceFile.PACKAGE_INITIALIZER
            else module.rpartition(".")[0]
        )
        statements = (
            node
            for node in ast.walk(ast.parse(path.read_text(), filename=str(path)))
            if isinstance(node, (ast.Import, ast.ImportFrom))
        )
        return tuple(
            loaded
            for statement in statements
            for loaded in self.module_graph.loaded_modules(
                statement=statement, package=package
            )
        )

    @functools.cache
    def loaded_test_modules(self, *, module: str) -> frozenset[str]:
        """
        :param module: The dotted name of a module of the tests.
        :return: The module, the packages above it and every module of the tests it
            loads, directly or through other modules of the tests.
        """
        parts = module.split(".")
        pending = [".".join(parts[:length]) for length in range(1, len(parts) + 1)]
        loaded = set()
        while pending:
            name = pending.pop()
            if name in loaded or name not in self.module_graph.paths:
                continue
            loaded.add(name)
            pending.extend(
                imported
                for imported in self.imported_modules(module=name)
                if imported.partition(".")[0] == self.layout.tests_directory.name
            )
        return frozenset(loaded)

    @functools.cache
    def used_members(self, *, module: str) -> frozenset[WorkspaceMember]:
        """
        :param module: The dotted name of a module of the tests.
        :return: The members whose modules it imports, directly or through other
            modules of the tests.
        """
        return frozenset(
            self.module_owners[imported]
            for test_module in self.loaded_test_modules(module=module)
            for imported in self.imported_modules(module=test_module)
            if imported in self.module_owners
        )

    def affected_tests(
        self, *, changed_paths: Iterable[Path]
    ) -> tuple[LibraryTests, ...]:
        """
        :param changed_paths: The paths a change touches, relative to the repository
            root.
        :return: The tests that change can affect, per library in the order of
            :attr:`library_names`. A path outside every member, the tests and
            :attr:`ContinuousIntegrationJobLayout.documentation_paths` affects every
            test, as does a ``conftest.py``
            loading a changed test module affect every test below it.
        """
        changed_members = set()
        changed_modules = set()
        changed_test_directories = set()
        for path in changed_paths:
            if any(
                path.is_relative_to(document)
                for document in self.layout.documentation_paths
            ):
                continue
            if (
                path.is_relative_to(self.layout.tests_directory)
                and path in self.module_names
            ):
                changed_modules.add(self.module_names[path])
                continue
            test_directory = self.test_directory(path=path)
            if test_directory is not None:
                changed_test_directories.add(test_directory)
                continue
            owner = self.workspace.owner(path=path)
            if owner is None:
                return self.all_tests()
            changed_members.add(owner)
        affected_members = self.workspace.members_depending_on(members=changed_members)
        affected_configuration_directories = [
            path.parent
            for module, path in self.configuration_files.items()
            if self.loaded_test_modules(module=module) & changed_modules
        ]
        return self.library_tests(
            selected_modules={
                module
                for module, path in self.test_files.items()
                if self.library_members.get(self.test_libraries[module])
                in affected_members | {None}
                or self.used_members(module=module) & affected_members
                or self.loaded_test_modules(module=module) & changed_modules
                or self.test_directory(path=path) in changed_test_directories
                or any(
                    path.is_relative_to(directory)
                    for directory in affected_configuration_directories
                )
            }
        )

    def all_tests(self) -> tuple[LibraryTests, ...]:
        """
        :return: Every test, per library in the order of :attr:`library_names`.
        """
        return self.library_tests(selected_modules=set(self.test_files))

    def library_tests(self, *, selected_modules: set[str]) -> tuple[LibraryTests, ...]:
        """
        :param selected_modules: The dotted names of the selected test files.
        :return: The selected tests of each library that has one. A library runs its
            whole test directory when every test in it is selected and its own.
        """
        selected_paths = {name: [] for name in self.library_names}
        directory_libraries = {}
        for module, path in self.test_files.items():
            directory = self.test_directory(path=path)
            directory_libraries.setdefault(directory, set()).add(
                self.test_libraries[module]
            )
            if module in selected_modules:
                selected_paths[self.test_libraries[module]].append(path)
            else:
                directory_libraries[directory].add(None)
        library_tests = []
        for name, paths in selected_paths.items():
            if not paths:
                continue
            directory = self.test_directory(path=paths[0])
            test_paths = (
                (directory,)
                if directory_libraries[directory] == {name}
                else tuple(sorted(paths))
            )
            library_tests.append(LibraryTests(library_name=name, test_paths=test_paths))
        return tuple(library_tests)


# %% jobs


@dataclass(kw_only=True, frozen=True)
class SharedJob:
    """
    A CI job that runs the tests of several libraries, as
    :class:`ContinuousIntegrationJobLayout` declares it.
    """

    name: str
    """
    The name of the job, which its cache is saved under.
    """

    library_names: tuple[str, ...]
    """
    The libraries whose tests the job runs.
    """

    check_rules: bool = False
    """
    Whether the job first checks the code a pull request changes against the rules
    that have a checker, which makes it run for every change.
    """


@dataclass(kw_only=True, frozen=True)
class ContinuousIntegrationJob:
    """
    A CI job running the tests of one or more libraries.
    """

    name: str
    """
    The name of the job, which its cache is saved under.
    """

    libraries: tuple[LibraryTests, ...]
    """
    The tests the job runs.
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
            MatrixKey.LIBRARIES: json.dumps(
                [library.library_name for library in self.libraries]
            ),
            MatrixKey.TEST_PATHS: json.dumps(
                {
                    library.library_name: [str(path) for path in library.test_paths]
                    for library in self.libraries
                }
            ),
            MatrixKey.CHECK_RULES: self.check_rules,
        }


@dataclass(kw_only=True, frozen=True)
class ContinuousIntegrationJobLayout:
    """
    How the tests of the libraries are divided into CI jobs: the libraries of a shared
    job run together, every other library in a job of its own.
    """

    tests_directory: Path
    """
    The directory holding the test directory of each library, relative to the
    repository root. A library's test directory is named after it, followed by
    :attr:`LibraryTests.directory_suffix`.
    """

    documentation_paths: tuple[Path, ...]
    """
    The paths of the repository's own documentation, relative to the repository root,
    whose change affects no test.
    """

    shared_jobs: tuple[SharedJob, ...]
    """
    The jobs running the tests of several libraries.
    """

    assigned_tests: tuple[LibraryTests, ...]
    """
    The tests assigned to a library by hand: those that load the modules of members
    other than by importing them, such as in a subprocess.
    """

    test_environments: dict[str, dict[str, str]]
    """
    The environment variables the tests of a library need set before Python starts,
    by library.
    """

    def jobs_running(
        self,
        *,
        affected_tests: Iterable[LibraryTests],
        known_library_names: Iterable[str],
    ) -> tuple[ContinuousIntegrationJob, ...]:
        """
        :param affected_tests: The tests to run.
        :param known_library_names: Every library CI can list, in the order of the jobs
            that run one library alone.
        :return: The jobs running exactly those tests. A shared job that checks the
            rules is kept even when none of its libraries is affected.
        :raises UnknownContinuousIntegrationLibraryError: When a shared job, the
            assigned tests or the test environments name an unknown library.
        """
        known_library_names = tuple(known_library_names)
        library_names_by_table = {
            JobLayoutKey.ASSIGNED_TESTS: {
                assigned.library_name for assigned in self.assigned_tests
            },
            JobLayoutKey.TEST_ENVIRONMENTS: set(self.test_environments),
        }
        for table, library_names in library_names_by_table.items():
            unknown_table_library_names = library_names.difference(known_library_names)
            if unknown_table_library_names:
                raise UnknownContinuousIntegrationLibraryError(
                    section=table,
                    library_names=tuple(sorted(unknown_table_library_names)),
                    known_library_names=known_library_names,
                )
        affected_by_name = {tests.library_name: tests for tests in affected_tests}
        shared_library_names = set()
        shared_jobs = []
        for shared_job in self.shared_jobs:
            unknown_library_names = set(shared_job.library_names).difference(
                known_library_names
            )
            if unknown_library_names:
                raise UnknownContinuousIntegrationLibraryError(
                    section=shared_job.name,
                    library_names=tuple(sorted(unknown_library_names)),
                    known_library_names=known_library_names,
                )
            shared_library_names.update(shared_job.library_names)
            libraries = tuple(
                affected_by_name[name]
                for name in shared_job.library_names
                if name in affected_by_name
            )
            if libraries or shared_job.check_rules:
                shared_jobs.append(
                    ContinuousIntegrationJob(
                        name=shared_job.name,
                        libraries=libraries,
                        check_rules=shared_job.check_rules,
                    )
                )
        own_jobs = tuple(
            ContinuousIntegrationJob(name=name, libraries=(affected_by_name[name],))
            for name in known_library_names
            if name in affected_by_name and name not in shared_library_names
        )
        return own_jobs + tuple(shared_jobs)

    @classmethod
    def from_toml(cls, *, text: str) -> ContinuousIntegrationJobLayout:
        """
        :param text: The content of a file declaring a layout.
        :return: The layout it describes.
        """
        layout = tomllib.loads(text)
        return cls(
            tests_directory=Path(layout[JobLayoutKey.TESTS_DIRECTORY]),
            documentation_paths=tuple(
                Path(path) for path in layout.get(JobLayoutKey.DOCUMENTATION_PATHS, [])
            ),
            shared_jobs=tuple(
                SharedJob(
                    name=job[JobLayoutKey.NAME],
                    library_names=tuple(job[JobLayoutKey.LIBRARIES]),
                    check_rules=job.get(JobLayoutKey.CHECK_RULES, False),
                )
                for job in layout.get(JobLayoutKey.SHARED_JOBS, [])
            ),
            assigned_tests=tuple(
                LibraryTests(
                    library_name=library_name,
                    test_paths=tuple(Path(test_path) for test_path in test_paths),
                )
                for library_name, test_paths in layout.get(
                    JobLayoutKey.ASSIGNED_TESTS, {}
                ).items()
            ),
            test_environments=layout.get(JobLayoutKey.TEST_ENVIRONMENTS, {}),
        )


@dataclass(kw_only=True, frozen=True)
class ContinuousIntegrationJobMatrix:
    """
    The test jobs of one CI run, as the matrix its workflow expands.
    """

    jobs: tuple[ContinuousIntegrationJob, ...]
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

    def write_github_output(self, *, output_file: Path) -> None:
        """
        Append the matrix to a step's output file as its
        :attr:`GitHubActionsName.MATRIX_OUTPUT`.

        :param output_file: The file :attr:`GitHubActionsName.OUTPUT_FILE_VARIABLE`
            names.
        """
        with output_file.open("a") as output:
            output.write(f"{GitHubActionsName.MATRIX_OUTPUT}={self.to_json()}\n")
