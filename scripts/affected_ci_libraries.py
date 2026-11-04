"""
Prints the libraries whose CI test job a change can affect.

Reads the changed paths, one per line, from standard input and prints the names of the
libraries to test as a sorted JSON list. A library is one directory ``test/<library>_test``.

Runs on the bare CI runner before anything is installed, so it imports only the standard
library.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
import tomllib
from dataclasses import dataclass, field
from enum import Enum, StrEnum
from functools import cached_property
from pathlib import Path, PurePosixPath

# %% fixed names of the repository


class RepositoryPath(Enum):
    """
    The files and folders of the repository the selection reads, relative to its root.
    """

    PROJECT = PurePosixPath("pyproject.toml")
    """
    The root project, which lists the workspace members.
    """

    TESTS = PurePosixPath("test")
    """
    The folder holding one test folder per library and the harness they share.
    """

    TEST_HARNESS = PurePosixPath("test/conftest.py")
    """
    The conftest every test job imports.
    """

    AGENT_RULES = PurePosixPath("AGENTS.md")
    """
    The agent rules, which the job of :attr:`Workspace.agent_rules_library` checks.
    """

    DOCUMENTATION = PurePosixPath("doc")
    """
    The repository's documentation, which no test reads.
    """

    ORM_INTERFACES = PurePosixPath("cognitive_robot_abstract_machine/orm_interfaces.py")
    """
    The list of the workspace's ORM interfaces.
    """


class AlwaysTestedLibrary(StrEnum):
    """
    The libraries every selection tests: they check files of the whole repository, and
    they keep the selection from ever being empty.
    """

    VERSION = "version"
    """
    Checks the versions of every package.
    """

    REPOSITORY = "cognitive_robot_abstract_machine"
    """
    Checks the repository's own tooling.
    """


class PythonPattern(StrEnum):
    """
    Regular expressions over the lines of a Python file. They are read line by line
    instead of parsed, so a file that does not parse, such as a test's deliberately
    broken module, still counts; an import that only appears in a string counts too,
    which can only select more libraries, never fewer.
    """

    ABSOLUTE_FROM_IMPORT = r"^\s*from\s+([A-Za-z_]\w*)"
    """
    ``from package.module import name``, capturing the top-level package.
    """

    ABSOLUTE_IMPORT = r"^\s*import\s+([\w.]+(?:\s*,\s*[\w.]+)*)"
    """
    ``import package.module, other``, capturing the imported modules.
    """

    RELATIVE_IMPORT = r"^\s*from\s+(\.+)([\w.]*)\s+import\s+\(?\s*([\w,\s]*)"
    """
    ``from ..package.module import name``, capturing the dots, the module and the
    names imported.
    """

    ORM_INTERFACE = r"\bOrmInterface\(\s*\"([\w-]+)\""
    """
    The declaration of one generated ORM interface, capturing its package.
    """

    ORM_BUILD = r"\bregenerate_orm_interfaces\("
    """
    A test conftest building every ORM interface of the workspace.
    """


class PythonFile(StrEnum):
    """
    Names of the files a Python module is defined in.
    """

    MODULE_SUFFIX = ".py"
    """
    The suffix of a module's file.
    """

    PACKAGE = "__init__.py"
    """
    The file that makes a folder a package.
    """


class Suffix(StrEnum):
    """
    Suffixes of names the selection reads.
    """

    TEST_FOLDER = "_test"
    """
    Ends the name of a library's test folder.
    """

    MARKDOWN = ".md"
    """
    Ends the name of a Markdown document.
    """


# %% requirement names


@dataclass(frozen=True)
class RequirementName:
    """
    The name of the distribution a requirement asks for, normalized as in PEP 503 but
    joined with underscores, as the workspace's folders are.
    """

    normalized: str
    """
    The name in lower case, with every run of ``-``, ``_`` and ``.`` replaced by ``_``.
    """

    @classmethod
    def of(cls, requirement: str) -> RequirementName:
        """
        :param requirement: A requirement such as ``random-events>=1.0`` or
            ``krrood_spatial_math[casadi]``.
        :return: The name of the distribution it asks for.
        """
        name = re.match(r"\s*[A-Za-z0-9][A-Za-z0-9._-]*", requirement).group().strip()
        return cls(re.sub(r"[-_.]+", "_", name).lower())


# %% python sources


@dataclass(frozen=True)
class PythonSource:
    """
    One Python file, read for the modules it imports.
    """

    path: Path
    """
    The file.
    """

    @cached_property
    def text(self) -> str:
        """
        The file's content.
        """
        return self.path.read_text()

    @cached_property
    def absolute_imports(self) -> frozenset[str]:
        """
        The top-level names of the modules the file imports absolutely.
        """
        names = set(re.findall(PythonPattern.ABSOLUTE_FROM_IMPORT, self.text, re.M))
        for modules in re.findall(PythonPattern.ABSOLUTE_IMPORT, self.text, re.M):
            names.update(module.strip().split(".")[0] for module in modules.split(","))
        return frozenset(names)

    @cached_property
    def relative_imports(self) -> frozenset[Path]:
        """
        The files of this repository the file imports relatively.
        """
        files = set()
        for dots, module, names in re.findall(
            PythonPattern.RELATIVE_IMPORT, self.text, re.M
        ):
            package = self.path.parents[len(dots) - 1]
            if module:
                files.update(self.module_files(package / module.replace(".", "/")))
                continue
            for name in re.findall(r"\w+", names):
                files.update(self.module_files(package / name))
        return frozenset(files)

    @staticmethod
    def module_files(module: Path) -> list[Path]:
        """
        :param module: Where a module lives, without its suffix.
        :return: The file defining the module, if it is one of this repository.
        """
        candidates = [
            module.with_name(module.name + PythonFile.MODULE_SUFFIX),
            module / PythonFile.PACKAGE,
        ]
        return [candidate for candidate in candidates if candidate.is_file()]

    @cached_property
    def builds_orm_interfaces(self) -> bool:
        """
        Whether the file builds every ORM interface of the workspace.
        """
        return re.search(PythonPattern.ORM_BUILD, self.text) is not None

    @classmethod
    def closure(cls, sources: list[PythonSource]) -> list[PythonSource]:
        """
        :param sources: Python files.
        :return: The files and every file of the repository they import relatively,
            directly or through another such file.
        """
        reached = {source.path: source for source in sources}
        pending = list(sources)
        while pending:
            source = pending.pop()
            for path in source.relative_imports - reached.keys():
                reached[path] = PythonSource(path)
                pending.append(reached[path])
        return list(reached.values())


# %% workspace members


@dataclass(frozen=True)
class WorkspaceMember:
    """
    One package of the uv workspace.
    """

    name: str
    """
    The normalized name of its distribution.
    """

    folder: PurePosixPath
    """
    Its folder, relative to the repository root.
    """

    dependencies: frozenset[str]
    """
    The normalized names of every distribution it requires, its extras' included.
    """

    module_names: frozenset[str]
    """
    The top-level modules it ships.
    """

    @classmethod
    def read(cls, root: Path, folder: PurePosixPath) -> WorkspaceMember:
        """
        :param root: The repository root.
        :param folder: The member's folder, relative to the root.
        :return: The member its ``pyproject.toml`` declares.
        """
        project = tomllib.loads(
            (root / folder / RepositoryPath.PROJECT.value).read_text()
        )["project"]
        requirements = list(project.get("dependencies", []))
        for extra in project.get("optional-dependencies", {}).values():
            requirements.extend(extra)
        return cls(
            name=RequirementName.of(project["name"]).normalized,
            folder=folder,
            dependencies=frozenset(
                RequirementName.of(requirement).normalized
                for requirement in requirements
            ),
            module_names=frozenset(
                package.parent.name
                for package in (root / folder).glob(f"src/*/{PythonFile.PACKAGE}")
            ),
        )

    def contains(self, path: PurePosixPath) -> bool:
        """
        :param path: A path relative to the repository root.
        :return: Whether the path lies in this member's folder.
        """
        return path.is_relative_to(self.folder)


# %% the workspace


@dataclass
class Workspace:
    """
    The repository's libraries and what each of their test jobs reads.
    """

    root: Path
    """
    The repository root.
    """

    agent_rules_library: str = "krrood"
    """
    The library whose test job checks the agent rules.
    """

    @cached_property
    def members(self) -> dict[str, WorkspaceMember]:
        """
        The workspace members by name.
        """
        project = tomllib.loads((self.root / RepositoryPath.PROJECT.value).read_text())
        members = [
            WorkspaceMember.read(self.root, PurePosixPath(folder))
            for folder in project["tool"]["uv"]["workspace"]["members"]
        ]
        return {member.name: member for member in members}

    @cached_property
    def libraries(self) -> list[str]:
        """
        The names of the libraries, one per test folder, sorted.
        """
        return sorted(
            folder.name.removesuffix(Suffix.TEST_FOLDER)
            for folder in (self.root / RepositoryPath.TESTS.value).glob(
                f"*{Suffix.TEST_FOLDER}"
            )
            if folder.is_dir()
        )

    @cached_property
    def member_of_module(self) -> dict[str, str]:
        """
        The member shipping each top-level module, by module name.
        """
        return {
            module: member.name
            for member in self.members.values()
            for module in member.module_names
        }

    @cached_property
    def orm_members(self) -> frozenset[str]:
        """
        The members that have a generated ORM interface.
        """
        declarations = (self.root / RepositoryPath.ORM_INTERFACES.value).read_text()
        return frozenset(
            RequirementName.of(package).normalized
            for package in re.findall(PythonPattern.ORM_INTERFACE, declarations)
        )

    def members_imported_by(self, sources: list[PythonSource]) -> set[str]:
        """
        :param sources: Python files.
        :return: The members whose modules the files import.
        """
        return {
            self.member_of_module[module]
            for source in sources
            for module in source.absolute_imports
            if module in self.member_of_module
        }

    def dependency_closure(self, names: set[str]) -> frozenset[str]:
        """
        :param names: Names of members.
        :return: The members and every member they require, directly or indirectly.
        """
        reached = set()
        pending = list(names)
        while pending:
            name = pending.pop()
            if name in reached:
                continue
            reached.add(name)
            pending.extend(self.members[name].dependencies & self.members.keys())
        return frozenset(reached)

    def test_folder(self, library: str) -> Path:
        """
        :param library: A library.
        :return: The folder of its tests.
        """
        return self.root / RepositoryPath.TESTS.value / f"{library}{Suffix.TEST_FOLDER}"

    @cached_property
    def harness_members(self) -> set[str]:
        """
        The members the shared test harness imports, which every test job loads.
        """
        harness = PythonSource(self.root / RepositoryPath.TEST_HARNESS.value)
        return self.members_imported_by(PythonSource.closure([harness]))

    @cached_property
    def test_sources(self) -> dict[str, list[PythonSource]]:
        """
        The Python files each library's test job loads, by library: those of its test
        folder and every file they import relatively, such as another library's test
        helpers.
        """
        return {
            library: PythonSource.closure(
                [
                    PythonSource(path)
                    for path in sorted(
                        self.test_folder(library).rglob(f"*{PythonFile.MODULE_SUFFIX}")
                    )
                ]
            )
            for library in self.libraries
        }

    def members_read_by(self, library: str) -> frozenset[str]:
        """
        :param library: A library.
        :return: The members whose code the library's test job runs: the library itself,
            what its tests and the shared harness import, every ORM member if its tests
            build the ORM interfaces, and everything those require.
        """
        sources = self.test_sources[library]
        read = self.members_imported_by(sources) | self.harness_members
        if library in self.members:
            read.add(library)
        if any(source.builds_orm_interfaces for source in sources):
            read.update(self.orm_members)
        return self.dependency_closure(read)

    @cached_property
    def members_read_by_library(self) -> dict[str, frozenset[str]]:
        """
        The members each library's test job runs the code of, by library.
        """
        return {library: self.members_read_by(library) for library in self.libraries}

    def libraries_loading_tests_of(self, library: str) -> set[str]:
        """
        :param library: A library.
        :return: The libraries whose test job loads a file of its test folder.
        """
        folder = self.test_folder(library)
        return {
            loading
            for loading, sources in self.test_sources.items()
            if any(source.path.is_relative_to(folder) for source in sources)
        }

    def libraries_affected_by(self, path: PurePosixPath) -> set[str]:
        """
        :param path: A changed path, relative to the repository root.
        :return: The libraries whose test job the change can affect.
        """
        tests = RepositoryPath.TESTS.value
        if path.is_relative_to(tests) and len(path.parts) > 2:
            library = path.parts[1].removesuffix(Suffix.TEST_FOLDER)
            if path.parts[1].endswith(Suffix.TEST_FOLDER) and library in self.libraries:
                return self.libraries_loading_tests_of(library)
        changed_members = {
            member.name for member in self.members.values() if member.contains(path)
        }
        if changed_members:
            return {
                library
                for library, read in self.members_read_by_library.items()
                if read & changed_members
            }
        if path == RepositoryPath.AGENT_RULES.value:
            return {self.agent_rules_library}
        if path.is_relative_to(RepositoryPath.DOCUMENTATION.value) or (
            len(path.parts) == 1 and path.suffix == Suffix.MARKDOWN
        ):
            return set()
        return set(self.libraries)

    def libraries_to_test(self, paths: list[PurePosixPath]) -> list[str]:
        """
        :param paths: The changed paths, relative to the repository root.
        :return: The libraries to test, sorted.
        """
        selected = {library.value for library in AlwaysTestedLibrary}
        for path in paths:
            selected |= self.libraries_affected_by(path)
        return sorted(selected)


# %% command line


def main() -> None:
    """
    Print the libraries to test for the changed paths on standard input, or every
    library.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--all", action="store_true", help="print every library, whatever changed"
    )
    arguments = parser.parse_args()
    workspace = Workspace(Path(__file__).resolve().parent.parent)
    if arguments.all:
        print(json.dumps(workspace.libraries))
        return
    paths = [PurePosixPath(line.strip()) for line in sys.stdin if line.strip()]
    print(json.dumps(workspace.libraries_to_test(paths)))


if __name__ == "__main__":
    main()
