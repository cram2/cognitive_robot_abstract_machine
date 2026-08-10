from __future__ import annotations

import atexit
import os
import shutil
import tempfile
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path

import psutil
from krrood.singleton import SingletonMeta
from typing_extensions import ClassVar


class ProcessLiveness(ABC):
    """
    Answers whether a process is still running.
    """

    @abstractmethod
    def is_alive(self, process_id: int) -> bool:
        """
        :param process_id: The process to ask about.
        :return: Whether a process with that id is running.
        """


@dataclass
class RunningProcessLiveness(ProcessLiveness):
    """
    Answers from the process table of the machine this runs on.
    """

    def is_alive(self, process_id: int) -> bool:
        return psutil.pid_exists(process_id)


@dataclass
class MeshFileStorage(metaclass=SingletonMeta):
    """
    The place this process writes exported mesh files to.

    Each export gets a directory of its own beneath a single root, so a material or
    texture written beside a mesh belongs to that mesh alone. The root is removed when the
    process exits, which makes an exported path valid for exactly as long as the process
    that wrote it.

    ..note:: The root is created when this class is first instantiated, not on import, so
        a process that exports no mesh writes nothing.
    """

    root_prefix: ClassVar[str] = "semantic_digital_twin_meshes_"
    """
    Marks a temporary directory as a mesh session root of this package.
    """

    temporary_directory: Path = field(
        default_factory=lambda: Path(tempfile.gettempdir())
    )
    """
    The directory the root is created in, and the one searched for abandoned roots.
    """

    process_liveness: ProcessLiveness = field(default_factory=RunningProcessLiveness)
    """
    Decides whether the process a root is named after is still running.
    """

    owner_process_id: int = field(init=False, default_factory=os.getpid)
    """
    The process that created the root, and the only one permitted to remove it.
    """

    root: Path = field(init=False)
    """
    The directory every mesh this process exports lives beneath.
    """

    def __post_init__(self) -> None:
        self.root = Path(
            tempfile.mkdtemp(
                prefix=f"{self.root_prefix}{self.owner_process_id}_",
                dir=self.temporary_directory,
            )
        )
        atexit.register(self.remove)
        self.remove_abandoned_roots()

    @staticmethod
    def create_mesh_directory(parent: Path) -> Path:
        """
        Create a directory holding a single mesh export.

        The name carries the creating process, so a mesh file named after its directory
        stays unique even against exports placed under a different parent.

        :param parent: The directory to create the mesh's directory in.
        :return: The path of the created directory.
        """
        return Path(tempfile.mkdtemp(prefix=f"{os.getpid()}_", dir=parent))

    def allocate_directory(self) -> Path:
        """
        Create a directory holding a single mesh export inside this process's root.

        :return: The path of the created directory.
        """
        return self.create_mesh_directory(self.root)

    def remove(self) -> None:
        """
        Delete the root and every mesh exported into it.

        Does nothing in a process that inherited the root by forking, so a child exiting
        cannot take the files away from the parent that owns them.
        """
        if os.getpid() != self.owner_process_id:
            return
        atexit.unregister(self.remove)
        shutil.rmtree(self.root, ignore_errors=True)

    def remove_abandoned_roots(self) -> None:
        """
        Delete the roots of processes that are no longer running.

        A root outlives its process whenever that process is killed instead of allowed to
        exit, because nothing runs at shutdown to clean up after it then.

        ..warning:: A process is recognised by its id, which is unique only within a
            process namespace. Containers sharing a temporary directory see each other's
            ids and would need another way to tell a live root apart.
        """
        for candidate in self.temporary_directory.glob(f"{self.root_prefix}*"):
            if candidate == self.root:
                continue
            process_id = candidate.name[len(self.root_prefix) :].split("_")[0]
            if not process_id.isdigit():
                continue
            if self.process_liveness.is_alive(int(process_id)):
                continue
            shutil.rmtree(candidate, ignore_errors=True)
