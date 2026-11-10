#!/usr/bin/env python3
"""
Drive the live session kernel in a container on a remote server from this machine::

    live_kernel.py start [--template FILE]   start the kernel, then set the session up
    live_kernel.py perform FILE [--cells A:B] perform FILE's cells through the session
    live_kernel.py raw FILE                  run FILE as it is, outside the session
    live_kernel.py wait NAME                 keep following the output of an earlier run
    live_kernel.py status                    whether the kernel computes or sleeps
    live_kernel.py interrupt                 raise KeyboardInterrupt in the kernel

Commands run detached inside the container and write their output to the exchange
directory, because an attached ``docker exec`` can hang on exit. Every option defaults
to the ``LIVE_*`` environment variable of the same name.
"""

from __future__ import annotations

import argparse
import logging
import os
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import timedelta
from enum import IntEnum, StrEnum
from pathlib import Path, PurePosixPath

logger = logging.getLogger(__name__)

# %% names


class SkillPath(StrEnum):
    """
    The files of this skill, relative to the repository root.
    """

    PERFORM_CELLS = ".claude/skills/cram-live-session/scripts/perform_cells.py"
    """
    The script sending a cell script to the kernel, run inside the container.
    """

    START_SESSION = ".claude/skills/cram-live-session/scripts/start_session.sh"
    """
    The script starting the kernel, run inside the container.
    """

    SESSION_TEMPLATE = ".claude/skills/cram-live-session/templates/robocasa_session.py"
    """
    The cell script that sets a session up.
    """


class SessionFile(StrEnum):
    """
    The files of a session, in its session directory.
    """

    KERNEL_CONNECTION = "kernel.json"
    """
    The connection file the kernel is started with.
    """


class Command(StrEnum):
    """
    The commands of this script.
    """

    START = "start"
    """
    Start the kernel, then set the session up.
    """

    PERFORM = "perform"
    """
    Perform a cell script's cells through the session.
    """

    RAW = "raw"
    """
    Run a script as it is, outside the session.
    """

    WAIT = "wait"
    """
    Keep following the output of an earlier run.
    """

    STATUS = "status"
    """
    Tell whether the kernel computes or sleeps.
    """

    INTERRUPT = "interrupt"
    """
    Raise :class:`KeyboardInterrupt` in the kernel.
    """


class LogMarker(StrEnum):
    """
    The text that marks a line of a run's output.
    """

    EXIT = "EXIT "
    """
    Begins the last line of a run, followed by its exit status.
    """


class RunEnd(IntEnum):
    """
    How following a run ended, as the exit status of this script; a run that finished
    ends it with its own exit status instead.
    """

    DEADLOCKED = 2
    """
    The kernel slept without output, most likely in a deadlock.
    """

    STILL_RUNNING = 3
    """
    The run had not finished when the wait was over.
    """

    KERNEL_GONE = 4
    """
    The kernel stopped running during the run.
    """


# %% the remote container


@dataclass(kw_only=True)
class RemoteContainer:
    """
    A docker container on a server reachable by ssh.
    """

    server: str
    """
    The ssh destination of the server.
    """

    container: str
    """
    The name of the container.
    """

    def run_on_server(self, *, command: str) -> str:
        """
        :param command: A shell command for the server.
        :return: What the command printed.
        """
        return subprocess.run(
            ["ssh", self.server, command], capture_output=True, text=True
        ).stdout

    def copy_to_server(self, *, local_path: Path, server_path: PurePosixPath) -> None:
        """
        :param local_path: A file on this machine.
        :param server_path: Where it goes on the server.
        """
        subprocess.run(
            ["scp", "-q", str(local_path), f"{self.server}:{server_path}"], check=True
        )

    def start_in_container(self, *, command: str) -> None:
        """
        Start a shell command inside the container without waiting for it.

        :param command: The command.
        """
        self.run_on_server(
            command=f"docker exec -d {self.container} bash -c {shlex.quote(command)}"
        )

    def process_id_on_server(self, *, command_line_part: str) -> int | None:
        """
        :param command_line_part: Text the process's command line contains.
        :return: The server's id of the first such process in the container; none if no
            process matches.
        """
        for line in self.run_on_server(
            command=f"docker top {self.container} -o pid,args"
        ).splitlines()[1:]:
            process_id, arguments = line.strip().split(maxsplit=1)
            if command_line_part in arguments and "bash -c" not in arguments:
                return int(process_id)
        return None

    def cpu_ticks(self, *, process_id: int, duration: timedelta) -> int:
        """
        :param process_id: The server's id of a process.
        :param duration: How long to count.
        :return: The CPU ticks the process used in that time.
        """
        read_ticks = f"awk '{{print $14+$15}}' /proc/{process_id}/stat"
        return int(
            self.run_on_server(
                command=f"a=$({read_ticks}); sleep {duration.total_seconds()}; "
                f"b=$({read_ticks}); echo $((b-a))"
            )
        )


# %% the kernel


@dataclass(kw_only=True)
class LiveKernel:
    """
    The live session kernel in a remote container, and the directory both sides share.
    """

    container: RemoteContainer
    """
    The container the kernel runs in.
    """

    exchange_on_server: PurePosixPath
    """
    The server's side of a directory mounted into the container.
    """

    exchange_in_container: PurePosixPath
    """
    The container's side of that directory.
    """

    repository_in_container: PurePosixPath
    """
    The repository inside the container.
    """

    environment_script: PurePosixPath
    """
    The script inside the container that puts CRAM and ROS on the paths.
    """

    session_directory: PurePosixPath
    """
    The directory inside the container holding the kernel's connection file and log.
    """

    wait: timedelta
    """
    How long to follow a run before leaving it running.
    """

    silence: timedelta
    """
    How long the kernel may sleep without output before it counts as deadlocked.
    """

    poll_period: timedelta = timedelta(seconds=5)
    """
    How often a run's output is read.
    """

    busy_cpu_ticks: int = 20
    """
    The fewest CPU ticks in a poll period of a kernel that computes rather than sleeps.
    """

    @property
    def connection_file(self) -> PurePosixPath:
        """
        The kernel's connection file inside the container.
        """
        return self.session_directory / SessionFile.KERNEL_CONNECTION

    @property
    def process_id(self) -> int | None:
        """
        The server's id of the kernel's process; none if it does not run.
        """
        return self.container.process_id_on_server(
            command_line_part=f"-m ipykernel_launcher -f {self.connection_file}"
        )

    def start(self, *, template: Path) -> int:
        """
        Start the kernel, wait for it, and set the session up with a cell script.

        :param template: The cell script setting the session up, run raw.
        :return: The setup run's exit status, or how following it ended.
        """
        if self.process_id is not None:
            logger.info(
                "The kernel runs already; run the template with 'raw' to set the "
                "session up again."
            )
            return 0
        settings = {
            "LIVE_REPOSITORY": self.repository_in_container,
            "LIVE_ENVIRONMENT_SCRIPT": self.environment_script,
            "LIVE_SESSION_DIRECTORY": self.session_directory,
        }
        assignments = " ".join(f"{name}={value}" for name, value in settings.items())
        self.container.start_in_container(
            command=f"{assignments} bash "
            f"{self.repository_in_container / SkillPath.START_SESSION}"
        )
        kernel_start = self.run_detached(
            name="kernel_start",
            command=f"until [ -f {self.connection_file} ]; do sleep 1; done",
        )
        if kernel_start != 0:
            return kernel_start
        return self.send_script(script=template, perform_arguments=["--raw"])

    def send_script(self, *, script: Path, perform_arguments: list[str]) -> int:
        """
        Copy a cell script to the exchange directory and send it to the kernel.

        :param script: The cell script on this machine.
        :param perform_arguments: The arguments of the perform script after the file.
        :return: The run's exit status, or how following it ended.
        """
        name = script.stem
        self.container.copy_to_server(
            local_path=script, server_path=self.exchange_on_server / f"{name}.py"
        )
        arguments = [
            str(self.exchange_in_container / f"{name}.py"),
            *perform_arguments,
            "--connection-file",
            str(self.connection_file),
        ]
        return self.run_detached(
            name=name,
            command=f"cd {self.repository_in_container} && "
            f"source {self.environment_script} && python -u "
            f"{self.repository_in_container / SkillPath.PERFORM_CELLS} "
            f"{shlex.join(arguments)}",
        )

    def run_detached(self, *, name: str, command: str) -> int:
        """
        Run a command in the container, its output going to the exchange directory, and
        follow that output.

        :param name: The name of the run, which names its output file.
        :param command: The command.
        :return: The run's exit status, or how following it ended.
        """
        output = self.exchange_in_container / f"{name}.log"
        self.container.run_on_server(
            command=f"rm -f {self.exchange_on_server / f'{name}.log'}"
        )
        self.container.start_in_container(
            command=f"({command}) > {output} 2>&1; echo {LogMarker.EXIT}$? >> {output}"
        )
        return self.follow(name=name)

    def follow(self, *, name: str) -> int:
        """
        Log a run's output as it grows, until the run ends, :attr:`wait` passes, or the
        kernel sleeps through :attr:`silence` without output.

        :param name: The name of the run.
        :return: The run's exit status, or how following it ended.
        """
        output = self.exchange_on_server / f"{name}.log"
        logged_lines = 0
        silent = timedelta()
        waited = timedelta()
        while waited < self.wait:
            lines = self.container.run_on_server(
                command=f"tail -n +{logged_lines + 1} {output} 2>/dev/null || true"
            ).splitlines()
            logged_lines += len(lines)
            for line in lines:
                logger.info(line)
                if line.startswith(LogMarker.EXIT):
                    return int(line.removeprefix(LogMarker.EXIT))
            silent = timedelta() if lines else silent + self.poll_period
            if silent >= self.silence:
                process_id = self.process_id
                if process_id is None:
                    logger.warning("The kernel no longer runs.")
                    return RunEnd.KERNEL_GONE
                if self.is_sleeping(process_id=process_id):
                    logger.warning(
                        "The kernel sleeps without output for %s: probably deadlocked; "
                        "'interrupt' shows where.",
                        silent,
                    )
                    return RunEnd.DEADLOCKED
                silent = timedelta()
            time.sleep(self.poll_period.total_seconds())
            waited += self.poll_period
        logger.warning("Still running after %s; continue with 'wait %s'.", waited, name)
        return RunEnd.STILL_RUNNING

    def is_sleeping(self, *, process_id: int) -> bool:
        """
        :param process_id: The server's id of the kernel's process.
        :return: Whether the kernel used fewer than :attr:`busy_cpu_ticks` in a poll
            period.
        """
        return (
            self.container.cpu_ticks(process_id=process_id, duration=self.poll_period)
            < self.busy_cpu_ticks
        )

    def interrupt(self) -> None:
        """
        Raise :class:`KeyboardInterrupt` in the kernel.
        """
        self.container.start_in_container(
            command=f"kill -INT $(pgrep -f '^[^ ]*python[^ ]* -m ipykernel_launcher "
            f"-f {self.connection_file}')"
        )


# %% the command line


def option_with_environment_default(
    parser: argparse.ArgumentParser, name: str, default: str | None = None, **options
) -> None:
    """
    Add an option that defaults to the ``LIVE_*`` environment variable of its name.

    :param parser: The parser.
    :param name: The option's name, without dashes.
    :param default: The value when neither the option nor the variable is given; the
        option is required when there is none.
    :param options: Further arguments of the option.
    """
    variable = f"LIVE_{name.upper().replace('-', '_')}"
    value = os.environ.get(variable, default)
    parser.add_argument(
        f"--{name}",
        default=value,
        required=value is None,
        help=f"defaults to ${variable}",
        **options,
    )


def parse_command_line() -> argparse.Namespace:
    """
    :return: The parsed command line.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    option_with_environment_default(parser, "server")
    option_with_environment_default(parser, "container")
    option_with_environment_default(parser, "exchange-on-server", type=PurePosixPath)
    option_with_environment_default(
        parser, "exchange-in-container", "/root/live_logs", type=PurePosixPath
    )
    option_with_environment_default(
        parser, "repository-in-container", "/opt/cram", type=PurePosixPath
    )
    option_with_environment_default(
        parser, "environment-script", "/entrypoint.sh", type=PurePosixPath
    )
    option_with_environment_default(
        parser, "session-directory", "/root/work/live_session", type=PurePosixPath
    )
    option_with_environment_default(parser, "wait-seconds", "540", type=float)
    option_with_environment_default(parser, "silence-seconds", "90", type=float)
    commands = parser.add_subparsers(dest="command", required=True)
    start = commands.add_parser(Command.START)
    start.add_argument(
        "--template",
        type=Path,
        default=Path(__file__).resolve().parents[4] / SkillPath.SESSION_TEMPLATE,
    )
    perform = commands.add_parser(Command.PERFORM)
    perform.add_argument("file", type=Path)
    perform.add_argument(
        "--cells", default=":", help="a slice of the cells, FIRST:LAST"
    )
    commands.add_parser(Command.RAW).add_argument("file", type=Path)
    commands.add_parser(Command.WAIT).add_argument("name")
    commands.add_parser(Command.STATUS)
    commands.add_parser(Command.INTERRUPT)
    return parser.parse_args()


def main() -> None:
    """
    Run the command the command line names.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    arguments = parse_command_line()
    kernel = LiveKernel(
        container=RemoteContainer(
            server=arguments.server, container=arguments.container
        ),
        exchange_on_server=arguments.exchange_on_server,
        exchange_in_container=arguments.exchange_in_container,
        repository_in_container=arguments.repository_in_container,
        environment_script=arguments.environment_script,
        session_directory=arguments.session_directory,
        wait=timedelta(seconds=arguments.wait_seconds),
        silence=timedelta(seconds=arguments.silence_seconds),
    )
    match arguments.command:
        case Command.START:
            sys.exit(kernel.start(template=arguments.template))
        case Command.PERFORM:
            sys.exit(
                kernel.send_script(
                    script=arguments.file,
                    perform_arguments=["--cells", arguments.cells],
                )
            )
        case Command.RAW:
            sys.exit(
                kernel.send_script(script=arguments.file, perform_arguments=["--raw"])
            )
        case Command.WAIT:
            sys.exit(kernel.follow(name=arguments.name))
        case Command.STATUS:
            process_id = kernel.process_id
            if process_id is None:
                logger.info("The kernel does not run.")
                return
            ticks = kernel.container.cpu_ticks(
                process_id=process_id, duration=kernel.poll_period
            )
            logger.info(
                "Kernel process %s on the server: %s CPU ticks in %s",
                process_id,
                ticks,
                kernel.poll_period,
            )
        case Command.INTERRUPT:
            kernel.interrupt()
            logger.info("Interrupt sent.")


if __name__ == "__main__":
    main()
