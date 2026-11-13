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
directory, because an attached ``docker exec`` can hang on exit; that output streams
back as it is written. Every call to the server shares one ssh connection. Every option
defaults to the ``LIVE_*`` environment variable of the same name.
"""

from __future__ import annotations

import argparse
import logging
import os
import selectors
import shlex
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import timedelta
from enum import IntEnum, StrEnum
from pathlib import Path, PurePosixPath

from typing_extensions import BinaryIO

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


class RunEndExitStatus(IntEnum):
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

    CONNECTION_LOST = 5
    """
    The connection streaming the run's output closed before the run ended.
    """


class SecureShellOption(StrEnum):
    """
    The ssh client options this script sets.
    """

    CONTROL_MASTER = "ControlMaster"
    """
    Whether a call opens a connection later calls share.
    """

    CONTROL_PATH = "ControlPath"
    """
    The socket of the shared connection.
    """

    CONTROL_PERSIST = "ControlPersist"
    """
    How long the shared connection stays open after its last call.
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

    shared_connection_socket: Path = field(
        default_factory=lambda: Path.home() / ".ssh" / "live-kernel-%C"
    )
    """
    The socket of the ssh connection every call shares; ``%C`` stands for a hash of the
    destination. Its path must stay shorter than a Unix socket's limit of 107 bytes.
    """

    shared_connection_lifetime: timedelta = timedelta(minutes=10)
    """
    How long the shared connection stays open after its last call.
    """

    @property
    def secure_shell_command(self) -> list[str]:
        """
        The call of the ssh program to the server, sharing its connection, before the
        remote command.
        """
        options = {
            SecureShellOption.CONTROL_MASTER: "auto",
            SecureShellOption.CONTROL_PATH: self.shared_connection_socket,
            SecureShellOption.CONTROL_PERSIST: int(
                self.shared_connection_lifetime.total_seconds()
            ),
        }
        arguments = ["ssh"]
        for option, value in options.items():
            arguments += ["-o", f"{option}={value}"]
        return [*arguments, self.server]

    def run_on_server(self, *, command: str, standard_input: str = "") -> str:
        """
        :param command: A shell command for the server.
        :param standard_input: What the command reads from its standard input.
        :return: What the command printed.
        """
        return subprocess.run(
            [*self.secure_shell_command, command],
            input=standard_input,
            capture_output=True,
            text=True,
        ).stdout

    def start_on_server(self, *, command: str) -> subprocess.Popen[bytes]:
        """
        :param command: A shell command for the server.
        :return: The running command, its output unbuffered, to read as it is printed.
        """
        return subprocess.Popen(
            [*self.secure_shell_command, command],
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            bufsize=0,
        )

    def detached_container_command(self, *, command: str) -> str:
        """
        :param command: A shell command for inside the container.
        :return: The server's shell command starting it inside the container without
            waiting for it.
        """
        return f"docker exec -d {self.container} bash -c {shlex.quote(command)}"

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

    busy_check_period: timedelta = timedelta(seconds=5)
    """
    How long the kernel's CPU use is counted to tell whether it computes or sleeps.
    """

    busy_cpu_ticks: int = 20
    """
    The fewest CPU ticks in a :attr:`busy_check_period` of a kernel that computes rather
    than sleeps.
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
        self.container.run_on_server(
            command=self.container.detached_container_command(
                command=f"{assignments} bash "
                f"{self.repository_in_container / SkillPath.START_SESSION}"
            )
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
            script=script,
        )

    def run_detached(
        self, *, name: str, command: str, script: Path | None = None
    ) -> int:
        """
        Run a command in the container, its output going to the exchange directory, and
        follow that output.

        :param name: The name of the run, which names its output file and its script.
        :param command: The command.
        :param script: A file on this machine copied to the exchange directory before
            the command starts.
        :return: The run's exit status, or how following it ended.
        """
        output = self.exchange_in_container / f"{name}.log"
        steps = [f"rm -f {self.exchange_on_server / f'{name}.log'}"]
        if script is not None:
            steps.append(f"cat > {self.exchange_on_server / f'{name}.py'}")
        steps.append(
            self.container.detached_container_command(
                command=f"({command}) > {output} 2>&1; "
                f"echo {LogMarker.EXIT}$? >> {output}"
            )
        )
        self.container.run_on_server(
            command=" && ".join(steps),
            standard_input="" if script is None else script.read_text(),
        )
        return self.follow(name=name)

    def follow(self, *, name: str) -> int:
        """
        Log a run's output as it is written, until the run ends, :attr:`wait` passes, or
        the kernel sleeps through :attr:`silence` without output.

        :param name: The name of the run.
        :return: The run's exit status, or how following it ended.
        """
        tail = self.container.start_on_server(
            command=f"tail -n +1 -F {self.exchange_on_server / f'{name}.log'} "
            "2>/dev/null"
        )
        exit_status = RunOutputFollower(
            kernel=self, output=tail.stdout, name=name
        ).follow()
        tail.terminate()
        return exit_status

    def is_sleeping(self, *, process_id: int) -> bool:
        """
        :param process_id: The server's id of the kernel's process.
        :return: Whether the kernel used fewer than :attr:`busy_cpu_ticks` in a
            :attr:`busy_check_period`.
        """
        return (
            self.container.cpu_ticks(
                process_id=process_id, duration=self.busy_check_period
            )
            < self.busy_cpu_ticks
        )

    def interrupt(self) -> None:
        """
        Raise :class:`KeyboardInterrupt` in the kernel.
        """
        self.container.run_on_server(
            command=self.container.detached_container_command(
                command=f"kill -INT $(pgrep -f '^[^ ]*python[^ ]* -m ipykernel_launcher "
                f"-f {self.connection_file}')"
            )
        )


# %% following a run


@dataclass(kw_only=True)
class RunOutputFollower:
    """
    Logs a run's output as it is written, until the run ends, the kernel's
    :attr:`~LiveKernel.wait` passes, or the kernel sleeps through its
    :attr:`~LiveKernel.silence` without output.
    """

    kernel: LiveKernel
    """
    The kernel performing the run.
    """

    output: BinaryIO
    """
    The run's output as it is written, unbuffered.
    """

    name: str
    """
    The name of the run.
    """

    started: float = field(default_factory=time.monotonic)
    """
    When following began, on the :func:`time.monotonic` clock.
    """

    last_output: float = field(default_factory=time.monotonic)
    """
    When the run last printed or the kernel was last seen computing, on the
    :func:`time.monotonic` clock.
    """

    readable: selectors.DefaultSelector = field(
        default_factory=selectors.DefaultSelector
    )
    """
    Tells when :attr:`output` has a line to read.
    """

    def __post_init__(self) -> None:
        self.readable.register(self.output, selectors.EVENT_READ)

    @property
    def remaining_wait(self) -> timedelta:
        """
        How long following goes on before the run is left running.
        """
        return self.kernel.wait - timedelta(seconds=time.monotonic() - self.started)

    @property
    def silent(self) -> timedelta:
        """
        How long the run has printed nothing.
        """
        return timedelta(seconds=time.monotonic() - self.last_output)

    def follow(self) -> int:
        """
        :return: The run's exit status, or how following it ended.
        """
        while self.remaining_wait > timedelta():
            if self.line_arrives():
                exit_status = self.log_next_line()
            else:
                exit_status = self.check_silent_kernel()
            if exit_status is not None:
                return exit_status
        logger.warning(
            "Still running after %s; continue with 'wait %s'.",
            self.kernel.wait,
            self.name,
        )
        return RunEndExitStatus.STILL_RUNNING

    def line_arrives(self) -> bool:
        """
        :return: Whether a line arrives before the wait or the silence is over.
        """
        timeout = min(self.remaining_wait, self.kernel.silence - self.silent)
        return bool(self.readable.select(timeout=max(timeout.total_seconds(), 0)))

    def log_next_line(self) -> int | None:
        """
        :return: The run's exit status if the line ends the run, or
            :attr:`RunEndExitStatus.CONNECTION_LOST` if the output closed; none if the
            run goes on.
        """
        line = self.output.readline()
        if not line:
            logger.warning("The connection to the server closed.")
            return RunEndExitStatus.CONNECTION_LOST
        text = line.decode(errors="replace").rstrip("\n")
        logger.info(text)
        if text.startswith(LogMarker.EXIT):
            return int(text.removeprefix(LogMarker.EXIT))
        self.last_output = time.monotonic()
        return None

    def check_silent_kernel(self) -> int | None:
        """
        :return: How following ends if the run was silent for the kernel's
            :attr:`~LiveKernel.silence` and the kernel is gone or sleeps; none if the
            run goes on.
        """
        silent = self.silent
        if silent < self.kernel.silence:
            return None
        process_id = self.kernel.process_id
        if process_id is None:
            logger.warning("The kernel no longer runs.")
            return RunEndExitStatus.KERNEL_GONE
        if self.kernel.is_sleeping(process_id=process_id):
            logger.warning(
                "The kernel sleeps without output for %s: probably deadlocked; "
                "'interrupt' shows where.",
                silent,
            )
            return RunEndExitStatus.DEADLOCKED
        self.last_output = time.monotonic()
        return None


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
                process_id=process_id, duration=kernel.busy_check_period
            )
            logger.info(
                "Kernel process %s on the server: %s CPU ticks in %s",
                process_id,
                ticks,
                kernel.busy_check_period,
            )
        case Command.INTERRUPT:
            kernel.interrupt()
            logger.info("Interrupt sent.")


if __name__ == "__main__":
    main()
