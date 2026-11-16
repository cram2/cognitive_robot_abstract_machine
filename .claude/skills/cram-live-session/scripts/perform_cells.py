#!/usr/bin/env python3
"""
Send a cell script to the live session kernel and log what it prints as it prints it.
Runs where the kernel runs, with CRAM on the path::

    perform_cells.py SCRIPT               # perform every cell through the session
    perform_cells.py SCRIPT --cells 2:4   # perform cells 2 and 3
    perform_cells.py SCRIPT --raw         # run the whole file, outside the session

A performed cell is kept in the session's history, so it can be reverted or replayed; a
raw run is not. The exit status is 1 if the kernel reported an error.
"""

from __future__ import annotations

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import timedelta
from enum import StrEnum
from pathlib import Path

from jupyter_client import BlockingKernelClient
from typing_extensions import Any

from experiments.live_session.cells import Cell, CellScript
from krrood.patterns.called_positionally import called_positionally

# %% kernel messages


class KernelMessageType(StrEnum):
    """
    The types of the messages a kernel publishes while it executes code, which the
    output log shows.
    """

    STREAM = "stream"
    """
    Text the code printed.
    """

    EXECUTE_RESULT = "execute_result"
    """
    The value of the code's last expression.
    """

    DISPLAY_DATA = "display_data"
    """
    A value the code displayed.
    """

    ERROR = "error"
    """
    The exception the code raised.
    """


class ExecutionStatus(StrEnum):
    """
    How the kernel says an execution ended.
    """

    OK = "ok"
    """
    The code ran without raising.
    """


@dataclass(kw_only=True)
class KernelOutputLog:
    """
    Logs what a kernel prints, the cells it is sent and how each execution ended.
    """

    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))
    """
    The logger everything is written to.
    """

    @called_positionally
    def log_message(self, message: dict[str, Any]) -> None:
        """
        :param message: A message the kernel published while executing.
        """
        content = message["content"]
        match message["msg_type"]:
            case KernelMessageType.STREAM:
                self.logger.info(content["text"].rstrip("\n"))
            case KernelMessageType.EXECUTE_RESULT | KernelMessageType.DISPLAY_DATA:
                self.logger.info(content["data"].get("text/plain", ""))
            case KernelMessageType.ERROR:
                self.logger.error("\n".join(content["traceback"]))

    def log_cell_start(self, *, index: int, cell: Cell) -> None:
        """
        :param index: The position of the cell in its script.
        :param cell: The cell about to be performed.
        """
        self.logger.info("=== cell %s: %s", index, cell.title)

    def log_status(self, *, status: str) -> None:
        """
        :param status: How the kernel said an execution ended.
        """
        self.logger.info("[status: %s]", status)


# %% the kernel


@dataclass(kw_only=True)
class KernelConnection:
    """
    A connection to a running Jupyter kernel.
    """

    connection_file: Path
    """
    The connection file the kernel was started with.
    """

    timeout: timedelta
    """
    How long one execution may take before the client gives up waiting for it.
    """

    output_log: KernelOutputLog = field(default_factory=KernelOutputLog)
    """
    Where the kernel's output goes.
    """

    client: BlockingKernelClient = field(init=False)
    """
    The client talking to the kernel.
    """

    def __post_init__(self) -> None:
        self.client = BlockingKernelClient(connection_file=str(self.connection_file))
        self.client.load_connection_file()
        self.client.start_channels()

    def execute(self, *, code: str) -> bool:
        """
        Run code in the kernel, logging its output as it arrives.

        :param code: The code.
        :return: Whether the kernel reported no error.
        """
        reply = self.client.execute_interactive(
            code,
            output_hook=self.output_log.log_message,
            timeout=self.timeout.total_seconds(),
        )
        status = reply["content"]["status"]
        self.output_log.log_status(status=status)
        return status == ExecutionStatus.OK


# %% performing cells


def perform_code(cell: Cell, session_name: str) -> str:
    """
    :param cell: A cell.
    :param session_name: The name the kernel holds its live session under.
    :return: Code that makes the kernel's session perform the cell in the kernel's own
        namespace.
    """
    return (
        "from experiments.live_session.cells import Cell as _Cell\n"
        f"{session_name}.perform(cell=_Cell(title={cell.title!r}, "
        f"source={cell.source!r}), namespace=globals())"
    )


def main() -> None:
    """
    Parse the command line and send the script.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s", stream=sys.stdout)
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("script", type=Path)
    parser.add_argument("--cells", default=":", help="a slice of the cells, FIRST:LAST")
    parser.add_argument("--raw", action="store_true")
    parser.add_argument("--session-name", default="session")
    parser.add_argument(
        "--connection-file",
        type=Path,
        default=Path("/root/work/live_session/kernel.json"),
    )
    parser.add_argument("--timeout-seconds", type=float, default=3600)
    arguments = parser.parse_args()

    kernel = KernelConnection(
        connection_file=arguments.connection_file,
        timeout=timedelta(seconds=arguments.timeout_seconds),
    )
    if arguments.raw:
        sys.exit(0 if kernel.execute(code=arguments.script.read_text()) else 1)
    first, _, last = arguments.cells.partition(":")
    cells = CellScript.from_file(path=arguments.script).cells
    selected = slice(int(first) if first else None, int(last) if last else None)
    for index, cell in list(enumerate(cells))[selected]:
        kernel.output_log.log_cell_start(index=index, cell=cell)
        if not kernel.execute(code=perform_code(cell, arguments.session_name)):
            sys.exit(1)


if __name__ == "__main__":
    main()
