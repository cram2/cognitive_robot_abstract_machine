#!/usr/bin/env python3
"""
Send a cell script to the live-session kernel and print what the kernel prints, as it
prints it. Runs where the kernel runs, with CRAM on the path::

    perform_cells.py SCRIPT               # perform every cell through the session
    perform_cells.py SCRIPT --cells 2:4   # perform cells 2 and 3
    perform_cells.py SCRIPT --raw         # run the whole file, outside the session

A performed cell is kept in the session's history with the moment before it, so it can
be reverted or replayed; a raw run is not. The exit status is 1 if the kernel reported
an error.
"""

from __future__ import annotations

import argparse
import sys
from dataclasses import dataclass, field
from pathlib import Path

from jupyter_client import BlockingKernelClient

from experiments.live_session.cells import Cell, CellScript

# %% the kernel


@dataclass
class KernelConnection:
    """
    A connection to a running Jupyter kernel.
    """

    connection_file: Path
    """
    The connection file the kernel was started with.
    """

    timeout_seconds: float
    """
    How long one execution may take before the client gives up waiting for it.
    """

    client: BlockingKernelClient = field(init=False)
    """
    The client talking to the kernel.
    """

    def __post_init__(self) -> None:
        self.client = BlockingKernelClient(connection_file=str(self.connection_file))
        self.client.load_connection_file()
        self.client.start_channels()

    def execute(self, code: str) -> bool:
        """
        Run code in the kernel, printing its output as it arrives.

        :param code: The code.
        :return: Whether the kernel reported no error.
        """
        reply = self.client.execute_interactive(
            code, output_hook=print_kernel_output, timeout=self.timeout_seconds
        )
        status = reply["content"]["status"]
        print(f"[status: {status}]", flush=True)
        return status == "ok"


def print_kernel_output(message: dict) -> None:
    """
    :param message: A message the kernel published while executing.
    """
    content = message["content"]
    match message["msg_type"]:
        case "stream":
            print(content["text"], end="", flush=True)
        case "execute_result" | "display_data":
            print(content["data"].get("text/plain", ""), flush=True)
        case "error":
            print("\n".join(content["traceback"]), flush=True)


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
        f"{session_name}.perform(_Cell(title={cell.title!r}, source={cell.source!r}), "
        "globals())"
    )


def main() -> None:
    """
    Parse the command line and send the script.
    """
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

    kernel = KernelConnection(arguments.connection_file, arguments.timeout_seconds)
    if arguments.raw:
        sys.exit(0 if kernel.execute(arguments.script.read_text()) else 1)
    first, _, last = arguments.cells.partition(":")
    cells = CellScript.from_file(arguments.script).cells
    selected = slice(int(first) if first else None, int(last) if last else None)
    for index, cell in list(enumerate(cells))[selected]:
        print(f"=== cell {index}: {cell.title}", flush=True)
        if not kernel.execute(perform_code(cell, arguments.session_name)):
            sys.exit(1)


if __name__ == "__main__":
    main()
