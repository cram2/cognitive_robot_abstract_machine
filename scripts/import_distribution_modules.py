"""
Import every module of one distribution from where it is installed, so a module that
needs a distribution it does not declare fails where only its declared dependencies are.
"""

from __future__ import annotations

import argparse
import importlib
import sys
from pathlib import Path


def module_names(*, source_directory: Path) -> list[str]:
    """
    :param source_directory: The ``src`` directory of a distribution.
    :return: The dotted name of every module below it.
    """
    names = []
    for path in sorted(source_directory.rglob("*.py")):
        parts = path.relative_to(source_directory).with_suffix("").parts
        if parts[-1] == "__init__":
            parts = parts[:-1]
        names.append(".".join(parts))
    return names


def main() -> None:
    """
    Import the modules of the distribution whose ``src`` directory is given.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "source_directory", type=Path, help="The src directory of the distribution."
    )
    source_directory = parser.parse_args().source_directory.resolve()
    for name in module_names(source_directory=source_directory):
        module = importlib.import_module(name)
        if source_directory in Path(module.__file__).resolve().parents:
            print(f"{name} was imported from the checkout, not the install")
            sys.exit(1)
        print(f"imported {name}")


if __name__ == "__main__":
    main()
