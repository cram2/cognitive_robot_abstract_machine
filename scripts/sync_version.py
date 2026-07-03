"""
Synchronize the root VERSION file into all package _version.py files.
"""

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
VERSION_FILE = ROOT / "VERSION"

PACKAGES = [
    "random_events",
    "krrood",
    "coraplex",
    "giskardpy",
    "probabilistic_model",
    "robokudo",
    "physics_simulators",
    "experiments",
    "semantic_digital_twin",
    "cognitive_robot_abstract_machine",
]


def main() -> None:
    version = VERSION_FILE.read_text().strip()
    for package in PACKAGES:
        target = ROOT / package / "src" / package / "_version.py"
        if package == "cognitive_robot_abstract_machine":
            target = ROOT / package / "_version.py"
        target.write_text(f'__version__ = "{version}"\n')
        print(f"Updated {target}")


if __name__ == "__main__":
    main()
