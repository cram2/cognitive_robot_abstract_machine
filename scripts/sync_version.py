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
    "cramera",
    "giskardpy",
    "probabilistic_model",
    "robokudo",
    "physics_simulators",
    "experiments",
    "semantic_digital_twin",
    "cognitive_robot_abstract_machine",
    "basstler",
]

FLAT_LAYOUT_PACKAGES = {"cognitive_robot_abstract_machine", "basstler"}
"""
The packages that *are* their own directory rather than living under a ``src`` one, so
their ``_version.py`` sits one level up from where the rest of them keep it.
"""


def main() -> None:
    version = VERSION_FILE.read_text().strip()
    for package in PACKAGES:
        if package in FLAT_LAYOUT_PACKAGES:
            target = ROOT / package / "_version.py"
        else:
            target = ROOT / package / "src" / package / "_version.py"
        target.write_text(f'__version__ = "{version}"\n')
        print(f"Updated {target}")


if __name__ == "__main__":
    main()
