"""
Report where a separately installed CRAMERA wheel resolves its package.
"""

import json
from importlib.metadata import distribution
from pathlib import Path

import cramera


print(
    json.dumps(
        {
            "package": str(Path(cramera.__file__).resolve().parent),
            "providers": [
                entry.value
                for entry in distribution("cramera").entry_points
                if entry.group == "coraplex.visualizations"
            ],
        }
    )
)
