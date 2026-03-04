from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class ColorLegend:
    name: str = field(default="Other")
    color: str = field(default="white")
