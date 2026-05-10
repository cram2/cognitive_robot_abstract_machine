from __future__ import annotations

from dataclasses import dataclass


@dataclass
class AncestorBase:
    shared_field: str = ""

    def shared_method(self) -> int: ...

