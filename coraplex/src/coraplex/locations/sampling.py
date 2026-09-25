"""
How a rated set of pose candidates is drawn from.
"""

from __future__ import annotations

from dataclasses import dataclass

from typing_extensions import Optional


@dataclass
class CandidateDraw:
    """
    The terms a backend is asked to draw pose candidates on.
    """

    number_of_samples: int = 2000
    """
    How many candidates to draw.

    Far more than a caller judges properly, since a standing pose inside the furniture
    costs nothing to refuse.
    """

    seed: Optional[int] = None
    """
    Fixes the draw, so a run can be repeated exactly.

    ``None`` draws afresh every time, which is what drawing from a map buys over reading
    it off in the order the map rates it.
    """
