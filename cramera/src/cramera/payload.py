"""
The shape anything cramera sends to its frontend.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from typing_extensions import Any, Dict


@dataclass(kw_only=True)
class CrameraPayload(ABC):
    """
    One JSON answer to a frontend request.

    Every panel reads ``ok`` first and only then the payload's own keys, so that much is
    shared; :meth:`to_payload` is what each answer defines for itself.
    """

    ok: bool = True
    """
    Whether the request could be answered at all.
    """

    @abstractmethod
    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the frontend expects.
        """
