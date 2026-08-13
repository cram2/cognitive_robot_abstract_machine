"""
Reading a JSON file that a previous cramera run wrote.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from typing_extensions import Any, Optional

from cramera.logging_setup import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True)
class GeneratedJson:
    """
    A JSON artifact produced by an earlier run: a scene bundle, or the scan cache.

    A failed run can leave one half-written, so reading degrades to "absent" instead of
    refusing to start.
    """

    path: Path
    """
    Where the artifact lives.
    """

    def read(self) -> Optional[Any]:
        """
        The artifact's content, or None when it is missing or unreadable.

        A missing file is a normal state (nothing was generated yet) and stays silent;
        only an unreadable or corrupt file is worth a warning.
        """
        if not self.path.is_file():
            return None
        try:
            return json.loads(self.path.read_text(encoding="utf-8", errors="replace"))
        except (OSError, ValueError) as error:
            logger.warning("ignoring unreadable %s: %s", self.path, error)
            return None
