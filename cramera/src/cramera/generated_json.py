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


def write_json_atomically(
    path: Path, payload: Any, indent: Optional[int] = None
) -> None:
    """
    Write a JSON artifact, replacing it only once it is complete.

    A generated artifact (a scene bundle, an index) is read by a server that may be
    polling it at any moment; a failure part-way through a write must not leave a
    truncated file behind for a reader to trip over.

    :param path: Destination path of the file.
    :param payload: JSON-serializable content to write.
    :param indent: Indentation passed to :func:`json.dumps`, or None to compact it.
    """
    temporary = path.with_suffix(path.suffix + ".part")
    temporary.write_text(json.dumps(payload, indent=indent), encoding="utf-8")
    temporary.replace(path)


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
