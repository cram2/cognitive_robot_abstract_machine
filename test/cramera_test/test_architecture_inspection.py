"""
Inspect handwritten architecture without loading generated database mappings.
"""

from pathlib import Path
from unittest.mock import Mock

from pytest import MonkeyPatch

from cramera.knowledge.architecture_scan import ArchitectureScanner


# %% generated source boundaries
def test_architecture_scan_skips_generated_mappings(
    tmp_path: Path, monkeypatch: MonkeyPatch
) -> None:
    """
    Generated mapping names are rejected before any source file is opened.

    :param tmp_path: Empty repository root for the isolated scan.
    :param monkeypatch: Supplies a directory listing without creating mapping files.
    """
    scanner = ArchitectureScanner(root=str(tmp_path))
    directory_listing = Mock(
        return_value=[(str(tmp_path), [], ["ormatic_interface.py"])]
    )
    parse_module = Mock(return_value=None)
    monkeypatch.setattr(
        "cramera.knowledge.architecture_scan.os.walk", directory_listing
    )
    monkeypatch.setattr(scanner, "_parsed_module", parse_module)

    assert scanner.scan().classes == []
    parse_module.assert_not_called()
