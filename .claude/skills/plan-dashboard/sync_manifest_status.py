#!/usr/bin/env python3
"""
Auto-correct a plan.yaml's item statuses to ``done`` wherever GitHub confirms the item's
pull request is merged.

This is the one direction plan.yaml's manually-maintained ``status`` field
can be corrected without human judgment. Every other kind of drift
build_dashboard.py's drift banner can report - a pull request number that doesn't
resolve, an item marked done while its pull request is still open, a closed-but-
unmerged pull request against an active status - means something happened that only a
person can interpret (abandoned? reverted? mistyped?), and stays a drift
flag for a human to read, exactly as before. "merged on GitHub" has no such
ambiguity, so leaving it as a standing drift flag would just mean the same
fact gets reported forever until someone manually edits the manifest - this
script is that edit, meant to run automatically as part of every
/plan-dashboard refresh. See plan-schema.md's "Why status is deliberately
thin" section for the full design reasoning.

Usage:
    python3 sync_manifest_status.py \\
        --plan /tmp/plan.yaml \\
        --pr-data /tmp/pr_data.json \\
        [--output /tmp/plan.yaml]

Rewrites --plan in place, or writes to --output if given. Patches only the
exact ``status: <value>`` line of each corrected item - every other line is
left byte-for-byte untouched, so comments, key order, string wrapping, and
quoting all survive exactly. A full YAML load-mutate-dump round trip was
tried and rejected: even a library that claims to preserve formatting
(ruamel.yaml) re-flows long wrapped strings and normalizes ``null``
spellings on every write, turning a one-line status fix into an unreadable,
unrelated diff across the whole file.

pr_data.json shape: identical to build_dashboard.py's module docstring.

Prints a one-line JSON summary to stdout:
    {"corrected": [{"id": "<item id>", "previous_status": "<status>"}, ...]}
An empty ``corrected`` list means nothing needed fixing.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from build_dashboard import (
    ItemStatus,
    LiveState,
    PlanValidationError,
    classify_live_state,
    load_pull_requests_by_repository,
    validate_plan,
)

_ITEM_START_PATTERN = re.compile(r"^\s*- id:")
_STATUS_LINE_PATTERN = re.compile(r"^(\s*status:\s*)(\S+)\s*$")


class MissingStatusLineError(ValueError):
    """Raised when an item slated for correction has no ``status:`` line in
    its manifest block - the manifest text and the parsed data have gone out
    of sync."""


@dataclass
class StatusCorrection:
    """
    One item whose manifest ``status`` was corrected to ``done``.
    """

    item_identifier: str
    """
    The corrected item's effective id (``id``, or ``branch`` if unset).
    """

    previous_status: ItemStatus
    """
    The status the manifest held before correction.
    """

    def to_json_dict(self) -> dict[str, str]:
        """
        Render to the plain-dict shape the calling skill expects.
        """
        return {"id": self.item_identifier, "previous_status": self.previous_status}


def find_items_to_correct(
    plan: dict[str, Any],
    pull_requests_by_repository: dict[str, dict[str, Any]],
) -> list[dict[str, Any]]:
    """Find every item whose pull request is merged on GitHub but whose manifest
    ``status`` isn't already ``done``.

    :param plan: The raw, freshly-``yaml.safe_load``-ed plan.yaml content.
    :param pull_requests_by_repository: Live pull request state for every repository
        referenced by the plan's items.
    :return: The raw item mappings that need correcting, in manifest order.
    """
    default_repository = plan["default_repository"]
    return [
        item
        for item in plan.get("items", [])
        if item.get("status") != ItemStatus.DONE.value
        and classify_live_state(
            item.get("pull_request_number"),
            item.get("repository") or default_repository,
            pull_requests_by_repository,
        )
        is LiveState.MERGED
    ]


def apply_status_corrections(
    plan_text: str, items_to_correct: list[dict[str, Any]]
) -> tuple[str, list[StatusCorrection]]:
    """
    Patch ``plan_text``'s ``status:`` line for each item in ``items_to_correct`` to
    ``done``, leaving every other line untouched.

    :param plan_text: The plan.yaml file's raw text.
    :param items_to_correct: Raw item mappings (as found by
        :func:`find_items_to_correct`) whose status line should become
        ``done``.
    :raises ValueError: If an item's ``status:`` line can't be found in its
        block - the manifest and the parsed data have gone out of sync.
    :return: The patched text, and the correction actually applied to each
        item, in the same order as ``items_to_correct``.
    """
    identifiers_to_correct = {
        item.get("id") or item.get("branch") for item in items_to_correct
    }
    lines = plan_text.split("\n")
    item_starts = [
        index for index, line in enumerate(lines) if _ITEM_START_PATTERN.match(line)
    ]
    item_starts.append(len(lines))

    corrections: list[StatusCorrection] = []
    for start, end in zip(item_starts, item_starts[1:]):
        item_identifier = lines[start].strip().removeprefix("- id:").strip()
        if item_identifier not in identifiers_to_correct:
            continue
        status_line_index = next(
            (
                offset
                for offset, line in enumerate(lines[start:end])
                if _STATUS_LINE_PATTERN.match(line)
            ),
            None,
        )
        if status_line_index is None:
            raise MissingStatusLineError(
                f"item {item_identifier!r} has no status: line to correct"
            )
        line_index = start + status_line_index
        match = _STATUS_LINE_PATTERN.match(lines[line_index])
        corrections.append(
            StatusCorrection(
                item_identifier=item_identifier,
                previous_status=ItemStatus(match.group(2)),
            )
        )
        lines[line_index] = f"{match.group(1)}{ItemStatus.DONE.value}"

    return "\n".join(lines), corrections


def main() -> int:
    """
    Parse arguments, apply corrections, and print the summary.

    See the module docstring for the CLI contract.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, help="Path to plan.yaml")
    parser.add_argument(
        "--pr-data",
        required=True,
        help='Path to a JSON file: {"owner/repo": {"pr_number": {...}}}',
    )
    parser.add_argument(
        "--output",
        default=None,
        help="Path to write the corrected plan.yaml to - defaults to --plan (in place)",
    )
    arguments = parser.parse_args()

    plan_path = Path(arguments.plan)
    plan_text = plan_path.read_text()
    plan = yaml.safe_load(plan_text)

    try:
        validate_plan(plan)
    except PlanValidationError as error:
        print(f"plan.yaml failed validation: {error}", file=sys.stderr)
        return 1

    raw_pull_request_data = json.loads(Path(arguments.pr_data).read_text())
    pull_requests_by_repository = load_pull_requests_by_repository(
        raw_pull_request_data
    )

    items_to_correct = find_items_to_correct(plan, pull_requests_by_repository)
    corrected_text, corrections = apply_status_corrections(plan_text, items_to_correct)

    output_path = Path(arguments.output) if arguments.output else plan_path
    output_path.write_text(corrected_text)

    print(json.dumps({"corrected": [c.to_json_dict() for c in corrections]}))
    return 0


if __name__ == "__main__":
    sys.exit(main())
