#!/usr/bin/env python3
"""
Classify one item's dependencies as ready or not-ready to build on, reusing
build_dashboard.py's own live-state classification and readiness rule.

plan-item-kickoff and plan-item-resolve both need this exact question
answered - "is it actually safe to stack new work on top of item X's
dependencies?" - and previously re-derived the rule
(:meth:`build_dashboard.Item.is_ready_to_unblock_dependents`) in their own
SKILL.md prose instead of calling the code that already implements and tests
it. This script is that single call site.

Usage:
    python3 check_dependency_readiness.py \\
        --plan /tmp/plan.yaml \\
        --pr-data /tmp/pr_data.json \\
        --item <item-id>

pr_data.json shape: identical to build_dashboard.py's module docstring.

Prints a one-line JSON list to stdout, one entry per entry in the item's
``depends_on``, in that order:
    [{"identifier": "<dependency id>", "title": "<dependency title>",
      "live_state": "<LiveState value>", "is_ready": <bool>}, ...]
A dependency identifier that doesn't resolve to a known item is reported
with ``"title": null, "live_state": null, "is_ready": false`` - a broken
``depends_on`` reference is never silently treated as ready.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import yaml

from build_dashboard import (
    Plan,
    PlanValidationError,
    PullRequestsByRepository,
    classify_live_state,
    load_pull_requests_by_repository,
    validate_plan,
)


class UnknownItemError(ValueError):
    """
    Raised when the requested item id doesn't exist in the plan.
    """


def dependency_readiness(
    plan: Plan,
    item_identifier: str,
    pull_requests_by_repository: PullRequestsByRepository,
) -> list[dict[str, Any]]:
    """
    Classify every dependency of ``item_identifier`` as ready or not.

    :param plan: The already-validated plan.
    :param item_identifier: The effective identifier (``id`` or ``branch``) of the item
        whose dependencies should be checked.
    :param pull_requests_by_repository: Live pull request state for every repository
        referenced by the plan's items.
    :raises UnknownItemError: If ``item_identifier`` isn't in the plan.
    :return: One ready-to-serialize dict per entry in the item's ``depends_on``, in that
        order.
    """
    items_by_identifier = {item.identifier: item for item in plan.items}
    item = items_by_identifier.get(item_identifier)
    if item is None:
        raise UnknownItemError(f"no item {item_identifier!r} in plan {plan.id!r}")

    results: list[dict[str, Any]] = []
    for dependency_identifier in item.depends_on:
        dependency = items_by_identifier.get(dependency_identifier)
        if dependency is None:
            results.append(
                {
                    "identifier": dependency_identifier,
                    "title": None,
                    "live_state": None,
                    "is_ready": False,
                }
            )
            continue
        dependency.live_state = classify_live_state(
            dependency.pull_request_number,
            dependency.repository or plan.default_repository,
            pull_requests_by_repository,
        )
        results.append(
            {
                "identifier": dependency.identifier,
                "title": dependency.title,
                "live_state": dependency.live_state.value,
                "is_ready": dependency.is_ready_to_unblock_dependents(),
            }
        )
    return results


def main() -> int:
    """
    Parse arguments, classify the item's dependencies, and print the result.

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
    parser.add_argument("--item", required=True, help="The item id to check")
    arguments = parser.parse_args()

    raw_plan = yaml.safe_load(Path(arguments.plan).read_text())
    try:
        validate_plan(raw_plan)
    except PlanValidationError as error:
        print(str(error), file=sys.stderr)
        return 1
    plan = Plan.from_mapping(raw_plan)

    raw_pull_request_data = json.loads(Path(arguments.pr_data).read_text())
    pull_requests_by_repository = load_pull_requests_by_repository(
        raw_pull_request_data
    )

    try:
        results = dependency_readiness(
            plan, arguments.item, pull_requests_by_repository
        )
    except UnknownItemError as error:
        print(str(error), file=sys.stderr)
        return 1

    print(json.dumps(results))
    return 0


if __name__ == "__main__":
    sys.exit(main())
