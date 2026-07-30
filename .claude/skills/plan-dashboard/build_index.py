#!/usr/bin/env python3
"""
Render the master index of every plan from a list of plan summaries.

Generic - takes a list of already-computed plan summaries (see --plans
below) and renders them; it has no idea what a plan actually contains
beyond that. Pair with build_dashboard.py, whose --output JSON summary
gives you the done/total counts to build one of these entries from.

Usage:
    python3 build_index.py --plans /tmp/plans.json --output /tmp/index.html

plans.json shape: a JSON list of objects, each:
    {
        "id": "<plan-id>",
        "title": "...",
        "description": "...",
        "done": <int>,
        "total": <int>,
        "dashboard_url": "<url>" or null   // null if never published yet
    }
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from render_common import create_template_environment, sanitize_http_url


@dataclass
class PlanSummary:
    """
    One plan's entry in the master index, as gathered by the skill.
    """

    id: str
    """The plan's stable identifier: a short kebab-case slug (e.g.
    ``rdr-refactor``), not a UUID - it is chosen by whoever bootstraps the
    plan, doubles as its directory name, and is what a person types as the
    ``<plan-id>`` argument to ``/plan-dashboard``/``/plan-create``, so it
    needs to stay human-readable and human-typable."""

    title: str
    """
    The plan's display title.
    """

    description: str
    """A one-line description shown under the title."""

    done: int
    """
    How many of the plan's items are done.
    """

    total: int
    """How many items the plan has in total."""

    dashboard_url: str | None = None
    """
    The plan's published dashboard Artifact URL, or ``None`` if it has never been
    published yet.
    """

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PlanSummary:
        """
        Build a summary from one entry of ``plans.json``.
        """
        return cls(
            id=data["id"],
            title=data["title"],
            description=data.get("description", ""),
            done=data["done"],
            total=data["total"],
            dashboard_url=sanitize_http_url(data.get("dashboard_url")),
        )

    @property
    def is_complete(self) -> bool:
        """
        Whether every item in the plan is done.
        """
        return self.total > 0 and self.done == self.total

    @property
    def completion_percentage(self) -> float:
        """
        The plan's completion, from 0 to 100.
        """
        return (self.done / self.total * 100) if self.total else 0.0

    @property
    def progress_label(self) -> str:
        """
        The human-readable progress label shown on the plan's card.
        """
        return f"{self.done} / {self.total} done" if self.total else "no items yet"

    @property
    def completion_percentage_label(self) -> str:
        """
        :attr:`completion_percentage`, formatted to one decimal place with a
        trailing ``%`` - ready to drop straight into the progress bar's
        ``width`` style without the template doing any formatting itself.
        """
        return f"{self.completion_percentage:.1f}%"

    @property
    def css_class(self) -> str:
        """
        The plan card's CSS class list: ``"plan-card"``, plus ``"complete"`` once every
        item is done.
        """
        return "plan-card complete" if self.is_complete else "plan-card"


def render_index_page(plans: list[PlanSummary]) -> str:
    """
    Render the full master-index page for every known plan.

    :param plans: Every plan to list, in the order they should appear.
    :return: The rendered HTML page.
    """
    template = create_template_environment().get_template("index.html")
    return template.render(plans=plans)


def main() -> int:
    """
    Parse arguments, render the index page, and write it out.

    See the module docstring for the CLI contract.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--plans", required=True, help="Path to a JSON list of plan summaries"
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the index HTML to"
    )
    arguments = parser.parse_args()

    plans = [
        PlanSummary.from_mapping(entry)
        for entry in json.loads(Path(arguments.plans).read_text())
    ]
    output = render_index_page(plans)
    Path(arguments.output).write_text(output)
    return 0


if __name__ == "__main__":
    sys.exit(main())
