from __future__ import annotations

from dataclasses import dataclass, field

from typing_extensions import Any, Dict, List

# %% node details


@dataclass
class NodeInfoSection:
    """
    A group of details of a plan node, shown under one heading.
    """

    heading: str
    """
    The name the group is shown under.
    """

    entries: Dict[str, Any] = field(default_factory=dict)
    """
    The details of the group, keyed by the name each is shown under.
    """

    def to_lines(self) -> List[str]:
        """
        :return: The heading followed by one line per detail.
        """
        return [f"--- {self.heading} ---"] + [
            f"{name}: {value}" for name, value in self.entries.items()
        ]


@dataclass
class NodeInfo:
    """
    The details of a plan node, shown when it is clicked in the plan visualization.
    """

    sections: List[NodeInfoSection] = field(default_factory=list)
    """
    The groups of details, in the order they are shown.
    """

    def to_dict(self) -> Dict[str, Dict[str, Any]]:
        """
        :return: The details of each section, keyed by the heading of that section.
        """
        return {section.heading: section.entries for section in self.sections}

    def to_lines(self) -> List[str]:
        """
        :return: All sections rendered as the lines the visualization displays.
        """
        return [line for section in self.sections for line in section.to_lines()]
