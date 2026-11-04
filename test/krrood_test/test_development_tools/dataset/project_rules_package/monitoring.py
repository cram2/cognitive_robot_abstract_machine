"""
A section of a second module, next to a section it only imports.
"""

from __future__ import annotations

from krrood.development_tools.agents.rule import Section

from .operations import DeploymentRules  # noqa: F401


class AlertingRules(Section):
    """
    The repository's rules on alerts.
    """

    ALERT_ON_FAILURES = "A project rule about alerts"
