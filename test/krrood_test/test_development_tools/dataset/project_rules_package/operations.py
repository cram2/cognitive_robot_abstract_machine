"""
Sections declared out of alphabetical order.
"""

from __future__ import annotations

from krrood.development_tools.agents.rule import Section


class ReleasesRules(Section):
    """
    The repository's rules on releases.
    """

    TAG_RELEASES = "A project rule about releases"


class DeploymentRules(Section):
    """
    The repository's rules on deploying.
    """

    DEPLOY_FROM_TAGS = "A project rule about deploying"
