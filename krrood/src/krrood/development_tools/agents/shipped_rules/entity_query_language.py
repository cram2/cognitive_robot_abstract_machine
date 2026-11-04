"""
The rules on writing Entity Query Language queries.
"""

from __future__ import annotations

from krrood.development_tools.agents.rule import CodeReference, Section
from krrood.entity_query_language.factories import a, an


class EntityQueryLanguageRules(Section):
    """
    The rules on writing Entity Query Language queries.
    """

    ARTICLE_MATCHES_ITS_ENTITY = (
        f"Pick {CodeReference(a).name} or {CodeReference(an).name} so the query reads "
        "as correct English with the entity it quantifies: `a(Drawer)` and "
        "`an(Interval.around)`, never `an(Drawer)` or `a(Interval.around)`"
    )
