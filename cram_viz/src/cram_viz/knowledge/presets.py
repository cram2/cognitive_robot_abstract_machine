"""
Ready-made EQL queries for the EQL panel.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

from typing_extensions import List

from cram_viz.knowledge.knowledge_base import get_knowledge_base


@dataclass
class Preset:
    """
    One ready-made EQL query offered by the EQL panel.
    """

    text: str
    """
    Human-readable label shown in the presets list.
    """

    code: str
    """
    EQL source the panel runs when this preset is picked.
    """


#: static presets for the architecture side of the graph
ARCH_PRESETS: Tuple[Preset, ...] = (
    Preset(
        "CRAM packages by size",
        "set_of(pkg.name, pkg.class_count).ordered_by(pkg.class_count, descending=True)",
    ),
    Preset(
        "all Designator classes",
        "an(entity(cls).where(cls.name.endswith('Designator')))",
    ),
    Preset(
        "where does EQL live?",
        "set_of(cls.name, cls.module).where(in_('entity_query_language', cls.module)).limit(15)",
    ),
    Preset(
        "subclasses of Symbol",
        "an(entity(cls).where(in_('Symbol', cls.bases)))",
    ),
    Preset(
        "inside coraplex",
        "an(entity(sub).where(sub.package == 'coraplex'))",
    ),
)


def get_presets() -> List[Preset]:
    """
    Ready-made queries for the EQL panel.

    Scene presets are generated from the loaded scene, so they stay valid for any
    onboarded robot/environment; the architecture presets are static.
    """
    kb = get_knowledge_base()
    presets = [
        Preset("which robot is this?", "the(entity(rob))"),
        Preset("which arms does it have?", "an(entity(arm))"),
        Preset("each arm and its gripper", "set_of(arm.side, arm.gripper)"),
        Preset("what is in the scene?", "an(entity(obj))"),
        Preset("what gets moved?", "an(entity(ep.picks).where(ep.picks != None))"),
    ]
    first_object = next((entry for entry in kb.objects if entry.kind == "object"), None)
    if first_object:
        presets.append(
            Preset(
                "the %s" % first_object.label.lower(),
                "the(entity(obj).where(obj.name == %s))" % repr(first_object.name),
            )
        )
    manipulation = next((episode for episode in kb.episodes if episode.picks), None)
    if manipulation:
        if manipulation.places_at:
            presets.append(
                Preset(
                    "where does it place them?",
                    "the(entity(ep.places_at).where(ep.name == %s))"
                    % repr(manipulation.name),
                )
            )
        if manipulation.performed_by:
            presets.append(
                Preset(
                    "which arm does '%s'?" % manipulation.name,
                    "the(entity(ep.performed_by).where(ep.name == %s))"
                    % repr(manipulation.name),
                )
            )
    return presets + list(ARCH_PRESETS)
