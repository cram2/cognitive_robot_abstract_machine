"""
Ready-made EQL queries for the EQL panel.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from typing import Tuple

from typing_extensions import TYPE_CHECKING, List, Optional

from cramera.knowledge.eql_session import EqlSession
from cramera.knowledge.query_verbalization import QueryVerbalization
from cramera.knowledge.queryable_knowledge import QueryScope

from cramera.knowledge.detected_events import (
    EVENT_CLASS_SUFFIX,
    EVENT_VARIABLE,
    DetectedEventRecord,
)

if TYPE_CHECKING:
    from cramera.knowledge.query_runner import EqlQueryRunner


WORD_START = re.compile(r"(?<!^)(?=[A-Z])")
"""
Where a word begins inside a camel-case class name, which is where a space belongs when
that name is said out loud.
"""


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

    requires_live: bool = False
    """
    Whether answering this needs a running demo attached to the viewer.

    A bundle declares questions about the demo it was recorded from, which range over
    variables only that demo's live query source offers.
    """

    scope: QueryScope = QueryScope.CURRENT_STATE
    """
    Which of the demo's bodies of knowledge this question is about, and so the heading
    it is offered under.
    """

    verbalization: Optional[QueryVerbalization] = None
    """
    The question read back as English, or None while nothing that knows the query's
    variables has worded it (see :meth:`worded`).
    """

    def worded(self, runner: EqlQueryRunner) -> Preset:
        """
        This preset with its question read back as English by ``runner``.

        A preset whose code the runner cannot build keeps no verbalization: its button
        still shows its label, and running it reports its own error.

        :param runner: The runner whose variables the preset's code ranges over.
        """
        return replace(self, verbalization=runner.verbalize(self.code))

    @classmethod
    def of_scene(cls, scene: Optional[str] = None) -> List[Preset]:
        """
        The ready-made queries the EQL panel offers, worded by the scene's runner.

        The pair is the same for every scene: it asks about the robot the recording
        carries, which any onboarded bundle can answer. A demo's own questions range
        over variables only that demo offers and reach the panel from the live bridge.

        :param scene: Name of the scene whose runner words them, or None for the active
            one.
        """
        return cls._worded_by_scene(list(SCENE_PRESETS), scene)

    @classmethod
    def _worded_by_scene(
        cls, presets: List[Preset], scene: Optional[str]
    ) -> List[Preset]:
        """
        The presets with their questions read back as English by the scene's runner.

        A bundle-declared question ranges over a demo's own variables, which the
        recorded scene does not know; it is worded by the live bridge instead and stays
        unworded here.

        :param presets: The presets to word.
        :param scene: Name of the scene whose runner words them, or None for the active
            one.
        """
        runner = EqlSession.of_scene(scene).runner()
        return [preset.worded(runner) for preset in presets]


@dataclass(frozen=True)
class PresetsPerType:
    """
    The same question asked once per type a record can be, e.g. "give me all pick up
    events" for every kind of event a demo detects.

    Written out rather than shown as buttons: a panel has room for the question, not for
    one button per type it can name.
    """

    class_suffix: str
    """
    The word every one of these types' class names ends in, e.g. ``"Event"``.
    """

    class_names: Tuple[str, ...]
    """
    The class name of every type a question may name.
    """

    code: str
    """
    The query answering the question, with ``%s`` for the type's class name.
    """

    scope: QueryScope = QueryScope.CURRENT_STATE
    """
    Which body of knowledge these questions are about.
    """

    def questions(self) -> List[Preset]:
        """
        One question per type, worded the way it is asked out loud.
        """
        return [
            Preset(
                "give me all %s %s" % (self._spoken_type(name), self._plural_noun),
                self.code % name,
                scope=self.scope,
            )
            for name in self.class_names
        ]

    @property
    def _plural_noun(self) -> str:
        """
        What several of these records are called out loud, e.g. ``"events"``.
        """
        return self.class_suffix.lower() + "s"

    def _spoken_type(self, class_name: str) -> str:
        """
        One type as a question names it: ``"PickUpEvent"`` becomes ``"pick up"``.

        :param class_name: The type's class name.
        """
        return WORD_START.sub(" ", class_name[: -len(self.class_suffix)]).lower()


@dataclass(frozen=True)
class DetectedEventQuestions:
    """
    The questions asked about detected events, for whatever set of types is on offer.

    A running demo offers every type its detectors can produce; a recorded scene offers
    the types it actually detected, so it never asks about a moment it cannot answer
    with.
    """

    event_types: Tuple[str, ...]
    """
    The class name of every event type a question may name.
    """

    @classmethod
    def of_records(cls, events: List[DetectedEventRecord]) -> DetectedEventQuestions:
        """
        The questions a set of detections can answer, one per type among them.

        :param events: The detections on offer.
        """
        return cls(tuple(sorted({record.event_type for record in events})))

    def listed(self) -> List[Preset]:
        """
        The questions the panel offers as buttons, or none when nothing was detected.
        """
        if not self.event_types:
            return []
        return [
            Preset(
                "what was detected, and when?",
                "set_of(%s.name, %s.event_type, %s.timestamp)"
                % (EVENT_VARIABLE, EVENT_VARIABLE, EVENT_VARIABLE),
                scope=QueryScope.DETECTED_EVENTS,
            )
        ]

    def unlisted(self) -> List[Preset]:
        """
        "Give me all pick up events", written out once per type -- more questions than a
        panel has room to show as buttons.
        """
        return PresetsPerType(
            class_suffix=EVENT_CLASS_SUFFIX,
            class_names=self.event_types,
            code="an(entity(%s).where(%s.event_type == '%%s'))"
            % (EVENT_VARIABLE, EVENT_VARIABLE),
            scope=QueryScope.DETECTED_EVENTS,
        ).questions()


SCENE_PRESETS: Tuple[Preset, ...] = (
    Preset("which robot is this?", "the(entity(robot))"),
    Preset("which arm does it have?", "an(entity(arm))"),
)
"""
The questions the EQL panel offers for every scene.
"""
