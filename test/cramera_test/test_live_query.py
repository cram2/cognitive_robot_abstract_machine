"""
Tests for querying a running demo through the live bridge.
"""

from __future__ import annotations

import threading
from dataclasses import dataclass, field

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from semantic_digital_twin.spatial_types import Point3  # noqa: E402
from typing_extensions import Any, List  # noqa: E402

from cramera.knowledge.presets import Preset  # noqa: E402
from cramera.knowledge.query_domain import QueryDomain  # noqa: E402
from cramera.knowledge.query_runner import EqlQueryRunner  # noqa: E402
from cramera.knowledge.queryable_knowledge import (  # noqa: E402
    QueryableKnowledge,
    QueryEvaluation,
    QueryScope,
    UnknownQueryScope,
)
from cramera.live.bridge import Bridge  # noqa: E402
from cramera.live.query import LiveQuerySource, NoQuerySourceRegistered  # noqa: E402

from .dataset.queryable_records import NamedRecord  # noqa: E402


@dataclass(frozen=True)
class AnswersFromStorage(QueryEvaluation):
    """
    An evaluation standing in for one that reads a database instead of this process.
    """

    stored: List[NamedRecord] = field(default_factory=list)

    def evaluate(self, expression: Any) -> Any:
        return list(self.stored)


@dataclass
class GrowingRecordSource(LiveQuerySource):
    """
    A source whose records keep arriving, the way a demo's results do while it runs,
    alongside the ones its finished runs already stored.
    """

    records: List[NamedRecord] = field(default_factory=list)
    """
    What this source's current-state domain ranges over.
    """

    stored: List[NamedRecord] = field(default_factory=list)
    """
    What its episodic-memory domain answers with.
    """

    def title(self) -> str:
        """
        What the panel names this source.
        """
        return "record demo"

    def knowledge(self) -> List[QueryableKnowledge]:
        """
        The two bodies of knowledge this source offers.
        """
        return [
            QueryableKnowledge(
                scope=QueryScope.CURRENT_STATE,
                domains=[QueryDomain("record", NamedRecord, self.records)],
            ),
            QueryableKnowledge(
                scope=QueryScope.EPISODIC_MEMORY,
                domains=[QueryDomain("stored_record", NamedRecord)],
                evaluation=AnswersFromStorage(stored=self.stored),
                extra_names={"ALPHA": "alpha"},
            ),
        ]

    def presets(self) -> List[Preset]:
        """
        The ready-made queries this source offers.
        """
        return [
            Preset("all records", "an(entity(record))"),
            Preset(
                "everything stored",
                "an(entity(stored_record))",
                scope=QueryScope.EPISODIC_MEMORY,
            ),
        ]

    def unlisted_presets(self) -> List[Preset]:
        """
        The one-per-kind question this source recognizes without showing it.
        """
        return [
            Preset("give me all beta samples", "an(entity(record))"),
        ]


@dataclass
class CurrentStateOnlySource(LiveQuerySource):
    """
    A demo that keeps no record of its finished runs, so only its present is queryable.
    """

    def title(self) -> str:
        return "state-only demo"

    def knowledge(self) -> List[QueryableKnowledge]:
        return [
            QueryableKnowledge(
                scope=QueryScope.CURRENT_STATE,
                domains=[QueryDomain("record", NamedRecord, [])],
            )
        ]

    def presets(self) -> List[Preset]:
        return []


def make_record(name: str) -> NamedRecord:
    """
    One record to query, distinguished only by its name.

    :param name: The record's name.
    """
    return NamedRecord(name, "alpha", 1.0, Point3(0.0, 0.0, 0.0))


@pytest.fixture()
def source() -> GrowingRecordSource:
    return GrowingRecordSource(
        records=[make_record("first")], stored=[make_record("last week")]
    )


@pytest.fixture()
def bridge(source) -> Bridge:
    live_bridge = Bridge()
    live_bridge.register_query_source(source)
    return live_bridge


# %% answering from the running process
class TestQueryingARegisteredSource:
    def test_a_query_is_answered_from_the_sources_domain(self, bridge):
        result = bridge.run_query("an(entity(record))")

        assert result.ok
        assert [row["__entity__"] for row in result.rows] == ["first"]

    def test_a_query_sees_records_recorded_after_registration(self, bridge, source):
        """
        The point of querying live: an answer reflects what the demo has done by the
        time the question is asked, not by the time the bridge was wired up.
        """
        source.records.append(make_record("second"))

        result = bridge.run_query("an(entity(record))")

        assert [row["__entity__"] for row in result.rows] == ["first", "second"]

    def test_the_presets_are_the_sources_own(self, bridge):
        assert [preset.text for preset in bridge.query_presets()] == [
            "all records",
            "everything stored",
        ]

    def test_every_preset_of_a_source_runs_in_the_scope_it_declares(self, bridge):
        for preset in bridge.query_presets():
            assert bridge.run_query(preset.code, preset.scope).ok, preset.text


# %% presets worded per scope
class TestWordedLivePresets:
    """
    The bridge hands out each preset with its question read back as English, worded by
    the body of knowledge the preset declares — the source itself stays wording-free.
    """

    def test_each_preset_is_worded_by_the_scope_it_declares(self, bridge):
        current, stored = bridge.query_presets()

        assert current.verbalization == EqlQueryRunner(
            domains=[QueryDomain("record", NamedRecord, [])]
        ).verbalize(current.code)
        assert stored.verbalization == EqlQueryRunner(
            domains=[QueryDomain("stored_record", NamedRecord)]
        ).verbalize(stored.code)
        assert current.verbalization is not None
        assert stored.verbalization is not None

    def test_the_sources_own_presets_stay_unworded(self, source):
        assert all(preset.verbalization is None for preset in source.presets())


# %% recognizing a spoken question
class TestAskedQuestions:
    """
    The bridge recognizes which of the demo's presets a natural-language question is
    asking, so a transcript can run a query as if its button had been clicked.
    """

    def test_a_question_is_recognized_across_scopes(self, bridge):
        result = bridge.match_question("show me everything stored")

        assert result.matched
        assert result.preset.code == "an(entity(stored_record))"
        assert result.preset.scope is QueryScope.EPISODIC_MEMORY

    def test_the_recognized_preset_runs_as_if_clicked(self, bridge):
        result = bridge.match_question("show me all records")

        answer = bridge.run_query(result.preset.code, result.preset.scope)

        assert answer.ok
        assert [row["__entity__"] for row in answer.rows] == ["first"]

    def test_a_question_the_panel_does_not_show_is_recognized_too(self, bridge):
        """
        A source writes out one question per kind of thing it records, which the panel
        has no room to show; asking for one still runs it.
        """
        result = bridge.match_question("give me all beta samples")

        assert result.matched
        assert result.preset.text == "give me all beta samples"

    def test_an_unlisted_question_stays_off_the_buttons(self, bridge):
        assert "give me all beta samples" not in [
            preset.text for preset in bridge.query_presets()
        ]

    def test_an_unlisted_question_is_matched_by_its_label_alone(self, bridge):
        """
        Wording a question means building its query, which is too much work per asked
        question for a family the panel never shows.
        """
        result = bridge.match_question("give me all beta samples")

        assert result.preset.verbalization is None

    def test_a_source_writing_none_out_is_matched_against_its_buttons(self):
        live_bridge = Bridge()
        live_bridge.register_query_source(CurrentStateOnlySource())

        assert not live_bridge.match_question("give me all beta samples").matched

    def test_an_unrelated_question_is_refused(self, bridge):
        result = bridge.match_question("what's the weather like today")

        assert not result.matched

    def test_without_a_source_there_is_nothing_to_match(self):
        with pytest.raises(NoQuerySourceRegistered):
            Bridge().match_question("which robot is this")


# %% two bodies of knowledge, asked apart
class TestQueryingByScope:
    def test_the_current_state_is_answered_from_the_running_demo(self, bridge):
        result = bridge.run_query("an(entity(record))", QueryScope.CURRENT_STATE)

        assert [row["__entity__"] for row in result.rows] == ["first"]

    def test_episodic_memory_is_answered_from_where_it_was_stored(self, bridge):
        result = bridge.run_query(
            "an(entity(stored_record))", QueryScope.EPISODIC_MEMORY
        )

        assert [row["__entity__"] for row in result.rows] == ["last week"]

    def test_a_scope_only_sees_its_own_variables(self, bridge):
        """
        Each scope is its own vocabulary: asking about stored runs in the current-state
        scope is a mistake worth reporting rather than an empty answer.
        """
        with pytest.raises(NameError):
            bridge.run_query("an(entity(stored_record))", QueryScope.CURRENT_STATE)

    def test_a_scope_the_source_does_not_offer_is_refused(self):
        live_bridge = Bridge()
        live_bridge.register_query_source(CurrentStateOnlySource())

        with pytest.raises(UnknownQueryScope):
            live_bridge.run_query("an(entity(record))", QueryScope.EPISODIC_MEMORY)

    def test_a_scope_may_put_further_names_in_reach_of_its_questions(self, bridge):
        """
        A question about stored runs compares against the vocabulary those runs were
        recorded in, which a query has no other way to name.
        """
        result = bridge.run_query(
            "an(entity(stored_record).where(stored_record.category == ALPHA))",
            QueryScope.EPISODIC_MEMORY,
        )

        assert result.ok

    def test_the_scopes_on_offer_are_named_with_their_headings(self, bridge):
        assert bridge.query_scopes() == [
            QueryScope.CURRENT_STATE,
            QueryScope.EPISODIC_MEMORY,
        ]

    def test_status_reports_that_querying_is_available(self, bridge):
        assert bridge.status()["query"] is True

    def test_the_source_titles_the_answers(self, bridge):
        assert bridge.query_title() == "record demo"


# %% no demo to ask
class TestQueryingWithoutASource:
    def test_running_a_query_raises(self):
        with pytest.raises(NoQuerySourceRegistered):
            Bridge().run_query("an(entity(record))")

    def test_asking_for_presets_raises(self):
        with pytest.raises(NoQuerySourceRegistered):
            Bridge().query_presets()

    def test_status_reports_that_querying_is_unavailable(self):
        assert Bridge().status()["query"] is False


# %% concurrent viewers
class TestConcurrentQueries:
    def test_queries_from_several_threads_all_answer_correctly(self, bridge, source):
        """
        Krrood's SymbolGraph singleton is not threadsafe and the bridge serves several
        viewers from its own thread pool, so overlapping queries must not corrupt one
        another's answers.
        """
        source.records.extend(make_record(name) for name in ("second", "third"))
        counts: List[int] = []
        lock = threading.Lock()

        def ask() -> None:
            count = bridge.run_query("an(entity(record))").count
            with lock:
                counts.append(count)

        askers = [threading.Thread(target=ask) for _ in range(8)]
        for asker in askers:
            asker.start()
        for asker in askers:
            asker.join(timeout=30)

        assert counts == [3] * 8
