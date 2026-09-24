"""
Tests for recognizing which ready-made query a natural-language question asks.

A spoken question rarely repeats a preset's wording verbatim; the matcher has to
recognize honest paraphrases, and just as importantly refuse questions nothing on offer
answers, so the panel can say so instead of running the wrong query.
"""

from dataclasses import replace

import pytest

krrood = pytest.importorskip("krrood", reason="EQL requires krrood")

from cramera.knowledge.presets import Preset  # noqa: E402
from cramera.knowledge.query_verbalization import QueryVerbalization  # noqa: E402
from cramera.knowledge.question_matching import (  # noqa: E402
    MINIMUM_SIMILARITY,
    UNMATCHED_QUESTION_REPLY,
    QuestionMatcher,
    QuestionMatchResult,
)
from cramera.knowledge.queryable_knowledge import QueryScope  # noqa: E402

ROBOT_PRESET = Preset("which robot is this?", "the(entity(robot))")
SCENE_PRESET = Preset("what is in the scene?", "an(entity(scene_object))")
MOVED_PRESET = Preset("what gets moved?", "an(entity(episode.picks))")

WORDED_PRESET = Preset(
    "detections",
    "an(entity(event))",
    verbalization=QueryVerbalization(
        text="An event where the event was detected by the camera.",
        html="<span>An event where the event was detected by the camera.</span>",
    ),
)


def make_matcher() -> QuestionMatcher:
    """
    A matcher over three plainly-labelled presets.
    """
    return QuestionMatcher([ROBOT_PRESET, SCENE_PRESET, MOVED_PRESET])


# %% recognizing the asked question
class TestRecognizingAQuestion:
    """Exact wording and natural-language paraphrases resolve to their own presets."""

    def test_exact_collection_wording_beats_a_broader_fuzzy_match(self) -> None:
        """A collection's exact wording takes precedence over its broader base name."""
        broad = Preset("Find the degrees_of_freedom of a World", ROBOT_PRESET.code)
        specific = Preset(
            "Find the active_degrees_of_freedom of a World", ROBOT_PRESET.code
        )

        result = QuestionMatcher([broad, specific]).match(specific.text)

        assert result.preset is specific

    def test_the_exact_wording_is_recognized(self):
        result = make_matcher().match("which robot is this?")

        assert result.matched
        assert result.preset == ROBOT_PRESET

    def test_an_honest_paraphrase_is_recognized(self):
        result = make_matcher().match("can you tell me which robot this is")

        assert result.preset == ROBOT_PRESET

    def test_case_and_punctuation_do_not_matter(self):
        result = make_matcher().match("WHAT IS IN THE SCENE")

        assert result.preset == SCENE_PRESET

    def test_the_most_similar_preset_wins(self):
        result = make_matcher().match("what gets moved around")

        assert result.preset == MOVED_PRESET

    def test_the_verbalized_wording_recognizes_what_the_label_alone_would_not(self):
        """
        A preset's label can be terse; the verbalization of its query is a full
        sentence, and being recognized by either is being recognized.
        """
        matcher = QuestionMatcher([WORDED_PRESET])

        result = matcher.match("what was detected by the camera")

        assert result.matched
        assert result.preset == WORDED_PRESET

    @pytest.mark.parametrize(
        "question",
        [WORDED_PRESET.verbalization.text, "what was detected by the camera"],
    )
    def test_verbalization_is_matched_when_the_label_is_absent(
        self, question: str
    ) -> None:
        """
        A preset with no label still matches its native wording and paraphrases.

        :param question: An exact or paraphrased question about the verbalized query.
        """
        preset = replace(WORDED_PRESET, text=None)

        result = QuestionMatcher([preset]).match(question)

        assert result.preset is preset

    def test_an_unworded_preset_does_not_prevent_matching_other_presets(self) -> None:
        """
        Missing wording leaves the other presets available for fuzzy matching.
        """
        matcher = QuestionMatcher([Preset(None, ROBOT_PRESET.code), WORDED_PRESET])

        result = matcher.match("what was detected by the camera")

        assert result.preset is WORDED_PRESET


# %% refusing what nothing answers
class TestRefusingAQuestion:
    @pytest.mark.parametrize("minimum_similarity", [0.0, MINIMUM_SIMILARITY])
    def test_a_preset_without_a_label_or_verbalization_matches_nothing(
        self, minimum_similarity: float
    ) -> None:
        """
        Query source is not a wording alternative for an unworded preset.

        :param minimum_similarity: The threshold for recognizing a wording.
        """
        preset = Preset(None, ROBOT_PRESET.code)

        result = QuestionMatcher([preset], minimum_similarity).match(preset.code)

        assert result.preset is None
        assert result.similarity == 0.0

    def test_an_unrelated_question_is_not_matched(self):
        result = make_matcher().match("what's the weather like today")

        assert not result.matched
        assert result.preset is None

    def test_a_refusal_still_reports_how_close_the_closest_came(self):
        result = make_matcher().match("what's the weather like today")

        assert 0.0 < result.similarity < MINIMUM_SIMILARITY

    def test_no_presets_on_offer_matches_nothing(self):
        result = QuestionMatcher([]).match("which robot is this?")

        assert not result.matched
        assert result.similarity == 0.0

    def test_the_minimum_similarity_is_the_deciding_line(self):
        """
        The same question flips from refused to recognized as the matcher's own
        threshold moves across its similarity, so the threshold — not some second rule —
        is what decides.
        """
        text = "can you tell me which robot this is"
        similarity = make_matcher().match(text).similarity

        assert (
            QuestionMatcher([ROBOT_PRESET], minimum_similarity=similarity + 1.0)
            .match(text)
            .preset
            is None
        )
        assert (
            QuestionMatcher([ROBOT_PRESET], minimum_similarity=similarity)
            .match(text)
            .preset
            == ROBOT_PRESET
        )


# %% the payload the panel reads
class TestMatchPayloads:
    def test_a_match_travels_as_the_presets_own_payload(self):
        result = make_matcher().match("which robot is this?")

        payload = result.to_payload()

        assert payload["ok"] is True
        assert payload["matched"] is True
        assert payload["similarity"] == result.similarity
        assert payload["preset"]["code"] == ROBOT_PRESET.code
        assert payload["preset"]["scope"] == QueryScope.CURRENT_STATE

    def test_a_refusal_travels_as_the_sorry_reply(self):
        payload = make_matcher().match("what's the weather like today").to_payload()

        assert payload["ok"] is True
        assert payload["matched"] is False
        assert payload["reply"] == UNMATCHED_QUESTION_REPLY
        assert "preset" not in payload

    def test_the_result_defaults_to_the_module_threshold(self):
        assert QuestionMatcher([]).minimum_similarity == MINIMUM_SIMILARITY

    def test_matched_is_the_presets_presence(self):
        assert QuestionMatchResult(preset=ROBOT_PRESET, similarity=90.0).matched
        assert not QuestionMatchResult(preset=None, similarity=10.0).matched


# %% one question asked within another's words
class TestAShorterWordingWinsATie:
    def test_a_question_whose_words_sit_inside_anothers_is_the_one_recognized(self):
        """
        "give me all pick up actions" is word for word inside "give me all move and pick
        up actions", so both score a perfect word overlap; the shorter wording is the
        more specific one and has to win the tie.
        """
        exact = Preset("give me all pick up actions", "the pick up actions")
        containing = Preset(
            "give me all move and pick up actions", "the move and pick up actions"
        )
        matcher = QuestionMatcher([containing, exact])

        result = matcher.match("give me all pick up actions")

        assert result.preset == exact

    def test_a_longer_question_still_names_the_shorter_wording_it_contains(self):
        """
        The tie-break prefers the more specific wording only; asking with extra polite
        framing around it is still the same question.
        """
        exact = Preset("give me all pick up actions", "the pick up actions")
        matcher = QuestionMatcher(
            [Preset("give me all move and pick up actions", "the other"), exact]
        )

        result = matcher.match("please give me all pick up actions now")

        assert result.preset == exact


# %% questions that differ only in the words that matter

EVENT_TYPE_PRESETS = [
    Preset("give me all pick up events", "an(entity(event))"),
    Preset("give me all placing events", "an(entity(event))"),
    Preset("give me all support events", "an(entity(event))"),
    Preset("give me all contact events", "an(entity(event))"),
]
"""
One question per type of event, as a recording offers them: the same sentence except for
the words naming the type.
"""


def event_matcher() -> QuestionMatcher:
    """
    A matcher over the questions a recording offers about its detected events, plus the
    plainly-labelled ones a scene offers besides them.
    """
    return QuestionMatcher(
        EVENT_TYPE_PRESETS + [ROBOT_PRESET, SCENE_PRESET, MOVED_PRESET]
    )


class TestQuestionsSharingTheirWording:
    """
    Questions offered one per type share every word but the type's own, so the words that
    distinguish them are the few that have to decide the match.
    """

    def test_the_named_type_decides_which_question_is_recognized(self):
        result = event_matcher().match("show me all the pick up events")

        assert result.preset is EVENT_TYPE_PRESETS[0]

    def test_a_type_written_as_one_word_is_still_that_type(self):
        result = event_matcher().match("show me the pickup events")

        assert result.preset is EVENT_TYPE_PRESETS[0]

    def test_naming_no_type_matches_none_of_them(self):
        result = event_matcher().match("show me the events")

        assert result.preset not in EVENT_TYPE_PRESETS

    def test_a_question_about_the_weather_is_still_refused(self):
        result = event_matcher().match("what is the weather like today")

        assert not result.matched

    def test_the_type_that_was_asked_for_beats_the_others_clearly(self):
        matcher = event_matcher()

        asked = matcher.match("give me all support events")

        assert asked.preset is EVENT_TYPE_PRESETS[2]
