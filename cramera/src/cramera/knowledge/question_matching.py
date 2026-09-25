"""
Recognizing which ready-made query a natural-language question is asking.

A spoken (or typed) question is free text, and the panel can only run the presets it
has. The matcher compares the text against each preset's wording — the verbalization of
its query and the label on its button — and either names the preset worth running or
says that nothing on offer answers the question.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from functools import cached_property

from rapidfuzz import utils
from typing_extensions import Any, Dict, FrozenSet, List, Optional, Tuple

from cramera.knowledge.presets import Preset
from cramera.payload import CrameraPayload

MINIMUM_SIMILARITY = 70.0
"""
Similarity (0–100) below which no preset counts as being what the question asked.

Scored by how much of a wording's meaning the question covers, where a word counts for
as much as it tells the questions apart (see :meth:`QuestionMatcher.match`): a question
naming what a wording is about scores in the high 80s and above, while one that only
shares the framing the wordings have in common stays well below.
"""

UNMATCHED_QUESTION_REPLY = "Sorry, I cannot answer that question."
"""
The reply shown for a question no ready-made query matches.
"""


def words_of(text: str) -> List[str]:
    """
    The words a question or a wording is read as, in the order they were said.

    :param text: The question or wording to read.
    """
    return utils.default_process(text).split()


def spellings_of(text: str) -> FrozenSet[str]:
    """
    Every word the text offers to be recognized by: what it says, plus each pair of
    adjacent words glued together, so a question may write as one word what a wording
    writes as two -- "pickup" for "pick up".

    :param text: The question to read.
    """
    words = words_of(text)
    return frozenset(words) | {
        first + second for first, second in zip(words, words[1:])
    }


@dataclass(kw_only=True)
class QuestionMatchResult(CrameraPayload):
    """
    The outcome of asking a natural-language question: the preset recognized as that
    question, or the reply that nothing on offer answers it.
    """

    preset: Optional[Preset]
    """
    The ready-made query recognized as the asked question, or None when none was similar
    enough.
    """

    similarity: float
    """
    Similarity (0–100) between the asked question and the closest preset's wording.
    """

    @property
    def matched(self) -> bool:
        """
        Whether a preset was recognized as the asked question.
        """
        return self.preset is not None

    def to_payload(self) -> Dict[str, Any]:
        """
        The JSON-serializable shape the panel's voice consumer reads.
        """
        if self.preset is None:
            return {
                "ok": self.ok,
                "matched": False,
                "similarity": self.similarity,
                "reply": UNMATCHED_QUESTION_REPLY,
            }
        return {
            "ok": self.ok,
            "matched": True,
            "similarity": self.similarity,
            "preset": asdict(self.preset),
        }


@dataclass
class QuestionMatcher:
    """
    Recognizes which of the presets on offer a natural-language question is asking.
    """

    presets: List[Preset]
    """
    The ready-made queries on offer, each worded as well as its source knows how.
    """

    minimum_similarity: float = MINIMUM_SIMILARITY
    """
    Similarity (0–100) below which a question counts as unanswerable.
    """

    def match(self, text: str) -> QuestionMatchResult:
        """
        The preset most similar to the asked question, or the no-match outcome.

        A wording is scored by how much of it the question covers, and each of its words
        counts for as much as it tells the wordings apart: the framing that every
        question on offer shares ("give me all ... events") decides almost nothing, while
        the words only one of them uses decide almost everything.

        :param text: The question as asked, in natural language.
        """
        asked = spellings_of(text)
        best: Optional[Preset] = None
        best_pair = (0.0, -1.0)
        for preset in self.presets:
            pair = self._comparison(asked, text, preset)
            if pair > best_pair:
                best, best_pair = preset, pair
        best_similarity = best_pair[0]
        if best is None or best_similarity < self.minimum_similarity:
            return QuestionMatchResult(preset=None, similarity=best_similarity)
        return QuestionMatchResult(preset=best, similarity=best_similarity)

    @cached_property
    def _weights(self) -> Dict[str, float]:
        """
        What each word is worth: one over the share of the wordings that use it, so a
        word half of them use is worth half as much as one only a quarter use.

        The framing the questions on offer have in common ("give me all ... events") is
        thereby worth little, and the words only one of them uses are worth the most.
        """
        wordings = [set(words_of(wording)) for wording in self._wordings()]
        weights: Dict[str, float] = {}
        for wording in wordings:
            for word in wording:
                weights[word] = weights.get(word, 0.0) + 1.0
        return {word: len(wordings) / used_by for word, used_by in weights.items()}

    @cached_property
    def _unheard_weight(self) -> float:
        """
        What a word no wording uses is worth: more than any word they do, because a
        question saying it is asking about something none of them is about.

        Worth as much as a word half a wording would use, which is the rarest a word can
        be without being unheard of.
        """
        return 2.0 * len(self._wordings())

    def _weight(self, word: str) -> float:
        """
        What one word is worth when it decides which wording is meant.

        :param word: The word being weighed.
        """
        return self._weights.get(word, self._unheard_weight)

    def _wordings(self) -> List[str]:
        """
        Every way the presets on offer put their questions.
        """
        return [
            wording for preset in self.presets for wording in self._wordings_of(preset)
        ]

    @staticmethod
    def _wordings_of(preset: Preset) -> List[str]:
        """
        The ways one preset puts its question: the label on its button, and its query
        read back as English where the source worded it.

        :param preset: The preset to read.
        """
        if preset.verbalization is None:
            return [preset.text]
        return [preset.text, preset.verbalization.text]

    def _comparison(
        self, asked: FrozenSet[str], text: str, preset: Preset
    ) -> Tuple[float, float]:
        """
        How well the asked question names one preset: its best score, and how many of
        the words scoring it were its own.

        A preset is worded twice -- its query's verbalization and the label on its
        button -- and being recognized by either is being recognized.

        Word order and polite framing around the words that matter do not count against
        a question. Covering a wording's words alone ties a question with every wording
        that contains them, e.g. "give me all pick up actions" inside "give me all move
        and pick up actions", so the score is paired with the count of the wording's own
        words and a match prefers the wording that added the fewest: the more specific
        one.

        :param asked: The words of the question.
        :param text: The question as asked, for counting its words.
        :param preset: The preset whose wordings the question is compared against.
        :return: The score, and the negated count of words the best wording added
            beyond the question's own, so a higher pair means a better, more specific
            match.
        """
        said = words_of(text)
        best = (0.0, 0.0)
        for wording in self._wordings_of(preset):
            score = self._agreement(asked, said, wording)
            own_words = len(words_of(wording))
            pair = (score, -float(own_words - len(said)))
            if pair > best:
                best = pair
        return best

    def _agreement(self, asked: FrozenSet[str], said: List[str], wording: str) -> float:
        """
        How far a question and one wording say the same thing, 0-100.

        Read in both directions, and the closer one counts: a question may frame the
        words that matter politely, and it may put in three words what a wording spells
        out in a sentence. What it may not do is leave out the words that say which
        wording is meant, since those are the ones carrying the weight.

        :param asked: Every spelling the question offers.
        :param said: The words of the question, in order.
        :param wording: The wording to compare against.
        """
        return max(
            self._coverage(asked, words_of(wording)),
            self._coverage(spellings_of(wording), said),
        )

    def _coverage(self, asked: FrozenSet[str], wording: List[str]) -> float:
        """
        How much of one side's words the other one said, 0-100, weighing each word by
        what it says about which wording is meant.

        :param asked: Every spelling the covering side offers.
        :param wording: The words being covered, in order.
        """
        total = sum(self._weight(word) for word in set(wording))
        if not total:
            return 0.0
        covered = self._covered_words(asked, wording)
        return 100.0 * sum(self._weight(word) for word in covered) / total

    @staticmethod
    def _covered_words(asked: FrozenSet[str], wording: List[str]) -> FrozenSet[str]:
        """
        The words of a wording the question said, counting a pair the question wrote as
        one word as both of them.

        :param asked: Every spelling the question offers.
        :param wording: The words of the wording, in order.
        """
        covered = {word for word in wording if word in asked}
        for first, second in zip(wording, wording[1:]):
            if first + second in asked:
                covered |= {first, second}
        return frozenset(covered)
