"""
Contract test over the skill documents that send a session to the setup prerequisite
check.

The rule they share is stated once, in prerequisite-check.md: a skill whose setup check
fails runs ``/setup-personal-notes`` rather than asking whether to. A document that
instructs an offer instead contradicts it, and nothing else catches that - a branch that
forks the instruction into a new document of its own conflicts with no existing file, so
the contradiction arrives silently at merge time.

Filesystem only: no network, no credentials, no scratch repository.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from basstler.locations import PackageLocation

from .constants import SkillDirectory


class SkillCommand(StrEnum):
    """
    A skill, as a session is told to invoke it.
    """

    SETUP_PERSONAL_NOTES = "/setup-personal-notes"
    """
    The one-time setup of the personal-notes tooling.
    """


class OfferingVerb(StrEnum):
    """
    The verbs by which a document makes a command something to ask about rather than
    something to run.
    """

    OFFER = "offer"
    """
    The verb the instruction this rule replaced was written with.
    """

    PROPOSE = "propose"
    """
    A synonym that would reinstate the same question.
    """

    SUGGEST = "suggest"
    """
    A synonym that would reinstate the same question.
    """


# %% the documents under test


@dataclass(frozen=True)
class SkillDocument:
    """
    One markdown document belonging to a skill, read for the instructions it gives.
    """

    path: Path
    """
    Its location on disk.
    """

    text: str
    """
    Its full contents.
    """

    @classmethod
    def read_from(cls, path: Path) -> SkillDocument:
        """
        Read one document.

        :param path: The document's location on disk.
        :return: The document.
        """
        return cls(path=path, text=path.read_text(encoding="utf-8"))

    @property
    def relative_path(self) -> str:
        """
        The path as a reader of this repository would name it.
        """
        return str(self.path.relative_to(PackageLocation.REPOSITORY_ROOT.value))

    @property
    def offer_instructions(self) -> list[str]:
        """
        Every place this document tells a session to offer the setup rather than run it.
        """
        return [match.group(0) for match in self.offer_pattern().finditer(self.text)]

    @staticmethod
    def offer_pattern() -> re.Pattern[str]:
        """
        The instruction shape the no-ask rule replaced: a verb of offering or proposing,
        governing the setup command directly.

        Deliberately narrow - it is aimed at the sentence that was copied from document to
        document (``offer `/setup-personal-notes```), not at every possible paraphrase of
        asking, and it must not fire on prose *about* the rule, which discusses offering
        and asking at length.
        """
        verbs = "|".join(OfferingVerb)
        command = re.escape(SkillCommand.SETUP_PERSONAL_NOTES)
        return re.compile(
            rf"\b(?:{verbs})(?:s|ing)?\s+(?:to\s+run\s+)?`?{command}", re.IGNORECASE
        )


def skill_documents() -> list[SkillDocument]:
    """
    Every markdown document under the skills directory.

    Discovered rather than listed: the point of the sweep is to cover a document that
    does not exist yet, which is exactly what a list cannot do.

    :return: The documents.
    """
    return [
        SkillDocument.read_from(path)
        for path in sorted(
            (PackageLocation.REPOSITORY_ROOT / SkillDirectory.DIRECTORY).rglob("*.md")
        )
    ]


# %% the rule


def test_no_skill_document_offers_the_setup_instead_of_running_it() -> None:
    """
    A document that mentions the setup skill instructs running it, never offering it.
    """
    offenders = {
        document.relative_path: document.offer_instructions
        for document in skill_documents()
        if document.offer_instructions
    }
    assert offenders == {}


def test_the_sweep_reads_the_skill_documents_rather_than_nothing() -> None:
    """
    Guards the rule above against passing vacuously on a moved or renamed directory.

    Asserts only that documents were found, not which: skills are added and their
    documents move between them, and pinning the set here would make an unrelated change
    fail this module.
    """
    assert skill_documents() != []
