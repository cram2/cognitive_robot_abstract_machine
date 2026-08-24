#!/usr/bin/env python3
"""
Record a plan dashboard's published Artifact URL into the dashboard-URL cache.

The caller names the cache key and the title the dashboard is published under; the URL
is looked up in the account's Artifact listing and that URL is written, so no URL the
listing does not contain can be recorded. When several artifacts share the title, the
most recently updated one wins; ``--url`` overrides that choice and must name one of
them.

Usage:
    python3 record_dashboard_url.py \\
        --key <plan-id|_index> \\
        --expected-title "<the plan's title, or the index's own title>" \\
        --listing /tmp/artifact_listing.json \\
        --cache /tmp/dashboard-urls.yaml \\
        --output /tmp/updated-dashboard-urls.yaml \\
        [--url <one of the same-titled candidates>]

artifact_listing.json shape - one object per artifact the account owns, transcribed
from the Artifact tool's ``action: "list"`` output:
    [{"title": "...", "url": "...", "updated": "YYYY-MM-DD"}, ...]

The cache is patched a line at a time rather than round-tripped through a YAML dump, so
the header comment, key order and spacing survive and the diff shows only the entry that
moved.

Prints a one-line JSON summary to stdout:
    {"key": ..., "url": ..., "previous_url": ..., "changed": true|false}
``changed`` is false when the cache already named that artifact, in which case there is
nothing to push.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Any

ARTIFACT_URL_PATTERN = re.compile(
    r"^https://claude\.ai/code/artifact/"
    r"[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}$"
)
"""
The shape of a published Artifact URL, used to reject anything else on sight.
"""

# %% refusals


@dataclass
class DashboardUrlError(Exception, ABC):
    """
    Base class for every refusal to record a dashboard URL.

    Subclasses implement :meth:`error_message` and :meth:`suggest_correction`; both are
    evaluated at construction and composed into the exception message, with a non-empty
    correction rendered as a trailing ``Suggestion:`` line. Mirrors krrood's
    ``DataclassException`` idiom rather than importing it, since this module must stay
    importable with nothing but the standard library.
    """

    def __post_init__(self) -> None:
        """
        Compose the exception message from the subclass's two descriptions.
        """
        if getattr(type(self), "__abstractmethods__", None):
            raise TypeError(
                f"Can't instantiate abstract class {type(self).__name__} without an "
                f"implementation of "
                f"{', '.join(sorted(type(self).__abstractmethods__))}."
            )
        message = self.error_message()
        correction = self.suggest_correction()
        if correction:
            message = f"{message}\nSuggestion: {correction}"
        super().__init__(message)

    def __str__(self) -> str:
        """
        Render the composed message rather than a repr of the dataclass fields.
        """
        return Exception.__str__(self)

    @abstractmethod
    def error_message(self) -> str:
        """
        :return: A human-readable description of what went wrong.
        """

    @abstractmethod
    def suggest_correction(self) -> str:
        """
        :return: Advice on how to fix the error, or an empty string if there is none.
        """


@dataclass
class MalformedArtifactUrlError(DashboardUrlError):
    """
    Raised when a supplied URL is not shaped like a published Artifact URL.
    """

    supplied_url: str
    """
    The URL the caller passed.
    """

    def error_message(self) -> str:
        return f"{self.supplied_url!r} is not a published Artifact URL."

    def suggest_correction(self) -> str:
        return (
            "Pass a URL of the form https://claude.ai/code/artifact/<uuid>, copied from "
            "the Artifact tool's listing."
        )


@dataclass
class ArtifactNotPublishedError(DashboardUrlError):
    """
    Raised when no artifact in the listing carries the expected title.
    """

    expected_title: str
    """
    The title the dashboard was expected to be published under.
    """

    def error_message(self) -> str:
        return f"No artifact in the listing is titled {self.expected_title!r}."

    def suggest_correction(self) -> str:
        return (
            "Publish the dashboard before recording its URL, and check the title "
            "matches the plan's own title exactly."
        )


@dataclass
class UnlistedArtifactUrlError(DashboardUrlError):
    """
    Raised when a supplied URL is not among the artifacts carrying the expected title -
    either invented, or belonging to some other plan.
    """

    supplied_url: str
    """
    The URL the caller chose.
    """

    expected_title: str
    """
    The title whose artifacts were the only permitted choices.
    """

    candidate_urls: list[str]
    """
    Every artifact URL carrying that title.
    """

    def error_message(self) -> str:
        return (
            f"{self.supplied_url} is not among the artifacts titled "
            f"{self.expected_title!r}."
        )

    def suggest_correction(self) -> str:
        return f"Choose one of: {', '.join(self.candidate_urls)}."


# %% the artifact listing


@dataclass(frozen=True)
class ArtifactListingEntry:
    """
    One artifact the account owns, as reported by the Artifact tool's listing.
    """

    title: str
    """
    The artifact's title, matched against a plan's own ``title``.
    """

    url: str
    """
    The artifact's published URL.
    """

    updated: str
    """
    The date the artifact was last updated, as an ISO ``YYYY-MM-DD`` string, which is
    what separates same-titled artifacts.
    """


def load_artifact_listing(
    raw_listing: list[dict[str, str]],
) -> list[ArtifactListingEntry]:
    """
    Build typed listing entries from the parsed artifact-listing JSON.

    :param raw_listing: One mapping per artifact, each with ``title``, ``url`` and
        ``updated``.
    :return: The same artifacts as typed entries, in listing order.
    """
    return [
        ArtifactListingEntry(
            title=entry["title"], url=entry["url"], updated=entry["updated"]
        )
        for entry in raw_listing
    ]


def resolve_artifact_url(
    listing: list[ArtifactListingEntry], expected_title: str, chosen_url: str | None
) -> str:
    """
    Resolve which published artifact a cache key should point at.

    Same-titled artifacts are ranked by ``updated``, most recent first, and ties keep
    listing order - so the freshest page wins without anyone being asked.

    :param listing: Every artifact the account owns.
    :param expected_title: The title the dashboard is published under.
    :param chosen_url: An explicit override, which must carry ``expected_title``.
    :raises MalformedArtifactUrlError: If ``chosen_url`` is not an Artifact URL.
    :raises ArtifactNotPublishedError: If no artifact carries ``expected_title``.
    :raises UnlistedArtifactUrlError: If ``chosen_url`` carries a different title.
    :return: The URL to record.
    """
    if chosen_url is not None and not ARTIFACT_URL_PATTERN.match(chosen_url):
        raise MalformedArtifactUrlError(supplied_url=chosen_url)

    candidates = [entry for entry in listing if entry.title == expected_title]
    if not candidates:
        raise ArtifactNotPublishedError(expected_title=expected_title)

    candidate_urls = [entry.url for entry in candidates]
    if chosen_url is None:
        return most_recently_updated(candidates).url

    if chosen_url not in candidate_urls:
        raise UnlistedArtifactUrlError(
            supplied_url=chosen_url,
            expected_title=expected_title,
            candidate_urls=candidate_urls,
        )
    return chosen_url


def most_recently_updated(
    candidates: list[ArtifactListingEntry],
) -> ArtifactListingEntry:
    """
    Pick the freshest of several same-titled artifacts.

    :param candidates: Artifacts sharing a title, in listing order.
    :return: The one updated most recently; among equal dates, the earliest listed,
        since the Artifact listing itself is ordered most-recent-first.
    """
    return max(candidates, key=lambda entry: entry.updated)


# %% patching the cache


@dataclass(frozen=True)
class CacheEntryLine:
    """
    An existing key's line in the cache text.
    """

    index: int
    """
    The line's position in the cache text.
    """

    prefix: str
    """
    Everything up to and including the ``key:`` separator, preserved verbatim.
    """

    url: str
    """
    The URL the key currently holds.
    """


@dataclass(frozen=True)
class PatchedCache:
    """
    The cache text after writing one key, and what that key held beforehand.
    """

    text: str
    """
    The full cache text with the one entry written.
    """

    previous_url: str | None
    """
    What the key held before, or ``None`` if it had no entry.
    """


def find_cache_entry_line(lines: list[str], key: str) -> CacheEntryLine | None:
    """
    Locate ``key``'s existing line in the cache.

    :param lines: The cache text split into lines.
    :param key: The cache key to look for.
    :return: The matching line, or ``None`` if the key has no entry yet.
    """
    pattern = re.compile(rf"^({re.escape(key)}:\s*)(\S+)\s*$")
    for index, line in enumerate(lines):
        match = pattern.match(line)
        if match is not None:
            return CacheEntryLine(
                index=index, prefix=match.group(1), url=match.group(2)
            )
    return None


def append_position(lines: list[str]) -> int:
    """
    Find where a new entry belongs: after the last entry, before any trailing blank
    lines, so the file keeps its single trailing newline.

    :param lines: The cache text split into lines.
    :return: The index to insert a new entry at.
    """
    position = len(lines)
    while position > 0 and lines[position - 1] == "":
        position -= 1
    return position


def apply_url_record(cache_text: str, key: str, url: str) -> PatchedCache:
    """
    Write ``url`` as ``key``'s entry, rewriting its line or appending a new one.

    :param cache_text: The dashboard-urls.yaml file's raw text.
    :param key: The cache key to write.
    :param url: The URL to record for it.
    :return: The patched text and whatever the key held beforehand.
    """
    lines = cache_text.split("\n")
    existing = find_cache_entry_line(lines, key)
    if existing is None:
        position = append_position(lines)
        appended = lines[:position] + [f"{key}: {url}"] + lines[position:]
        return PatchedCache(text="\n".join(appended), previous_url=None)

    lines[existing.index] = f"{existing.prefix}{url}"
    return PatchedCache(text="\n".join(lines), previous_url=existing.url)


# %% the recorded outcome


@dataclass
class UrlRecord:
    """
    The outcome of recording one key's URL.
    """

    key: str
    """
    The cache key written - a plan id, or ``_index`` for the master index.
    """

    url: str
    """
    The URL now recorded for that key.
    """

    previous_url: str | None
    """
    What the key held beforehand, or ``None`` if it had no entry.
    """

    @property
    def changed(self) -> bool:
        """
        Whether the cache moved, and so needs pushing back.
        """
        return self.previous_url != self.url

    def to_json_dict(self) -> dict[str, Any]:
        """
        Render to the plain-dict shape the calling skill expects.
        """
        return {
            "key": self.key,
            "url": self.url,
            "previous_url": self.previous_url,
            "changed": self.changed,
        }


def main() -> int:
    """
    Parse arguments, resolve the URL, patch the cache, and print the summary.

    See the module docstring for the CLI contract.
    """
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "--key", required=True, help="Cache key: a plan id, or _index for the index"
    )
    parser.add_argument(
        "--expected-title",
        required=True,
        help="The title the dashboard is published under",
    )
    parser.add_argument(
        "--listing",
        required=True,
        help='Path to a JSON file: [{"title": ..., "url": ..., "updated": ...}, ...]',
    )
    parser.add_argument("--cache", required=True, help="Path to dashboard-urls.yaml")
    parser.add_argument(
        "--output", required=True, help="Path to write the updated cache to"
    )
    parser.add_argument(
        "--url",
        default=None,
        help="Override which same-titled artifact to keep, instead of the freshest",
    )
    arguments = parser.parse_args()

    listing = load_artifact_listing(json.loads(Path(arguments.listing).read_text()))
    try:
        url = resolve_artifact_url(listing, arguments.expected_title, arguments.url)
    except DashboardUrlError as error:
        print(f"refusing to record a dashboard URL: {error}", file=sys.stderr)
        return 1

    patched = apply_url_record(Path(arguments.cache).read_text(), arguments.key, url)
    Path(arguments.output).write_text(patched.text)

    record = UrlRecord(key=arguments.key, url=url, previous_url=patched.previous_url)
    print(json.dumps(record.to_json_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
