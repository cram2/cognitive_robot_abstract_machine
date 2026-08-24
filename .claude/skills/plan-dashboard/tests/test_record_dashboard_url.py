"""
Tests for record_dashboard_url.py - the deterministic dashboard-URL cache write.

The behaviour under test is that only a URL the account's Artifact listing actually
contains can reach the cache, however confidently a caller supplies one.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import yaml

from record_dashboard_url import (
    ArtifactListingEntry,
    ArtifactNotPublishedError,
    DashboardUrlError,
    MalformedArtifactUrlError,
    UnlistedArtifactUrlError,
    append_position,
    apply_url_record,
    find_cache_entry_line,
    load_artifact_listing,
    main,
    most_recently_updated,
    resolve_artifact_url,
)

FIXTURES_DIRECTORY = Path(__file__).parent / "fixtures"
CACHE_TEXT = (FIXTURES_DIRECTORY / "dashboard-urls.yaml").read_text()
LISTING = load_artifact_listing(
    json.loads((FIXTURES_DIRECTORY / "artifact-listing.json").read_text())
)

ALPHA_URL = "https://claude.ai/code/artifact/11111111-1111-4111-8111-111111111111"
STALE_BETA_URL = "https://claude.ai/code/artifact/22222222-2222-4222-8222-222222222222"
INDEX_URL = "https://claude.ai/code/artifact/33333333-3333-4333-8333-333333333333"
BETA_URL = "https://claude.ai/code/artifact/44444444-4444-4444-8444-444444444444"
OLDER_TWIN_URL = "https://claude.ai/code/artifact/55555555-5555-4555-8555-555555555555"
NEWER_TWIN_URL = "https://claude.ai/code/artifact/66666666-6666-4666-8666-666666666666"
FIRST_TIED_URL = "https://claude.ai/code/artifact/77777777-7777-4777-8777-777777777777"
SECOND_TIED_URL = "https://claude.ai/code/artifact/88888888-8888-4888-8888-888888888888"


# %% resolving a url from the listing


def test_url_is_resolved_from_the_listing_by_title() -> None:
    """
    A uniquely-titled artifact resolves without the caller naming any URL.
    """
    assert resolve_artifact_url(LISTING, "Beta plan title", None) == BETA_URL


def test_listing_entries_carry_title_url_and_updated_date() -> None:
    """
    The listing loads into typed entries rather than raw mappings.
    """
    assert (
        ArtifactListingEntry(
            title="Beta plan title", url=BETA_URL, updated="2026-08-02"
        )
        in LISTING
    )


# %% a fabricated url cannot enter the cache


def test_url_absent_from_the_listing_is_refused() -> None:
    """
    The regression this script exists for: a plausible but unpublished URL is rejected
    instead of being written to the cache.
    """
    fabricated = "https://claude.ai/code/artifact/69b0dd3c-10cf-414b-8572-f389dbf21825"
    with pytest.raises(UnlistedArtifactUrlError):
        resolve_artifact_url(LISTING, "Beta plan title", fabricated)


def test_url_that_is_not_an_artifact_url_is_refused() -> None:
    """
    A URL of the wrong shape fails on shape, before any listing lookup.
    """
    with pytest.raises(MalformedArtifactUrlError):
        resolve_artifact_url(LISTING, "Beta plan title", "https://example.com/page")


def test_title_absent_from_the_listing_is_refused() -> None:
    """
    A plan whose dashboard was never published has nothing to record.
    """
    with pytest.raises(ArtifactNotPublishedError):
        resolve_artifact_url(LISTING, "Never published title", None)


# %% refusals describe themselves


def test_a_refusal_composes_its_message_and_suggestion() -> None:
    """
    Both halves of a refusal reach the exception message, the suggestion last.
    """
    error = ArtifactNotPublishedError(expected_title="Never published title")
    assert (
        str(error)
        == f"{error.error_message()}\nSuggestion: {error.suggest_correction()}"
    )


def test_a_refusal_carries_its_context_as_typed_fields() -> None:
    """
    What a refusal is about is readable off the exception rather than parsed back out of
    its wording.
    """
    error = UnlistedArtifactUrlError(
        supplied_url=ALPHA_URL,
        expected_title="Twinned plan title",
        candidate_urls=[OLDER_TWIN_URL, NEWER_TWIN_URL],
    )
    assert error.supplied_url == ALPHA_URL
    assert error.candidate_urls == [OLDER_TWIN_URL, NEWER_TWIN_URL]


def test_every_refusal_is_a_dashboard_url_error() -> None:
    """
    One except clause catches the whole family.
    """
    for error in (
        MalformedArtifactUrlError(supplied_url="nonsense"),
        ArtifactNotPublishedError(expected_title="absent"),
        UnlistedArtifactUrlError(
            supplied_url=ALPHA_URL, expected_title="absent", candidate_urls=[]
        ),
    ):
        assert isinstance(error, DashboardUrlError)


def test_the_refusal_base_class_cannot_be_raised_on_its_own() -> None:
    """
    A refusal must say what went wrong and what to do, so the base class is abstract.
    """
    with pytest.raises(TypeError):
        DashboardUrlError()


# %% a duplicated title resolves to the freshest artifact


def test_duplicate_title_resolves_to_the_most_recently_updated() -> None:
    """
    Two artifacts sharing a title need no question: the newer page is the live one.
    """
    assert resolve_artifact_url(LISTING, "Twinned plan title", None) == NEWER_TWIN_URL


def test_equally_recent_duplicates_keep_listing_order() -> None:
    """
    The Artifact listing is ordered most-recent-first, so among equal dates the earliest
    listed is the freshest, and the later one is not chosen.
    """
    resolved = resolve_artifact_url(LISTING, "Tied plan title", None)
    assert resolved == FIRST_TIED_URL
    assert resolved != SECOND_TIED_URL


def test_most_recently_updated_picks_the_later_date() -> None:
    """
    Ranking is by the updated date, not by listing position.
    """
    candidates = [entry for entry in LISTING if entry.title == "Twinned plan title"]
    assert most_recently_updated(candidates).url == NEWER_TWIN_URL


def test_duplicate_title_accepts_an_override_from_the_candidates() -> None:
    """
    An explicit choice among the same-titled candidates overrides recency.
    """
    assert (
        resolve_artifact_url(LISTING, "Twinned plan title", OLDER_TWIN_URL)
        == OLDER_TWIN_URL
    )


def test_duplicate_title_refuses_an_override_belonging_to_another_title() -> None:
    """
    An override may not repoint a key at some other plan's artifact.
    """
    with pytest.raises(UnlistedArtifactUrlError):
        resolve_artifact_url(LISTING, "Twinned plan title", ALPHA_URL)


# %% patching the cache text


def test_existing_key_is_repointed_and_previous_url_reported() -> None:
    """
    Rewriting a key reports what it held before.
    """
    patched = apply_url_record(CACHE_TEXT, "beta-plan", BETA_URL)
    assert patched.previous_url == STALE_BETA_URL
    assert yaml.safe_load(patched.text)["beta-plan"] == BETA_URL


def test_unrelated_keys_and_comments_survive_a_write() -> None:
    """
    Only the addressed key's line changes; the header comment and every other plan's
    entry are left byte-for-byte alone.
    """
    patched = apply_url_record(CACHE_TEXT, "beta-plan", BETA_URL)
    changed_lines = set(CACHE_TEXT.split("\n")) ^ set(patched.text.split("\n"))
    assert changed_lines == {
        f"beta-plan: {STALE_BETA_URL}",
        f"beta-plan: {BETA_URL}",
    }


def test_a_new_key_is_appended() -> None:
    """
    A plan publishing for the first time gains an entry.
    """
    patched = apply_url_record(CACHE_TEXT, "twinned-plan", NEWER_TWIN_URL)
    assert patched.previous_url is None
    assert yaml.safe_load(patched.text)["twinned-plan"] == NEWER_TWIN_URL


def test_an_appended_key_keeps_the_single_trailing_newline() -> None:
    """
    A new entry goes before the trailing blank, not after it.
    """
    patched = apply_url_record(CACHE_TEXT, "twinned-plan", NEWER_TWIN_URL)
    assert patched.text.endswith(f"twinned-plan: {NEWER_TWIN_URL}\n")


def test_rewriting_a_key_with_its_current_url_leaves_the_text_unchanged() -> None:
    """
    An in-place update that did not move the page is a no-op, so nothing is pushed for
    it.
    """
    patched = apply_url_record(CACHE_TEXT, "alpha-plan", ALPHA_URL)
    assert patched.previous_url == ALPHA_URL
    assert patched.text == CACHE_TEXT


def test_an_existing_entry_line_is_located_with_its_prefix_and_url() -> None:
    """
    The located line carries what is needed to rewrite it without reformatting.
    """
    located = find_cache_entry_line(CACHE_TEXT.split("\n"), "alpha-plan")
    assert located.prefix == "alpha-plan: "
    assert located.url == ALPHA_URL


def test_a_key_with_no_entry_is_not_located() -> None:
    """
    A key absent from the cache reports as absent rather than as a blank entry.
    """
    assert find_cache_entry_line(CACHE_TEXT.split("\n"), "twinned-plan") is None


def test_append_position_skips_trailing_blank_lines() -> None:
    """
    The insertion point is after the last entry, not after the trailing newline.
    """
    assert append_position(["a: 1", ""]) == 1


# %% the command line


def _write_inputs(tmp_path: Path) -> tuple[Path, Path, Path]:
    """
    Write the shared cache and listing fixtures into ``tmp_path``, and return the cache,
    listing and output paths.
    """
    cache_path = tmp_path / "dashboard-urls.yaml"
    cache_path.write_text(CACHE_TEXT)
    listing_path = tmp_path / "artifact-listing.json"
    listing_path.write_text((FIXTURES_DIRECTORY / "artifact-listing.json").read_text())
    return cache_path, listing_path, tmp_path / "updated-dashboard-urls.yaml"


def test_command_line_records_a_drifted_url_and_reports_the_change(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A cache entry pointing at a dead URL is repaired from the listing, and the change is
    reported so the caller knows to push it.
    """
    cache_path, listing_path, output_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_dashboard_url.py",
            "--key",
            "beta-plan",
            "--expected-title",
            "Beta plan title",
            "--listing",
            str(listing_path),
            "--cache",
            str(cache_path),
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "key": "beta-plan",
        "url": BETA_URL,
        "previous_url": STALE_BETA_URL,
        "changed": True,
    }
    assert yaml.safe_load(output_path.read_text())["beta-plan"] == BETA_URL


def test_command_line_reports_no_change_when_the_cache_is_already_right(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    A cache already naming the live artifact needs no push.
    """
    cache_path, listing_path, output_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_dashboard_url.py",
            "--key",
            "_index",
            "--expected-title",
            "Plan Dashboards",
            "--listing",
            str(listing_path),
            "--cache",
            str(cache_path),
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()

    assert exit_code == 0
    assert json.loads(capsys.readouterr().out) == {
        "key": "_index",
        "url": INDEX_URL,
        "previous_url": INDEX_URL,
        "changed": False,
    }


def test_command_line_refuses_a_fabricated_url(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """
    The failure is loud and the cache is left untouched, rather than silently recording
    a URL nobody can open.
    """
    cache_path, listing_path, output_path = _write_inputs(tmp_path)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "record_dashboard_url.py",
            "--key",
            "beta-plan",
            "--expected-title",
            "Beta plan title",
            "--url",
            "https://claude.ai/code/artifact/69b0dd3c-10cf-414b-8572-f389dbf21825",
            "--listing",
            str(listing_path),
            "--cache",
            str(cache_path),
            "--output",
            str(output_path),
        ],
    )

    exit_code = main()

    assert exit_code == 1
    assert not output_path.exists()
    assert "is not among the artifacts titled" in capsys.readouterr().err
