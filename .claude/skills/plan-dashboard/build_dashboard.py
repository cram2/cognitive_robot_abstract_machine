#!/usr/bin/env python3
"""Render a single plan's dashboard HTML from its manifest + live GitHub data.

Generic, plan-agnostic: every plan-specific value (title, items, tracking
link, ...) comes from the inputs below, never hardcoded here. This is the
deterministic half of /plan-dashboard - the skill (SKILL.md) is responsible
for gathering the inputs (git show on the personal-notes branch, GitHub API
calls) and for the one step this script cannot do itself: calling the
Artifact tool to publish the HTML this script produces.

Usage:
    python3 build_dashboard.py \\
        --plan /tmp/plan.yaml \\
        --roadmap /tmp/roadmap.md \\
        --pr-data /tmp/pr_data.json \\
        --output /tmp/dashboard.html \\
        [--tracking-url "https://github.com/<owner>/<repo>/issues/<n>"]

pr_data.json shape: {"<owner>/<repo>": {"<pr_number>": {"state": "open"|
"closed", "draft": bool, "merged_at": str|null, "labels": [str, ...]}}} -
one entry per pull request number referenced by any item. A closed entry
must carry merged_at explicitly, null included; omitting it is rejected
rather than read as unmerged. See pr-data-fetching.md (next to this script)
for how a session should gather it.

Prints a one-line JSON summary to stdout (status counts, drift count,
ready-to-start/blocker-maybe-cleared item titles) so the calling skill can
report back without re-parsing the HTML it just wrote.

Requires PyYAML (manifest parsing), Jinja2 (page rendering, via
render_common.create_template_environment), and the ``markdown`` package
(roadmap.md -> HTML).
"""

from __future__ import annotations

import argparse
import json
import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any, ClassVar

import yaml

from render_common import (
    create_template_environment,
    render_markdown_to_html,
    sanitize_http_url,
)

MAXIMUM_DEPENDENCY_STACK_LEVEL = 4
"""Same-track dependency chains deeper than this wrap back to indent level 0."""


class ItemStatus(StrEnum):
    """The thin, manually-maintained status a plan.yaml item carries.

    Deliberately thin: everything about a pull request's actual GitHub state
    (open/draft/merged/CI/review) is never stored here - it is always
    live-fetched and represented separately by :class:`LiveState`.
    """

    NOT_STARTED = "not_started"
    IN_PROGRESS = "in_progress"
    BLOCKED = "blocked"
    DEFERRED = "deferred"
    DONE = "done"

    @property
    def display_label(self) -> str:
        """The human-readable label shown in the dashboard UI for this status."""
        match self:
            case ItemStatus.NOT_STARTED:
                return "Not started"
            case ItemStatus.IN_PROGRESS:
                return "In progress"
            case ItemStatus.BLOCKED:
                return "Blocked"
            case ItemStatus.DEFERRED:
                return "Deferred"
            case ItemStatus.DONE:
                return "Done"


class LiveState(StrEnum):
    """An item's live GitHub pull request state, classified fresh on every run.

    :attr:`NO_PULL_REQUEST` is a real member, not represented as ``None``:
    "this item has no pull request yet" is itself a meaningful, displayable state
    (with its own label and CSS class) rather than an absence value every
    caller would otherwise have to special-case around the enum.
    """

    NO_PULL_REQUEST = "none"
    MERGED = "merged"
    OPEN_DRAFT = "open_draft"
    OPEN_READY = "open_ready"
    CLOSED_UNMERGED = "closed_unmerged"
    NOT_FOUND = "not_found"

    @property
    def display_label(self) -> str:
        """The human-readable label shown in the dashboard UI for this state."""
        match self:
            case LiveState.NO_PULL_REQUEST:
                return "No pull request yet"
            case LiveState.MERGED:
                return "Merged"
            case LiveState.OPEN_DRAFT:
                return "Open · Draft"
            case LiveState.OPEN_READY:
                return "Open · Ready"
            case LiveState.CLOSED_UNMERGED:
                return "Closed (unmerged)"
            case LiveState.NOT_FOUND:
                return "Not found on GitHub"


class PullRequestState(StrEnum):
    """GitHub's own coarse-grained pull request state, as returned by its API."""

    OPEN = "open"
    CLOSED = "closed"


class PullRequestLabel(StrEnum):
    """The GitHub labels this repo's own convention attaches to a pull
    request by hand, that this codebase's logic or a session cares about
    recognizing.

    Not exhaustive of every label a real pull request can carry - GitHub's
    own label vocabulary is open-ended, and other automation on this repo
    may add labels this dashboard never needs to know about. See
    :attr:`PullRequestRecord.identified_labels` for how an unrecognized
    label is handled (silently excluded, not an error), and
    ``.claude/hooks/README.md``'s "labels the dashboard reads" list for what
    each member means and who applies it.
    """

    MERGED = "merged"
    IN_REVIEW = "in-review"
    BUG = "bug"


class ValidationProblem(ABC):
    """A single problem found while validating a plan.yaml - see plan-schema.md.

    One dataclass subclass per validation rule, each carrying the specific
    fields that rule cares about rather than a pre-formatted string - so a
    caller that wants to react to (say) only duplicate-id problems can
    ``isinstance()``-check for :class:`DuplicateItemId` instead of comparing
    against a generic ``kind`` tag. Not an :class:`Exception` subclass:
    these are collected as data by :func:`validate_plan` and never
    individually raised - only the aggregate :class:`PlanValidationError`
    ever is.
    """

    @abstractmethod
    def describe(self) -> str:
        """The human-readable description of this problem, shown to the user."""


@dataclass
class InvalidManifestRoot(ValidationProblem):
    """The manifest didn't parse to a mapping - an empty file (``yaml.safe_load``
    returns ``None``) or a manifest whose top level is a list or scalar."""

    actual_value: Any
    """Whatever the manifest actually parsed to."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"plan.yaml must parse to a mapping, got {type(self.actual_value).__name__}: {self.actual_value!r}"


@dataclass
class InvalidSchemaVersion(ValidationProblem):
    """The manifest's ``schema_version`` is missing or not ``1``."""

    actual_value: Any
    """Whatever ``schema_version`` actually held."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"schema_version must be 1, got {self.actual_value!r}"


@dataclass
class DuplicateItemId(ValidationProblem):
    """Two or more items resolve to the same effective id."""

    duplicate_identifiers: list[str]
    """Every identifier that occurred more than once."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"duplicate item id(s): {sorted(self.duplicate_identifiers)}"


@dataclass
class UnknownTrack(ValidationProblem):
    """An item's ``track`` doesn't resolve to a declared track."""

    item_identifier: str
    """The offending item's effective id."""

    track: Any
    """Whatever ``track`` actually held."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"item {self.item_identifier!r} has unknown track {self.track!r}"


@dataclass
class UnknownStatus(ValidationProblem):
    """An item's ``status`` isn't one of :class:`ItemStatus`'s values."""

    item_identifier: str
    """The offending item's effective id."""

    status: Any
    """Whatever ``status`` actually held."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"item {self.item_identifier!r} has unknown status {self.status!r}"


@dataclass
class InvalidDependsOn(ValidationProblem):
    """An item's ``depends_on`` isn't a list.

    A plain string is iterable character-by-character in Python, so without
    this check a string ``depends_on`` would silently be misread as one
    dependency per character instead of failing loudly.
    """

    item_identifier: str
    """The offending item's effective id."""

    actual_type: type
    """The type ``depends_on`` actually held, instead of ``list``."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"item {self.item_identifier!r} depends_on must be a list, got {self.actual_type.__name__}"


@dataclass
class UnknownDependency(ValidationProblem):
    """An item's ``depends_on`` names an id that doesn't resolve to another item."""

    item_identifier: str
    """The offending item's effective id."""

    dependency_identifier: str
    """The unresolvable id named in ``depends_on``."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"item {self.item_identifier!r} depends_on unknown id {self.dependency_identifier!r}"


@dataclass
class UnknownWave(ValidationProblem):
    """A track's ``wave`` doesn't resolve to a declared wave."""

    track_identifier: str
    """The offending track's id."""

    wave: Any
    """Whatever ``wave`` actually held."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"track {self.track_identifier!r} has unknown wave {self.wave!r}"


@dataclass
class DependencyCycle(ValidationProblem):
    """A ``depends_on`` chain loops back on itself.

    Undetected, a cycle causes affected items to silently disappear from
    the rendered dependency stack instead of being reported."""

    cycle_identifiers: list[str]
    """The item ids forming the cycle, in order, with the first id repeated
    at the end to show where it closes."""

    def describe(self) -> str:
        """See :meth:`ValidationProblem.describe`."""
        return f"depends_on cycle: {' -> '.join(self.cycle_identifiers)}"


class PlanValidationError(Exception):
    """Raised when a plan.yaml fails schema validation - see plan-schema.md."""

    def __init__(self, problems: list[ValidationProblem]) -> None:
        self.problems = problems
        """Every problem found - collected rather than stopping at the first
        one, the same way a linter reports every violation in one pass
        instead of one-error-at-a-time: a broken manifest is itself
        something the user needs the full picture of, not a single
        symptom they have to rediscover the rest of by trial and error."""
        super().__init__("; ".join(problem.describe() for problem in problems))


def _find_dependency_cycle(
    item_identifiers: list[str], depends_on_by_identifier: dict[str, list[str]]
) -> list[str] | None:
    """Find one cycle in the ``depends_on`` graph, if any exists.

    :param item_identifiers: Every item's effective id.
    :param depends_on_by_identifier: Each item id's own ``depends_on`` list,
        already known-valid list ids (an id naming something outside
        ``item_identifiers`` is a separate, already-reported problem and is
        skipped here rather than followed).
    :return: The cycle's item ids in order, with the first id repeated at
        the end, or ``None`` if the graph is acyclic.
    """
    UNVISITED, IN_PROGRESS, DONE = 0, 1, 2
    state = {identifier: UNVISITED for identifier in item_identifiers}
    path: list[str] = []

    def visit(identifier: str) -> list[str] | None:
        state[identifier] = IN_PROGRESS
        path.append(identifier)
        for dependency_identifier in depends_on_by_identifier.get(identifier, []):
            if dependency_identifier not in state:
                continue
            if state[dependency_identifier] == IN_PROGRESS:
                cycle_start = path.index(dependency_identifier)
                return [*path[cycle_start:], dependency_identifier]
            if state[dependency_identifier] == UNVISITED:
                cycle = visit(dependency_identifier)
                if cycle is not None:
                    return cycle
        path.pop()
        state[identifier] = DONE
        return None

    for identifier in item_identifiers:
        if state[identifier] == UNVISITED:
            cycle = visit(identifier)
            if cycle is not None:
                return cycle
    return None


def validate_plan(plan: dict[str, Any]) -> None:
    """Check the same schema rules plan-create is required to produce
    manifests that pass.

    :param plan: The raw, freshly-``yaml.safe_load``-ed plan.yaml content.
        Deliberately a loosely-typed dict rather than the strongly-typed
        :class:`Plan` dataclass: this function must tolerate partially
        malformed input well enough to collect every problem in it, which
        a dataclass constructor (failing outright on the first missing or
        wrong-typed field) cannot do.
    :raises PlanValidationError: If any rule is violated, carrying every
        problem found.
    """
    if not isinstance(plan, dict):
        raise PlanValidationError([InvalidManifestRoot(plan)])

    problems: list[ValidationProblem] = []

    if plan.get("schema_version") != 1:
        problems.append(InvalidSchemaVersion(plan.get("schema_version")))

    item_identifiers = [
        item.get("id") or item.get("branch") for item in plan.get("items", [])
    ]
    if len(item_identifiers) != len(set(item_identifiers)):
        seen: set[str] = set()
        duplicate_identifiers = {
            identifier
            for identifier in item_identifiers
            if identifier in seen or seen.add(identifier)
        }
        problems.append(DuplicateItemId(sorted(duplicate_identifiers)))

    track_identifiers = {track["id"] for track in plan.get("tracks", [])}
    wave_identifiers = {wave["id"] for wave in plan.get("waves", [])}
    item_identifier_set = set(item_identifiers)
    depends_on_by_identifier: dict[str, list[str]] = {}

    for item in plan.get("items", []):
        item_identifier = item.get("id") or item.get("branch")
        if item.get("track") not in track_identifiers:
            problems.append(UnknownTrack(item_identifier, item.get("track")))
        if item.get("status") not in {status.value for status in ItemStatus}:
            problems.append(UnknownStatus(item_identifier, item.get("status")))
        depends_on = item.get("depends_on")
        if depends_on is not None and not isinstance(depends_on, list):
            problems.append(InvalidDependsOn(item_identifier, type(depends_on)))
        else:
            depends_on_by_identifier[item_identifier] = list(depends_on or [])
            for dependency_identifier in depends_on or []:
                if dependency_identifier not in item_identifier_set:
                    problems.append(
                        UnknownDependency(item_identifier, dependency_identifier)
                    )

    cycle = _find_dependency_cycle(item_identifiers, depends_on_by_identifier)
    if cycle is not None:
        problems.append(DependencyCycle(cycle))

    for track in plan.get("tracks", []):
        if track.get("wave") not in wave_identifiers:
            problems.append(UnknownWave(track["id"], track.get("wave")))

    if problems:
        raise PlanValidationError(problems)


class MissingMergeTimestampError(ValueError):
    """Raised when a closed pull request's ``pr_data.json`` entry omits the
    ``merged_at`` key altogether, which leaves no way to tell a merged pull
    request from one closed unmerged."""


@dataclass
class PullRequestRecord:
    """The live GitHub state of one pull request, as gathered by the skill."""

    state: PullRequestState
    """GitHub's own coarse-grained state."""

    draft: bool = False
    """Whether the pull request is currently a draft."""

    merged_at: datetime | None = None
    """The pull request's merge timestamp, or ``None`` if GitHub never
    recorded a merge through its own merge API."""

    labels: list[str] = field(default_factory=list)
    """The pull request's raw GitHub labels, exactly as GitHub reports them.
    Kept as ``list[str]`` rather than ``list[PullRequestLabel]``: a real pull
    request can carry labels this codebase has no reason to know about (other
    automation's own labels, GitHub's defaults), so this field must tolerate
    anything - see :attr:`identified_labels` for the subset this codebase
    actually recognizes. Needed because :attr:`merged_at` alone misses a real
    case in this repo's history: a pull request merged out-of-band (its
    branch pushed directly, then the pull request closed by hand rather than
    through GitHub's merge button) never gets ``merged_at`` set, so the
    closer manually adds the :attr:`PullRequestLabel.MERGED` label to record
    what actually happened - see :meth:`was_merged`."""

    @property
    def identified_labels(self) -> frozenset[PullRequestLabel]:
        """The subset of :attr:`labels` that match a known
        :class:`PullRequestLabel` - any other raw label this pull request
        happens to carry is silently excluded, not an error, since
        :attr:`labels` is deliberately not limited to labels this codebase
        recognizes."""
        known_values = {member.value for member in PullRequestLabel}
        return frozenset(
            PullRequestLabel(label) for label in self.labels if label in known_values
        )

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> PullRequestRecord:
        """Build a record from one entry of ``pr_data.json``.

        A closed entry must carry ``merged_at`` explicitly, ``null`` included:
        a gatherer that never requested the field would otherwise be
        indistinguishable from GitHub reporting no merge, silently turning
        every merged pull request into a closed-unmerged one.

        :raises MissingMergeTimestampError: If a closed entry omits ``merged_at``.
        """
        state = PullRequestState(data["state"])
        if state is PullRequestState.CLOSED and "merged_at" not in data:
            raise MissingMergeTimestampError(
                "a closed pull request entry must carry merged_at (null included)"
            )
        merged_at = data.get("merged_at")
        return cls(
            state=state,
            draft=data.get("draft", False),
            merged_at=datetime.fromisoformat(merged_at) if merged_at else None,
            labels=list(data.get("labels") or []),
        )

    @property
    def was_merged(self) -> bool:
        """Whether this pull request's changes actually landed - GitHub's own
        :attr:`merged_at`, or (for an out-of-band merge GitHub never
        recorded) the manually-applied :attr:`PullRequestLabel.MERGED`
        label."""
        return (
            self.merged_at is not None
            or PullRequestLabel.MERGED in self.identified_labels
        )


class MalformedPullRequestDataError(ValueError):
    """Raised when one entry of ``pr_data.json`` cannot be parsed into a
    :class:`PullRequestRecord` - a missing or invalid ``state`` field, or an
    unparsable ``merged_at`` timestamp."""


PullRequestsByRepository = dict[str, dict[str, PullRequestRecord]]


def classify_live_state(
    pull_request_number: int | None,
    repository: str,
    pull_requests_by_repository: PullRequestsByRepository,
) -> LiveState:
    """Classify one item's live GitHub state from its pull request number and repository.

    Standalone so callers other than :class:`DashboardRenderer` (notably
    ``sync_manifest_status.py``, which needs the same classification to
    decide what to auto-correct) don't have to duplicate this logic.

    :param pull_request_number: The item's tracked pull request number, or ``None`` if
        it has no pull request yet.
    :param repository: The ``"owner/repo"`` to look the pull request up under.
    :param pull_requests_by_repository: Live pull request state for every repository
        referenced by the plan's items.
    :return: The classified state.
    """
    if pull_request_number is None:
        return LiveState.NO_PULL_REQUEST
    repository_pull_requests = pull_requests_by_repository.get(repository, {})
    pull_request = repository_pull_requests.get(str(pull_request_number))
    if pull_request is None:
        return LiveState.NOT_FOUND
    if pull_request.was_merged:
        return LiveState.MERGED
    if pull_request.state is PullRequestState.CLOSED:
        return LiveState.CLOSED_UNMERGED
    if pull_request.draft:
        return LiveState.OPEN_DRAFT
    return LiveState.OPEN_READY


@dataclass
class Wave:
    """A sequential phase of the initiative - wave 2 generally starts once
    wave 1 has landed."""

    id: str
    """The wave's stable identifier, referenced by :attr:`Track.wave`."""

    name: str
    """The wave's display name."""

    description: str | None = None
    """An optional one-line note about the wave, shown in the dashboard header."""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Wave:
        """Build a wave from one entry of plan.yaml's ``waves[]`` - only
        called after :func:`validate_plan` has already confirmed the data
        is well-formed."""
        return cls(
            id=data["id"],
            name=data["name"],
            description=data.get("description"),
        )


@dataclass
class Track:
    """A parallel line of work within a wave - its items can proceed
    independently of other tracks in the same wave."""

    id: str
    """The track's stable identifier, referenced by :attr:`Item.track`."""

    name: str
    """The track's display name."""

    wave: str
    """The :attr:`Wave.id` this track belongs to."""

    description: str | None = None
    """Shown in place of an item list when the track has no items yet."""

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Track:
        """Build a track from one entry of plan.yaml's ``tracks[]`` - only
        called after :func:`validate_plan` has already confirmed the data
        is well-formed."""
        return cls(
            id=data["id"],
            name=data["name"],
            wave=data["wave"],
            description=data.get("description"),
        )


@dataclass
class DependencyChip:
    """One ready-to-render ``needs`` chip on an item's card - see
    :attr:`Item.dependency_chips`. Precomputed so the template never has to
    look a dependency identifier up in ``items_by_identifier`` or fall back
    to the raw identifier itself."""

    identifier: str
    """The chip's display text: the dependency's own effective identifier,
    or the raw ``depends_on`` entry if it doesn't resolve to a known item."""

    tooltip: str
    """The chip's hover title: the dependency's title, or its identifier
    again if it doesn't resolve to a known item."""

    is_ready: bool
    """Whether the dependency is actually safe to build on right now
    (:meth:`Item.is_ready_to_unblock_dependents`) - ``False`` for an
    unresolved identifier, since an item this plan doesn't know about can
    never be considered ready. Drives the chip's ``chip-unmet`` styling, the
    dashboard's one visual cue that an item is blocked on this dependency."""


@dataclass(frozen=True)
class ItemAction(ABC):
    """One not-done item's actionable dashboard button - see
    :attr:`Item.action`. The label matches what the status actually calls
    for (starting fresh work reads differently from resolving a blocker),
    but every non-``done`` status gets one: there's always something
    actionable to do next besides waiting.

    Not a plain enum: each action carries the plan/item it targets, so its
    members can't be fixed singletons - one subclass per ``/plan-item-...``
    skill instead. Not instantiated directly - see :class:`StartNowAction`/
    :class:`ResolveAction`."""

    label: str
    """The button's text, e.g. ``"Start now"`` or ``"Resolve"``."""

    plan_id: str
    """The plan this action's command targets."""

    item_identifier: str
    """The item this action's command targets."""

    skill_command_name: ClassVar[str]
    """The ``/plan-item-...`` skill this action invokes - fixed per
    subclass, since which skill an action routes to never varies per
    instance (only :attr:`label` does, e.g. ``"Resolve"`` vs. ``"Resume"``
    for the same :class:`ResolveAction`)."""

    @property
    def command(self) -> str:
        """The full command copied to the clipboard when the button is
        clicked."""
        return f"{self.skill_command_name} {self.plan_id} {self.item_identifier}"


@dataclass(frozen=True)
class StartNowAction(ItemAction):
    """Routes a not-started item, once every dependency is ready, to
    ``plan-item-kickoff``."""

    skill_command_name: ClassVar[str] = "/plan-item-kickoff"


@dataclass(frozen=True)
class ResolveAction(ItemAction):
    """Routes a blocked, in-progress, or deferred item to
    ``plan-item-resolve`` - :attr:`ItemAction.label` is worded to match
    which of those three it actually is."""

    skill_command_name: ClassVar[str] = "/plan-item-resolve"


@dataclass(frozen=True)
class ModelOption:
    """One choice in a dashboard action button's model dropdown."""

    value: str
    """The model id to prepend as ``/model <value>`` before the copied
    command, or ``""`` to copy the command alone and inherit whatever
    model the pasted-into session is already running."""

    label: str
    """The human-readable label shown in the dropdown."""


AVAILABLE_MODELS: list[ModelOption] = [
    ModelOption(value="", label="Session default"),
    ModelOption(value="claude-opus-5", label="Opus 5"),
    ModelOption(value="claude-sonnet-5", label="Sonnet 5"),
    ModelOption(value="claude-haiku-4-5-20251001", label="Haiku 4.5"),
    ModelOption(value="claude-fable-5", label="Fable 5"),
]
"""Every model offered in an action button's dropdown, dashboard-wide - not
plan-specific, so declared once at module level rather than threaded through
:class:`Plan`."""


@dataclass
class Item:
    """One tracked unit of work (typically one branch/pull request) within a plan."""

    title: str
    """The item's display title."""

    branch: str
    """The git branch this item is implemented on."""

    track: str
    """The :attr:`Track.id` this item belongs to."""

    status: ItemStatus
    """The manually-maintained status - see :class:`ItemStatus`."""

    id: str | None = None
    """The item's stable identifier, defaulting to :attr:`branch` if unset."""

    pull_request_number: int | None = None
    """The pull request number tracking this item, if one exists yet."""

    repository: str | None = None
    """Overrides the plan's ``default_repository`` for this item, if set."""

    session: str | None = None
    """A link to the session that produced this item, if any."""

    notes: str | None = None
    """Free-text notes shown on the item's card."""

    depends_on: list[str] = field(default_factory=list)
    """The identifiers of items that must complete before this one can start."""

    blockers: list[str] = field(default_factory=list)
    """Free-text descriptions of what's currently blocking this item."""

    live_state: LiveState = field(default=LiveState.NO_PULL_REQUEST, init=False)
    """This item's live GitHub state, filled in by :meth:`DashboardRenderer.render`."""

    drift_description: str | None = field(default=None, init=False)
    """Why :attr:`status` disagrees with :attr:`live_state`, if it does."""

    pull_request_url: str | None = field(default=None, init=False)
    """This item's pull request URL on GitHub, filled in by
    :meth:`DashboardRenderer.render` - ``None`` if it has no pull request yet."""

    dependency_chips: list[DependencyChip] = field(default_factory=list, init=False)
    """Ready-to-render chips for :attr:`depends_on`, filled in by
    :meth:`DashboardRenderer.render`."""

    action: ItemAction | None = field(default=None, init=False)
    """This item's dashboard action button, filled in by
    :meth:`DashboardRenderer.render` - ``None`` only for a ``done`` item
    (nothing left to do) or a not-started item whose dependencies aren't
    all ready yet (starting now would build on unsafe state). Every other
    status always gets one: something is always actionable next."""

    needs_review: bool = field(default=False, init=False)
    """Whether this item's pull request is open, still a draft, and actually worth
    reviewing right now, filled in by :meth:`DashboardRenderer.render`.
    This plan's convention keeps every pull request in draft until its author has
    reviewed it themselves - so a draft pull request is exactly the population that
    still needs that review, and flipping it to "ready for review" *is*
    the record of having done so. False for a ``deferred`` item even with
    an open draft pull request - deferred means intentionally paused or superseded,
    so there is nothing to actually review yet. Drives the dashboard's
    "Review" button and the "ready to review" sidebar list - distinct from
    :attr:`Item.action`, since reviewing a draft pull request and resuming/resolving
    the underlying work are different next steps that can both apply to
    the same item at once."""

    @property
    def has_open_pull_request(self) -> bool:
        """Whether this item currently has an open (draft or ready) pull request -
        used to decide whether a dependent's pull request is safe to review yet."""
        return self.live_state in (LiveState.OPEN_DRAFT, LiveState.OPEN_READY)

    @property
    def identifier(self) -> str:
        """The item's effective identifier: :attr:`id`, or :attr:`branch` if unset."""
        return self.id or self.branch

    @property
    def status_and_drift_css_class(self) -> str:
        """The item card's dynamic CSS class suffix: ``status-<value>``,
        plus ``has-drift`` once :attr:`drift_description` is set."""
        drift_suffix = " has-drift" if self.drift_description else ""
        return f"status-{self.status.value}{drift_suffix}"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Item:
        """Build an item from one entry of plan.yaml's ``items[]`` - only
        called after :func:`validate_plan` has already confirmed the data
        is well-formed."""
        notes = data.get("notes")
        return cls(
            title=data["title"],
            branch=data["branch"],
            track=data["track"],
            status=ItemStatus(data["status"]),
            id=data.get("id"),
            pull_request_number=data.get("pull_request_number"),
            repository=data.get("repository"),
            session=sanitize_http_url(data.get("session")),
            notes=notes.strip() if notes else None,
            depends_on=list(data.get("depends_on") or []),
            blockers=list(data.get("blockers") or []),
        )

    def is_effectively_done(self) -> bool:
        """Whether this item can unblock a dependent, by manifest status or live state."""
        return self.status is ItemStatus.DONE or self.live_state is LiveState.MERGED

    def is_ready_to_unblock_dependents(self) -> bool:
        """Whether a dependent item can safely start stacking its own branch
        on this one: done, merged, or its pull request is open and out of draft (ready
        for review). Stacking on a same-track dependency is this repo's
        normal workflow well before that dependency merges - a still-open
        draft is the one state that isn't safe to build on top of, since it
        can still see heavy rework."""
        return self.is_effectively_done() or self.live_state is LiveState.OPEN_READY


@dataclass
class Plan:
    """A full initiative spanning multiple pull requests and sessions, as read from plan.yaml."""

    id: str
    """The plan's stable identifier - the directory name under ``plans/``."""

    title: str
    """The plan's display title."""

    description: str
    """A one-line description shown under the title."""

    default_repository: str
    """The ``"owner/repo"`` items resolve pull requests against unless they override it."""

    waves: list[Wave]
    """The plan's sequential phases, in order."""

    tracks: list[Track]
    """The plan's parallel lines of work, each tagged with a wave."""

    items: list[Item]
    """The plan's tracked units of work, each tagged with a track."""

    tracking_issue: int | None = None
    """The coordination-mailbox issue or pull request number for structural changes, if any."""

    @property
    def repository_url(self) -> str:
        """:attr:`default_repository`'s GitHub URL, ready for the masthead link."""
        return f"https://github.com/{self.default_repository}"

    @classmethod
    def from_mapping(cls, data: dict[str, Any]) -> Plan:
        """Build a plan from a freshly-loaded plan.yaml - only called after
        :func:`validate_plan` has already confirmed the data is well-formed."""
        return cls(
            id=data["id"],
            title=data["title"],
            description=data["description"],
            default_repository=data["default_repository"],
            waves=[Wave.from_mapping(wave) for wave in data.get("waves", [])],
            tracks=[Track.from_mapping(track) for track in data.get("tracks", [])],
            items=[Item.from_mapping(item) for item in data.get("items", [])],
            tracking_issue=data.get("tracking_issue"),
        )


@dataclass
class DashboardSummary:
    """The one-line JSON summary this script prints to stdout on success."""

    status_counts: dict[ItemStatus, int]
    """How many items carry each :class:`ItemStatus`."""

    drift_items: list[str]
    """Titles of items whose manifest status disagrees with live GitHub state."""

    ready_to_start: list[str]
    """Titles of not-started items whose dependencies are all ready."""

    blocker_maybe_cleared: list[str]
    """Titles of blocked items with at least one dependency ready."""

    ready_to_review: list[str]
    """Titles of items with an open draft pull request, not blocked, whose
    dependencies (if any) already have their own open pull request."""

    def to_json_dict(self) -> dict[str, Any]:
        """Render to the plain-dict shape the calling skill expects."""
        return {
            "counts": {
                status.value: count for status, count in self.status_counts.items()
            },
            "drift_count": len(self.drift_items),
            "drift_items": self.drift_items,
            "ready_to_start": self.ready_to_start,
            "blocker_maybe_cleared": self.blocker_maybe_cleared,
            "ready_to_review": self.ready_to_review,
        }


@dataclass
class StackedItem:
    """One item's position within its track's dependency stack, for the
    ``item_card`` template macro to render.

    Carries two independent indent computations - as normally shown, and as
    shown once done items are hidden - so the page can switch between them
    client-side (see ``hide-done`` in dashboard.html) without a re-render:
    a done dependency, once hidden, no longer visually justifies indenting
    its dependents, so they dedent as if they had no dependency on it at
    all rather than merely one level shallower."""

    item: Item
    """The item itself."""

    indent_level: int
    """How deeply nested this item is under its same-track dependency chain,
    capped at :data:`MAXIMUM_DEPENDENCY_STACK_LEVEL`."""

    wrap_parent: Item | None
    """The item this one visually continues from, if the chain wrapped back
    to indent level 0 past the cap; ``None`` otherwise."""

    indent_level_with_done_hidden: int
    """:attr:`indent_level`, recomputed as if every ``done`` item in the
    same-track dependency chain weren't a dependency at all."""

    wrap_parent_with_done_hidden: Item | None
    """:attr:`wrap_parent`, recomputed the same way - never itself a
    ``done`` item, since a done wrap-parent would be invisible in that view."""

    @property
    def indent_style(self) -> str:
        """The card's ``style`` attribute value: both indent levels as CSS
        custom properties, so plain CSS (keyed off the page's ``hide-done``
        class) picks whichever applies without any per-toggle re-render."""
        return (
            f"--indent-level: {self.indent_level}; "
            f"--indent-level-hidden-done: {self.indent_level_with_done_hidden};"
        )


@dataclass
class TrackSection:
    """One track's rendering context: its declared info plus its items,
    already ordered into a dependency stack."""

    track: Track
    """The track itself."""

    stacked_items: list[StackedItem]
    """The track's items, stacked - empty if the track has none yet."""

    @property
    def empty_state_message(self) -> str:
        """What to show in place of an item list when the track has none yet:
        the track's own :attr:`Track.description`, or a generic fallback."""
        return self.track.description or "No tracked items."


@dataclass
class WaveSection:
    """One wave's rendering context: its declared info plus its tracks, in order."""

    wave: Wave
    """The wave itself."""

    tracks: list[TrackSection]
    """The wave's tracks, in declaration order."""


@dataclass
class DashboardRenderer:
    """Renders one :class:`Plan` (plus its live pull request data and roadmap text)
    into the dashboard's HTML.

    A dataclass rather than a bag of closures over shared state: each
    concern (classifying live state, computing next steps, stacking a
    track's items by dependency) is an independently named, independently
    testable method. The HTML itself is produced by handing the computed
    data to the ``dashboard.html`` Jinja2 template, not built up as Python
    strings here.
    """

    plan: Plan
    """The plan being rendered."""

    roadmap_text: str
    """The plan's ``roadmap.md`` narrative content."""

    pull_requests_by_repository: PullRequestsByRepository
    """Live pull request state for every repository referenced by the plan's items."""

    tracking_url: str | None
    """The tracking issue's or pull request's ``html_url``, if the plan has one."""

    items_by_identifier: dict[str, Item] = field(init=False)
    """Every item, keyed by :attr:`Item.identifier`."""

    def __post_init__(self) -> None:
        self.items_by_identifier = {item.identifier: item for item in self.plan.items}

    def render(self) -> tuple[str, DashboardSummary]:
        """Classify live state/drift for every item, then render the full page.

        :return: The rendered HTML, and the summary to print on stdout.
        """
        self._classify_items()
        drift_items = [item for item in self.plan.items if item.drift_description]
        ready_to_start, blocker_maybe_cleared = self._compute_next_steps()
        ready_to_review = self._compute_ready_to_review()

        template = create_template_environment().get_template("dashboard.html")
        output = template.render(
            title=self.plan.title,
            description=self.plan.description,
            repository=self.plan.default_repository,
            repository_url=self.plan.repository_url,
            total_items=len(self.plan.items),
            tracking_url=self.tracking_url,
            item_statuses=list(ItemStatus),
            status_counts=self._status_counts(),
            drift_items=drift_items,
            ready_to_start=ready_to_start,
            blocker_maybe_cleared=blocker_maybe_cleared,
            ready_to_review=ready_to_review,
            roadmap_html=render_markdown_to_html(self.roadmap_text),
            waves=self._build_wave_sections(),
            available_models=AVAILABLE_MODELS,
        )

        summary = DashboardSummary(
            status_counts=self._status_counts(),
            drift_items=[item.title for item in drift_items],
            ready_to_start=[item.title for item in ready_to_start],
            blocker_maybe_cleared=[item.title for item in blocker_maybe_cleared],
            ready_to_review=[item.title for item in ready_to_review],
        )
        return output, summary

    def _classify_items(self) -> None:
        """Fill in every item's :attr:`Item.live_state`,
        :attr:`Item.drift_description`, :attr:`Item.pull_request_url`,
        :attr:`Item.needs_review`, :attr:`Item.dependency_chips`, and
        :attr:`Item.action` from live pull request data and the plan's other items,
        in place.

        Runs in two passes: :attr:`Item.live_state` must be filled in for
        every item before :meth:`_action_for` can check whether *another*
        item's dependencies are ready, since dependencies can appear later
        in :attr:`Plan.items` than their dependents."""
        for item in self.plan.items:
            item.live_state = self._live_state_of(item)
            item.drift_description = self._drift_description_of(item)
            item.pull_request_url = self._pull_request_url_of(item)
            item.needs_review = (
                item.live_state is LiveState.OPEN_DRAFT
                and item.status is not ItemStatus.DEFERRED
            )
        for item in self.plan.items:
            item.dependency_chips = self._dependency_chips_of(item)
            item.action = self._action_for(item)

    def _pull_request_url_of(self, item: Item) -> str | None:
        """Build one item's pull request URL on GitHub, or ``None`` if it has no pull request yet."""
        if item.pull_request_number is None:
            return None
        repository = item.repository or self.plan.default_repository
        return f"https://github.com/{repository}/pull/{item.pull_request_number}"

    def _action_for(self, item: Item) -> ItemAction | None:
        """Build one item's dashboard action button, or ``None`` if it
        isn't applicable.

        A ``done`` item has nothing left to do. A not-started item only
        gets a button once every dependency is actually safe to build on -
        starting now against an unready dependency would build on unsafe
        state. Every other status (blocked, in progress, deferred) always
        gets one: there's always something actionable to investigate next,
        so each routes to the same ``/plan-item-resolve`` skill, worded to
        match what that status actually calls for."""
        match item.status:
            case ItemStatus.DONE:
                return None
            case ItemStatus.NOT_STARTED:
                if not self._dependencies_are_ready(item):
                    return None
                return StartNowAction(
                    label="Start now",
                    plan_id=self.plan.id,
                    item_identifier=item.identifier,
                )
            case ItemStatus.BLOCKED:
                label = "Resolve"
            case ItemStatus.IN_PROGRESS:
                label = "Resume"
            case ItemStatus.DEFERRED:
                label = "Reconsider"
        return ResolveAction(
            label=label,
            plan_id=self.plan.id,
            item_identifier=item.identifier,
        )

    def _dependencies_are_ready(self, item: Item) -> bool:
        """Whether every entry in :attr:`Item.depends_on` names an item
        that's itself ready to be built upon
        (:meth:`Item.is_ready_to_unblock_dependents`) - vacuously true for
        an item with no dependencies."""
        return all(
            self.items_by_identifier[
                dependency_identifier
            ].is_ready_to_unblock_dependents()
            for dependency_identifier in item.depends_on
            if dependency_identifier in self.items_by_identifier
        )

    def _dependency_chips_of(self, item: Item) -> list[DependencyChip]:
        """Build one ready-to-render :class:`DependencyChip` per entry in
        :attr:`Item.depends_on`, resolving each against :attr:`items_by_identifier`."""
        chips: list[DependencyChip] = []
        for dependency_identifier in item.depends_on:
            dependency = self.items_by_identifier.get(dependency_identifier)
            if dependency is None:
                chips.append(
                    DependencyChip(
                        identifier=dependency_identifier,
                        tooltip=dependency_identifier,
                        is_ready=False,
                    )
                )
            else:
                chips.append(
                    DependencyChip(
                        identifier=dependency.identifier,
                        tooltip=dependency.title,
                        is_ready=dependency.is_ready_to_unblock_dependents(),
                    )
                )
        return chips

    def _live_state_of(self, item: Item) -> LiveState:
        """Classify one item's live GitHub state from :attr:`pull_requests_by_repository`."""
        return classify_live_state(
            item.pull_request_number,
            item.repository or self.plan.default_repository,
            self.pull_requests_by_repository,
        )

    @staticmethod
    def _drift_description_of(item: Item) -> str | None:
        """Describe why :attr:`Item.status` disagrees with its live state, if it does."""
        live_state = item.live_state
        match live_state, item.status:
            case LiveState.NOT_FOUND, _:
                return f"pull request #{item.pull_request_number} not found on GitHub"
            case (LiveState.OPEN_DRAFT | LiveState.OPEN_READY), ItemStatus.DONE:
                return f"marked done, but pull request #{item.pull_request_number} is still open"
            case LiveState.MERGED, (
                ItemStatus.NOT_STARTED | ItemStatus.BLOCKED | ItemStatus.DEFERRED
            ):
                return f"marked {item.status.value}, but pull request #{item.pull_request_number} is already merged"
            case (LiveState.MERGED | LiveState.CLOSED_UNMERGED), (
                ItemStatus.IN_PROGRESS | ItemStatus.BLOCKED
            ):
                return f"marked {item.status.value}, but pull request #{item.pull_request_number} is {live_state.value.replace('_', ' ')}"
            case LiveState.CLOSED_UNMERGED, ItemStatus.DONE:
                return f"marked done, but pull request #{item.pull_request_number} was closed without merging"
            case _:
                return None

    def _compute_next_steps(self) -> tuple[list[Item], list[Item]]:
        """Compute the "ready to start" and "blocker may be cleared" lists
        for the sidebar, from each item's dependencies' effective status.

        A blocked item never lands in "ready to start", even once every
        dependency is ready - it's still blocked, and its actionable next
        step is to resolve that blocker, not to start fresh. It always
        lands in "blocker may be cleared" instead once at least one
        dependency is ready, whether that's all of them or only some.

        :return: ``(ready_to_start, blocker_maybe_cleared)``.
        """
        ready_to_start: list[Item] = []
        blocker_maybe_cleared: list[Item] = []
        for item in self.plan.items:
            dependencies = [
                self.items_by_identifier[identifier]
                for identifier in item.depends_on
                if identifier in self.items_by_identifier
            ]
            if item.status not in (
                ItemStatus.NOT_STARTED,
                ItemStatus.BLOCKED,
            ):
                continue
            ready_count = sum(
                dependency.is_ready_to_unblock_dependents()
                for dependency in dependencies
            )
            if item.status is ItemStatus.NOT_STARTED and self._dependencies_are_ready(
                item
            ):
                ready_to_start.append(item)
            elif item.status is ItemStatus.BLOCKED and ready_count > 0:
                blocker_maybe_cleared.append(item)
        return ready_to_start, blocker_maybe_cleared

    def _compute_ready_to_review(self) -> list[Item]:
        """Items with an open draft pull request that are actually reviewable right
        now: not blocked, and every dependency (if any) already has its own
        open pull request - reviewing a stacked pull request before its base even has one open
        yet is premature, even though the base need not itself be past
        review. A deferred item never reaches here in the first place -
        :attr:`Item.needs_review` is already ``False`` for it."""
        ready_to_review: list[Item] = []
        for item in self.plan.items:
            if not item.needs_review or item.status is ItemStatus.BLOCKED:
                continue
            dependencies = [
                self.items_by_identifier[identifier]
                for identifier in item.depends_on
                if identifier in self.items_by_identifier
            ]
            if all(dependency.has_open_pull_request for dependency in dependencies):
                ready_to_review.append(item)
        return ready_to_review

    def _status_counts(self) -> dict[ItemStatus, int]:
        """Count the plan's items by :class:`ItemStatus`, including zero counts."""
        counts = {status: 0 for status in ItemStatus}
        for item in self.plan.items:
            counts[item.status] += 1
        return counts

    def _build_wave_sections(self) -> list[WaveSection]:
        """Group the plan's tracks by wave and its items by track, stacking
        each track's items by dependency, ready for the template to render."""
        tracks_by_wave: dict[str, list[Track]] = {}
        for track in self.plan.tracks:
            tracks_by_wave.setdefault(track.wave, []).append(track)

        items_by_track: dict[str, list[Item]] = {}
        for item in self.plan.items:
            items_by_track.setdefault(item.track, []).append(item)

        return [
            WaveSection(
                wave=wave,
                tracks=[
                    TrackSection(
                        track=track,
                        stacked_items=self._build_track_stack(
                            items_by_track.get(track.id, [])
                        ),
                    )
                    for track in tracks_by_wave.get(wave.id, [])
                ],
            )
            for wave in self.plan.waves
        ]

    @staticmethod
    def _build_track_stack(track_items: list[Item]) -> list[StackedItem]:
        """Order a track's items into a dependency stack (same-track
        depends_on only), assign an indent level per item capped at
        :data:`MAXIMUM_DEPENDENCY_STACK_LEVEL`, wrapping back to level 0
        (with a reference back to the real parent) past the cap. Also
        computes each item's indent as it would be with done items hidden -
        see :class:`StackedItem`."""
        items_by_identifier = {item.identifier: item for item in track_items}

        def same_track_parent(item: Item) -> Item | None:
            """The first same-track entry in :attr:`Item.depends_on`, if any."""
            for dependency_identifier in item.depends_on:
                dependency = items_by_identifier.get(dependency_identifier)
                if dependency is not None:
                    return dependency
            return None

        children_by_parent: dict[str, list[Item]] = {}
        roots: list[Item] = []
        for item in track_items:
            parent = same_track_parent(item)
            if parent is None:
                roots.append(item)
            else:
                children_by_parent.setdefault(parent.identifier, []).append(item)

        visible_stack_cache: dict[str, tuple[int, Item | None]] = {}

        def visible_stack_position(item: Item) -> tuple[int, Item | None]:
            """This item's indent level and wrap-parent once ``done`` items
            are excluded from the same-track dependency chain entirely - a
            done dependency is treated exactly as if it weren't a
            dependency at all, so its dependents dedent back to level 0
            rather than merely one level shallower."""
            if item.identifier in visible_stack_cache:
                return visible_stack_cache[item.identifier]
            parent = same_track_parent(item)
            if parent is None or parent.status is ItemStatus.DONE:
                result = (0, None)
            else:
                parent_level, parent_wrap_parent = visible_stack_position(parent)
                next_level = parent_level + 1
                if next_level > MAXIMUM_DEPENDENCY_STACK_LEVEL:
                    result = (0, parent)
                else:
                    result = (next_level, parent_wrap_parent)
            visible_stack_cache[item.identifier] = result
            return result

        stacked_items: list[StackedItem] = []

        def walk(item: Item, level: int, wrap_parent: Item | None) -> None:
            """Depth-first-visit one item and its same-track dependents,
            appending a :class:`StackedItem` per visit."""
            next_level = level + 1
            wrap_for_children = None
            if next_level > MAXIMUM_DEPENDENCY_STACK_LEVEL:
                next_level = 0
                wrap_for_children = item
            visible_level, visible_wrap_parent = visible_stack_position(item)
            stacked_items.append(
                StackedItem(
                    item=item,
                    indent_level=level,
                    wrap_parent=wrap_parent,
                    indent_level_with_done_hidden=visible_level,
                    wrap_parent_with_done_hidden=visible_wrap_parent,
                )
            )
            for child in children_by_parent.get(item.identifier, []):
                walk(child, next_level, wrap_for_children)

        for root in roots:
            walk(root, 0, None)

        return stacked_items


def load_pull_requests_by_repository(
    raw_pull_request_data: dict[str, Any],
) -> PullRequestsByRepository:
    """Parse ``pr_data.json``'s raw JSON into :class:`PullRequestRecord`\\ s.

    :param raw_pull_request_data: The parsed JSON, keyed by ``"owner/repo"``
        then by pull request number as a string - see the module docstring for the
        exact shape.
    :raises MalformedPullRequestDataError: If any entry can't be parsed into
        a :class:`PullRequestRecord`.
    :return: The same structure, with each leaf mapping parsed into a
        :class:`PullRequestRecord`.
    """
    pull_requests_by_repository: PullRequestsByRepository = {}
    for repository, pull_requests in raw_pull_request_data.items():
        pull_requests_by_repository[repository] = {}
        for pull_request_number, record in pull_requests.items():
            try:
                pull_requests_by_repository[repository][pull_request_number] = (
                    PullRequestRecord.from_mapping(record)
                )
            except (KeyError, ValueError) as error:
                raise MalformedPullRequestDataError(
                    f"{repository}#{pull_request_number}: {error}"
                ) from error
    return pull_requests_by_repository


def main() -> int:
    """Parse arguments, validate the manifest, render the dashboard, and
    print its summary. See the module docstring for the CLI contract."""
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--plan", required=True, help="Path to plan.yaml")
    parser.add_argument("--roadmap", required=True, help="Path to roadmap.md")
    parser.add_argument(
        "--pr-data",
        required=True,
        help='Path to a JSON file: {"owner/repo": {"pr_number": {...}}}',
    )
    parser.add_argument(
        "--output", required=True, help="Path to write the dashboard HTML to"
    )
    parser.add_argument(
        "--tracking-url",
        default=None,
        help="The plan's tracking_issue html_url, if it has one",
    )
    arguments = parser.parse_args()

    raw_plan = yaml.safe_load(Path(arguments.plan).read_text())
    roadmap_text = Path(arguments.roadmap).read_text()
    raw_pull_request_data = json.loads(Path(arguments.pr_data).read_text())

    try:
        validate_plan(raw_plan)
    except PlanValidationError as error:
        print(f"plan.yaml failed validation: {error}", file=sys.stderr)
        return 1

    plan = Plan.from_mapping(raw_plan)
    pull_requests_by_repository = load_pull_requests_by_repository(
        raw_pull_request_data
    )
    renderer = DashboardRenderer(
        plan=plan,
        roadmap_text=roadmap_text,
        pull_requests_by_repository=pull_requests_by_repository,
        tracking_url=arguments.tracking_url,
    )
    output, summary = renderer.render()

    Path(arguments.output).write_text(output)
    print(json.dumps(summary.to_json_dict()))
    return 0


if __name__ == "__main__":
    sys.exit(main())
