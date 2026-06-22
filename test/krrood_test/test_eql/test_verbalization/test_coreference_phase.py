"""
Unit tests for the CoreferenceProcessor — the document-order pass that resolves referring
noun phrases (first / repeat / pronoun).

A discourse scope is opened not by a marker but by a fragment's ``source`` (the query node it was
built from): the pass asks the :class:`DiscourseModel` who the focus of that scope is. The
``_scope`` helper mirrors that — it stamps a child with a stand-in query node and pairs it with a
matching discourse model.
"""

from __future__ import annotations

import uuid
from dataclasses import replace

from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    Fragment,
    NounPhrase,
    PhraseFragment,
    PossessiveChain,
    RoleFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.rendering.coreference_processor import (
    CoreferenceProcessor,
)
from krrood.entity_query_language.verbalization.rendering.determiner_processor import (
    DeterminerProcessor,
)
from krrood.entity_query_language.verbalization.rendering.discourse import (
    DiscourseModel,
    EMPTY_DISCOURSE,
)


class _ScopeNode:
    """A minimal query-node stand-in carrying an ``_id_`` (what the discourse model keys on)."""

    def __init__(self) -> None:
        self._id_ = uuid.uuid4()


def _scope(child: Fragment, focus_id):
    """Make *child* a discourse scope with focus *focus_id* — a query-sourced fragment plus the
    matching discourse model (``focus_id=None`` for a scope with no single subject)."""
    node = _ScopeNode()
    return replace(child, source=node), DiscourseModel({node._id_: focus_id})


def _noun(text: str) -> RoleFragment:
    return RoleFragment(text=text, role=SemanticRole.VARIABLE)


def _text(fragment) -> str:
    return flatten_fragment_to_plain_text(CoreferenceProcessor().process(fragment))


def test_noop_on_plain_tree():
    tree = PhraseFragment(parts=[WordFragment(text="Find"), _noun("Robot")])
    assert _text(tree) == "Find Robot"


def test_non_referring_noun_phrase_is_preserved():
    np = NounPhrase(head=_noun("Robot"))
    out = CoreferenceProcessor().process(np)
    assert isinstance(out, NounPhrase)
    assert out.referent_id is None
    assert flatten_fragment_to_plain_text(out.head) == "Robot"


def test_query_sourced_fragment_is_processed_transparently():
    child = PhraseFragment(parts=[WordFragment(text="its"), _noun("parent")])
    scoped, discourse = _scope(child, focus_id=uuid.uuid4())
    out = CoreferenceProcessor(discourse=discourse).process(scoped)
    assert flatten_fragment_to_plain_text(out) == "its parent"


def test_recurses_into_noun_phrase_modifiers():
    np = NounPhrase(
        head=_noun("sum"),
        modifiers=[WordFragment(text="of"), NounPhrase(head=_noun("amounts"))],
    )
    out = CoreferenceProcessor().process(np)
    assert isinstance(out, NounPhrase)
    assert isinstance(out.modifiers[1], NounPhrase)  # nested NP preserved, recursed


# ── referring resolution (first / repeat / numbered) ─────────────────────────────


def _realise(fragment, discourse=EMPTY_DISCOURSE) -> str:
    """Coreference then determiner phase → plain text (the pipeline's first two stages)."""
    resolved = CoreferenceProcessor(discourse=discourse).process(fragment)
    return flatten_fragment_to_plain_text(DeterminerProcessor().process(resolved))


def test_repeat_mention_is_downgraded_to_definite():
    rid = uuid.uuid4()
    tree = PhraseFragment(
        parts=[
            NounPhrase(head=_noun("Robot"), referent_id=rid),  # first → "a Robot"
            WordFragment(text="and"),
            NounPhrase(head=_noun("Robot"), referent_id=rid),  # repeat → "the Robot"
        ]
    )
    assert _realise(tree) == "a Robot and the Robot"


def test_first_mention_modifiers_dropped_on_repeat():
    rid = uuid.uuid4()
    full = NounPhrase(
        head=_noun("Robot"),
        referent_id=rid,
        modifiers=[WordFragment(text="of"), _noun("Cabinet")],  # "a Robot of Cabinet"
    )
    repeat = NounPhrase(head=_noun("Robot"), referent_id=rid)
    tree = PhraseFragment(parts=[full, WordFragment(text="and"), repeat])
    assert _realise(tree) == "a Robot of Cabinet and the Robot"


def test_numbered_referent_never_downgrades():
    rid = uuid.uuid4()
    tree = PhraseFragment(
        parts=[
            NounPhrase(
                head=_noun("Robot 2"), definiteness=Definiteness.BARE, referent_id=rid
            ),
            WordFragment(text="and"),
            NounPhrase(
                head=_noun("Robot 2"), definiteness=Definiteness.BARE, referent_id=rid
            ),
        ]
    )
    assert _realise(tree) == "Robot 2 and Robot 2"


def test_distinct_referents_do_not_interfere():
    a, b = uuid.uuid4(), uuid.uuid4()
    tree = PhraseFragment(
        parts=[
            NounPhrase(head=_noun("Robot"), referent_id=a),
            WordFragment(text="and"),
            NounPhrase(head=_noun("Cabinet"), referent_id=b),
        ]
    )
    assert _realise(tree) == "a Robot and a Cabinet"


def test_unique_first_mention_then_definite_repeat():
    rid = uuid.uuid4()
    tree = PhraseFragment(
        parts=[
            NounPhrase(
                head=_noun("Robot"), definiteness=Definiteness.UNIQUE, referent_id=rid
            ),
            WordFragment(text="and"),
            NounPhrase(
                head=_noun("Robot"), definiteness=Definiteness.UNIQUE, referent_id=rid
            ),
        ]
    )
    assert _realise(tree) == "the unique Robot and the Robot"


def test_entity_repeat_drops_restriction_modifiers():
    rid = uuid.uuid4()
    full = NounPhrase(
        head=_noun("Robot"),
        referent_id=rid,
        modifiers=[PhraseFragment(parts=[WordFragment(text="whose battery is high")])],
    )
    repeat = NounPhrase(head=_noun("Robot"), referent_id=rid)
    tree = PhraseFragment(parts=[full, WordFragment(text="and"), repeat])
    assert _realise(tree) == "a Robot whose battery is high and the Robot"


# ── possessive chains: pronominal vs possessive ──────────────────────────────────


def _attr_part(name):
    return PathStep(name, None)


def test_chain_rooted_at_subject_pronominalises():
    rid = uuid.uuid4()
    chain = PossessiveChain(
        parts=[_attr_part("battery")],
        root_fragment=NounPhrase(head=_noun("Robot"), referent_id=rid),
        root_referent_id=rid,
    )
    scoped, discourse = _scope(
        PhraseFragment(parts=[NounPhrase(head=_noun("Robot"), referent_id=rid), chain]),
        focus_id=rid,
    )
    # subject introduced first ("a Robot"), then the chain rooted at it → "its battery"
    assert _realise(scoped, discourse) == "a Robot its battery"


def test_chain_not_rooted_at_subject_is_possessive():
    subj, other = uuid.uuid4(), uuid.uuid4()
    chain = PossessiveChain(
        parts=[_attr_part("battery")],
        root_fragment=NounPhrase(head=_noun("Robot"), referent_id=other),
        root_referent_id=other,
    )
    scoped, discourse = _scope(chain, focus_id=subj)
    # root is not the subject → possessive "the battery of a Robot" (root resolved as first mention)
    assert _realise(scoped, discourse) == "the battery of a Robot"


def test_chain_rooted_at_plural_subject_pronominalises_with_their():
    """A plural-subject scope (e.g. an aggregation source population) yields *"their …"*."""
    rid = uuid.uuid4()
    chain = PossessiveChain(
        parts=[_attr_part("battery")],
        root_fragment=NounPhrase(head=_noun("Robot"), referent_id=rid),
        root_referent_id=rid,
    )
    intro = NounPhrase(
        head=_noun("Robot"), number=Number.PLURAL, referent_id=rid
    )  # "Robots" (bare plural population intro)
    # The pass derives "their" from the plural population intro (referent_id == focus) it walks
    # before the chain — no number is supplied by the scope.
    scoped, discourse = _scope(
        PhraseFragment(parts=[intro, chain]), focus_id=rid
    )
    # _realise runs coreference + determiner only; the head inflects to "Robots" later in the
    # morphology pass (the full pipeline is pinned by test_deeply_nested_subqueries_golden).
    assert _realise(scoped, discourse) == "Robot their battery"
