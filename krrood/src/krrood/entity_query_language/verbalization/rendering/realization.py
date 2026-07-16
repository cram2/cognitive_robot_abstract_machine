from __future__ import annotations

import uuid
from typing_extensions import Iterable, List, Mapping, Optional

from krrood.entity_query_language.verbalization.fragments.base import (
    flatten_fragment_to_plain_text,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.rendering.coreference_processor import (
    CoreferenceProcessor,
)
from krrood.entity_query_language.verbalization.rendering.discourse import (
    DiscourseView,
    EMPTY_DISCOURSE,
)
from krrood.entity_query_language.verbalization.rendering.determiner_processor import (
    DeterminerProcessor,
)
from krrood.entity_query_language.verbalization.rendering.morphology_processor import (
    MorphologyProcessor,
)
from krrood.entity_query_language.verbalization.rendering.orthography_processor import (
    OrthographyProcessor,
)
from krrood.entity_query_language.verbalization.rendering.passes import RealizationPass

# The stateless lowering passes are shared module-level instances, in pipeline order. The
# coreference pass is stateful per walk (parameterised by discourse and the prior-build referents),
# so it is created fresh per call and prepended to the pipeline in realize_tree.
_LOWERING_PASSES: List[RealizationPass] = [
    DeterminerProcessor(),
    MorphologyProcessor(),
    OrthographyProcessor(),
]


def realize_tree(
    fragment: VerbalizationFragment,
    previously_introduced_referents: Optional[Iterable[uuid.UUID]] = None,
    discourse: DiscourseView = EMPTY_DISCOURSE,
    numbered_labels: Optional[Mapping[uuid.UUID, str]] = None,
) -> VerbalizationFragment:
    """
    Run the ordered realisation passes over *fragment* — the one place the lowering passes and
    their order are defined: coreference resolution → determiner lowering → morphology →
    orthography (punctuation spacing). Both the whole-expression build and the local realisation
    of an opaque template need this same ordered sequence.

    Reference: :cite:t:`gatt2009simplenlg` — the ordered realisation stages.

    :param fragment: Root of the fragment tree.
    :param previously_introduced_referents: Referents introduced by prior builds on a shared context.
    :param discourse: The focus-per-scope view the coreference pass consults (empty for a local
        sub-tree, which has no query scope of its own).
    :param numbered_labels: Disambiguation numbers for referents the rules cannot label themselves
        (relational referents) — applied by the coreference pass.
    :return: The fully realised fragment tree.

    This is the pass-running step: it returns a lowered fragment *tree*, so the example wraps it in
    :func:`flatten_fragment_to_plain_text` to read the text out; :func:`realize_subtree` runs the
    same passes and returns that plain string directly.

    >>> from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
    >>> tree = EQLVerbalizer().build(a(entity(variable(Robot, []))))
    >>> flatten_fragment_to_plain_text(realize_tree(tree))
    'Find a Robot'
    """
    pipeline: List[RealizationPass] = [
        CoreferenceProcessor(
            discourse=discourse,
            numbered_labels=dict(numbered_labels or {}),
            previously_introduced_referents=tuple(
                previously_introduced_referents or ()
            ),
        ),
        *_LOWERING_PASSES,
    ]
    realised = fragment
    for realisation_pass in pipeline:
        realised = realisation_pass.process(realised)
    return realised


def realize_subtree(fragment: VerbalizationFragment) -> str:
    """
    Fully realise a sub-tree to plain text — the realisation passes, then flatten.

    For an opaque leaf (a user template that string-formats its children), the children must be
    realised here, locally, rather than deferred to the global passes.

    :param fragment: Root of the sub-tree.
    :return: The realised plain-text string.

    Its contribution over :func:`realize_tree` is the final flatten: it returns the plain *string*
    *Find a Robot*, not a fragment tree — the form an opaque template needs for its locally realised
    children.

    >>> from krrood.entity_query_language.verbalization.verbalizer import EQLVerbalizer
    >>> realize_subtree(EQLVerbalizer().build(a(entity(variable(Robot, [])))))
    'Find a Robot'
    """
    return flatten_fragment_to_plain_text(realize_tree(fragment))
