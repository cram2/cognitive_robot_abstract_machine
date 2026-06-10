"""
Chain **assembler** — realise a
:class:`~krrood.entity_query_language.core.mapped_variable.MappedVariable` chain
(Attribute / Index / Call) into its surface phrase.

Chains have no separate *plan*: there is no content selection to decide, only the
surface form to render (a boolean terminal attribute → predicative *"<nav> is [not]
<attr>"*; anything else → the possessive path *"the attr of the Root"*, optionally
pronominalised to *"its …"* when the root is the current coreference subject).  It is
therefore a realisation-only :class:`Assembler` (``Assembler[None]``); its sub-steps are
methods sharing ``self.ctx`` (recursion via ``self.ctx.child``).  Entity-rooted chains
defer to :meth:`QueryAssembler.inline_noun`.

Reference: Gatt & Reiter (2009), SimpleNLG — surface realisation.
"""

from __future__ import annotations

from typing_extensions import List, Optional

from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.query.quantifiers import ResultQuantifier
from krrood.entity_query_language.query.query import Entity
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.verbalization.chain_utils import (
    build_path_parts,
    walk_chain,
)
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    PossessiveChain,
    RoleFragment,
    VerbFragment,
)
from krrood.entity_query_language.verbalization.fragments.factory import phrase, word
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    Number,
)
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole
from krrood.entity_query_language.verbalization.grammar.assembly.base import Assembler
from krrood.entity_query_language.verbalization.grammar.assembly.query import (
    QueryAssembler,
)
from krrood.entity_query_language.verbalization.grammar.conditions.recognition import (
    is_bool_attr_chain,
)
from krrood.entity_query_language.verbalization.rendering.possessive import (
    possessive_path,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Copulas,
    Prepositions,
)


class ChainAssembler(Assembler[MappedVariable, None]):
    """Realise a MappedVariable chain (possessive / predicative / pronominal forms).

    Realisation-only (``planner = None``): a chain has no content to decide, only a surface
    form, so there is no plan — :meth:`realize` ignores it.
    """

    def realize(self, node, plan: None = None) -> VerbFragment:
        """Default chain rendering (the :class:`Assembler` entry point)."""
        return self.chain(node)

    # ── entry forms ──────────────────────────────────────────────────────────

    def chain(
        self, expression: MappedVariable, *, negated: bool = False
    ) -> VerbFragment:
        """Boolean terminal → predicative *"<nav> is [not] <attr>"*; else possessive path.

        When a plural is requested (``ctx.number``) and this is a single attribute on a
        variable, build the bare plural *"attrs of Roots"*; otherwise render singular.
        """
        if self.ctx.number is Number.PLURAL:
            plural = self._plural_attribute(expression)
            if plural is not None:
                return plural
        chain, leaf = walk_chain(expression)
        if is_bool_attr_chain(expression):
            return self._bool_predicative(chain, leaf, negated)
        parts = build_path_parts(chain)
        root_fragment = self._chain_root(leaf)
        if isinstance(leaf, Variable):
            # Defer the pronominal-vs-possessive choice to the coreference pass: it knows
            # whether the root is the current subject (a build-time fact no longer consulted here).
            return PossessiveChain(
                parts=parts, root_fragment=root_fragment, root_referent_id=leaf._id_
            )
        return possessive_path(parts, root_fragment)

    def _plural_attribute(self, expression: MappedVariable) -> Optional[VerbFragment]:
        """*"attrs of Roots"* when *expression* is a single ``Attribute`` on a ``Variable``,
        else ``None`` (caller falls through to the singular rendering).  Tags both leaves
        plural for the morphology pass; marks the root introduced for cross-build seeding.
        """
        chain, root = walk_chain(expression)
        if not (
            isinstance(root, Variable)
            and len(chain) == 1
            and isinstance(chain[0], Attribute)
        ):
            return None
        type_name = root._type_.__name__
        label = self.ctx.refer.disambiguation_map.get(root._id_, type_name)
        self.ctx.refer.mark_introduced(root)
        numbered = label != type_name
        attribute = chain[0]
        root_np = NounPhrase(
            head=RoleFragment.for_variable(label, root),
            number=Number.SINGULAR if numbered else Number.PLURAL,
            definiteness=Definiteness.BARE if numbered else Definiteness.INDEFINITE,
            referent_id=root._id_,
        )
        return NounPhrase(
            head=RoleFragment.for_attribute(
                attribute._owner_class_, attribute._attribute_name_
            ),
            number=Number.PLURAL,
            definiteness=Definiteness.INDEFINITE,
            modifiers=[Prepositions.OF.as_fragment(), root_np],
        )

    def _chain_root(self, leaf: object) -> VerbFragment:
        """Noun phrase for the chain root — inline-noun for Entity roots, else recurse."""
        inner = leaf
        while isinstance(inner, ResultQuantifier):
            inner = inner._child_
        if isinstance(inner, Entity):
            return QueryAssembler(self.ctx).inline_noun(inner)
        return self.ctx.child(leaf)

    def _bool_predicative(
        self, chain: List[MappedVariable], leaf: object, negated: bool
    ) -> VerbFragment:
        """*"<nav> is [not] <attr>"* — chains ending in an int Index get ordinal navigation."""
        root_fragment = self._chain_root(leaf)
        nav_chain = chain[:-1]

        if not nav_chain:
            nav_fragment = root_fragment
        elif isinstance(nav_chain[-1], Index) and isinstance(nav_chain[-1]._key_, int):
            ordinal = morphology.ordinal(nav_chain[-1]._key_)
            pre_frag = possessive_path(build_path_parts(nav_chain[:-1]), root_fragment)
            nav_fragment = phrase(
                Articles.THE.as_fragment(),
                word(ordinal),
                Prepositions.OF.as_fragment(),
                pre_frag,
            )
        else:
            nav_fragment = possessive_path(build_path_parts(nav_chain), root_fragment)

        copula = Copulas.IS_NOT.as_fragment() if negated else Copulas.IS.as_fragment()
        terminal = chain[-1]
        return phrase(
            nav_fragment,
            copula,
            RoleFragment.for_attribute(
                terminal._owner_class_, terminal._attribute_name_
            ),
        )
