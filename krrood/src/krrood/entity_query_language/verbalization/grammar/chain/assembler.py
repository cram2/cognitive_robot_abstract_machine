from __future__ import annotations

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import (
    MappedVariable,
)
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.verbalization.fragments.base import (
    NounPhrase,
    PhraseFragment,
    PossessiveChain,
    RoleFragment,
    VerbalizationFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.chain.planner import (
    ChainPlan,
    ChainPlanner,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    possessive_path,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Copulas,
    Logicals,
    Prepositions,
)


class ChainAssembler(Assembler[MappedVariable, ChainPlan]):
    """Render a ``MappedVariable`` chain into one of its surface forms.

    The :class:`ChainPlanner` decides the chain's content — its root, display path-parts, and
    whether it ends in a boolean attribute. *Which* surface form to use is no longer decided here:
    that choice lives in the grammar as guarded ``MappedVariable`` rules (plural-attribute /
    boolean-predicative / possessive). This assembler is the shared rendering toolkit those rules
    call: each form is a public method, and :meth:`realize` is the unguarded possessive default.

    The forms are the possessive path *"the attribute of the Root"* (optionally pronominalised to
    *"its …"* when the root is the current coreference subject), the predicative *"<navigation> is
    [not] <attribute>"* for a boolean terminal, and the bare plural *"attributes of Roots"*.

    >>> verbalize_expression(variable(Robot, []).battery)
    'the battery of a Robot'

    Reference: :cite:t:`gatt2009simplenlg` — surface realisation.
    """

    planner = ChainPlanner

    def realize(self, node: MappedVariable, plan: ChainPlan) -> VerbalizationFragment:
        """
        :param node: The chain to render.
        :param plan: The chain plan computed for *node*.
        :return: The possessive rendering — the unguarded default form.
        """
        return self.possessive(plan)

    # %% surface forms

    def possessive(self, plan: ChainPlan) -> VerbalizationFragment:
        """
        :param plan: The analysed chain.
        :return: The possessive path *"the attribute of the Root"*; for a variable root, deferred to
            the coreference pass as a :class:`PossessiveChain` (it knows whether the root is the
            current discourse subject, for *"its …"*).

        >>> verbalize_expression(variable(Task, []).name)
        'the name of a Task'
        """
        root_fragment = self._chain_root(plan.root)
        if isinstance(plan.root, Variable):
            return PossessiveChain(
                parts=plan.parts,
                root_fragment=root_fragment,
                root_referent_id=plan.root._id_,
                node_id=plan.chain[-1]._id_,
            )
        return possessive_path(plan.parts, root_fragment)

    def plural_attribute(self, plan: ChainPlan) -> VerbalizationFragment:
        """
        :param plan: The analysed chain (a single attribute on a variable — see
            :attr:`ChainPlan.is_single_variable_attribute`).
        :return: The bare plural *"attributes of Roots"*.

        >>> verbalize_expression(sum(variable(Robot, []).battery))
        'the sum of batteries of Robots'
        """
        attribute = plan.chain[0]
        # The root's plural noun phrase ("Robots" / "Robot 2") is the variable rule's job; recurse
        # for it rather than rebuilding its number/definiteness/label here.
        root_noun_phrase = self.context.child(
            plan.root, number=GrammaticalNumber.PLURAL
        )
        return NounPhrase(
            head=RoleFragment.for_attribute(
                attribute._owner_class_, attribute._attribute_name_
            ),
            number=GrammaticalNumber.PLURAL,
            definiteness=Definiteness.INDEFINITE,
            modifiers=[Prepositions.OF.as_fragment(), root_noun_phrase],
        )

    def _chain_root(self, root: SymbolicExpression) -> VerbalizationFragment:
        """
        :param root: The chain root.
        :return: The noun phrase for the chain root. Recursed in ``inline`` position, so an entity
            root is dispatched to the inline-noun form and anything else renders normally — the
            chain assembler doesn't decide which, nor call the query assembler.

        >>> verbalize_expression(variable(Mission, []).assigned_to)
        'the Robot to which a Mission is assigned'
        """
        return self.context.child(root, inline=True)

    def boolean_predicative(
        self, plan: ChainPlan, negated: bool = False
    ) -> VerbalizationFragment:
        """
        :param plan: The analysed chain (a boolean terminal — see
            :attr:`ChainPlan.is_boolean_terminal`).
        :param negated: Whether to negate the predicative.
        :return: The predicative *"<navigation> is [not] <attribute>"*.

        >>> verbalize_expression(variable(Robot, []).operational)
        'a Robot is operational'
        """
        navigation_fragment, attribute_fragment = self._boolean_parts(plan)
        copula = Copulas.IS_NOT.as_fragment() if negated else Copulas.IS.as_fragment()
        return PhraseFragment(parts=[navigation_fragment, copula, attribute_fragment])

    def boolean_alternative(self, plan: ChainPlan) -> VerbalizationFragment:
        """
        :param plan: The analysed boolean-terminal chain.
        :return: The unconstrained-boolean predicative *"<navigation> is either <attribute> or not"* —
            for a boolean attribute compared to a domain holding both ``True`` and ``False`` (the
            value is left open, so neither polarity is asserted).

        >>> verbalize_expression(variable(Task, []).completed == variable(bool, [True, False]))
        'a Task is either completed or not'
        """
        navigation_fragment, attribute_fragment = self._boolean_parts(plan)
        return PhraseFragment(
            parts=[
                navigation_fragment,
                Copulas.IS.as_fragment(),
                Logicals.EITHER.as_fragment(),
                attribute_fragment,
                Conjunctions.OR.as_fragment(),
                Logicals.NOT.as_fragment(),
            ]
        )

    def _boolean_parts(self, plan: ChainPlan) -> tuple:
        """:return: ``(navigation_fragment, attribute_fragment)`` for a boolean-terminal chain — the
        navigation prefix and the terminal attribute noun, shared by the predicative and the
        open-alternative forms.

        The prefix is the chain minus its boolean terminal, recursed through the standard grammar
        (``terminal._child_``) rather than re-rendered here: it is itself a navigable referent, so
        the recursion supplies its surface form (root *"a Mission"*, ordinal *"the first of …"*,
        relational *"the Robot to which a Mission is assigned"*) and — crucially — pronominalises it
        to the discourse subject (*"the Robot to which it is assigned"*) when one is in scope.

        >>> verbalize_expression(variable(Worker, []).tasks[0].completed)
        'the first of the tasks of a Worker is completed'
        """
        terminal = plan.chain[-1]
        navigation_fragment = self.context.child(terminal._child_, inline=True)
        return navigation_fragment, RoleFragment.for_attribute(
            terminal._owner_class_, terminal._attribute_name_
        )
