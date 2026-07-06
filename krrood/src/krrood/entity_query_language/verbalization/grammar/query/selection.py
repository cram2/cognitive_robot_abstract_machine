"""
Rendering a query's *selection* — the variables / columns it selects — into prose.

A selection is said differently by shape: natural Oxford-comma prose *"a, b, and c"* (no code-like
brackets), a parenthesised tuple *"(a, b)"* for a ranked set-of whose *"top three"* pre-head needs
the tuple grouped, or a plural population *"Employees"* for an ordered report. Contiguous attributes
of one owner fold into a shared genitive (*"the department and salary of an Employee"*) via the
shared :func:`~…coordination.group_consecutive_by_owner` aggregation primitive.

Split out of :class:`~…query.assembler.QueryAssembler` so the selection-rendering responsibility is
its own cohesive collaborator (the assembler delegates to it).
"""

from __future__ import annotations

import uuid
from dataclasses import dataclass

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.verbalization.fragments.base import (
    VerbalizationFragment,
    NounPhrase,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
    Separator,
)
from krrood.entity_query_language.verbalization.grammar.chain.planner import (
    ChainPlanner,
)
from krrood.entity_query_language.verbalization.grammar.framework.phrase_rule import (
    RuleContext,
)
from krrood.entity_query_language.verbalization.microplanning.coordination import (
    group_consecutive_by_owner,
    OwnerGroup,
)
from krrood.entity_query_language.verbalization.microplanning.possessive import (
    attribute_fragment,
    coordinated_genitive,
)
from krrood.entity_query_language.verbalization.navigation_path import PathStep
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Conjunctions,
    Punctuation,
)


def subject_referent_id(variable: SymbolicExpression) -> Optional[uuid.UUID]:
    """:return: The referent id for a subject/selection variable (``None`` when *variable* is not a
    single variable, which suppresses pronominalisation).

    >>> robot = variable(Robot, [])
    >>> subject_referent_id(robot) == robot._id_
    True
    >>> subject_referent_id(robot.battery) is None
    True
    """
    return variable._id_ if isinstance(variable, Variable) else None


@dataclass
class SelectionAssembler:
    """Render a query's selected variables / columns into prose — the single owner of *how a
    selection is said* (natural prose, parenthesised tuple, plural population, co-owned genitive).

    >>> employee = variable(Employee, [])
    >>> verbalize_expression(a(set_of(employee.department, employee.name)))
    'Find the department and name of an Employee'
    """

    context: RuleContext
    """The per-node context (recursion entry and microplanning services)."""

    def prose(
        self,
        variables: List[SymbolicExpression],
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
    ) -> VerbalizationFragment:
        """:return: the selections joined as natural prose *"a, b, and c"* (Oxford comma, no
        parentheses). A plural *number* lists them as populations (*"Employees"*) for an ordered
        report; otherwise contiguous attributes of one owner fold into a shared genitive (*"the
        department and salary of an Employee"*).

        This is the natural-prose path (as opposed to :meth:`parenthesised`): the two attributes share
        the Employee owner, so it emits the comma-free shared genitive *the department and name of an
        Employee* rather than a parenthesised tuple.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, employee.name)))
        'Find the department and name of an Employee'
        """
        if number is GrammaticalNumber.PLURAL:
            selections = [self.one(variable, number) for variable in variables]
        else:
            selections = self._folded(variables)
        return oxford_comma(selections, Conjunctions.AND.as_fragment())

    def one(
        self, variable: SymbolicExpression, number: GrammaticalNumber
    ) -> VerbalizationFragment:
        """:return: a single selection, as a bare plural population (*"Employees"*) when *number* is
        plural and the selection is a variable, else its default referring form.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee)).ordered_by(employee.salary, descending=True))
        'Report Employees ordered by their salaries from highest to lowest'
        """
        if number is GrammaticalNumber.PLURAL and isinstance(variable, Variable):
            return NounPhrase(
                head=RoleFragment.for_variable(
                    variable._type_.__name__, variable, number=GrammaticalNumber.PLURAL
                ),
                number=GrammaticalNumber.PLURAL,
                definiteness=Definiteness.INDEFINITE,
                referent_id=subject_referent_id(variable),
            )
        return self.context.child(variable)

    def parenthesised(
        self, variables: List[SymbolicExpression]
    ) -> VerbalizationFragment:
        """:return: the selections as a parenthesised tuple *"(a, b)"* — for a ranked set-of, whose
        *"the top three"* pre-head needs the tuple grouped.

        >>> employee = variable(Employee, [])
        >>> robot = variable(Robot, [])
        >>> verbalize_expression(a(set_of(employee.name, robot.name)).ordered_by(
        ...     employee.salary, descending=True).limit(2))
        'Find the top two (the name of an Employee, the name of a Robot)'
        """
        tuple_phrase = PhraseFragment(
            parts=[self.context.child(variable) for variable in variables],
            separator=Separator.COMMA,
        )
        return PhraseFragment(
            parts=[
                Punctuation.OPEN_PAREN.as_fragment(),
                tuple_phrase,
                Punctuation.CLOSE_PAREN.as_fragment(),
            ]
        )

    def _folded(
        self, variables: List[SymbolicExpression]
    ) -> List[VerbalizationFragment]:
        """:return: the rendered selections, with each maximal run of plain attributes sharing one
        owner folded into a single coordinated genitive, and every other selection rendered alone.

        It performs the fold itself: *department* and *salary* share the Employee owner, so instead of
        *the department of an Employee and the salary of an Employee* it emits the single coordinated
        genitive *the department and salary of an Employee*:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(the(set_of(employee.department, employee.salary)))
        'Find the department and salary of an Employee'
        """
        fragments: List[VerbalizationFragment] = []
        for item in group_consecutive_by_owner(variables, self._foldable_attribute):
            if isinstance(item, OwnerGroup):
                fragments.append(
                    coordinated_genitive(
                        [attribute_fragment(terminal) for terminal in item.items],
                        self.context.child(item.owner, inline=True),
                    )
                )
            else:
                fragments.append(self.one(item, GrammaticalNumber.SINGULAR))
        return fragments

    def _foldable_attribute(
        self, selection: SymbolicExpression
    ) -> Optional[Tuple[SymbolicExpression, PathStep]]:
        """:return: ``(owner, terminal_step)`` when *selection* is a plain genitive attribute that
        can share an owner with siblings, else ``None`` — a relational terminal (*"the Robot to
        which …"*) does not coordinate cleanly, so it is left alone.

        It is the predicate :meth:`_folded` consults: it accepts *department* and *salary* (returning
        their shared Employee owner), which is what lets them collapse into *the department and salary
        of an Employee*; returning ``None`` would keep them as separate genitives.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(the(set_of(employee.department, employee.salary)))
        'Find the department and salary of an Employee'
        """
        if not isinstance(selection, Attribute):
            return None
        plan = self.context.microplan.plan_for(selection, ChainPlanner)
        if not plan.parts or plan.parts[-1].is_relation:
            return None
        return selection._child_, plan.parts[-1]
