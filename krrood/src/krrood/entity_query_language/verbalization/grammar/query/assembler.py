from __future__ import annotations

from dataclasses import replace

from typing_extensions import List, Optional, Tuple

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.expression_structure import walk_chain
from krrood.entity_query_language.core.mapped_variable import Attribute
from krrood.entity_query_language.core.variable import Variable
from krrood.entity_query_language.operators.aggregators import Aggregator
from krrood.entity_query_language.verbalization import morphology
from krrood.entity_query_language.query.query import Query, SetOf
from krrood.entity_query_language.verbalization.fragments.base import (
    BlockFragment,
    NounPhrase,
    oxford_comma,
    PhraseFragment,
    RoleFragment,
    VerbalizationFragment,
    WordFragment,
)
from krrood.entity_query_language.verbalization.fragments.features import (
    Definiteness,
    GrammaticalNumber,
)
from krrood.entity_query_language.verbalization.grammar.aggregation.assembler import (
    AggregationValueAssembler,
)
from krrood.entity_query_language.verbalization.grammar.framework.assembler import (
    Assembler,
)
from krrood.entity_query_language.verbalization.grammar.clauses.composer import (
    ClauseComposer,
)
from krrood.entity_query_language.verbalization.grammar.query.planner import (
    QueryPlan,
    QueryPlanner,
    SortDirection,
    RankingKeyRelation,
    ReportKind,
    ReportPlan,
    SelectionKind,
)
from krrood.entity_query_language.verbalization.grammar.query.ranking import (
    ranking_surface,
    RankingRequest,
)
from krrood.entity_query_language.verbalization.grammar.query.selection import (
    SelectionAssembler,
    subject_referent_id,
)
from krrood.entity_query_language.verbalization.microplanning.referring import (
    referring_noun_with_restrictions,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles,
    Conjunctions,
    FallbackNouns,
    GroupingPhrases,
    Keywords,
    Prepositions,
    Punctuation,
    RankingWords,
)


class QueryAssembler(Assembler[Query, QueryPlan]):
    """
    Realise a query / nested-entity / set-of from its query plan into the *"Find … such that …
    grouped by … having … ordered by …"* block (and the nested noun-phrase form).

    It dispatches on the selection shape, builds the selection and its restrictions, and combines
    the trailing clauses. Coreference is resolved later: the assembler emits referring noun
    phrases and subject-scope markers, and a document-order pass decides first/subsequent/pronoun
    afterwards :cite:p:`reiter2000building`.

    Reference: :cite:t:`gatt2009simplenlg` — surface realisation.

    >>> robot = variable(Robot, [])
    >>> verbalize_expression(an(entity(robot).where(robot.battery > 50)))
    'Find a Robot whose battery is greater than 50'
    """

    planner = QueryPlanner

    # %% entry points

    def realize(self, node: Query, plan: QueryPlan) -> VerbalizationFragment:
        """
        :param node: The query being rendered.
        :param plan: The query plan.
        :return: The top-level imperative form *"Find X such that …"*, dispatched on the
            selection shape.

        It is the dispatch entry: it picks the handler for the plan's selection shape, which for the
        plain-variable subject of the class example assembles the whole *"Find a Robot whose battery
        is greater than 50"*.
        """
        if plan.report is not None and plan.report.kind is ReportKind.GROUPING:
            return self._assemble_grouped_report(node, plan, plan.report)
        handlers = {
            SelectionKind.ENTITY_SELECTOR: self._realize_entity_selector,
            SelectionKind.EMPTY: self._realize_empty,
            SelectionKind.SUBJECT: self._assemble_subject,
        }
        return handlers[plan.kind](node, plan)

    def _realize_entity_selector(
        self, node: Query, plan: QueryPlan
    ) -> VerbalizationFragment:
        """:return: *"Find <a Robot where …> such that …"* — the selected variable is itself an entity.

        It handles the entity-selector shape: the selection is rendered as a noun and wrapped in the
        query body, so here the whole result is just *"Find a Task"*:

        >>> verbalize_expression(an(entity(an(entity(variable(Task, []))))))
        'Find a Task'
        """
        selection = self._as_noun(node.selected_variable)
        return self._query_body(
            node, plan, selection, where_items=[self._where_clause(plan)]
        )

    def _realize_empty(self, node: Query, plan: QueryPlan) -> VerbalizationFragment:
        """:return: *"Find entities such that …"* — no selected variable (the fallback form).

        ..note:: Unreachable through the public API: ``entity()`` always carries a selected
            variable, so a query never plans to :attr:`SelectionKind.EMPTY`.
        """
        return self._query_body(
            node,
            plan,
            FallbackNouns.ENTITY.plural_fragment(),
            where_items=[self._where_clause(plan)],
        )

    def assemble_nested(self, node: Query) -> VerbalizationFragment:
        """
        :param node: The nested entity.
        :return: The noun-phrase form for a nested entity (never emits *"Find …"*).

        In the shown sentence it contributes only the inner noun *"a Task"*; the enclosing *"Find a
        Worker whose tasks contains …"* belongs to the outer top-level query:

        >>> worker = variable(Worker, [])
        >>> verbalize_expression(
        ...     an(entity(worker).where(contains(worker.tasks, an(entity(variable(Task, []))))))
        ... )
        'Find a Worker whose tasks contains a Task'
        """
        plan = self.plan(node)
        if plan.is_aggregation_subquery:
            return AggregationValueAssembler(self.context).realize(node, plan)
        return self._as_noun(node)

    def assemble_set_of(self, node: SetOf) -> VerbalizationFragment:
        """
        :param node: The set-of query.
        :return: *"Find v1 and v2 such that …"* for a search; *"Report <columns>"* /
            *"For each <keys>, report <columns>"* for an aggregation report; *"Report v1 and v2
            ordered by …"* (plural) for an ordered listing.

        It selects the set-of surface from the plan; the plain search shown has no report or ranking,
        so it assembles the whole *"Find a Robot and a Task"*:

        >>> verbalize_expression(an(set_of(variable(Robot, []), variable(Task, []))))
        'Find a Robot and a Task'
        """
        plan = self.plan(node)
        report = plan.report
        if report is not None and plan.ranking is not None:
            return self._assemble_ranked_report(node, plan, report)
        if report is not None and report.kind is ReportKind.GROUPING:
            return self._assemble_grouped_report(node, plan, report)
        if report is not None and report.kind is ReportKind.AGGREGATION:
            return self._assemble_aggregation_report(node, plan, report)
        if plan.ranking is not None:
            return self._assemble_ranked_set_of(node, plan)
        return self._query_body(
            node,
            plan,
            self._set_of_selection(node, plan),
            where_items=[self._where_clause(plan)],
            find_header=self._set_of_header(plan),
        )

    def _assemble_ranked_report(
        self, node: SetOf, plan: QueryPlan, report: ReportPlan
    ) -> VerbalizationFragment:
        """:return: a grouped/aggregated report whose rows are ranked *by an aggregate* — framed as
        *"For the <entity> with the highest <aggregate>, report <columns>"* (singular for
        ``limit(1)``, *"the three <entities> with the highest …"* for ``n > 1``). Naming the entity
        first lets the columns pronominalise to it (*"its period"*) and the aggregate, named once in
        the frame, reduce to *"the sum"* in the body. A ranking by a plain attribute (or a tuple with
        no single root) keeps the attribute-keyed reframe, which already names its basis.
        """
        subject = self._tuple_subject(node, plan)
        aggregate = self._ranked_aggregate_column(node, plan.ranking)
        if subject is None or aggregate is None:
            return self._assemble_ranked_set_of(node, plan)
        number = (
            GrammaticalNumber.PLURAL
            if plan.ranking.limit_number > 1
            else GrammaticalNumber.SINGULAR
        )
        subject_noun = NounPhrase(
            head=RoleFragment.for_variable(
                subject._type_.__name__, subject, number=number
            ),
            number=number,
            definiteness=Definiteness.DEFINITE,
            pre_head=(
                WordFragment(text=morphology.cardinal(plan.ranking.limit_number))
                if plan.ranking.limit_number > 1
                else None
            ),
            modifiers=[
                self._highest_aggregate_modifier(aggregate, plan.ranking.direction)
            ],
            referent_id=subject_referent_id(subject),
        )
        header = PhraseFragment(
            parts=[
                self._sentence_initial(Keywords.FOR.as_fragment()),
                subject_noun,
                Punctuation.COMMA.as_fragment(),
                Keywords.REPORT.as_fragment(),
            ]
        )
        return self._query_body(
            node,
            plan,
            self._selections.prose(node._selected_variables_),
            where_items=[self._where_clause(plan)],
            find_header=header,
        )

    def _ranked_aggregate_column(
        self, node: SetOf, ranking
    ) -> Optional[SymbolicExpression]:
        """:return: the selected aggregate column the query is ranked by (matched to the order key
        structurally), or ``None`` when the order key is not an aggregate or names no column.
        """
        if not isinstance(ranking.order_key, Aggregator):
            return None
        for selection in node._selected_variables_:
            if self._is_order_key(selection, ranking.order_key):
                return selection
        return None

    def _highest_aggregate_modifier(
        self, aggregate: SymbolicExpression, direction: SortDirection
    ) -> VerbalizationFragment:
        """:return: the post-nominal ranking modifier *"with the highest/lowest <aggregate>"* — the
        aggregate's first (full) mention, which the body's repeat reduces to *"the sum"*.
        """
        return PhraseFragment(
            parts=[
                Prepositions.WITH.as_fragment(),
                self._with_superlative(self.context.child(aggregate), direction),
            ]
        )

    def _is_order_key(
        self, selection: SymbolicExpression, order_key: Optional[SymbolicExpression]
    ) -> bool:
        """:return: whether *selection* is the column the query is ordered by — matched
        structurally, not by identity, so a re-stated aggregate (``ordered_by(sum(x))`` for a
        selected ``sum(x)``) still lines up."""
        if order_key is None:
            return False
        if selection._id_ == order_key._id_:
            return True
        return self._expression_signature(selection) == self._expression_signature(
            order_key
        )

    def _expression_signature(self, expression: SymbolicExpression) -> Tuple:
        """:return: a structural key for *expression* — its kind, root variable, and attribute path —
        so two distinct objects describing the same navigation/aggregate compare equal.
        """
        if isinstance(expression, Aggregator):
            kind = type(expression).__name__
            chain, root = walk_chain(expression._chain_expression_)
        else:
            kind = ""
            chain, root = walk_chain(expression)
        root_id = root._id_ if isinstance(root, Variable) else None
        path = tuple((step._attribute_name_, step._owner_class_) for step in chain)
        return (kind, root_id, path)

    def _with_superlative(
        self, fragment: VerbalizationFragment, direction: SortDirection
    ) -> VerbalizationFragment:
        """:return: *fragment* qualified by the ranking superlative — *"the highest …"* (descending) /
        *"the lowest …"* (ascending) — attached as the noun's pre-head so it reads *"the highest sum
        of …"*."""
        quality = (
            RankingWords.LOWEST
            if direction is SortDirection.ASCENDING
            else RankingWords.HIGHEST
        )
        if isinstance(fragment, NounPhrase):
            return replace(fragment, pre_head=quality.as_fragment())
        return PhraseFragment(parts=[quality.as_fragment(), fragment])

    def _set_of_selection(self, node: SetOf, plan: QueryPlan) -> VerbalizationFragment:
        """:return: the set-of's rendered selection — a plural listing for an ordered report, else
        natural Oxford-comma prose.

        It produces the selection span after *"Find"*; this set-of has no report, so it picks the
        coordinated prose form *"the department and salary of an Employee"*:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, employee.salary)))
        'Find the department and salary of an Employee'
        """
        if plan.report is not None:
            return self._selections.prose(
                node._selected_variables_, number=GrammaticalNumber.PLURAL
            )
        return self._selections.prose(node._selected_variables_)

    def _assemble_ranked_set_of(
        self, node: SetOf, plan: QueryPlan
    ) -> VerbalizationFragment:
        """:return: a ranked set-of reframed onto the entity its columns describe — *"Report, for the
        top three Employees by salary, their department and name"* — so the tuple reads as a
        possessive listing instead of a code-like *"(a, b)"*. The ranking attaches to the subject
        noun (its order key named when it is an attribute of the subject), and the columns
        pronominalise to it. A set-of whose columns share no single root keeps the bracketed tuple.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, employee.name).ordered_by(
        ...     employee.salary, descending=True).limit(3)))
        'Report, for the top three Employees by salary, their departments and their names'
        """
        subject = self._tuple_subject(node, plan)
        if subject is None:
            return self._query_body(
                node,
                plan,
                self._selections.parenthesised(node._selected_variables_),
                where_items=[self._where_clause(plan)],
                find_header=self._set_of_header(plan),
            )
        ranking = replace(
            plan.ranking,
            relation=self._key_relation_to(subject, plan.ranking.order_key),
        )
        surface = ranking_surface(RankingRequest(plan=ranking))
        subject_noun = NounPhrase(
            head=RoleFragment.for_variable(
                subject._type_.__name__, subject, number=surface.number
            ),
            number=surface.number,
            definiteness=Definiteness.DEFINITE,
            pre_head=surface.pre_head,
            modifiers=surface.modifiers,
            referent_id=subject_referent_id(subject),
        )
        columns = self._ranked_columns(node, ranking)
        if not columns:
            return self._query_body(
                node,
                plan,
                subject_noun,
                where_items=[self._where_clause(plan)],
                find_header=self._sentence_initial(Keywords.REPORT.as_fragment()),
            )
        header = PhraseFragment(
            parts=[
                self._sentence_initial(Keywords.REPORT.as_fragment()),
                Punctuation.COMMA.as_fragment(),
                Keywords.FOR.as_fragment(),
                subject_noun,
                Punctuation.COMMA.as_fragment(),
            ]
        )
        # Each column is rendered on its own (not owner-folded) so it pronominalises to the
        # already-introduced subject ("their department and their name"), agreeing in number.
        column_list = oxford_comma(
            [self.context.child(column) for column in columns],
            Conjunctions.AND.as_fragment(),
        )
        return self._query_body(
            node,
            plan,
            column_list,
            where_items=[self._where_clause(plan)],
            find_header=header,
        )

    def _tuple_subject(self, node: SetOf, plan: QueryPlan) -> Optional[Variable]:
        """:return: the single root variable every selected column navigates from (the entity the
        ranking reframes onto), or ``None`` when the columns share no single root.

        Here it returns the shared Employee root; that is the entity the ranking reframes onto, which
        is why the surface fronts *"the top three Employees"* and the columns pronominalise to
        *"their"*:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, employee.name).ordered_by(
        ...     employee.salary, descending=True).limit(3)))
        'Report, for the top three Employees by salary, their departments and their names'
        """
        if plan.discourse_root is None:
            return None
        for selection in node._selected_variables_:
            _, root = walk_chain(selection)
            if isinstance(root, Variable) and root._id_ == plan.discourse_root:
                return root
        return None

    def _key_relation_to(
        self, subject: Variable, order_key: Optional[SymbolicExpression]
    ) -> RankingKeyRelation:
        """:return: how *order_key* relates to *subject* — ``SELF`` (it is the subject), ``ATTRIBUTE``
        (a chain on it, so the ranking reads *"by <attribute>"*), or ``OTHER`` (a different root, e.g.
        an aggregate — the key is then left to the visible column).

        This is the decision that shapes the ranking phrase: here *salary* is a chain on the Employee
        subject, so it returns ``ATTRIBUTE`` and the surface names the key as *"by salary"*:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, employee.name).ordered_by(
        ...     employee.salary, descending=True).limit(3)))
        'Report, for the top three Employees by salary, their departments and their names'
        """
        if order_key is None:
            return RankingKeyRelation.SELF
        chain, root = walk_chain(order_key)
        if not isinstance(root, Variable) or root._id_ != subject._id_:
            return RankingKeyRelation.OTHER
        return RankingKeyRelation.ATTRIBUTE if chain else RankingKeyRelation.SELF

    def _ranked_columns(self, node: SetOf, ranking) -> List[SymbolicExpression]:
        """:return: the reported columns — the selected tuple, with the order key removed when the
        ranking already names it (*"by salary"*), so it is not listed twice.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, employee.name, employee.salary
        ...     ).ordered_by(employee.salary, descending=True).limit(3)))
        'Report, for the top three Employees by salary, their departments and their names'
        """
        selected = list(node._selected_variables_)
        names_key = ranking.relation in (
            RankingKeyRelation.ATTRIBUTE,
            RankingKeyRelation.SELF,
        )
        if names_key and ranking.order_key is not None:
            selected = [s for s in selected if s._id_ != ranking.order_key._id_]
        return selected

    @property
    def _selections(self) -> SelectionAssembler:
        """:return: The selection-rendering collaborator for this node's context.

        It adds no words of its own: it merely supplies the collaborator that renders the selection
        *"a Robot and a Task"* shown after *"Find"*:

        >>> verbalize_expression(an(set_of(variable(Robot, []), variable(Task, []))))
        'Find a Robot and a Task'
        """
        return SelectionAssembler(self.context)

    def _assemble_aggregation_report(
        self, node: SetOf, plan: QueryPlan, report: ReportPlan
    ) -> VerbalizationFragment:
        """:return: a calculation/report — *"Report <columns>"*, or *"For each <keys>, report
        <columns>"* when grouped (the grouping stated first, so it frames the whole report and the
        trailing *"grouped by"* clause is dropped as redundant).

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, sum(employee.salary)).grouped_by(
        ...     employee.department)))
        'For each department, report the sum of salaries of Employees'
        """
        header = (
            self._for_each_header(report.group_keys, node)
            if report.is_grouped
            else self._sentence_initial(Keywords.REPORT.as_fragment())
        )
        return self._query_body(
            node,
            plan,
            self._selections.prose(report.columns),
            where_items=[self._where_clause(plan)],
            find_header=header,
        )

    def _assemble_grouped_report(
        self, node: Query, plan: QueryPlan, report: ReportPlan
    ) -> VerbalizationFragment:
        """:return: a grouped report with no aggregates — *"For each <keys>, report all <columns>"*
        (the columns listed as per-group populations), or *"Report the distinct <keys>"* when the
        selection is exactly the group key (so there is nothing left to report but the keys
        themselves). The trailing *"grouped by"* clause is dropped as redundant.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department).grouped_by(employee.department)))
        'Report the distinct departments'
        """
        if not report.columns:
            return self._query_body(
                node,
                plan,
                self._distinct_keys(report.group_keys),
                where_items=[self._where_clause(plan)],
                find_header=self._sentence_initial(Keywords.REPORT.as_fragment()),
            )
        selection = PhraseFragment(
            parts=[
                GroupingPhrases.ALL.as_fragment(),
                self._selections.prose(report.columns, number=GrammaticalNumber.PLURAL),
            ]
        )
        return self._query_body(
            node,
            plan,
            selection,
            where_items=[self._where_clause(plan)],
            find_header=self._for_each_header(report.group_keys, node),
        )

    def _distinct_keys(self, keys: List[SymbolicExpression]) -> VerbalizationFragment:
        """:return: *"the distinct <keys>"* — the group keys as a plural population listing, for a
        grouped query that reports nothing but its keys.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department).grouped_by(employee.department)))
        'Report the distinct departments'
        """
        labels = oxford_comma(
            [self._group_label(key, GrammaticalNumber.PLURAL) for key in keys],
            Conjunctions.AND.as_fragment(),
        )
        return PhraseFragment(
            parts=[
                Articles.THE.as_fragment(),
                GroupingPhrases.DISTINCT.as_fragment(),
                labels,
            ]
        )

    def _for_each_header(
        self, keys: List[SymbolicExpression], node: Query
    ) -> VerbalizationFragment:
        """:return: the fronted *"For each <key>, report"* frame — the grouping first (it is the row
        dimension of the result), the keys as bare singular labels, then the lowercase verb. A HAVING
        filter is woven onto the key as a *"whose <aggregate> is …"* group restriction.

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, sum(employee.salary)).grouped_by(
        ...     employee.department)))
        'For each department, report the sum of salaries of Employees'
        """
        labels = oxford_comma(
            [self._group_label(key) for key in keys], Conjunctions.AND.as_fragment()
        )
        having = node._having_expression_
        key_phrase = labels
        if having is not None:
            with self.context.configuration.possessive_aggregate_scope():
                having_fragment = self.context.child(having.condition)
            key_phrase = PhraseFragment(
                parts=[labels, Keywords.WHOSE.as_fragment(), having_fragment]
            )
        return PhraseFragment(
            parts=[
                Keywords.FOR_EACH.as_fragment(),
                key_phrase,
                Punctuation.COMMA.as_fragment(),
                Keywords.REPORT.as_fragment(),
            ]
        )

    def _group_label(
        self,
        key: SymbolicExpression,
        number: GrammaticalNumber = GrammaticalNumber.SINGULAR,
    ) -> VerbalizationFragment:
        """:return: a group key as a bare label in *number* — *"department"* / *"departments"* for an
        attribute key, the type name for a variable key — naming the group itself rather than one
        member's navigation (*"the department of an Employee"*).

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(employee.department, sum(employee.salary)).grouped_by(
        ...     employee.department)))
        'For each department, report the sum of salaries of Employees'
        """
        if isinstance(key, Attribute):
            return RoleFragment.for_attribute(
                key._owner_class_, key._attribute_name_, number=number
            )
        return RoleFragment.for_type(key._type_, number=number)

    @staticmethod
    def _sentence_initial(fragment: RoleFragment) -> VerbalizationFragment:
        """:return: *fragment* with its first letter capitalised — a keyword carries its mid-sentence
        (lowercase) form, capitalised here when it opens the sentence (*"report"* → *"Report"*).

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(a(set_of(sum(employee.salary))))
        'Report the sum of salaries of Employees'
        """
        return replace(fragment, text=fragment.text[:1].upper() + fragment.text[1:])

    def _set_of_header(self, plan: QueryPlan) -> VerbalizationFragment:
        """:return: The set-of header — *"Find the <top three>"* when a ``limit`` ranks the tuples
        (the order key is suppressed; it is a visible tuple element), else the plain verb
        (*"Find"* / *"Report"*). The literal *"sets of"* is intentionally dropped — the selection
        carries the set shape.

        It emits only the leading *"Find the bottom three"* header; the parenthesised tuple after it
        is the separately rendered selection:

        >>> employee = variable(Employee, [])
        >>> department = variable(Department, [])
        >>> verbalize_expression(a(set_of(employee.salary, department.name).ordered_by(
        ...     employee.salary).limit(3)))
        'Find the bottom three (the salary of an Employee, the name of a Department)'
        """
        if plan.ranking is None:
            return self._verb(plan)
        surface = ranking_surface(RankingRequest(plan=plan.ranking))
        return PhraseFragment(
            parts=[
                Keywords.FIND.as_fragment(),
                Articles.THE.as_fragment(),
                surface.pre_head,
            ]
        )

    def _verb(self, plan: QueryPlan) -> VerbalizationFragment:
        """:return: the opening verb — *"Report"* when the query is a report (its plan aggregates,
        groups, or orders the results — the three :class:`ReportKind` variants, all of which present
        a summary rather than search for individual matches), a backend-chosen *"Generate"* /
        *"Find"* when a performative override is set, else *"Find"* (a search).

        It emits only the leading word of the shown output: this ordered query presents results, so
        it produces *"Report"* rather than *"Find"*:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).ordered_by(employee.salary)))
        'Report Employees ordered by their salaries from lowest to highest'
        """
        if plan.report is not None:
            return self._sentence_initial(Keywords.REPORT.as_fragment())
        override = self.context.services.performative_override
        if override is not None:
            return override.as_fragment()
        return Keywords.FIND.as_fragment()

    def _subject_number(self, plan: QueryPlan) -> GrammaticalNumber:
        """:return: the grammatical number of the rendered subject — plural for a ranking of several
        (*"the top three Employees"*) or an ordered report (*"Report Employees"*), else singular. The
        subject's restriction and possessives agree with it (*"whose salaries are …"*).

        Here this ordered report yields ``PLURAL``, which is why the subject surfaces as *"Employees"*
        and its possessive as *"their salaries"* rather than the singular forms:

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).ordered_by(employee.salary)))
        'Report Employees ordered by their salaries from lowest to highest'
        """
        if plan.ranking is not None:
            return ranking_surface(RankingRequest(plan=plan.ranking)).number
        if plan.report is not None and plan.report.kind in (
            ReportKind.ORDERING,
            ReportKind.GROUPING,
        ):
            return GrammaticalNumber.PLURAL
        return GrammaticalNumber.SINGULAR

    # %% subject selection

    def _assemble_subject(self, node: Query, plan: QueryPlan) -> VerbalizationFragment:
        """:return: *"Find a Robot whose battery is high, such that … [clauses]"* — the
        plain-variable selection with its WHERE woven in.

        It assembles the whole sentence for the plain-variable (SUBJECT) shape: the selection
        *"a Robot"* with its restriction folded in as *"whose battery is greater than 50"*, as in the
        class example.
        """
        variable = node.selected_variable
        selected = self._build_selection(node, variable, plan)
        selected, where_items = self._apply_subject_restrictions(plan, selected)
        # No scope marker: the engine stamps this body with its query node, and the coreference
        # pass reads the focus for that query from the discourse view.
        return self._query_body(node, plan, selected, where_items=where_items)

    def _build_selection(
        self, node: Query, variable: SymbolicExpression, plan: QueryPlan
    ) -> VerbalizationFragment:
        """:return: the selection's referring noun phrase — a ``limit`` ranking phrase (*"the top
        three Robots"*), else *"the unique Robot"* (``eql.the``) / *"a Robot"*.

        It produces only the selection noun after *"Find"*; with no ranking, report, or ``eql.the``
        here it falls to the plain indefinite *"an Employee"*:

        >>> verbalize_expression(an(entity(variable(Employee, []))))
        'Find an Employee'
        """
        if plan.ranking is not None:
            return self._build_ranking_selection(variable, plan)
        if plan.report is not None and plan.report.kind is ReportKind.ORDERING:
            # An ordered listing presents all the matching results, so the subject is plural
            # ("Report Employees …"); a plural subject pronominalises to "their".
            return self._selections.one(variable, self._subject_number(plan))
        if plan.is_the:
            # "the unique <type>" first mention; the coreference pass reduces a repeat to
            # "the <type>" (UNIQUE downgrades to DEFINITE) — so it is a referring noun phrase.
            return NounPhrase(
                head=RoleFragment.for_variable(plan.selected_type, variable),
                definiteness=Definiteness.UNIQUE,
                referent_id=subject_referent_id(variable),
            )
        # context.child(variable) → VariableRule referring noun phrase; the entity shares its referent.
        return self.context.child(variable)

    def _build_ranking_selection(
        self, variable: SymbolicExpression, plan: QueryPlan
    ) -> VerbalizationFragment:
        """:return: The ranking selection — *"the first two Robots"* / *"the top three Employees by
        salary"* / *"the Employee with the highest salary"*. A ranking is inherently definite
        (*"the"*), so it ignores ``is_the``; it stays a referring noun phrase so a repeat mention
        reduces to *"the Robot"* and a WHERE pronominalises (*"its"* / *"their"*).

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(entity(employee).ordered_by(employee.salary, descending=True).limit(3))
        'Find the top three Employees by salary'
        """
        surface = ranking_surface(RankingRequest(plan=plan.ranking))
        return NounPhrase(
            head=RoleFragment.for_variable(
                plan.selected_type, variable, number=surface.number
            ),
            number=surface.number,
            definiteness=Definiteness.DEFINITE,
            pre_head=surface.pre_head,
            modifiers=surface.modifiers,
            referent_id=subject_referent_id(variable),
        )

    def _apply_subject_restrictions(
        self, plan: QueryPlan, selected: VerbalizationFragment
    ) -> Tuple[VerbalizationFragment, List[Optional[VerbalizationFragment]]]:
        """:return: The selection with its inline superlative modifiers attached, and the WHERE's
        clause items — the *"whose"* group (a sub-list of points in hierarchical) then a separate
        *"such that <residual>"* clause (each ``None`` when absent).

        In the class example it contributes the single *"whose battery is greater than 50"* clause
        item (no superlative modifier and no *"such that"* residual), which the body appends to the
        selection.
        """
        rendered = ClauseComposer(self.context).restriction(
            plan, self._subject_number(plan)
        )
        if rendered is None:
            return selected, []
        post_noun: List[VerbalizationFragment] = list(rendered.inline_modifiers)
        if rendered.relative_clauses:
            post_noun.append(
                oxford_comma(rendered.relative_clauses, Conjunctions.AND.as_fragment())
            )
        if post_noun:
            selected = PhraseFragment(parts=[selected, *post_noun])
        residual = (
            PhraseFragment(parts=[Keywords.SUCH_THAT.as_fragment(), rendered.residual])
            if rendered.residual is not None
            else None
        )
        return selected, [rendered.whose, residual]

    # %% noun forms

    def _as_noun(self, entity: Query) -> VerbalizationFragment:
        """
        A referring noun phrase — *"a/the unique <type>"* with the restrictions as appositive
        modifiers.

        :param entity: The nested entity selector.
        :return: The standalone-noun form *"a Robot where …"*.

        It emits the entity as a referring noun with its WHERE folded in as an appositive *"whose"*
        modifier — the *"a Worker whose tasks contains a Task"* span carried after the verb:

        >>> worker = variable(Worker, [])
        >>> verbalize_expression(
        ...     an(entity(worker).where(contains(worker.tasks, an(entity(variable(Task, []))))))
        ... )
        'Find a Worker whose tasks contains a Task'
        """
        plan = self.plan(entity)
        variable = entity.selected_variable
        definiteness = Definiteness.UNIQUE if plan.is_the else Definiteness.INDEFINITE
        # A referring noun phrase is the subject of its own modifiers (the coreference pass infers
        # this from the modifiers slot), so no scope marker is emitted here. The "whose" block
        # flattens to "whose a, and b" inline (the renderer expands a block into points only at the
        # item level, not inside a phrase).
        return referring_noun_with_restrictions(
            variable,
            plan.selected_type,
            definiteness,
            ClauseComposer(self.context).restriction(plan),
        )

    # %% query-body clauses

    def _query_body(
        self,
        node: Query,
        plan: QueryPlan,
        selection: VerbalizationFragment,
        where_items: List[Optional[VerbalizationFragment]],
        find_header: Optional[VerbalizationFragment] = None,
    ) -> VerbalizationFragment:
        """:return: *"Find <selection>"* + the present clauses (the subject restriction's *"whose"*
        / *"such that"*, then *grouped by … having … ordered by …*) as block items — absent
        clauses (``None``) are simply skipped.

        It joins the pieces into the final block: in the class example the *"Find a Robot"* header
        with the single *"whose battery is greater than 50"* clause item appended.
        """
        if find_header is None:
            find_header = self._verb(plan)
        header = PhraseFragment(parts=[find_header, selection])
        clauses = [
            clause
            for clause in [*where_items, *self._trailing_clauses(node, plan)]
            if clause is not None
        ]
        return BlockFragment(header=header, items=clauses)

    def _trailing_clauses(
        self, node: Query, plan: QueryPlan
    ) -> List[Optional[VerbalizationFragment]]:
        """:return: The post-selection clauses, in canonical reading order (``None`` when absent).

        The standalone *"ordered by …"* clause is suppressed when a ranking selection already
        conveys the ordering — i.e. a ``limit`` on a plain-variable (SUBJECT) selection. Ordering
        without a ``limit`` keeps the clause; a limited set-of keeps it too (no ranking selection).

        >>> employee = variable(Employee, [])
        >>> verbalize_expression(an(entity(employee).ordered_by(employee.salary)))
        'Report Employees ordered by their salaries from lowest to highest'
        """
        composer = ClauseComposer(self.context)
        ranked = plan.ranking is not None and plan.kind in (
            SelectionKind.SUBJECT,
            SelectionKind.SET_OF,
        )
        # A grouped report fronts its grouping as "For each …", so the trailing "grouped by" clause
        # would merely repeat it.
        fronts_grouping = plan.report is not None and plan.report.is_grouped
        return [
            None if fronts_grouping else composer.grouped_by(node),
            # A grouped report fronts its HAVING onto the group key ("For each department whose …"),
            # so the trailing filter would repeat it.
            None if fronts_grouping else composer.having(node),
            None if ranked else composer.ordered_by(node),
        ]

    def _where_clause(self, plan: QueryPlan) -> Optional[VerbalizationFragment]:
        """:return: *"such that <condition>"*, or ``None`` when the query has no WHERE.

        It emits only the trailing *"such that the battery of the Robot is greater than 50"* span; the
        *"Find a Robot and a Task"* before it comes from the selection:

        >>> robot = variable(Robot, [])
        >>> verbalize_expression(a(set_of(robot, variable(Task, [])).where(robot.battery > 50)))
        'Find a Robot and a Task such that the battery of the Robot is greater than 50'
        """
        if plan.where_condition is None:
            return None
        return PhraseFragment(
            parts=[
                Keywords.SUCH_THAT.as_fragment(),
                self.context.child(plan.where_condition),
            ]
        )

    def inline_noun(self, entity: Query) -> VerbalizationFragment:
        """
        The entity's WHERE condition is deferred to the binding scope so it can
        be emitted as a *"such that …"* clause after all binding overrides are
        registered.

        :param entity: The entity used as a chain root inside an instantiated variable.
        :return: The inline-noun form for *entity*.

        It contributes only the chain-root noun *"a Robot"*; the surrounding *"the name of"* is
        supplied by the enclosing attribute chain:

        >>> verbalize_expression(an(entity(variable(Robot, []))).name)
        'the name of a Robot'
        """
        entity.build()
        variable = entity.selected_variable
        type_name = FallbackNouns.ENTITY.name_of(variable)

        where_expression = entity._where_expression_
        if where_expression is not None:
            self.context.scope.defer_constraint(where_expression.condition)

        # A referring noun phrase (referent_id below) — a repeat reduces to "the <type>" in the pass.
        return NounPhrase(
            head=RoleFragment.for_variable(type_name, variable),
            referent_id=subject_referent_id(variable),
        )
