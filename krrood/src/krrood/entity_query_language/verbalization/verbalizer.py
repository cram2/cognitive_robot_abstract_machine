from __future__ import annotations

import ast
import datetime as _dt
import inspect
import operator
import re
from dataclasses import dataclass
from textwrap import dedent
from typing import Optional

from krrood.entity_query_language.exceptions import WrongPropertyReturnStatementImplementation
from krrood.entity_query_language.verbalization.utils import inflect_engine, _camel_to_words, _ordinal, _ensure_plural, \
    _apply_binding_aliases
from krrood.patterns.code_parsing_utils import get_accessed_attribute_name_in_return_statement_of_property
from krrood.singleton import SingletonMeta

from krrood.entity_query_language.core.base_expressions import SymbolicExpression
from krrood.entity_query_language.core.mapped_variable import (
    Attribute,
    Call,
    FlatVariable,
    Index,
    MappedVariable,
)
from krrood.entity_query_language.core.variable import (
    InstantiatedVariable,
    Literal,
    Variable,
)
from krrood.entity_query_language.operators.aggregators import (
    Average,
    Count,
    CountAll,
    Max,
    Min,
    Mode,
    MultiMode,
    Sum,
)
from krrood.entity_query_language.operators.comparator import Comparator, not_contains
from krrood.entity_query_language.operators.core_logical_operators import AND, OR, Not
from krrood.entity_query_language.operators.logical_quantifiers import Exists, ForAll
from krrood.entity_query_language.predicate import Verbalizable, Triple
from krrood.entity_query_language.query.operations import (
    GroupedBy,
    Having,
    OrderedBy,
    Where,
)
from krrood.entity_query_language.query.quantifiers import An, ResultQuantifier, The
from krrood.entity_query_language.query.query import Entity, SetOf, Query
from krrood.entity_query_language.verbalization.context import VerbalizationContext, _article
from krrood.entity_query_language.verbalization.rule_analysis import (
    AggregationStatus,
    RuleAnalyzer,
    RuleStructure,
)

_OP_WORDS = {
    operator.eq: "is",
    operator.ne: "is not",
    operator.lt: "is less than",
    operator.le: "is at most",
    operator.gt: "is greater than",
    operator.ge: "is at least",
    operator.contains: "contains",
    not_contains: "does not contain",
}

_OP_WORDS_COMPACT = {
    operator.eq: "equals",
    operator.ne: "does not equal",
    operator.lt: "less than",
    operator.le: "at most",
    operator.gt: "greater than",
    operator.ge: "at least",
    operator.contains: "contains",
    not_contains: "does not contain",
}

_NEGATED_OP_WORDS = {
    operator.gt: "is not greater than",
    operator.lt: "is not less than",
    operator.ge: "is not at least",
    operator.le: "is not at most",
    operator.eq: "is not",
    operator.ne: "is",
    operator.contains: "does not contain",
    not_contains: "contains",
}

_NEGATED_OP_WORDS_COMPACT = {
    operator.gt: "not greater than",
    operator.lt: "not less than",
    operator.ge: "not at least",
    operator.le: "not at most",
    operator.eq: "does not equal",
    operator.ne: "equals",
    operator.contains: "does not contain",
    not_contains: "contains",
}

_OP_WORDS_TEMPORAL = {
    operator.lt: "is before",
    operator.gt: "is after",
    operator.le: "is no later than",
    operator.ge: "is no earlier than",
    operator.eq: "is at",
    operator.ne: "is not at",
}

_OP_WORDS_TEMPORAL_COMPACT = {
    operator.lt: "before",
    operator.gt: "after",
    operator.le: "no later than",
    operator.ge: "no earlier than",
    operator.eq: "at",
    operator.ne: "not at",
}

_NEGATED_OP_WORDS_TEMPORAL = {
    operator.lt: "is no earlier than",
    operator.gt: "is no later than",
    operator.le: "is after",
    operator.ge: "is before",
    operator.eq: "is not at",
    operator.ne: "is at",
}

_NEGATED_OP_WORDS_TEMPORAL_COMPACT = {
    operator.lt: "no earlier than",
    operator.gt: "no later than",
    operator.le: "after",
    operator.ge: "before",
    operator.eq: "not at",
    operator.ne: "at",
}


@dataclass
class EQLVerbalizer(metaclass=SingletonMeta):
    """
    Visitor-based verbalizer: maps an EQL expression tree to readable English.

    Usage::

        verbalizer = EQLVerbalizer()
        text = verbalizer.verbalize(query)

    Each ``_v_<ClassName>_`` method handles one node type.  Unknown types fall
    back to :meth:`_v_default_` which returns the node's ``_name_`` property.
    """

    # ── Dispatcher ─────────────────────────────────────────────────────────────

    def verbalize(
        self,
        expr: SymbolicExpression,
        ctx: Optional[VerbalizationContext] = None,
    ) -> str:
        if ctx is None:
            ctx = VerbalizationContext.from_expression(expr)
        method = getattr(self, f"_v_{type(expr).__name__}_", self._v_default_)
        return method(expr, ctx)

    # ── Leaves ─────────────────────────────────────────────────────────────────

    def _v_Variable_(self, expr: Variable, ctx: VerbalizationContext) -> str:
        return ctx.noun_for(expr)

    def _v_Literal_(self, expr: Literal, ctx: VerbalizationContext) -> str:
        return ctx.type_name_of_value(expr._value_)

    def _v_ExternallySetVariable_(self, expr, ctx: VerbalizationContext) -> str:
        type_name = expr._type_.__name__ if getattr(expr, "_type_", None) else "variable"
        return f"{_article(type_name)} {type_name}"

    # ── MappedVariables ────────────────────────────────────────────────────────

    def _v_Attribute_(self, expr: Attribute, ctx: VerbalizationContext) -> str:
        return self._verbalize_mapped_chain_(expr, ctx)

    def _v_Index_(self, expr: Index, ctx: VerbalizationContext) -> str:
        return self._verbalize_mapped_chain_(expr, ctx)

    def _v_Call_(self, expr: Call, ctx: VerbalizationContext) -> str:
        return self._verbalize_mapped_chain_(expr, ctx)

    def _v_FlatVariable_(self, expr: FlatVariable, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr._child_, ctx)

    def _verbalize_plural_(self, expr, ctx: VerbalizationContext) -> str:
        """
        Return a plural noun phrase for *expr* — used by ForAll and aggregators.

        * ``Variable(T)``              → ``"Employees"``
        * ``FlatVariable(child)``      → recurse into child
        * ``Attribute`` (single-hop)   → ``"Cabinets' containers"``
        * anything else                → fall back to singular ``verbalize()``
        """
        if isinstance(expr, FlatVariable):
            return self._verbalize_plural_(expr._child_, ctx)

        if isinstance(expr, Variable):
            type_name = expr._type_.__name__
            label = ctx.disambiguation_map.get(expr._id_, type_name)
            ctx.seen[expr._id_] = label
            if label != type_name:
                return label
            return inflect_engine.plural(type_name)

        if isinstance(expr, Attribute):
            # Walk the chain to find root variable and single attribute hop.
            chain: list = []
            current = expr
            while isinstance(current, MappedVariable):
                chain.append(current)
                current = current._child_
            root = current
            if isinstance(root, Variable) and len(chain) == 1 and isinstance(chain[0], Attribute):
                type_name = root._type_.__name__
                label = ctx.disambiguation_map.get(root._id_, type_name)
                ctx.seen[root._id_] = label
                root_plural = label if label != type_name else inflect_engine.plural(type_name)
                attr_plural = _ensure_plural(chain[0]._attribute_name_)
                return f"{attr_plural} of {root_plural}"

        return self.verbalize(expr, ctx)

    @staticmethod
    def _walk_chain_(expr: MappedVariable) -> tuple:
        """Walk a MappedVariable chain and return (root-first list, leaf node)."""
        chain: list[MappedVariable] = []
        current = expr
        while isinstance(current, MappedVariable):
            chain.append(current)
            current = current._child_
        chain.reverse()  # root-side first
        return chain, current

    @staticmethod
    def _render_path_(parts: list, root_text: str) -> str:
        """Render an attribute path as ``"the a of the b of root"``."""
        if not parts:
            return root_text
        inner = " of the ".join(reversed(parts))
        return f"the {inner} of {root_text}"

    def _verbalize_chain_root_(self, leaf, ctx: VerbalizationContext) -> str:
        """Resolve the root text for a chain leaf, unwrapping any ResultQuantifier to find an Entity."""
        # _update_children_ replaces a freshly-built Entity with its An/The wrapper;
        # unwrap to recover the underlying Entity and defer its where-conditions.
        inner = leaf
        while isinstance(inner, ResultQuantifier):
            inner = inner._child_
        if isinstance(inner, Entity):
            return self._verbalize_entity_as_inline_noun_(inner, ctx)
        return self.verbalize(leaf, ctx)

    def _verbalize_mapped_chain_(self, expr: MappedVariable, ctx: VerbalizationContext,
                                 negated: bool = False) -> str:
        """
        Natural-language path for a MappedVariable chain.

        * Boolean terminal ``Attribute``: predicative — ``"nav is [not] active"``.
        * All other chains: ``"of"`` form — ``"battery of Robot"``,
          ``"name of tasks[0] of the Robot"``.
        """
        chain, leaf = self._walk_chain_(expr)
        root_text = self._verbalize_chain_root_(leaf, ctx)
        terminal = chain[-1]
        if isinstance(terminal, Attribute) and terminal._type_ is bool:
            nav_text = self._verbalize_navigation_chain_(chain[:-1], root_text)
            verb = "is not" if negated else "is"
            return f"{nav_text} {verb} {terminal._attribute_name_}"
        return self._render_path_(self._build_path_parts_(chain), root_text)

    def _verbalize_navigation_chain_(self, nav_chain: list, root_text: str) -> str:
        """
        Verbalize the navigation portion of a chain (everything before a boolean terminal).

        An integer ``Index`` at the end of the chain is converted to an ordinal:
        ``[Attribute("tasks"), Index(0)]`` → ``"the first of tasks of the Robot"``.
        """
        if not nav_chain:
            return root_text
        if isinstance(nav_chain[-1], Index) and isinstance(nav_chain[-1]._key_, int):
            ordinal = _ordinal(nav_chain[-1]._key_)
            pre_text = self._render_path_(self._build_path_parts_(nav_chain[:-1]), root_text)
            return f"the {ordinal} of {pre_text}"
        return self._render_path_(self._build_path_parts_(nav_chain), root_text)

    def _build_path_parts_(self, chain: list) -> list[str]:
        """Build readable string fragments for a root-to-leaf MappedVariable chain."""
        parts: list[str] = []
        i = 0
        while i < len(chain):
            node = chain[i]
            if isinstance(node, Attribute):
                name = node._attribute_name_
                # Eagerly absorb immediately following Index nodes into the attr name.
                while i + 1 < len(chain) and isinstance(chain[i + 1], Index):
                    i += 1
                    name += f"[{repr(chain[i]._key_)}]"
                parts.append(name)
            elif isinstance(node, Index):
                parts.append(f"[{repr(node._key_)}]")
            elif isinstance(node, Call):
                parts.append("()")
            elif isinstance(node, FlatVariable):
                pass  # FlatVariable handled by _v_FlatVariable_
            i += 1
        return parts

    # ── Instantiated (predicates / inference variables) ────────────────────────

    def _v_InstantiatedVariable_(
        self, expr: InstantiatedVariable, ctx: VerbalizationContext
    ) -> str:
        """
        Verbalize an InstantiatedVariable (Predicate, or an Inferred Class) using the given template.
        """
        try:
            if isinstance(expr._type_, type) and issubclass(expr._type_, Verbalizable):
                template = expr._type_._verbalization_template_()
                return self._verbalize_template_(expr, ctx, template)
        except NotImplementedError:
            # Means that the `_verbalization_template_` is not implemeted for this Predicate type.
            pass
        return self._verbalize_instantiated_natural_(expr, ctx)

    def _verbalize_template_(
        self, expr: InstantiatedVariable, ctx: VerbalizationContext, template: str
    ) -> str:
        """
        Verbalize an expression using the given template.

        :param expr: The expression to be verbalized.
        :param ctx: VerbalizationContext.
        :param template: The template to be verbalized.
        :return: The verbalized expression as a string.
        """
        if issubclass(expr._type_, Triple):
            subject_name = get_accessed_attribute_name_in_return_statement_of_property(expr._type_.subject, expr._type_)
            object_name = get_accessed_attribute_name_in_return_statement_of_property(expr._type_.object, expr._type_)
            template = template.replace("{subject}", f"{subject_name}")
            template = template.replace("{object}", f"{object_name}")
        kwargs = {
            name: self.verbalize(child, ctx)
            for name, child in expr._child_vars_.items()
        }
        return template.format(**kwargs)

    def _verbalize_predicate_no_template_(
        self, expr: InstantiatedVariable, ctx: VerbalizationContext
    ) -> str:
        type_name = getattr(expr._type_, "__name__", str(expr._type_))
        if len(expr._child_vars_) == 2:
            items = list(expr._child_vars_.items())
            left, right = items[0][1], items[1][1]
            predicate_text = _camel_to_words(type_name)
            left_text = self.verbalize(left, ctx)
            right_text = self.verbalize(right, ctx)
            return f"{left_text} {predicate_text} {right_text}"
        if expr._child_vars_:
            args_str = ", ".join(
                f"{name}={self.verbalize(child, ctx)}"
                for name, child in expr._child_vars_.items()
            )
            return f"{_article(type_name)} {type_name}({args_str})"
        return f"{_article(type_name)} {type_name}"

    def _verbalize_instantiated_natural_(
        self, expr: InstantiatedVariable, ctx: VerbalizationContext
    ) -> str:
        type_name = getattr(expr._type_, "__name__", str(expr._type_))

        if expr._id_ in ctx.seen:
            return f"the {ctx.seen[expr._id_]}"
        ctx.seen[expr._id_] = type_name

        ctx.push_constraint_frame()

        binding_parts: list[str] = []
        binding_alias_map: dict[str, str] = {}
        for field_name, child_expr in expr._child_vars_.items():
            field_ref = f"the {field_name} of the {type_name}"
            if inflect_engine.singular_noun(field_name):
                plural_value = self._verbalize_plural_(child_expr, ctx)
                binding_parts.append(f"{field_ref} are {plural_value}")
            else:
                value_text = self.verbalize(child_expr, ctx)
                binding_parts.append(f"{field_ref} is {value_text}")
                definite_value = re.sub(r"\b(a|an) ([A-Z])", r"the \2", value_text)
                if re.search(r"\bthe [A-Z]", definite_value) and definite_value not in binding_alias_map:
                    binding_alias_map[definite_value] = field_ref

        constraints = ctx.pop_constraint_frame()

        ctx.binding_aliases.update(binding_alias_map)
        if constraints and binding_alias_map:
            constraints = [_apply_binding_aliases(c, binding_alias_map) for c in constraints]

        result = f"{_article(type_name)} {type_name}"
        if binding_parts:
            result += ", where " + " and ".join(binding_parts)
        if constraints:
            result += ", such that " + " and ".join(constraints)
        return result

    # ── Logical operators ──────────────────────────────────────────────────────

    def _v_AND_(self, expr: AND, ctx: VerbalizationContext) -> str:
        parts = [self.verbalize(c, ctx) for c in ctx.flatten_same_type(expr, AND)]
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def _v_OR_(self, expr: OR, ctx: VerbalizationContext) -> str:
        parts = [self.verbalize(c, ctx) for c in ctx.flatten_same_type(expr, OR)]
        if len(parts) == 1:
            return parts[0]
        return "either " + ", ".join(parts[:-1]) + f", or {parts[-1]}"

    def _v_Not_(self, expr: Not, ctx: VerbalizationContext) -> str:
        child = expr._child_
        # Case 1: negate a comparator — inline the negated verb word.
        if isinstance(child, Comparator):
            left = self.verbalize(child.left, ctx)
            right = self.verbalize(child.right, ctx)
            is_temporal = self._is_temporal_(child.left) or self._is_temporal_(child.right)
            if is_temporal:
                neg_table = _NEGATED_OP_WORDS_TEMPORAL_COMPACT if ctx.compact_predicates else _NEGATED_OP_WORDS_TEMPORAL
                fallback_table = _OP_WORDS_TEMPORAL_COMPACT if ctx.compact_predicates else _OP_WORDS_TEMPORAL
            else:
                neg_table = _NEGATED_OP_WORDS_COMPACT if ctx.compact_predicates else _NEGATED_OP_WORDS
                fallback_table = _OP_WORDS_COMPACT if ctx.compact_predicates else _OP_WORDS
            op_word = neg_table.get(
                child.operation, f"not {fallback_table.get(child.operation, child._name_)}"
            )
            return f"{left} {op_word} {right}"
        # Case 2: negate a boolean attribute chain — inline "is not".
        if isinstance(child, MappedVariable):
            chain, _ = self._walk_chain_(child)
            if isinstance(chain[-1], Attribute) and chain[-1]._type_ is bool:
                return self._verbalize_mapped_chain_(child, ctx, negated=True)
        # Case 3: fallback — wrap with "not (…)".
        return f"not ({self.verbalize(child, ctx)})"

    # ── Quantifiers ────────────────────────────────────────────────────────────

    def _v_ForAll_(self, expr: ForAll, ctx: VerbalizationContext) -> str:
        var_text = self._verbalize_plural_(expr.variable, ctx)
        cond_text = self.verbalize(expr.condition, ctx)
        return f"for all {var_text}, {cond_text}"

    def _v_Exists_(self, expr: Exists, ctx: VerbalizationContext) -> str:
        var_text = self.verbalize(expr.variable, ctx)
        cond_text = self.verbalize(expr.condition, ctx)
        return f"there exists {var_text} such that {cond_text}"

    # ── Comparators ────────────────────────────────────────────────────────────

    def _is_temporal_(self, expr) -> bool:
        """Return True if *expr* is or produces a datetime.datetime value."""
        if isinstance(expr, Literal):
            return isinstance(expr._value_, _dt.datetime)
        if isinstance(expr, Variable):
            return getattr(expr, "_type_", None) is _dt.datetime
        if isinstance(expr, MappedVariable):
            chain, current = [], expr
            while isinstance(current, MappedVariable):
                chain.append(current)
                current = current._child_
            return bool(chain) and getattr(chain[-1], "_type_", None) is _dt.datetime
        return False

    def _v_Comparator_(self, expr: Comparator, ctx: VerbalizationContext) -> str:
        left = self.verbalize(expr.left, ctx)
        right = self.verbalize(expr.right, ctx)
        if self._is_temporal_(expr.left) or self._is_temporal_(expr.right):
            table = _OP_WORDS_TEMPORAL_COMPACT if ctx.compact_predicates else _OP_WORDS_TEMPORAL
        else:
            table = _OP_WORDS_COMPACT if ctx.compact_predicates else _OP_WORDS
        op_word = table.get(expr.operation, expr._name_)
        return f"{left} {op_word} {right}"

    # ── Aggregators ────────────────────────────────────────────────────────────

    def _v_Count_(self, expr: Count, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "number of {}")

    def _v_CountAll_(self, expr: CountAll, ctx: VerbalizationContext) -> str:
        return "count of all"

    def _v_Sum_(self, expr: Sum, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "sum of {}")

    def _v_Average_(self, expr: Average, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "average of {}")

    def _v_Max_(self, expr: Max, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "maximum {}")

    def _v_Min_(self, expr: Min, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "minimum {}")

    def _v_Mode_(self, expr: Mode, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "mode of {}")

    def _v_MultiMode_(self, expr: MultiMode, ctx: VerbalizationContext) -> str:
        return self._verbalize_aggregator_(expr, ctx, "all modes of {}")

    def _verbalize_aggregator_(self, expr, ctx: VerbalizationContext, template: str) -> str:
        """
        Verbalize an aggregator with coreference: first mention returns the plain phrase;
        any subsequent mention of the same expression prefixes it with "the".
        """
        child_text = self._verbalize_plural_(expr._child_, ctx)
        phrase = template.format(child_text)
        if expr._id_ in ctx.seen:
            return f"the {phrase}"
        ctx.seen[expr._id_] = phrase
        return phrase

    # ── Rule (If … then …) verbalization ─────────────────────────────────────

    _rule_analyzer = RuleAnalyzer()

    def _verbalize_rule_(self, expr: Entity, ctx: VerbalizationContext) -> str:
        structure = self._rule_analyzer.analyze(expr)
        if_parts = self._verbalize_rule_if_(structure, ctx)
        then_parts = self._verbalize_rule_then_(structure, ctx)
        return f"If {if_parts}, then {then_parts}"

    def _verbalize_rule_if_(self, s: RuleStructure, ctx: VerbalizationContext) -> str:
        from krrood.entity_query_language.query.query import Entity as _Entity

        # Bucket extra WHERE conditions by which antecedent owns their left side.
        ant_by_root_id = {self._antecedent_var_id_(a): a for a in s.antecedents}
        extra_by_ant: dict = {self._antecedent_var_id_(a): [] for a in s.antecedents}
        unmatched: list = []
        for cond in s.extra_where_conditions:
            owner_id = self._condition_left_owner_id_(cond)
            if owner_id in extra_by_ant:
                extra_by_ant[owner_id].append(cond)
            else:
                unmatched.append(cond)

        # An antecedent is "primary" (introduced explicitly in IF) if it has
        # own conditions OR is the left-side owner of at least one extra WHERE.
        primary_ids = {
            self._antecedent_var_id_(a)
            for a in s.antecedents
            if a.own_conditions or extra_by_ant.get(self._antecedent_var_id_(a))
        }

        clauses: list[str] = []
        for ant in s.antecedents:
            root = ant.root
            type_name = ant.type_name
            ant_id = self._antecedent_var_id_(ant)

            if ant_id not in primary_ids:
                # Non-primary: just register in ctx.seen for later coreference.
                ctx.seen[root._id_] = type_name
                if isinstance(root, _Entity):
                    root.build()
                    sel = root.selected_variable
                    if sel is not None and hasattr(sel, "_id_"):
                        ctx.seen[sel._id_] = type_name
                continue

            # Introduce the antecedent
            if ant.aggregation_status == AggregationStatus.AGGREGATED:
                intro = f"there are {inflect_engine.plural(type_name)}"
            else:
                intro = f"there's {_article(type_name)} {type_name}"

            ctx.seen[root._id_] = type_name
            if isinstance(root, _Entity):
                root.build()
                sel = root.selected_variable
                if sel is not None and hasattr(sel, "_id_"):
                    ctx.seen[sel._id_] = type_name

            # Own conditions + extra WHERE conditions owned by this antecedent
            all_conditions = ant.own_conditions + extra_by_ant.get(ant_id, [])
            whose = self._whose_clauses_from_conditions_(
                all_conditions, root, type_name, ant.aggregation_status, s.antecedents, ctx
            )
            clauses.append(f"{intro}, {whose}" if whose else intro)

        # Unmatched conditions added to the last clause
        for cond in unmatched:
            cond_text = self.verbalize(cond, ctx)
            if clauses:
                clauses[-1] = clauses[-1] + f", and {cond_text}"
            else:
                clauses.append(cond_text)

        return ", and ".join(clauses) if clauses else "true"

    def _verbalize_rule_then_(self, s: RuleStructure, ctx: VerbalizationContext) -> str:
        type_name = s.consequent_type
        ctx.seen[id(s)] = type_name  # placeholder; real id set below

        intro = f"there's {_article(type_name)} {type_name}"

        whose_parts: list[str] = []
        for binding in s.consequent_bindings:
            field = binding.field_name
            if binding.is_plural_field:
                value_text = _ensure_plural(self._verbalize_plural_(binding.value_expr, ctx))
                if binding.aggregation_status == AggregationStatus.AGGREGATED:
                    # Second mention — use "the Xs"
                    value_text = f"the {value_text}"
                whose_parts.append(f"whose {field} are {value_text}")
            elif binding.aggregation_status == AggregationStatus.GROUP_KEY:
                value_text = self._verbalize_group_key_value_(binding.value_expr, ctx)
                whose_parts.append(f"whose {field} is {value_text}")
            else:
                value_text = self.verbalize(binding.value_expr, ctx)
                whose_parts.append(f"whose {field} is {value_text}")

        if whose_parts:
            return intro + ", " + ", and ".join(whose_parts)
        return intro

    def _whose_clauses_from_conditions_(
        self, conditions, root, type_name: str,
        agg: AggregationStatus, all_antecedents, ctx: VerbalizationContext
    ) -> str:
        """Build comma-joined 'whose X is Y' phrases from a list of conditions."""
        parts: list[str] = []
        for cond in conditions:
            text = self._try_whose_from_condition_(cond, all_antecedents, ctx)
            parts.append(text if text else self.verbalize(cond, ctx))
        if not parts:
            return ""
        if len(parts) == 1:
            return parts[0]
        return ", ".join(parts[:-1]) + f", and {parts[-1]}"

    def _try_whose_from_condition_(
        self, cond, antecedents, ctx: VerbalizationContext
    ) -> Optional[str]:
        """
        If *cond* is ``left == right`` where ``left`` is an Attribute of a known
        antecedent variable, return ``"whose {attr} is {right}"``.
        """
        from krrood.entity_query_language.operators.comparator import Comparator
        from krrood.entity_query_language.core.mapped_variable import Attribute, MappedVariable
        from krrood.entity_query_language.query.quantifiers import ResultQuantifier
        import operator as _op

        if not isinstance(cond, Comparator) or cond.operation is not _op.eq:
            return None
        left = cond.left
        if not isinstance(left, Attribute):
            return None

        current = left
        attr_names: list[str] = []
        while isinstance(current, MappedVariable):
            if hasattr(current, "_attribute_name_"):
                attr_names.append(current._attribute_name_)
            current = current._child_
        while isinstance(current, ResultQuantifier):
            current = current._child_

        matched_ant = self._find_matching_antecedent_(current, antecedents)
        if matched_ant is None or not attr_names:
            return None

        raw_attr = attr_names[-1]
        aggregated = matched_ant.aggregation_status == AggregationStatus.AGGREGATED

        attr_word = _ensure_plural(raw_attr) if aggregated else raw_attr
        right_text = (
            self._verbalize_plural_(cond.right, ctx)
            if aggregated
            else self.verbalize(cond.right, ctx)
        )
        copula = "are" if aggregated else "is"
        return f"whose {attr_word} {copula} {right_text}"

    @staticmethod
    def _antecedent_var_id_(ant) -> Optional[object]:
        from krrood.entity_query_language.query.query import Entity as _Entity
        root = ant.root
        if isinstance(root, _Entity):
            root.build()
            sel = root.selected_variable
            return getattr(sel, "_id_", None)
        return getattr(root, "_id_", None)

    def _condition_left_owner_id_(self, cond) -> Optional[object]:
        """Return the variable ID of the left-side root of an equality condition, if any."""
        from krrood.entity_query_language.operators.comparator import Comparator
        from krrood.entity_query_language.core.mapped_variable import MappedVariable
        from krrood.entity_query_language.query.quantifiers import ResultQuantifier
        import operator as _op

        if not isinstance(cond, Comparator) or cond.operation is not _op.eq:
            return None
        current = cond.left
        while isinstance(current, MappedVariable):
            current = current._child_
        while isinstance(current, ResultQuantifier):
            current = current._child_
        return getattr(current, "_id_", None)

    @staticmethod
    def _find_matching_antecedent_(var_node, antecedents):
        """Return the antecedent whose root variable matches *var_node*, or None."""
        from krrood.entity_query_language.query.query import Entity as _Entity
        node_id = getattr(var_node, "_id_", None)
        for ant in antecedents:
            root = ant.root
            if isinstance(root, _Entity):
                root.build()
                sel = root.selected_variable
                ant_id = getattr(sel, "_id_", None)
            else:
                ant_id = getattr(root, "_id_", None)
            if ant_id is not None and ant_id == node_id:
                return ant
        return None

    def _verbalize_group_key_value_(self, expr, ctx: VerbalizationContext) -> str:
        """
        Render a GROUP_KEY binding value as ``"the common {attr} of the {PluralRoot}"``.
        """
        from krrood.entity_query_language.core.mapped_variable import MappedVariable
        from krrood.entity_query_language.core.variable import Variable

        chain: list = []
        current = expr
        while isinstance(current, MappedVariable):
            chain.append(current)
            current = current._child_
        chain.reverse()  # root-side first

        if not chain or not isinstance(current, Variable):
            return self.verbalize(expr, ctx)

        root_type = current._type_.__name__ if getattr(current, "_type_", None) else "entity"
        root_plural = inflect_engine.plural(root_type)
        # Mark root as seen (plural mention)
        ctx.seen[current._id_] = root_type

        parts = self._build_path_parts_(chain)
        if not parts:
            return f"the common {root_type} of the {root_plural}"

        outermost = list(reversed(parts))[0]
        return f"the common {outermost} of the {root_plural}"

    # ── Query: Entity and SetOf ────────────────────────────────────────────────

    def _v_Entity_(self, expr: Entity, ctx: VerbalizationContext) -> str:
        if expr._id_ in ctx.seen:
            return f"the {ctx.seen[expr._id_]}"

        expr.build()

        # Rule form: entity whose selected variable is an inference
        if self._rule_analyzer.can_handle(expr):
            return self._verbalize_rule_(expr, ctx)

        is_the = (
            expr._quantifier_builder_ is not None
            and expr._quantifier_builder_.type is The
        )
        var = expr.selected_variable

        if isinstance(var, Entity):
            selected = self._verbalize_entity_as_noun_(var, ctx)
        elif var is None:
            selected_type = "entity"
            ctx.seen[expr._id_] = selected_type
            selected = "entities"
        elif is_the:
            selected_type = var._type_.__name__ if getattr(var, "_type_", None) else "entity"
            ctx.seen[var._id_] = selected_type
            ctx.seen[expr._id_] = selected_type
            selected = f"the unique {selected_type}"
        else:
            selected = self.verbalize(var, ctx)
            selected_type = ctx.seen.get(getattr(var, "_id_", None), "entity")
            ctx.seen[expr._id_] = selected_type

        return self._verbalize_query_body_(expr, ctx, f"Find {selected}")

    def _verbalize_entity_as_noun_(self, expr: Entity, ctx: VerbalizationContext) -> str:
        """
        Compact form used when an ``Entity`` acts as the selected variable of an outer query.

        Produces ``"the unique Container where its name equals …"`` rather than
        ``"Find the unique Container, such that …"``.
        """
        if expr._id_ in ctx.seen:
            return f"the {ctx.seen[expr._id_]}"

        expr.build()
        is_the = (
            expr._quantifier_builder_ is not None
            and expr._quantifier_builder_.type is The
        )
        var = expr.selected_variable
        selected_type = var._type_.__name__ if var and getattr(var, "_type_", None) else "entity"

        ctx.seen[expr._id_] = selected_type
        if var is not None:
            ctx.seen[var._id_] = selected_type

        if is_the:
            article_noun = f"the unique {selected_type}"
        else:
            article_noun = f"{_article(selected_type)} {selected_type}"

        where_expr = expr._where_expression_
        if where_expr is not None:
            cond = self.verbalize(where_expr.condition, ctx)
            return f"{article_noun} where {cond}"
        return article_noun

    def _verbalize_entity_as_inline_noun_(self, entity: Entity, ctx: VerbalizationContext) -> str:
        """
        Render an ``Entity`` as a bare noun phrase for use as the root of an
        ``Attribute`` chain inside an ``InstantiatedVariable``.

        On the **first** encounter the entity's type name is registered in
        *ctx.seen* (so subsequent mentions use "the") and its where-conditions
        are deferred into the current constraint frame.  On **subsequent**
        encounters the method returns ``"the <TypeName>"`` immediately with no
        side effects.
        """
        if entity._id_ in ctx.seen:
            return f"the {ctx.seen[entity._id_]}"

        entity.build()
        var = entity.selected_variable
        type_name = var._type_.__name__ if var and getattr(var, "_type_", None) else "entity"

        ctx.seen[entity._id_] = type_name
        if var is not None and hasattr(var, "_id_"):
            ctx.seen[var._id_] = type_name

        where_expr = entity._where_expression_
        if where_expr is not None:
            cond_text = self.verbalize(where_expr.condition, ctx)
            ctx.add_constraint(cond_text)

        return f"{_article(type_name)} {type_name}"

    def _v_SetOf_(self, expr: SetOf, ctx: VerbalizationContext) -> str:
        expr.build()
        vars_str = ", ".join(self.verbalize(v, ctx) for v in expr._selected_variables_)
        return self._verbalize_query_body_(expr, ctx, f"Find sets of ({vars_str})")

    @staticmethod
    def combine_in_a_bracket(parts: list[str]) -> str:
        if len(parts) == 1:
            return parts[0]
        return f"({EQLVerbalizer.combine(parts, 'and')})"

    @staticmethod
    def combine(parts: list[str], conjunction: str = "and") -> str:
        if len(parts) == 1:
            return parts[0]
        conjunction = f" {conjunction} " if conjunction else " "
        combined = ", ".join(parts[:-1]) + f",{conjunction}{parts[-1]}"
        return combined

    def _verbalize_query_body_(self, expr, ctx: VerbalizationContext, prefix: str) -> str:
        """Append where / grouped-by / having / ordered-by clauses to *prefix*."""
        parts = [prefix]

        where_expr = expr._where_expression_
        grouped_expr = expr._grouped_by_expression_
        having_expr = expr._having_expression_

        aliases = ctx.binding_aliases

        if where_expr is not None:
            where_text = _apply_binding_aliases(self.verbalize(where_expr.condition, ctx), aliases)
            parts.append(f"such that {where_text}")

        if grouped_expr is not None and grouped_expr.variables_to_group_by:
            group_key_root_ids = self._root_var_ids_(grouped_expr.variables_to_group_by)
            groups = [
                _apply_binding_aliases(self.verbalize(v, ctx), aliases)
                for v in grouped_expr.variables_to_group_by
            ]
            aggregated = self._aggregated_noun_phrases_(expr, group_key_root_ids, ctx)
            groups_str = self.combine_in_a_bracket(groups)
            if aggregated:
                aggregated_str = self.combine(aggregated, '')
                parts.append(
                    f"and the {aggregated_str} are grouped by {groups_str}"
                )
            else:
                parts.append(f"grouped by {groups_str}")

        if having_expr is not None:
            ctx.compact_predicates = True
            having_text = _apply_binding_aliases(self.verbalize(having_expr.condition, ctx), aliases)
            ctx.compact_predicates = False
            parts.append(f"having {having_text}")

        ob = expr._ordered_by_builder_
        if ob is not None:
            direction = "descending" if ob.descending else "ascending"
            ordered_text = _apply_binding_aliases(self.verbalize(ob.variable, ctx), aliases)
            parts.append(f"ordered by {ordered_text} ({direction})")

        return ", ".join(parts)

    # ── Result quantifiers (transparent wrappers) ──────────────────────────────

    def _v_An_(self, expr: An, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr._child_, ctx)

    def _v_The_(self, expr: The, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr._child_, ctx)

    def _v_ResultQuantifier_(self, expr: ResultQuantifier, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr._child_, ctx)

    # ── Filter wrappers (delegate to their condition) ──────────────────────────

    def _v_Where_(self, expr: Where, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr.condition, ctx)

    def _v_Having_(self, expr: Having, ctx: VerbalizationContext) -> str:
        return self.verbalize(expr.condition, ctx)

    def _v_GroupedBy_(self, expr: GroupedBy, ctx: VerbalizationContext) -> str:
        if expr.variables_to_group_by:
            groups = [self.verbalize(v, ctx) for v in expr.variables_to_group_by]
            return f"grouped by {', '.join(groups)}"
        return "grouped"

    def _v_OrderedBy_(self, expr: OrderedBy, ctx: VerbalizationContext) -> str:
        direction = "descending" if expr.descending else "ascending"
        return f"ordered by {self.verbalize(expr.variable, ctx)} ({direction})"

    # ── Grouped-by helpers ─────────────────────────────────────────────────────

    def _root_var_ids_(self, exprs) -> set:
        """Return the set of Variable._id_ values at the root of each expression."""
        ids: set = set()
        for e in exprs:
            current = e
            while isinstance(current, MappedVariable):
                current = current._child_
            if isinstance(current, Variable):
                ids.add(current._id_)
        return ids

    def _aggregated_noun_phrases_(
        self, query_expr, group_key_root_ids: set, ctx: VerbalizationContext
    ) -> list[str]:
        """
        Return plural noun phrases for variables in the selection that are not group keys.

        For Entity queries whose selected variable is an InstantiatedVariable, the
        child_vars are inspected: any child whose root Variable is not a group key is
        considered aggregated.  For other query shapes, non-group-key selected variables
        are used instead.
        """
        from krrood.entity_query_language.query.query import Entity
        from krrood.entity_query_language.core.variable import InstantiatedVariable

        texts: list[str] = []
        selected_var = query_expr.selected_variable if isinstance(query_expr, Entity) else None

        if isinstance(selected_var, InstantiatedVariable):
            for child_expr in selected_var._child_vars_.values():
                root = child_expr
                while isinstance(root, MappedVariable):
                    root = root._child_
                if isinstance(root, Variable) and root._id_ in group_key_root_ids:
                    continue
                texts.append(self._verbalize_plural_(child_expr, ctx))
        elif isinstance(query_expr, Query):
            for var in query_expr._selected_variables_:
                if var._id_ not in group_key_root_ids:
                    texts.append(self._verbalize_plural_(var, ctx))

        return texts

    # ── Fallback ───────────────────────────────────────────────────────────────

    def _v_default_(self, expr: SymbolicExpression, ctx: VerbalizationContext) -> str:
        return expr._name_


_default_verbalizer = EQLVerbalizer()


def verbalize_expression(expr) -> str:
    """
    Verbalize any EQL expression — including sub-expressions, conditions, predicates,
    and aggregators — into a human-readable English phrase.

    :param expr: Any ``SymbolicExpression`` instance.
    :return: A natural-language description of the expression.
    """
    if isinstance(expr, Query):
        expr.build()
    return _default_verbalizer.verbalize(expr)
