---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Verbalization Internals

This guide explains the architecture of the EQL verbalization subsystem for developers who want to understand, extend, or debug it.  End-user documentation lives in {doc}`../user/verbalization`.

## Overview

The verbalization subsystem translates any EQL symbolic expression into a human-readable English string (or a structured fragment tree that can be rendered in plain text, ANSI colour, or HTML).

The single entry point is {py:class}`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`, with {py:func}`~krrood.entity_query_language.verbalization.pipeline.verbalize_expression` as the plain-text shortcut:

```python
# Simplest — plain text, no colour (== VerbalizationPipeline.plain().verbalize(...))
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
text = verbalize_expression(query)

# Colour / layout / links — choose a pipeline factory
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline

VerbalizationPipeline.plain().verbalize(query)                    # plain prose
VerbalizationPipeline.ansi().verbalize(query)                     # ANSI true-colour prose
VerbalizationPipeline.ansi(hierarchical=True).verbalize(query)    # ANSI, indented bullets
VerbalizationPipeline.html(link_resolver=resolver).verbalize(query)  # HTML <span> + links

# Full control — construct a pipeline from a renderer + formatter directly
from krrood.entity_query_language.verbalization.rendering.formatter import HTMLFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import HierarchicalRenderer
pipeline = VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter(), resolver))
html = pipeline.verbalize(query)
```

Pass a shared {py:class}`~krrood.entity_query_language.verbalization.context.MicroplanningServices` to `verbalize` across calls to get cross-mention coreference (*"a Robot"* … *"the Robot"*).  A construct with no grammar rule raises {py:class}`~krrood.entity_query_language.verbalization.exceptions.UnverbalizableExpressionError` — coverage gaps fail loudly rather than degrading to a bare class name.

---

## Architecture: build → realise → render

```{mermaid}
graph LR
    A[EQL Expression] --> B[fold + realisation passes]
    B -- realised Fragment tree --> C[FragmentRenderer]
    C -- formatted string --> D[Output]
    E[grammar RULES / select] -. dispatch .-> B
    F[MicroplanningServices] -. services .-> B
    G[Formatter] -. markup .-> C
    H[SourceLinkResolver] -. URLs .-> C
```

### Layer 1 — Fragment building + realisation (`EQLVerbalizer.build`)

{py:class}`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer` is the internal fragment builder behind the pipeline (use it directly only when you want the fragment tree itself, e.g. in tests).  `build(expression, services)`:

1. **Folds** the EQL tree into a {py:class}`~krrood.entity_query_language.verbalization.fragments.base.Fragment` tree via the grammar (see [Rule dispatch](#rule-dispatch-the-fold)).
2. Runs the ordered **realisation passes** ({py:func}`~krrood.entity_query_language.verbalization.rendering.realization.realize_tree`): coreference → determiner → morphology → orthography (see [Realisation passes](#realisation-passes)).

It never produces strings — formatting is Layer 2/3.

### Layer 2 — Fragment rendering (`FragmentRenderer`)

{py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.FragmentRenderer` traverses the realised tree and produces a single string.

| Renderer | Output style |
|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer` | Flat prose; `BlockFragment`s joined inline |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer` | Indented bullet lists; each `BlockFragment` nesting level adds one indent |

### Layer 3 — Format markup (`Formatter`)

{py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter` injects format-specific characters into the renderer output.

| Formatter | Colour encoding | Space | Newline | Links |
|---|---|---|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter` | none | `" "` | `"\n"` | no |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.ANSIFormatter` | `\033[38;2;R;G;Bm` | `" "` | `"\n"` | OSC 8 |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.HTMLFormatter` | `<span style="color:…">` | `&nbsp;` | `<br>` | `<a href>` |

---

## Fragment Type Hierarchy

All verbalization output is expressed as a tree of `Fragment` subclasses.  There are **leaf** nodes (text), **structural** containers (hold children), and one **coreference marker** (`PossessiveChain`) the realisation passes consume and strip.

```{mermaid}
classDiagram
    class Fragment {
        <<abstract>>
    }
    class WordFragment {
        role-less text: articles, punctuation, connectives
    }
    class RoleFragment {
        text + SemanticRole + optional SourceReference (for hyperlinking)
    }
    class PhraseFragment {
        inline sequence of children joined by a separator
    }
    class NounPhrase {
        head + Definiteness + Number + modifiers + referent_id (a DP spec, lowered later)
    }
    class BlockFragment {
        header + item fragments (flattens or indents on render)
    }
    class PossessiveChain {
        a chain whose its/of form coreference decides (stripped after coreference)
    }
    Fragment <|-- WordFragment
    Fragment <|-- RoleFragment
    Fragment <|-- PhraseFragment
    Fragment <|-- NounPhrase
    Fragment <|-- BlockFragment
    Fragment <|-- PossessiveChain
```

`NounPhrase` is a *spec*, not a lowered phrase: rules emit it with grammatical features (`Definiteness`, `Number`) but **no** determiner; the [determiner pass](#realisation-passes) lowers it to a `PhraseFragment`.  The recursion helpers over this tree are {py:func}`~krrood.entity_query_language.verbalization.fragments.base.fold_fragment` (a catamorphism to any value — used by the renderer/flatten), {py:func}`~krrood.entity_query_language.verbalization.fragments.base.map_structural_children` / {py:func}`~krrood.entity_query_language.verbalization.fragments.base.map_fragment` (structure-preserving rebuilds — used by the realisation passes), and {py:func}`~krrood.entity_query_language.verbalization.fragments.base.flatten_fragment_to_plain_text`.

### SemanticRole and Colours

{py:class}`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole` is a *presentation* classification (decoupled from the EQL type taxonomy) that determines the colour applied by formatters, and is read by the morphology pass for copula agreement (`OPERATOR`).  Colours match the `QueryGraph.ColorLegend` palette.

| Role          | Example | Colour |
|---------------|---|---|
| `KEYWORD`     | *Find*, *If*, *such that* | yellow `#eded18` |
| `VARIABLE`    | *Robot*, *Employee 1* | cornflower blue |
| `AGGREGATION` | *sum of*, *number of* | red-orange `#F54927` |
| `OPERATOR`    | *is greater than*, *is* | orange `#ff7f0e` |
| `LOGICAL`     | *and*, *or*, *not*, *for all* | green `#2ca02c` |
| `LITERAL`     | `42`, `"hello"` | gray `#949292` |
| `ATTRIBUTE`   | *battery*, *tasks* | teal `#8FC7B8` |
| `PLAIN`       | *of*, *the*, *,* (`WordFragment`, role-less) | none |

### Building Fragments

Construct fragments **directly** (there are no factory helpers), and prefer the vocabulary constants for any fixed word so no English string is ever inlined:

```python
from krrood.entity_query_language.verbalization.fragments.base import (
    WordFragment, RoleFragment, PhraseFragment, NounPhrase, BlockFragment, oxford_comma,
)
from krrood.entity_query_language.verbalization.vocabulary.english import (
    Articles, Conjunctions, Punctuation,
)

# Fixed words / punctuation → vocabulary constants (avoids typos, carries the role)
the   = Articles.THE.as_fragment()
comma = Punctuation.COMMA.as_fragment()

# Coloured word with a source reference (for hyperlinking variables / attributes)
role_frag = RoleFragment.for_variable("Robot", robot_var)   # VARIABLE role + source link
attr_frag = RoleFragment.for_attribute(Robot, "battery")    # ATTRIBUTE role + link
op_frag   = RoleFragment.for_operator("is greater than")    # OPERATOR role, no link

# Inline sequence (default " " separator)
phrase = PhraseFragment(parts=[the, role_frag, op_frag])

# Oxford-comma join (serial comma for 3+; no comma for a pair — pass pair_comma=True to keep it)
list_frag = oxford_comma([frag_a, frag_b, frag_c], Conjunctions.AND.as_fragment())

# Noun-phrase spec (the determiner is added by the determiner pass, not here)
np = NounPhrase(head=RoleFragment.for_variable("Robot", robot_var), referent_id=robot_var._id_)

# Block structure (renders as bullets in HierarchicalRenderer)
block = BlockFragment(header=keyword_frag, items=[item1, item2])
```

---

## Rule dispatch (the fold)

{py:func}`~krrood.entity_query_language.verbalization.engine.fold` is the **single** place the EQL tree is recursed — an F-algebra catamorphism.  For each node it:

1. Checks `context.scope.binding_overrides` for the node's `_id_` — returns the override immediately if present (used for `InstantiatedVariable` field references).
2. {py:func}`~krrood.entity_query_language.verbalization.grammar.framework.phrase_rule.select`s the most-specific {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.phrase_rule.PhraseRule` and calls its `build`, handing it a fresh {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.phrase_rule.RuleContext` whose `child` re-enters `fold`.
3. If no rule matches, raises {py:class}`~krrood.entity_query_language.verbalization.exceptions.UnverbalizableExpressionError`.

A rule never recurses by hand — it calls `context.child(sub_expression)`.  `child` accepts optional per-recursion render flags (`number`, `inline`, `as_value`), bundled internally as a {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.phrase_rule.RenderOptions`; a single `root_context` factory builds the `RuleContext` so the continuation is defined in one place (a new flag is one field on `RenderOptions`, not a signature change at every call site).

**Query scoping is declarative.**  A rule whose construct *is* a query body (the top-level / nested `Entity` and `SetOf` rules) declares `enters_query_scope = True`; the engine then runs its `build` inside `context.configuration.query_depth_scope()`, so any Entity found within renders as a nested noun phrase (`query_depth > 0`) rather than emitting *"Find …"*.  Neither rules nor assemblers ever push the scope by hand.  `when` runs *outside* the scope — it guards on the rule's own position (`query_depth == 0` for the top-level form).

### Specificity

`select` ranks the rules whose `construct` matches (`isinstance`) and whose `when` guard passes, by the key **(construct MRO depth, guarded over unguarded, rule-class MRO depth)**, highest wins.  Specificity comes primarily from the *construct* class, so rules stay flat; the final component lets a rule that is a genuine *special case* of another express that by subclassing it (when both guards hold, the more-derived rule class wins).  For example `Literal <: Variable`, so `LiteralRule` (deeper construct) shadows `VariableRule`; a guarded boolean-attribute chain rule outranks the plain possessive chain rule.

The same most-specific-wins selection is reused, off the EQL tree, by the {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.specificity.SpecificityRule` registries — several small families of guarded alternatives ranked by class depth: the {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.forms.ConditionForm` surface-form registry, the {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.transforms.PredicateTransform` predicate registry, the {py:class}`~krrood.entity_query_language.verbalization.grammar.query.ranking.RankingForm` selection-ranking registry, and the {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.restriction.RestrictionSubjectRule` registry.  Adding an alternative is a new subclass; nothing else changes.

### How to Extend — A Worked Example

This walkthrough adds verbalization for a hypothetical `Between` operator (`between(x, lo, hi)`) rendering as *"x is between lo and hi"*.

#### Step 1 — Where does the code go?

The grammar is organised **per construct** under `grammar/<construct>/` (`terms`, `chain`, `conditions`, `query`, `inference`, `aggregation`, `clauses`, `instantiated`, `match`).  Each folder has a `rules.py`; the involved constructs also have a `planner.py` + `assembler.py`.

* **A simple construct** (one phrase, no content decisions) → add one `PhraseRule` subclass to the relevant construct's `rules.py` (a comparator-like operator belongs in {py:mod}`grammar.conditions.rules <krrood.entity_query_language.verbalization.grammar.conditions.rules>`).  That's the whole change.
* **A complex construct** (needs *what to say* analysis separate from *how to say it*) → add a `Planner` + `Assembler` pair in the construct's folder ({py:class}`~krrood.entity_query_language.verbalization.grammar.framework.planner.Planner` / {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.assembler.Assembler` bases) and have the rule's `build` delegate to the assembler.

`Between` is simple, so it's a single rule in `grammar/conditions/rules.py`.

#### Step 2 — Write the rule

Set `construct`, optionally `name`/`when` (and `enters_query_scope = True` if the construct is itself a query body), and implement `build(node, context)`.  Start the docstring with the **target string** in the established convention (*"x is between lo and hi"*) — that line is how readers get the output intuition at a glance, and the rule-doctest harness ({py:mod}`test_rule_doctests <test.krrood_test.test_eql.test_verbalization.test_rule_doctests>`) executes it:

```python
class BetweenRule(PhraseRule):
    """Verbalizes a Between operator as *"x is between lo and hi"*.

    >>> verbalize_expression(between(x, lo, hi))
    'x is between lo and hi'
    """

    construct = Between
    name = "between"

    def build(self, node, context: RuleContext) -> Fragment:
        return PhraseFragment(
            parts=[
                context.child(node.left),                   # recurse via the fold
                RangePhrases.IS_BETWEEN.as_fragment(),       # fixed phrase from the lexicon
                oxford_comma(
                    [context.child(node.lo), context.child(node.hi)],
                    Conjunctions.AND.as_fragment(),
                ),
            ]
        )
```

#### Step 3 — Registration is automatic

`RULES` is **auto-discovered**: {py:data}`~krrood.entity_query_language.verbalization.grammar.framework.registry.RULES` instantiates every concrete `PhraseRule` subclass via `recursive_subclasses(PhraseRule)` (abstract bases filtered out).  Importing the rule module is enough — the registry module imports every construct package so the subclasses exist before `RULES` is built.  (`select` is specificity-based, so definition order is irrelevant.)

#### Step 4 — Recurse with `context.child`, decide with `context` services

`context.child(sub_expression)` re-enters the fold, so sub-expressions get coreference, binding overrides, and pronoun resolution for free.  Reach cross-cutting state only through the `RuleContext` services — `context.refer` (referring expressions), `context.scope` (binding), `context.configuration` (render flags), `context.microplan` (the plan read model) — never reach into the service internals directly.  Never call `verbalize_expression(child)` from a rule: that starts a fresh services bundle and breaks coreference.

#### Step 5 — Use constructors + vocabulary constants

Build fragments with the constructors (`WordFragment`/`RoleFragment`/`PhraseFragment`/`NounPhrase`), and source every fixed word from the lexicon (`Keywords`, `Copulas`, `Operators`, `Punctuation`, …) via `.as_fragment()`.  No English string — including punctuation — belongs in a rule.

| Enum | Example | `.as_fragment()` role |
|---|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Keywords` | `Keywords.FIND` | KEYWORD |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Copulas` | `Copulas.IS` / `Copulas.for_number(n)` | OPERATOR |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Operators` | `Operators.from_callable(op.le).select(...)` | OPERATOR |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Logicals` | `Logicals.FOR_ALL` | LOGICAL |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Aggregations` | `Aggregations.COUNT` | AGGREGATION |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Articles` | `Articles.THE` | PLAIN |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Conjunctions` | `Conjunctions.AND` | PLAIN |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Prepositions` | `Prepositions.OF` | PLAIN |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Punctuation` | `Punctuation.COMMA` | PLAIN |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Pronouns` | `Pronouns.ITS` | PLAIN |

For coloured / linked fragments referencing a Python class or attribute, use the `RoleFragment` class methods:

```python
RoleFragment.for_variable("Robot", var)       # VARIABLE role + SourceReference to type
RoleFragment.for_attribute(Robot, "battery")  # ATTRIBUTE role + SourceReference to attr
RoleFragment.for_operator("is greater than")  # OPERATOR role, no link
```

#### Step 6 — Write a test

Add a golden-string case to `test/krrood_test/test_eql/test_verbalization/test_verbalization.py`:

```python
def test_verbalize_between():
    assert verbalize_expression(between(x, lo, hi)) == "x is between lo and hi"
```

```bash
pytest test/krrood_test/test_eql/test_verbalization -x
```

---

## The planner / assembler split (complex constructs)

Queries, inference rules, instantiated variables, matches, chains, and aggregations are too involved for a one-method rule, so they split *what to say* from *how to say it*:

* a {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.planner.Planner` performs pure structural analysis into a frozen plan (no fragments, no context mutation, no recursion).  Plans are computed once and shared via the {py:class}`~krrood.entity_query_language.verbalization.microplanning.microplan.Microplan` read model (`context.microplan.plan_for(node, SomePlanner)`);
* an {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.assembler.Assembler` realises that plan into fragments (it owns recursion and any render-scope mutations).

The rule's `build` just calls `XAssembler(context).assemble(node)` (or, for plan-less constructs, a dedicated assembler method).  The orchestrating {py:class}`~krrood.entity_query_language.verbalization.grammar.query.assembler.QueryAssembler` delegates cohesive sub-forms to their own components: the trailing clauses to {py:mod}`~krrood.entity_query_language.verbalization.grammar.clauses` (`assembler` + `composer`), the WHERE partition to the {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.forms.ConditionForm` registry via {py:func}`~krrood.entity_query_language.verbalization.grammar.conditions.forms.as_subject_restrictions`, and the aggregation value-subquery to {py:class}`~krrood.entity_query_language.verbalization.grammar.aggregation.assembler.AggregationValueAssembler`.

A query *searches* (Find) or *presents* (Report) by one plan-level shape, not a render-time `isinstance`: the {py:class}`~krrood.entity_query_language.verbalization.grammar.query.planner.QueryPlanner` sets a {py:class}`~krrood.entity_query_language.verbalization.grammar.query.planner.ReportPlan` (with a {py:class}`~krrood.entity_query_language.verbalization.grammar.query.planner.ReportKind`) when the query presents results rather than hunting for a match.  Report-ness and conditions are **orthogonal** axes — Find/Report answers *"hunt or present?"*, the WHERE answers *"over which subset?"* — so a report may still be filtered.  Two kinds present results: `AGGREGATION` (a `set_of` computing an aggregate) and `ORDERING` (any `ordered_by` listing, since ordering ranges over the whole result set).  The opening verb (`QueryAssembler._verb`) and the subject's number both derive from the plan, so the same body builders serve both modes: a search reads as natural prose *"Find a and b"* (Oxford-comma, no code-like parentheses); an `AGGREGATION` report opens with *"Report …"* and, when grouped, fronts its grouping as *"For each <key>, report …"* (the result's row dimension stated first — reads as the GROUP BY idiom and drops the redundant key column and trailing *"grouped by"* clause, keys rendered as the bare group label); an `ORDERING` report opens with *"Report …"* and renders the subject **plural** (*"Report Employees ordered by their salary"* — a plural subject pronominalises to *"their"* via the same `Number` plumbing the ranking surface uses).  A *ranked* query (`limit`) is excluded from the report shape and keeps the *"Find the top three …"* ranking form, whose count-bearing pre-head is a distinct surface.  Within a selection list, contiguous plain attributes sharing one owner fold into a coordinated genitive (`coordinated_genitive`: *"the department and salary of an Employee"*, not *"… and its salary"*) — the assembler groups by the attributes' shared `_child_` owner and renders the owner once.  *"report"* is held lowercase in the vocabulary and capitalised by the assembler only when it opens the sentence (there is no global sentence-case pass — leading capitals are positional in the keywords).

Navigation chains analyse the chain once into a {py:class}`~krrood.entity_query_language.verbalization.grammar.chain.planner.ChainPlan` ({py:class}`~krrood.entity_query_language.verbalization.grammar.chain.planner.ChainPlanner`); {py:class}`~krrood.entity_query_language.verbalization.grammar.chain.assembler.ChainAssembler` then renders the plural / boolean-predicative / possessive form, with the possessive/pronominal surface built by {py:mod}`~krrood.entity_query_language.verbalization.microplanning.possessive` over {py:func}`~krrood.entity_query_language.verbalization.navigation_path.build_path_parts`.  The boolean predicative *"<navigation> is <attribute>"* does not re-render its navigation prefix: the prefix is the chain minus its boolean terminal (`terminal._child_`), recursed through the standard grammar, so it reuses the very possessive / relational / pronominal surfaces above — and pronominalises to the discourse subject (*"the Robot to which **it** is assigned is operational"*) for free.

A hop whose attribute is a **relation** (a past participle + preposition, e.g. `assigned_to`) renders as a relative clause — *"the `<Type>` `<prep>` which `<owner>` is `<participle>`"* (*"the Robot to which a Mission is assigned"*) — instead of the genitive *"the `<attr>` of `<owner>`"*.  The head is the hop's `_type_`; the verb is split by {py:func}`~krrood.entity_query_language.verbalization.grammar.conditions.recognition.relational_verb` into participle + preposition (one recognizer, with `relational_verb_phrase` — the stranded *"assigned to"* absence uses — a thin wrapper over it), so the preposition pied-pipes before *which*.  Detection is *injected* into `build_path_parts` (the detector is passed by `ChainPlanner`, so `navigation_path` stays free of any grammar import), and the relational flag rides on the {py:class}`~krrood.entity_query_language.verbalization.navigation_path.PathStep` (`relation`) so every navigation surface — possessive, boolean prefix, deferred chain, and pronominal — renders it uniformly through the shared `possessive`/`pronominal` hop builders.  Keeping the owner the verb's subject means the meaning never flips for agentive relations (*"the Person by which a Book is owned"*); when the owner is the discourse subject it pronominalises to the nominative *it* / *they*, the copula agreeing (*"… to which it is assigned"* / *"… to which they are assigned"*).  The relative clause is itself a *referring* {py:class}`~krrood.entity_query_language.verbalization.fragments.base.NounPhrase` keyed on the navigated value node's id (repeated source-level navigations share that node, so they share the id): the first mention spells the clause out, and a repeat of the same navigation reduces to a bare *"the Robot"* through the ordinary coreference downgrade — so *"the battery of the Robot to which it is assigned … and the power of **the Robot**"* says the clause once.  This is why the coreference pass *walks* the built chain (rather than returning it raw): the freshly-emitted relative-clause noun phrases must pass through first/repeat resolution.  An attribute reached *through* the **local centre** — the relational referent the immediately preceding chain was *about* (the owner of its outermost attribute) — goes one step further and pronominalises: *"the power of the Robot to which it is assigned"* becomes *"its power"* (`_relational_possessive`, mirroring the subject's *"its …"* but anchored on the relational referent rather than the chain root).  The centre is the one piece of running state the pass keeps besides `_seen`: each chain sets it to its own topic after resolving, and a chain about something else clears it — so *"its"* binds only to the referent named *directly before*, and stays unambiguous even among numbered referents (*"Robot 1 … its power … Robot 2"*; but *"Robot 1 … Robot 2 … the power of Robot 1"*, never a stranded *"its"* reaching across the intervening *Robot 2*).  This is centering theory (Grosz, Joshi & Weinstein 1995) in its smallest useful form.  Two *distinct* relational referents of one type (two missions' robots) would both reduce to an ambiguous *"the Robot"*, so they are numbered *"Robot 1"* / *"Robot 2"* — the same {py:class}`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions` disambiguation map that numbers colliding *variables*, now also counting relational referents (keyed on the value node id).  A relational referent has no rule to label it (its clause is built in the microplanner), so the number is stamped on by the coreference pass from `numbered_labels`, and a numbered (`BARE`) clause reduces to its bare label on repeat.

Conditions are owned by {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.assembler.ConditionAssembler` (the surface forms) plus the {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.forms.ConditionForm` registry (which form attaches where).  Its `predicate` form does not hard-code the special cases either: it dispatches over the {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.transforms.PredicateTransform` registry, so the generic *"<left> <op> <right>"* is one transform alongside the absence and boolean-polarity (*"is [not] <attr>"*) forms.  Pure structural predicates live in {py:mod}`~krrood.entity_query_language.verbalization.grammar.conditions.recognition` and the operator phrase in {py:func}`~krrood.entity_query_language.verbalization.grammar.conditions.operator_phrase.comparator_operator`.

`render_absence` (a `== None` comparison) picks its wording by the attribute's *part of speech*: a plain noun reads *"<owner> has no <attribute>"*, while a **relational** attribute — a past participle plus a preposition (`assigned_to`, `owned_by`) recognised by {py:func}`~krrood.entity_query_language.verbalization.grammar.conditions.recognition.relational_verb_phrase` — reads as a passive verb naming the attribute's declared type, *"<owner> has not been assigned to any <Type>"*.  The participle test is the deterministic, offline {py:func}`~krrood.entity_query_language.verbalization.morphology.is_past_participle` (a `lemminflect` dictionary+rule lookup, **not** a statistical POS tagger), so it is reproducible and needs no model download; nouns that merely end in a preposition (`color_in`) are correctly left as *"has no …"*.

---

## Realisation passes

After the fold, {py:func}`~krrood.entity_query_language.verbalization.rendering.realization.realize_tree` runs four ordered passes over the fragment tree:

1. **{py:class}`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor`** — resolves referring expressions in document order and strips `PossessiveChain` markers (below).
2. **{py:class}`~krrood.entity_query_language.verbalization.rendering.determiner_processor.DeterminerProcessor`** — lowers every `NounPhrase` to a determiner-bearing `PhraseFragment` via the concord table (INDEFINITE×SINGULAR → *a/an*; INDEFINITE×PLURAL → bare; DEFINITE → *the*; UNIQUE → *the unique*; BARE → no determiner), and tags the head with its `Number`.
3. **{py:class}`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor`** — inflects every leaf tagged `Number.PLURAL` (pluralise nouns; copula suppletion *is*→*are*).  Domain exceptions can be registered via {py:func}`~krrood.entity_query_language.verbalization.morphology.register_plural` / {py:func}`~krrood.entity_query_language.verbalization.morphology.register_indefinite_article`.
4. **{py:class}`~krrood.entity_query_language.verbalization.rendering.orthography_processor.OrthographyProcessor`** — removes the space adjacent to glued punctuation (a `Punctuation` token carries a spacing feature), so rules emit *","* / *"("* / *")"* as ordinary, normally-separated tokens and still get *"x, y"* / *"(x)"*.

### Coreference

Rules emit the *first-mention* form — a `NounPhrase` tagged with a `referent_id` (and `Definiteness` INDEFINITE / BARE, or UNIQUE for a *"the unique X"* `the(entity(...))` selection), with variable-rooted chains emitted as a `PossessiveChain`.  The discourse focus (which referent is the current subject, for *"its …"*) is **not** marked by rules; it is projected once from the shared plan read model by {py:class}`~krrood.entity_query_language.verbalization.rendering.discourse.DiscourseModel` (`DiscourseModel.from_expression(expression, microplan)`), and the coreference pass consults it.  The pass then, in output order:

* keeps the first mention of a referent (e.g. *"a Robot"*) and marks it introduced;
* downgrades a repeat **singular** mention to definite, dropping the first-mention modifiers (*"the Robot"*; a UNIQUE *"the unique Robot"* likewise downgrades to a plain DEFINITE *"the Robot"*); numbered labels (*"Robot 2"*, `BARE`) never downgrade;
* renders a `PossessiveChain` as *"its …"* when its root is the discourse subject, else as the possessive *"the … of …"*;
* reduces a repeat of a query's **selected quantity** — an aggregation's measured attribute, which `DiscourseModel` also collects (`is_selected_quantity`) — to a bare *"the battery"*, so a WHERE on the very attribute being aggregated does not repeat the whole possessive (*"the average of the battery of the Robot to which a Mission is assigned such that **the battery** is greater than 5"*). The `PossessiveChain` carries its own `node_id` for this match; the first mention (in the aggregate) spells out in full, the repeat reduces.

The local centre (`_chain_topic`) is the **grammatical subject of the clause just said** — the antecedent a reader binds *"its"* to (centering theory: the backward-looking centre is the highest-ranked entity, subject before oblique). A relational referent is that subject only when nothing is predicated *of an attribute of it*: a boolean predicative (*"the Robot to which it is assigned is operational"*) or a bare relational mention — there the referent heads the subject phrase, so a following attribute reads *"its battery"*. But a genitive attribute (*"the **battery** of the Robot … is greater than 5"*) is itself the subject head; the Robot is an oblique inside it. So a following *"its power"* would bind to the battery (*"battery power"*!), not the Robot — wrong — and the owner is spelled out instead: *"the power of the Robot"*. This is one rule, not a special case: an aggregation's measured quantity (*"the average of the **battery** …"*) is likewise headed by the attribute, so it clears the centre the same way. One refinement keeps a *run* consistent: a clause that is itself an *"its …"* continuation of the current centre **keeps** it (a centering CONTINUE — the pronoun signals the topic persists), so a string of attributes on one referent reads uniformly *"its battery … its power"* rather than mixing *"its battery … the power of the Robot"*. The asymmetry that protects the rule above survives: *"its"* can only *start* after the referent was a clause subject, but once started it continues.

The build is therefore free of in-fold coreference mutation.  {py:class}`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions` holds only the pre-computed disambiguation map and a `seen` **set** of introduced referent ids — the latter solely to *seed* the pass across builds sharing one services bundle.  `numbered_label(var)` is the single source of the disambiguation lookup; `noun_for_parts(var)` builds on it for the first-mention `Definiteness`.

### Disambiguation map

Built by `MicroplanningServices.from_expression(expression)`, which pre-scans the tree.  Types with a single variable keep the plain type name; collisions get numbered labels:

```
Robot    (single)  →  "Robot"
Apple 1  (first)   →  "Apple 1"
Apple 2  (second)  →  "Apple 2"
```

---

## Microplanning services

A single {py:class}`~krrood.entity_query_language.verbalization.context.MicroplanningServices` threads four single-responsibility services through one build; rules reach them via the `RuleContext` properties.

| Service (`RuleContext` accessor) | Responsibility |
|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions` (`context.refer`) | disambiguation map + introduced-referent `seen` set |
| {py:class}`~krrood.entity_query_language.verbalization.microplanning.binding_scope.BindingScope` (`context.scope`) | deferred-constraint frames + field-reference overrides |
| {py:class}`~krrood.entity_query_language.verbalization.microplanning.config.RenderConfiguration` (`context.configuration`) | render-mode flags (query depth, compact predicates) |
| {py:class}`~krrood.entity_query_language.verbalization.microplanning.microplan.Microplan` (`context.microplan`) | the plan read model — each node's plan computed once and shared (lazy / memoised) |

Value lexicalisation (Python value → text — e.g. `None` → *"nothing"*, an enum member → its name, a `datetime` → *"May 23, 2026"*) is the pure module-level function {py:func}`~krrood.entity_query_language.verbalization.value_lexicon.value_phrase`, not a method on the services.

### Constraint frames (BindingScope)

Used by the `InstantiatedVariable` path: when an `Entity` is a chain root inside an `InstantiatedVariable`, its WHERE condition is *deferred* into a frame rather than verbalized inline.  After all binding overrides are registered, the deferred expressions are emitted as a *"such that …"* clause.

```python
context.scope.push_constraint_frame()
context.scope.defer_constraint(expression)
deferred = context.scope.pop_constraint_frame()
```

### Binding overrides (BindingScope)

`context.scope.binding_overrides` maps `expression._id_` → `Fragment`.  `fold` checks it before dispatch, so when a bound variable appears again as a WHERE value it reuses the *"the field of the Type"* fragment instead of re-verbalizing the raw variable.

---

## Coordination (aggregation / conjunction reduction)

{py:mod}`~krrood.entity_query_language.verbalization.microplanning.coordination` owns the EQL-level conjunction-reduction microplanning task.  {py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.reduce_conjuncts` is the entry point a caller applies to a flat conjunct list; it folds complementary lower/upper bound comparisons on one chain into a {py:class}`~krrood.entity_query_language.verbalization.microplanning.coordination.RangeFold` ({py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.fold_range_pairs`, rendered *"… is between lo and hi"* by {py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.build_between`), and folds co-indexed comparisons across two prefixes into a {py:class}`~krrood.entity_query_language.verbalization.microplanning.coordination.CoindexedFold` (*"the begin and end … have the same month and year"*).  Fragment-level Oxford-comma joining is {py:func}`~krrood.entity_query_language.verbalization.fragments.base.oxford_comma`.

---

## Source References and Link Resolvers

{py:class}`~krrood.entity_query_language.verbalization.fragments.source_reference.SourceReference` identifies the Python entity a `RoleFragment` represents:

```python
SourceReference.for_type(Robot)                 # class reference
SourceReference.for_attribute(Robot, "battery") # attribute reference
```

A {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver` maps these to URLs; the built-in {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.AutoAPIResolver` builds Sphinx AutoAPI URLs:

```python
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import AutoAPIResolver
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline

resolver = AutoAPIResolver.for_package("krrood")
pipeline = VerbalizationPipeline.html(link_resolver=resolver)
```

The resolver is passed to the renderer at construction (via the `VerbalizationPipeline` factories).  When a renderer meets a `RoleFragment` with a non-`None` `source_reference` and a resolver, it wraps the coloured text in a hyperlink.

---

## Module map

| Area | Modules |
|---|---|
| Engine | `engine.py` (`fold`), `verbalizer.py` (`EQLVerbalizer`), `pipeline.py` (`VerbalizationPipeline`, `verbalize_expression`), `context.py` (`MicroplanningServices`), `exceptions.py` (`UnverbalizableExpressionError`) |
| Fragment IR | `fragments/base.py` (the `Fragment` hierarchy + `fold_fragment` / `map_structural_children` / `map_fragment` / `oxford_comma` / `flatten_fragment_to_plain_text`), `fragments/features.py`, `fragments/roles.py`, `fragments/source_reference.py` |
| Lexicon | `vocabulary/english.py`, `vocabulary/words.py` — **all** English words/phrases/punctuation |
| Grammar framework | `grammar/framework/` — `phrase_rule` (`PhraseRule`, `RuleContext`, `select`), `registry` (`RULES` via `recursive_subclasses`), `specificity` (`SpecificityRule`, `most_applicable`), `assembler` (`Assembler` base), `planner` (`Planner` base) |
| Grammar (per construct) | `grammar/<construct>/` — `terms`, `chain`, `conditions`, `query`, `inference`, `aggregation`, `clauses`, `instantiated`, `match`; each has `rules.py`, and the involved ones a `planner.py` + `assembler.py` |
| Conditions | `grammar/conditions/` — `rules`, `assembler` (`ConditionAssembler`), `forms` (`ConditionForm` registry, `place` / `as_subject_restrictions`), `transforms` (`PredicateTransform` registry), `recognition`, `operator_phrase`, `restriction` |
| Microplanning | `microplanning/` — `referring`, `binding_scope`, `config` (`RenderConfiguration`), `microplan` (`Microplan`), `coordination`, `possessive` |
| Rendering | `rendering/` — `realization` (the passes), `coreference_processor`, `determiner_processor`, `morphology_processor`, `orthography_processor`, `discourse` (`DiscourseModel`), `renderer`, `formatter`, `source_link_resolver` |
| Utilities | `morphology.py` (word-morphology facade over `inflect` + `lemminflect`: pluralise / article / ordinal / `is_past_participle`), `navigation_path.py` (`build_path_parts`, `PathStep`), `value_lexicon.py` (`value_phrase`), `subquery.py`; chain walking is `core.expression_structure.walk_chain` |

---

## How to Add a New Output Format

Subclass {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter` (and optionally a renderer):

```python
from dataclasses import dataclass
from krrood.entity_query_language.verbalization.rendering.formatter import Formatter
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole

@dataclass
class MarkdownFormatter(Formatter):
    """Renders colour as Markdown bold (no true colour in plain Markdown)."""

    def colorize(self, text: str, role: SemanticRole) -> str:
        if role in (SemanticRole.KEYWORD, SemanticRole.VARIABLE):
            return f"**{text}**"
        return text

    @property
    def space(self) -> str:
        return " "

    @property
    def newline(self) -> str:
        return "\n"

    def wrap_link(self, text: str, url: str) -> str:
        return f"[{text}]({url})"
```

Then build a pipeline from it:

```python
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.renderer import ParagraphRenderer

text = VerbalizationPipeline(ParagraphRenderer(MarkdownFormatter())).verbalize(query)
```

---

## API Reference

### Entry points & engine

- {py:class}`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline` — the single public entry point (`.plain()` / `.ansi()` / `.html()` + `.verbalize`)
- {py:func}`~krrood.entity_query_language.verbalization.pipeline.verbalize_expression` — plain-text shortcut
- {py:class}`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer` — internal fragment builder (`build`)
- {py:func}`~krrood.entity_query_language.verbalization.engine.fold` / {py:class}`~krrood.entity_query_language.verbalization.exceptions.UnverbalizableExpressionError`
- {py:class}`~krrood.entity_query_language.verbalization.context.MicroplanningServices`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.features.Definiteness` / {py:class}`~krrood.entity_query_language.verbalization.fragments.features.Number`

### Grammar / dispatch

- {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.phrase_rule.PhraseRule` / {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.phrase_rule.RuleContext` / {py:func}`~krrood.entity_query_language.verbalization.grammar.framework.phrase_rule.select`
- {py:data}`~krrood.entity_query_language.verbalization.grammar.framework.registry.RULES`
- {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.specificity.SpecificityRule` / {py:func}`~krrood.entity_query_language.verbalization.grammar.framework.specificity.most_specific`
- {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.assembler.Assembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.framework.planner.Planner`
- {py:class}`~krrood.entity_query_language.verbalization.grammar.query.assembler.QueryAssembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.chain.assembler.ChainAssembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.aggregation.assembler.AggregationValueAssembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.match.assembler.MatchAssembler`
- {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.assembler.ConditionAssembler` / {py:class}`~krrood.entity_query_language.verbalization.grammar.conditions.forms.ConditionForm` / {py:func}`~krrood.entity_query_language.verbalization.grammar.conditions.forms.as_subject_restrictions`
- {py:func}`~krrood.entity_query_language.verbalization.grammar.conditions.operator_phrase.comparator_operator`

### Fragment hierarchy

- {py:class}`~krrood.entity_query_language.verbalization.fragments.base.Fragment` / `WordFragment` / `RoleFragment` / `PhraseFragment` / `NounPhrase` / `BlockFragment` / `PossessiveChain`
- {py:func}`~krrood.entity_query_language.verbalization.fragments.base.fold_fragment` / {py:func}`~krrood.entity_query_language.verbalization.fragments.base.map_structural_children` / {py:func}`~krrood.entity_query_language.verbalization.fragments.base.map_fragment` / {py:func}`~krrood.entity_query_language.verbalization.fragments.base.oxford_comma`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole` / {py:class}`~krrood.entity_query_language.verbalization.fragments.source_reference.SourceReference`

### Rendering passes & output

- {py:func}`~krrood.entity_query_language.verbalization.rendering.realization.realize_tree`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.coreference_processor.CoreferenceProcessor` / {py:class}`~krrood.entity_query_language.verbalization.rendering.determiner_processor.DeterminerProcessor` / {py:class}`~krrood.entity_query_language.verbalization.rendering.morphology_processor.MorphologyProcessor` / {py:class}`~krrood.entity_query_language.verbalization.rendering.orthography_processor.OrthographyProcessor`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.discourse.DiscourseModel`
- {py:mod}`~krrood.entity_query_language.verbalization.microplanning.possessive`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer` / `HierarchicalRenderer`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter` / `ANSIFormatter` / `HTMLFormatter`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver` / `AutoAPIResolver`

### Microplanning & utilities

- {py:class}`~krrood.entity_query_language.verbalization.microplanning.referring.ReferringExpressions` / {py:class}`~krrood.entity_query_language.verbalization.microplanning.binding_scope.BindingScope` / {py:class}`~krrood.entity_query_language.verbalization.microplanning.config.RenderConfiguration` / {py:class}`~krrood.entity_query_language.verbalization.microplanning.microplan.Microplan`
- {py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.reduce_conjuncts` / {py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.fold_range_pairs` / {py:func}`~krrood.entity_query_language.verbalization.microplanning.coordination.build_between`
- {py:func}`~krrood.entity_query_language.verbalization.navigation_path.build_path_parts` / {py:class}`~krrood.entity_query_language.verbalization.grammar.chain.planner.ChainPlan`

### Vocabulary (lexicon)

- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Keywords` / `Logicals` / `Aggregations` / `Copulas` / `Operators` / `Articles` / `Prepositions` / `Conjunctions` / `Punctuation` / `Pronouns` / `RangePhrases` / `CoindexedOperators` / `CoindexedPhrases` / `Absence` / `NonExistence` / `SetMembership` / `Specificity` / `ExistentialPhrase` / `FallbackNouns`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.words.PlainWord` / `RoleWord` / `VocabEnum`
