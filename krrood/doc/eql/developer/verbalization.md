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

The verbalization subsystem translates any EQL symbolic expression into a human-readable English string or a structured fragment tree that can be rendered in plain text, ANSI colour, or HTML.

The entry points are:

```python
# Simplest — plain text, no colour
from krrood.entity_query_language.verbalization.verbalizer import verbalize_expression
text = verbalize_expression(query)

# Colour and layout — pass a renderer to verbalize_expression
from krrood.entity_query_language.verbalization.rendering.formatter import HTMLFormatter, ANSIFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import HierarchicalRenderer, ParagraphRenderer

# ANSI-coloured prose
text = verbalize_expression(query, renderer=ParagraphRenderer(ANSIFormatter()))

# HTML, indented bullets, with hyperlinks
text = verbalize_expression(query, renderer=HierarchicalRenderer(HTMLFormatter(), resolver))

# Full control — VerbalizationPipeline is the underlying implementation
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
pipeline = VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter(), resolver))
html = pipeline.verbalize(query)
```

---

## Architecture: the Three-Layer Pipeline

```{mermaid}
graph LR
    A[EQL Expression] --> B[EQLVerbalizer]
    B -- VerbFragment tree --> C[FragmentRenderer]
    C -- formatted string --> D[Output]
    E[RuleEngine] -. dispatches .-> B
    F[VerbalizationContext] -. shared state .-> B
    G[Formatter] -. markup .-> C
    H[SourceLinkResolver] -. URLs .-> C
```

### Layer 1 — Fragment Building (`EQLVerbalizer`)

{py:class}`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer` walks the EQL expression tree and produces a parallel tree of
{py:class}`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment` nodes.

It does not produce strings directly. Every call to `build(expression, context)` returns a `VerbFragment` — rendering (plain/ANSI/HTML) is deferred to Layer 2.

`EQLVerbalizer` delegates to a single {py:class}`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine`,
which dispatches every expression to the first matching
{py:class}`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule`.
Rules are the **only** extension unit — there are no sub-verbalizer classes.
Each expression type has exactly one rule as the single source of truth for its
verbalization.

### Layer 2 — Fragment Rendering (`FragmentRenderer`)

{py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.FragmentRenderer` traverses the `VerbFragment` tree and produces a single string.

Two concrete renderers:

| Renderer | Output style |
|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer` | Flat prose; BlockFragments joined inline |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer` | Indented bullet lists; each BlockFragment nesting level adds one indent |

### Layer 3 — Format Markup (`Formatter`)

{py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter` injects format-specific characters into the renderer output.

| Formatter | Colour encoding | Space | Newline | Links |
|---|---|---|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter` | none | `" "` | `"\n"` | no |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.ANSIFormatter` | `\033[38;2;R;G;Bm` | `" "` | `"\n"` | OSC 8 |
| {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.HTMLFormatter` | `<span style="color:…">` | `&nbsp;` | `<br>` | `<a href>` |

---

## Fragment Type Hierarchy

All verbalization output is expressed as a tree of `VerbFragment` subclasses before rendering.  Understanding this hierarchy is essential for writing new rules or renderers.

```{mermaid}
classDiagram
    class VerbFragment {
        <<abstract>>
    }
    class WordFragment {
        plain text: articles, punctuation, connectives
    }
    class RoleFragment {
        text + SemanticRole + optional SourceRef (for hyperlinking)
    }
    class PhraseFragment {
        inline sequence of child fragments joined by a separator
    }
    class BlockFragment {
        header + list of item fragments (flattens or indents on render)
    }
    VerbFragment <|-- WordFragment
    VerbFragment <|-- RoleFragment
    VerbFragment <|-- PhraseFragment
    VerbFragment <|-- BlockFragment
```

### SemanticRole and Colours

{py:class}`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole` determines the colour applied by formatters.  Colours match the `QueryGraph.ColorLegend` palette for visual consistency with query graph visualizations.

| Role          | Example | Colour |
|---------------|---|---|
| `KEYWORD`     | *Find*, *If*, *such that* | yellow `#eded18` |
| `VARIABLE`    | *Robot*, *Employee 1* | cornflower blue |
| `AGGREGATION` | *sum of*, *number of* | red-orange `#F54927` |
| `OPERATOR`    | *is greater than*, *is* | orange `#ff7f0e` |
| `LOGICAL`     | *and*, *or*, *not*, *for all* | green `#2ca02c` |
| `LITERAL`     | `42`, `"hello"` | gray `#949292` |
| `ATTRIBUTE`   | *battery*, *tasks* | teal `#8FC7B8` |
| `PLAIN (Not a Role)`      | *of*, *the*, *,* | none |

### Building Fragments

Convenience factory methods avoid repetitive construction:

```python
from krrood.entity_query_language.verbalization.fragments.base import (
    WordFragment, RoleFragment, PhraseFragment, BlockFragment,
    join_with, oxford_and,
)
from krrood.entity_query_language.verbalization.vocabulary.english import Articles
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole

# Plain word (`The` is a plain word)
# can be constructed directly… 
word = WordFragment(text="the") 
# or via vocabulary constants (Preferred way, as it avoids typos):
word = Articles.THE.as_fragment()

# Coloured word with source object reference (for variable, and attribute)
role_frag = RoleFragment.for_variable("Robot", robot_var)      # VARIABLE role + source link
attr_frag = RoleFragment.for_attribute("battery", Robot)  # ATTRIBUTE role + link
op_frag   = RoleFragment.for_operator("is greater than")       # OPERATOR role, no link

# Inline sequence
phrase = PhraseFragment([word, role_frag, op_frag])

# Oxford-comma join
list_frag = oxford_and([frag_a, frag_b, frag_c], Conjunctions.AND.as_fragment())

# Block structure (renders as bullets in HierarchicalRenderer)
block = BlockFragment(header=keyword_frag, items=[item1, item2])
```

---

## Rule Dispatch Mechanism

`EQLVerbalizer.build(expression, context)` delegates to `RuleEngine.build(expression, context, verbalizer)`, which:

1. Checks `context.binding_overrides` for the expression's `_id_` — if found, returns the override fragment immediately (see [Binding Overrides](#binding-overrides) below).
2. Iterates the sorted rule list and calls `rule_cls.applies(expression, context)`.
3. Calls `rule_cls.transform(expression, context, verbalizer)` on the first matching rule.
4. Falls back to `WordFragment(text=expression._name_)` when no rule matches.

### MRO-Depth Sorting

Rules are sorted by `__mro__.index(VerbalizationRule)` (descending) at `RuleEngine` construction time.  A deeper MRO index means the class is more specific (closer to `VerbalizationRule` in the hierarchy).  This ensures subclass rules shadow parent rules without requiring explicit priority integers.

Example from `rules/logical.py`:

```{mermaid}
classDiagram
    class NotRule {
        depth 1 — generic fallback
    }
    class NotComparatorRule {
        depth 2 — tried first: Not(Comparator)
    }
    class NotBoolAttrRule {
        depth 2 — tried first: Not(bool Attribute)
    }
    NotRule <|-- NotComparatorRule
    NotRule <|-- NotBoolAttrRule
```

### How to Extend — A Worked Example

This walkthrough adds verbalization for a hypothetical ``Between`` operator
(``between(x, lo, hi)``) that should render as *"x is between lo and hi"*.

#### Step 1 — Where Does the Code Go?

Ask these questions in order:

1. **Is this a new expression *family*?** (e.g. a new operator type that has no rule
   module yet) → Create a new file in ``rules/`` and import it from
   ``rules/registry.py`` (any import of the module triggers auto-registration).
2. **Is it a variant of an existing expression?** (e.g. a more-specific case of
   ``Comparator``, ``AND``, or ``InstantiatedVariable``) → Subclass the existing rule
   in the same file.  The subclass rule will have deeper MRO and take priority
   automatically.
3. **Is it a new expression within an existing family?** → Add the rule class to the
   existing ``rules/<family>.py`` file.

For ``Between`` — a comparator-like range constraint — we create a rule in
``rules/comparator.py`` (or its own file if we prefer).

#### Step 2 — Write the Rule

Subclass {py:class}`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule`
(or an existing intermediate rule for automatic higher priority) and implement
``applies()`` and ``transform()``:

```python
from krrood.entity_query_language.verbalization.rule_engine import VerbalizationRule
from krrood.entity_query_language.verbalization.fragments.factory import phrase, word
from krrood.entity_query_language.verbalization.vocabulary.english import RangePhrases

class BetweenRule(VerbalizationRule):
    """Verbalizes a Between operator as *"x is between lo and hi"*."""

    @classmethod
    def applies(cls, expression, context) -> bool:
        return isinstance(expression, Between)

    @classmethod
    def transform(cls, expression, context, verbalizer):
        left_fragment = verbalizer.build(expression.left, context)
        lower_fragment = verbalizer.build(expression.lo, context)
        upper_fragment = verbalizer.build(expression.hi, context)
        return phrase(
            left_fragment,
            RangePhrases.IS_BETWEEN.as_fragment(),
            phrase(lower_fragment, word("and"), upper_fragment),
        )
```

#### Step 3 — Registration Is Automatic

**That's it.**  Because concrete ``VerbalizationRule`` subclasses self-register
via ``__init_subclass__``, importing the module that defines ``BetweenRule`` is
enough.  If you created a new file, add an import line to
{py:data}`~krrood.entity_query_language.verbalization.rules.registry.ALL_RULES`
(the module import triggers registration).  If you added the rule to an existing
file, nothing else is needed — the existing import already covers it.

No hand-maintained list.  No registry update.  Just define the class and import
its module.

#### Step 4 — Use ``verbalizer.build(child, context)`` for Recursive Sub-Expressions

``EQLVerbalizer`` is passed as the ``verbalizer`` parameter to every
``transform()``.  Always call ``verbalizer.build(child, context)`` to render sub-expressions
— this re-enters the full rule dispatch, so your rule's sub-expressions benefit from
coreference tracking, binding overrides, pronoun resolution, and any future rules.

Never call ``verbalize_expression(child)`` from within a rule — that creates a fresh
context and breaks coreference chains across the expression tree.

#### Step 5 — Use Fragment Constructors and Vocabulary Constants

Prefer the factory functions from
{py:mod}`~krrood.entity_query_language.verbalization.fragments.factory` for dynamic
fragments, and vocabulary enum members (``.as_fragment()``) for fixed words:

```python
from krrood.entity_query_language.verbalization.fragments.factory import phrase, role, word
from krrood.entity_query_language.verbalization.vocabulary.english import Keywords, Copulas

# Fixed words → vocabulary constants (avoids typos, carries SemanticRole)
header = Keywords.FIND.as_fragment()
copula = Copulas.IS.as_fragment()

# Dynamic text → factory functions
label = role("Robot", SemanticRole.VARIABLE)
join = word(",")
inline = phrase(label, copula, value)
```

Key vocabulary enums and their ``.as_fragment()`` return type:

| Enum | Example | Returns |
|---|---|---|
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Keywords` | ``Keywords.FIND.as_fragment()`` | ``WordFragment`` (KEYWORD role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Copulas` | ``Copulas.IS.as_fragment()`` | ``RoleFragment`` (OPERATOR role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Operators` | ``Operators.from_callable(op.le).select(...).as_fragment()`` | ``RoleFragment`` (OPERATOR role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Logicals` | ``Logicals.FOR_ALL.as_fragment()`` | ``WordFragment`` (LOGICAL role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Aggregations` | ``Aggregations.COUNT.as_fragment()`` | ``RoleFragment`` (AGGREGATION role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Articles` | ``Articles.THE.as_fragment()`` | ``WordFragment`` (PLAIN role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Conjunctions` | ``Conjunctions.AND.as_fragment()`` | ``WordFragment`` (PLAIN role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Prepositions` | ``Prepositions.OF.as_fragment()`` | ``WordFragment`` (PLAIN role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Pronouns` | ``Pronouns.ITS.as_fragment()`` | ``WordFragment`` (PLAIN role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.ExistentialPhrase` | ``ExistentialPhrase.singular("Robot")`` | ``PhraseFragment`` |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.RangePhrases` | ``RangePhrases.IS_BETWEEN.as_fragment()`` | ``PhraseFragment`` (OPERATOR role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.SortDirections` | ``SortDirections.ASC.as_fragment()`` | ``WordFragment`` (PLAIN role) |
| {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.FallbackNouns` | ``FallbackNouns.ENTITY.as_fragment()`` | ``WordFragment`` (PLAIN role) |

For coloured / linked fragments that reference a Python class or attribute, use
the class methods on {py:class}`~krrood.entity_query_language.verbalization.fragments.base.RoleFragment`:

```python
RoleFragment.for_variable("Robot", var)              # VARIABLE role + SourceRef to type
RoleFragment.for_attribute("battery", Robot)         # ATTRIBUTE role + SourceRef to attr
RoleFragment.for_operator("is greater than")         # OPERATOR role, no link
```

#### Step 6 — Understanding MRO Priority

The rule engine sorts rule classes by MRO depth — the number of inheritance steps
from ``VerbalizationRule``.  Greater depth = tried first.

```{mermaid}
classDiagram
    class VerbalizationRule {
        <<abstract>>
        depth 0 — never tried
    }
    class NotRule {
        depth 1 — generic negation fallback
    }
    class NotComparatorRule {
        depth 2 — tried BEFORE NotRule
    }
    class NotBoolAttrRule {
        depth 2 — tried BEFORE NotRule
    }
    class LogicalRule {
        <<abstract>>
        depth 1 — abstract
    }
    class AndRule {
        depth 2
    }
    class RangeConjunctionRule {
        depth 3 — tried first
    }
    VerbalizationRule <|-- NotRule
    VerbalizationRule <|-- LogicalRule
    NotRule <|-- NotComparatorRule
    NotRule <|-- NotBoolAttrRule
    LogicalRule <|-- AndRule
    AndRule <|-- RangeConjunctionRule
```

**When to subclass an existing rule vs ``VerbalizationRule``:**

| Situation | Base class | Why |
|---|---|---|
| New expression with no parent rule | ``VerbalizationRule`` | Fresh start |
| More-specific case of an existing rule | The existing rule class (e.g. ``AndRule``) | Automatic MRO priority — no need to worry about ordering |
| Sibling special cases (same depth) | The common parent (e.g. ``NotRule`` for two ``Not`` variants) | Both get depth+1, ordered by definition within the module |

Sibling rules at the same MRO depth **must** have mutually exclusive
``applies()`` preconditions — the engine's stable sort preserves definition order
as a tiebreaker, but the correct answer is to make sure only one rule ever matches
a given expression.

#### Step 7 — Write a Test

Add an entry to the parametrized test data in
``test/krrood_test/test_eql/test_verbalization/test_verbalization.py``.
The parametrize decorator feeds each ``(query, expected_text)`` pair to
``verbalize_expression`` and asserts the output matches:

```python
# In the parametrize list:
(Between(booking_date, date1, date2), "booking_date is between date1 and date2"),
```

Run the suite (ensure that you are in your local development virtual environment):

```bash
pytest test/krrood_test/test_eql/test_verbalization -x
```

All tests must stay green.

---

## VerbalizationContext Internals

A single {py:class}`~krrood.entity_query_language.verbalization.context.VerbalizationContext`
instance is threaded through the entire `EQLVerbalizer.build()` call tree.

### Coreference Tracking (`seen`)

```python
context.seen: dict   # maps expression._id_ → display label
```

The first time a `Variable` is encountered, `noun_for_parts()` records it in `seen` and returns `INDEFINITE` (→ "a Robot").  Subsequent encounters return `DEFINITE` (→ "the Robot").

### Disambiguation Map

Created by `VerbalizationContext.from_expression(expression)`, which pre-scans the full expression tree.  Types with a single variable keep the plain type name; collisions get numbered labels:

```
Robot    (single)  →  "Robot"
Apple 1  (first)   →  "Apple 1"
Apple 2  (second)  →  "Apple 2"
```

### Constraint Frames

Used by the `InstantiatedVariable` verbalization path:

```python
context.push_constraint_frame()   # open a frame
context.defer_constraint(expression)    # add expression to the top frame
deferred = context.pop_constraint_frame()  # retrieve and close
```

When an `Entity` is used as a chain root inside an `InstantiatedVariable`, its WHERE condition is deferred into the top frame rather than verbalized inline.  After all binding overrides are registered, the deferred expressions are verbalized and emitted as a *"such that …"* clause.

### Binding Overrides

```python
context.binding_overrides: dict   # maps expression._id_ → VerbFragment
```

Populated by `_verbalize_instantiated_natural` for each field binding.  Before any rule is consulted, `RuleEngine.build` checks this dict.  This ensures that when a variable appears a second time as a WHERE condition value, the renderer uses the same *"the field of the Type"* fragment rather than re-verbalizing the raw variable.

---

## Source References and Link Resolvers

{py:class}`~krrood.entity_query_language.verbalization.fragments.source_ref.SourceRef`
is a frozen dataclass that identifies the Python entity a `RoleFragment` represents:

```python
SourceRef(cls=Robot)                          # class reference
SourceRef(cls=Robot, attribute="battery")     # attribute reference
```

A {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver`
maps these to URL strings:

```python
class SourceLinkResolver(Protocol):
    def resolve(self, ref: SourceRef) -> Optional[str]: ...
```

The built-in implementation is {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.AutoAPIResolver`,
which builds Sphinx AutoAPI URLs:

```python
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import AutoAPIResolver
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline

# Auto-detect local docs
resolver = AutoAPIResolver.for_package("krrood")
pipeline = VerbalizationPipeline.html(link_resolver=resolver)
```

The resolver is passed to the renderer at construction time (via `VerbalizationPipeline` factory methods).  When `_render_role` encounters a `RoleFragment` with a non-`None` `source_ref` and a non-`None` `_link_resolver`, it calls `resolver.resolve(ref)` and wraps the coloured text with a hyperlink.

---

## Rule Organisation

Rules live in ``krrood/entity_query_language/verbalization/rules/``.  Each file owns one
expression family.  When adding a new rule, check the table for the matching family.

| Module | Expression types | Output form | Add a rule here when … |
|---|---|---|---|
| ``rules/query.py`` | ``Entity``, ``SetOf``, ``GroupedBy``, ``OrderedBy``, ``Filter``, ``ResultQuantifier`` | *"Find X such that …"*, noun phrases, clause assembly | Adding a new query variant, select clause, or noun form |
| ``rules/inference_rule.py`` | ``Entity`` (selected var is ``InstantiatedVariable``) | *"IF … THEN …"* blocks | Changing IF/THEN phrasing or antecedent rendering |
| ``rules/chains.py`` | ``MappedVariable`` (Attribute, Index, Call, FlatVariable) | Possessive paths, bool predicates, pronominal chains | Adding a new chain node type or path rendering variant |
| ``rules/logical.py`` | ``AND``, ``OR``, ``Not``, range-folded conjunctions | Oxford-comma lists, negation, range folding | Adding a new logical operator or a negated variant of an expression |
| ``rules/comparator.py`` | ``Comparator`` | *"<left> <operator> <right>"* (via ``operator_phrase.py``) | Adding a new comparator variant (not a new operator — operator phrases live in vocabulary) |
| ``rules/aggregators.py`` | ``Aggregator`` subtypes | *"the sum of …"*, *"the number of …"* | Adding a new aggregation function |
| ``rules/quantifiers.py`` | ``ForAll``, ``Exists`` | *"for all …"*, *"there exists …"* | Adding a new logical quantifier |
| ``rules/variables.py`` | ``Variable``, ``Literal``, ``InstantiatedVariable`` | Type names, literals, binding forms | Adding a new variable subtype or binding rendering variant |

**If your expression doesn't fit any existing family**, create a new file under
``rules/`` and add an import line to
{py:data}`~krrood.entity_query_language.verbalization.rules.registry.ALL_RULES`
(the module import triggers auto-registration — no list to maintain).

Rule bodies live directly in ``transform()`` methods or in module-level helpers within the
same file.  For example, the query-body assembly (``_verbalize_query_body_``,
``_grouped_by_clause``, etc.) and the noun forms (``as_noun``, ``as_inline_noun``) are all
module-level functions in ``rules/query.py``, called directly by ``EntityRule.transform``
and ``SetOfRule.transform``.

Chain rules (``rules/chains.py``) and inference rules (``rules/inference_rule.py``) import
``as_inline_noun`` from ``rules/query.py`` for the cases where an ``Entity`` appears as a
chain root or antecedent — a simple cross-module function call, not a delegate method.

---

## How to Add a New Output Format

Subclass {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter` and optionally
{py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.FragmentRenderer`:

```python
from dataclasses import dataclass
from krrood.entity_query_language.verbalization.rendering.formatter import Formatter
from krrood.entity_query_language.verbalization.fragments.roles import SemanticRole, ROLE_COLORS

@dataclass
class MarkdownFormatter(Formatter):
    """Renders colour as Markdown bold (no true colour support in plain Markdown)."""

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

Then pass it to `verbalize_expression` or directly to any `FragmentRenderer`:

```python
from krrood.entity_query_language.verbalization.rendering.renderer import ParagraphRenderer

text = verbalize_expression(query, renderer=ParagraphRenderer(MarkdownFormatter()))
```

---

## API Reference

### Core

- {py:class}`~krrood.entity_query_language.verbalization.verbalizer.EQLVerbalizer`
- {py:func}`~krrood.entity_query_language.verbalization.verbalizer.verbalize_expression` — unified entry point; accepts optional ``renderer`` kwarg for colour/layout
- {py:class}`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline`
- {py:class}`~krrood.entity_query_language.verbalization.context.VerbalizationContext`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.features.Definiteness`
- {py:class}`~krrood.entity_query_language.verbalization.rule_engine.VerbalizationRule`
- {py:class}`~krrood.entity_query_language.verbalization.rule_engine.RuleEngine`

### Fragment Hierarchy

- {py:class}`~krrood.entity_query_language.verbalization.fragments.base.VerbFragment`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.base.WordFragment`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.base.RoleFragment`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.base.PhraseFragment`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.base.BlockFragment`
- {py:func}`~krrood.entity_query_language.verbalization.fragments.base.join_with`
- {py:func}`~krrood.entity_query_language.verbalization.fragments.base.oxford_and`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.roles.SemanticRole`
- {py:data}`~krrood.entity_query_language.verbalization.fragments.roles.ROLE_COLORS`
- {py:func}`~krrood.entity_query_language.verbalization.fragments.roles.role_for`
- {py:class}`~krrood.entity_query_language.verbalization.fragments.source_ref.SourceRef`
- {py:func}`~krrood.entity_query_language.verbalization.fragments.factory.word`
- {py:func}`~krrood.entity_query_language.verbalization.fragments.factory.phrase`
- {py:func}`~krrood.entity_query_language.verbalization.fragments.factory.role`

### Rendering

- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.FragmentRenderer`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.Formatter`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.ANSIFormatter`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.HTMLFormatter`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.SourceLinkResolver`
- {py:class}`~krrood.entity_query_language.verbalization.rendering.source_link_resolver.AutoAPIResolver`

### Rule Modules and Analysis

- {py:mod}`~krrood.entity_query_language.verbalization.rules.query`
- {py:mod}`~krrood.entity_query_language.verbalization.rules.inference_rule`
- {py:mod}`~krrood.entity_query_language.verbalization.rules.chains`
- {py:mod}`~krrood.entity_query_language.verbalization.rules.logical`
- {py:mod}`~krrood.entity_query_language.verbalization.rules.comparator`
- {py:mod}`~krrood.entity_query_language.verbalization.rules.aggregators`
- {py:mod}`~krrood.entity_query_language.verbalization.rules.quantifiers`
- {py:mod}`~krrood.entity_query_language.verbalization.rules.variables`
- {py:class}`~krrood.entity_query_language.verbalization.rule_analysis.RuleAnalyzer`
- {py:class}`~krrood.entity_query_language.verbalization.rule_analysis.RuleStructure`
- {py:class}`~krrood.entity_query_language.verbalization.rule_analysis.AntecedentInfo`
- {py:class}`~krrood.entity_query_language.verbalization.rule_analysis.ConsequentBinding`
- {py:class}`~krrood.entity_query_language.verbalization.rule_analysis.AggregationStatus`
- {py:func}`~krrood.entity_query_language.verbalization.operator_phrase.comparator_phrase`
- {py:func}`~krrood.entity_query_language.verbalization.operator_phrase.comparator_operator`

### Utilities

- {py:func}`~krrood.entity_query_language.verbalization.chain_utils.walk_chain`
- {py:func}`~krrood.entity_query_language.verbalization.chain_utils.chain_root`
- {py:func}`~krrood.entity_query_language.verbalization.chain_utils.build_path_parts`
- {py:func}`~krrood.entity_query_language.verbalization.chain_utils.verbalize_plural`
- {py:func}`~krrood.entity_query_language.verbalization.utils._str`
- {py:func}`~krrood.entity_query_language.verbalization.utils._camel_to_words`
- {py:func}`~krrood.entity_query_language.verbalization.utils._ordinal`
- {py:func}`~krrood.entity_query_language.verbalization.utils._ensure_plural`

### Vocabulary

- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Keywords`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Logicals`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Aggregations`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Copulas`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Operators`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.Articles`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.english.ExistentialPhrase`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.words.PlainWord`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.words.RoleWord`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.words.OperatorPhrase`
- {py:class}`~krrood.entity_query_language.verbalization.vocabulary.words.VocabEnum`

### Rule Registry

- {py:data}`~krrood.entity_query_language.verbalization.rules.registry.ALL_RULES`
