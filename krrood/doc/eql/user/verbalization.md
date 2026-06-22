---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.19.3
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Verbalization

EQL queries are Python objects — they can be inspected, composed, stored, and now: **read aloud**.

Verbalization turns any EQL expression into a plain-English sentence. This is useful for:

- **Debugging** — instantly understand what a complex query actually asks.
- **Explainability** — surface query intent in logs, UIs, or reports.
- **Testing** — assert on what a query *means* (its intent and structure), not just what it returns. Two queries can return the same rows on a fixture yet express different intent (e.g. `and_` vs `or_`); a verbalization assertion pins the intent regardless of the data.

## The Quick API

The simplest way to verbalize any EQL expression is `verbalize_expression`. With no extra
arguments it returns plain text. Pass a *renderer* to control color and layout.

```{code-cell} ipython3
from krrood.entity_query_language.factories import variable, entity, an
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.example_domain import Robot

robots = [Robot("R2D2", 95), Robot("C3PO", 20), Robot("BB8", 80)]
r = variable(Robot, domain=robots)

query = an(entity(r).where(r.battery > 50))
print(verbalize_expression(query))
```

The output reads like a natural sentence describing exactly what the query selects.

## More Conditions, Still Readable

Adding more `.where()` conditions does not break the sentence — EQL connects them naturally.

```{code-cell} ipython3
query = an(entity(r).where(r.battery > 50, r.name != "BB8"))
print(verbalize_expression(query))
```

## Cross-Variable Conditions

Verbalization handles cross-variable comparisons too — the sentence describes the *relationship* between variables.

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.example_domain import Mission

missions = [Mission(robots[0], 1), Mission(robots[1], 3)]
m = variable(Mission, domain=missions)

query = an(entity(r).where(m.assigned_to == r, m.priority > 2))
print(verbalize_expression(query))
```

## Logical Operators

Conditions combined with `and_`, `or_`, and `not_` are verbalized with natural connectives.

```{code-cell} ipython3
from krrood.entity_query_language.factories import variable, and_, or_, not_

x = variable(int, [1, 5, 12])
print(verbalize_expression(and_(x > 1, x < 10, x != 5)))
print(verbalize_expression(or_(x > 10, x < 0)))
print(verbalize_expression(not_(x > 5)))
```

Notice the `or_` form opens with *"either … or …"* for readability, and chained `and_` conditions
separate each clause with a comma and the final *"and"*.

## Boolean and Indexed Attributes

An attribute whose type is `bool` uses a predicative form — *"<nav-path> is <attribute>"* —
rather than the possessive form used for non-boolean attributes.

```{code-cell} ipython3
from krrood.entity_query_language.factories import variable, not_
from krrood.entity_query_language.verbalization.example_domain import Task, Worker

w = variable(Worker, domain=None)
print(verbalize_expression(w.tasks[0].completed))
print(verbalize_expression(not_(w.tasks[0].completed)))
```

A numeric index like `[0]` becomes an ordinal (*"the first …"*), and the terminal boolean
field maps to *"is completed"* / *"is not completed"*.

Comparing a boolean attribute to a boolean **value** folds the value into the verb's polarity,
rather than tacking on *"is True"*:

```{code-cell} ipython3
t = variable(Task, domain=None)
print(verbalize_expression(t.completed == True))    # "a Task is completed"
print(verbalize_expression(t.completed == False))   # "a Task is not completed"
print(verbalize_expression(t.completed == variable(bool, [True, False])))  # left open
```

`== True` reads *"is completed"*, `== False` (and `!= True`) reads *"is not completed"*, and a
boolean variable whose domain holds both values reads *"is either completed or not"*.

## Relational Attributes

When an attribute is named as a *relation* — a past participle plus a preposition, like
`assigned_to`, `owned_by`, or `written_by` — navigating through it reads as a **relative clause**
naming the related type, rather than the bare genitive *"the assigned_to of …"*. The preposition
moves in front of *"which"* (*"to which"*, *"by which"*):

```{code-cell} ipython3
m = variable(Mission, domain=None)
print(verbalize_expression(m.assigned_to))               # the Robot to which a Mission is assigned
print(verbalize_expression(m.assigned_to.operational))   # … is operational
```

The head noun (*"the Robot"*) is the attribute's declared type, and the owner stays the subject of
the verb — so even agentive *by* relations read correctly (*"the Person by which a Book is owned"*,
never the reversed *"the Person owned by a Book"*). When the owner is the query's subject it
pronominalises (*"the battery of the Robot **to which it is assigned**"*). A plain noun attribute is
unaffected and keeps the genitive *"the name of the department of an Employee"*; a noun that merely
ends in a preposition (e.g. `color_in`) is not treated as a relation.

When the robot is the **subject** of a clause — a boolean attribute, *"the Robot to which it is
assigned is operational"* — the attributes that follow read *"its battery … its power"*, uniformly
(once *"its"* refers to the robot it keeps it as the topic):

```{code-cell} ipython3
query = an(entity(m).where(
    m.assigned_to.operational, m.assigned_to.battery > 5, m.assigned_to.power > 1,
))
print(verbalize_expression(query))
# Find a Mission such that the Robot to which it is assigned is operational, its battery
# is greater than 5, and its power is greater than 1
```

But when the clause is *about an attribute* — *"the battery of the Robot … is greater than 5"* — the
battery, not the robot, is its subject, so a following *"its power"* would read as the battery's
power. To stay unambiguous the owner is spelled out instead:

```{code-cell} ipython3
query = an(entity(m).where(m.assigned_to.battery > 5, m.assigned_to.power > 10))
print(verbalize_expression(query))
# Find a Mission such that the battery of the Robot to which it is assigned is greater
# than 5, and the power of the Robot is greater than 10
```

## Absence Conditions (`== None`)

A comparison to `None` is read as an *absence*, not as a value. The exact wording adapts to the
attribute:

- a plain **noun** attribute reads *"<owner> **has no** <attribute>"* (*"a Pose has no orientation"*);
- a **relational** attribute — one named as a past participle plus a preposition (`assigned_to`,
  `owned_by`, `shipped_to`) — reads as a passive verb naming the related type:
  *"<owner> **has not been** <verb> **any** <Type>"*. The related type is taken automatically from
  the attribute's declared type;
- a bare variable (no attribute to name) reads *"<subject> **does not exist"***.

```{code-cell} ipython3
m = variable(Mission, domain=None)
print(verbalize_expression(m.assigned_to == None))          # "a Mission has not been assigned to any Robot"
print(verbalize_expression(variable(Mission, domain=None) == None))  # "... does not exist"
```

Whether an attribute is "relational" is decided morphologically — the part before the preposition
must be a real past participle — so `assigned_to` (verb) becomes the passive form while a noun that
merely ends in a preposition (e.g. `color_in`) stays *"has no color_in"*. Inside a query the absence
is said as its own clause (*"such that the Mission has not been assigned to any Robot"*) — it never
folds into the *"whose …"* group, because the subject/object flip cannot sit there.

## Domain-Constrained Values

When a value-typed variable (an `int`/`float`/`str`/`bool` or an `enum`) carries a small explicit
domain, the verbalizer lists the candidates as *"one of …"* in value position.

```{code-cell} ipython3
r = variable(Robot, domain=robots)
print(verbalize_expression(r.battery == variable(int, [10, 50, 90])))
print(verbalize_expression(r.battery == variable(int, [10, 50])))   # a pair → "one of 10 or 50"
```

Three or more candidates use the serial comma (*"one of 10, 50, or 90"*); a pair drops it
(*"one of 10 or 50"*).  An *entity*-typed variable's domain is its inferred population and is never
listed — it stays *"a Robot"*.

## Concrete Object Values

Comparing against a concrete domain object means its *identity*, so the verbalizer says
*"a specific <Type>"* rather than printing the object's (possibly huge) `repr`.  When the class has
an obvious identifying field it is appended:

```{code-cell} ipython3
m = variable(Mission, domain=None)
print(verbalize_expression(m.assigned_to == robots[0]))   # robots[0] is Robot("R2D2", …)
```

The identifying field(s) are taken from the class's `_identifying_attributes_` classmethod if it
declares one, otherwise the first present of `name` / `id` / `label` / `key` / `uuid`
(`Robot` has `name`, so this reads *"a specific Robot with name 'R2D2'"*).  With none of those, it
falls back to a bare *"a specific Robot"*.

## Factoring Repeated Comparisons

Two comparisons that pair the same attribute across sibling chains fold into one natural clause —
*"the begin and end of the period have the same month and year"* — instead of repeating each.

```{code-cell} ipython3
from dataclasses import dataclass

@dataclass
class Date:
    month: int
    year: int

@dataclass
class Period:
    begin: Date
    end: Date

p = variable(Period, domain=None)
query = an(entity(p).where(p.begin.month == p.end.month, p.begin.year == p.end.year))
print(verbalize_expression(query))
```

## Underspecified Constructions (`match` / `underspecified`)

An `underspecified(...)` construction is a *generative* request — *"Generate a … given that …"*.
Several scalar assignments on one object coordinate into a single *"… respectively"* point (capped
at three; longer or phrase-valued assignments are said separately, and a `None` becomes a
*"has no"* point):

```{code-cell} ipython3
from krrood.entity_query_language.factories import underspecified

@dataclass
class Point:
    x: float
    y: float
    z: float

print(verbalize_expression(underspecified(Point)(x=1, y=2, z=3)))
```

## Aggregations

Aggregation functions (`count`, `sum`, `average`, `max`, `min`) are wrapped with
the definite article and a descriptive phrase when verbalized.  Here we need a domain with
a numeric field:

```{code-cell} ipython3
import datetime
from krrood.entity_query_language.factories import variable
import krrood.entity_query_language.factories as eql
from krrood.entity_query_language.verbalization.example_domain import (
    AmountDetails,
    BankTransaction,
)

t = variable(BankTransaction, domain=None)

print(verbalize_expression(eql.count(t)))
print(verbalize_expression(eql.sum(t.amount_details.amount)))
print(verbalize_expression(eql.average(t.amount_details.amount)))
print(verbalize_expression(eql.max(t.amount_details.amount)))
print(verbalize_expression(eql.min(t.amount_details.amount)))
```

All aggregations use the definite article (*"the number of"*, *"the sum of"*, ...).
The attribute chain following the aggregation uses a possessive *"of the ..."* path.

When a WHERE condition filters the *very attribute being aggregated*, that attribute is named in
full once (in the aggregate) and the condition refers back to it as a bare *"the battery"*:

```{code-cell} ipython3
query = an(entity(eql.average(m.assigned_to.battery)).where(m.assigned_to.battery > 5))
print(verbalize_expression(query))
# Find the average of the battery of the Robot to which a Mission is assigned such that
# the battery is greater than 5
```

## Reports (presenting results, not searching)

Some queries *present* results rather than *search* for a match — a calculation, or an ordered
listing. These open with **"Report"** rather than "Find". Report-ness and conditions are
orthogonal: a report may still be filtered.

A `set_of` that *computes* an aggregate is a calculation, so it opens with "Report" and drops the
code-like parentheses:

```{code-cell} ipython3
employee = variable(Employee, domain=None)
print(verbalize_expression(a(set_of(eql.sum(employee.salary)))))
# Report the sum of salaries of Employees
```

When it is **grouped**, the grouping is stated first as a **"For each <key>"** frame (the natural
reading of GROUP BY). The key is named once — as the bare group label — and is not restated as a
column:

```{code-cell} ipython3
query = a(
    set_of(employee.department, eql.sum(employee.salary)).grouped_by(employee.department)
)
print(verbalize_expression(query))
# For each department, report the sum of salaries of Employees
```

An **ordered** query is also a report — ordering presents *all* the (matching) results in
sequence, which is a listing, not a hunt for one match. The subject is therefore plural, and a
filter does not change that (it just narrows the list):

```{code-cell} ipython3
print(verbalize_expression(an(entity(employee).ordered_by(employee.salary))))
# Report Employees ordered by their salary (ascending)

print(verbalize_expression(
    an(entity(employee).where(employee.salary > 5).ordered_by(employee.salary))
))
# Report Employees whose salary is greater than 5, ordered by their salary (ascending)
```

A `limit` is the exception: it ranks (*"Find the top three …"*), a distinct count-bearing form.

A reported aggregate is a *computed quantity*: named in full where it is first reported, a later
mention of the same aggregate (in `having`, or an ordering) reduces to its bare head:

```{code-cell} ipython3
total = eql.sum(employee.salary)
print(verbalize_expression(
    a(set_of(employee.department, total).grouped_by(employee.department).having(total > 30000))
))
# For each department, report the sum of salaries of Employees having the sum greater than 30000
```

A plain (non-aggregating, unordered) `set_of` stays a search and also drops the parentheses.
Several attributes of the *same* owner fold into a shared genitive rather than repeating the owner:

```{code-cell} ipython3
print(verbalize_expression(a(set_of(employee.department, employee.name))))
# Find the department and name of an Employee
```

## Date Range Folding

When a lower-bound and an upper-bound comparison on the same datetime attribute appear
together, the verbalizer folds them into a single *"between ... and ..."* phrase.

```{code-cell} ipython3
bt = variable(BankTransaction, domain=None)
lo = datetime.datetime(2026, 5, 15)
hi = datetime.datetime(2026, 5, 30)

query = an(entity(bt).where(bt.booking_date >= lo, bt.booking_date <= hi))
print(verbalize_expression(query))
```

The output uses *"is between ... and ..."* with the datetime values formatted in a
human-readable form.  The same folding happens when the comparisons appear on an
aggregation sub-query's WHERE clause (see the next section).

## Nested Sub-Queries and Aggregation Scoping

Aggregation sub-queries nest naturally.  A scoped aggregation — *"the sum of amounts
among BankTransactions whose booking_date is between ..."* — is produced when an aggregate
appears inside an `entity()` wrapper with its own WHERE conditions.

```{code-cell} ipython3
bt  = variable(BankTransaction, domain=None)
bt_sum = variable(BankTransaction, domain=None)
start = datetime.datetime(2026, 5, 15)
end   = datetime.datetime(2026, 5, 30)

sum_val = an(entity(eql.sum(bt_sum.amount_details.amount)).where(
    bt_sum.booking_date >= start, bt_sum.booking_date <= end,
))
query = an(entity(bt).where(bt.amount_details.amount == sum_val))
print(verbalize_expression(query))
```

The possessive pronoun *"its"* replaces a repeated *"of the BankTransaction"* on the
outer condition.  The scoped aggregation automatically uses the preposition *"among"*
and the WHERE conditions inside the sub-query continue to work (date-range folding,
pronouns, etc.).

A maximum-value variant produces a similarly compact form:

```{code-cell} ipython3
bt_max = variable(BankTransaction, domain=None)
max_val = an(entity(eql.max(bt_max.amount_details.amount)))
query = an(entity(bt).where(bt.amount_details.amount == max_val))
print(verbalize_expression(query))
```

And a scoped aggregation can stand alone as the main query — no outer entity needed:

```{code-cell} ipython3
cutoff = datetime.datetime(2024, 5, 17)
scoped_sum = an(entity(eql.sum(bt.amount_details.amount)).where(bt.booking_date < cutoff))
print(verbalize_expression(scoped_sum))
```

## Same-Type Variable Disambiguation

When two variables of the same type appear in a query, the verbalizer distinguishes
them by appending a numeric index — *"Employee 1"*, *"Employee 2"*.

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.example_domain import Employee

emp1 = variable(Employee, domain=None)
emp2 = variable(Employee, domain=None)
query = an(entity(emp1).where(emp1.salary > emp2.salary))
print(verbalize_expression(query))
```

The same mechanism also handles disambiguation when an aggregate and an entity share
a type:

```{code-cell} ipython3
emp = variable(Employee, domain=None)
query_agg = an(entity(eql.average(emp.salary)).where(emp.starting_salary > 20000))
print(verbalize_expression(query_agg))
```

It also counts the *related entity* a relational attribute introduces, so two missions assigned to
two different robots read *"Robot 1"* / *"Robot 2"* rather than two indistinguishable *"the Robot"*s
— and a repeat of either reduces to its numbered label:

```{code-cell} ipython3
pair = variable(Pair, domain=None)
query = an(entity(pair).where(
    pair.primary.assigned_to.battery > 5,
    pair.primary.assigned_to.power > 1,
    pair.secondary.assigned_to.battery > 3,
))
print(verbalize_expression(query))
# Find a Pair such that the battery of Robot 1 to which its primary is assigned is
# greater than 5, the power of Robot 1 is greater than 1, and the battery of Robot 2
# to which its secondary is assigned is greater than 3
```

## Custom Predicates

A custom predicate can control its verbalization by implementing
`_verbalization_template_`.  The template is a string with ``{field_name}`` placeholders
corresponding to the predicate's dataclass fields.

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.example_domain import (
    Location,
    IsReachable,
)

loc = variable(Location, domain=None)
print(verbalize_expression(IsReachable(loc)))
```

Predicates with multiple fields receive their arguments in positional order:

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.example_domain import (
    Department,
    StaffMember,
    WorksIn,
)

dept = variable(Department, domain=None)
emp = variable(StaffMember, domain=None)
print(verbalize_expression(WorksIn(emp, dept)))
```

When a predicate does not define `_verbalization_template_`, the verbalizer falls
back to a generic description.

## Grouped Queries (`set_of` + `grouped_by` + `having`)

Queries built with `set_of`, `grouped_by`, and `having` are verbalized as a structured
sentence with the selection, GROUP BY, and HAVING clauses clearly separated.

```{code-cell} ipython3
from krrood.entity_query_language.factories import a, set_of

emp = variable(Employee, domain=None)
avg_salary = eql.average(emp.salary)
query = a(
    set_of(emp.department, avg_salary)
    .grouped_by(emp.department)
    .having(avg_salary > 30000)
)
print(verbalize_expression(query))
```

The HAVING clause uses the *compact* (copula-less) operator form — *"greater than 30000"*
instead of *"is greater than 30000"* — matching SQL-style conciseness.  The GROUP BY
clause states only the grouping key without restating the full selection tuple.

## Colored Terminal Output

For richer output in a terminal, use `VerbalizationPipeline` with a renderer (`verbalize_expression`
itself only returns plain text). Each part of the sentence is color-coded by its semantic role.

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.formatter import ANSIFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import ParagraphRenderer

print(VerbalizationPipeline(ParagraphRenderer(ANSIFormatter())).verbalize(query))
```

Color legend:

| Color | Role | Example |
|---|---|---|
| Cornflower blue | **Variable type** | `Robot`, `Mission` |
| Teal | **Attribute** | `battery`, `assigned_to` |
| Orange | **Operator** | `is greater than`, `is not` |
| Green | **Logical connective** | `and`, `or`, `such that` |
| Gray | **Literal value** | `50`, `"BB8"` |
| Yellow | **Keyword / rule structure** | `If`, `then`, `whose` |

## HTML Output for Notebooks

Build a `VerbalizationPipeline` with an `HTMLFormatter` renderer to produce `<span>` tags for direct use in Jupyter or any HTML context.

```{code-cell} ipython3
from IPython.display import HTML
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.formatter import HTMLFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import ParagraphRenderer

HTML(VerbalizationPipeline(ParagraphRenderer(HTMLFormatter())).verbalize(query))
```

### Hierarchical HTML

Use `HierarchicalRenderer` to get an indented bullet structure — great for rule trees.

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.rendering.renderer import HierarchicalRenderer

HTML(VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter())).verbalize(query))
```

## Verbalizing Rule Trees

Verbalization really shines on rule trees. The if/then structure is rendered clearly.

```{code-cell} ipython3
from krrood.entity_query_language.factories import (
    variable, entity, an, deduced_variable, inference
)
from krrood.entity_query_language.verbalization.example_domain import (
    Bird,
    LoveBirds,
    BirdView,
    StrongLoveBird,
)

birds = [Bird('tweety'), Bird('snappy'), Bird('sleepy')]
bird = birds[0]
love_birds = [
    LoveBirds(birds[0], birds[1], True),
    LoveBirds(birds[1], birds[2], False),
]

love_birds = variable(LoveBirds, domain=love_birds)
bird_view = deduced_variable(BirdView)
rule_query = an(entity(inference(StrongLoveBird)(bird=love_birds.bird_1)).where(love_birds.strong_love))

HTML(VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter())).verbalize(rule_query))
```

The hierarchical renderer shows the **If/then** structure with each condition and conclusion
on its own line, indented under the relevant clause.

### Deep Nesting in Hierarchical Mode

The hierarchical view really shines on rules with *deeply nested* attribute chains — the
bullet structure makes the relationship between conditions visually clear.  Here is a
drawer-detection rule with a multi-hop path:

```{code-cell} ipython3
from krrood.entity_query_language.factories import variable, entity, an, inference
from krrood.entity_query_language.verbalization.example_domain import (
    Handle,
    Container,
    FixedConnection,
    PrismaticConnection,
    Drawer,
)

fc = variable(FixedConnection, domain=None)
pc = variable(PrismaticConnection, domain=None)
h  = variable(Handle, domain=None)
drawer_rule = an(entity(inference(Drawer)(
    container=fc.parent,
    handle=fc.child,
)).where(
    fc.parent == pc.child,
    fc.child == h,
))

HTML(VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter())).verbalize(drawer_rule))
```

And a cabinet rule that aggregates over multiple drawers — notice the *aggregated* antecedent
uses *"there are"* and the THEN clause bindings use plural *"are"*:

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.example_domain import Cabinet

pc = variable(PrismaticConnection, domain=None)
dr  = variable(Drawer, domain=None)
cabinet_rule = an(entity(inference(Cabinet)(
    container=pc.parent,
    drawers=dr,
)).where(pc.child == dr.container))

HTML(VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter())).verbalize(cabinet_rule))
```

## Verbalization as an Explanation Tool

Because EQL tracks *how* inferences were made, you can verbalize the query that produced any
inferred result — not just hand-written queries.

```{code-cell} ipython3
from krrood.entity_query_language.explanation.explanation import explain_inference

inferred_views = list(rule_query.evaluate())
inferred_object = inferred_views[0]

explanation = explain_inference(inferred_object)
print(verbalize_expression(explanation.query_root))
```

This outputs the exact query that matched and produced `inferred_object`, described in English.
It is directly useful for displaying *why* a robot perceives something as a Drawer, a Door, etc.

## Hyperlinks to Source Code

Pass `link_resolver=AutoAPIResolver(...)` when constructing the renderer and class/attribute
names become clickable links — opening the corresponding Sphinx AutoAPI documentation page.

`AutoAPIResolver` works in two modes:

| Mode | How to construct | When it works |
|---|---|---|
| **Local** | `AutoAPIResolver.for_package("krrood")` | After `sphinx-build doc doc/_build/html` |
| **GitHub Pages** | `AutoAPIResolver(base_url="https://cram2.github.io/…/krrood")` | Always, no local build needed |

The demo below uses `Robot` and `Mission` from
`krrood.entity_query_language.verbalization.example_domain` — real classes that Sphinx AutoAPI
documents like any other module under `src`, so their hyperlinks resolve to the generated AutoAPI
pages both on GitHub Pages and in a locally built docs tree (no hand-maintained mock API needed).

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.example_domain import Robot, Mission
from krrood.entity_query_language.factories import variable, entity, an
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.formatter import HTMLFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import HierarchicalRenderer
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import AutoAPIResolver

vd_robots = [Robot("R2D2", 95), Robot("C3PO", 20)]
vd_missions = [Mission(vd_robots[0], 3)]
r = variable(Robot, domain=vd_robots)
m = variable(Mission, domain=vd_missions)
linked_query = an(entity(r).where(m.assigned_to == r, m.priority > 2))

# Local — requires docs to be built first: sphinx-build doc doc/_build/html
resolver = AutoAPIResolver.for_package("krrood")
HTML(VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter(), resolver)).verbalize(linked_query))
```

```{code-cell} ipython3
# GitHub Pages — always available, no local build needed.
resolver = AutoAPIResolver(base_url="https://cram2.github.io/cognitive_robot_abstract_machine/krrood")
HTML(VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter(), resolver)).verbalize(linked_query))
```

## API Reference

- {py:func}`~krrood.entity_query_language.verbalization.pipeline.verbalize_expression` — the simplest entry point; returns plain text (``verbalize_expression(expr)``)
- {py:class}`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline` — colour / layout / hyperlinks via a renderer
  - ``VerbalizationPipeline.plain().verbalize(expr)`` — plain prose (what ``verbalize_expression`` uses)
  - ``VerbalizationPipeline.ansi().verbalize(expr)`` — ANSI-coloured prose
  - ``VerbalizationPipeline.html(hierarchical=True).verbalize(expr)`` — HTML indented bullets
  - ``VerbalizationPipeline.html(hierarchical=True, link_resolver=resolver).verbalize(expr)`` — with source hyperlinks
  - ``VerbalizationPipeline(renderer).verbalize(expr)`` — from a renderer you build directly
- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer` — flat prose layout
- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer` — indented bullet list layout
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter` — no colour markup
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.ANSIFormatter` — terminal escape codes
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.HTMLFormatter` — ``<span>`` tags for notebooks
