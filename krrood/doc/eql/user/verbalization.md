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

```{admonition} A few grammar words, in plain English
:class: note

You don't need any grammar background to read this page. A handful of terms come up now and then — here
is what each means in plain words, with the technical name in brackets in case you're curious:

- **the "X of Y" form** — *"the battery of a Robot"* (the *possessive* / *genitive*).
- **a "that …" description** — extra words hung on a noun to describe it, as in *"the Robot **that is
  assigned to a Mission**"* (a *relative clause*).
- **the subject** — the thing a sentence is about, or the thing doing the action (*a Robot* in *"a Robot
  is operational"*).
- **active vs. passive** — the doer first (*"the Person **who owns** a Book"*) versus the thing it is
  done to first (*"a Book **is owned by** a Person"*).
- **"is" vs. "are"** — the verb switches to match one thing versus several (*"its salary **is** …"* vs.
  *"their salaries **are** …"*).
- **a "by" relation** — a field like `owned_by`, where the *other* thing does the action (its *agent*).
```

## The Quick API

The simplest way to verbalize any EQL expression is `verbalize_expression`. With no extra
arguments it returns plain text. Pass a *renderer* to control color and layout.

```{code-cell} ipython3
from krrood.entity_query_language.factories import variable, entity, an
from krrood.entity_query_language.verbalization.pipeline import verbalize_expression
from krrood.entity_query_language.verbalization.example_domain import Robot

robots = [Robot("R2D2", 95, True), Robot("C3PO", 20, False), Robot("BB8", 80, True)]
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

Notice the `or_` form joins the options with a plain *"… or …"* — an *"or"* that allows either one or
both (an *inclusive or*). It deliberately does **not** start with *"either"*, which would suggest only
one of them can hold. Chained `and_` conditions are separated by commas with a final *"and"*.

## Boolean and Indexed Attributes

A `bool` attribute is read as a short *"… is …"* statement — *"the first task of a Worker **is
completed**"* — instead of the *"X of Y"* form (the *possessive*, e.g. *"the battery of a Robot"*) used
for other attributes. Grammar calls the *"is <attribute>"* form *predicative*.

```{code-cell} ipython3
from krrood.entity_query_language.factories import variable, not_
from krrood.entity_query_language.verbalization.example_domain import Task, Worker

w = variable(Worker, domain=None)
print(verbalize_expression(w.tasks[0].completed))
print(verbalize_expression(not_(w.tasks[0].completed)))
```

A numeric index like `[0]` becomes a position word — *"first"*, *"second"* (an *ordinal*) — and merges
into the singular noun, so it reads *"the first task"* rather than *"the first of the tasks"*. The final
yes/no field then reads *"is completed"* / *"is not completed"*.

Comparing a yes/no attribute to a yes/no **value** builds the answer into the verb itself, rather than
tacking on *"is True"*:

```{code-cell} ipython3
t = variable(Task, domain=None)
print(verbalize_expression(t.completed == True))
print(verbalize_expression(t.completed == False))
print(verbalize_expression(t.completed == variable(bool, [True, False])))  # left open
```

`== True` reads *"is completed"*, `== False` (and `!= True`) reads *"is not completed"*, and a
boolean variable whose domain holds both values reads *"is either completed or not"*.

## Relational Attributes

Some attributes are named like a little action — a verb in its *"-ed"* form plus a small linking word
(a *past participle* plus a *preposition*), like `assigned_to`, `owned_by`, or `written_by`. Navigating
through one of these reads as a *"that …"* / *"which …"* description of the related thing (a *relative
clause*), instead of the plain *"the assigned_to of …"* (the *"X of Y"* / *possessive* form). The
linking word moves in front of *"which"* (*"to which"*, *"by which"*):

```{code-cell} ipython3
m = variable(Mission, domain=None)
print(verbalize_expression(m.assigned_to))
print(verbalize_expression(m.assigned_to.operational))
```

The main noun (*"the Robot"*) is the attribute's declared type. A *"by"* relation (`owned_by`,
`written_by`) — where the *other* thing does the action (its *agent*) — is read with the doer first (the
*active* voice): *"the Person **who owns** a Book"*, *"the Author **who writes** a Document"*, rather
than the thing-it-happens-to first (the *passive*) *"the Person by which a Book is owned"*. Every other
relation keeps the *"… which … is <verb>"* wording. When the owner is the thing the whole query is about,
it is referred back to with *"it"* / *"its"* the second time (rather than repeating its name):
*"the battery of the Robot **to which it is assigned**"*, *"the Person **who owns it**"*. A plain noun
attribute is untouched and keeps the *"X of Y"* form (*"the name of the department of an Employee"*); a
noun that merely ends in a preposition (e.g. `color_in`) is not treated as a relation.

When the robot is the thing the sentence is about (its *subject*) — as with a yes/no attribute, *"the
Robot to which it is assigned is operational"* — the attributes that follow read *"its battery … its
power"* consistently (once *"its"* points at the robot, it keeps pointing there):

```{code-cell} ipython3
query = an(entity(m).where(
    m.assigned_to.operational, m.assigned_to.battery > 5, m.assigned_to.power > 1,
))
print(verbalize_expression(query))
```

But when the sentence is *about an attribute* — *"the battery of the Robot … is greater than 5"* — the
battery, not the robot, is what it is about, so a following *"its power"* would sound like the battery's
power. To stay clear, the owner is spelled out instead:

```{code-cell} ipython3
query = an(entity(m).where(m.assigned_to.battery > 5, m.assigned_to.power > 10))
print(verbalize_expression(query))
```

### Nesting a Related Entity's Restriction

When a query links a related thing through one of these relations (``m.assigned_to == r``) and
separately puts a condition on that thing (``r.battery > 50``), the two are combined into a single
*"that …"* description on the main noun — the relation names the thing, and its condition hangs right off
it — rather than a separate trailing *"such that …"* sentence that repeats *"the Robot"*. A single
attribute compared with *"greater/less than"* (an *order* comparison) reads as a short *"with <attribute>
<comparison>"* (other shapes keep the *"whose … is …"* wording):

```{code-cell} ipython3
r = variable(Robot, domain=None)
query = an(entity(m).where(m.assigned_to == r, r.battery > 50))
print(verbalize_expression(query))
```

## Absence Conditions (`== None`)

A comparison to `None` is read as an *absence*, not as a value. The exact wording adapts to the
attribute:

- a plain **noun** attribute reads *"<owner> **has no** <attribute>"* (*"a Pose has no orientation"*);
- an **action-style** attribute (the *"-ed"*-verb-plus-linking-word kind above, like `assigned_to`,
  `owned_by`, `shipped_to`) reads as *"<owner> **has not been** <verb> **any** <Type>"* — the related
  type is filled in automatically from the attribute's declared type;
- a bare variable (nothing to name) reads *"<subject> **does not exist"***.

```{code-cell} ipython3
m = variable(Mission, domain=None)
print(verbalize_expression(m.assigned_to == None))
print(verbalize_expression(variable(Mission, domain=None) == None))
```

Whether an attribute counts as *action-style* is decided by the shape of the word — the part before the
linking word must be a real *"-ed"* verb form (a *past participle*) — so `assigned_to` (a verb) becomes
the *"has not been assigned to"* form, while a noun that merely ends in a small word (e.g. `color_in`)
stays *"has no color_in"*. Inside a query this absence is said as its own sentence (*"such that the
Mission has not been assigned to any Robot"*) — it never joins the *"whose …"* group, because that would
force the sentence to swap who-does-what, which doesn't fit there.

## Domain-Constrained Values

When a single-value variable (an `int`/`float`/`str`/`bool` or an `enum`) is given a small fixed set of
allowed values, the verbalizer lists them as *"one of …"* where the value would go.

```{code-cell} ipython3
r = variable(Robot, domain=robots)
print(verbalize_expression(r.battery == variable(int, [10, 50, 90])))
print(verbalize_expression(r.battery == variable(int, [10, 50])))
```

Three or more values use a comma before *"or"* (the *serial comma*: *"one of 10, 50, or 90"*); a pair
drops it (*"one of 10 or 50"*).  A variable that stands for a whole object (not a single value) is never
listed out — it stays *"a Robot"*.

## Concrete Object Values

Comparing against a concrete domain object means its *identity*, so the verbalizer says
*"a specific <Type>"* rather than printing the object's (possibly huge) `repr`.  When the class has
an obvious identifying field it is appended:

```{code-cell} ipython3
m = variable(Mission, domain=None)
print(verbalize_expression(m.assigned_to == robots[0]))   # robots[0] is Robot("R2D2", …)
```

The identifying field(s) are the dataclass fields marked with `GrammarMetadata.is_identifying_field`
(via `field(metadata=FieldMetadata(other_metadata=[GrammarMetadata(is_identifying_field=True)]).as_dict())`),
otherwise the first present of `name` / `id` / `label` / `key` / `uuid` (`Robot` has `name`, so this
reads *"a specific Robot with name 'R2D2'"*). With none of those, it falls back to a bare *"a
specific Robot"*.

## Factoring Repeated Comparisons

Two comparisons that check the same fields on two parallel parts of an object — here a period's begin
and end — fold into one natural sentence, *"the beginning and end of the period have the same month and
year"*, instead of repeating each.

A field whose attribute name reads awkwardly can register a `display_name` in its
`GrammarMetadata`; the verbalizer then uses that word wherever the field appears (here `begin`
surfaces as *"beginning"*).

```{code-cell} ipython3
from dataclasses import dataclass, field
from krrood.patterns.field_metadata import FieldMetadata, GrammarMetadata

@dataclass
class Date:
    month: int
    year: int

@dataclass
class Period:
    begin: Date = field(
        metadata=FieldMetadata(
            other_metadata=[GrammarMetadata(display_name="beginning")]
        ).as_dict()
    )
    end: Date = None

p = variable(Period, domain=None)
query = an(entity(p).where(p.begin.month == p.end.month, p.begin.year == p.end.year))
print(verbalize_expression(query))
```

## Underspecified Constructions (`match` / `underspecified`)

An `underspecified(...)` construction asks the system to *make* something rather than find it —
*"Generate a … given that …"*. Several single-value settings on one object are combined into one
*"… respectively"* line (up to three; longer or phrase-length values are said separately, and a `None`
becomes a *"has no"* line):

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

All aggregations start with *"the"* (the *definite article*): *"the number of"*, *"the sum of"*, ….
The attribute after it uses the *"of the ..."* (*possessive*) form.

When a WHERE condition filters the *very attribute being aggregated*, that attribute is named in
full once (in the aggregate) and the condition refers back to it as a bare *"the battery"*:

```{code-cell} ipython3
query = an(entity(eql.average(m.assigned_to.battery)).where(m.assigned_to.battery > 5))
print(verbalize_expression(query))
```

## Reports (presenting results, not searching)

Some queries *present* results rather than *search* for a match — a calculation, or an ordered
listing. These open with **"Report"** rather than "Find". Report-ness and conditions are
orthogonal: a report may still be filtered.

A `set_of` that *computes* an aggregate is a calculation, so it opens with "Report" and drops the
code-like parentheses:

```{code-cell} ipython3
from krrood.entity_query_language.factories import a, set_of
from krrood.entity_query_language.verbalization.example_domain import Employee

employee = variable(Employee, domain=None)
print(verbalize_expression(a(set_of(eql.sum(employee.salary)))))
```

When it is **grouped**, the grouping is stated first as a **"For each <key>"** frame (the natural
reading of GROUP BY). The key is named once — as the bare group label — and is not restated as a
column:

```{code-cell} ipython3
query = a(
    set_of(employee.department, eql.sum(employee.salary)).grouped_by(employee.department)
)
print(verbalize_expression(query))
```

An **ordered** query is also a report — ordering shows *all* the matching results in sequence, a listing
rather than a hunt for one. So the subject is plural, and a filter doesn't change that (it just shortens
the list). When there is more than one, the wording matches it: the verb becomes *"are"* (*"whose
salaries are …"*) and a per-item value spreads to each (*"their salaries"*):

```{code-cell} ipython3
print(verbalize_expression(an(entity(employee).ordered_by(employee.salary))))

print(verbalize_expression(
    an(entity(employee).where(employee.salary > 5).ordered_by(employee.salary))
))
```

The same matching applies wherever the subject is plural — including a `limit` that ranks several
(*"the top three Employees … whose salaries are greater than 1000"*).

A `limit` is the exception: it ranks and counts (*"Find the top three …"*).

A grouped query's `having` filter is attached to the group name as a *"whose <total> is …"* condition
(the total read as a plain possession of the group), so it is clear which group the condition filters:

```{code-cell} ipython3
total = eql.sum(employee.salary)
print(verbalize_expression(
    a(set_of(employee.department, total).grouped_by(employee.department).having(total > 30000))
))
```

A plain (non-aggregating, unordered) `set_of` stays a search and also drops the parentheses.
Several attributes of the *same* owner fold into one shared *"X of Y"* phrase rather than repeating the
owner:

```{code-cell} ipython3
print(verbalize_expression(a(set_of(employee.department, employee.name))))
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

The word *"its"* stands in for a repeated *"of the BankTransaction"* on the outer
condition.  The scoped aggregation automatically uses *"among"* and the WHERE conditions
inside the sub-query continue to work (date-range folding, pronouns, etc.).

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
from dataclasses import dataclass

@dataclass
class Pair:
    primary: Mission
    secondary: Mission

pair = variable(Pair, domain=None)
query = an(entity(pair).where(
    pair.primary.assigned_to.battery > 5,
    pair.primary.assigned_to.power > 1,
    pair.secondary.assigned_to.battery > 3,
))
print(verbalize_expression(query))
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

The HAVING clause is attached to the group name as *"For each department whose <total> is
greater than 30000"* — putting the filter on the group it restricts, rather than a trailing
*"having …"* that would look like it describes the whole list of results.  The GROUP BY part names
only the grouping key, without repeating the full set of selected columns.

## Colored Terminal Output

For richer output in a terminal, use `VerbalizationPipeline` with a renderer (`verbalize_expression`
itself only returns plain text). Each part of the sentence is color-coded by what kind of thing it is
(its role).

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
on its own line, indented under the relevant part.

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

And a cabinet rule that aggregates over multiple drawers — notice the *"if"* part (the condition) uses
*"there are"* and the *"then"* part uses the plural *"are"* to match the several drawers:

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

For links rendered *inside* this documentation site, use `AutoAPIResolver.for_in_site_docs()`: it
emits links **relative** to the current page, so they resolve to the sibling `autoapi/` tree in any
build or host — a local `_build/html`, a Pages preview, or the published site — with no hard-coded
URL. (For a live preview from an IDE instead, `AutoAPIResolver.for_package("krrood")` targets the
IDE's `localhost` server against a local `sphinx-build doc doc/_build/html`.)

The demo below uses `Robot` and `Mission` from
`krrood.entity_query_language.verbalization.example_domain` — real classes that Sphinx AutoAPI
documents like any other module under `src`, so their hyperlinks resolve to the generated AutoAPI
pages in any built docs tree (no hand-maintained mock API needed).

```{code-cell} ipython3
from krrood.entity_query_language.verbalization.example_domain import Robot, Mission
from krrood.entity_query_language.factories import variable, entity, an
from krrood.entity_query_language.verbalization.pipeline import VerbalizationPipeline
from krrood.entity_query_language.verbalization.rendering.formatter import HTMLFormatter
from krrood.entity_query_language.verbalization.rendering.renderer import HierarchicalRenderer
from krrood.entity_query_language.verbalization.rendering.source_link_resolver import AutoAPIResolver

vd_robots = [Robot("R2D2", 95, True), Robot("C3PO", 20, False)]
vd_missions = [Mission(vd_robots[0], 3)]
r = variable(Robot, domain=vd_robots)
m = variable(Mission, domain=vd_missions)
linked_query = an(entity(r).where(m.assigned_to == r, m.priority > 2))

# `for_in_site_docs()` emits links relative to this page (which lives at eql/user/), so they
# resolve to the sibling AutoAPI tree in any build — a local _build/html, a Pages preview, or the
# published site — without hard-coding a host.
resolver = AutoAPIResolver.for_in_site_docs()
HTML(VerbalizationPipeline(HierarchicalRenderer(HTMLFormatter(), resolver)).verbalize(linked_query))
```

## API Reference

- {py:func}`~krrood.entity_query_language.verbalization.pipeline.verbalize_expression` — the simplest entry point; returns plain text (``verbalize_expression(expr)``)
- {py:class}`~krrood.entity_query_language.verbalization.pipeline.VerbalizationPipeline` — colour / layout / hyperlinks via a renderer
  - ``VerbalizationPipeline.plain().verbalize(expression)`` — plain prose (what ``verbalize_expression`` uses)
  - ``VerbalizationPipeline.ansi().verbalize(expression)`` — ANSI-coloured prose
  - ``VerbalizationPipeline.html(hierarchical=True).verbalize(expression)`` — HTML indented bullets
  - ``VerbalizationPipeline.html(hierarchical=True, link_resolver=resolver).verbalize(expression)`` — with source hyperlinks
  - ``VerbalizationPipeline(renderer).verbalize(expression)`` — from a renderer you build directly
- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.ParagraphRenderer` — flat prose layout
- {py:class}`~krrood.entity_query_language.verbalization.rendering.renderer.HierarchicalRenderer` — indented bullet list layout
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.PlainFormatter` — no colour markup
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.ANSIFormatter` — terminal escape codes
- {py:class}`~krrood.entity_query_language.verbalization.rendering.formatter.HTMLFormatter` — ``<span>`` tags for notebooks
