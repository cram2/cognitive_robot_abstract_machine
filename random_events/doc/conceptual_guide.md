---
jupytext:
  cell_metadata_filter: -all
  formats: md:myst
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.11.5
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# Conceptual Guide

In probability theory, sigma algebras play a fundamental role by formalizing the notion of events within a sample space.
This chapter delves into the development and application of a specific sigma-algebra, the *product sigma algebra* 
tailored for the specific needs of tractable probabilistic inference.

This book walks through the motivation, definition and construction of an algebra that is suitable for tractable 
probabilistic reasoning.

## Motivation

While foundational concepts like sigma-algebras may appear abstract at first, a thorough understanding of their 
properties and, specifically, product sigma-algebras, is crucial for rigorous probability theory and its applications 
across various scientific disciplines, like robotics. Key motivating arguments are

- *Foundations of probability theory*: sigma algebras are the building blocks for defining probability in a rigorous 
way. By understanding them, a deeper understanding of how probabilities are assigned to events is gained.
- *Working with complex events:* In real-world scenarios, events can be intricate. Sigma algebras describe not just 
simple events but also unions, intersections, and complements of these events, and hence are a powerful tool to analyze 
probabilities of more complex situations.
- *Connection to advanced math:* Sigma algebras bridge the gap between set theory and advanced mathematical concepts 
like measure theory and integration. Studying them opens doors to these powerful tools used in various scientific 
fields.

Especially in robotics they are important, since
- *Reasoning with uncertainty:* Robots often operate in environments with uncertainty. 
 Sigma algebras provide a mathematical foundation to represent uncertain events and reason about the probability of 
different events happening (like sensor readings or obstacles appearing).
- *Decision-making under probability:* Many robotic tasks involve making decisions based on probabilities. By 
understanding sigma algebras, algorithms can be build that consider the chance of different outcomes and choose the 
action with the highest probability of success.
- *Planning and control under uncertainty:* Planning robot actions often requires considering various possibilities. 
Sigma algebras allow for the creation of probabilistic models of the environment, enabling robots to plan and control 
their movements while accounting for uncertainties.

Research has shown that events that are described by independent constraints (rules) are most likely the only events 
where probability estimation is tractable. {cite}`choi2020probabilistic`
Spaces that are constructed by independent constraints are called product spaces. Understanding the shape of such 
events is a key competence to building (new) tractable probabilistic models.

## Sigma Algebra

A sigma algebra ($sigma$-algebra) is a set of sets that contains all set differences that can be constructed by 
combining arbitrary subsets of the said set. Furthermore, it contains all countable unions of sets and all infinite 
intersections of the set.

````{prf:definition} Sigma Algebra
:label: def-sigma-algebra

Let {math}`E` be a space of elementary events. Consider the powerset {math}`2^E` and let {math}`\Im \subset 2^E` be a 
set of subsets of {math}`E`. Elements of {math}`\Im` are called random events. 
If {math}`\Im` satisfies the following properties, it is called a sigma-algebra ({math}`\sigma`-algebra).

```{math}
:label: eq-sigma-algebra
1. & \hspace{.5em}  E \in \Im \\
2. & \hspace{.5em}  (A, B) \in \Im \Rightarrow (A - B) \in \Im \\
3. & \hspace{.5em}  (A_1, A_2, ... \in \Im) \Rightarrow \left( \bigcup_{i=1}^\mathbb{N} A_i \in \Im \wedge 
\bigcap_{i=1}^\infty A_i \in \Im \right)
```

The tuple {math}`(E, \Im)` is called a **measurable space**.
````

An example of such a set of sets is the following:

```{code-cell} ipython3
:tags: []
from itertools import chain, combinations
    

def powerset(iterable):
    s = list(iterable)
    result = list(chain.from_iterable(combinations(s, r) for r in range(len(s) + 1)))
    return [set(x) for x in result]


E = {"a", "b", "c"}
powerset_of_E = powerset(E)
powerset_of_E
```

We can see that this is a correct sigma-algebra by verifying all axioms. First, check if it contains the space of 
elementary Events $E$:

```{code-cell} ipython3
:tags: []

E in powerset_of_E
```

Next, check if it contains all set differences:

```{code-cell} ipython3
:tags: []

for A, B in combinations(powerset_of_E, 2):
    if A - B not in powerset_of_E:
        print(f"Set difference {A - B} not in powerset")
```

Finally, check if it contains all countable unions and intersections:

```{code-cell} ipython3
:tags: []

for A, B in combinations(powerset_of_E, 2):
    if A.union(B) not in powerset_of_E:
        print(f"Union {A.union(B)} not in powerset")
    if A.intersection(B) not in powerset_of_E:
        print(f"Intersection {A.intersection(B)} not in powerset")
```

We have constructed a $\sigma$-algebra. This is a basic example, but it is important to understand the concept 
of such a system of sets.

The set operations union and difference can be constructed from complement and intersection.

## Intervals
A more advanced example is the set of intervals. Intervals are a common example of a sigma-algebra that is not trivial 
but yet easy to construct.

First, I introduce the concept of simple intervals and composite intervals in {prf:ref}`def-interval`.

````{prf:definition} Interval
:label: def-interval

A simple interval is a subset of  {math}`\mathbb{R}` denoted by

```{math}
:label: eq-interval
(a,b) &= \{x\in \mathbb{R} \mid a<x<b\},    \\
[a,b) &= \{x\in \mathbb{R} \mid a\le x<b\}, \\
(a,b] &= \{x\in \mathbb{R} \mid a<x\le b\}, \\
[a,b] &= \{x\in \mathbb{R} \mid a\le x\le b\}.
```

A composite interval, or just interval, is a union of simple intervals.

$$
I = I_1 \cup I_2 \cup ... \cup I_n
$$
````

According to {prf:ref}`def-sigma-algebra`, we now have to look at the intersection and complement of simple intervals.

The intersection of two simple intervals is a simple interval.

```{code-cell} ipython3
:tags: []
from random_events.interval import SimpleInterval, Interval, Bound

i1 = SimpleInterval(1, 3, Bound.CLOSED, Bound.CLOSED)
i2 = SimpleInterval(2, 4, Bound.OPEN, Bound.CLOSED)
i2.intersection_with(i1)
````

The complement of a simple interval is an interval.

```{code-cell} ipython3
:tags: []
Interval(*i2.complement())
````

Since every other set operation can be done using only the intersection, union and difference of two sets, we now 
investigate the last interesting property I would like to discuss. 

### Disjoint unions
[Sigma additivity](https://en.wikipedia.org/wiki/Sigma-additive_set_function) is an important property of measures.
Since probability is a measure, it would be nice to deal with disjoint unions of sets instead of regular unions.
An interval can be converted into a disjoint union of simple intervals using {prf:ref}`algo-make-disjoint` and 
{prf:ref}`algo-split-into-disjoint-and-non-disjoint`.

```{prf:algorithm} make_disjoint
:label: algo-make-disjoint

**Inputs** A union of sets  {math}`S = \bigcup_{i=1}^N s_i` that is not nescessarily dijsoint.

**Output**  A disjoint union of sets {math}`S^* = \bigcup_{i=1}^N s_i^*` such that  {math}`S = S^*`.

1. {math}`S^*, S^{'} \leftarrow` split_into_disjoint_and_non_disjoint({math}`S`)

2. while {math}`S^{'} \neq \emptyset`:

    3. {math}`S_{disjoint}`, {math}`S^{'} \leftarrow` split_into_disjoint_and_non_disjoint {math}`(S^{'})`
    
    4. {math}`S^* \leftarrow S^* \cup S_{disjoint}`

return {math}`S^*`
```

```{prf:algorithm} split_into_disjoint_and_non_disjoint
:label: algo-split-into-disjoint-and-non-disjoint

**Inputs** A union of sets  {math}`S = \bigcup_{i=1}^N s_i` that is not nescessarily dijsoint.

**Output**  A disjoint union of sets {math}`A = \bigcup_{i=1}^N a_i` and 
a non disjoint union of sets {math}`B = \bigcup_{i=1}^N b_i` such that  {math}`S = A \cup B`.

1. {math}`A \leftarrow \emptyset`

2.  {math}`B \leftarrow \emptyset`

3. for {math}`s_i \in S`

    3.1 {math}` A \leftarrow  A \cup  \left( s_i \setminus \{s_j | s_j \in S \text{ if } i \neq j \} \right)`
    
    3.2 {math}` B \leftarrow  B \cup \left( \bigcup_{j \neq i} s_i \cap s_j \right)`
    
return {math}`A, B`

```

As neither {prf:ref}`algo-make-disjoint` nor {prf:ref}`algo-split-into-disjoint-and-non-disjoint` use the fact that 
 {math}`S` is an interval, we can use them to convert any set of sets into a disjoint union of sets.


## Product Sigma Algebra

In machine learning, problems are typically constructed over a set of variables and not just intervals or simple sets.
Hence, a multidimensional algebra is needed.
A multidimensional algebra is constructed by taking the cartesian product of the univariate algebras. For instance, 
let {math}`A = \{a_1, a_2, a_3\}` and {math}`B = \{b_1, b_2, b_3\}` be two spaces of elementary events.
Constructing an algebra over both can be down by taking the cartesian product of the two sets 
{math}`E = A \times B = \{(a_1, b_1), (a_1, b_2), (a_1, b_3), (a_2, b_1), (a_2, b_2), (a_2, b_3), (a_3, b_1), 
(a_3, b_2), (a_3, b_3)\}`.

Formally, the product sigma algebra is defined in {prf:ref}`def-product-sigma-algebra`.

````{prf:definition} Product Sigma Algebra 
:label: def-product-sigma-algebra

Let {math}`(E_1,\Im_1)` and {math}`(E_2,\Im_2)` be measurable spaces.
The product sigma-algebra of {math}`\Im_1` and {math}`\Im_2` is denoted {math}`\Im_1 \otimes \Im_2`, and defined as:
{math}`\Im_1 \otimes \Im_2 := \sigma(\{S_1 \times S_2 : S_1 \in \Im_1 \wedge S_2 \in \Im_2\})`
where {math}`\sigma` denotes generated sigma-algebra and {math}`\times` denotes Cartesian product.
This is a sigma-algebra on the Cartesian product {math}`E_1 \times E_2`. {cite}`hunter2011data`
````

In machine learning, the sets {math}`E_1, E_2, ... E_n` are typically referred to as variables.

As you can probably imagine, it is very inefficient to directly work with cartesian products of sets directly due to 
their exponential size. 

The rest of this guide addresses an efficient representation of random events of the product sigma algebra.

Instead of storing all combinations that are constructed by the cartesian product, we can store the constraints that 
apply to every variable separately.
The datastructure that does so is called a *Simple Event*.
A union of simple events is called *Event*.
The intersection of two simple events is straight forward shown in {prf:ref}`lemma-intersection-simple-event`.


````{prf:lemma} Intersection of two Random Events in the Product Sigma Algebra 
:label: lemma-intersection-simple-event

The intersection of two simple events is given by the variable-wise intersections

```{math}
(A_1 \times B_1) \cap (A_2 \times B_2) = (A_1 \cap A_2) \times (B_1 \cap B_2). 
``` 
{cite}`hunter2011data`
````

````{prf:lemma} Complement of a Random Event in the Product Sigma Algebra 
:label: lemma-complement-product-sigma-algebra

The complement of a simple event is given by

```{math}
:label: eq-complement-product-sigma-algebra
(A \times B)^c = (A^c \times B) \cup (A \times B^c) \cup (A^c \times B^c). 
``` 
{cite}`hunter2011data`
````

While the complement of a simple event as stated in {prf:ref}`lemma-complement-product-sigma-algebra` is correct,
it is exponential heavy to calculate.
However, the proof below shows how to calculate the complement of a simple event that results in linear many terms.

````{prf:proof} Complement of a Simple Event in Linear Time.

Let
\begin{align*}
    \mathbb{A} &= A \cup A^c \, , \\
    \mathbb{B} &= B \cup B^c \text{ and }\\
    \mathbb{C} &= C \cup C^c.
\end{align*}

**Induction Assumption**

\begin{align*}
    (A \times B)^c = (A^c \times \mathbb{B}) \cup (A \times B^C)
\end{align*}
*Proof:*
\begin{align*}
    (A \times B)^c &= (A^c \times B) \cup (A \times B^c) \cup (A^c \times B^c) \\
    &= (A^c \times B) \cup (A^c \times B^c) \cup (A \times B^c) \\
    &= ( A^c \times (B \cup B^c) ) \cup   (A \times B^c) \\
    &= (A^c \times \mathbb{B}) \cup (A \times B^C) \hspace{0.5em}\square
\end{align*}

**Induction Step**

\begin{align*}
    (A \times B \times C)^c = (A^c \times \mathbb{B} \times \mathbb{C}) \cup (A \times B^C \times \mathbb{C} ) \cup 
    (A \times B \times C^c)
\end{align*}
*Proof:*
\begin{align*}
    (A \times B \times C)^c &= (A^c \times B \times C) \cup (A \times B^c \times C) \cup (A \times B \times C^c) \\ 
    &\cup (A^c \times B^c \times C) \cup (A^c \times B \times C^c) \cup (A \times B^c \times C^c) \\ 
    &\cup (A^c \times B^c \times C^c) \\
    &= (C \times \underbrace{(A^c \times B) \cup (A \times B^c) \cup (A^c \times B^c))}_{\text{Induction Assumption}} \\
    &\cup (C^c \times  \underbrace{(A^c \times B) \cup (A \times B^c) \cup (A^c \times B^c))}_{\text{Induction Assumption}} \\ 
    &\cup (A \times B \times C^c) \\
    &= (C \times (A^c \times \mathbb{B}) \cup (A \times B^C)) \cup 
       (C^c \times (A^c \times \mathbb{B}) \\ 
       &\cup (A \times B^C)) \cup (A \times B \times C^c)\\
    &= (A^c \times \mathbb{B} \times \mathbb{C}) \cup (A \times B^C \times \mathbb{C} ) \cup (A \times B \times C^c)
\end{align*}
````

A less formal and more intuitive explanation of why the complement is only linear big is obtained by thinking
about it geometrically. 
Consider the unit square.

```{code-cell} ipython3
:tags: []

from random_events.variable import *
from random_events.product_algebra import *
from random_events.interval import *
x = Continuous("x")
y = Continuous("y")

unit_square = SimpleEvent({x: closed(0, 1), y: closed(0, 1)}).as_composite_set()
fig = go.Figure(unit_square.plot(), unit_square.plotly_layout())
fig.update_layout(title="Unit Square")
fig.show()
```

If we now want to construct the complement of said square, we can do so by first constructing an event that
contains the parts left and right of the square.
It will be infinitely wide on the y-axis.
Be aware that since infinite wide areas cannot be plotted, the events are limited to a visible range.
However, this does not influence the line of thinking.

```{code-cell} ipython3
:tags: []


not_x_and_R = SimpleEvent({x: (~closed(0, 1)) & closed(-1, 2), y: closed(-1, 2)}).as_composite_set()
fig = go.Figure(not_x_and_R.plot(), not_x_and_R.plotly_layout())
fig.update_layout(title="Left and right part of the complement of the unit square.")
fig.show()
```

The last part missing is the top and bottom part of the complement of the unit square. For this, we can take the 
complement of y and keep x the way it was.

```{code-cell} ipython3
:tags: []


not_y_and_x = SimpleEvent({x: closed(0, 1), y: (~closed(0, 1)) & closed(-1, 2)}).as_composite_set()
fig.add_traces(not_y_and_x.plot())
fig.update_layout(title="All parts of the complement of the unit square.")
fig.show()
```

As we can see, this process constructs the correct complement.

## Connections to Logic

Algebraic concepts are hard to grasp.
Since you, the reader is very likely a computer scientist, I will re-explain events from the perspective of logic.
We can rewrite the assignment of a variable to a set as a boolean variable. For example,
{math}`Item_{\{\text{bowl}, \text{cup}\}} = item \in \{\text{bowl}, \text{cup}\}`
is a boolean variable that is true if the item is a bowl or a cup.
We can rewrite the statement of the union as a logical statement.

```{math}
\left( Item_{\{\text{bowl}\}} \land Color_{\{\text{blue}\}} \right) \lor \left( Item_{\{\text{cup}\}} 
\land Color_{\{\text{red}\}} \right)
```
This logical statement describes either a blue bowl or a red cup.
The event can always be thought of as a disjunction of conjunctions, hence a logical statement in the 
[disjunctive normal form](https://en.wikipedia.org/wiki/Disjunctive_normal_form).
This connection between the measurable space of a sigma algebra and logic is important for the combination of 
correct and consistent probabilistic reasoning.


## Application

You may ask yourself where the product algebra matters in real applications.
Consider your kitchen.
You most likely have some regions where you are able to stand, and some regions where you can't.
If you look at the floor plan of your kitchen, you could perhaps describe it as the following event.  

```{code-cell} ipython3
:tags: []

kitchen = SimpleEvent({x: closed(0, 6.6), y: closed(0, 7)}).as_composite_set()
refrigerator = SimpleEvent({x: closed(5, 6), y: closed(6.3, 7)}).as_composite_set()
top_kitchen_island = SimpleEvent({x: closed(0, 5), y: closed(6.5, 7)}).as_composite_set()
left_cabinets = SimpleEvent({x: closed(0, 0.5), y: closed(0, 6.5)}).as_composite_set()

center_island = SimpleEvent({x: closed(2, 4), y: closed(3, 5)}).as_composite_set()

occupied_spaces = refrigerator | top_kitchen_island | left_cabinets | center_island
fig = go.Figure(occupied_spaces.plot(), occupied_spaces.plotly_layout())
fig.show()
```
Now posing the question on where you can stand in your kitchen,
you can calculate the complement of the occupied space with the kitchen.

```{code-cell} ipython3
:tags: []

free_space = kitchen.difference_with(occupied_spaces)
fig = go.Figure(free_space.plot(), free_space.plotly_layout())
fig.show()
```

Now this already sounds somewhat useful.
However, just the events are of limited use. 
The real power of the product algebra comes when you start to calculate probabilities of events.
For this, 
you can check out this tutorial on [probability theory](https://probabilistic-model.readthedocs.io/en/latest/examples/probability_theory.html).

```{bibliography}
```