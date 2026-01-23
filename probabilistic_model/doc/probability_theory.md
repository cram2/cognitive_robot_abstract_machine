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

# Probability Theory

This book walks through essential concepts of probability theory with practical examples and
aims to make probability theory graspable for roboticists and machine learning engineers.

At the end of this book, you will understand ...

- Basic terms of probability theory
- Random events and how they interact
- Quantities of interest in probabilistic modeling
- Learning probabilistic models from data
- Applying probabilistic models to practical problems

## Concepts of Probability

The first section walks through the essential concepts of probability that lay the foundation to understand modern
probabilistic modelling.
You should get familiar with the sigma-algebra first. For that, I recommend [this book](https://random-events.readthedocs.io/en/latest/conceptual_guide.html).

The reason why the sigma-algebra is the set of interest for probability theory is, bluntly speaking, knowing the
probability of every atomic event is knowing the probability of every possible event.
Practically, the sigma-algebra used in real world modeling is always the powerset of some elementary events. While
it would be possible to construct a sigma-algebra that is not the powerset of its elementary events, it serves up to
everything I have seen so far, no purpose to the real world.

The sigma algebra of interest for practical probabilistic reasoning is
the [product sigma-algebra](https://random-events.readthedocs.io/en/latest/conceptual_guide.html#product-sigma-algebra).
Next, we define the probabilities of events in {prf:ref}`def-probability-measure`.

### Probability Measure

````{prf:definition} Probability Measure
:label: def-probability-measure

Let $(E, \Im)$ be a $\sigma$-algebra. A non-negative real function $P \rightarrow \mathbb{R}_{0, +}$
is called a measure if it satisfies the following properties:

1. $P(\emptyset) = 0$
2. For any countable sequence $\{A_i \in \Im \}_{i=1,...,}$ of pairwise disjoints sets $A_i \cap A_j = \emptyset$ if $i
   \neq j, P$ satisfies countable additivity ($\sigma$-additivity):
   
   $$P \left( \bigcup_{i=1}^\infty A_i \right) = \sum_{i=1}^\infty P(A_i)$$
   
3. $P(A \cup B) = P(A) + P(B) - P(A,B)$
4. $P(E) = 1$

The triple $(E, \Im, P)$ is called a probability space.
````

The probability measure just tells that for non-intersecting sets, you can determine the probability of the union by
adding the atomic probabilities. Furthermore, for intersecting sets you have to subtract the intersection because it is
added in there twice otherwise. You may realise that in the [random events package](https://random-events.readthedocs.io/en/latest/) 
the composite random event always contains a disjoint union of events. 
The sigma-additivity is the sole reason for that.

A common way to visually think about those things is through venn diagrams. 
The size of the objects depicted in the diagrams is proportionate to the probability of the respective event.


```{code-cell} ipython3
from random_events.set import SetElement, Set
from random_events.variable import Symbolic, Continuous, Integer
from random_events.product_algebra import Event, SimpleEvent
from random_events.interval import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools

x = Continuous("x")
y = Continuous("y")
```

```{code-cell} ipython3
event_a = SimpleEvent({x: closed(0, 2), y: closed(0, 2)}).as_composite_set()
event_b = SimpleEvent({x: closed(2.5, 3), y: closed(1, 2.5)}).as_composite_set()
fig = go.Figure(event_a.plot(color="red") + event_b.plot(color="blue"), event_a.plotly_layout())
fig.update_layout(title="Two disjoint events")
fig.show()
```

In this venn diagram, we can see that event $A$ and event $B$ are not intersecting. Hence, the probability (size) of
both events can be added to get the size of their union (sigma-additivity).

```{code-cell} ipython3
event_a.simple_sets[0][x] = closed(2, 2.75)
fig = go.Figure(event_a.plot(color="red") + event_b.plot(color="blue"), event_a.plotly_layout())
fig.update_layout(title="Two non-disjoint events")
fig.show()
```

In the second venn diagram, we see that the events are intersecting. Hence, the size of their union is smaller by the
sum of both sizes by exactly the intersecting part (3. Axiom).

In general, thinking of probabilities as size of sets is a good intuition. Most things you can think about for
geometrical sizes of shapes also apply to probability. However, probabilities never exceed unity (4. Axiom).
From the definition of the probability measure, we can derive {prf:ref}`theo-sum-rule`.


### Sum Rule

```{prf:theorem} Sum Rule
:label: theo-sum-rule

From $A + \neg A = E$ we get

$$P(A) + P(\neg A) = P(E) = 1 \text{, thus } P(A) = 1 - P(\neg A).$$


Furthermore, from $A = A \cap (B + \neg B)$, using the notation $P(A, B) = P(A \cap B)$ for the joint probability 
of $A$ and $B$, we get the Sum Rule

$$P(A) = P(A,B) + P(A, \neg B).$$

```
The sum rule is a way to express a belief over only the event $A$ if said probability is not known. Instead, the joint
probability of $A$ and $B$ is known, and by adding (marginalizing) over all cases of $B$ we can get the probability of
$A$.

```{code-cell} ipython3
fig = make_subplots(rows=1, cols=2, subplot_titles=["Event A and B", "Event A and not B"])

event_a_and_b = event_a & event_b
event_a_and_not_b = event_a & ~event_b
fig.add_traces(event_a_and_b.plot(), rows=1, cols=2)
fig.add_traces(event_a_and_not_b.plot(), rows=1, cols=1)
fig.update_xaxes(range=[2, 2.75], )
fig.update_yaxes(range=[0, 2], )
fig.update_layout(title="Visualization of the Sum Rule")
fig.show()

```

In the interactive plot (play around with it as you want) you can observe that the event $A$ is precisely described by
the union of both events $A, \neg B$ and $A, B$. Furthermore, both events are disjoint and the probability can be
calculated using sigma$additivity.

Before we dive into the next block of definitions, there is a little preface. As these next theorems are so strongly
intertwined with each other, there is not much sense of making a plot for each of them. 
There is an interactive plot after Bayes theorem that gives light to how these definitions work.
Onward, to {prf:ref}`def-truncated-probability`.
### Conditional Probability

````{prf:definition} Conditional Probability
:label: def-truncated-probability

If $P(A) > 0$, the quotient

$$P(B|A) = \frac{P(A, B)}{P(A)}$$

is called the truncated probability of $B$ given $A$. 
It immediately gives the product rule

$$P(A,B) = P(B|A)P(A) = P(A|B)P(B).$$

It is easy to show that $P(B|A) \geq 0, P(E|A) = 1$ and for $B\cap C = \emptyset$, we have 
$P(B+C |A) = P(B|A) + P(C|A)$. 

Thus, for a fixed $A$, $(E, \Im, P(\cdot|A))$ is a probability space as well.
````

Instead of explaining this concept in my own words, I will refer to Todd Kemps phrasing about it:
> We often think of truncated probability intuitively in terms of a two-stage experiment.
> In the first stage of the experiment, we see whether one particular event $A$ has occurred or not.
> If it has, it may influence whether a second event $B$, which we're going to measure in a second stage,
> will occur or not. Therefore, we may want to update our information on probabilities of the later
> events given information about whether the earlier event has occurred or not.
> In this language we'll refer to the original probability measure $P(B)$ as the prior probability of
> an event $B$, and after we observe that event $A$ has occurred, we refer to the new updated truncated
> probability as the posterior probability of that same event $B$.
> In more practical applications $A$ is referenced as the evidence since it is an event that evidently
> happened and $B$ is the query, since it is the event of interest.
> There are two very elementary but extraordinarily important results that come from that line of
> thinking the so-called law of total probability and most critically, Bayes theorem.

Todd Kemp, [30.1 Conditional Probability](https://youtu.be/CX013Hfgv-o?si=RQM5e7bcHJ8wuZeb&t=121])

### Law of total probability

```{prf:theorem} Law of total probability
:label: theo-total-probability

Let $A_1 + A_2 + \cdots + A_n = E$ and $A_i \cap A_j = \emptyset$ if $i \neq j$. 

Then for any $X \in \Im,$

$$P(X) = \sum_{i=1}^n P(X|A_i)P(A_i).$$

Because $X = E \cap X = \bigcup_{i=1}^n (A_i \cap X),$ we get that

$$P(X) = \sum_{i=1}^n P(A_i, X) = \sum_{i=1}^n P(X|A_i) P(A_i).$$
```


### Bayes Theorem

```{prf:theorem} Bayes Theorem
:label: theo-bayes-theorem

Let $A_1 + A_2 + \cdots + A_n = E$ and $A_i \cap A_j = \emptyset$ if $i \neq j$. 

Then for any $X \in \Im,$

$$P(A_i|X) = \frac{P(X|A_i)P(A_i)}{\sum_{j=1}^n P(X|A_j) P(A_j)}.$$
```


```{code-cell} ipython3
:tags: [hide-input]

def event_trace(event: SimpleEvent, name: str) -> go.Scatter:
    """
    Create the trace for a rectangle event.
    """
    x_0 = event["x"].simple_sets[0].lower
    x_1 = event["x"].simple_sets[0].upper
    y_0 = event["y"].simple_sets[0].lower
    y_1 = event["y"].simple_sets[0].upper
    trace = go.Scatter(x=[x_0, x_0, x_1, x_1, x_0], y=[y_0, y_1, y_1, y_0, y_0], fill="toself", name=name)
    return trace


x_event = SimpleEvent({x: closed(0, 1), y: closed(0, 1)})

# create figure and add event
fig = go.Figure()
fig.add_trace(event_trace(x_event, "X"))

# add disjoint event A_1 to A_n
for index, (x_interval, y_interval) in enumerate(itertools.product([closed(-0.25, 0.5).simple_sets[0],
                                                                    closed(0.5, 1.25).simple_sets[0]],
                                                                   [closed(-0.25, 0.5).simple_sets[0],
                                                                    closed(0.5, 1.25).simple_sets[0]])):
    sub_event = SimpleEvent({x: x_interval, y: y_interval})
    sub_event_trace = event_trace(sub_event, f"A_{index + 1}")
    fig.add_trace(sub_event_trace)
    intersection_trace = event_trace(sub_event.intersection_with(x_event), f"A_{index + 1} and X")
    fig.add_trace(intersection_trace)
fig.update_layout(title="Visualization of the law of total probability and Bayes theorem")
fig.show()
```

In the interactive plot, you can see the event $X$ and the disjoint events $A_1, A_2, A_3, A_4$. 
The intersection of $X$ and $A_i$ is also plotted. 
You can see that the sum of the intersections is equal to the event $X$. 
This is the law of total probability. 
Conditioning on an event can be seen as zooming into the events and only accounting for what is
inside the zoomed area. 
The Bayes theorem is a way to reverse the conditioning. 
It is a way to update the prior belief of an event given new evidence and just a consequence of the definitions given 
so far.

## Random Variables

While there exists a definition for random variables, I find it quite confusing and disconnected to how things are
approached in practice. Instead, I will give a more practical and engineering motivated definition on what a 
random variable is.

```{prf:definition} Random Variable
:label: def-random-variable

A random variable is a term in a language that can take different values. 
The range of values of a random variable, $\text{dom}(X)$, is exhaustive and mutually exclusive. 

A tuple $X$ of random variables with $X=\left<X_{1}, \ldots, X_{n}\right>$ is a complex random variable with the domain 

$$\text{dom}(X)=\text{dom}(X_{1}) \times \ldots \times \text{dom}(X_{n}).$$

A possible world $x=\left<X_1=v_1,\ldots,X_n=v_n\right>$ specifies a value assignment for each random variable $X_i$
under consideration.

The set $\mathcal{X}$ of all possible worlds is called the universe.

```

Let's look at a practical example.
In real world scenarios, data is most of the time provided as a table, dataframe, array, etc. 
The rows of such an object are referred to as samples or instances and the columns as variables or features.

```{code-cell} ipython3
from sklearn.datasets import load_iris

data = load_iris(as_frame=True)

dataframe = data.data
target = data.target.astype(str)
target[target == "0"] = "Setosa"
target[target == "1"] = "Versicolour"
target[target == "2"] = "Virginica"
dataframe["plant"] = target
dataframe
```

In this example, we see a dataset with 150 instances and 5 variables. 
Probability distributions can now be defined over events that are constructed over the complex random variable 
of this dataset.

### Independence

```{prf:definition} Independence
:label: def-independence
We say that $A$ is independent of $B$ is $(PA|B) = P(A)$ or equivalently that

$$P(A \cap B) = P(A) P(B).$$

Notation: $A \perp \!\!\! \perp B$. Information about $B$ does not give information about $A$ and vice versa.
```

In {prf:ref}`def-independence` $A$ and $B$ could be both events or event entire random variables.
Let's look at a practical example where we construct a joint probability distribution over colors and shapes.

```{code-cell} ipython3
from probabilistic_model.distributions.multinomial import MultinomialDistribution
import numpy as np
from enum import IntEnum
from random_events.set import Set

class Color(IntEnum):
    BLUE = 0
    RED = 1
    

class Shape(IntEnum):
    CIRCLE = 0
    RECTANGLE = 1
    TRIANGLE = 2


color = Symbolic("color", Set.from_iterable(Color))
shape = Symbolic("shape", Set.from_iterable(Shape))

probabilities = np.array([[2 / 15, 1 / 15, 1 / 5],
                          [1 / 5, 1 / 10, 3 / 10]])

distribution = MultinomialDistribution((color, shape), probabilities)
color_event = SimpleEvent({color: Color.BLUE}).as_composite_set()
shape_event = SimpleEvent({shape: (Shape.CIRCLE, Shape.TRIANGLE)}).as_composite_set()
joint_event = color_event & shape_event
print(f"P({joint_event}) = {distribution.probability(joint_event)}")
print(
    f"P({color_event}) * P({shape_event}) = {distribution.probability(color_event) * distribution.probability(shape_event)}")
```

We can see that both events are independent as their joint probability can be decomposed to the product of the marginal
probabilities.
However, in most applications one is interested in independence between entire variables (dimensions).
Checking independence over entire dimensions requires either analysis of the computation or
checking independence over all possible events.
Let's verify if the variables are independent.

```{code-cell} ipython3
for color_value, shape_value in itertools.product(color.domain.simple_sets, shape.domain.simple_sets):
        joint_event = SimpleEvent({color: color_value, shape: shape_value}).as_composite_set()
        color_event = SimpleEvent({color: color_value}).as_composite_set()
        shape_event = SimpleEvent({shape: shape_value}).as_composite_set()
        print(f"P({joint_event}) = {distribution.probability(joint_event)}")
        print(f"P({color_event}) * P(shape={shape_event}) = {distribution.probability(color_event) * distribution.probability(shape_event)}")
        print(np.allclose(distribution.probability(joint_event), 
                          distribution.probability(color_event) * distribution.probability(shape_event)))
        print("-" * 80)
```

As we can see, the entire variables are independent.


The definition of independence can be expanded to truncated independence. Formally,

### Conditional Independence

```{prf:definition} Conditional Independence
:label: def-truncated-independence

Two variables (events) $A$ and $B$ are conditionally independent given variable (event) $C$, 
if and only if their truncated distribution factorizes,

$$P(A,B|C) = P(A|C) P(B|C).$$

In that case we have $P(A|B, C) = P(A|C)$, i.e. in light of information $C$, $B$ provides no (further) information about
$A$. 

Notation: $A \perp \!\!\! \perp B \, | \, C$.
```

Let's explore this in another example.

```{code-cell} ipython3
:tag:

class Size(IntEnum):
    SMALL = 0
    LARGE = 1
    
size = Symbolic("size", Set.from_iterable(Size))

probabilities = np.array([[[2 / 30, 1 / 30, 1 / 10], [0, 0.3, 0.05]],
                          [[1 / 10, 1 / 20, 3 / 20], [ 0.15, 0, 0.,]]])
distribution = MultinomialDistribution((color, size, shape), probabilities)

small_event = SimpleEvent({size: Size.SMALL}).as_composite_set()
p_small = distribution.probability(small_event)

large_event = ~small_event
p_large = distribution.probability(large_event)

color_event = SimpleEvent({color: Color.BLUE}).as_composite_set()
color_and_small = color_event & small_event
color_and_large = color_event & large_event
p_color_and_small = distribution.probability(color_and_small)
p_color_and_large = distribution.probability(color_and_large)

shape_event = SimpleEvent({shape: Shape.CIRCLE}).as_composite_set()
shape_and_small = shape_event & small_event
shape_and_large = shape_event & large_event
p_shape_and_small = distribution.probability(shape_and_small)
p_shape_and_large = distribution.probability(shape_and_large)

shape_and_color_and_small = shape_event & color_event & small_event
p_shape_and_color_and_small = distribution.probability(shape_and_color_and_small)

shape_and_color_and_large = shape_event & color_event & large_event
p_shape_and_color_and_large = distribution.probability(shape_and_color_and_large)

p_shape_given_small = p_shape_and_small / p_small
p_color_given_small = p_color_and_small / p_small
p_shape_and_color_given_small = p_shape_and_color_and_small / p_small

p_shape_given_large = p_shape_and_large / p_large
p_color_given_large = p_color_and_large / p_large
p_shape_and_color_given_large = p_shape_and_color_and_large / p_large

print(f"P(Shape|Small) = {p_shape_given_small}")
print(f"P(Color|Small) = {p_color_given_small}")
print(f"P(Shape, Color|Small) = {p_shape_and_color_given_small}")
print(np.allclose(p_shape_given_small * p_color_given_small, p_shape_and_color_given_small))

print(f"P(Shape|Large) = {p_shape_given_large}")
print(f"P(Color|Large) = {p_color_given_large}")
print(f"P(Shape, Color|Large) = {p_shape_and_color_given_large}")
print(np.allclose(p_shape_given_large * p_color_given_large, p_shape_and_color_given_large))
````

In this example, we can observe that the variables are conditionally independent if the size of the object is small.
They are conditionally dependent if the size of the object is large.
In contrast to the previous example, the variables are not independent over the entire domain, only certain 
configurations (events) of the variables are independent.

### Continuous Random Variables

We just saw how to construct a joint probability table by enumerating all elementary events. 
However, regarding continuous spaces, this is not possible since one cannot enumerate all real numbers.

For real spaces, the probability density function is used. 
Firstly, though, we have to understand events in continuous spaces. 
In continuous spaces, we are interested in events that describe intervals on the real line. 
For instance, $P(0 \leq x \leq 1) $ is the probability that $x$ is between $0$ and $1$. 
Giving a formal description of the Borel algebra requires topological arguments, which I will not discuss. 
For most applications, it is enough to understand that we want to reason about intervals.

### Probability Density Function (PDF)

```{prf:definition} Probability Density Function (PDF)
:label: def-pdf

Let $\mathfrak{B}$ be the Borel sigma-algebra on $\mathbb{R}^d$. 
A probability measure $P$ on $(\mathbb{R}^d, \mathfrak{B})$ has a density $p$ if $p$ is a non-negative (Borel) 
measurable function on $\mathbb{R^d}$ satisfying for all $B \in \mathfrak{B}$

$$P(B) = \int_B p(x) dx =: \int_B p(x_1,...,x_d) dx_1 ... dx_d.$$
```

Bluntly speaking, the pdf is an always positive function that is high where we think that this region is highly likely
and low where we think it's not.

The integral over a pdf is called the cumulative distribution function and is the object we use to calculate the
probability of an event. Formally,

### Cumulative Distribution Function (CDF)

```{prf:definition} Cumulative Distribution Function (CDF)
:label: def-cdf

For probability measures $P$ on $(R^d, \mathfrak{B})$, the cumulative distribution function is the function

$$F(x) = P\left( \prod_{i=1}^d (X_i < x_i) \right).$$

If $F$ is sufficiently differentiable, then $P$ has a density given by

$$p(x) = \frac{\partial^dF}{\partial x_1, ..., \partial x_d}.$$

```

In this sense, the cdf over many variables describes the probability of the event that every variable is below some 
value (vector).
For example, if we evaluate $F \begin{pmatrix} 1 \\ 2 \end{pmatrix}$ we calculate the probability of the event that 
$x_1 < 1 \wedge x_2 < 2$.

```{code-cell} ipython3
from probabilistic_model.distributions.gaussian import GaussianDistribution
normal = GaussianDistribution(Continuous("x"), 0, 1)
fig = go.Figure(normal.plot(), normal.plotly_layout())
fig.for_each_trace(lambda trace: trace.update(visible='legendonly') if trace.name in ["Expectation", "Mode"] else ())
```

The figure above visualizes the objects we just discussed for a univariate normal distribution.


```{bibliography}
```