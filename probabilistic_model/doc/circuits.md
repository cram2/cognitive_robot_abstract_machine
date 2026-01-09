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

# Probabilistic Circuits

Unifying Probabilistic Modeling: There are various formalisms used for tractable probabilistic models. 
Probabilistic circuits offer a general and unified framework that can encompass these diverse approaches. 
This fosters a more cohesive understanding and facilitates advancements in the field.
The main resource for this section is {cite}`choi2020probabilistic`.

## Units

Probabilistic Circuits describe computational graphs that can be used to represent probability distributions.
The computational graphs are used to analyze the cost of answering queries.

Formally, 

````{prf:definition} Probabilistic Circuit
:label: def-probabilistic-circuit

A probabilistic circuit (PC) defined over variables $X$ is represented by a parameterized Directed Acyclic
Graph (DAG) with a single root node $n_r$. 
Every leaf node in the DAG represents an input node that defines a primitive 
distribution over some variable $x \in X$. 
Every inner node $n$ is either a sum node or a product node, which merges the distributions encoded by its children, 
denoted $ch(n)$, to construct more complex distributions. 
The distribution represented by every node is defined recursively as:

\begin{align*}
p_n(x) :=
\begin{cases}
f_n(x)  & \text{n is an input node,}\\
\prod_{c \in ch(n))} p_c(x) & \text{n is a product node }\\
\sum_{c \in ch(n))} \theta_{c, n} \cdot p_c(x) & \text{n is a sum node.}  
\end{cases}
\end{align*}

where $f_n(x)$ is an univariate input distribution (e.g., Gaussian, Categorical), and $\theta_{n,c}$ denotes the 
parameter corresponding to edge $(n, c)$. 
Intuitively, sum nodes model mixtures of their input distributions, which require the mixture
log_weights to be in the probability simplex: $\sum_{c \in ch(n)} \theta_{n,c} = 1$
and $\forall c \in ch(n), \theta_{n,c} \geq 0$. 
And product nodes build factorized distributions over their inputs. 
The size of a PC, denoted $|p|$, is the number of edges in its DAG. {cite}`liu2024scaling`
````


## Properties

The cost of calculating the quantities described in {ref}`chapter:queries` can now be described with respect to 
properties of the units in the circuit. 
While {cite}`choi2020probabilistic` describes and proofs these properties in detail, I will give a brief overview here.

### Decomposability

Whenever a query requires an integral decomposability is the most important property.
Before we look at this property, we have to define the scope of a unit.

````{prf:definition} Scope
:label: def-scope

Let $C = (G, \theta)$ be a PC with variables $X$. The computational graph $G$ is equipped with a scope function $\phi$ which associates 
to each unit $n \in G$ a subset of $X$, i.e., $\phi(n) \subseteq X$. 
For each non-input unit 
$$
n \in G, \phi(n) = \bigcup_{c \in ch(n)} \phi(c)
$$. 
The scope of the root of $C$ is $X$. {cite}`choi2020probabilistic`
````

Intuitively, the scope of a unit is all variables that are involved in the computation of the unit.

````{prf:definition} Decomposability
:label: def-decomposability

A product node $n$ is decomposable if the scopes of
its input units do not share variables: 
$$
\phi(c_i) \cap \phi(c_j) = \emptyset, \forall c_i, c_j \in ch(n), i \neq j.
$$

A PC is decomposable if all of its product units are decomposable. {cite}`choi2020probabilistic`

````


Next to decomposability being very important, it is also very restrictive and forbids the use of something like
a [linear layer](https://docs.nvidia.com/deeplearning/performance/dl-performance-fully-connected/index.html).

Decomposability is necessary the factor rule

$$
\int \int p_1(x) \cdot p_2(y) dx dy = \int p_1(x) dx \cdot \int p_2(y) dy
$$

only applies if the functions do not share variables. 

If they do, a haunting formula from your calculus class will come
back to you

$$
\int u(x) \cdot v(x) dx = \int u(x) dx \cdot \int v(x) dx + \int u'(x) \cdot v'(x) dx.
$$

Using the $uv$ substitution is often intractable and requires the use of numerical methods or unbounded many nested
substitutions. 
You perhaps remember the problems in figuring out the correct substitution and the resulting mess of terms from your
studies of calculus.  
Hence, decomposability is necessary for the queries {ref}`chapter:marginal_distributions`, 
{ref}`chapter:conditionals` and {ref}`chapter:moments`.

### Smoothness

The next important but less severe property is smoothness.
    
````{prf:definition} Smoothness
:label: def-smoothness

A sum node $n$ is smooth if its inputs all have identical scopes: 

$$
\phi(c) = \phi(n), \forall c \in ch(n).
$$ 

A circuit is smooth if all of its sum units are smooth.  {cite}`choi2020probabilistic`

````

We usually want smoothness to ensure a proper integral.
Long story short if we have a non-smooth sum unit, we would end up with an integral that looks like

$$\int \int p(x) dx dy$$

and hence, it has no useful probabilistic interpretation.

Smoothness is less severe than decomposability since it can be easily ensured. 

Smoothness is also a property needed for integration and hence required for the queries 
{ref}`chapter:marginal_distributions`, {ref}`chapter:conditionals` and {ref}`chapter:moments`.

### Determinism

The last interesting property is determinism.
Before we can define determinism, we have to define the support of a distribution.

````{prf:definition} Support
:label: def-support

The support of a distribution is the set of all elementary events that have a non-zero probability of occurring

$$
supp(p) = \{x \in X | p(x) > 0\}.
$$

````

````{prf:definition} Support as Random Event
The support of a smooth and decomposable circuit is an element of the product algebra with

$$
P(supp(p)) = 1.
$$

````

The support of a distribution can also be seen as the region of influence of a function. 
Now, with determinism, we can argue on {ref}`chapter:mode` calculations

````{prf:definition} Determinism
:label: def-determinism

A sum node $n$ is deterministic if, for any fully-instantiated input, 
the output of at most one of its children is nonzero

\supp(c_i) \cap \supp(c_j) = \emptyset, \forall c_i, c_j \in ch(n), i \neq j.
 
A circuit is deterministic if all of its sum nodes are deterministic. (Adapted from {cite}`choi2020probabilistic`)

````

Determinism intuitively ensures that the mode of the distribution can be calculated by ensuring that the regions of 
influence of every part of the circuit are disjoint. 
If they are disjoint (the circuit is deterministic), 
we can just get the mode by replacing every sum node with a max operation w.r.t. the likelihood.

The figure below shows a univariate Gaussian mixture model where we take the mode of the children as the mode of the 
distribution. 
We can easily see that this is not the real mode, since the regions of influence overlap. 
In general, whenever having distributions with infinite support in a mixture (sum unit), 
the mode is not obtained by replacing the sum node with a max operator.

```{code-cell} ipython3
:tags: [hide-input]

from random_events.variable import Continuous
from random_events.interval import *

import plotly.graph_objects as go
from probabilistic_model.distributions import *
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
import numpy as np

```

```{code-cell} ipython3

x = Continuous("X")

model = ProbabilisticCircuit()
s1 = SumUnit(probabilistic_circuit = model)
s1.add_subcircuit(leaf(GaussianDistribution(x, 0, 0.5), model), np.log(0.1))
s1.add_subcircuit(leaf(GaussianDistribution(x, 1, 2), model), np.log(0.9))

wrong_mode, wrong_max_likelihood = model.root.subcircuits[1].distribution.mode()
wrong_max_likelihood = model.likelihood(np.array([[wrong_mode.simple_sets[0][x].simple_sets[0].lower]]))[0]
mode_trace = model.univariate_mode_traces(wrong_mode, wrong_max_likelihood)

wrong_mode, wrong_max_likelihood = model.root.subcircuits[0].distribution.mode()
wrong_max_likelihood = model.likelihood(np.array([[wrong_mode.simple_sets[0][x].simple_sets[0].lower]]))[0]
mode_trace += model.univariate_mode_traces(wrong_mode, wrong_max_likelihood)

fig = go.Figure(model.plot(), model.plotly_layout())
fig.add_traces(mode_trace)
fig.show()
```

The next figure shows that if we truncated the children of the sum node to a disjoint support, we get the correct mode.

```{code-cell} ipython3

model = ProbabilisticCircuit()
s1 = SumUnit(probabilistic_circuit = model)
s1.add_subcircuit(leaf(TruncatedGaussianDistribution(x, open_closed(-np.inf, 0.5).simple_sets[0], 0, 0.5), model), np.log(0.1))
s1.add_subcircuit(leaf(TruncatedGaussianDistribution(x, open(0.5, np.inf).simple_sets[0], 1, 2), model), np.log(0.9))

fig = go.Figure(model.plot(), model.plotly_layout())
fig.show()
```

## Limitations

While circuits are a great and wonderful tool to represent complex probability distributions, 
they have their limitations.
One of the limitations comes from the sum units.
Sum Units can be interpreted as latent variables.

````{prf:theorem} Latent Variable Interpretation
:label: theo-latent-variable

A sum unit $n$ can be interpreted as a latent symbolic variable.
The domain of the variable are the children of the sum unit.
The probabilities of the sum unit are given by

$$
p(c) = \theta_{c, n}.
$$

Since sum units are convex, i. e. $\sum_{c \in ch(n)} \theta_{n,c} = 1$ and $\forall c \in ch(n), \theta_{n,c} \geq 0$,
the sum unit can be interpreted as a latent variable. 
When performing inference, the latent variables are marginalized.
````

```{margin} Continuous Mxitures
Integration over continuous latent variables is further discussed here {cite}`gala2024probabilistic` and here
{cite}`martires2024probabilistic`.
```

In {prf:ref}`theo-latent-variable`, we see that the sum unit can be interpreted as a *discrete* latent variable.
Whenever we would use a sum unit with a continuous latent variable, or a discrete latent variable with infinite components,
inference would require marginalizing the latent variable which results in intractable integrals. 


The second limitation comes from decomposability.
A typical approach to modern computational graphs is heavy use of linear algebra.
This is not possible with decomposable product units. 
Consider a linear transformation 

$$
f(x, y) = \begin{pmatrix} 1 & 2 \\ 3 & 4 \end{pmatrix} \begin{pmatrix} x \\ y \end{pmatrix}.
$$

The probability distributions would then be expressed over the transformed variables and require us to use a change 
of variables here.

\begin{align*}
x_{new} &= x + 2y  &y_{new} &= 3x + 4y\\
x &= x_{new} - 2y &y &= -\frac{3}{2}x_{new} + 2y_{new}.
\end{align*}

When one would perform inference on these, one is still interested in the original variables $x$ and $y$.
Performing an integral over the transformed variables would require the use of the Jacobian of the transformation
and integrate over it, which is intractable for all interesting transformations.

This behaviour of the probability and the change of variables is the last theorem we will discuss 
(which I didn't tell you yet since it is rather complicated).

````{prf:theorem} Change of Variables
:label: theo-change-of-variables

Let $X = (X_1, \dots, X_d)$ have a joint density $p_x$. Let $g: \mathbb{R}^d \rightarrow \mathbb{R}^d$
be continuously differentiable and injective, with non vanishing Jacobian $J_g$. Then $Y = g(X)$ has
density
\begin{align*}
    p_Y(y) = 
    \begin{cases}
        p_x(g^{-1}(y) \cdot |J_{g^{-1}}(y)|) & \text{if y is in the range of g} \\
        0 & \text{otherwise}    
    \end{cases}
\end{align*}

{cite}`hennig2020pml`
````

## Normalizing Flows

Normalizing flows are a way to overcome the limitations of probabilistic circuits at the price of tractable inference.
Informally, a normalizing flow consists of an easy, usually fully factorized  gaussian latent distribution, 
and a series of invertible and differentiable transformations that transform the latent distribution into the target 
distribution.

One can imagine this like a portal from the real world to a different world where things are easier to calculate.
However, the portal is not free and requires the calculation of the Jacobian of the transformation.

Hence, integration is not possible in the real world, and the integral approximation {ref}`chapter:monte-carlo` 
is used to evaluate the integral.

A common flow is the multivariate gaussian where the transformation is a linear transformation,
and the latent distribution is indeed a fully factorized gaussian.
This section only points to this idea, and I recommend reading further in here {cite}`papamakarios2021normalizing` 
if you are interested.

```{bibliography}
```


