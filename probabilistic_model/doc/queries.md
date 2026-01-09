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

(chapter:queries)=
# Queries
The second chapter walks through the most important queries in probabilistic modeling.
There are common quantities that are of interest regarding probabilities. 
The process of calculating these quantities is called inference or querying.
You, most likely, have seen them in one or another form, but I will introduce them formally accompanied by some 
examples.


## Likelihoods
The likelihood query is the evaluation of the joint probability distribution at a point (possible world).
In such a query, no variable is left unassigned and no integration is performed.
Likelihood queries just calculate $p(x)$. 
The example below shows such a query.


```{code-cell} ipython3
:tags: [hide-input]

from random_events.set import SetElement, Set
from random_events.variable import Symbolic, Continuous, Integer
from random_events.product_algebra import Event, SimpleEvent
from random_events.interval import *
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import itertools
from probabilistic_model.distributions.multinomial import MultinomialDistribution
import numpy as np
from enum import IntEnum


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

from sklearn.datasets import load_iris

data = load_iris(as_frame=True)

dataframe = data.data
target = data.target.astype(str)
target[target == "0"] = "Setosa"
target[target == "1"] = "Versicolour"
target[target == "2"] = "Virginica"
dataframe["plant"] = target

```

```{code-cell} ipython3
possible_world = np.array([[Color.BLUE, Shape.CIRCLE]])
print(f"p({color.name}={possible_world[0, 0]}, {shape.name}={possible_world[0, 1]}) = {distribution.likelihood(possible_world)}")
```

In the abstract interface to probabilistic models in this package, the likelihood query is implemented [here](https://probabilistic-model.readthedocs.io/en/latest/autoapi/probabilistic_model/probabilistic_model/index.html#probabilistic_model.probabilistic_model.ProbabilisticModel.likelihood).


## Maximum Likelihood Estimate

While the likelihood query has little use in probabilistic applications, 
it is essential for the construction and learning of models.

It is the basic formula this is evaluated a vast number of times in the maximum likelihood based learning methods. 

Formally,

````{prf:definition} Maximum Likelihood Estimate
:label: def-mle

$L(\theta)$ is called the likelihood function. 
Goal: Determine 

$$\hat{\theta}_{MLE}=\underset{\theta\in\Theta}{arg \,max} L(\theta)=
\underset{\theta\in\Theta}{arg \,max}\prod_{i=1}^NP(\mathcal{D}_i | \theta),$$

which is called the Maximum Likelihood Estimate (MLE), 
where $\theta$ are the parameters of the distribution $p$ and $N$ being the number of observations.
````
As gradient descent is a common way to maximize the likelihood function, 
the gradient of said function is also of interest. 
Calculating the gradient of a big product of functions is problematic due to the complexity of the [product rule](https://en.wikipedia.org/wiki/Product_rule). 
Instead, the log-likelihood function is used, since the maximum of the logarithm of a function is also the maximum of 
the function itself. 

Formally,

````{prf:theorem} Maximum Log-Likelihood Estimate
:label: def-mlle

The function

$$\hat{\theta}_{MLLE}=\underset{\theta\in\Theta}{arg \,max} log(L(\theta)) = 
\underset{\theta\in\Theta}{arg \,max}\sum_{i=1}^N log(P(\mathcal{D}_i | \theta))$$

calculates the log-likelihood of the data $\mathcal{D}$ given the parameters $\theta$.
The log-likelihood function has the same optimal parameter-vector as the likelihood function, 
$\hat{\theta}_{MLLE} = \hat{\theta}_{MLE}$.

````

## Independently and Identically Distributed

As we can see in the maximum likelihood estimate, the likelihood function is a product of many probabilities. However, multiplying probabilities to achieve a joint probability is only allowed if the events are independent, as we saw in the definition of independence. In practice, we often assume that the data is independently and identically distributed (i.i.d.) in order to allow efficient calculations. The maximum likelihood estimate can be summarized as:
>   Given some arbitrary but fixed parameter $\theta$, what is the probability that we get the
    observed data D assuming each world being independently drawn from the identical
    underlying distribution (i.i.d. assumption)
    
Since determining the exact computations needed for a maximum likelihood estimate is subject to calculus, 
I will not annoy you with that and instead show you a practical example.
For a Normal Distribution, we know that the maximum likelihood estimate for the mean is the sample mean and for the 
variance the sample variance.




```{code-cell} ipython3
from probabilistic_model.distributions import *
mean = dataframe["sepal length (cm)"].mean()
variance = dataframe["sepal length (cm)"].std()

sepal_length = Continuous("sepal length (cm)")
distribution = GaussianDistribution(sepal_length, mean, variance)
fig = go.Figure(distribution.plot(), distribution.plotly_layout())
fig.show()
```

The plot shows the normal distribution that is the maximum likelihood estimate for the data if we assume the data 
is i. i. d. and drawn from a normal distribution.

(chapter:marginals)=
## Marginals

The marginal query is the next interesting quantity we investigate.
A common scenario in applications of probability theory is reasoning under uncertainty. 
Consider the distribution from above that describes the sepal length in centimeters. 
How probable is it that a plant has a sepal length between 6 and 7 cm?
Answering such questions requires the integration over the described area.  
In this scenario, we can get the probability by the following piece of code.

```{code-cell} ipython3
event = SimpleEvent({sepal_length: closed(6, 7)}).as_composite_set()
distribution.probability(event)
```

We can see that the probability of such an event is approximately $34\%$.
Formally, we can describe such a query as:

````{prf:definition} Marginal Query class
:label: def-marginal

Let $p(X)$ be a joint distribution over random variables $X$. 
The class of marginal queries over $p$ is the set of functions that compute

$$p(E = e, Z \in \mathcal{I}) = \int_\mathcal{I} p(z, e) dZ.$$

where $e \in dom(E)$ is a partial state for any subset of random variables $E \subseteq X$, 
and  $Z = X \setminus E$ is the set of $k$ random variables to be integrated over intervals 
$I = I_1 \times \cdots \times I_k$ each of which is defined over the domain of its corresponding 
random variables in $Z: I_i \subseteq dom(Z_i) $ for $ i = 1, \cdots, k.$

````

While this definition may be a bit weird to think about, 
it essentially says that marginal queries are integrations over axis aligned bounding boxes. 
Furthermore, marginal queries can also contain (partial) point descriptions, just as in the likelihood. 

[This tutorial](https://random-events.readthedocs.io/en/latest/conceptual_guide.html) dives deeper in the product 
algebra that is constructed by the marginal query class. 

(chapter:conditionals)=
## Conditionals

While marginal queries already allow for the calculations of truncated probabilities using the definition of the 
truncated probability, it is, in most scenarios, more interesting to consider the truncated probability space. 
We can construct such a thing by invoking the `truncated` method with the corresponding event.

```{code-cell} ipython3
event = SimpleEvent({sepal_length: closed(6, 7)}).as_composite_set()
distribution, probability = distribution.truncated(event)
fig = go.Figure(distribution.plot(), distribution.plotly_layout())
fig.show()
```

We can see that conditioning on the event that the sepal length is between 6 and 7 cm, the resulting distribution is a 
zoomed in version of the original distribution. 
The resulting type of distribution is not a Gaussian distribution anymore, but a truncated Gaussian distribution. 
The probability of the event that the sepal length is between 6 and 7 cm is now $100\%$. 
Since the truncated Gaussian is a much more complicated object than the ordinary Gaussian distribution, y
ou can read more about it [here](https://probabilistic-model.readthedocs.io/en/latest/examples/truncated_gaussians.html).

```{code-cell} ipython3
distribution.probability(event)
```

(chapter:marginal_distributions)=
## Marginal Distributions

The next interesting object that belongs to the marginal query class is the marginal distribution. The marginal distribution is the distribution over a subset of random variables. To achieve this, all other random variables are integrated out. The marginal distribution is the distribution over the subset of random variables. The marginal distribution can be obtained by invoking the `marginal` method with the corresponding subset of random variables.
Consider the multidimensional distribution over colors and shapes from above. We can obtain the marginal distribution over colors by the following piece of code.

```{code-cell} ipython3
import tabulate

probabilities = np.array([[2/15, 1/15, 1/5],
                          [1/5, 1/10, 3/10]])

distribution = MultinomialDistribution((color, shape), probabilities)
color_distribution = distribution.marginal([color])
print(tabulate.tabulate(color_distribution.to_tabulate(), tablefmt="fancy_grid"))
```

A final remark on the marginal method, is that every inference method that is part of the marginal query class is 
still efficiently calculable. 
A problem is that, it sometimes destroys a property that is needed for finding the mode of the distribution. 
We will investigate later what that means.

(chapter:moments)=
## Moments

The final interesting quantity that belongs to the marginal query class is the moment of a distribution. 

````{prf:definition} Moment Query class
:label: def-moment

The (central) moment of a distribution is defined as the integral

$$\mathbb{M}_n(x) = E((x-E(x))^n) = \int (x-\mu)^n p(x) dx,$$

where $n$ is the order of the moment and $\mu$ is the center of the moment.

````

The most common moments are the expectation given as

$$E(x) = \int x \, p(x) dx,$$

which is a moment query with $n=1$ and center $0$. 

The variance is the second central moment and given by

$$Var(x) = \int (x-E(x))^2 p(x) dx, $$

which is a moment query with the mean as the center and $n=2$. 

While higher order moments exist $(n>2)$, they have little practical use and are not further discussed here.

The interface to calculate moments is the `moment` method of the distribution object. 

```{code-cell} ipython3
from random_events.product_algebra import VariableMap
distribution = GaussianDistribution(sepal_length, mean, variance)
print("mean:", distribution.expectation(distribution.variables))
print("variance:", distribution.moment(VariableMap({sepal_length: 2}), distribution.expectation(distribution.variables)))
print("Raw variance:", distribution.moment(VariableMap({sepal_length: 2}), VariableMap({sepal_length: 0})))
print("Third moment with center 3:", distribution.moment(VariableMap({sepal_length: 3}), VariableMap({sepal_length: 3})))
```

As we can see, the expectation and variance are shortcut by their names since they are so common. 

Furthermore, we can calculate all moments that exist by plugging in the order and center that we want as `VariableMap`.


(chapter:mode)=
## Mode query

The next important quantity of a probability distribution is the mode. 
The mode refers to the region where the density is maximized. 
Formally,

````{prf:definition} Mode Query class

The mode of a distribution is defined as the set where the density is maximal, i.e.,

$$\hat{x} = \underset{x \in \mathcal{X}}{arg \,max} p(x). $$

````

While common literature describes the mode under a condition, we can omit such a description since we already defined that the truncated distribution is part of the marginal query class. Hence, the mode under a condition is just the chain of the condition and mode methods.

A common perception of the mode is that it is the single point of highest density, such as in the example below.

```{code-cell} ipython3
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *
distribution = ProbabilisticCircuit()
leaf(GaussianDistribution(Continuous("x"), 0, 1), distribution)
fig = go.Figure(distribution.plot(), distribution.plotly_layout())
fig.show()
```

However, the mode is more accurately described as the set of points with the highest density. Consider the example below.

```{code-cell} ipython3
condition = closed(-float("inf"), -1) | closed(1, float("inf"))
distribution, _ = distribution.truncated(SimpleEvent({distribution.variables[0]: condition}).as_composite_set())
go.Figure(distribution.plot(), distribution.plotly_layout()).show()
```

We can see that conditioning a Gaussian on such an event already creates a mode that has two points. Furthermore, modes can be sets of infinite many points, such as shown below.

```{code-cell} ipython3
uniform = leaf(UniformDistribution(Continuous("x"), open(-1, 1).simple_sets[0]), ProbabilisticCircuit()).probabilistic_circuit
go.Figure(uniform.plot(), uniform.plotly_layout()).show()
```

The mode of the uniform distribution is the entire interval of the uniform distribution $(-1, 1) $. 
The mode is particularly useful when we want to find the best (most likely) solution to a problem and not just any.


## Sampling

Sampling refers to the generation of random samples from a distribution. 
Defining sampling formally is a bit tricky, as it is not a query in the sense of the other queries.
However, it is an important part of probabilistic modeling, as it allows us to generate examples from a distribution.
If you are interested in a good definition of sampling, I recommend {cite:p}`Sampling2017StackExchange`.

For most practical purposes, we can define sampling as the generation of random samples from a distribution.
Let's look at an example from the Gaussian distribution.

```{code-cell} ipython3
distribution = GaussianDistribution(Continuous("x"), 0, 1)
samples = distribution.sample(2)
print(samples)
```

As we can see, we just draw two samples from the Gaussian distribution.

(chapter:monte-carlo)=
## Monte Carlo Estimate

In {prf:ref}`def-marginal`, we defined the marginal query class as the integration over events of the product algebra.
In practice, quantities such as the marginal probability are often intractable to compute.
Furthermore, there are integrals that are not part of the marginal query class and yet interesting to calculate.
One way to approximate such integrals is by using Monte Carlo methods.

````{prf:definition} Monte Carlo Estimate
:label: def-monte-carlo

Consider k indipendent samples $x_1, ..., x_k$ from a multidimensional random variable with a certain pdf $p(x)$. 
Then a Monte Carlo estimate would be a way of approximating multidimensional integrals of the form

$$
\int f(x) p(x) dx
$$

by using the empirical expectation of the function $f$ evaluated at the samples

$$
\int f(x) p(x) dx \approx \frac{1}{k} \sum_{i=1}^k f(x_i).
$$

The Monte Carlo estimate is sometimes reffered to as the expectation of the function $f$ under the distribution $p$.
````

The Monte Carlo estimate is a powerful tool to approximate integrals that have unconstrained form.

In general, Monte Carlo Methods are integral approximations and not tied to probability distributions. 
Until now, we described our integrals as random events. 
Let $E$ be a random event as we know it, consider the function

$$
\mathbb{1}_E(x) = \begin{cases} 1 & \text{if } E \models x \\ 0 & \text{else} \end{cases}
$$

then,

$$
\int \mathbb{1}_E(x) p(x) dx
$$

calculates $P(E)$. 
The Monte Carlo Estimate yields an approximation by

$$
P(E) \approx = \frac{1}{k} \sum_{i=1}^k \mathbb{1}_E(x_i).
$$

For instance, consider a two-dimensional random variable $X = (X_1, X_2)$ with a standard normal distribution 
$p(x) = \mathcal{N}(x_1 | 0, 1) \cdot \mathcal{N}(x_2 | 0, 1) $.

```{code-cell} ipython3

x1 = Continuous("x1")
x2 = Continuous("x2")

model = ProbabilisticCircuit()
prod = ProductUnit(probabilistic_circuit = model)
p_x1 = leaf(GaussianDistribution(x1, 0, 1), model)
p_x2 = leaf(GaussianDistribution(x2, 0, 1), model)
prod.add_subcircuit(p_x1)
prod.add_subcircuit(p_x2)

fig = go.Figure(model.plot(), model.plotly_layout())
fig.show()

samples = model.sample(10000)
```
We can now calculate the probability that $x_1 \in (0.5, 1.)
\land x_2 \in (0.75, 1.) $ not only by using integration but also by a monte carlo estimate.

```{code-cell} ipython3
event = SimpleEvent({x1: closed(0.25, 1.), x2: closed(0., 1.)}).as_composite_set()
monte_carlo_probability = sum([event.contains(sample) for sample in samples]) / len(samples)

fig.add_traces(event.plot())
fig.show()

print("Monte Carlo Probability:", monte_carlo_probability)
print("Exact Probability:", model.probability(event))
``` 
As we can see, the Monte Carlo estimate is a good but rough approximation of the exact probability.

Since Monte Carlo estimates are not constrained by any form of $f$, we can use them to approximate any integral, such as
$P(x_1 < x_2)$ or similar complex integrals. 
These integrals are often intractable to solve analytically and can only be calculated by Monte Carlo estimates.

```{code-cell} ipython3
filtered_samples = [sample for sample in samples if sample[0] < sample[1]]
monte_carlo_probability = len(filtered_samples) / len(samples)
print("Monte Carlo Probability:", monte_carlo_probability)
```


For further examples on Monte Carlo estimates,
I recommend [this Monte Carlo Integration example](https://en.wikipedia.org/wiki/Monte_Carlo_method).

## Practical Example

In practice, probabilities can be used in robotics. 
Consider the kitchen scenario from the [product algebra notebook](https://random-events.readthedocs.io/en/latest/conceptual_guide.html#application).

```{code-cell} ipython3

x = Continuous("x")
y = Continuous("y")

kitchen = SimpleEvent({x: closed(0, 6.6), y: closed(0, 7)}).as_composite_set()
refrigerator = SimpleEvent({x: closed(5, 6), y: closed(6.3, 7)}).as_composite_set()
top_kitchen_island = SimpleEvent({x: closed(0, 5), y: closed(6.5, 7)}).as_composite_set()
left_cabinets = SimpleEvent({x: closed(0, 0.5), y: closed(0, 6.5)}).as_composite_set()

center_island = SimpleEvent({x: closed(2, 4), y: closed(3, 5)}).as_composite_set()

occupied_spaces = refrigerator | top_kitchen_island | left_cabinets | center_island
free_space = kitchen - occupied_spaces
fig = go.Figure(free_space.plot(), free_space.plotly_layout())
fig.show()
```

Let's now say that we want a position to access the fridge. 
We want to be as close to the fridge as possible. 
Naturally, we need a gaussian model for that. 
We now construct a gaussian distribution over the free space
to describe locations and their probabilities to access the fridge.

```{code-cell} ipython3
pc = ProbabilisticCircuit()

p_x = leaf(GaussianDistribution(x, 5.5, 0.5), pc)
p_y = leaf(GaussianDistribution(y, 6.65, 0.5), pc)
p_xy = ProductUnit(probabilistic_circuit = pc)
p_xy.add_subcircuit(p_x)
p_xy.add_subcircuit(p_y)
p_xy = p_xy.probabilistic_circuit
fig = go.Figure(p_xy.plot(number_of_samples=500), p_xy.plotly_layout())
fig.show()
```

We now want to filter for all positions that are available in the kitchen. 
Hence, we need to condition our probability distribution on the free space of the kitchen. 
We can do this by invoking the `truncated` method of the distribution object.

```{code-cell} ipython3
distribution, _ = p_xy.truncated(free_space)
fig = go.Figure(distribution.plot(number_of_samples=500), distribution.plotly_layout())
fig.show()
```

As you can see, we can express our belief of good accessing positions for the fridge using probability theory. 
This idea scales to many more complex scenarios,
such as the localization of a robot in a room or the prediction of the next state of a system.
However, this is out of scope for this tutorial.

```{bibliography}
```
