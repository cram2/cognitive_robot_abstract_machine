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

# Truncated Gaussian

Conditioning a probabilistic circuit on a random event requires conditioning of any kind of distribution to an interval 
on the real line. 
Unfortunately, the literature regarding the usage of a truncated Gaussian in a probabilistic circuit is not very 
extensive. 
In this notebook, we will explore the usage of a truncated Gaussian in a probabilistic circuit.

First, let us create a Normal distribution:

```{code-cell} ipython3
from random_events.interval import closed
from random_events.variable import Continuous
from random_events.product_algebra import Event, SimpleEvent
from probabilistic_model.distributions import GaussianDistribution, TruncatedGaussianDistribution
import plotly
plotly.offline.init_notebook_mode()
import plotly.graph_objects as go

variable = Continuous("x")
distribution = GaussianDistribution(variable, location=0, scale=1)
fig = go.Figure(distribution.plot())
fig.update_layout(title="Normal Distribution", xaxis_title=distribution.variable.name)
fig.show()
```

Whenever one conditions a Gaussian Distribution to an event, the result of that action will give a truncated Gaussian. 
Let us see what happens:

```{code-cell} ipython3
evidence = SimpleEvent({variable: closed(0.5, 2)}).as_composite_set()
conditional_distribution, evidence_probability = distribution.truncated(evidence)
fig = go.Figure(conditional_distribution.plot())
fig.update_layout(title="Normal Distribution", xaxis_title=distribution.variable.name)
fig.show()
```

Perhaps it is a bit surprising that computing (non-centered) moments for a truncated Gaussian is actually pretty hard.
For this, one has to calculate the integral

$$
\int_a^b (x-center)^{order} p(x) dx 
$$


While this integral shares some similarities with the ones from a Gaussian distribution, notice how the Expectation 
operator does not enjoy a full support (i.e., the real line), indeed, the above integral would be equivalent to:

$$ 
\mathbb{E} \left[ \left( X-center \right) ^{order} \right]\mathbb{1}_{\left[ lower , upper \right]}(x)
$$


A solution to this problem could be found by recursively using the raw moments of the truncated distribution and the 
binomial expansion of the inner factor inside the operator, as in:

$$
\mathbb{E} \left[ \left( X-center \right) \right]\mathbb{1}_{\left[ lower , upper \right]}(x) = \mathbb{E} \left[ \left( X \right) \right] \mathbb{1}_{\left[ lower , upper \right]}(x) - center 
$$

$$ 
\mathbb{E} \left[ \left( X-center \right)^2 \right]\mathbb{1}_{\left[ lower , upper \right]}(x) = \mathbb{E} \left[ X^2 \right] \mathbb{1}_{\left[ lower , upper \right]}(x) + center^2 - 2 center \mathbb{E} \left[ X \right]\mathbb{1}_{\left[ lower , upper \right]}(x) 
$$

And so on.
Thus, one can easily obtain the following result:

$$ 
\mathbb{E} \left[ \left( X-center \right)^{order} \right]\mathbb{1}_{\left[ lower , upper \right]}(x) = \sum _{i=0}^{order} \binom{order}{i} \mathbb{E}[X^i] \mathbb{1}_{\left[ lower , upper \right]}(x) (-center)^{(order-i)} 
$$

Moreover, {cite}`jawitz2003truncated` has shown that the following result holds for raw moments:

$$ 
\mathbb{E}[X^i] \mathbb{1}_{\left[ lower , upper \right]}(x) = \frac{1}{\sqrt{\pi}}\sum_{k=0}^{i}\binom{i}{k}\mu^{(i-k)}\left( \sigma \sqrt{2} \right)^k \int_{y(lower)}^{y(upper)}y^k e^{(-y^2)}dy 
$$

Where $\mu$ and $\sigma$ are the mean and the variance of the starting Gaussian distribution, 
and the variable $y$ is obtained by the following change of variables:

$$ 
y = \frac{x-\mu}{\sigma\sqrt{2}} 
$$ 

and 

$$ 
dy = \frac{dx}{\sigma\sqrt{2}} 
$$

However, it is known that conventional recursive formulas for truncated moments tend to suffer from 
subtracted cancellation errors associated, with the differences of large values giving inaccuracies as their orders become high {cite}`pollak2019doublerecursion`.
Furthermore, the computation time of higher order moments can explode whenever one deals with recursive formulas. 
That is why our implementations follow the paper by {cite}`ogasawara2022moments`, 
as a non-recursive method is defined for dealing with orders of any kind (this includes real-valued orders).  

For the simple case of a double truncated Gaussian, the formula proposed by {cite}`ogasawara2022moments` is as follows:

$$
\sigma^{order} \frac{1}{\Phi(upper)-\Phi(lower)} \sum_{k=0}^{order} \binom{order}{k} I_k (-center)^{(order-k)} 
$$

Where:

$$ 
I_k = \frac{2^{\frac{k}{2}}}{\sqrt{\pi}}\Gamma \left( \frac{k+1}{2} \right) \left[ sgn \left(upper\right) \mathbb{1}\left \{ k=2 \nu \right \} + \mathbb{1} \left\{k = 2\nu -1 \right\} \frac{1}{2} F_{\Gamma} \left( \frac{upper^2}{2},\frac{k+1}{2} \right) -  sgn \left(lower\right) \mathbb{1}\left \{ k=2 \nu \right \} + \mathbb{1} \left\{k = 2\nu -1 \right\} \frac{1}{2} F_{\Gamma} \left( \frac{lower^2}{2},\frac{k+1}{2} \right) \right] 
$$

for $k, \nu = 0,1,\ldots$.

The term $F_{\Gamma}(x,a)$ denotes the cdf of the gamma distribution at $x$ when the shape parameter is $a$ with the 
unit scale parameter.
