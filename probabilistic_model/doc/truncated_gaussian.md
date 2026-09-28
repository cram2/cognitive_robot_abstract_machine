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
from probabilistic_model.distributions.gaussian import GaussianDistribution, TruncatedGaussianDistribution
import plotly
plotly.offline.init_notebook_mode()
import plotly.graph_objects as go

variable = Continuous("x")
distribution = GaussianDistribution(variable=variable, location=0, scale=1)
fig = go.Figure(distribution.plot())
fig.update_layout(title="Normal Distribution", xaxis_title=distribution.variable.name)
fig.show()
```

Whenever one conditions a Gaussian Distribution to an event, the result of that action will give a truncated Gaussian. 
Let us see what happens:

```{code-cell} ipython3
evidence = SimpleEvent.from_data({variable: closed(0.5, 2)}).as_composite_set()
conditional_distribution, evidence_probability = distribution.truncated(evidence)
fig = go.Figure(conditional_distribution.plot())
fig.update_layout(title="Normal Distribution", xaxis_title=distribution.variable.name)
fig.show()
```

Sampling, the likelihood, the cumulative distribution function and the moments of a truncated Gaussian are computed by
[`scipy.stats.truncnorm`](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.truncnorm.html).
A moment about a center $c$,

$$
\mathbb{E} \left[ \left( X - c \right)^{order} \right],
$$

is computed as the raw moment of the truncated Gaussian shifted by $-c$.
Expanding it into raw moments of $X$ instead would subtract large, nearly equal numbers far in the tails of the
Gaussian, where the variance is small compared to the squared mean.

```{code-cell} ipython3
from random_events.product_algebra import VariableMap

mean = conditional_distribution.moment(VariableMap({variable: 1}), VariableMap({variable: 0}))[variable]
variance = conditional_distribution.moment(VariableMap({variable: 2}), VariableMap({variable: mean}))[variable]
mean, variance
```
