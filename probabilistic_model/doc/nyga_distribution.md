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

(nyga-distribution)=
# Learning univariate, model-free and deterministic distributions

````{margin}
## History
The concept of a similar distribution to the Nyga distribution was first introduced in a paper by 
Daniel Nyga {cite}`nyga2023joint`.
Daniels' creation was the so-called quantile distribution. 
Back in the day, it was not connected to circuits or a maximum likelihood estimation. 
The idea was to create a distribution that is able to approximate any distribution without any 
assumptions. 
Daniel used the mean squared error between the learned quantile function and the 
[empirical CDF](https://en.wikipedia.org/wiki/Empirical_distribution_function).
However, this sometimes leads to undesired results, especially with containing jumps in the distribution
 or completely missing parts of the distribution. 
Now, with the MLE proof on how to do it properly, we can
create distributions that overcome these issues. 
Yet the idea and hence the name lives on.

````

In this tutorial, I go through the line of thinking, math and usage of learning univariate, 
model-free and deterministic distribution from raw data.

In most machine learning scenarios, the distribution of the data is unknown. 
Learning a distribution is then usually done by assuming a functional form and fitting the parameters of the function 
to the data.
This is called parametric learning.
 
However, we learn a distribution without assuming a functional form. This is called non-parametric learning.

Let $\mathcal{D}$ be a dataset of $N$ samples $\mathcal{D} = \{d_1, d_2, \dots, d_N\}$, where $x_i \in \mathbb{R}$. 
The dataset is constructed from a mixture of normal distributions $p(x) = 0.5 \cdot N(0, 1) + 0.5 \cdot N(5, 0.5)$.

```{code-cell} ipython3
import numpy as np
np.random.seed(69)
dataset = np.concatenate((np.random.normal(0, 1, size=(1000, 1)), np.random.normal(5, 0.5, size=(1000, 1))))
dataset
```

The idea is to create as many components in a mixture of uniform distributions as is necessary 
to achieve a good fit to the data. 
The fitness of the model is measured by the weighted log likelihood of the data under the model (compare with {prf:ref}`def-mle`).
Every sample $d_i$ carries a weight $w_i$, for example the number of times it occurs in the data.
The weighted log likelihood of the data under the model is given by

$$
log \left( L(\mathcal{D}| \boldsymbol{\Theta}) \right) =  \sum_{i=1}^N w_i \, log(p(d_i)),
$$

where $\boldsymbol{\Theta}$ represents the parameters of the model.

When the uniform distribution is assumed for the data, the weighted log likelihood is given by

$$
log \left(L(\mathcal{D}| \boldsymbol{\Theta}) \right) = \sum_{i=1}^N w_i \, log \left(\frac{1}{b-a} \right).
$$

Since the density of the uniform distribution is constant, the weighted log likelihood can be restated as

$$ - W \, log \left(b-a \right), \qquad W = \sum_{i=1}^N w_i,$$
 
which is particularly easy to calculate. 

In case of a mixture of two uniform distributions, the likelihood is given by

$$
p(x_i) = w_{left} \cdot U_{left} + w_{right} \cdot U_{right}
$$


We will solve this problem using a recursive partitioning scheme. 
First off, we will sort the data, make it unique.
Next, we iterate through every possible partitioning that can be made on the data, such that we get two datasets

$$
\mathcal D_{left} = {d_1, d_2, \dots, d_k} \\
\mathcal D_{right} = {d_{k+1}, d_{k+2}, \dots, d_N}
$$

The split values are calculated by the distance maximizing value between $d_k$ and $d_{k+1}$.

$$
d_{split} = \frac{d_k + d_{k+1}}{2}
$$


The likelihood of such a split is given by assuming a uniform distribution on the left and right side of the split. 
This constructs a deterministic mixture of uniform distributions where the log_weights are given by the relative sum of 
log_weights on the left and right side of the split.

\begin{align*}
w_{total} &= \sum_{i=1}^N w_i \\
w_{left} &= \frac{\sum_{i=1}^k w_i}{w_{total}}\\
w_{right} &= \frac{\sum_{i=k+1}^N w_i}{w_{total}} 
\end{align*}

The density of the mixture is given by 

$$
p(x) = w_{left} \, U_{left} + w_{right} \, U_{right} = \frac{w_{left}}{d_{split}-d_1} + \frac{w_{right}}{d_N - d_{split}}.
$$

The determinism allows simplifying the likelihood calculation for all samples in the left split to just

\begin{align*}
p_{left}(x) &= w_{left} \, U_{left} + w_{right} \, \underbrace{U_{right}}_{ = 0} = w_{left} \, U_{left} \\
log(p_{left}(x)) &= log(w_{left}) - log(d_{split}-d_1)
\end{align*}

and for the right split to

\begin{align*}
p_{right}(x) = w_{right} \, U_{right} + w_{left} \, \underbrace{ U_{left}}_{ = 0} = w_{right} \, U_{right} \\
log(p_{right}(x)) = log(w_{right}) - log(d_N - d_{split})
\end{align*}

Furthermore, we can observe that the density is constant for every sample in the left split and constant for every sample in the right split.


Plugging it in, we get the likelihood for the split

\begin{align*}
log \left( L(\mathcal{D}| d_1, d_{split}, d_N) \right) &= \sum_{i=1}^{k} w_i \, log(p_{left}(d_i)) + \sum_{i=k+1}^{N} w_i \, log(p_{right}(d_i)) \\
&=  \underbrace{W_{left} \left( log(w_{left}) - log(d_{split} - d_1) \right)}_{\text{Left Hand Side (LHS)}} + \underbrace{W_{right} \left( log(w_{right}) - log(d_N - d_{split}) \right)}_{\text{Right Hand Side (RHS)}},
\end{align*}

where $W_{left} = \sum_{i=1}^{k} w_i$ and $W_{right} = \sum_{i=k+1}^{N} w_i$ are the weights on either side of the split.

The most likely split is selected, and the process is repeated recursively on the left and right side of the split.

The likelihood to compare against is given by the weighted log likelihood of a uniform without splitting, i.e.

$$
log \left( L(\mathcal{D}| d_1, d_N) \right) = - W \,log \left(d_N - d_1 \right).
$$

The gain of a split is the difference of the two. With $q = W_{left} / W$ and
$\lambda = (d_{split} - d_1) / (d_N - d_1)$ it equals

$$
gain = W \cdot KL \left( Bernoulli(q) \, || \, Bernoulli(\lambda) \right) \geq 0,
$$

so a split never lowers the likelihood.
A threshold on the likelihood alone would therefore never stop the induction.
Instead, every split has to pay for the two parameters it adds, its boundary and its weight, with the penalty of the
Bayesian information criterion {cite}`schwarz1978estimating`. 
A split is made if and only if

$$
gain \cdot \frac{n}{W} > \alpha \, log(n), \qquad n = \frac{W^2}{\sum_{i=1}^N w_i^2},
$$

where $n$ is the effective sample size of the weights. 
For samples that are weighted by how often they occur, $n = W$ and the criterion reads $gain > \alpha \, log(W)$.
The factor $\alpha$ is referenced to as `split_penalty_factor` in the code, and $\alpha = 1$ is exactly the Bayesian 
information criterion.

The recursive splitting is greedy and therefore not guaranteed to find the partition with the highest penalized
likelihood. 
Since the boundaries are fixed by the indices of the data, the best partition can also be found exactly by dynamic 
programming {cite}`jackson2005algorithm` with the pruning of {cite}`killick2012optimal`. 
It is slower on large datasets, so the induction searches for the best partition only up to
`maximum_unique_values_for_optimal_partition` unique values and splits greedily above that.

This algorithm is implemented in the [NygaInduction](https://cram2.github.io/cognitive_robot_abstract_machine/probabilistic_model/autoapi/probabilistic_model/learning/nyga_induction/index.html#probabilistic_model.learning.nyga_induction.NygaInduction).

# Theory Example

Consider the dataset $\mathcal{D} = \{ 1, 2, 3, 4, 7, 9 \}$ with uniform weights, i.e., $w_i = 1, \forall i$, and
hence $W = 6$.

The log likelihood of a uniform distribution without a split over $\mathcal{D}$ is given by

\begin{align*}
log \left( L(\mathcal{D}| \boldsymbol{\Theta}) \right) &= - 6 \, log(9-1) \\
&\approx -12.48
\end{align*}

We can now calculate every possible split and get the list $splits = [1.5,\, 2.5,\, 3.5,\, 5.5,\, 8]$
Let's calculate the log likelihood for every split.

1. Iteration: $k=1, d_{split} = 1.5$

\begin{align*}
 LHS &= W_{left} \left( log(w_{left}) - log(d_{split} - d_1) \right) \\
 &= 1 \cdot \left( log \left( \frac{1}{6} \right) - log(1.5 - 1) \right) \\
 &\approx -1.1\\
 RHS &= W_{right} \left( log(w_{right}) - log(d_N - d_{split}) \right) \\
 &= 5 \cdot \left( log \left( \frac{5}{6} \right) - log(9 - 1.5) \right) \\
 &\approx -10.99 \\
 log \left( L(\mathcal{D}| 1, 1.5, 9) \right) &\approx -1.0986 - 10.9861  \\
 &\approx -12.08
\end{align*}



2. Iteration  $k=2,  d_{split} = 2.5$

\begin{align*}
 LHS &= 2 \left( log(2) - log(6) - log(1.5) \right)\\
 &\approx -3.01\\
 RHS &= 4  \left( log(4) - log(6) - log(6.5) \right)\\
 &\approx -9.11 \\
 log \left( L(\mathcal{D}| 1, 2.5, 9) \right) &\approx -12.12
\end{align*}

3. Iteration $k=3,  d_{split} = 3.5$

\begin{align*}
 LHS &= 3 \left( log(3) - log(6) - log(2.5) \right)\\
 &\approx -4.83\\
 RHS &= 3  \left( log(3) - log(6) - log(5.5) \right)\\
 &\approx -7.19\\
 log \left( L(\mathcal{D}| 1, 3.5, 9) \right) &\approx -12.02
\end{align*}

4. Iteration $k=4,  d_{split} = 5.5$

\begin{align*}
 LHS &= 4 \left( log(4) - log(6) - log(4.5) \right)\\
 &\approx -7.64\\
 RHS &= 2  \left( log(2) - log(6) - log(3.5) \right)\\
 &\approx -4.7\\
 log \left( L(\mathcal{D}| 1, 5.5, 9) \right) &\approx -12.34
\end{align*}

5. Iteration $k=5,  d_{split} = 8$

\begin{align*}
 LHS &= 5 \left( log(5) - log(6) - log(7) \right)\\
 &\approx -10.64\\
 RHS &= 1  \left( log(1) - log(6) - log(1) \right)\\
 &\approx -1.79\\
 log \left( L(\mathcal{D}| 1, 8, 9) \right) &\approx -12.43
\end{align*}


We can see that the split at 3.5 has the highest likelihood. But is it worth its two parameters?
Its gain is
$$
-12.0220 - (-12.4766) \approx 0.45,
$$
while the penalty with $\alpha = 1$ is $log(6) \approx 1.79$. 
Six samples are too few to justify a split, so the induction keeps a single uniform distribution.

## Practical Example

Finally, if we apply the algorithm to the dataset, we get the following result.
We can see that it looks very similar to the gaussian mixture we sampled from.

```{code-cell} ipython3
from probabilistic_model.learning.nyga_induction import NygaInduction
from random_events.variable import Continuous

import plotly
plotly.offline.init_notebook_mode()
import plotly.graph_objects as go

distribution = NygaInduction(Continuous("x"), minimum_samples_per_quantile=100)
distribution = distribution.fit(dataset)
fig = go.Figure(distribution.plot(), distribution.plotly_layout())
fig.show()
```

Comparing this to the Gaussian distribution we sampled from, we can see that the result is very similar.

```{code-cell} ipython3
from probabilistic_model.distributions.gaussian import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *

mixture = ProbabilisticCircuit()
gaussian_1 = leaf(GaussianDistribution(variable=Continuous("x"), location=0, scale=1), mixture)
gaussian_2 = leaf(GaussianDistribution(variable=Continuous("x"), location=5, scale=0.5), mixture)
s1 = SumUnit(probabilistic_circuit = mixture)
s1.add_subcircuit(gaussian_1, np.log(0.5))
s1.add_subcircuit(gaussian_2, np.log(0.5))
fig.add_traces(mixture.plot())
```

In summary, the Nyga Distribution is a universal, univariate, smooth and deterministic distribution approximation 
without any assumptions on functional form. 
