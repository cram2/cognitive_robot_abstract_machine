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
The weighted log likelihood is derived from the weighted likelihood. 
The weighted likelihood of the data under the model is given by

$$
L(\mathcal{D}| \boldsymbol{\Theta}) = \prod_{i=1}^N  w_i \, p(d_i),
$$

where $\boldsymbol{\Theta}$ represents the parameters of the model.

By applying the logarithm to the weighted likelihood, we get the log likelihood

$$
log \left( L(\mathcal{D}| \boldsymbol{\Theta}) \right) =  \sum_{i=1}^N log(w_i) + log(p(d_i)).
$$


When the uniform distribution is assumed for the data, the weighted log likelihood is given by

\begin{align*}
log \left(L(\mathcal{D}| \boldsymbol{\Theta}) \right) &= \sum_{i=1}^N log(w_i) + log \left(\frac{1}{b-a} \right)\\
&= \sum_{i=1}^N log(w_i) + \sum_{i=1}^N - log \left(b-a \right).
\end{align*}

Since the density of the uniform distribution is constant, the weighted log likelihood can be restated as

$$ \sum_{i=1}^N log(w_i) -N log \left(b-a \right),$$
 
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
log \left( L(\mathcal{D}| d_1, d_{split}, d_N) \right) &= \sum_{i=1}^N log(w_i) + log(p(x_i))\\
&=  \sum_{i=1}^{left} log(w_i) + log(p(x_i)) + \sum_{i=right}^{N} log(w_i) + log(p(x_i)) \\
&=  \sum_{i=1}^{left} log(w_i) + log(w_{left} \cdot p_{left}(x_i)) + \sum_{i=right}^{N} log(w_i) + log(w_{right} \cdot p_{right}(x_i)) \\
&=  \underbrace{\sum_{i=1}^{left} log(w_i) + log(w_{left}) + log(U_{left})}_{\text{Left Hand Side (LHS)}} + \underbrace{\sum_{i=right}^{N} log(w_i) + log(w_{right}) + log(U_{right})}_{\text{Right Hand Side (RHS)}} \\
LHS &=  L \cdot ( log(w_{left}) + log(U_{left})) + \sum_{i=1}^{left} log(w_i)\\
&= L \cdot \left( log \left( \sum_{i=1}^{left} w_i \right) - log \left( \sum_{i=1}^{N} w_i \right) - log(D_{split} - D_{left}) \right) + \sum_{i=1}^{left} log(w_i)\\
RHS &= (N - R) \cdot \left( log \left( \sum_{i=right}^{N} w_i \right) - log \left( \sum_{i=1}^{N} w_i \right) - log(D_{right} - D_{split}) \right) + \sum_{i=right}^{N} log(w_i)\\
\end{align*}


The most likely split is selected, and the process is repeated recursively on the left and right side of the split.

If the likelihood improvement of the best split, compared to no splitting, is smaller than a given threshold, the process is terminated. 
The likelihood to compare against is given by the weighted log likelihood of a uniform without splitting, i.e.

$$
log \left( L(\mathcal{D}|, d_1, d_N) \right) = \sum_{i=1}^N log(w_i) - N \,log \left(d_N - d_1 \right).
$$

The best threshold is given by selecting the maximum weighted likelihood improvement over all possible splits.
The improvement value is given by

$$
max \left( log \left( L(\mathcal{D}| d_1, d_{split}, d_N) \right) \right) > log(\epsilon) + log \left( L(\mathcal{D}|, d_1, d_N) \right)
$$


In simpler terms, the induction is terminated as soon as the likelihood does not improve by more than $\epsilon\,\%$ 
anymore if a split is made. 

This parameter is referenced to as `min_likelihood_improvement` in the code.
This algorithm is implemented in the [NygaDistribution](https://probabilistic-model.readthedocs.io/en/latest/autoapi/probabilistic_model/learning/nyga_distribution/index.html#probabilistic_model.learning.nyga_distribution.NygaDistribution).

# Theory Example

Consider the dataset $\mathcal{D} = \{ 1, 2, 3, 4, 7, 9 \}$ with uniform log_weights, i.e., $w_i = 1, \forall i$.

The log likelihood of a uniform distribution without a split over $\mathcal{D}$ is given by

\begin{align*}
log \left( L(\mathcal{D}| \boldsymbol{\Theta}) \right) &= \sum_{i=1}^6 log(1) - log(9-1) \\
&\approx -12.48
\end{align*}

We can now calculate every possible split and get the list $splits = [1.5,\, 2.5,\, 3.5,\, 5.5,\, 8]$
Let's calculate the log likelihood for every split.

1. Iteration: $k=1, D_{split} = 1.5$

\begin{align*}
 log \left( L(\mathcal{D}| 1, 1.5, 9) \right) &= LHS + RHS \\
 LHS &= L \cdot \left( log \left( \sum_{i=1}^{left} w_i \right) - log \left( \sum_{i=1}^{N} w_i \right) - log(D_{split} - D_{left}) \right) + \sum_{i=1}^{left} log(w_i) \\
 &= 1 \cdot \left( log \left( \sum_{i=1}^{1} 1 \right) - log \left( \sum_{i=1}^{6} 1 \right) - log(1.5 - 1) \right) + \sum_{i=1}^{1} log(1) \\
 &=  log (1) - log(6) - log(0.5) + log(1) \\
 &\approx -1.1\\
 RHS &= (N - R) \cdot \left( log \left( \sum_{i=right}^{N} w_i \right) - log \left( \sum_{i=1}^{N} w_i \right) - log(D_{right} - D_{split}) \right) + \sum_{i=right}^{N} log(w_i)\\
 &= (6 - 1) \cdot \left( log \left( \sum_{i=2}^{6} 1 \right) - log \left( \sum_{i=1}^{6} 1 \right) - log(9 - 1.5) \right) + \sum_{i=2}^{6} log(1)\\
 &= 5 \cdot \left( log(5) - log(6) - log(7.5) \right) \\
 &\approx -10.99 \\
 log \left( L(\mathcal{D}| 1, 1.5, 9) \right) &= -1.1 -10.99  \\
 &\approx -12.09
\end{align*}



2. Iteration  $k=2,  D_{split} = 2.5$

\begin{align*}
 LHS &= 2 \left( log(2) - log(6) - log(1.5) \right)\\
 &\approx -3.01\\
 RHS &= 4  \left( log(4) - log(6) - log(6.5) \right)\\
 &\approx -9.11 \\
 log \left( L(\mathcal{D}| 1, 2.5, 9) \right) &\approx -12.12
\end{align*}

3. Iteration $k=3,  D_{split} = 3.5$

\begin{align*}
 LHS &= 3 \left( log(3) - log(6) - log(2.5) \right)\\
 &\approx -4.83\\
 RHS &= 3  \left( log(3) - log(6) - log(5.5) \right)\\
 &\approx -7.19\\
 log \left( L(\mathcal{D}| 1, 3.5, 9) \right) &\approx -12.02
\end{align*}

4. Iteration $k=4,  D_{split} = 5.5$

\begin{align*}
 LHS &= 4 \left( log(4) - log(6) - log(4.5) \right)\\
 &\approx -7.64\\
 RHS &= 2  \left( log(2) - log(6) - log(3.5) \right)\\
 &\approx -4.7\\
 log \left( L(\mathcal{D}| 1, 5.5, 9) \right) &\approx -12.34
\end{align*}

5. Iteration $k=5,  D_{split} = 8$

\begin{align*}
 LHS &= 5 \left( log(5) - log(6) - log(7) \right)\\
 &\approx -10.64\\
 RHS &= 1  \left( log(1) - log(6) - log(1) \right)\\
 &\approx -1.79\\
 log \left( L(\mathcal{D}| 1, 8, 9) \right) &\approx -12.43
\end{align*}


We can see that the split at 3.5 has the highest likelihood. But how much better is it? 
The relative improvement is given by
$$
\frac{exp(-12.02) }{exp(-12.48)} = 1.58,
$$
which means that the resulting likelihood is 58% higher than the likelihood without a split.

## Practical Example

Finally, if we apply the algorithm to the dataset, we get the following result.
We can see that it looks very similar to the gaussian mixture we sampled from.

```{code-cell} ipython3
from probabilistic_model.learning.nyga_distribution import NygaDistribution
from random_events.variable import Continuous

import plotly
plotly.offline.init_notebook_mode()
import plotly.graph_objects as go

distribution = NygaDistribution(Continuous("x"), min_samples_per_quantile=100, min_likelihood_improvement=0.01)
distribution = distribution.fit(dataset)
fig = go.Figure(distribution.plot(), distribution.plotly_layout())
fig.show()
```

Comparing this to the Gaussian distribution we sampled from, we can see that the result is very similar.

```{code-cell} ipython3
from probabilistic_model.distributions import GaussianDistribution
from probabilistic_model.probabilistic_circuit.rx.probabilistic_circuit import *

mixture = ProbabilisticCircuit()
gaussian_1 = leaf(GaussianDistribution(Continuous("x"), 0, 1), mixture)
gaussian_2 = leaf(GaussianDistribution(Continuous("x"), 5, 0.5), mixture)
s1 = SumUnit(probabilistic_circuit = mixture)
s1.add_subcircuit(gaussian_1, np.log(0.5))
s1.add_subcircuit(gaussian_2, np.log(0.5))
fig.add_traces(mixture.plot())
```

In summary, the Nyga Distribution is a universal, univariate, smooth and deterministic distribution approximation 
without any assumptions on functional form. 
