# Welcome to probabilistic_models's documentation!

The `probabilistic_models` package contains fast and flexible implementations for various probabilistic models.
This package provides a clean, unifying and well documented API to probabilistic models. 
Just like sklearn does for classical machine learning models.

Install the package via pip:

```bash 
pip install probabilistic_model
```

## Supported Models

- Continuous Distributions
  - Gaussian Distribution
  - Uniform Distribution
- Discrete Distributions
  - Categorical Distribution
  - Integer Distribution
- Bayesian Networks
- Probabilistic Circuits / Sum Product Networks
  - Random and Tensorized SPNs
  - Nyga Distributions
  - Joint Probability Trees
  - Conditional SPNs

## Supported Inferences
- Likelihoods
- Sampling
- Marginal Probabilities
- Marginal Distributions
- Conditional Distributions
- Modes
- Moments
- $L_1$ distances


## Citing probabilistic_model
If you use this software for publications, please cite it as below.

```bibtex
@software{schierenbeck2024pm,
author = {Schierenbeck, Tom},
title = {probabilistic_model: A Python package for probabilistic models},
url = {https://github.com/tomsch420/probabilistic_model},
version = {7.1.0},
}
```
