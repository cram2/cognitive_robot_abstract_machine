# Welcome to the Probabilistic Model's package. 
[![Python Tests](https://github.com/tomsch420/probabilistic_model/actions/workflows/python-tests.yml/badge.svg)](https://github.com/tomsch420/probabilistic_model/actions/workflows/python-tests.yml)

This package contains an interface for any kind of probabilistic models.
The aim of this package is to provide a clean, unifying, well documented API to
probabilistic models. Just like sklearn does for classical machine learning models.

Read the docs here https://probabilistic-model.readthedocs.io/en/latest/.

## Development

### Testing

The project uses unittest for testing. To run the tests locally:

```bash
pip install -r requirements-dev.txt
pip install -e .
python -m unittest discover test
```

### Continuous Integration

This project uses GitHub Actions for continuous integration. The CI pipeline runs automatically on push to main/master branches and on pull requests to these branches. It sets up a Python environment, installs dependencies, and runs the tests.

You can see the status of the CI pipeline in the GitHub Actions tab of the repository.
