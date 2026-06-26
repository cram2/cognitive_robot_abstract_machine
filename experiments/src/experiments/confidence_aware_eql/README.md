# Confidence-Aware Evaluation Engine

This package provides a domain-agnostic framework for learning a tractable probabilistic model from object features, setting a data-driven threshold, and flagging anomalous or incomplete inputs.

## Layout

```text
confidence_aware_eql/
├── domains/                # domain-specific plugin modules
├── engine/                 # reusable modeling and evaluation logic
├── tests/                  # example and regression tests
├── registry.py             # discovers domain plugins
├── run.py                  # command-line entry point
└── conftest.py             # pytest import setup
```

## Installation

Install the package dependencies in the active environment:

```bash
pip install -e experiments -e random_events -e probabilistic_model
pip install scikit-learn pytest pytest-xdist pytest-cov pytest-asyncio
```

## Running the demo

From the repository root:

```bash
python3 experiments/src/experiments/confidence_aware_eql/run.py --list
python3 experiments/src/experiments/confidence_aware_eql/run.py --domain kitchen
python3 experiments/src/experiments/confidence_aware_eql/run.py --all
```

## Running tests

```bash
PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 python3 -m pytest -p xdist -o addopts='-sv' experiments/src/experiments/confidence_aware_eql/tests/test_kitchen.py
```

## Adding a new domain

1. Create a new module in the domains directory.
2. Define `DOMAIN` and `SPEC` in that module.
3. Add example objects in `FAMILIAR`, `ANOMALOUS`, or `INCOMPLETE` if needed.
4. Run the command-line entry point again to use the new domain.

A domain module must expose `DOMAIN` and `SPEC`.

## How the engine works

1. A domain defines ordered features.
2. Synthetic or real data is passed to `CircuitModel.fit`.
3. The fitted model is compiled into a tractable probabilistic circuit.
4. A threshold strategy is used to decide whether a sample is unusual.
5. The evaluator scores new objects and reports warnings for incomplete or low-confidence inputs.

## Notes

- Categorical features are currently numeric-encoded in the model.
- The current examples use synthetic data generated from the package.
- The implementation supports full-evidence scoring and can be extended for more specific checks.
```
