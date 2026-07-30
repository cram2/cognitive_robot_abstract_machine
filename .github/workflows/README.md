# Workflow Local Testing Guide

This README describes how to test GitHub Actions workflows locally using [act](https://github.com/nektos/act) and [actionlint](https://github.com/rhysd/actionlint). 

## Tools Setup

### 1. act (GitHub Actions Local Runner)
Installs: 
- Docker [install guide](https://docs.docker.com/get-docker/)
- `act` tool: run `curl --proto '=https' --tlsv1.2 -sSf https://raw.githubusercontent.com/nektos/act/master/install.sh | sudo bash`

Usage Example:
```bash
# Run coraplex workflow locally
act -W .github/workflows/coraplex-publish-to-pypi.yml
```

or run a single job:

```bash
act -W .github/workflows/krrood-publish-to-pypi.yml -j "check-version"
```

### 2. actionlint (Workflow Linter)
Installs:
- Go: go to [go website](https://go.dev/doc/install), download and install go
- `actionlint` tool: `go install github.com/rhysd/actionlint/cmd/actionlint@latest`, restart terminal session afterwards

Usage Example:
```bash
# Lint all workflow YAML files
actionlint .github/workflows/*.yml
```

## Workflow Testing Workflow
1. **Linting:** Run `actionlint` to ensure there are no syntax or logical errors in the YAML files.
2. **Local Execution:** Use `act` to simulate the workflow execution and verify that scripts (like version checks) behave as expected in a containerized environment.

