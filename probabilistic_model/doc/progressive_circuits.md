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

# Progressive Probabilistic Circuits

A progressive probabilistic circuit (PPC) learns tasks one after another without
forgetting the earlier ones. Following progressive neural networks
{cite}`rusu2016progressive`, every task gets a circuit of its own, called a column. A new
column may reuse the earlier columns, which stay unchanged.

## Structure

Every column is a copy of the *template*, a probabilistic circuit. Adding a column for a
new task

1. copies the template into the PPC,
2. makes the sum unit at the same position in every earlier column an additional child
   of each sum unit of the new column, and
3. adds the new column below the root of the PPC, a sum unit that mixes all columns.
   The new column gets no weight there until it is learned.

Edges only point from newer columns to older ones, so an earlier column never depends on
a later one. Aligned sum units model the same variables, so the PPC stays smooth and
decomposable. It is not deterministic: a sum unit's own children and the aligned units of
earlier columns can model the same rows.

Columns align because they are copies of the same template. If the template is changed
after columns were added, the new column no longer matches the earlier ones, and adding
it raises a {py:class}`~probabilistic_model.learning.progressive.exceptions.ColumnStructureMismatchError`.

## Learning

{py:class}`~probabilistic_model.learning.progressive.expectation_maximization.ProgressiveExpectationMaximization`
learns one column at a time with expectation maximization:

- While a column is learned, the root gives it the whole weight, so it is trained as the
  only model of its task. It still reaches the earlier columns through its edges to them.
- Only the weights and leaves of the learned column are updated. The other columns keep
  their parameters, which prevents forgetting.
- Afterwards the root weights every column by its share of all rows the columns were
  learned from.

Gaussian, discrete and symbolic leaves can be learned; any other leaf raises an
{py:class}`~probabilistic_model.learning.progressive.exceptions.UnsupportedLeafDistributionError`.

```{warning}
Later columns read from earlier ones, so learning an earlier column again also changes
the later columns.
```
