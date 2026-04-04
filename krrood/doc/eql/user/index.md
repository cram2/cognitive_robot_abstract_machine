---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.16.4
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# User Guide
For most users, we recommend following the documentation in this order:
1.  **[Writing Basic Queries](writing_queries.md)**: Learn the fundamentals of EQL, including defining variables, selecting entities, and applying basic filters.
2.  **[Mapped Variables and Attribute Access](mapped_variable.md)**: Discover how to traverse object relationships, access attributes symbolically, and handle nested data collections.
3.  **[Comparators](comparators.md)** and **[Logical Operators](logical_operators.md)**: Detailed references for the symbolic comparison and logical operators used to build query conditions.
4.  **[Aggregators](aggregators.md)**: Learn how to perform calculations like `count` and `average` over your result sets.
5.  **[Result Quantifiers](result_quantifiers.md)** and **[Result Processors](result_processors.md)**: Tools for controlling the expected cardinality (e.g., `an`, `the`) and the final presentation (sorting, limiting, grouping) of your results.
6.  **[Predicates and Symbolic Functions](predicate_and_symbolic_function.md)**: Instructions on how to integrate custom Python logic and predicates directly into your symbolic queries.
7.  **[Pattern Matching](match.md)**: Simpler pattern matching against structured data.
8.  **[EQL for SQL Experts](eql_for_sql_experts.md)**: A comparative guide for those with a relational database background, mapping EQL concepts to SQL equivalents.
9.  **[Writing Rule Trees](writing_rule_trees.md)**: An advanced topic for building complex reasoning systems with EQL.


## Developer Guide
For developers, you can read the sections mentioned in **[Developer Guide](../developer/index.md)**