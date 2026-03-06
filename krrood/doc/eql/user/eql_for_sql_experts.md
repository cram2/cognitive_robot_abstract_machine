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

# EQL for SQL Experts

If you are coming from a relational database background, EQL will feel familiar but with several powerful abstractions
that simplify complex queries. This guide compares EQL concepts with their SQL equivalents.

## Implicit JOINs vs. Path Following

The biggest difference between SQL and EQL is how relationships are handled. In SQL, you must explicitly JOIN tables
using foreign keys. In EQL, you simply follow the attribute path.

### SQL (Explicit JOIN)

```sql
SELECT p.name
FROM robots r
         JOIN parts p ON r.id = p.robot_id
WHERE r.type = 'Astromech'
  AND p.status = 'Broken';
```

### EQL (Implicit Path Following)

```python
# EQL automatically traverses the relationship from robot to parts
query = entity(r.parts.name).where(
    r.type == "Astromech",
    r.parts.status == "Broken"
)
```

## Filter Early, Filter Late

In SQL, all filtering happens in the `WHERE` or `HAVING` clause. EQL also maintains this distinction.

| SQL Clause | EQL Method      | Purpose                                        |
|:-----------|:----------------|:-----------------------------------------------|
| `WHERE`    | `.where()`      | Filters individual entities *before* grouping. |
| `GROUP BY` | `.grouped_by()` | Defines the grouping keys.                     |
| `HAVING`   | `.having()`     | Filters aggregated groups *after* calculation. |
| `ORDER BY` | `.ordered_by()` | Sorts the final result set.                    |
| `LIMIT`    | `.limit()`      | Restricts the number of returned rows.         |

## Result Cardinality

SQL queries always return a result set (even if empty). EQL uses quantifiers to express expectations about the result
set size, which helps catch data integrity issues early.

- **SQL**: Always returns 0..N rows.
- **EQL `an()`**: Expects 0..N results.
- **EQL `the()`**: Expects exactly 1 result (like `SELECT ... LIMIT 1` but with an assertion).

```{note}
EQL's `the()` is a great way to enforce domain cardinality logic (e.g., "every robot must have exactly
one serial number") directly in your query.
```

## Set Operations

| SQL      | EQL                                                                      |
|:---------|:-------------------------------------------------------------------------|
| `UNION`  | {py:class}`~krrood.entity_query_language.operators.set_operations.Union` |
| `EXISTS` | {py:func}`~krrood.entity_query_language.factories.exists`                |
| `IN`     | {py:func}`~krrood.entity_query_language.factories.in_`                   |

## Summary Example: A Complex Report

SQL query:

```sql
SQL version:
SELECT type, COUNT(*), AVG(battery)
FROM robots
WHERE online = true
GROUP BY type
HAVING COUNT(*) > 2
ORDER BY AVG(battery) DESC
LIMIT 5
```

EQL query:

```python 
r = variable(ExampleRobot, domain=all_robots)

query = set_of(r.type, count(r), average(r.battery)) \
    .where(r.online) \
    .grouped_by(r.type) \
    .having(count(r) > 2) \
    .ordered_by(average(r.battery), descending=True) \
    .limit(5)
```

```{warning}
While EQL is powerful, remember that it operates on objects in memory. For extremely large
datasets
typically found in data warehouses, SQL is still the preferred tool. EQL is optimized for complex symbolic reasoning
over structured object models.
If your case includes calling functions/predicates as part of the query, or if you prefer to work on live domain objects
then EQL is the way to go.
```
