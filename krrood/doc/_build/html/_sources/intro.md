### Knowledge Representation & Reasoning Through Object-Oriented Design
 
KRROOD is a Python framework that unifies symbolic knowledge representation, expressive querying, 
and rule-based reasoning with clear, object-oriented abstractions.

KRROOD is designed to help you:
- Model your domain as Python dataclasses and objects
- Query entities with a pythonic, relational language (EQL)
- Create and apply rules for inference and reasoning
- Work in-memory or persist to SQL through an ORM interface (ORMatic)

Overall, KRROOD turns your python package into a knowledge base.
The approach draws inspiration from classic knowledge representation and reasoning like Prolog, First Order Logic, SQL 
and Description Logic while emphasizing clean object-oriented design. 
A major inspiration for the development of KRROOD is the book
[Knowledge Representation and Reasoning](https://bista.sites.dmi.unipg.it/didattica/CSP/knowledge-representation-and-reasoning-the-morgan-kaufmann-series-in-artificial-intelligence.pdf).

### Core Components

- Entity Query Language (EQL)
  - A pythonic relational query language for entities and relationships
  - Supports joins, nesting, predicates, rule-based inference, and more
  - Works uniformly against in-memory data, and via ORMatic, SQL backends

- ORMatic
  - Bridges your object model to relational storage
  - Generates and manages mappings so you can write EQL over SQL-backed data
  - Provides an EQL-to-SQL interface to keep queries portable across backends

- Reasoning and Rules
  - Express symbolic rules and rule trees over your domain model
  - Run inference to derive new knowledge from existing facts and relationships

### When to Use KRROOD

- You have a rich domain model and want first-class, object-oriented knowledge representation
- You need expressive, composable queries that feel natural in Python
- You want reasoning capabilities (rules and inference) coupled with your data
- You want to target both in-memory and SQL stores without rewriting queries

### Key Ideas

- Object-Oriented Knowledge: Entities, attributes, and relations are represented as Python objects and classes
- Uniform Querying: The same query patterns apply to in-memory and SQL sources
- Rules as First-Class Citizens: Rules and rule trees integrate directly with queries for inference

### Quick Start

- Install the package:

```bash
pip install krrood
```


- Explore EQL
  - Start with the EQL introduction and examples covering cached symbols, logical connectives, nesting, joins, predicates, indexing, flattening, concatenation, universal quantification, and rule-based inference

- Add Persistence with ORMatic
  - Use ORMatic for automatic or alternative mappings and run the same EQL queries against SQL databases

### Design Philosophy

- Object-Oriented First: Model the domain with classes and relationships
- Expressive but Safe: Queries and interfaces aim to be clear, composable, and hard to misuse
- Single Source of Truth: Keep one conceptual model for both reasoning and persistence

If you are new to KRROOD, start with the EQL introduction, then add ORMatic as soon as you want SQL-backed persistence. 
The examples are designed to be incremental and self-contained.


## To Cite:

```bib
@software{bassiouny2025krrood,
author = {Bassiouny, Abdelrhman AND Schierenbeck, Tom},
title = {Knowledge Representation & Reasoning Through Object-Oriented Design},
url = {https://github.com/code-iai/krrood},
version = {1.0.0},
}
```