# Ripple Down Rules


Welcome to the Ripple Down Rules package!
The ripple_down_rules is a python package that implements the various ripple down rules versions, including 
Single Classification (SCRDR), Multi Classification (MCRDR), and Generalised Ripple Down Rules (GRDR).

SCRDR, MCRDR, and GRDR are rule-based classifiers that are built incrementally, and can be used to classify data cases.
The rules are refined as new data cases are classified, this is done by prompting the user to add a new rule when a case
is misclassified or not classified at all. This allows the system to adapt to new data without the need for retraining.

SCRDR, MCRDR, and GRDR logic were inspired from the book: 
["Ripple Down Rules: An Alternative to Machine Learning"](https://www.taylorfrancis.com/books/mono/10.1201/9781003126157/ripple-rules-paul-compton-byeong-ho-kang) by Paul Compton, Byeong Ho Kang.


## 🚀 Key Features

### 🧠 Data Model (Ontology) + Rule Base as One Entity
- 🧬 Unified data structure: Data Model (Ontology) and rules use the same Python data structures. 
- 🔄 Automatic sync: Updates to the data model instantly reflect in the rule base. 
- 📦 Version controlled: The rule base is a Python module, versioned with your project.

### 🔁 Supports First, Second & Higher-Order Logic
- 🧩 Unlimited expressiveness: Rule conditions and conclusions are plain Python functions — anything Python can do, your rules can too!

### 🛡️ Automatic Rule Base Maintenance 
- ⚠️ Contradiction detection: New rules are auto-checked for logical conflicts. 
- 🔧 Prompted refinements: If a contradiction arises, you're guided to add a refinement rule. 

### 📝 Transparent & Editable Rule Base
- 📖 Readable rules: Rules are clean, understandable Python code. 
- 🔄 Reload-friendly: Easily edit and reload rules manually as needed.

### 💻 Developer-Centric Interface
- 👨‍💻 Feels like home: Seamless integration with your favorite IDE.
- ✨ Modern coding experience: Auto-completion and suggestions supported via IDE plugins.

### 🤖 LLM-Powered Rule Writing
- 💡 AI-assisted authoring: Ask AI for help or suggestions directly within the IDE.
- ⚡ Smart completion: Context-aware completions streamline rule writing.

### 🎯 Flexible Rule Specificity
- 🧪 Instance-level precision: Write rules for highly specific object scenarios.
- 🏛️ Generalization-ready: Create broad rules for superclass relationships.

### 🖼️ GUI for Rule Exploration
- 🧭 Object Explorer Panel: Navigate and inspect objects easily.
- 🧯 Interactive Diagram: Expandable/collapsible object diagram to guide rule creation visually.

This work aims to provide a flexible and powerful rule-based system that can be used in various applications,
from simple classification tasks to complex decision-making systems. Furthermore, one of the main goals is to
provide an easy-to-use interface that allows users to write rules in a natural way, without the need for
complex configurations or setups, and without the need to learn old or deprecated programming languages.

Future (and current) work will focus on improving the user experience, adding more features, and enhancing the
performance, so stay tuned for updates! and feel free to contribute, give feedback, or report issues on the
[GitHub repository](https://github.com/AbdelrhmanBassiouny/ripple_down_rules/issues)

## To Cite:

```bib
@software{bassiouny2025rdr,
author = {Bassiouny, Abdelrhman},
title = {Ripple-Down-Rules},
url = {https://github.com/AbdelrhmanBassiouny/ripple_down_rules},
version = {0.6.47},
}
```

## Installation
```bash
sudo apt-get install graphviz graphviz-dev
pip install ripple_down_rules
```
For GUI support, also install:

```bash
sudo apt-get install libxcb-cursor-dev
```

## Technical Overview

### CaseQuery

The {py:class}`ripple_down_rules.datastructures.dataclasses.CaseQuery` class is a data structure that represents a query for a case in the ripple down rules system.
It mainly requires the queried object which is referred to as `case`, the target attribute that is being queried, the type(s) of the target, and a boolean indicating whether the target is mutually exclusive or not.

### RippleDownRules

{py:class}`ripple_down_rules.rdr.RippleDownRules` This is the main abstract class for the ripple down rules. From this class, the different versions of
ripple down rules are derived. So the {py:class}`ripple_down_rules.rdr.SingleClassRDR`, {py:class}`ripple_down_rules.rdr.MultiClassRDR`, and {py:class}`ripple_down_rules.rdr.GeneralRDR` classes
are all derived from this class.

For most cases, you will use the `GeneralRDR` class, which is the most general version of the ripple down rules, 
and it internally uses the `SingleClassRDR` and `MultiClassRDR` classes to handle the classification tasks depending
on the case query.

This class has four main methods that you will mostly use:

- {py:func}`ripple_down_rules.rdr.RippleDownRules.fit`: This method is used to fit the rules to the data. It takes a list of `CaseQuery` objects that contain the
  data cases and their targets, and it will prompt the user to add rules when a case is misclassified or not classified at all.
- {py:func}`ripple_down_rules.rdr.RippleDownRules.fit_case`: This method is used to fit a single case to the rules.
- {py:func}`ripple_down_rules.rdr.RippleDownRules.classify`: This method is used to classify a data case. It takes a data case (any python object) and returns the
predicted target.
- {py:func}`ripple_down_rules.rdr.RippleDownRules.save`: This method is used to save the rules to a file. It will save the rules as a python module that can be imported
  in your project. In addition, it will save the rules as a JSON file with some metadata about the rules.
- {py:func}`ripple_down_rules.rdr.RippleDownRules.load`: This method is used to load the rules from a file. It will load the rules from a JSON file and then update the
  rules from the python module (which is important in case the user manually edited the rules in the python module).

### SingleClassRDR
This is the single classification version of the ripple down rules. It is used to classify data cases into a single
target class. This is used when your classification is mutually exclusive, meaning that a data case can only belong
to one class. For example, an animal can be either a "cat" or a "dog", but not both at the same time.

### MultiClassRDR
This is the multi classification version of the ripple down rules. It is used to classify data cases into multiple
target classes. This is used when your classification is not mutually exclusive, meaning that a data case can belong
to multiple classes at the same time. For example, an animal's habitat can be both "land" and "water" at the same time.

### GeneralRDR
This is the general version of the ripple down rules. It has the following features:
- It can handle both single and multi classification tasks by creating instances of `SingleClassRDR` and `MultiClassRDR`
  internally depending on the case query.
- It performs multiple passes over the rules and uses conclusions from previous passes to refine the classification,
    which allows it to handle complex classification tasks.

## Expert

The {py:class}`ripple_down_rules.experts.Expert` is an interface between the ripple down rules and the rule writer. Currently, only a {py:class}`ripple_down_rules.experts.Human` expert is
implemented, but it is designed to be easily extendable to other types of experts, such as LLMs or other AI systems.

The main APIs are:

- {py:func}`ripple_down_rules.experts.Expert.ask_for_conclusion`: This method is used to ask the expert for a conclusion or a target for a data case.
- {py:func}`ripple_down_rules.experts.Expert.ask_for_conditions`: This method is used to ask the expert for the conditions that should be met for a rule to be
applied or evaluated.

## RDRDecorator

The {py:class}`ripple_down_rules.rdr_decorators.RDRDecorator` is a decorator that can be used to create rules in a more convenient way. It allows you to write 
functions normally and then decorate them with the `@RDRDecorator().decorator`. This will allow the function to be
to use ripple down rules to provide its output. This also allows you to write your own initial function logic, and then
this will be used input or feature to the ripple down rules.


## Example Usage

- [Relational Example](relational_example_tutorial.md): This example shows how to use the Ripple Down Rules to classify objects in a relational model.
- [Relational Example with Decorator](relational_example_with_decorator_tutorial.md): This example shows how to use the Ripple Down Rules with a decorator to overload methods.
- [Propositional Example](propositional_example_tutorial.md): This example shows generic usage of the Ripple Down Rules to classify objects in a propositional setting.
