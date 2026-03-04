### Propositional Example Tutorial

By propositional, I mean that each rule conclusion is a propositional logic statement with a constant value.

For this example, we will use the [UCI Zoo dataset](https://archive.ics.uci.edu/ml/datasets/zoo) to classify animals
into their species based on their features. The dataset contains 101 animals with 16 features, and the target is th
e species of the animal.

To install the dataset:
```bash
pip install ucimlrepo
```

### Prepare the Data

We first import the dataset and prepare the data for the Ripple Down Rules classifier.
We also define a utility function to convert the target values to `Species` enum values.

```python
from __future__ import annotations
from krrood.ripple_down_rules.datastructures.case import create_cases_from_dataframe
from ucimlrepo import fetch_ucirepo
from enum import Enum

class Species(str, Enum):
    """Enum for the species of the animals in the UCI Zoo dataset."""
    mammal = "mammal"
    bird = "bird"
    reptile = "reptile"
    fish = "fish"
    amphibian = "amphibian"
    insect = "insect"
    molusc = "molusc"
    
    @classmethod
    def from_str(cls, value: str) -> Species:
        return getattr(cls, value)

# fetch dataset
zoo = fetch_ucirepo(id=111)

# data (as pandas dataframes)
X = zoo.data.features
y = zoo.data.targets

# This is a utility that allows each row to be a Case instance,
# which simplifies access to column values using dot notation.
all_cases = create_cases_from_dataframe(X, name="Animal")

# The targets are the species of the animals
category_names = ["mammal", "bird", "reptile", "fish", "amphibian", "insect", "molusc"]
category_id_to_name = {i + 1: name for i, name in enumerate(category_names)}
targets = [Species.from_str(category_id_to_name[i]) for i in y.values.flatten()]
```

### Define the Case Queries
Create a new python script to define the case queries and the Ripple Down Rules classifier.
For every target, we create a `CaseQuery` that specifies the case, the target attribute, the type of the target,
and whether the classification is mutually exclusive or not. In this case, we set `mutually_exclusive` to `True` since
each animal belongs to only one species.
```python
from krrood.ripple_down_rules import CaseQuery

case_queries = [CaseQuery(case, 'species', type(target), True, _target=target)
                for case, target in zip(all_cases[:10], targets[:10])]
```

### Create and Use the Ripple Down Rules Classifier

```python
# Optionally Enable GUI if available
from krrood.ripple_down_rules.helpers import enable_gui
enable_gui()


from krrood.ripple_down_rules import GeneralRDR
from krrood.ripple_down_rules.utils import render_tree


# Now that we are done with the data preparation, we can create and use the Ripple Down Rules classifier.
grdr = GeneralRDR(save_dir="./", model_name="species_rdr")

# Fit the GRDR to the data
grdr.fit(case_queries, animate_tree=True)

# Render the tree to a file
render_tree(grdr.start_rules[0].node, use_dot_exporter=True, filename="species_rdr")

# Classify a case
cat = grdr.classify(all_cases[50])
assert cat['species'] == targets[50]
```

When prompted to write conditions for the given target, I press the "Edit" buttond (%edit in the Ipython interface) and I wrote the following
inside the template function that the Ripple Down Rules created:
```python
return case.milk == 1
```
Then, I press the "Load" button (%load in Ipython), this loads the function I just wrote such that I can test it inside
the Ipython interface.
After that, I press the "Accept" button (return func_name(case) in Ipython), this will save the rule permanently.

When prompted for conditions for the next target, I wrote the following inside the template function that the
Ripple Down Rules created:
```python
return case.aquatic == 1
```

I keep doing this for all the prompts until I have fitted the rules such that the classifier correctly classifies the
animals in the dataset. This took around 10 prompts in total, and the rules correctly classified 101 animals in the dataset.

The rule tree generated from fitting all the dataset will look like this:
![species_rdr](https://raw.githubusercontent.com/AbdelrhmanBassiouny/ripple_down_rules/main/images/scrdr.png)