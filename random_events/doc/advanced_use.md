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


# Advanced Use-case

In this tutorial, we will look at a humorous application of random events. 
This examples shows that elements from the product algebra can take almost any shape, such as a tomato. 
First, we import the necessary packages and define two variables.

```{code-cell} ipython3
:tags: []

import os.path
from random_events.product_algebra import Event, SimpleEvent
from random_events.variable import Continuous
from random_events.interval import *
from PIL import Image
import numpy as np
import plotly.graph_objects as go

x = Continuous("x")
y = Continuous("y")
```

Next, let's load the logo of this package.

```{code-cell} ipython3
:tags: []
path = os.path.join("Tomato.png")
image = im=Image.open(path)
image
```

We can express this image as an event that can be reasoned about.

```{code-cell} ipython3
:tags: []
image = np.array(image.resize((18, 17), Image.NEAREST))
colors = np.unique(image.reshape((image.shape[0] * image.shape[1], image.shape[2])), axis=0)[1:]
def indices_to_complex_event(indices: np.array) -> Event:
    result = []
    for index in indices:
        event = SimpleEvent({y: closed_open(-index[0] - 1., -index[0]),
                       x: closed_open(index[1], index[1] + 1.)})
        result.append(event)
    return Event(*result).make_disjoint()

fig = go.Figure()

complex_events = []

for color in colors:
    pixel_indices = np.transpose(np.nonzero(np.all(image == color, axis=-1)))
    complex_event = indices_to_complex_event(pixel_indices)
    complex_events.append(complex_event)
    traces = complex_event.plot(f"rgb({color[0]},{color[1]},{color[2]})")
    fig.update_layout(complex_event.plotly_layout())
    fig.add_traces(traces)

fig.update_layout(title="Random Events Tomato")
fig.show()
```

While the shape of a tomato as an event that can be used for probabilistic reasoning serves no particular interest, 
it showcases that random events can take approximately any shape and not "just" rectangles.
The entire tomate as measurable space is obtained by union of all the events.
    
```{code-cell} ipython3
:tags: []

entire_event = complex_events[0] | complex_events[1] | complex_events[2]
fig = go.Figure(entire_event.plot(), entire_event.plotly_layout())
fig.update_layout(title="Random Events Tomato as one Event")
fig.show()
```

I hope this bizarre examples aids you in understanding of the product algebra capabilities. 
Using this event, you can calculate things like the probability of a tomato or the conditional distribution given a tomato.

