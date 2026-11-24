# Glossary

```{glossary}
PGM
  [From Wikipedia](https://en.wikipedia.org/wiki/Graphical_model): 
  A probabilistic graphical model (PGM) or structured probabilistic model is a probabilistic model for which a graph 
  expresses the truncated dependence structure between random variables. 
  They are commonly used in probability theory, statistics—particularly Bayesian statistics—and machine learning. 

MAG
  A maximal ancestral graph: a graph over the recorded variables of a directed acyclic
  graph with hidden variables. Its edges are directed or bidirected, and an edge shows a
  tail at a node that causes the other end and an arrow at one that does not.

PAG
  A partial ancestral graph: the maximal ancestral graphs the data cannot tell apart,
  drawn as one graph. A circle marks an edge end on which they disagree.

Visible edge
  A directed edge $X \to W$ of an ancestral graph for which some variable not adjacent
  to $W$ has an edge into $X$, or a path into $X$ whose inner nodes are colliders and
  parents of $W$. Only a visible edge rules out a hidden variable causing both ends.

Amenable
  An ancestral graph is amenable for the effect of $X$ on $Y$ when every possibly
  directed path from $X$ to $Y$ starts with a visible edge out of $X$. Only then can
  an adjustment set identify the effect.

Forbidden set
  The nodes other than $X$ on a possibly directed path from $X$ to $Y$, with their
  possible descendants. No valid adjustment set holds one of them.


```