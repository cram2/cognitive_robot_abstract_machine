# Mutagenesis Causal Query: Results

This experiment uses a relational causal circuit on the CTU Mutagenesis dataset. We
take the 188-molecule dataset, mark branching-atom count as a candidate cause of
mutagenicity right in the query, and ask what backdoor adjustment says once we
control for the dataset's known `ind1` indicator. It follows on from fitting and
validating the relational circuit on the full dataset (covered separately), and it
runs on the whole dataset without subsampling.

The cause and effect are declared where the rest of the query is built, not
registered afterward as a separate step:

```python
query = a(MutagenesisMolecule)(
    indicator_1=confounder,
    logp=..., lumo=..., mutagenic=...,
    branching_atom_count=cause,
    atoms=[a(MutagenesisAtom)(element=..., atom_type=..., charge=..., bond_count=...) for _ in range(2)],
    bonds=[a(MutagenesisBond)(bond_type=...) for _ in range(1)],
)
query.causes_effect(query.variable.mutagenic == True)
causal_circuit = RelationalCircuitRegistry(relational_probabilistic_circuit=model).get_model(
    UnderspecifiedParameters(query)
)
```

`RelationalCircuitRegistry` reads the `cause`/`confounder`/`causes_effect` markers,
grounds the query, and returns a verified `CausalCircuit` in one call.

## What we did

1. Fit a relational circuit on all 188 molecules, with the class circuit stratified by
   branching-atom count so it is support-deterministic over that variable.
2. Built a query for a two-atom molecule with every atom's element, atom type,
   charge and bond count left unspecified, marking branching-atom count as the cause,
   `ind1` as a confounder to adjust for, and mutagenicity as the effect.
3. Resolved that query through `RelationalCircuitRegistry`, which grounds it, trims
   the circuit to just those three variables, registers the cause and effect, and
   verifies the result is support-deterministic.
4. Ran backdoor adjustment and compared it against naive conditioning at every
   branching-atom-count value the grounded circuit's support covers.

Code: `causal_query.py` (`BranchingAtomCountCausalQuery.run`), `dataset.py`,
`domain.py` (`MutagenesisAtom.bond_count`,
`MutagenesisMoleculeAggregations.branching_atom_count`).
Tests: `test/causal_reasoning_test/test_causal_query.py`.

## What a branching atom is

The CTU dataset's `bonds` table records which two atoms each bond connects
(`atom1_id`, `atom2_id`). Each atom's bond count, the number of bonds it
participates in, is derived from that table when the molecule is loaded. An atom
with three or four bonds sits at a ring-fusion or branch point in the molecular
graph, rather than along a plain chain (one bond) or a simple ring position (two
bonds). Branching-atom count is the number of such atoms in a molecule, so
registering it as the cause here uses the dataset's own atom-to-atom connectivity
directly, not just a per-atom or per-bond tally.

## The data

Every one of the 188 molecules has at least one branching atom, and the count spans
a wide range:

| Branching atoms | Molecules | Share |
|---:|---:|---:|
| 7 | 7 | 3.7% |
| 8 | 12 | 6.4% |
| 9 | 12 | 6.4% |
| 10 | 13 | 6.9% |
| 11 | 7 | 3.7% |
| 12 | 5 | 2.7% |
| 13 | 16 | 8.5% |
| 14 | 21 | 11.2% |
| 15 | 20 | 10.6% |
| 16 | 12 | 6.4% |
| 17 | 19 | 10.1% |
| 18 | 14 | 7.4% |
| 19 | 7 | 3.7% |
| 20 | 1 | 0.5% |
| 21 | 15 | 8.0% |
| 22 | 4 | 2.1% |
| 24 | 2 | 1.1% |
| 25 | 1 | 0.5% |

## Why branching-atom count?

Branching-atom count is present in every molecule and ranges from 7 to 25 across the
dataset, so grounding retains a variable with real spread rather than one that is
mostly a single dominant value. It also splits clearly by mutagenicity: molecules
that go on to test mutagenic average 16.4 branching atoms, non-mutagenic ones
average 10.6. Ring density and structural complexity, which branching-atom count is
a direct proxy for, are known correlates of genotoxicity in the mutagenesis QSAR
literature. The results below back that up: naive P(mutagenic) climbs from 0 at the
lowest branching-atom counts to 1 at the highest.

## Why not ground this analytically instead?

`GroundingMode.EXACT` grounds by enumerating the fitted circuit's own exact partition
over the cause variable instead of Monte Carlo sampling, but only when that partition
is already disjoint on its own; otherwise it logs a warning and falls back to
`GroundingMode.SAMPLED`. That precondition does not hold for branching-atom count on
this dataset: it is not, today, a split the induced tree makes on its own. `EXACT`
grounding is not reachable yet for this experiment. `SAMPLED` grounding is not a
fallback chosen over a working alternative; it is the mode that runs to completion
here.

## Results

`P(mutagenic = True)` at each branching-atom-count value, naive versus
backdoor-adjusted for `ind1`, on the full dataset:

| Branching atoms | Region P(branching atoms) | Naive P(mutagenic) | Adjusted P(mutagenic) |
|---:|---:|---:|---:|
| 7 | 0.0372 | 0.0000 | 0.0000 |
| 8 | 0.0638 | 0.3333 | 0.3333 |
| 9 | 0.0638 | 0.1667 | 0.1667 |
| 10 | 0.0691 | 0.1538 | 0.1538 |
| 11 | 0.0372 | 0.1429 | 0.1429 |
| 12 | 0.0266 | 0.4000 | 0.4000 |
| 13 | 0.0851 | 0.6875 | 0.6370 |
| 14 | 0.1117 | 0.6190 | 0.6575 |
| 15 | 0.1064 | 0.8500 | 0.8305 |
| 16 | 0.0638 | 0.8333 | 0.8493 |
| 17 | 0.1011 | 1.0000 | 1.0000 |
| 18 | 0.0745 | 1.0000 | 1.0000 |
| 19 | 0.0372 | 1.0000 | 1.0000 |
| 20 | 0.0053 | 1.0000 | 1.0000 |
| 21 | 0.0798 | 1.0000 | 1.0000 |
| 22 | 0.0213 | 1.0000 | 1.0000 |
| 24 | 0.0106 | 1.0000 | 1.0000 |
| 25 | 0.0053 | 1.0000 | 1.0000 |

The circuit passes support-determinism verification, and the whole thing runs end to
end in well under a minute.

For reference, majority-voting mutagenicity per (branching-atom count, `ind1`) pair
turns these two fields into a plain classifier. Its confusion matrix against the
actual labels, on the same 188 molecules:

| | Predicted mutagenic | Predicted non-mutagenic |
|---|---:|---:|
| Actually mutagenic | 108 | 17 |
| Actually non-mutagenic | 9 | 54 |

That is 162 correct out of 188, 86.2% accuracy, from branching-atom count and `ind1`
together. Branching-atom count alone gets 84.6%; adding `ind1` closes only part of
the gap. Pushing further would need a properly held-out train/test split and a real
classifier fit on more of the dataset's fields, not a majority vote read off two
fields directly, a different, separate piece of work from what this experiment sets
out to show.

## Final takeway from this experiment

**What it does prove:** The full pipeline works end-to-end on real, structured relational data, divided fitting, cause/confounder/causes_effect query marking, grounding through RelationalCircuitRegistry, support-determinism verification, and backdoor adjustment all produce a coherent result that matches a real chemistry signal (mutagenicity rising with branching-atom count, consistent with known QSAR correlates). That's genuine validation of the pipeline. 

**What it doesn't prove:** It doesn't prove branching-atom count causes mutagenicity, that's inherent to backdoor adjustment itself (it's only as sound as the assumed confounder set), but not this pipeline gap.