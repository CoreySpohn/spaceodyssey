---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
kernelspec:
  display_name: Python 3
  language: python
  name: python3
---

# The ledger

```{admonition} The question for this page
Where does the budget go, and how can a run prove that it was spent correctly?
```

A telescope has finite time, and a program has a finite allocation. A simulator that
lets a policy overspend gives a yield nobody could achieve. The ledger is the part of
the carry (the full state of a run, defined in [Getting started](../getting-started.md))
that holds every balance and every lane's state, and it is the first place
to look when an observation is refused.

```{code-cell} python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import Capacities, HostExecutor
from spaceodyssey.studies import builder, toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = builder.survey_1(caps=caps)
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
key = jr.PRNGKey(0)
result = HostExecutor().run(study, truth, key, 40)
ledger = result.carry.ledger
```

The study is the five-observation toy from [Getting started](../getting-started.md).

## The budget is a table of balances

The budget has one row for each asset, one for the program and one shared row, and one
column for each of three resources. A channel's cost has one entry per resource. Each
resource is charged to its own row: the first to the asset that owns the lane, the
second to the program, and the third to the shared row. So the toy's one-unit cost
moves the diagonal of the table.

```{code-cell} python
print("initial balances")
print(np.asarray(study.budget))
print("cost of one observation on channel 10:", np.asarray(study.channels.cost[0]))
print("final balances")
print(np.asarray(ledger.balance))
```

Five observations at one unit each should leave the telescope with 5 of its 10 units,
and the program and shared rows with 95 of their 100. The final balances show exactly
that.

## An admission debits at the start of the exposure

The debit happens when the policy's observation is admitted, at tick 0 for the first
one, not when the exposure ends. The cell stops the run after a single step, which
handles the opening wake, the decision and the admission. The balances have already
dropped, and the log has no record yet, because the record is written when the
exposure ends at tick 110.

```{code-cell} python
early = HostExecutor().run(study, truth, key, 1)
print("balance diagonal", np.diag(np.asarray(early.carry.ledger.balance)))
print("records written ", int(np.asarray(early.carry.log.valid).sum()))
```

## Every debit is also written to a counter

Beside the balances, the ledger keeps two counters of the same shape: `debited`, the
total admissions have taken, and `refunded`, the total a cancellation has given back.
Nothing in the transaction rules reads them. They exist so that a finished run can be
audited from its own state.

The audit is one equation. Each balance must equal its initial value, minus what was
debited, plus what was refunded. This run cancelled nothing, so `refunded` is zero.

```{code-cell} python
net = np.asarray(study.budget) - np.asarray(ledger.debited) + np.asarray(ledger.refunded)
print("debited diagonal ", np.diag(np.asarray(ledger.debited)))
print("refunded total   ", float(np.asarray(ledger.refunded).sum()))
assert np.allclose(net, np.asarray(ledger.balance), rtol=0, atol=1e-12)
```

The equation holds in every run, including ones that stop early or refuse requests,
and it is one of the generic invariants that `check_run` evaluates for any study.

## A refused observation leaves the ledger as it was

Cutting the telescope's budget to 2 units lets two observations through and refuses the
third for funds. The refusal debits nothing, so the telescope balance stops at zero and
the counters stop at 2. The program and shared rows are charged only for admitted
observations.

```{code-cell} python
poor = eqx.tree_at(lambda s: s.budget, study, study.budget.at[0, 0].set(2.0))
broke = HostExecutor().run(poor, truth, key, 40)
after = broke.carry.ledger
print("balance diagonal ", np.diag(np.asarray(after.balance)))
print("debited diagonal ", np.diag(np.asarray(after.debited)))
net = np.asarray(poor.budget) - np.asarray(after.debited) + np.asarray(after.refunded)
assert np.allclose(net, np.asarray(after.balance), rtol=0, atol=1e-12)
```

## The ledger also tracks each lane

A lane runs one activity at a time. The ledger records the tick at which each lane
frees up, which is how admission knows a new observation would collide. After the full
run the lane is busy until tick 630, the end of the fifth exposure. The final stop also
closed the lane so that nothing more could be admitted.

```{code-cell} python
print("lane busy until", int(ledger.lane_busy_until[0]), "closed", bool(ledger.lane_closed[0]))
assert int(ledger.lane_busy_until[0]) == int(np.asarray(result.carry.log.end_tick).max())
```

## What this page established

Admission debits the budget table at the start of an exposure and writes the same
amount to a counter, so a finished run can prove its balances from its own state. A
refusal debits nothing. The three concept pages so far describe one world. The how-to
pages show how to replace each part of it, starting with the channel.
