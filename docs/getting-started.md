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

# Getting started

```{admonition} The question for this page
What does a simulated observing campaign leave behind when it finishes?
```

A campaign is a schedule played out against a simulated sky. This page builds the
smallest one that is still complete, runs it, and reads the three things it leaves
behind: a log of records, a ledger of spent budget, and the beliefs the study formed.

## Install

```bash
pip install spaceodyssey
```

spaceodyssey runs on JAX in 64-bit mode, because event times are 64-bit integers.
It refuses to build a study without that mode and never changes the setting itself,
so every script starts with the first cell below.

```{code-cell} python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import (
    ALL_EFFECTS,
    PUBLIC,
    Agenda,
    AssetTable,
    Capacities,
    ChannelTable,
    Clock,
    CoincidentOrder,
    HostExecutor,
    Study,
    histories_equal,
    preflight,
)
from spaceodyssey.studies import toy
```

## What is the smallest campaign worth running?

Take two stars and one telescope. The telescope observes star 0, star 1, star 0,
star 1 and star 0, in that order. One observation slews for 10 ticks, exposes for
100 ticks, and has its result released to the study 20 ticks later. Each observation
costs one unit from three budgets. The study then asks one question: how many stars
reached a signal-to-noise ratio above 3?

The words below name the parts of that story, and the rest of the page builds each one.

| Word | Meaning in the toy campaign |
| --- | --- |
| tick | One step of simulated time. The clock here makes a tick one second. |
| subject | A star the campaign can observe. |
| asset | A telescope or instrument whose budget is spent. |
| lane | A place where one activity runs at a time, here the single telescope pointing. |
| channel | A kind of observation: its duration, cost and who may see the result. |
| record | One row of the log: what an observation produced. |

## Capacities fix every array size up front

JAX needs array shapes at compile time, so a study declares how many of everything
it can hold before it runs. `Capacities` carries those numbers. Changing any of them
changes the shapes and recompiles the programs, which is why they live in one place.

The registry is the other half of the declaration. It lists the record kinds the
study can write. The toy registry holds a measurement kind with three payload columns
(the measured value, the noise draw that produced it, and the exposure fraction) and
an auxiliary kind. Lowering the registry into the capacities records those columns.

```{code-cell} python
caps = toy.toy_registry().lower(
    Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
)
print(caps.n_subjects, caps.n_channels, caps.payload_width)
```

## Two tables describe the instrument and the budget

The channel table says what one kind of observation is. Channel 10 runs on lane 0,
occupies it for 100 ticks, releases its result 20 ticks later to everyone (`PUBLIC`
means both the observer and the policy), and costs one unit of each of the three
resources. The asset table says that lane 0 belongs to asset 0 and that pointing it
costs a 10-tick slew.

The budget has one row for each asset, one for the program (the campaign as a whole)
and one shared row. The telescope may spend 10 units, and the program and the shared
row 100 each.

```{code-cell} python
channels = ChannelTable(
    id=jnp.array([10]),
    lane=jnp.array([0]),
    occupancy_ticks=jnp.array([100]),
    latency_ticks=jnp.array([20]),
    kind=jnp.array([toy.RECORD_MEASUREMENT]),
    audience=jnp.array([int(PUBLIC)]),
    cost=jnp.ones((1, 3)),
    charge=jnp.ones((1, 3)),
    coincident_order=jnp.array([CoincidentOrder.FAULT_FIRST]),
)
assets = AssetTable(
    lane_asset=jnp.array([0]),
    transition_ticks=jnp.array([10]),
    freeze_ticks=jnp.zeros(1, jnp.int64),
    disruptive_resource=jnp.full(1, -1, jnp.int64),
)
budget = jnp.array([[10.0, 10.0, 10.0], [100.0, 100.0, 100.0], [100.0, 100.0, 100.0]])
```

Three fields in those tables are set to their neutral values here. `coincident_order`
only matters when two events fall on the same tick, `freeze_ticks` is a pause after
a disruptive change of the lane, and `disruptive_resource` names the resource such a
change spends (minus one means none).

## Six components supply the behavior

The tables hold numbers. The behavior comes from six components, each an Equinox
module that implements one small contract. The toy components keep every one of them
a few lines long.

| Component | Its job | Toy class |
| --- | --- | --- |
| world source | Draws one simulated world, the truth. | `ToyWorldSource` |
| channel model | Turns the truth into a measured value. | `ToyPhotometry` |
| observer | Folds released records into a belief. | `ToyRunningMean` |
| policy | Decides what to do next from what it can see. | `FixedPlanPolicy` |
| design | Holds continuous settings, here a gain. | `ToyDesign` |
| objective | Scores the finished run. | `ToyObjective` |

The `process` argument is a seventh slot for a process that seeds its own events into the queue, such as a
scheduled fault. `NullProcess` seeds none.

The fixed plan lists (subject, channel) pairs, so the policy here never reacts to
what it sees. The agenda holds the policies together with the tick at which each
phase ends and the effects each phase may ask for. One phase lasting to tick 1000
is enough.

```{code-cell} python
plan = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))
study = Study(
    caps=caps,
    clock=Clock(1.0),
    channels=channels,
    channel_model=toy.ToyPhotometry(noise=jnp.array([0.5])),
    assets=assets,
    budget=budget,
    horizon_tick=1000,
    agenda=Agenda((toy.FixedPlanPolicy(plan=plan),), (1000,), (ALL_EFFECTS,)),
    process=toy.NullProcess(),
    observer=toy.ToyRunningMean(),
    design=toy.ToyDesign(gain=1.0),
    world_source=toy.ToyWorldSource(),
    objective=toy.ToyObjective(threshold=3.0),
)
```

## The study can check itself before it runs

`preflight` reads what each component declares about itself, such as the truth it
reads or the effects it asks for, and checks that the declarations fit together and
fit the executor. It runs no simulation. An empty list means no finding. A clean
result is not a proof of correct science, only that the wiring is consistent.

```{code-cell} python
print(preflight(study, executor="host"))
```

## A run needs a truth, a key and a step budget

The truth is the simulated world: the true brightness of each star. Here we choose it
by hand, 1.0 and 1.2, so that later cells can compare what the study believed with
what is true. The root key seeds the measurement noise. The step budget caps how many
queued events the run may handle, so a mistake cannot loop forever.

The host executor runs one world with a Python loop over events. It returns a
`RunResult`, and `complete` is true only when the study reached its own end. Check it
before reading any score, because a run that ran out of steps or capacity also
returns.

```{code-cell} python
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
result = HostExecutor().run(study, truth, jr.PRNGKey(0), 40)
print(bool(result.complete), int(result.status))
```

## The log holds one record per observation

The log is a fixed-size table, so most rows are unused. The `valid` column marks the
rows that hold records. Reading the five valid rows shows when each observation
started, when its exposure ended, and when its result reached the study.

The policy decides only when a result arrives, so observations run back to back.
Each one then takes 10 + 100 + 20 = 130 ticks from decision to release, which puts
the starts at 10, 140, 270, 400 and 530. The cell computes those numbers and checks
the log against them.

```{code-cell} python
log = result.carry.log
valid = np.asarray(log.valid)
cols = {
    name: np.asarray(getattr(log, name))[valid]
    for name in ("subject", "start_tick", "end_tick", "released_tick")
}
value = np.asarray(log.payload)[valid][:, 0]

expected_start = 10 + 130 * np.arange(5)
assert (cols["start_tick"] == expected_start).all()
assert (cols["end_tick"] == expected_start + 100).all()
assert (cols["released_tick"] == expected_start + 120).all()

print("subject  start  end  released  value")
for i in range(5):
    print(
        f"{cols['subject'][i]:7d} {cols['start_tick'][i]:6d} {cols['end_tick'][i]:4d}"
        f" {cols['released_tick'][i]:9d} {value[i]:7.3f}"
    )
```

## The ledger shows where the budget went

The ledger keeps the balances of the budget table and a running total of everything
debited. Five observations at one unit each should lower each used balance by 5.
The telescope row is charged on the first resource, the program row on the second and
the shared row on the third, so the changes sit on the diagonal.

```{code-cell} python
ledger = result.carry.ledger
print(np.asarray(ledger.balance))
print(np.asarray(ledger.debited))
assert (np.diag(np.asarray(ledger.debited)) == 5.0).all()
```

## The belief is what the study concluded

The observer folded each released record into a running mean per subject, and the
objective scores those beliefs. A subject counts as detected when its mean times the
square root of its measurement count, divided by the noise level 0.5, exceeds 3.

The plan observed star 0 three times and star 1 twice. If every measurement equalled
the truth, the ratios would be 1.0 * sqrt(3) / 0.5 = 3.46 and 1.2 * sqrt(2) / 0.5 =
3.39, both above 3. The noisy run can fall on either side of each, and the cell shows
which.

```{code-cell} python
belief = result.carry.belief
noise = 0.5
count = np.asarray(belief.n)
snr = np.asarray(belief.mean) * np.sqrt(count) / noise
ideal = np.asarray(truth.signal) * np.sqrt(count) / noise
print("count   ", count)
print("ideal   ", ideal.round(2))
print("mean    ", np.asarray(belief.mean).round(3), "truth", np.asarray(truth.signal))
print("snr     ", snr.round(2))
print("detected", int(study.objective.evaluate(result.carry, study)))
```

## One call builds the same study

The library ships this toy study as `survey_1`, which builds the tables, the
components and the study in one call. Running it with the same capacities and the same
truth gives the same history: every integer equal and every float within 1e-12. The
history is the canonical record of a run, and comparing histories is how the library
compares executors, as the next page shows.

```{code-cell} python
other = toy.survey_1(caps=caps)
other_result = HostExecutor().run(other, truth, jr.PRNGKey(0), 40)
print(histories_equal(result.history(), other_result.history()).go)
```

## What this page established

A study is two tables, a registry of record kinds and six components. A run turns it
into a log of records, a ledger and a belief, and `complete` says whether the run
finished. The next page runs the same study over many simulated worlds at once.
