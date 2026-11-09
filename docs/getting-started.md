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

A campaign is a schedule played out against a simulated sky. This page runs the
smallest complete one first, reads the three things it leaves behind (a log of
records, a ledger of spent budget and the beliefs the study formed), and only then
takes the study apart to see how it was built.

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

from spaceodyssey.core import Capacities, HostExecutor, RunStatus
from spaceodyssey.studies import toy
```

## What is the smallest campaign worth running?

Take two stars and one telescope. The telescope observes star 0, star 1, star 0,
star 1 and star 0, in that order. One observation slews for 10 ticks, exposes for
100 ticks, and has its result released to the study 20 ticks later. Each observation
costs one unit from each of three budgets. The study then asks one question: how many
stars reached a signal-to-noise ratio above 3?

The words below name the parts of that story. Later sections define the rest as they
meet them.

| Word | Meaning in the toy campaign |
| --- | --- |
| tick | One step of simulated time. The clock here makes a tick one second. |
| subject | A star the campaign can observe. |
| asset | A telescope or instrument whose budget is spent. |
| lane | A place where one activity runs at a time, here the single telescope pointing. |
| channel | A kind of observation: its duration, cost and who may see the result. |
| record | One row of the log: what an observation produced. |
| phase | A stretch of the campaign with its own policy. The toy has one phase. |
| carry | Everything a run holds at one moment: the event queue, the log, the ledger, the belief and the flags that say how the run is going. |
| belief | What the study has concluded so far, here a mean brightness per star. |
| view | The part of the log and the belief that the policy is allowed to see. |
| trace | One row per step of the run, saying which event was handled. |
| history | The comparable summary of a run (its log, ledger, belief and trace) that tests use to check two runs agree. |

## One call builds the study and one call runs it

The library ships this toy as `toy.survey_1`. It needs only the capacities, which are
the sizes of the arrays the study may fill: two subjects, one asset, one lane and one
channel. Running it needs a truth (the simulated world, here the true brightness of
each star, chosen by hand as 1.0 and 1.2 so later cells can compare belief with
truth), a random key that seeds the measurement noise, and a step budget.

The step budget caps how many events the run may handle, so a mistake cannot loop
forever. This run handles 12 events (five observations that each leave two, an opening
event and a closing event, as the concepts section shows), so 40 steps is generous.

```{code-cell} python
caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = toy.survey_1(caps=caps)
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
result = HostExecutor().run(study, truth, jr.PRNGKey(0), 40)
print(RunStatus(result.status).name, bool(result.complete))
```

`HostExecutor` runs one world with a Python loop over events. It returns a
`RunResult`. `complete` is true only when the study reached its own end, so check it
before reading any score: a run that ran out of steps or capacity also returns, with
a different status.

## The log holds one record per observation

The log is a fixed-size table, so most rows are unused. The `valid` column marks the
rows that hold records. Reading the five valid rows shows when each observation
started, when its exposure ended and when its result reached the study.

The policy decides only when a result arrives, so observations run back to back.
Each one takes $10 + 100 + 20 = 130$ ticks from decision to release, which puts the
starts at 10, 140, 270, 400 and 530. The cell computes those numbers and checks the
log against them.

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

The ledger keeps the balances of the budget and a running total of everything
debited. The budget is a table with one row per asset, one for the program (the
campaign as a whole) and one shared row, and one column for each of three resources:
an asset-scoped one, a program-scoped one and a shared one. Each resource is charged to
its own row, so five observations at one unit each lower the diagonal of the table by
5. [The ledger](concepts/ledger.md) explains the layout and what the ledger proves.

```{code-cell} python
ledger = result.carry.ledger
print(np.asarray(ledger.balance))
print(np.asarray(ledger.debited))
assert (np.diag(np.asarray(ledger.debited)) == 5.0).all()
```

## The belief is what the study concluded

The observer folded each released record into a running mean per subject, and the
objective scores those beliefs. A subject counts as detected when its signal-to-noise
ratio exceeds 3, where the ratio is

$$
\mathrm{SNR} = \frac{\bar{y}\,\sqrt{n}}{\sigma},
$$

with $\bar y$ the mean measurement, $n$ the number of measurements and $\sigma = 0.5$
the noise level. Read it as: more measurements and brighter stars both raise the ratio.

The plan observed star 0 three times and star 1 twice. If every measurement equalled
the truth, the ratios would be $1.0\sqrt{3}/0.5 = 3.46$ and $1.2\sqrt{2}/0.5 = 3.39$,
both above 3. The noisy run can fall on either side of each, and the cell shows which.

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

## Taking the study apart

`survey_1` built three kinds of thing: array sizes, two tables and six components.
The rest of the page builds the same study by hand and checks that it behaves the
same.

### Capacities fix every array size up front

JAX needs array shapes at compile time, so a study declares how many of everything it
can hold before it runs. `Capacities` carries those numbers, and changing any of them
changes the shapes and recompiles the programs, which is why they live in one place.
Here `n_subjects` is the number of stars, `n_assets` the number of instruments,
`n_lanes` the number of lanes, `n_channels` the number of observation kinds and `n_an`
the number of subjects the study can analyze at once (it may not exceed `n_subjects`).

The registry is the other half of the declaration. It lists the record kinds the
study can write. `lower` combines the two: it widens the payload, the number of
numeric columns in each record, to fit the widest kind in the registry. The toy
measurement kind has three payload columns (the measured value, the noise draw that
produced it and the exposure fraction), and `payload_width` has a floor of 4, so the
cell prints `2 1 4`.

```{code-cell} python
from spaceodyssey.core import (
    ALL_EFFECTS,
    PUBLIC,
    Agenda,
    AssetTable,
    ChannelTable,
    Clock,
    CoincidentOrder,
    Study,
    histories_equal,
    preflight,
)

caps = toy.toy_registry().lower(caps)
print(caps.n_subjects, caps.n_channels, caps.payload_width)
```

### Two tables describe the instrument and the budget

The channel table says what one kind of observation is. Channel 10 runs on lane 0,
occupies it for 100 ticks, releases its result 20 ticks later to everyone (`PUBLIC`
means both the observer and the policy), and costs one unit of each of the three
resources. The `charge` entries must equal `cost`; they matter only for studies that
refund. The asset table says that
lane 0 belongs to asset 0 and that pointing it costs a 10-tick slew.

In the budget, row 0 is the telescope (asset 0), row 1 the program and row 2 the
shared pool, and the three columns are the asset-scoped, program-scoped and shared
resources. The telescope may spend 10 units of its own resource, and the program and
shared rows have 100 each.

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
#                 asset-scoped  program-scoped  shared
budget = jnp.array(
    [
        [10.0, 10.0, 10.0],  # row 0: the telescope
        [100.0, 100.0, 100.0],  # row 1: the program
        [100.0, 100.0, 100.0],  # row 2: the shared pool
    ]
)
```

Three fields in those tables are set to their neutral values here. `coincident_order`
only matters when two events fall on the same tick, `freeze_ticks` is a pause after
a disruptive change of the lane, and `disruptive_resource` names the resource such a
change spends (minus one means none).

### Six components supply the behavior

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

The `process` argument is a seventh slot for a process that seeds its own events into
the queue, such as a scheduled fault. `NullProcess` seeds none.

The fixed plan lists (subject, channel) pairs, so the policy here never reacts to
what it sees. The agenda holds the policies together with the tick at which each
phase ends and the effects each phase may ask for. The toy has one phase, and it ends
at tick 1000. The study's `horizon_tick` is also 1000: it is the last tick at which an
observation may end, a separate limit that admission checks, and setting the two
equal means the phase never ends before the horizon does.

```{code-cell} python
plan = ((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))
by_hand = Study(
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

### The study can check itself before it runs

`preflight` reads what each component declares about itself, such as the truth it
reads or the effects it asks for, and checks that the declarations fit together and
fit the executor. It runs no simulation. An empty list means no finding. A clean
result is not a proof of correct science, only that the wiring is consistent.

```{code-cell} python
print(preflight(by_hand, executor="host"))
```

### The hand-built study gives the same history

Running it with the same truth and key gives the same history as `survey_1`: every
integer equal and every float within $10^{-12}$. Comparing histories is how the
library compares executors, as the next page shows.

```{code-cell} python
by_hand_result = HostExecutor().run(by_hand, truth, jr.PRNGKey(0), 40)
print(histories_equal(result.history(), by_hand_result.history()).go)
```

## What this page established

A study is two tables, a registry of record kinds and six components. A run turns it
into a log of records, a ledger and a belief, and `complete` says whether the run
finished. The next page runs the same study over many simulated worlds at once.
