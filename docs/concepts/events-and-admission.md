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

# Events and admission

```{admonition} The question for this page
In what order do things happen in a campaign, and who decides that an observation is
allowed?
```

A campaign is not a loop over targets. An exposure ends at some tick, its result
arrives later, and a decision is made only when something wakes the policy. The core
keeps a queue of events and handles one per step. When a decision asks for an
observation, the core checks it before committing to it, and the check can refuse.
The ledger and the carry named below are defined in
[The ledger](ledger.md) and [Getting started](../getting-started.md).

```{code-cell} python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import (
    AdmissionCode,
    Capacities,
    EffectKind,
    EventKind,
    HostExecutor,
    RunStatus,
    Stage,
)
from spaceodyssey.studies import builder, toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = builder.survey_1(caps=caps)
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
key = jr.PRNGKey(0)


def show(result):
    """Print the events a run handled, with the decision each one woke."""
    trace = result.trace
    print("step  tick  stage    event    decision  admission")
    for i in np.flatnonzero(np.asarray(trace.handled)):
        woke = bool(trace.wake[i])
        decision = EffectKind(int(trace.eff_kind[i][0])).name if woke else "-"
        code = AdmissionCode(int(trace.admit_code[i][0])).name if woke else "-"
        print(
            f"{i:4d} {int(trace.tick[i]):5d}  {Stage(int(trace.stage[i])).name:8s}"
            f" {EventKind(int(trace.kind[i])).name:8s} {decision:9s} {code}"
        )
    print("run ended:", RunStatus(int(result.status)).name)


result = HostExecutor().run(study, truth, key, 40)
show(result)
```

The study is the five-observation toy from [Getting started](../getting-started.md).
The table lists every event the run handled, one row per step. Its columns are:

| Column | What it shows |
| --- | --- |
| step | The index of the step, counting from 0. |
| tick | The simulated time at which the event was handled. |
| stage | Where in the tick the event ran, explained below. |
| event | The kind of event: `DONE` (an exposure ended), `RELEASE` (a result was delivered), `WAIT` (the opening wake) or `STOP` (the campaign ends). |
| decision | The effect the policy asked for, or `-` if the event did not wake it. |
| admission | The result of the admission check on that effect (`OK` or a refusal), or `-`. |

The table is built from the run's trace, which holds one row per step.

## A campaign is a queue of events, handled one per step

The table has twelve rows. Each of the five observations leaves two: a `DONE` event
when the exposure ends and the record is written, and a `RELEASE` event when the
result reaches its readers. One more `WAIT` event at tick 0 is the opening wake that
lets the policy make its first decision, and a final `STOP` event ends the campaign
after the policy asked to stop. The policy's stop asks to drain: it takes no new work
and lets anything already in flight finish before the campaign ends. Here nothing is in
flight at that moment, so the run ends at once.

```{code-cell} python
handled = int(np.asarray(result.trace.handled).sum())
print(handled, "events =", 1 + 5 * 2 + 1)
assert handled == 1 + 5 * 2 + 1
```

The ticks follow from the channel. The first exposure starts after a 10-tick slew,
ends 100 ticks later at tick 110, and is released 20 ticks after that at tick 130. The
next decision happens at that release, so the cycle repeats every 130 ticks.

## Events at the same tick are ordered by stage

A queued event carries a tick and a stage, and the core always handles the smallest
pair, tick first. Within one tick the stages run in a fixed order: `SETTLE` (an
occupancy ends), `RELEASE` (a result is delivered), `ANALYZE`, `DECIDE` and `ADMIT`.
Each event kind has its stage: a `DONE` event is handled at `SETTLE`, a `RELEASE`
event at `RELEASE`, and the `WAIT` and `STOP` events at `ADMIT`, as the table shows.
So a result can never be released before the exposure that made it has settled, even
when both happen at the same tick. (Events that tie on both tick and stage are broken
by further ordering keys that the core carries, which this toy never needs.)

The `ANALYZE` and `DECIDE` stages are for analysis work between a release and the next
decision, and for the decision itself. The toy does no analysis and folds its running
mean as it reads records, so no event in these runs is handled at either stage; the
policy's decision runs inside the step of the event that woke it.

Setting the channel latency to zero makes an exposure end and its result arrive at the
same tick. The table shows each pair in stage order, `DONE` first.

The cell below changes one field of the study. A study is an immutable Equinox
module, so it cannot be edited in place: `eqx.tree_at(where, study, new)` returns a
copy in which the leaf picked out by `where` is replaced by `new`. Every variant of
the toy in these pages is made this way.

```{code-cell} python
no_latency = eqx.tree_at(lambda s: s.channels.latency_ticks, study, jnp.array([0]))
show(HostExecutor().run(no_latency, truth, key, 40))
```

The channel also declares what wins when a fault and a settlement coincide, through
its coincident order. This toy channel never faults, so that setting is not exercised
here.

## A release wakes the policy, which decides in the same step

A step handles one event. If that event was released to the policy audience, the
policy is woken, and in the same step it decides, its effects are admitted, and the
decision is closed. The `decision` column shows what the policy asked for after each
wake. The opening `WAIT` event and the first four releases each led to an `OBSERVE`,
which is five in all, and the fifth release led to a `STOP`. The `WAIT` event is a
different thing from the `WAIT` effect: the event is the opening wake the core queues
at tick 0, and the effect is something a policy can ask for to do nothing until the
next wake.

An event the policy cannot read wakes nothing. The `DONE` rows have no decision,
because the exposure ending is not visible to the policy. Only the release is.

## Admission checks an effect before the campaign commits to it

A decision returns effects, such as `OBSERVE`, `WAIT` or `STOP`. An observation is
admitted only if its lane is open, the lane is free, the exposure ends inside the
horizon, the budgets cover its cost and the event queue has room. The checks run in
that order (the codes are `LANE_CLOSED`, `LANE_BUSY`, `HORIZON`, `NO_FUNDS` and
`NO_SLOT`), the first failure names the refusal in the `admission` column, and a
refused effect consumes nothing and writes no record.

Shortening the horizon to 300 ticks shows one refusal. The third observation would
start at tick 270 and end at 370, past the horizon.

```{code-cell} python
short = eqx.tree_at(lambda s: s.horizon_tick, study, jnp.asarray(300))
refused = HostExecutor().run(short, truth, key, 40)
show(refused)
print("records written:", int(np.asarray(refused.carry.log.valid).sum()))
```

Cutting the telescope budget from 10 units to 2 shows another. Two observations spend
it, and the third is refused for funds.

```{code-cell} python
poor = eqx.tree_at(lambda s: s.budget, study, study.budget.at[0, 0].set(2.0))
broke = HostExecutor().run(poor, truth, key, 40)
show(broke)
```

## A refusal is a status, not an error

Neither refused run raised an exception. Each ended `IDLE`: the queue emptied while
the study had not reached its own end, because the policy was only woken by results,
and after the refusal no result was on its way. A refused effect is information the
policy could have used, and a policy that cannot see it can stall the run, which is why
`complete` is the first thing to check after any run.

```{code-cell} python
print(bool(result.complete), bool(refused.complete), bool(broke.complete))
```

## What this page established

The core handles one event per step, in the order of tick, then stage. A release wakes
the policy, which decides in that same step, and admission checks the lane, the horizon,
the funds and the queue room, in that order, before anything is committed. The next page follows one record from the
end of its exposure to the moment a reader can see it.
