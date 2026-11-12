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

# Records and release

```{admonition} The question for this page
When does a result exist, and who is allowed to read it?
```

An exposure that has ended has produced a result, but the study does not learn it at
that instant. Real pipelines take time to calibrate and deliver data, and a policy that
acted on a result before it arrived would be cheating. The log records both facts: when
a result exists and when each reader receives it.

```{code-cell} python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import Audience, Capacities, HostExecutor, RecordStatus
from spaceodyssey.studies import builder, toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = builder.survey_1(caps=caps)
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
views = []
result = HostExecutor().run(study, truth, jr.PRNGKey(0), 40, view_log=views)

log = result.carry.log
valid = np.asarray(log.valid)


def column(name, log=log, valid=valid):
    """The values of one log column over the rows that hold records."""
    return np.asarray(getattr(log, name))[valid]
```

The study is the five-observation toy from [Getting started](../getting-started.md),
run with its decision views kept: the `view_log` argument of `run` fills a list with the
view each decision was given, which the cells below read.

## A record has several clocks

A record is one row of the log. Besides what was observed, it holds four ticks that
mark the life of the result. `start_tick` is when the exposure began and `end_tick`
when it ended. `available_tick` is when the result exists to be delivered, which is
the end plus the channel's release latency. `released_tick` is when the policy
audience received it, and `accepted_tick` is when the observer took it in.

For the first observation the exposure runs from 10 to 110. The channel latency is 20
ticks, so the result is available at 130, and the channel releases to everyone, so the
policy and the observer both receive it at 130. The cell prints the ticks of every
record and checks them against that rule.

```{code-cell} python
latency = int(study.channels.latency_ticks[0])
ticks = ("start_tick", "end_tick", "available_tick", "released_tick", "accepted_tick")
print(" ".join(f"{t:>14s}" for t in ticks))
for row in zip(*(column(t) for t in ticks)):
    print(" ".join(f"{t:>14d}" for t in row))

assert (column("available_tick") == column("end_tick") + latency).all()
assert (column("released_tick") == column("available_tick")).all()
assert (column("accepted_tick") == column("available_tick")).all()
```

In this run `released_tick` and `accepted_tick` are equal in every row, because the
channel releases to the policy and the observer together. They separate when the
audiences differ, as the observer-only run below shows.

## A result exists before anyone can read it

Between tick 110 and tick 130 the first record is in the log and in nobody's view. The
policy is woken by the release, not by the end of the exposure, so its decisions fall
at the release ticks. The first view holds no rows, and each later view holds one more.

```{code-cell} python
print("decision ticks", [int(v.tick) for v in views])
print("rows in view  ", [int(v.n_rows) for v in views])
assert [int(v.tick) for v in views[1:]] == list(column("released_tick"))
```

Release latency sets the length of this gap. A longer latency pushes back every belief
update and every decision that waits on one.

## A channel chooses the audience a record is released to

The channel table names an audience for each channel, built from the `OBSERVER` and
`POLICY` flags. The toy channel used `PUBLIC`, which is both. The cell reruns the study
with an observer-only channel. The record is still written and the observer still
folds it, but the policy never receives it, and a policy that is woken only by
releases is never woken again.

```{code-cell} python
hidden = eqx.tree_at(
    lambda s: s.channels.audience, study, jnp.array([int(Audience.OBSERVER)])
)
quiet = HostExecutor().run(hidden, truth, jr.PRNGKey(0), 40)
quiet_log = quiet.carry.log
quiet_valid = np.asarray(quiet_log.valid)

print("observer count", np.asarray(quiet.carry.belief.n))
print("policy count  ", np.asarray(quiet.carry.pbelief.n))  # pbelief: see below
print("released tick ", np.asarray(quiet_log.released_tick)[quiet_valid])
print("accepted tick ", np.asarray(quiet_log.accepted_tick)[quiet_valid])
print("run complete  ", bool(quiet.complete))
```

The belief the observer holds is the full `belief`. The policy never sees it; it sees
`pbelief`, the policy summary, a reduced copy that the carry (the full state of a run, see
[Getting started](../getting-started.md)) updates only from records
released to the policy. The observer folded one record and accepted it at tick 130. The
policy summary counted none, the release tick stayed at -1 (so here `released_tick` and
`accepted_tick` differ), and the run stopped early because nothing woke the policy
again. An audience is part of what a study declares about its information, which is
why `preflight` reads it.

## The status says how an acquisition ended

Every record carries a status. All the toy records are `COMPLETED`. The others name
what a fault, a cancel or an empty measurement leaves behind. An observer decides what
to do with each: the toy observer folds only `COMPLETED` rows.

```{code-cell} python
print([status.name for status in RecordStatus])
print({RecordStatus(int(s)).name for s in column("status")})
```

[Records](../records.md) in the reference defines every status, the revision and
supersession fields, and how an observer folds a corrected record.

## What this page established

A record has a start, an end, an availability time, and a release to each audience.
The policy sees a record only after its release, and only if the channel names the
policy in its audience. The next page follows the money: the ledger that every
admitted observation debits.
