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

# How to write an observer

```{admonition} The task
Replace the running mean with a belief that follows your own rule for turning released
records into an estimate.
```

The observer is the study's inference. It folds each released record into a belief, and
the policy reads a summary of that belief when it decides. This guide writes an observer
that weights recent measurements more than old ones, an exponentially weighted mean, and
then checks it against the log.

```{code-cell} python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import (
    I64,
    AbstractObserver,
    Capacities,
    CompiledExecutor,
    HostExecutor,
    MeanBelief,
    RecordStatus,
    preflight,
    replay_belief,
)
from spaceodyssey.core.conformance import check_folded_once
from spaceodyssey.studies import toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = toy.survey_1(caps=caps)
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
```

## An observer implements one method, fold

Subclass `AbstractObserver` and write `fold(belief, rec, caps, key=None)`. The core
calls it once for each record the observer accepts, with the belief so far and the
record as a row of arrays. It returns the new belief. The default belief is a
`MeanBelief`, with a per-subject `mean`, a per-subject count `n` and a small auxiliary
array `aux`, so an observer that keeps that shape needs nothing else: the defaults
supply `init`, the layout and the policy summary.

`fold` sees every record the observer accepts, including kinds and statuses it does not
use, so it must leave the belief unchanged for those rather than branch on them. The
rule below takes a record only when its kind is a measurement and its status is
`COMPLETED`, and masks the update with `jnp.where`. The first measurement of a subject
sets the mean, and each later one moves it a fraction `alpha` of the way to the new
value.

```{code-cell} python
class EwmaMean(AbstractObserver):
    """Exponentially weighted mean of completed measurements for each subject."""

    alpha: float = 0.5
    measurement_kind: int = eqx.field(default=toy.RECORD_MEASUREMENT, static=True)

    def fold(self, belief, rec, caps, key=None):
        """Move the subject's mean toward a completed measurement."""
        t = rec.subject
        take = (
            rec.valid
            & (rec.kind == self.measurement_kind)
            & (rec.status == RecordStatus.COMPLETED)
        )
        x = rec.payload[0]
        first = belief.n[t] == 0
        new = jnp.where(first, x, belief.mean[t] + self.alpha * (x - belief.mean[t]))
        return MeanBelief(
            mean=belief.mean.at[t].set(jnp.where(take, new, belief.mean[t])),
            n=belief.n.at[t].add(take.astype(I64)),
            aux=belief.aux,
        )
```

The belief carries a count because the core uses it: the fingerprint of every policy
snapshot reads the count leaf, so the count must change exactly when the belief does.

## The observer goes into the study

```{code-cell} python
ewma = eqx.tree_at(lambda s: s.observer, study, EwmaMean())
print(preflight(ewma, executor="compiled"))
result = HostExecutor().run(ewma, truth, jr.PRNGKey(0), 40)
print(bool(result.complete), np.asarray(result.carry.belief.mean))
```

The study checks at construction that the layout the observer declares describes the
belief it builds, so a mismatch of shapes fails when the study is made, not mid-run.

## The belief must match the rule applied to the log

The log holds every measurement in release order, so the belief can be recomputed from
it by plain Python. For each subject, start from the first value and move a half of the
way to each later one. The recomputed means should equal the belief. The running mean
of the same values differs, which shows that the observer did something other than
average.

```{code-cell} python
log = result.carry.log
valid = np.asarray(log.valid)
values = np.asarray(log.payload)[valid][:, 0]
subjects = np.asarray(log.subject)[valid]

for star in (0, 1):
    x = values[subjects == star]
    level = x[0]
    for later in x[1:]:
        level += 0.5 * (later - level)
    print(star, round(float(level), 6), round(float(result.carry.belief.mean[star]), 6),
          "plain mean", round(float(x.mean()), 6))
    assert abs(level - float(result.carry.belief.mean[star])) < 1e-12
```

Star 0 was measured three times, so its weighted mean and plain mean differ. Star 1
was measured twice, and the weighted mean with `alpha` of one half is the plain mean
of two values, so they agree.

## The belief can be rebuilt from the log alone

`replay_belief` takes the recorded log of a run and folds each released row once,
in release order, through the study's observer. The result must equal the belief the
run carried. A fold that depended on anything outside the record, such as a Python
global, would break this equality.

```{code-cell} python
rebuilt = replay_belief(ewma, result.carry, result.trace)
print(np.array_equal(np.asarray(rebuilt.mean), np.asarray(result.carry.belief.mean)))
```

## The conformance kit checks that no record is folded twice

The kit in `spaceodyssey.core.conformance` runs generic checks on any observer.
`check_folded_once` takes a batched run and delivers one accepted record a second time,
and passes only if the belief does not change and the core flags the duplicate. A
fold that is not idempotent in this sense would double count a measurement.

```{code-cell} python
batch = jax.tree_util.tree_map(lambda a: jnp.stack([a] * 4), truth)
roots = jr.split(jr.PRNGKey(2), 4)
run = CompiledExecutor().run_worlds(ewma, batch, roots, 40)
print(check_folded_once(ewma, run, batch, roots))
```

The kit has further checks for observers that refit, that are statistical, or that
correct earlier records. [Core](../core.md) lists them.
