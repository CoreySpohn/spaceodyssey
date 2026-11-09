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

# Truth and view

```{admonition} The question for this page
What does a policy know about the simulated world, and how does the library keep it
from knowing more?
```

A simulated campaign has an advantage a real one never has: it holds the true state of
the sky. A policy that could read that state would choose perfectly and prove nothing.
The library therefore keeps two things apart. The **truth** is the simulated world.
The **view** is the masked snapshot a policy receives when it decides.

```{code-cell} python
import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import Capacities, HostExecutor
from spaceodyssey.core.battery import truth_swap
from spaceodyssey.studies import toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = toy.survey_1(caps=caps, policies=(toy.ThresholdStopPolicy(),))
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
views = []
result = HostExecutor().run(study, truth, jr.PRNGKey(0), 60, view_log=views)
print(bool(result.complete))
```

The study is the toy from [Getting started](../getting-started.md) with a policy that
reacts. It observes whichever of the two stars it has measured less, and stops once
either star reaches a signal-to-noise ratio above 3. The true brightnesses are 1.0 and
1.2, and the `view_log` argument keeps every view a decision was given.

## The truth enters the study through one door

Only the channel model receives the truth. Its `realize` method takes the truth, the
subject observed and a random key, and returns the measured value together with the
standard normal draw that produced the noise, called the variate. The toy channel
computes value = signal + 0.5 * variate for a full exposure, where 0.5 is the noise
level and the design gain is 1.

The cell checks that formula against the first record in the log. The value and
variate in the log are the ones the study's own run produced, so the check ties the
formula to the run.

```{code-cell} python
log = result.carry.log
first = int(np.argmax(np.asarray(log.valid)))
value, variate = np.asarray(log.payload)[first, :2]
subject = int(log.subject[first])

noise = float(study.channel_model.noise[0])
recovered = value - noise * variate
print(f"value {value:.4f}, variate {variate:.4f}, recovered {recovered:.4f}")
assert abs(recovered - float(truth.signal[subject])) < 1e-12
```

Subtracting the noise recovers the true brightness exactly. Anyone who holds both the
value and the variate holds the truth, so the variate must stay out of reach of every
reader that makes decisions.

## A column declares which readers may see it

Each payload column of a record kind names the audiences allowed to read it. The
`OBSERVER` audience folds records into beliefs, the `POLICY` audience decides, and the
`SCORER` audience reads the log afterward to score the run. The toy measurement kind
lets everyone read the value and the exposure fraction, and lets only the scorer read
the variate.

```{code-cell} python
from spaceodyssey.core import Audience

for column in toy.MEASUREMENT_KIND.columns:
    readers = [a.name for a in Audience if column.audience & a]
    print(f"{column.name:9s} {readers}")
```

## The view carries the same record with the variate zeroed

A snapshot copies the released rows the policy may see and zeroes every column the
policy may not read. The cell compares the last view of the run with the log. The value
and fraction columns agree, and the variate column is zero in the view and nonzero in
the log.

```{code-cell} python
last = views[-1]
n = int(last.n_rows)
seen = np.asarray(last.log.payload)[:n]
held = np.asarray(log.payload)[np.asarray(log.valid)]
print("column      view     log")
for name, j in (("value", 0), ("variate", 1), ("fraction", 2)):
    print(f"{name:9s} {seen[0, j]:7.3f} {held[0, j]:7.3f}")
assert (seen[:, 1] == 0).all() and (held[:, 1] != 0).all()
```

The column mask is the only thing between the policy and the truth. It protects the
variate because the channel model declared it scorer-only. A column the channel
declared readable would reach the policy unchanged.

## A view grows only when a result is released

A policy is woken when a record is released to it, and the view it receives holds the
released rows and nothing else. The first decision happens at tick 0 with nothing
released. Each later decision follows one release, so the view holds one more row.

```{code-cell} python
print("decision  rows in view  tick")
for i, view in enumerate(views):
    print(f"{i:8d} {int(view.n_rows):14d} {int(view.tick):5d}")
```

An observation that has ended but whose result is not yet released is in the log and
absent from every view. The next page shows when that gap opens and closes.

## Swapping the truth cannot change a decision until a record shows the difference

The generated check `truth_swap` tests the whole boundary on a real run. It runs the
study on two batches of truths with the same random keys. Up to the first released
record whose policy-readable columns differ, every trace field must be bitwise equal:
event order, decisions, admissions and policy draws. The check fails when no record
differs, because a swap that changes nothing proves nothing.

```{code-cell} python
truths_a = toy.ToyTruth(signal=jnp.array([[1.0, 1.2], [0.8, 1.4], [1.1, 0.9]]))
truths_b = toy.ToyTruth(signal=truths_a.signal + 0.3)
check = truth_swap(study, truths_a, truths_b, jr.split(jr.PRNGKey(1), 3), 60)
print(check, check.detail)
```

The detail lists, for each of three worlds, the step at which a differing record first
reached the policy. Before that step the two runs are indistinguishable to it.

## What this page established

The truth reaches the study only through the channel model. The policy receives a
snapshot of released rows with scorer-only columns zeroed, and `truth_swap` confirms
on a run that nothing else leaks. The next page follows the events that move a record
from an exposure to a view.
