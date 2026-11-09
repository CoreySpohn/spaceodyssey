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

# Many worlds

```{admonition} The question for this page
When one study runs over hundreds of simulated worlds at once, what changes and what
must stay the same?
```

A single run is one draw of the sky and one draw of the measurement noise. What a
campaign is worth is an average over many such draws. The compiled executor runs a
whole batch of worlds in one program, so this page repeats the study from
[Getting started](getting-started.md) over 512 worlds and compares the result with
the host run.

```{code-cell} python
import time

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
from jax.scipy.special import ndtr

from spaceodyssey.core import (
    Capacities,
    CompiledExecutor,
    HostExecutor,
    RunStatus,
    histories_equal,
    index_world,
)
from spaceodyssey.studies import toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = toy.survey_1(caps=caps)
```

The last line builds the study of the previous page in one call. It has two stars,
one telescope and a plan of five observations, three of star 0 and two of star 1.

## A batch of worlds needs a batch of truths and keys

The world source draws one truth from one key. Splitting a root key gives one key per
world, and `vmap` turns the single draw into a batch whose leaves gain a leading
world axis. The measurement noise needs its own keys, so each world carries a truth
key and a separate root key.

```{code-cell} python
n_worlds = 512
truth_keys, roots = jax.vmap(jr.split)(jr.split(jr.PRNGKey(7), n_worlds)).transpose(
    1, 0, 2
)
truths = jax.vmap(lambda key: study.world_source.draw(key, caps))(truth_keys)
print(truths.signal.shape, roots.shape)
```

## One call runs every world

`run_worlds` takes the batch and returns a `RunResult` whose leaves all have the world
axis in front. The first call compiles the program for this study and this batch
size, so it is much slower than later calls. A second call with different truths reuses the
program and only pays for the run.

```{code-cell} python
executor = CompiledExecutor()

start = time.perf_counter()
result = executor.run_worlds(study, truths, roots, 40)
result.carry.tick.block_until_ready()
first = time.perf_counter() - start

other_keys = jr.split(jr.PRNGKey(8), n_worlds)
other_truths = jax.vmap(lambda key: study.world_source.draw(key, caps))(other_keys)
start = time.perf_counter()
again = executor.run_worlds(study, other_truths, roots, 40)
again.carry.tick.block_until_ready()
second = time.perf_counter() - start

print(f"first call {first:.2f} s, second call {second:.3f} s")
print(result.carry.belief.mean.shape)
```

## Every world reports its own status

A batch can finish unevenly, so `status` holds one code per world and `complete` one
boolean per world. An average over worlds is meaningful only when every world ran to
its own end, so the check comes before the average.

```{code-cell} python
print(int(result.complete.sum()), "of", n_worlds, "worlds complete")
assert bool(result.complete.all())
```

## The average over worlds estimates the yield

The objective scores one finished world, so `vmap` scores the batch. For this toy
there is an estimate to compare with. A star measured $n$ times has a signal-to-noise
ratio that is normal with unit standard deviation around $2 \sqrt{n}\, s$, where $s$
is its true brightness and 0.5 is the noise level. It is detected when that ratio
exceeds 3, which has probability $\Phi(2 \sqrt{n}\, s - 3)$, with $\Phi$ the normal
cumulative distribution. Averaging over $s$ uniform on 0.5 to 1.5, for $n = 3$ and
$n = 2$, and adding the two stars gives the expected number of detections per world.

```{code-cell} python
s = jnp.linspace(0.5, 1.5, 20001)
expected = sum(float(ndtr(2 * np.sqrt(n) * s - 3).mean()) for n in (3, 2))

scores = jax.vmap(lambda carry: study.objective.evaluate(carry, study))(result.carry)
standard_error = float(scores.std()) / np.sqrt(n_worlds)
print(f"estimate {expected:.3f}, batch mean {float(scores.mean()):.3f}")
print(f"standard error of the batch mean {standard_error:.3f}")
assert abs(float(scores.mean()) - expected) < 4 * standard_error
```

The batch mean sits within a few standard errors of the estimate. One world cannot
show that, because a single world scores 0, 1 or 2.

## What stays the same: the history of each world

The compiled executor and the host executor apply the same transaction rules, so a
world's history does not depend on which one ran it. A history is the canonical
account of a run: integers such as ticks, stages and counts agree exactly, and floats
agree to within 1e-12. The cell reruns three of the worlds on the host executor and
compares each with its slice of the batch.

```{code-cell} python
for w in range(3):
    host = HostExecutor().run(study, index_world(truths, w), roots[w], 40)
    print(w, histories_equal(host.history(), result.history(w)).go)
```

## What changes: shapes, compilation and policy form

Four things differ from the host run, and none of them changes a result.

| | Host executor | Compiled executor |
| --- | --- | --- |
| Worlds | One per `run`; `run_worlds` loops. | One program over a leading world axis. |
| Compilation | Each stage compiles on first use. | One program, compiled for the batch size and step budget. |
| Policy | May be written with Python control flow (`decide_py`). | Must be array code with fixed shapes and no Python branch on a traced value. |
| Inspection | Can collect each decision view as it runs. | Returns the final carry and the trace. |

The step budget and the number of worlds are shapes, so changing either recompiles.
The study's capacities are shapes too. A budget that is too small shows up as a
status, not an error, and the check above is what catches it.

```{code-cell} python
first_four = jax.tree_util.tree_map(lambda a: a[:4], truths)
short = executor.run_worlds(study, first_four, roots[:4], 8)
print([RunStatus(int(code)).name for code in short.status])
```

`TRUNCATED` means the eight steps ended before the study did.

## What this page established

The compiled executor runs a batch of worlds in one program and keeps the history of
each world equal to its host history. The number to read is an average over worlds,
and only complete worlds count. The next four pages explain what the log, the events
and the ledger of a single world mean, starting with the line the toy drew between
the truth and what the study can see.
