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

# How to write a channel

```{admonition} The task
Replace the toy's Gaussian measurement with your own model of how an observation turns
the simulated truth into a measured value.
```

A channel model is the one component that reads the truth. The toy channel adds noise
of a fixed size. This guide writes a channel whose noise grows with the square root of
the signal, as photon shot noise does, and checks that it runs on both executors.

```{code-cell} python
import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import (
    AbstractChannelModel,
    Capacities,
    CompiledExecutor,
    HostExecutor,
    histories_equal,
    preflight,
)
from spaceodyssey.studies import builder, toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = builder.survey_1(caps=caps)
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
```

## A channel model implements one method

Subclass `AbstractChannelModel` and write `realize`. It receives the truth, the
subject and channel observed, the fraction of the exposure that completed, a random
key, the study's design and its channel table. It returns two numbers: the measured
value, and the standard normal draw that produced the noise, called the variate. The
core writes the value, the variate and the fraction into the first three payload
columns of the record.

The method runs inside compiled code, so it must be a pure JAX function of its
arguments: no Python branching on traced values and no random numbers except from the
key. The class is an Equinox module, so its parameters are fields. `AbstractChannelModel`
supplies the `version` field (default 1), which the document codec reads, so the
subclass does not declare it.

Two of the arguments need a word. `design` is the study's design component, which holds
continuous settings; the toy design has one, a gain that multiplies the signal, set to 1
here. `channels` is the channel table, which a model can read for per-channel settings;
this model ignores it.

```{code-cell} python
class ShotNoisePhotometry(AbstractChannelModel):
    """Measured value with noise proportional to the square root of the signal."""

    noise: jax.Array  # noise level at unit signal, one entry per channel

    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        """Return the measured value and the standard normal variate behind it."""
        z = jr.normal(key, dtype=jnp.float64)
        signal = design.gain * truth.signal[subject]
        scale = self.noise[channel_index] * jnp.sqrt(signal)
        return signal + scale * z / jnp.sqrt(jnp.maximum(fraction, 1e-3)), z
```

The division by the square root of the fraction makes a shorter exposure noisier, as
averaging fewer photons would. The fraction is floored at 0.001 so that an exposure that
completed none of its time (a cancelled one) divides by a small number rather than by
zero. For a full exposure, the fraction is 1 and drops out.

## The new channel replaces the old one in the study

A `Study` is an immutable pytree, so the replacement goes through `eqx.tree_at`, which
returns a copy with one field swapped. The noise level 0.5 is the toy's, so a star of
brightness 1 has the same noise as before.

```{code-cell} python
shot = eqx.tree_at(
    lambda s: s.channel_model, study, ShotNoisePhotometry(noise=jnp.array([0.5]))
)
print(preflight(shot, executor="compiled"))
```

An empty list means the declarations are consistent for the compiled executor.

## The record must agree with the formula

The variate is stored in the log, so the formula can be checked against what the run
wrote. Rebuilding each value from the truth and its variate should reproduce the logged
value exactly, whatever the key was. The check below omits the fraction term, which
is valid because every exposure in this study completes in full, so the fraction is 1.

```{code-cell} python
result = HostExecutor().run(shot, truth, jr.PRNGKey(0), 40)
log = result.carry.log
valid = np.asarray(log.valid)
value, variate = np.asarray(log.payload)[valid][:, :2].T
signal = np.asarray(truth.signal)[np.asarray(log.subject)[valid]]

rebuilt = signal + 0.5 * np.sqrt(signal) * variate
print(np.c_[value, rebuilt].round(4))
assert np.allclose(value, rebuilt, rtol=0, atol=1e-12)
```

## The scatter grows with the square root of the signal

Shot noise means a brighter star scatters more. The cell draws 20000 measurements of a
dim star and of a bright star straight from `realize` and compares the standard
deviation of each with 0.5 times the square root of the brightness.

```{code-cell} python
def scatter(model, signal, n=20000):
    """Standard deviation of n measurements of a star of the given brightness."""
    keys = jr.split(jr.PRNGKey(3), n)
    values = jax.vmap(
        lambda k: model.realize(
            toy.ToyTruth(signal=jnp.array([signal, signal])),
            subject=0,
            channel_index=0,
            fraction=1.0,
            key=k,
            design=study.design,
            channels=study.channels,
        )[0]
    )(keys)
    return float(values.std())


model = ShotNoisePhotometry(jnp.array([0.5]))
for brightness in (0.5, 1.5):
    expected = 0.5 * float(np.sqrt(brightness))
    print(brightness, round(scatter(model, brightness), 3), "expected", round(expected, 3))
```

The measured scatter agrees with the expectation to within the sampling error of 20000
draws. The relative error of a standard deviation estimated from $N$ draws is about

$$
\frac{1}{\sqrt{2N}},
$$

which is 0.5 percent for $N = 20000$. Read it as: quadrupling the draws halves the
error.

## The channel runs on both executors with the same history

Both executors call `realize` with the same keys, so one world gives one history
whichever executor ran it. The compiled run takes a batch with a leading world axis,
here a batch of one.

```{code-cell} python
batch = jax.tree_util.tree_map(lambda a: a[None], truth)
compiled = CompiledExecutor().run_worlds(shot, batch, jr.PRNGKey(0)[None], 40)
print(histories_equal(result.history(), compiled.history(0)).go)
```

## Where to go next

A channel that needs more than the first three payload columns uses version 2 of the
contract: set `version = 2`, accept a keyword-only `context` argument holding the
tick, acquisition index and fraction of the record, and return the whole payload row.
The study must register a record kind with that many columns, and
[Records](../records.md) describes how. To save a study that holds your channel, see
[how to save and load a study](save-and-load.md).
