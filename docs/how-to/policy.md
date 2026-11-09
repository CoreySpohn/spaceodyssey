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

# How to write a policy

```{admonition} The task
Write a policy that chooses the next observation from what it can see, and stops the
campaign when it has seen enough.
```

The toy's fixed plan never looks at its view. This guide writes a policy that observes
whichever star has the fewest measurements and stops once every star has two, then
runs it on both executors and checks what it declares about itself.

```{code-cell} python
import dataclasses

import jax

jax.config.update("jax_enable_x64", True)

import equinox as eqx
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core import (
    AbstractPolicy,
    Capacities,
    CompiledExecutor,
    Declaration,
    Disposition,
    EffectKind,
    HostExecutor,
    Scope,
    effects_none,
    histories_equal,
    preflight,
)
from spaceodyssey.core.battery import declared_matches_run
from spaceodyssey.studies import toy

caps = Capacities(n_subjects=2, n_an=2, n_assets=1, n_lanes=1, n_channels=1)
study = toy.survey_1(caps=caps)
truth = toy.ToyTruth(signal=jnp.array([1.0, 1.2]))
```

## A policy implements one method

Subclass `AbstractPolicy` and write `decide(pstate, view, study, key)`. It returns
three things: an effect table that says what to do, the policy's next state, and a
uniform draw it used (return 0.0 when it drew nothing). The policy keeps its own state
in `pstate`, a small float array it may read and update. It reads the world only
through `view`, a snapshot of what has been released to it. [Truth and
view](../concepts/truth-and-view.md) lists what the snapshot holds.

The policy below reads the counts of measurements per star from `view.belief.n` and
observes the star with the fewest. When every star has `target` measurements, it asks
to stop.

## An effect table is a few rows of fixed size

An effect is one request: observe this subject on this channel, wait until a tick, or
stop. A decision returns a table of `caps.n_effects` rows, two by default, and empty
rows do nothing. The helper `effects_none` builds an empty table, and
`dataclasses.replace` fills the rows the policy uses. The `group` column ties effects
of one decision together, because the core admits or refuses them as a unit. The
decision index `view.n_dec` is a convenient group number.

A `STOP` row names a scope and a disposition. Campaign scope with the drain
disposition ends the campaign once the work already admitted has finished. The
`channel_index` column is a position in the channel table, not a channel id, and
position 0 is the toy's only channel.

```{code-cell} python
class BalancedPolicy(AbstractPolicy):
    """Observe the least-measured star until every star has `target` measurements."""

    target: int = eqx.field(default=2, static=True)

    def decide(self, pstate, view, study, key):
        """Choose the next observation, or stop when every star is measured enough."""
        counts = view.belief.n
        finished = jnp.all(counts >= self.target)
        empty = effects_none(study.caps)
        effects = dataclasses.replace(
            empty,
            kind=empty.kind.at[0].set(
                jnp.where(finished, EffectKind.STOP, EffectKind.OBSERVE)
            ),
            subject=empty.subject.at[0].set(jnp.argmin(counts)),
            group=jnp.full_like(empty.group, view.n_dec),
            scope=empty.scope.at[0].set(Scope.CAMPAIGN),
            disposition=empty.disposition.at[0].set(Disposition.DRAIN),
        )
        return effects, pstate, jnp.asarray(0.0, jnp.float64)
```

Every operation is a JAX operation on arrays of fixed shape, with no Python `if` on a
traced value. That is what lets the compiled executor run the policy inside its scan.

## The policy goes into the agenda

The study keeps its policies in the agenda, one per phase. The toy has one phase, so
the new policy replaces the tuple.

```{code-cell} python
balanced = eqx.tree_at(lambda s: s.agenda.policies, study, (BalancedPolicy(),))
result = HostExecutor().run(balanced, truth, jr.PRNGKey(0), 60)
log = result.carry.log
subjects = np.asarray(log.subject)[np.asarray(log.valid)]
print(bool(result.complete), subjects, np.asarray(result.carry.belief.n))
assert list(subjects) == [0, 1, 0, 1]
```

With two stars and a target of two measurements each, the least-measured rule
alternates, which gives subjects 0, 1, 0, 1. The ties at the start go to star 0 because
`argmin` takes the first.

## The compiled executor gives the same history

```{code-cell} python
batch = jax.tree_util.tree_map(lambda a: a[None], truth)
compiled = CompiledExecutor().run_worlds(balanced, batch, jr.PRNGKey(0)[None], 60)
print(histories_equal(result.history(), compiled.history(0)).go)
```

## A policy declares the stops it asks for

`preflight` reads a component's `declare` method and never runs the study, so the
policy must say what it is going to ask for. `Declaration(stops=...)` lists the
(scope, disposition) pairs the policy may emit. The generated check
`declared_matches_run` then runs the study and fails when the policy asks for a stop
it did not declare.

The first policy never declared its stop, so the check fails and names it. The second
adds `declare` and passes.

```{code-cell} python
class DeclaredBalancedPolicy(BalancedPolicy):
    """The same policy, declaring the campaign drain it asks for."""

    def declare(self):
        """The one stop this policy emits."""
        return Declaration(stops=((Scope.CAMPAIGN, Disposition.DRAIN),))


truths = toy.ToyTruth(signal=jnp.array([[1.0, 1.2], [0.8, 1.4]]))
roots = jr.split(jr.PRNGKey(1), 2)
declared = eqx.tree_at(lambda s: s.agenda.policies, study, (DeclaredBalancedPolicy(),))

print(declared_matches_run(balanced, truths, roots, 60))
print(declared_matches_run(declared, truths, roots, 60))
print(preflight(declared, executor="compiled"))
```

Declarations are assertions by the component. A clean `preflight` does not prove a
policy keeps them, which is why the check compares them with a run.

## A policy may use Python on the host executor

Control flow on traced values is not allowed under the compiled executor, but the
host executor can call a second method, `decide_py`, written with ordinary Python on
host values. Its default calls `decide`. A policy that needs it
declares `requires=frozenset({"python_policy"})` in its `Declaration`, a capability
only the host executor provides, so that `preflight` refuses the compiled executor
for it.
