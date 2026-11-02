"""Executors compiled."""

import equinox as eqx
import jax
import jax.numpy as jnp
from jax import lax

from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.executors.base import AbstractExecutor, RunResult
from spaceodyssey.core.transaction import transition, wake_by_audience


@eqx.filter_jit
def _scan_worlds(executor, study, truths, roots, wake_fn, n_steps):
    def one(study_, truth, root):
        c0 = init_carry(study_, truth)
        return lax.scan(
            lambda c, i: executor.transition(
                c, i, root, study_, truth, wake_fn=wake_fn
            ),
            c0,
            jnp.arange(n_steps),
        )

    return jax.vmap(one, in_axes=(None, 0, 0))(study, truths, roots)


class CompiledExecutor(AbstractExecutor):
    """Batch worlds around a compiled scan of the shared transaction."""

    def transition(self, carry, i, root, study, truth, *, wake_fn=wake_by_audience):
        """Advance one transaction."""
        return transition(carry, i, root, study, truth, wake_fn=wake_fn)

    def run(self, study, truths, roots, n_steps, *, wake_fn=wake_by_audience):
        """Run a batch of worlds for a fixed number of steps."""
        require_x64()
        carry, trace = _scan_worlds(self, study, truths, roots, wake_fn, n_steps)
        return RunResult(carry, trace)
