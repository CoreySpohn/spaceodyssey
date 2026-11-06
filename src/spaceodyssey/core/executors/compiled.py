"""Executors compiled."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.carry import empty_trace, init_carry
from spaceodyssey.core.enums import I64
from spaceodyssey.core.events import where_tree
from spaceodyssey.core.executors.base import AbstractExecutor, RunResult, index_world
from spaceodyssey.core.refresh import answer_refreshes
from spaceodyssey.core.service import answer_pending
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


@eqx.filter_jit
def _init_worlds(study, truths):
    return jax.vmap(init_carry, in_axes=(None, 0))(study, truths)


@eqx.filter_jit
def _scan_chunk(executor, study, carries, truths, roots, left, wake_fn, n_steps):
    """Continue each world for up to `n_steps` transitions or until it must wait.

    A step is taken only while the world has budget `left` and holds no unanswered
    request or an uninstalled refresh checkpoint; a step not taken leaves the carry
    alone and emits an empty trace row, so waiting for the host costs no budget.
    """

    def one(c0, truth, root, left0):
        def body(state, i):
            c, budget = state
            go = (budget > 0) & ~c.request.valid & ~c.refresh.valid
            c2, tr = executor.transition(c, i, root, study, truth, wake_fn=wake_fn)
            tr = where_tree(go, tr, empty_trace(study.caps))
            return (where_tree(go, c2, c), budget - go.astype(I64)), tr

        (c, budget), trace = lax.scan(body, (c0, left0), jnp.arange(n_steps))
        return c, budget, trace

    return jax.vmap(one)(carries, truths, roots, left)


def _put_world(batch, w, one):
    """Write one world's carry back into the batch."""
    return jax.tree_util.tree_map(lambda whole, x: whole.at[w].set(x), batch, one)


class CompiledExecutor(AbstractExecutor):
    """Batch worlds around a compiled scan of the shared transaction."""

    def transition(self, carry, i, root, study, truth, *, wake_fn=wake_by_audience):
        """Advance one transaction."""
        return transition(carry, i, root, study, truth, wake_fn=wake_fn)

    def run(
        self,
        study,
        truths,
        roots,
        n_steps,
        *,
        wake_fn=wake_by_audience,
        service=None,
        refit=None,
        chunk=32,
    ):
        """Run a batch of worlds for a fixed number of steps.

        With a `service` (see `AbstractHostService`) the scan runs in chunks of `chunk`
        steps and yields to the host between them: each world that reached a request
        is answered, then the scan resumes. A world takes `n_steps` steps in all, the
        same budget the host executor gives it, however many chunks that needs. The
        trace has empty rows where a world waited.

        The chunked driver is used whenever there is a `service` or the study's
        refresh schedule checkpoints (`study.refresh.every_ticks > 0`): worlds that
        reached a checkpoint are answered together through `refit` (an
        `AbstractRefitBoundary`), then the scan resumes. With a checkpointing study and
        no `refit`, each world stops at its first checkpoint and reports
        `RunStatus.AWAITING`.
        """
        require_x64()
        if service is not None or int(study.refresh.every_ticks) > 0:
            return self._run_hosted(
                study, truths, roots, n_steps, wake_fn, service, refit, chunk
            )
        carry, trace = _scan_worlds(self, study, truths, roots, wake_fn, n_steps)
        return RunResult(carry, trace, study.observer.layout(study.caps))

    def run_worlds(self, study, truths, roots, n_steps, **kw):
        """Run a batch of worlds; the same call as `run`."""
        return self.run(study, truths, roots, n_steps, **kw)

    def _run_hosted(
        self, study, truths, roots, n_steps, wake_fn, service, refit, chunk
    ):
        carries = _init_worlds(study, truths)
        n_worlds = roots.shape[0]
        left = jnp.full(n_worlds, n_steps, I64)
        traces = []
        while True:
            carries, left, trace = _scan_chunk(
                self, study, carries, truths, roots, left, wake_fn, chunk
            )
            traces.append(trace)
            if service is not None:
                for w in np.flatnonzero(np.asarray(carries.request.valid)):
                    answered = answer_pending(index_world(carries, w), study, service)
                    carries = _put_world(carries, w, answered)
            if refit is not None:
                carries = answer_refreshes(carries, roots, study, refit)
            # a world frozen on something nobody answers cannot go on
            stuck = (np.asarray(carries.request.valid) & (service is None)) | (
                np.asarray(carries.refresh.valid) & (refit is None)
            )
            finished = (
                np.asarray(carries.done | carries.idle | carries.stalled)
                | np.asarray(carries.overflow)
                | (np.asarray(left) <= 0)
                | stuck
            )
            if finished.all():
                break
        trace = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs, axis=1), *traces)
        return RunResult(carries, trace, study.observer.layout(study.caps))
