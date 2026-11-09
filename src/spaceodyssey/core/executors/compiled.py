"""Executors compiled."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax

from spaceodyssey.core import checkpoint as ckpt
from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.carry import empty_trace, init_carry
from spaceodyssey.core.enums import I64
from spaceodyssey.core.events import where_tree
from spaceodyssey.core.executors.base import AbstractExecutor, RunResult, index_world
from spaceodyssey.core.lift import lift_traced
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
def _scan_bucket(executor, study, truths, roots, wake_fn, budget, n_rows):
    """The default scan with `n_rows` fixed and the step count a traced `budget`.

    Rows after the first `budget` take no step and emit an empty trace row, so one
    program serves every step count up to `n_rows`. The initial carry is built inside
    the program, as in `_scan_worlds`: a program that takes the carry as an input
    compiles the same arithmetic to different last bits.
    """

    def one(truth, root):
        def body(state, i):
            c, left = state
            go = left > 0
            c2, tr = executor.transition(c, i, root, study, truth, wake_fn=wake_fn)
            tr = where_tree(go, tr, empty_trace(study.caps))
            return (where_tree(go, c2, c), left - go.astype(I64)), tr

        (c, _), trace = lax.scan(
            body, (init_carry(study, truth), budget), jnp.arange(n_rows)
        )
        return c, trace

    return jax.vmap(one)(truths, roots)


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
        bucket=None,
        checkpoint=None,
        resume=False,
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

        With `bucket` (an int) and neither of those, `n_steps` is rounded up to a
        multiple of `bucket` for the compiled scan and the step count is passed as an
        array, so every `n_steps` in the same bucket reuses one program instead of
        compiling its own. Steps past `n_steps` are not taken and the trace is cut to
        `n_steps` rows: the carry and trace are the default path's, bit for bit. The
        default (`bucket=None`) is the unchanged scan.

        With a `checkpoint` (a `CheckpointSink`) the chunked driver is used and the run
        saves at the end of every chunk (after each refit install, `inference`; at the
        end, `final`, or `hold` when a budget ended with a refresh checkpoint held). The
        boundaries are chunk ends and holds: there is no per-event boundary inside a
        compiled scan. `resume` continues from the sink: held checkpoints and requests
        are answered first, once, and `n_steps` is a fresh budget for this call. A
        resumed run equals the uninterrupted run at the same `chunk`, not the single
        scan, and `chunk` is part of the run identity.
        """
        require_x64()
        given = study
        study = lift_traced(study)
        if bucket is not None and (
            isinstance(bucket, bool) or not isinstance(bucket, int) or bucket <= 0
        ):
            raise ValueError(f"bucket must be a positive int, not {bucket!r}")
        if (
            checkpoint is not None
            or service is not None
            or (int(study.refresh.every_ticks) > 0)
        ):
            return self._run_hosted(
                study,
                truths,
                roots,
                n_steps,
                wake_fn,
                service,
                refit,
                chunk,
                (checkpoint, resume, given),
            )
        if bucket is not None and n_steps > 0:
            rows = -(-n_steps // bucket) * bucket
            carry, trace = _scan_bucket(
                self, study, truths, roots, wake_fn, jnp.asarray(n_steps, I64), rows
            )
            trace = jax.tree_util.tree_map(lambda x: x[:, :n_steps], trace)
            return RunResult(carry, trace, study.observer.layout(study.caps), roots)
        carry, trace = _scan_worlds(self, study, truths, roots, wake_fn, n_steps)
        return RunResult(carry, trace, study.observer.layout(study.caps), roots)

    def run_worlds(self, study, truths, roots, n_steps, **kw):
        """Run a batch of worlds; the same call as `run`."""
        return self.run(study, truths, roots, n_steps, **kw)

    def _run_hosted(
        self, study, truths, roots, n_steps, wake_fn, service, refit, chunk, ck=None
    ):
        checkpoint, resume, given = ck or (None, False, None)
        carries = _init_worlds(study, truths)
        n_worlds = roots.shape[0]
        left = jnp.full(n_worlds, n_steps, I64)
        traces = []
        prefix, saved = None, 0
        if checkpoint is not None:
            carries, prefix = self._begin(
                checkpoint,
                (given, truths, roots),
                (study, carries),
                resume,
                (wake_fn, service, refit, chunk),
            )
            if service is not None:
                carries = self._answer_requests(carries, study, service)
            if refit is not None:
                carries = answer_refreshes(
                    carries, roots, study, refit, left, keep=True
                )
        while True:
            if checkpoint is not None and self._all_finished(
                carries, left, service, refit
            ):
                break
            carries, left, trace = _scan_chunk(
                self, study, carries, truths, roots, left, wake_fn, chunk
            )
            traces.append(trace)
            if service is not None:
                carries = self._answer_requests(carries, study, service)
            had = np.asarray(carries.refresh.valid)
            if refit is not None:
                carries = answer_refreshes(
                    carries, roots, study, refit, left, keep=checkpoint is not None
                )
            installed = bool((had & ~np.asarray(carries.refresh.valid)).any())
            finished = self._all_finished(carries, left, service, refit)
            if checkpoint is not None:
                boundary = (
                    (
                        ckpt.AT_HOLD
                        if bool(np.asarray(carries.refresh.valid).any())
                        else ckpt.FINAL
                    )
                    if finished
                    else ckpt.AFTER_INFERENCE
                    if installed
                    else ckpt.AT_CHUNK
                )
                if checkpoint.wants(boundary):
                    new = jax.tree_util.tree_map(
                        lambda *xs: np.concatenate([np.asarray(x) for x in xs], 1),
                        *traces[saved:],
                    )
                    rows = sum(int(t.handled.shape[1]) for t in traces)
                    checkpoint.save(
                        boundary,
                        carries,
                        new,
                        rows + (0 if prefix is None else _rows(prefix)),
                    )
                    saved = len(traces)
            if finished:
                break
        if traces:
            trace = jax.tree_util.tree_map(
                lambda *xs: jnp.concatenate(xs, axis=1), *traces
            )
        else:
            trace = jax.tree_util.tree_map(
                lambda x: jnp.empty((n_worlds, 0, *x.shape), x.dtype),
                empty_trace(study.caps),
            )
        if prefix is not None:
            trace = jax.tree_util.tree_map(
                jnp.asarray, ckpt.concat_rows(prefix, trace, 1)
            )
        held = None
        if checkpoint is not None:
            held = (
                carries.refresh.valid
                if refit is not None
                else jnp.zeros(n_worlds, bool)
            )
        return RunResult(carries, trace, study.observer.layout(study.caps), roots, held)

    @staticmethod
    def _answer_requests(carries, study, service):
        """Answer the pending request of each world that holds one."""
        for w in np.flatnonzero(np.asarray(carries.request.valid)):
            answered = answer_pending(index_world(carries, w), study, service)
            carries = _put_world(carries, w, answered)
        return carries

    @staticmethod
    def _all_finished(carries, left, service, refit):
        """Whether no world can take another step in this call."""
        # a world frozen on something nobody answers cannot go on
        stuck = (np.asarray(carries.request.valid) & (service is None)) | (
            np.asarray(carries.refresh.valid) & (refit is None)
        )
        # a kept hold has no budget to pay for its completion step
        waiting = np.asarray(carries.refresh.valid) & (np.asarray(left) <= 0)
        done = (
            np.asarray(carries.done | carries.idle | carries.stalled)
            | np.asarray(carries.overflow)
            | (np.asarray(left) <= 0)
            | stuck
            | waiting
        )
        return bool(done.all())

    def _begin(self, checkpoint, ids, parts, resume, how):
        """Open the sink: the batched carry to continue from, and the saved rows."""
        given, truths, roots = ids
        study, carries = parts
        wake_fn, service, refit, chunk = how
        n_worlds = roots.shape[0]
        row = jax.tree_util.tree_map(
            lambda x: np.broadcast_to(np.asarray(x), (n_worlds, *np.shape(x))),
            empty_trace(study.caps),
        )
        config = {
            "executor": "compiled",
            "row_axis": 1,
            "chunk": int(chunk),
            "wake_fn": ckpt.callable_name(wake_fn),
            "service": None if service is None else ckpt.callable_name(type(service)),
            "refit": None if refit is None else ckpt.binding_record(refit),
        }
        got = checkpoint.begin(
            given,
            truths,
            roots,
            config=config,
            templates={"carry": carries, "trace": row},
            resume=resume,
        )
        if got is None:
            return carries, None
        return jax.tree_util.tree_map(jnp.asarray, got.carry), got.trace


def _rows(trace):
    """Row count of a stacked batched trace."""
    return int(jax.tree_util.tree_leaves(trace)[0].shape[1])
