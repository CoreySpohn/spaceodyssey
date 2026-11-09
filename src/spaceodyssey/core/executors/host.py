"""Executors host."""

import typing

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import checkpoint as ckpt
from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.carry import empty_trace, init_carry
from spaceodyssey.core.enums import Audience, EventKind
from spaceodyssey.core.executors.base import AbstractExecutor, RunResult, index_world
from spaceodyssey.core.lift import lift_traced
from spaceodyssey.core.refresh import answer_refresh
from spaceodyssey.core.service import answer_pending
from spaceodyssey.core.snapshot import effects_none, snapshot_view
from spaceodyssey.core.transaction import (
    admit,
    advance,
    close_decision,
    decide,
    decision_fingerprint,
    decision_key,
    finish_trace,
    mask_effects,
    phase_at,
    wake_by_audience,
)

_init = eqx.filter_jit(init_carry)
_adv = eqx.filter_jit(advance)
_dec = eqx.filter_jit(decide)
_adm = eqx.filter_jit(admit)
_close = eqx.filter_jit(close_decision)
_fin = eqx.filter_jit(finish_trace)
_fp = eqx.filter_jit(decision_fingerprint)


class _Saved(typing.NamedTuple):
    """How much of the run the last checkpoint already holds."""

    rows: int
    views: int
    effects: int


class HostExecutor(AbstractExecutor):
    """Run the shared transaction in a host loop."""

    def advance(self, carry, root, study, truth, *, wake_fn=wake_by_audience):
        """Advance one event with the shared handler table."""
        return _adv(carry, root, lift_traced(study), truth, wake_fn=wake_fn)

    def decide(self, carry, root, study):
        """Run the policy of the current phase."""
        return _dec(carry, root, lift_traced(study))

    def run(
        self,
        study,
        truths,
        roots,
        n_steps,
        *,
        policy_form="array",
        wake_fn=wake_by_audience,
        view_log=None,
        effects_log=None,
        service=None,
        refit=None,
        checkpoint=None,
        resume=False,
    ):
        """Python loop over steps; the decision stage may use a Python-form policy.

        With `view_log` a list, the policy view of every decision is appended to
        it; with `effects_log` a list, the effects of every decision (woken
        steps only). With a `service` (see `AbstractHostService`), a request the
        transaction reaches is answered before the next step; without one it stays
        pending and the run reports `RunStatus.AWAITING`. A refresh checkpoint the
        transaction reaches is answered the same way through `refit` (an
        `AbstractRefitBoundary`), in the iteration that held it.

        With a `checkpoint` (a `CheckpointSink`) the run saves after each acquisition
        (a handled `DONE` that wrote a record), after each refit it installs, and at
        its end. `resume` continues from what the sink holds: the held refresh
        checkpoint or request is answered first, the saved trace rows come back in
        front of the new ones, and `n_steps` is a fresh budget for this call. A budget
        that ends with a refresh checkpoint held keeps the hold (no `drop_hold`) and
        the result reports `RunStatus.TRUNCATED_AT_HOLD`. With `resume` false a sink
        that already holds a checkpoint is refused.
        """
        require_x64()
        given = study
        study = lift_traced(study)
        adv = lambda c, r, s, t: self.advance(c, r, s, t, wake_fn=wake_fn)  # noqa: E731
        dec, adm, close, fin = self.decide, _adm, _close, _fin
        carry = _init(study, truths)
        caps = study.caps
        layout = study.observer.layout(caps)
        prefix, base = None, 0
        if checkpoint is not None:
            carry, prefix = self._begin(
                checkpoint,
                (given, truths, roots),
                (study, carry, wake_fn, policy_form, service, refit),
                (view_log, effects_log, resume),
            )
            if prefix is not None:
                base = int(jax.tree_util.tree_leaves(prefix)[0].shape[0])
            if service is not None:
                carry = answer_pending(carry, study, service)
            if refit is not None:
                carry = answer_refresh(
                    carry, roots, study, refit, can_finish=n_steps > 0, keep=True
                )
        elif n_steps == 0:
            trace = jax.tree_util.tree_map(
                lambda x: jnp.empty((0, *x.shape), dtype=x.dtype), empty_trace(caps)
            )
            return RunResult(carry, trace, layout, roots)
        traces = []
        saved = _Saved(0, len(view_log or ()), len(effects_log or ()))
        for step in range(n_steps):
            carry, trace, wake = adv(carry, roots, study, truths)
            if bool(wake):
                if view_log is not None:
                    view_log.append(
                        jax.tree_util.tree_map(
                            np.asarray,
                            eqx.filter_jit(snapshot_view)(carry, Audience.POLICY, caps),
                        )
                    )
                if policy_form == "array":
                    eff, pstate, u = dec(carry, roots, study)
                else:
                    view = eqx.filter_jit(snapshot_view)(carry, Audience.POLICY, caps)
                    view_np = jax.tree_util.tree_map(np.asarray, view)
                    ph = int(phase_at(carry, study))
                    ps = (
                        np.zeros_like(carry.pstate)
                        if ph != int(carry.phase)
                        else np.asarray(carry.pstate)
                    )
                    key = decision_key(carry, roots, study)
                    eff, pstate, u = study.agenda.policies[ph].decide_py(
                        ps, view_np, study, key
                    )
            else:
                eff, pstate, u = effects_none(caps), carry.pstate, jnp.asarray(0.0)
            eff = mask_effects(eff, wake)
            if effects_log is not None and bool(wake):
                effects_log.append(eff)
            fp = _fp(carry, study) if bool(wake) else jnp.asarray(0, jnp.int64)
            carry, codes = adm(carry, eff, study, fp)
            carry = close(carry, eff, pstate, wake, study)
            traces.append(fin(trace, eff, codes, u, wake, carry, study, fp))
            if service is not None:
                carry = answer_pending(carry, study, service)
            boundary = None
            if checkpoint is not None:
                row = traces[-1]
                if (
                    bool(row.handled)
                    and int(row.kind) == EventKind.DONE
                    and int(row.n_written) > 0
                ):
                    boundary = ckpt.AFTER_ACQUISITION
            if refit is not None:
                had = bool(carry.refresh.valid)
                carry = answer_refresh(
                    carry,
                    roots,
                    study,
                    refit,
                    can_finish=step + 1 < n_steps,
                    keep=checkpoint is not None,
                )
                if checkpoint is not None and had and not bool(carry.refresh.valid):
                    boundary = ckpt.AFTER_INFERENCE
            if boundary is not None and checkpoint.wants(boundary):
                saved = self._save(
                    checkpoint,
                    boundary,
                    carry,
                    traces,
                    saved,
                    view_log,
                    effects_log,
                    base,
                )
        held = None
        if checkpoint is not None:
            held = jnp.asarray(refit is not None and bool(carry.refresh.valid))
            end = ckpt.AT_HOLD if bool(held) else ckpt.FINAL
            if len(traces) > saved.rows and checkpoint.wants(end):
                self._save(
                    checkpoint, end, carry, traces, saved, view_log, effects_log, base
                )
        trace = (
            jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *traces)
            if traces
            else None
        )
        if prefix is not None:
            trace = jax.tree_util.tree_map(
                jnp.asarray, ckpt.concat_rows(prefix, trace, 0)
            )
        elif trace is None:
            trace = jax.tree_util.tree_map(
                lambda x: jnp.empty((0, *x.shape), dtype=x.dtype), empty_trace(caps)
            )
        return RunResult(carry, trace, layout, roots, held)

    def _begin(self, checkpoint, ids, parts, logs):
        """Open the sink: the state to continue from, and the saved trace rows."""
        given, truths, roots = ids
        study, carry, wake_fn, policy_form, service, refit = parts
        view_log, effects_log, resume = logs
        caps = study.caps
        templates = {"carry": carry, "trace": empty_trace(caps)}
        if view_log is not None:
            templates["view"] = jax.tree_util.tree_map(
                np.asarray,
                eqx.filter_jit(snapshot_view)(carry, Audience.POLICY, caps),
            )
        if effects_log is not None:
            templates["effects"] = effects_none(caps)
        config = {
            "executor": "host",
            "row_axis": 0,
            "policy_form": policy_form,
            "logs": {"view": view_log is not None, "effects": effects_log is not None},
            "bound": {"wake_fn": wake_fn, "service": service, "refit": refit},
        }
        got = checkpoint.begin(
            given, truths, roots, config=config, templates=templates, resume=resume
        )
        if got is None:
            return carry, None
        if view_log is not None:
            view_log.extend(got.view_rows)
        if effects_log is not None:
            effects_log.extend(got.effect_rows)
        return jax.tree_util.tree_map(jnp.asarray, got.carry), got.trace

    def _save(
        self, checkpoint, boundary, carry, traces, saved, view_log, effects_log, base
    ):
        """Publish the carry and the rows produced since the last save."""
        new = traces[saved.rows :]
        rows = (
            jax.tree_util.tree_map(
                lambda *xs: np.stack([np.asarray(x) for x in xs]), *new
            )
            if new
            else None
        )
        views = tuple(view_log[saved.views :]) if view_log is not None else ()
        effects = tuple(effects_log[saved.effects :]) if effects_log is not None else ()
        checkpoint.save(
            boundary,
            carry,
            rows,
            base + len(traces),
            view_rows=views,
            effect_rows=effects,
        )
        return _Saved(
            len(traces), saved.views + len(views), saved.effects + len(effects)
        )

    def run_worlds(self, study, truths, roots, n_steps, **kw):
        """Run each world of a batch in turn and stack the results on a world axis."""
        if kw.get("checkpoint") is not None:
            raise ValueError(
                "a host checkpoint holds one world: call run once per world, each "
                "with its own sink, or use the compiled executor for a batch"
            )
        runs = [
            self.run(study, index_world(truths, w), roots[w], n_steps, **kw)
            for w in range(roots.shape[0])
        ]
        return RunResult(
            jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[r.carry for r in runs]),
            jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[r.trace for r in runs]),
            study.observer.layout(study.caps),
            roots,
        )
