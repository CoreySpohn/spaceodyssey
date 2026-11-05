"""Executors host."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.carry import empty_trace, init_carry
from spaceodyssey.core.enums import Audience
from spaceodyssey.core.executors.base import AbstractExecutor, RunResult, index_world
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


class HostExecutor(AbstractExecutor):
    """Run the shared transaction in a host loop."""

    def advance(self, carry, root, study, truth, *, wake_fn=wake_by_audience):
        """Advance one event with the shared handler table."""
        return _adv(carry, root, study, truth, wake_fn=wake_fn)

    def decide(self, carry, root, study):
        """Run the policy of the current phase."""
        return _dec(carry, root, study)

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
    ):
        """Python loop over steps; the decision stage may use a Python-form policy.

        With `view_log` a list, the policy view of every decision is appended to
        it; with `effects_log` a list, the effects of every decision (woken
        steps only). With a `service` (see `AbstractHostService`), a request the
        transaction reaches is answered before the next step; without one it stays
        pending and the run reports `RunStatus.AWAITING`.
        """
        require_x64()
        adv = lambda c, r, s, t: self.advance(c, r, s, t, wake_fn=wake_fn)  # noqa: E731
        dec, adm, close, fin = self.decide, _adm, _close, _fin
        carry = _init(study, truths)
        caps = study.caps
        if n_steps == 0:
            trace = jax.tree_util.tree_map(
                lambda x: jnp.empty((0, *x.shape), dtype=x.dtype), empty_trace(caps)
            )
            return RunResult(carry, trace)
        traces = []
        for _ in range(n_steps):
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
            traces.append(fin(trace, eff, codes, u, wake, carry, fp))
            if service is not None:
                carry = answer_pending(carry, study, service)
        return RunResult(
            carry, jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *traces)
        )

    def run_worlds(self, study, truths, roots, n_steps, **kw):
        """Run each world of a batch in turn and stack the results on a world axis."""
        runs = [
            self.run(study, index_world(truths, w), roots[w], n_steps, **kw)
            for w in range(roots.shape[0])
        ]
        return RunResult(
            jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[r.carry for r in runs]),
            jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *[r.trace for r in runs]),
        )
