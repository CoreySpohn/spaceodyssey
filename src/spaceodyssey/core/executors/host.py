"""Executors host."""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.capacities import require_x64
from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.enums import Audience
from spaceodyssey.core.executors.base import AbstractExecutor, RunResult
from spaceodyssey.core.keys import policy_key
from spaceodyssey.core.snapshot import effects_none, snapshot_view
from spaceodyssey.core.transaction import (
    admit,
    advance,
    close_decision,
    decide,
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


class HostExecutor(AbstractExecutor):
    """Run the shared transaction in a host loop."""

    def advance(self, carry, root, study, truth, *, wake_fn=wake_by_audience):
        """Advance one event with the shared handler table."""
        return _adv(carry, root, study, truth, wake_fn=wake_fn)

    def run(
        self,
        study,
        truth,
        root,
        n_steps,
        *,
        policy_form="array",
        wake_fn=wake_by_audience,
        view_log=None,
        effects_log=None,
    ):
        """Python loop over steps; the decision stage may use a Python-form policy.

        With `view_log` a list, the policy view of every decision is appended to
        it; with `effects_log` a list, the effects of every decision (woken
        steps only).
        """
        require_x64()
        adv = lambda c, r, s, t: self.advance(c, r, s, t, wake_fn=wake_fn)  # noqa: E731
        dec, adm, close, fin = _dec, _adm, _close, _fin
        carry = _init(study, truth)
        caps = study.caps
        traces = []
        for _ in range(n_steps):
            carry, trace, wake = adv(carry, root, study, truth)
            if bool(wake):
                if view_log is not None:
                    view_log.append(
                        jax.tree_util.tree_map(
                            np.asarray, snapshot_view(carry, Audience.POLICY, caps)
                        )
                    )
                if policy_form == "array":
                    eff, pstate, u = dec(carry, root, study)
                else:
                    view = snapshot_view(carry, Audience.POLICY, caps)
                    view_np = jax.tree_util.tree_map(np.asarray, view)
                    ph = int(phase_at(carry, study))
                    ps = (
                        np.zeros_like(carry.pstate)
                        if ph != int(carry.phase)
                        else np.asarray(carry.pstate)
                    )
                    key = policy_key(root, carry.n_dec)
                    eff, pstate, u = study.agenda.policies[ph].decide_py(
                        ps, view_np, study, key
                    )
            else:
                eff, pstate, u = effects_none(caps), carry.pstate, jnp.asarray(0.0)
            eff = mask_effects(eff, wake)
            if effects_log is not None and bool(wake):
                effects_log.append(eff)
            carry, codes = adm(carry, eff, study)
            carry = close(carry, eff, pstate, wake, study)
            traces.append(fin(trace, eff, codes, u, wake, carry))
        return RunResult(
            carry, jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *traces)
        )
