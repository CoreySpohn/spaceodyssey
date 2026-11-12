"""Rule (d): gradients three ways on the transaction core.

d1  reverse mode through the scan against a central-difference curve, on a frozen
discrete path (`ReplayPolicy`); d2  REINFORCE with a leave-one-out baseline against an
enumerated reference, for a policy whose decisions are random; d3  the pathwise gradient
of a smoothed count against the analytic derivative of the expected hard count.

Worlds run through `lax.map` in chunks of `CHUNK`; the scan body is `jax.checkpoint`ed.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
from jax import lax
from jax.scipy.stats import norm

from spaceodyssey import core
from spaceodyssey.core.enums import F64
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import toy as kinds
from tests.studies.scenarios import toy_gradient

D1_H = tuple(10.0 ** (-i) for i in range(1, 9))
N_STEPS = 80
PLAN_D3 = tuple((0, 10) for _ in range(5))
N_FORCED = 6


def final_carry(study, truth, root, n_steps=N_STEPS):
    """The carry after `n_steps` transitions of one world, with a checkpointed body."""
    body = jax.checkpoint(
        lambda c, i: (core.transition(c, i, root, study, truth)[0], None)
    )
    c, _ = lax.scan(body, core.init_carry(study, truth), jnp.arange(n_steps))
    return c


def _map_worlds(fn, truths, roots, chunk=1024):
    n = roots.shape[0]
    return lax.map(lambda a: fn(*a), (truths, roots), batch_size=min(chunk, n))


@eqx.filter_jit
def _soft_values(study, truths, roots, width, n_steps):
    def one(truth, root):
        return toy_gradient.credit_soft(
            final_carry(study, truth, root, n_steps), study, width
        )

    return _map_worlds(one, truths, roots)


@eqx.filter_jit
def _gap_values(study, truths, roots, width, n_steps):
    """Sum over subjects of |soft term - hard term|.

    Sum over subjects of |soft term - hard term|: it bounds |soft - hard| of the
    totals,

    and is monotone in the width term by term (a signed total can cancel across
    subjects).
    """

    def one(truth, root):
        snr = kinds.carry_snr(final_carry(study, truth, root, n_steps), study)
        x = snr - study.objective.threshold
        return jnp.sum(jnp.abs(jax.nn.sigmoid(x / width) - (x > 0)))

    return _map_worlds(one, truths, roots)


@eqx.filter_jit
def _hard_values(study, truths, roots, n_steps):
    def one(truth, root):
        c = final_carry(study, truth, root, n_steps)
        return kinds.credit_hard(c, study).astype(F64), c.done, c.pstate[0]

    return _map_worlds(one, truths, roots)


@eqx.filter_jit
def _mean_soft_and_grad(dyn, static, study, truths, roots, width, n_steps):
    def loss(d):
        s = eqx.tree_at(lambda x: x.design, study, eqx.combine(d, static))
        return jnp.mean(_soft_values(s, truths, roots, width, n_steps))

    return eqx.filter_value_and_grad(loss)(dyn)


@eqx.filter_jit
def _per_world_grad(dyn, static, study, truths, roots, width, n_steps):
    def one(truth, root):
        def loss(d):
            s = eqx.tree_at(lambda x: x.design, study, eqx.combine(d, static))
            return toy_gradient.credit_soft(
                final_carry(s, truth, root, n_steps), s, width
            )

        return eqx.filter_grad(loss)(dyn).gain

    return _map_worlds(one, truths, roots)


def paired_z(est_w, ref_w):
    """Handle a transaction state update.

    z of the paired difference, its standard error, and the resolving power `se / |mean
    reference|` (a statistic of the test, not a probability).
    """
    est_w, ref_w = np.asarray(est_w, float), np.asarray(ref_w, float)
    d = est_w - ref_w
    n = d.shape[0]
    se = float(np.std(d, ddof=1) / np.sqrt(n))
    mean_d = float(np.mean(d))
    z = mean_d / se if se > 0 else (0.0 if mean_d == 0 else float("inf"))
    ref = abs(float(np.mean(ref_w)))
    power = se / ref if ref > 0 else float("inf")
    return z, se, power


def record_effects(study, truth, root, n_steps=N_STEPS):
    """The effects of every decision of a host-loop run, stacked into one table."""
    log = []
    HostExecutor().run(study, truth, root, n_steps, effects_log=log)
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *log)


def replay_study(study, truth, root, n_steps=N_STEPS):
    """`study` with its phase-0 policy replaced by a replay of its own recorded run."""
    table = record_effects(study, truth, root, n_steps)
    return eqx.tree_at(
        lambda s: s.agenda.policies, study, (toy_gradient.ReplayPolicy(effects=table),)
    )


# ---- d1 -----------------------------------------------------------------------


def d1_reverse_and_fd(
    study,
    truths,
    roots,
    hs=D1_H,
    *,
    width=0.1,
    n_steps=60,
    grad_scale=1.0,
):
    """Reverse mode against central differences on the frozen discrete path.

    The objective is the world mean of `credit_soft(width)`. `rel_err[h]` is `|grad
    - fd[h]| / |grad|`; `reference` is the difference quotient at the h where that
    curve is smallest; `slope` is the least-squares slope of log10 rel_err on log10
    h over h = 1e-1 to 1e-4, nan when fewer than two h fall in that range.
    """
    width = jnp.asarray(width, F64)  # an array, so a new width does not recompile
    truth0 = jax.tree_util.tree_map(lambda a: a[0], truths)
    study = replay_study(study, truth0, roots[0], n_steps)
    dyn, static = eqx.partition(study.design, eqx.is_inexact_array)
    payload, g = _mean_soft_and_grad(dyn, static, study, truths, roots, width, n_steps)
    grad = float(g.gain) * grad_scale
    gain0 = float(study.design.gain)

    def at(gain):
        d = jax.tree_util.tree_map(lambda a: jnp.asarray(gain, F64), dyn)
        return float(
            _mean_soft_and_grad(d, static, study, truths, roots, width, n_steps)[0]
        )

    fd, rel_err = {}, {}
    for h in hs:
        fd[h] = (at(gain0 + h) - at(gain0 - h)) / (2.0 * h)
        rel_err[h] = abs(grad - fd[h]) / abs(grad) if grad != 0 else float("inf")
    h_min = min(rel_err, key=rel_err.get)
    fit = [h for h in hs if 1e-4 * 0.99 <= h <= 0.1 * 1.01]
    if len(fit) < 2:
        slope = float("nan")
    else:
        slope = float(
            np.polyfit(
                np.log10(fit), np.log10([max(rel_err[h], 1e-300) for h in fit]), 1
            )[0]
        )
    return dict(
        payload=float(payload),
        grad=grad,
        fd=fd,
        rel_err=rel_err,
        slope=slope,
        h_min=h_min,
        reference=fd[h_min],
    )


# ---- d2 -----------------------------------------------------------------------


def _forced_credit(study, truths, roots, m, n_steps):
    s = eqx.tree_at(
        lambda x: x.agenda.policies,
        study,
        (
            toy_gradient.ReplayPolicy(
                effects=toy_gradient.forced_effects(study.caps, study, m)
            ),
        ),
    )
    return _hard_values(s, truths, roots, n_steps)[0]


def d2_reinforce_and_enumerated(study, truths, roots, n_steps=N_STEPS):
    """Handle a transaction state update.

    REINFORCE with a leave-one-out baseline across worlds against the enumerated per-
    world gradient `sum_m f_w(m) pmf(m) (m - n p)`; `f_w(m)` comes from replayed runs
    that force m observes.
    """
    pol = study.agenda.policies[0]
    n_obs = pol.n_obs
    p = float(jax.nn.sigmoid(pol.theta))
    credit, done, m_w = _hard_values(study, truths, roots, n_steps)
    credit, m_w = np.asarray(credit), np.asarray(m_w)
    n = credit.shape[0]
    b_loo = (credit.sum() - credit) / (n - 1)
    score = m_w - n_obs * p
    est_w = (credit - b_loo) * score
    ms = np.arange(n_obs + 1)
    from scipy.stats import binom

    pmf = binom.pmf(ms, n_obs, p)
    f = np.stack(
        [np.asarray(_forced_credit(study, truths, roots, int(m), n_steps)) for m in ms]
    )
    ref_w = (f * (pmf * (ms - n_obs * p))[:, None]).sum(axis=0)
    n_flip = int(np.sum(np.ptp(f, axis=0) > 0))
    z, se, power = paired_z(est_w, ref_w)
    return dict(
        estimate=float(est_w.mean()),
        reference=float(ref_w.mean()),
        se_paired=se,
        z=z,
        power=power,
        n_flip=n_flip,
        all_done=bool(np.all(np.asarray(done))),
        m_mean=float(m_w.mean()),
        n_worlds=int(n),
    )


def d2_exact_two_world(p, f):
    """Exact expectation of the REINFORCE estimator for a one-decision, two-world toy:

    each world draws m in {0, 1} with probability (1 - p, p), has outcomes
    `f[w][m]`, and uses the leave-one-out baseline of the other world's drawn
    outcome. Returns (expected estimator, true gradient) with the gradient taken of
    the mean over worlds.
    """
    score = (-p, 1.0 - p)
    pmf = (1.0 - p, p)
    est = 0.0
    for m0 in (0, 1):
        for m1 in (0, 1):
            w = pmf[m0] * pmf[m1]
            e0 = (f[0][m0] - f[1][m1]) * score[m0]
            e1 = (f[1][m1] - f[0][m0]) * score[m1]
            est += w * 0.5 * (e0 + e1)
    truth = 0.5 * sum((f[w][1] - f[w][0]) * p * (1.0 - p) for w in (0, 1))
    return est, truth


# ---- d3 -----------------------------------------------------------------------


def d3_study(study):
    """`study` with the frozen five-acquisition_index plan on subject 0."""
    return eqx.tree_at(lambda s: s.agenda.policies, study, (_frozen_plan(PLAN_D3),))


def _frozen_plan(plan):
    return kinds.FixedPlanPolicy(plan=plan)


def d3_pathwise_and_analytic(
    study, truths, roots, widths=(0.5, 0.2, 0.1, 0.05), n_steps=60
):
    """Handle a transaction state update.

    Pathwise `d credit_soft(width) / d gain` per world against `pdf(thr - mu) * signal *
    sqrt(5) / noise`, `mu = gain * signal * sqrt(5) / noise`, which is the derivative of
    the expected hard count. `n_flip` is the number of worlds within three standard
    deviations of the threshold (where the hard count can flip).
    """
    study = d3_study(study)
    dyn, static = eqx.partition(study.design, eqx.is_inexact_array)
    sig = np.asarray(truths.signal[:, 0])
    noise = float(study.channel_model.noise[0])
    gain = float(study.design.gain)
    thr_ = float(study.objective.threshold)
    mu = gain * sig * np.sqrt(5.0) / noise
    ref_w = np.asarray(norm.pdf(thr_ - mu)) * sig * np.sqrt(5.0) / noise
    done = _hard_values(study, truths, roots, n_steps)[1]
    n_flip = int(np.sum(np.abs(mu - thr_) < 3.0))
    out, gaps = {}, []
    for w in widths:
        est_w = np.asarray(
            _per_world_grad(
                dyn, static, study, truths, roots, jnp.asarray(w, F64), n_steps
            )
        )
        gaps.append(
            np.asarray(_gap_values(study, truths, roots, jnp.asarray(w, F64), n_steps))
        )
        z, se, power = paired_z(est_w, ref_w)
        out[w] = dict(
            estimate=float(est_w.mean()),
            reference=float(ref_w.mean()),
            se_paired=se,
            z=z,
            power=power,
            n_flip=n_flip,
        )
    return dict(
        by_width=out,
        gaps=np.stack(gaps),
        widths=tuple(widths),
        all_done=bool(np.all(np.asarray(done))),
        n_worlds=int(roots.shape[0]),
    )
