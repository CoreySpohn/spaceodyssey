"""Shared pieces of the posterior-kernel tests: visit rows realized from the fixture.

A request holds the realized `visit_report` rows of chosen stars, entered for both
audiences, so a test runs the posterior boundary on rows a study would hold.
"""

import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core.kinds import RealizeContext
from spaceodyssey.core.records import record_none
from spaceodyssey.core.refresh import RefitRequest


def visit_context(channel, start_d, mode):
    """The realization context of a visit of `mode` that starts on day `start_d`."""
    built = channel.assets()
    per_day = int(built.context.clock.ticks_per_day)
    settle = int(built.context.settle_ticks)
    start = round(start_d * per_day)
    end = start + settle + round(channel.exposures_s[mode] * 1000)
    zero = jnp.asarray(0)
    return RealizeContext(
        jnp.asarray(start), jnp.asarray(end), zero, jnp.asarray(1.0), zero, zero
    )


def realize_visits(channel, truth, star, mode, days, seed=0):
    """`[(row, end_tick)]` of one star's visits on `days`, one noise key per visit."""
    out = []
    for i, day in enumerate(days):
        ctx = visit_context(channel, day, mode)
        key = jax.random.fold_in(jax.random.key(seed), i)
        row = channel.realize(truth, star, mode, 1.0, key, None, None, context=ctx)
        out.append((np.asarray(row), int(ctx.end_tick)))
    return out


def visit_request(study, visits, *, tick, seed=0, mask_policy=False):
    """A one-world request from `visits[star] = [(row, end_tick)]`, every star dirty.

    Entry `[0, 0, s]` is the observer's and `[0, 1, s]` the policy's. With
    `mask_policy` the policy's payload reads zero, as a column hidden from it does.
    """
    caps = study.caps
    n_subjects = caps.n_subjects
    kind = caps.kind_names.index("visit_report")
    base = jax.tree_util.tree_map(np.asarray, record_none(caps))
    shape = (1, 2, n_subjects, caps.rows_per_subject)
    rows = jax.tree_util.tree_map(
        lambda x: np.broadcast_to(x, shape + x.shape).copy(), base
    )
    for star, seen in visits.items():
        for j, (row, end) in enumerate(seen):
            for a in range(2):
                rows.payload[0, a, star, j, : len(row)] = (
                    0.0 if (mask_policy and a == 1) else row
                )
                rows.valid[0, a, star, j] = True
                rows.kind[0, a, star, j] = kind
                rows.end_tick[0, a, star, j] = end
    rng = np.random.default_rng(seed)
    own = np.arange(2 * n_subjects).reshape(2, n_subjects)
    dirty = np.zeros((1, 2, n_subjects), bool)
    for star in visits:
        dirty[0, :, star] = True
    return RefitRequest(
        tick=np.full(1, tick, np.int64),
        ordinal=np.zeros((1, 2, n_subjects), np.int64),
        dirty=dirty,
        n_rows=np.full((1, 2, n_subjects), caps.rows_per_subject, np.int64),
        rows=rows,
        row_keys=np.zeros((1, 2, n_subjects, caps.rows_per_subject, 2), np.uint32),
        keys=rng.integers(0, 2**31, (1, 2, n_subjects, 2)).astype(np.uint32),
        source=own[None].copy(),
    )
