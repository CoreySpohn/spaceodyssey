"""The linear-memory refit gather equals the pairwise reference it replaced."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import Capacities
from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.enums import I64, Audience, B
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.keys import fold_keys, refit_key
from spaceodyssey.core.records import (
    derived_rows,
    empty_log,
    log_records,
    masked_record,
    revision_set,
    revision_set_all,
)
from spaceodyssey.core.refresh import (
    AUDIENCES,
    RefitRequest,
    _bits,
    _canonical,
    gather_refit_rows,
    revision_set_digest,
)
from spaceodyssey.core.snapshot import _superseded_in_view, visible_rows
from spaceodyssey.studies import toy
from tests.core.helpers import worlds

N_SUBJECTS, N_LOG, N_ROWS = 6, 96, 4


# ---- the reference: the pairwise implementation this change replaced ---------------


def _old_revision_set(log, caps, audience, subject):
    seen = log.accepted_tick if audience == Audience.OBSERVER else log.released_tick
    mine = log.valid & (seen >= 0) & (log.subject == subject)
    mine = mine & ~derived_rows(log, caps)
    same = (log.acquisition_id[:, None] == log.acquisition_id[None, :]) & mine[None, :]
    later = same & (log.revision[None, :] > log.revision[:, None])
    newer = later.any(axis=1)
    replaced = jnp.max(jnp.where(later, log.revision[None, :], -1), axis=1)
    kinds = jnp.asarray(caps.refit_kinds, I64)
    statuses = jnp.asarray(caps.refit_statuses, I64)
    counts = mine & ~newer & jnp.isin(log.kind, kinds) & jnp.isin(log.status, statuses)
    return counts, replaced


def _old_refit_rows(log, caps, audience, subject):
    counts, replaced = _old_revision_set(log, caps, audience, subject)
    last = (~counts).astype(I64)
    order = jnp.lexsort((log.revision, log.acquisition_index, log.channel, last))
    rows = log_records(
        jax.tree_util.tree_map(lambda x: x[order], log),
        valid=counts[order],
        superseded_by=replaced[order],
    )
    return masked_record(rows, caps, audience)


def _cut(x, n):
    if n <= x.shape[0]:
        return x[:n]
    return jnp.pad(x, [(0, n - x.shape[0])] + [(0, 0)] * (x.ndim - 1))


def _old_entry_rows(log, caps, audience, subject, n):
    rows = _old_refit_rows(log, caps, audience, subject)
    count = rows.valid.sum().astype(I64)
    return jax.tree_util.tree_map(lambda x: _cut(x, n), _canonical(rows)), count


def _old_gather(carry, root, study):
    caps = study.caps
    n, count = caps.rows_per_subject, caps.n_subjects
    subjects = jnp.arange(count)
    per = [
        jax.vmap(lambda s, a=a: _old_entry_rows(carry.log, caps, a, s, n))(subjects)
        for a in AUDIENCES
    ]
    rows = jax.tree_util.tree_map(lambda x, y: jnp.stack([x, y]), per[0][0], per[1][0])
    n_rows = jnp.stack([per[0][1], per[1][1]])
    row_keys = jax.vmap(jax.vmap(lambda r: fold_keys(root, r, study)))(rows)
    digest = jax.vmap(jax.vmap(revision_set_digest))(rows)
    keys = jax.vmap(
        lambda d: jax.vmap(lambda dig, s: refit_key(root, s, dig))(d, subjects)
    )(digest)
    same = jnp.ones(count, B)
    for leaf in jax.tree_util.tree_leaves(rows):
        bits = _bits(leaf)
        same = same & (bits[0] == bits[1]).reshape(count, -1).all(axis=1)
    dirty = carry.refresh_dirty
    share = same & dirty[0] & dirty[1] & (keys[0] == keys[1]).all(axis=-1)
    source = jnp.stack([subjects, jnp.where(share, subjects, count + subjects)])
    return RefitRequest(
        tick=carry.refresh.tick,
        ordinal=carry.refresh.acquisition_index,
        dirty=dirty,
        n_rows=n_rows,
        rows=rows,
        row_keys=row_keys,
        keys=keys,
        source=source.astype(I64),
    )


# ---- randomized logs ---------------------------------------------------------------


def _study():
    caps = Capacities(
        n_subjects=N_SUBJECTS, n_log=N_LOG, rows_per_subject=N_ROWS, refit_lanes=2
    )
    return toy.survey_refresh(100, caps=caps)


def _random_log(caps, seed):
    """Corrections, repeated acquisition ids, one-audience revisions, invalid and
    derived rows, subjects outside the study and tied canonical keys."""
    rng = np.random.default_rng(seed)
    n = caps.n_log
    fill = int(rng.integers(n // 4, n + 1))
    pick = lambda hi, lo=0: jnp.asarray(rng.integers(lo, hi, n), I64)  # noqa: E731
    seen = lambda: jnp.where(  # noqa: E731
        jnp.asarray(rng.random(n) < 0.6), pick(50), jnp.asarray(-1, I64)
    )
    log = empty_log(caps)
    # mostly refit kinds and statuses, and a few subjects per seed, so sets are large
    kind = jnp.where(jnp.asarray(rng.random(n) < 0.7), 1, pick(4))
    status = jnp.where(jnp.asarray(rng.random(n) < 0.75), 2 * pick(2), pick(3))
    spread = int(rng.choice((2, 4, caps.n_subjects + 2)))
    return dataclasses.replace(
        log,
        valid=jnp.asarray((np.arange(n) < fill) & (rng.random(n) < 0.9)),
        kind=kind,
        subject=pick(spread, -1),
        channel=pick(3),
        acquisition_index=pick(3),
        acquisition_id=pick(n // int(rng.choice((2, 4)))),
        revision=pick(4),
        status=status,
        accepted_tick=seen(),
        released_tick=seen(),
        payload=jnp.asarray(rng.normal(size=log.payload.shape)),
        cost=jnp.asarray(rng.normal(size=log.cost.shape)),
    )


def _same_bits(a, b):
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    for x, y in zip(
        jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b), strict=True
    ):
        x, y = np.asarray(x), np.asarray(y)
        assert x.dtype == y.dtype and x.shape == y.shape
        assert x.tobytes() == y.tobytes()


def test_gather_refit_rows_equals_the_pairwise_reference():
    """Fails if the one-pass gather picks, orders, cuts, masks or counts the rows of
    a subject differently from the per-subject pairwise path: a correction, a
    repeated id, an observer-only revision, an invalid or derived row, a tie in the
    canonical key, or a subject whose set exceeds `rows_per_subject`."""
    study = _study()
    truths, roots = worlds(1, study.caps)
    carry = init_carry(study, index_world(truths, 0))
    new = eqx.filter_jit(gather_refit_rows)
    old = eqx.filter_jit(_old_gather)
    truncated, counted = 0, 0
    for seed in range(12):
        log = _random_log(study.caps, seed)
        carry = eqx.tree_at(lambda c: c.log, carry, log)
        a, b = new(carry, roots[0], study), old(carry, roots[0], study)
        _same_bits(a, b)
        truncated += int((np.asarray(a.n_rows) > N_ROWS).any())
        counted += int(np.asarray(a.n_rows).sum())
    # controls: the sweep reached truncation and non-empty sets
    assert truncated >= 3 and counted > 100


def test_revision_set_equals_the_pairwise_reference():
    """Fails if `revision_set` or `revision_set_all` differs from the pairwise rule:
    rows outside the subject still take `replaced` from its rows of the same id, strict
    `>` on revision, observer reads accepted_tick and policy released_tick."""
    caps = _study().caps
    old = jax.jit(_old_revision_set, static_argnums=(1, 2))
    new = jax.jit(revision_set, static_argnums=(1, 2))
    every = jax.jit(revision_set_all, static_argnums=(1, 2, 3))
    for seed in range(8):
        log = _random_log(caps, 100 + seed)
        for audience in AUDIENCES:
            counts_all, replaced_all = every(log, caps, audience, caps.n_subjects)
            for s in range(caps.n_subjects):
                want = old(log, caps, audience, s)
                _same_bits(new(log, caps, audience, s), want)
                mine = np.asarray(log.subject) == s
                assert (np.asarray(counts_all)[mine] == np.asarray(want[0])[mine]).all()
                assert (
                    np.asarray(replaced_all)[mine] == np.asarray(want[1])[mine]
                ).all()


def _old_superseded_in_view(log, vis, caps):
    vis = vis & ~derived_rows(log, caps)
    same = (log.acquisition_id[:, None] == log.acquisition_id[None, :]) & vis[None, :]
    newer = same & (log.revision[None, :] > log.revision[:, None])
    out = jnp.max(jnp.where(newer, log.revision[None, :], -1), axis=1)
    return jnp.where(derived_rows(log, caps), -1, out)


def test_superseded_in_view_equals_the_pairwise_reference():
    """Fails if the sorted `superseded_by` of a snapshot view differs from the
    pairwise rule: a derived row is -1, a row the audience cannot see replaces
    nothing, and a hidden row still takes the latest visible revision of its id."""
    caps = _study().caps
    old = jax.jit(_old_superseded_in_view, static_argnums=(2,))
    new = jax.jit(_superseded_in_view, static_argnums=(2,))
    for seed in range(8):
        log = _random_log(caps, 200 + seed)
        log = dataclasses.replace(
            log, audience=jnp.asarray(np.random.default_rng(seed).integers(1, 4, N_LOG))
        )
        for audience in AUDIENCES:
            vis = visible_rows(log, audience)
            _same_bits(new(log, vis, caps), old(log, vis, caps))
