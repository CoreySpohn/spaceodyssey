"""Pedigree layout, derived kinds in revision sets and the channel pre-flight check."""

import dataclasses
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.core import Capacities
from spaceodyssey.core.capacities import (
    PEDIGREE_HEAD,
    PEDIGREE_TRIPLE,
    pedigree_width,
)
from spaceodyssey.core.enums import I64, Audience
from spaceodyssey.core.preflight import FindingCode, preflight
from spaceodyssey.core.records import (
    empty_log,
    pedigree_of,
    refit_rows,
    revision_set,
)
from spaceodyssey.core.refresh import pedigree_words
from spaceodyssey.core.snapshot import snapshot_fingerprint
from spaceodyssey.studies import toy
from tests.core.refresh_helpers import oracle_refresh_study, refresh_study


def test_pedigree_width_follows_rows_and_is_zero_without_refresh():
    assert pedigree_width(0) == 0
    assert pedigree_width(8) == PEDIGREE_HEAD + PEDIGREE_TRIPLE * 8
    assert pedigree_width(9) - pedigree_width(8) == PEDIGREE_TRIPLE
    assert Capacities().product_width == 0
    assert refresh_study().caps.product_width == pedigree_width(8)


def test_pedigree_of_reads_the_layout_it_documents():
    caps = refresh_study().caps
    log = empty_log(caps)
    row = np.zeros(caps.product_width, np.int64)
    row[:PEDIGREE_HEAD] = [int(Audience.POLICY), 0, 2]
    row[PEDIGREE_HEAD : PEDIGREE_HEAD + 6] = [10, 3, 0, 10, 4, 1]
    log = eqx.tree_at(lambda x: x.product, log, log.product.at[5].set(jnp.asarray(row)))
    audience, status, triples = pedigree_of(log, 5)
    assert audience == int(Audience.POLICY)
    assert status == 0
    assert triples == ((10, 3, 0), (10, 4, 1))
    # control: an untouched slot names no rows
    assert pedigree_of(log, 4)[2] == ()


def test_derived_row_is_never_in_a_revision_set_or_refit_rows():
    caps = refresh_study().caps
    log = empty_log(caps)
    mk = lambda **kw: dataclasses.replace(log, **kw)  # noqa: E731
    z = lambda v: jnp.asarray([v] + [0] * (caps.n_log - 1), I64)  # noqa: E731
    one = jnp.zeros(caps.n_log, bool).at[0].set(True)
    measurement = toy.RECORD_MEASUREMENT
    seen = mk(valid=one, kind=z(measurement), accepted_tick=z(1), subject=z(0))
    counts, _ = revision_set(seen, caps, Audience.OBSERVER, 0)
    assert bool(counts[0])  # control: the same row as a measurement counts
    product = mk(valid=one, kind=z(toy.PRODUCT_KIND), accepted_tick=z(1), subject=z(0))
    counts, _ = revision_set(product, caps, Audience.OBSERVER, 0)
    assert not bool(counts.any())
    rows = refit_rows(product, caps, Audience.OBSERVER, 0)
    assert not bool(np.asarray(rows.valid).any())


def _with_channels(study, **named):
    schedule = dataclasses.replace(study.refresh, **named)
    return eqx.tree_at(lambda s: s.refresh, study, schedule)


def _k002(study):
    return [f for f in preflight(study) if f.code == FindingCode.K002]


def test_distinct_channels_raise_no_k002():
    assert _k002(refresh_study()) == []
    study, _ = oracle_refresh_study()
    assert _k002(study) == []


def test_product_channel_on_an_admission_channel_is_k002_with_locator():
    study = refresh_study()
    admitted = int(np.asarray(study.channels.id)[0])
    found = _k002(_with_channels(study, observer_channel=admitted))
    assert len(found) == 1
    assert found[0].subject == "refresh"
    assert found[0].member == "observer_channel"
    again = _k002(_with_channels(study, policy_channel=admitted))
    assert [f.member for f in again] == ["policy_channel"]


def test_two_product_channels_sharing_an_id_is_k002():
    study = refresh_study()
    found = _k002(_with_channels(study, policy_channel=study.refresh.observer_channel))
    assert [f.member for f in found] == ["policy_channel"]


def _hand_log(caps):
    """Four counting rows of subject 0, logged against the canonical order.

    Row slots hold (channel, index) as (12, 2), (10, 3), (11, 0), (10, 1). All four are
    accepted by the observer; only the first two are released to the policy.
    """
    log = empty_log(caps)
    pad = lambda v: jnp.asarray(list(v) + [0] * (caps.n_log - len(v)), I64)  # noqa: E731
    on = jnp.arange(caps.n_log) < 4
    return dataclasses.replace(
        log,
        valid=on,
        kind=pad([toy.RECORD_MEASUREMENT] * 4),
        channel=pad([12, 10, 11, 10]),
        acquisition_index=pad([2, 3, 0, 1]),
        acquisition_id=pad([1, 2, 3, 4]),
        accepted_tick=pad([1, 1, 1, 1]),
        released_tick=pad([1, 1, -1, -1]),
        audience=pad([int(Audience.OBSERVER | Audience.POLICY)] * 2 + [1, 1]),
    )


def test_pedigree_triples_are_canonical_and_per_audience():
    caps = refresh_study().caps
    log = _hand_log(caps)
    both = [refit_rows(log, caps, a, 0) for a in (Audience.OBSERVER, Audience.POLICY)]
    k = caps.rows_per_subject  # a request holds the first rows_per_subject
    rows = jax.tree_util.tree_map(lambda *xs: jnp.stack(xs)[:, None, :k], *both)
    request = types.SimpleNamespace(rows=rows)
    got = {}
    for a in (0, 1):
        words = pedigree_words(request, a, 0, 0, caps)
        out = dataclasses.replace(empty_log(caps), product=jnp.zeros_like(log.product))
        out = eqx.tree_at(lambda x: x.product, out, out.product.at[0].set(words))
        got[a] = pedigree_of(out, 0)
    # canonical order by (channel, index, revision), not by arrival or log slot
    assert got[0][2] == ((10, 1, 0), (10, 3, 0), (11, 0, 0), (12, 2, 0))
    # the policy's pedigree names only what it received, never the observer-only rows
    assert got[1][2] == ((10, 3, 0), (12, 2, 0))
    assert got[0][0] != got[1][0]


def _view(caps, product):
    log = dataclasses.replace(
        empty_log(caps), valid=jnp.arange(caps.n_log) < 1, product=product
    )
    return types.SimpleNamespace(
        log=log,
        n_rows=jnp.asarray(1, I64),
        belief=refresh_study().observer.init(caps),
        tick=jnp.asarray(0, I64),
        n_dec=jnp.asarray(0, I64),
        release_version=jnp.asarray(0, I64),
    )


def test_fingerprint_covers_the_pedigree_and_width_zero_is_unchanged():
    study = refresh_study()
    caps = study.caps
    layout = study.observer.layout(caps)
    base = (
        jnp.zeros((caps.n_log, caps.product_width), I64)
        .at[0, :5]
        .set(jnp.asarray([1, 0, 1, 10, 3], I64))
    )
    moved = base.at[0, 4].set(4)  # a triple's acquisition index
    fp = lambda v: int(snapshot_fingerprint(v, layout))  # noqa: E731
    assert fp(_view(caps, base)) == fp(_view(caps, base))
    assert fp(_view(caps, base)) != fp(_view(caps, moved))
    # a width-0 column adds nothing: the digest equals one over the other leaves
    slim = dataclasses.replace(caps, rows_per_subject=0, refit_lanes=0, product_width=0)
    empty = jnp.zeros((slim.n_log, 0), I64)
    # pinned from the digest over the leaves a Slice 1 view always had
    assert fp(_view(slim, empty)) == 5814203023881743964


def test_a_derived_row_never_supersedes_a_source_row():
    from spaceodyssey.core.snapshot import _superseded_in_view

    caps = refresh_study().caps
    log = empty_log(caps)
    pad = lambda v: jnp.asarray(list(v) + [0] * (caps.n_log - len(v)), I64)  # noqa: E731
    mk = lambda **kw: dataclasses.replace(  # noqa: E731
        log,
        valid=jnp.arange(caps.n_log) < 2,
        acquisition_id=pad([7, 7]),
        revision=pad([0, 1]),
        accepted_tick=pad([1, 1]),
        released_tick=pad([1, 1]),
        **kw,
    )
    derived = mk(kind=pad([toy.RECORD_MEASUREMENT, toy.PRODUCT_KIND]))
    counts, replaced = revision_set(derived, caps, Audience.OBSERVER, 0)
    assert bool(counts[0]) and not bool(counts[1])
    assert int(replaced[0]) == -1
    vis = derived.valid
    assert int(_superseded_in_view(derived, vis, caps)[0]) == -1
    # control: the same ids as two revisions of one measurement do replace
    source = mk(kind=pad([toy.RECORD_MEASUREMENT] * 2))
    counts, replaced = revision_set(source, caps, Audience.OBSERVER, 0)
    assert not bool(counts[0]) and bool(counts[1]) and int(replaced[0]) == 1
    assert int(_superseded_in_view(source, vis, caps)[0]) == 1


def test_committed_pedigrees_are_exact_per_audience():
    """Gather, boundary and commit on a hand log: the product column per audience."""
    from spaceodyssey.core import init_carry
    from spaceodyssey.core.events import empty_event_row
    from spaceodyssey.core.refresh import answer_refresh
    from tests.core.helpers import one_world
    from tests.core.refresh_helpers import masked

    study = refresh_study()
    caps = study.caps
    truth, root = one_world()
    carry = init_carry(study, truth)
    carry = dataclasses.replace(
        carry,
        log=_hand_log(caps),
        n_log=jnp.asarray(4, I64),
        next_sid=jnp.asarray(10, I64),
        refresh_dirty=jnp.ones_like(carry.refresh_dirty),
        refresh=dataclasses.replace(
            empty_event_row(), valid=jnp.asarray(True), tick=jnp.asarray(100, I64)
        ),
    )
    out = answer_refresh(carry, root, study, masked())
    # subject 0 writes its observer row then its policy row, after the four source rows
    observer = pedigree_of(out.log, 4)
    policy = pedigree_of(out.log, 5)
    assert observer == (
        int(Audience.OBSERVER),
        0,
        ((10, 1, 0), (10, 3, 0), (11, 0, 0), (12, 2, 0)),
    )
    assert policy == (int(Audience.POLICY), 0, ((10, 3, 0), (12, 2, 0)))
