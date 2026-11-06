"""Pedigree layout, derived kinds in revision sets and the channel pre-flight check."""

import dataclasses

import equinox as eqx
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
