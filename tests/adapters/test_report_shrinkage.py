"""The shrinkage kernel of the report refit on realized fixture visits.

Each test names the property or number it protects. The first tier runs at a small
draw count: the tests check structure (independence, sharing, codes), not accuracy.
"""

import functools

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("photomancy")
pytest.importorskip("photomancy.orbit.refit", reason="needs the report refit")

from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.core.enums import RefitStatus
from spaceodyssey.studies import bracket as br
from tests.adapters.posterior_helpers import realize_visits, visit_request
from tests.adapters.test_report_posterior_refit import COLUMNS, _thin
from tests.studies.bracket_helpers import (
    N_SLOTS,
    N_STEPS,
    N_WORLDS,
    build,
    fixture_source,
)

pytestmark = [
    pytest.mark.slow(reason="each test compiles the shrinkage and first-tier programs"),
    pytest.mark.clear_caches_between_tests,
]

SMALL = dict(n_first=256)
TICK = 250 * 86_400_000
CODES = {ph.SLOT_OK, ph.SLOT_NO_SUPPORT, ph.SLOT_DEGRADED, ph.SLOT_FAILED}


@functools.cache
def _ln_lam():
    return ph.population_ln_lam_range(fixture_source().population)


def _boundary(study, lanes, kernel="shrinkage", **override):
    settings = {"ln_lam_range": _ln_lam(), **SMALL, **override}
    groups = ph.SHRINK_GROUPS if kernel == "shrinkage" else ()
    return ph.ReportRefit(
        kernel=kernel,
        truth_groups=groups,
        occupancy=0.5,
        lanes=lanes,
        caps=study.caps,
        **settings,
    )


def _with_catalog(request, study, channel, truth, stars, *, drop=()):
    """`request` with each star's catalog row (as a study writes it) in the last row."""
    caps = study.caps
    kind = caps.kind_names.index("truth_catalog")
    last = caps.rows_per_subject - 1
    rows = request.rows
    for star in stars:
        if star in drop:
            continue
        row = np.asarray(channel._catalog_row(truth, star, 0, None, None))
        for a in range(2):
            rows.payload[0, a, star, last, : len(row)] = row
            rows.valid[0, a, star, last] = True
            rows.kind[0, a, star, last] = kind
    return request


def _slices(study, result, audience, star, world=0):
    names = [name for name, _, _ in ph.slice_spec(study)]
    pairs = zip(names, result.slices, strict=True)
    return {n: np.asarray(x[world, audience, star]) for n, x in pairs}


@pytest.fixture(scope="module")
def world():
    source, study, *_ = build(run=False, arm="shrinkage", occupancy=0.5, **SMALL)
    channel = study.channel_model
    truth = jax.tree_util.tree_map(lambda x: x[0], source.population)
    days = [30.0 * i for i in range(1, 7)]
    rich = {s: realize_visits(channel, truth, s, 4, days) for s in (0, 1)}
    return study, channel, truth, rich


def _request(world, visits, **kw):
    study, channel, truth, _ = world
    request = visit_request(study, visits, tick=TICK)
    return _with_catalog(request, study, channel, truth, visits, **kw)


def test_an_entry_does_not_depend_on_its_lane_neighbors(world):
    """Property: an entry's slices are bitwise equal alone, in any batch and any order.

    Entry A (star 1: detected and dark slots) is refit at a fixed lane count in three
    requests: first, after another entry, and beside a different neighbor. Fails if a
    lane reads another lane's data or a padding lane is not isolated. The statuses
    follow the shared slot codes.
    """
    study, channel, truth, rich = world
    other = realize_visits(channel, truth, 0, 3, [20.0 * i for i in range(1, 5)])
    stack = lambda *r: jax.tree_util.tree_map(lambda *x: np.concatenate(x), *r)  # noqa: E731
    first = stack(_request(world, {1: rich[1]}), _request(world, {0: rich[0]}))
    moved = stack(_request(world, {0: rich[0]}), _request(world, {1: rich[1]}))
    swapped = stack(_request(world, {1: rich[1]}), _request(world, {0: other}))
    boundary = _boundary(study, 3)
    mine = _slices(study, boundary.refit_batch(study, first), 0, 1)
    assert (mine["n_det"] > 0).any() and (mine["n_det"] == 0).any()
    assert set(np.unique(mine["status"])) <= CODES
    for request, world_index in ((moved, 1), (swapped, 0)):
        got = _slices(study, boundary.refit_batch(study, request), 0, 1, world_index)
        for name, value in mine.items():
            assert np.array_equal(value, got[name], equal_nan=True), name


def test_the_detected_slots_are_the_summary_of_the_shrinkage_cloud(world):
    """Each detected slot equals `summarize` of its cloud, drawn from the folded key.

    The cloud is `shrinkage_cloud` around the catalog elements at the width of the
    slot's detection count (one detection 0.20, three or more 0.025), with the
    entry's key folded with the slot, then the same `summarize` as every kernel.
    The least-detected slot of star 1 keeps only its first detection, so the entry
    holds a slot detected once beside one detected three or more times.
    """
    study, channel, truth, rich = world
    request = _request(world, {1: _first_only(rich[1], _least_detected(rich[1]))})
    result = _boundary(study, 1).refit_batch(study, request)
    got = _slices(study, result, 0, 1)
    elements = np.asarray(channel.truth_elements(truth))[1]
    key = jnp.asarray(request.keys[0, 0, 1], jnp.uint32)
    seen = {int(n) for n in got["n_det"] if n}
    assert 1 in seen and max(seen) >= 3
    for slot, n_det in enumerate(got["n_det"]):
        if n_det == 0:
            continue
        width = ph.shrinkage_width(jnp.asarray(n_det))
        folded = jax.random.fold_in(jax.random.fold_in(key, slot), 1)
        cloud = ph.shrinkage_cloud(elements[slot], folded, ph.SHRINK_DRAWS, width)
        ends = np.asarray(request.rows.end_tick)[0, 0, 1]
        leaves = ph.summarize(
            cloud,
            jnp.zeros(ph.SHRINK_DRAWS),
            1,
            channel,
            TICK,
            last_det_d=max(
                e * channel.tick_days
                for e, flag in zip(ends, _flags(request, slot), strict=True)
                if flag
            ),
        )
        assert got["status"][slot] == ph.SLOT_OK
        assert got["ess"][slot] == ph.SHRINK_DRAWS
        for name in ("p_det", "p_block", "scatter", "pos_mean", "p_class"):
            np.testing.assert_allclose(
                got[name][slot],
                np.asarray(getattr(leaves, name)),
                rtol=1e-8,
                atol=1e-12,
            )
        np.testing.assert_allclose(got["ln_period_q"][slot], leaves.ln_period_q)
    wide = got["scatter"][np.argmax(got["n_det"] == 1)]
    tight = got["scatter"][np.argmax(got["n_det"] >= 3)]
    assert wide > tight > 0


def _least_detected(visits):
    """The detected slot of `visits` with the fewest detections."""
    flags = [COLUMNS.index(f"detected_{s}") for s in range(N_SLOTS)]
    n_det = np.array([sum(row[f] > 0.5 for row, _ in visits) for f in flags])
    return int(np.argmin(np.where(n_det > 0, n_det, np.iinfo(np.int64).max)))


def _first_only(visits, slot):
    """`visits` with only the first detection of `slot` kept; later ones read null."""
    flag = COLUMNS.index(f"detected_{slot}")
    count = COLUMNS.index(f"counts_{slot}")
    kept = False
    out = []
    for row, end in visits:
        row = row.copy()
        if row[flag] > 0.5 and kept:
            row[flag] = 0.0
            row[count] = 0.0
        kept |= row[flag] > 0.5
        out.append((row, end))
    return out


def _flags(request, slot):
    """Which rows of star 1's entry detect `slot` (the visits fill the first rows)."""
    return np.asarray(request.rows.payload)[0, 0, 1, :, slot] > 0.5


def test_an_undetected_slot_takes_the_posterior_first_tier(world):
    """Dark slots and `p_new` are bitwise those of the posterior kernel on one entry."""
    study, _, _, rich = world
    request = _request(world, {0: _thin(rich[0])})
    shrink = _slices(study, _boundary(study, 1).refit_batch(study, request), 0, 0)
    post = _slices(
        study, _boundary(study, 1, "posterior").refit_batch(study, request), 0, 0
    )
    dark = shrink["n_det"] == 0
    assert dark.any() and (shrink["n_det"] > 0).any()
    assert np.any(shrink["p_new"])
    assert np.array_equal(shrink["p_new"], post["p_new"])
    for name in ("p_class", "ln_period_q", "ess", "status"):
        assert np.array_equal(shrink[name][dark], post[name][dark]), name


def test_a_detected_slot_without_a_catalog_row_installs_nothing(world):
    """No catalog row: detected slots are `SLOT_NO_SUPPORT` with zero leaves.

    The undetected slots do not read the catalog, so their slices equal those of the
    same entry with its row present.
    """
    study, _, _, rich = world
    visits = {0: _thin(rich[0])}
    boundary = _boundary(study, 1)
    result = boundary.refit_batch(study, _request(world, visits, drop=(0,)))
    got = _slices(study, result, 0, 0)
    live = got["n_det"] > 0
    assert live.any() and (~live).any()
    assert (got["status"][live] == ph.SLOT_NO_SUPPORT).all()
    assert not got["p_det"][live].any() and not got["ess"][live].any()
    assert int(result.status[0, 0, 0]) != int(RefitStatus.OK)
    full = _slices(study, boundary.refit_batch(study, _request(world, visits)), 0, 0)
    assert (full["status"][live] == ph.SLOT_OK).all()
    assert np.array_equal(got["p_new"], full["p_new"])
    for name in ("p_class", "ln_period_q", "ess", "status"):
        assert np.array_equal(got[name][~live], full[name][~live]), name


def test_the_shrinkage_arm_runs_beside_the_oracle_on_the_fixture():
    """`run_arms` of `oracle` and `shrinkage` completes with the expected columns."""
    arms, truths, roots = {}, None, None
    for name, settings in (("oracle", {}), ("shrinkage", {"occupancy": 0.5, **SMALL})):
        _, study, executor, truths, roots, _ = build(run=False, arm=name, **settings)
        arms[name] = (study, executor.boundary)
    out = br.run_arms(arms, truths, roots, N_STEPS)
    assert set(out.table) == {"oracle", "shrinkage"}
    reference = out.table["oracle"]
    mine = out.table["shrinkage"]
    for part in ("slot", "star", "world"):
        assert set(getattr(mine, part)) == set(getattr(reference, part))
    mean, se, _, n = out.gap("oracle", "shrinkage", "confirmed")
    assert n == N_WORLDS and np.isfinite(mean) and np.isfinite(se)
    assert (np.asarray(mine.world["low_ess"]) == 0).all()
    jax.clear_caches()
