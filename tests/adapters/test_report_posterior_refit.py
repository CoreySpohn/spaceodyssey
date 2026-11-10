"""The posterior kernel of the report refit on realized fixture visits.

Each test names the property or number it protects. The fits use small proposal
counts: they check structure (independence, masks, statuses), not accuracy.
"""

import functools

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Rearth2AU

pytest.importorskip("photomancy")
pytest.importorskip("photomancy.orbit.refit", reason="needs the report refit")

from photomancy.orbit.updater import UpdaterStatus

from spaceodyssey.adapters import jaxedith as jx
from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.core.enums import RefitStatus
from spaceodyssey.studies import bracket as br
from tests.adapters.posterior_helpers import realize_visits, visit_request
from tests.studies.bracket_helpers import (
    N_SLOTS,
    build,
    fixture_source,
    prior_summary,
)

pytestmark = [
    pytest.mark.slow(reason="each test compiles the posterior fits"),
    pytest.mark.clear_caches_between_tests,
]

SMALL = dict(
    n_round=2**11,
    n_rounds=2,
    n_centers=64,
    n_constants=2,
    max_rounds=2,
    n_first=256,
)
TICK = 250 * 86_400_000
COLUMNS = jx.visit_columns(N_SLOTS)


def _thin(visits):
    """Keep only the first detection of the visits; every other row reads as a null."""
    flags = [COLUMNS.index(f"detected_{s}") for s in range(N_SLOTS)]
    counts = [COLUMNS.index(f"counts_{s}") for s in range(N_SLOTS)]
    kept = False
    out = []
    for row, end in visits:
        row = row.copy()
        for flag, count in zip(flags, counts, strict=True):
            if row[flag] > 0.5 and not kept:
                kept = True
            else:
                row[flag] = 0.0
                row[count] = 0.0
        out.append((row, end))
    return out


@pytest.fixture(scope="module")
def world():
    source, study, _, _, _, _ = build(run=False, arm="replan", occupancy=0.5, **SMALL)
    channel = study.channel_model
    truth = jax.tree_util.tree_map(lambda x: x[0], source.population)
    star0 = _thin(realize_visits(channel, truth, 0, 4, [30.0 * i for i in range(1, 7)]))
    star1 = realize_visits(channel, truth, 1, 4, [30.0 * i for i in range(1, 5)])
    star1 = [(np.where(np.arange(len(r)) < N_SLOTS, 0.0, r), e) for r, e in star1]
    return study, channel, star0, star1


@functools.cache
def _ln_lam():
    """The scale prior of the fixture's population."""
    return ph.population_ln_lam_range(fixture_source().population)


def _boundary(study, lanes, **override):
    settings = {"ln_lam_range": _ln_lam(), **SMALL, **override}
    return ph.ReportRefit(
        kernel="posterior",
        occupancy=0.5,
        lanes=lanes,
        caps=study.caps,
        **settings,
    )


def _slices(study, result, audience, star):
    names = [name for name, _, _ in ph.slice_spec(study)]
    pairs = zip(names, result.slices, strict=True)
    return {n: np.asarray(x[0, audience, star]) for n, x in pairs}


def _stack_worlds(*requests):
    """One request whose worlds are the single worlds of `requests`, in order."""
    return jax.tree_util.tree_map(lambda *x: np.concatenate(x), *requests)


def _equal_slices(a, b, bitwise):
    """Largest relative gap between two entries' slices (zero when `bitwise`)."""
    worst = 0.0
    for name in a:
        x, y = a[name].astype(float), b[name].astype(float)
        if bitwise:
            assert np.array_equal(x, y, equal_nan=True), name
        elif x.size:
            both = np.isfinite(x) & np.isfinite(y)
            assert np.array_equal(np.isfinite(x), np.isfinite(y)), name
            scale = max(1.0, float(np.max(np.abs(x[both])))) if both.any() else 1.0
            gap = float(np.max(np.abs(x[both] - y[both]))) / scale if both.any() else 0
            worst = max(worst, gap)
    return worst


def test_an_entry_does_not_depend_on_its_lane_neighbors(world):
    """The single, collapsed and first-tier fits of one entry do not see their lanes.

    Entry A (star 0 with the full detections, so its slots run through the single
    and the collapsed engines and the first tier) is fit at a fixed lane count in
    three requests: first, after another entry, and beside a different neighbor. Its
    slices are bitwise equal in all three. Across lane counts (1 and 3) they agree to
    1e-11 (relative; 3.3e-12 measured, the compiled programs reassociate sums). Fails
    if a lane reads another lane's data, or if a padding lane is not isolated.
    """
    study, channel, _, _ = world
    truth = jax.tree_util.tree_map(lambda x: x[0], fixture_source().population)
    days = [30.0 * i for i in range(1, 7)]
    rich = {s: realize_visits(channel, truth, s, 4, days) for s in (0, 1)}
    other = realize_visits(channel, truth, 1, 3, [20.0 * i for i in range(1, 5)])
    names = [name for name, _, _ in ph.slice_spec(study)]

    def request(star, visits):
        return visit_request(study, {star: visits}, tick=TICK)

    first = _stack_worlds(request(0, rich[0]), request(1, rich[1]))
    moved = _stack_worlds(request(1, rich[1]), request(0, rich[0]))
    swapped = _stack_worlds(request(0, rich[0]), request(1, other))
    boundary = _boundary(study, 3, ess_floor=1.0)
    one = boundary.refit_batch(study, first)
    mine = {n: np.asarray(x[0, 0, 0]) for n, x in zip(names, one.slices, strict=True)}
    assert (mine["n_det"] > 0).any() and (mine["n_det"] == 0).any()
    assert (mine["n_det"] >= 2).any(), "no slot reaches the collapsed engine"
    got = boundary.refit_batch(study, moved)
    other_world = {
        n: np.asarray(x[1, 0, 0]) for n, x in zip(names, got.slices, strict=True)
    }
    _equal_slices(mine, other_world, bitwise=True)
    got = boundary.refit_batch(study, swapped)
    beside = {n: np.asarray(x[0, 0, 0]) for n, x in zip(names, got.slices, strict=True)}
    _equal_slices(mine, beside, bitwise=True)
    alone = _boundary(study, 1, ess_floor=1.0).refit_batch(study, first)
    lone = {n: np.asarray(x[0, 0, 0]) for n, x in zip(names, alone.slices, strict=True)}
    assert _equal_slices(mine, lone, bitwise=False) < 1e-11


def test_an_audience_that_cannot_read_the_columns_installs_nothing(world):
    """A policy blind to the measurements gets NO_SUPPORT and all-zero slices."""
    _, study, *_ = build(run=False, arm="prior_planned", occupancy=0.5, **SMALL)
    channel, star0 = world[1], world[2]
    boundary = _boundary(study, 2)
    assert list(boundary._readable(study, channel)) == [True, False]
    request = visit_request(study, {0: star0}, tick=TICK, mask_policy=True)
    result = boundary.refit_batch(study, request)
    assert int(result.status[0, 1, 0]) == int(RefitStatus.NO_SUPPORT)
    assert int(result.status[0, 0, 0]) in (
        int(RefitStatus.OK),
        int(RefitStatus.DEGRADED),
    )
    assert np.any(_slices(study, result, 0, 0)["p_new"])
    for name, value in _slices(study, result, 1, 0).items():
        assert not np.any(value), name


def test_a_low_ess_fit_is_retried_then_installed_as_degraded(world):
    """An unreachable ESS floor: the entry is DEGRADED, retried once and counted."""
    study, _, star0, _ = world
    request = visit_request(study, {0: star0}, tick=TICK)
    result = _boundary(study, 2, ess_floor=1e12).refit_batch(study, request)
    assert int(result.status[0, 0, 0]) == int(RefitStatus.DEGRADED)
    n_reports, n_retried, n_degraded, n_failed = (
        float(x) for x in result.diagnostics[0, 0, 0]
    )
    assert n_reports == len(star0)
    assert n_retried >= 1 and n_degraded >= 1 and n_failed == 0
    status = _slices(study, result, 0, 0)["status"]
    assert (status == ph.SLOT_DEGRADED).any()
    assert not np.isin(status, ph.SLOT_KEPT).any()


def test_a_slot_never_fit_has_a_status_of_its_own():
    """The empty summary marks slots SLOT_NEVER_FIT, which no fit outcome equals."""
    zeros = ph.DecisionSummary.zeros(2, N_SLOTS, 4, 2, 2)
    assert (np.asarray(zeros.status) == ph.SLOT_NEVER_FIT).all()
    codes = {int(x) for x in RefitStatus} | {int(x) for x in UpdaterStatus}
    assert ph.SLOT_NEVER_FIT not in codes


def test_a_blind_policy_keeps_its_prior_summary_to_the_horizon():
    """Mask, not zeros: the prior_planned policy summary stays the initial one."""
    _, study, _, _, _, run = build(arm="prior_planned", occupancy=0.5, **SMALL)
    channel = study.channel_model
    prior = prior_summary(channel, len(channel.catalog_mass_kg))
    got = jax.tree_util.tree_map(np.asarray, run.carry.pbelief)
    for name in ("p_new", "p_det", "p_block", "n_det", "status", "p_class"):
        expected = np.asarray(getattr(prior, name))
        held = np.asarray(getattr(got, name))
        assert np.array_equal(held, np.broadcast_to(expected, held.shape)), name


def _giant_visits(world, factor=11.0):
    """Visits of star 1 when its slot 1 is `factor` times larger than the fixture's."""
    _, channel, _, _ = world
    truth = jax.tree_util.tree_map(lambda x: x[0], fixture_source().population)
    big = eqx.tree_at(lambda t: t.Rp_AU, truth, truth.Rp_AU.at[1, 1].multiply(factor))
    days = [30.0 * i for i in range(1, 7)]
    return {1: realize_visits(channel, big, 1, 4, days)}


def test_a_bright_planet_fits_when_the_scale_prior_holds_the_population(world):
    """A planet 11 times larger than the fixture's, in a population that holds it:
    the slot fits (status OK, detections kept); under the default Earth-twin window
    (0.6 to 1.7 Earth radii at albedo 0.3) it does not fit, the entry is
    DEGRADED, not FAILED, the other slots still install, and the failed slot carries
    its code.
    """
    study, _, _, _ = world
    seen = _giant_visits(world)
    population = fixture_source().population
    population = eqx.tree_at(
        lambda t: t.Rp_AU, population, population.Rp_AU.at[:, 1, 1].multiply(11.0)
    )
    wide = ph.population_ln_lam_range(population)
    request = visit_request(study, seen, tick=TICK)
    got = _boundary(study, 1, ess_floor=1.0, ln_lam_range=wide).refit_batch(
        study, request
    )
    slot = _slices(study, got, 0, 1)
    assert int(got.status[0, 0, 1]) == int(RefitStatus.OK)
    assert slot["n_det"][1] >= 2 and slot["status"][1] == ph.SLOT_OK
    assert slot["ess"][1] > 1.0
    ref = float(np.log(0.3 * Rearth2AU**2))
    narrow = _boundary(study, 1, ess_floor=1.0, ln_lam_range=(ref - 1.0, ref + 1.0))
    got = narrow.refit_batch(study, request)
    slot = _slices(study, got, 0, 1)
    assert slot["status"][1] in ph.SLOT_KEPT
    assert int(got.status[0, 0, 1]) == int(RefitStatus.DEGRADED)
    dark = slot["n_det"] == 0
    assert dark.any() and (slot["status"][dark] == ph.SLOT_OK).all()
    assert np.any(slot["p_new"]) and np.all(np.isfinite(slot["p_new"]))


def _null_lane(channel, st, star, n_rows, key=(1, 2)):
    """A first-tier lane of `star` read through `n_rows` null reports."""
    width = channel.widths()[0]
    columns = jx.visit_columns(N_SLOTS)
    mode = 4
    rows = np.zeros((n_rows, width))
    for i in range(n_rows):
        duration = channel.exposures_s[mode]
        rows[i, columns.index("time_d")] = channel.epoch_jd + 30.0 * (i + 1)
        rows[i, columns.index("science_s")] = duration
        rows[i, columns.index("occupied_s")] = duration
        rows[i, columns.index("valid_science")] = 1.0
    entry = ph._RowLane(
        payload=jnp.asarray(rows), use=jnp.ones(n_rows, bool), subject=jnp.asarray(star)
    )
    g = ph._grid_entry(channel, entry, st.grid_points)
    zeros = jnp.zeros(n_rows)
    return ph._FitLane(
        subject=jnp.asarray(star),
        tick=jnp.asarray(TICK),
        key=jnp.asarray(key, jnp.uint32),
        slot=jnp.asarray(N_SLOTS),
        n_undetected=jnp.asarray(N_SLOTS),
        epochs=g["epochs"],
        sep=g["sep"],
        dmag0=g["dmag0"],
        limit=g["limit"],
        detected=jnp.zeros(n_rows, bool),
        counts=zeros,
        valid=g["valid"].all(axis=0),
        ra=zeros,
        dec=zeros,
        ra_err=zeros,
        dec_err=zeros,
        corr=zeros,
        end_tick=jnp.zeros(n_rows, jnp.int64),
        live=jnp.asarray(True),
    )


@pytest.mark.parametrize("n_rows", [1, 6])
def test_the_search_probability_follows_the_posterior_occupancy(world, n_rows):
    """`p_new` equals `1 - (1 - q' p1) ** n` with `q' = q L / (1 - q + q L)` per slot.

    `L` is the first tier's null evidence, `q` the prior occupancy per undetected
    slot and `p1` the cloud's chance that an occupied slot is class 1 and detected
    (found from the same cloud at `q = 1`). More nulls give a smaller `L`, so a
    smaller `p_new` than the constant prior gives.
    """
    study, channel, _, _ = world
    boundary = _boundary(study, 1)
    st = boundary._post_settings(channel)
    lane = _null_lane(channel, st, 0, n_rows)
    cloud, _ = ph._first_tier_cloud(channel, lane, st, 1)
    like = float(jnp.exp(cloud.log_evidence))
    n = N_SLOTS
    q = st.occupancy / n
    q_post = q * like / (1.0 - q + q * like)
    elements = ph.channel_elements(cloud.elements)
    zeros = jnp.zeros(elements.shape[0])
    args = (zeros,)

    def search(occupancy):
        return ph.summarize_search(
            elements,
            *args,
            occupancy,
            n,
            0,
            channel,
            jnp.asarray(TICK),
            n_grid=st.n_grid,
            spacing_d=st.spacing_d,
            albedo=st.albedo,
        )

    p_all = np.asarray(search(float(n)))  # q = 1: 1 - (1 - p1) ** n
    p1 = 1.0 - (1.0 - p_all) ** (1.0 / n)
    expected = 1.0 - (1.0 - q_post * p1) ** n
    got = np.asarray(ph._first_tier_lane(channel, lane, st, 1)[0])
    np.testing.assert_allclose(got, expected, rtol=1e-10, atol=1e-14)
    prior = 1.0 - (1.0 - q * p1) ** n
    assert np.all(got <= prior + 1e-15) and got.max() < prior.max()


def test_degraded_slots_reach_the_outcome_table(world):
    """The outcome table counts the kernel's `SLOT_DEGRADED` codes as `low_ess`.

    No fixture run detects a planet, so no study run reaches a fit; the refit's slot
    codes at the unreachable size floor are written into a finished run's summary and
    the table reads them back. A claim never rests on such a slot.
    """
    study, _, star0, _ = world
    request = visit_request(study, {0: star0}, tick=TICK)
    result = _boundary(study, 2, ess_floor=1e12).refit_batch(study, request)
    status = _slices(study, result, 0, 0)["status"]
    n_degraded = int((status == ph.SLOT_DEGRADED).sum())
    assert n_degraded > 0
    _, built, _, truths, _, run = build(arm="prior_planned", occupancy=0.5, **SMALL)
    belief = run.carry.belief
    belief = eqx.tree_at(
        lambda b: b.status, belief, belief.status.at[0, 0].set(jnp.asarray(status))
    )
    run = eqx.tree_at(lambda r: r.carry.belief, run, belief)
    table = br.outcome_table(run, truths, built)
    assert int(table.world["low_ess"][0]) == n_degraded
    assert int(table.world["low_ess"][1]) == 0
    assert not np.asarray(br.claimed_slots(belief, 1.0)).any()
