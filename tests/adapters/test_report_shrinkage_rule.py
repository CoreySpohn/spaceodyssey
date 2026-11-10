"""The shrinkage rule's cloud and its registry check, with no channel.

The rule: after a detection the slot's cloud is 500 orbits around the catalog orbit,
`a` and `Ag` log-normal at a fractional width of 0.20, 0.10 and 0.025 after one, two
and three or more detections, every other element at the catalog value.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("photomancy")

from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import Audience
from spaceodyssey.studies import bracket as br
from tests.adapters.test_report_refit import make_study

TRUTH = np.array([6.0, 0.2, 0.3, 1.0, 2.0, 40.0, 0.0, -2.0])
N_KEYS = 400


def _spread(width):
    """`(a, ln a, Ag)` of `N_KEYS` independent shrinkage clouds, one row per cloud."""
    keys = jax.random.split(jax.random.PRNGKey(3), N_KEYS)
    clouds = jax.jit(
        jax.vmap(lambda k: ph.shrinkage_cloud(TRUTH, k, ph.SHRINK_DRAWS, width))
    )(keys)
    a = np.asarray(ph.semimajor_axis_au(clouds))
    ag = np.exp(np.asarray(clouds[..., 7]) - TRUTH[7])
    return clouds, a, ag


def test_the_spread_of_a_at_three_detections_is_a_fractional_0_025():
    """Number: std(a) / mean(a) over 500-orbit clouds is 0.025 within 3 standard errors.

    The standard error is the spread of the per-cloud value over 400 independent
    clouds. The log standard deviation of `a` and of `Ag` is held to the same bound.
    """
    width = float(ph.shrinkage_width(jnp.asarray(3)))
    assert width == 0.025
    _, a, ag = _spread(width)
    for draw in (a, ag):
        frac = draw.std(axis=1, ddof=1) / draw.mean(axis=1)
        mean, se = frac.mean(), frac.std(ddof=1) / np.sqrt(N_KEYS)
        assert abs(mean - 0.025) < 3.0 * se, (mean, se)
        logs = np.log(draw).std(axis=1, ddof=1)
        assert abs(logs.mean() - 0.025) < 3.0 * logs.std(ddof=1) / np.sqrt(N_KEYS)


@pytest.mark.parametrize(
    ("n_det", "width"), [(1, 0.20), (2, 0.10), (3, 0.025), (9, 0.025)]
)
def test_the_width_follows_the_detection_count(n_det, width):
    """One detection is 0.20, two 0.10, three or more 0.025."""
    assert float(ph.shrinkage_width(jnp.asarray(n_det))) == width


def test_the_other_elements_stay_at_the_catalog_values():
    """Only `a` (through the period at fixed mass) and `Ag` (in `ln_lam`) move."""
    clouds, _, _ = _spread(0.2)
    clouds = np.asarray(clouds)
    for i in (1, 2, 3, 4, 5, 6):
        assert (clouds[..., i] == TRUTH[i]).all(), i
    assert clouds[..., 0].std() > 0 and clouds[..., 7].std() > 0
    # the draws of `a` and `Ag` are independent
    corr = np.corrcoef(clouds[0, :, 0], clouds[0, :, 7])[0, 1]
    assert abs(corr) < 0.2


def test_the_cloud_is_a_function_of_its_key():
    """The same key gives the same cloud; another key gives another."""
    k1, k2 = jax.random.PRNGKey(1), jax.random.PRNGKey(2)
    a = ph.shrinkage_cloud(TRUTH, k1, 50, 0.1)
    assert np.array_equal(a, ph.shrinkage_cloud(TRUTH, k1, 50, 0.1))
    assert not np.array_equal(a, ph.shrinkage_cloud(TRUTH, k2, 50, 0.1))


def _boundary(caps):
    return ph.ReportRefit(
        kernel="shrinkage",
        truth_groups=ph.SHRINK_GROUPS,
        occupancy=0.5,
        ln_lam_range=(-30.0, -20.0),
        caps=caps,
    )


def test_construction_raises_if_the_registry_hides_the_ephemeris_group():
    """A hidden orbit column reads as zero, so construction and every call refuse it."""
    _boundary(make_study().caps)
    hidden = make_study(open_groups=("existence", "class"))
    with pytest.raises(ValueError, match="shrinkage kernel needs"):
        _boundary(hidden.caps)
    with pytest.raises(ValueError, match="hidden"):
        _boundary(None).refit_batch(hidden, None)


def test_construction_raises_if_the_registry_hides_the_photometric_scale():
    """`ln_lam`, the column `Ag` is centered on, sits in the class group."""
    hidden = make_study(open_groups=("existence", "ephemeris"))
    with pytest.raises(ValueError, match="class"):
        _boundary(hidden.caps)


def test_the_kernel_declares_the_groups_it_reads_and_the_first_tier_settings():
    """The orbit and class groups, the occupancy and the scale range are required."""
    with pytest.raises(ValueError, match="exactly"):
        ph.ReportRefit(kernel="shrinkage", truth_groups=("ephemeris",), occupancy=0.5)
    with pytest.raises(ValueError, match="occupancy"):
        ph.ReportRefit(kernel="shrinkage", truth_groups=ph.SHRINK_GROUPS)
    with pytest.raises(ValueError, match="ln_lam_range"):
        ph.ReportRefit(kernel="shrinkage", truth_groups=ph.SHRINK_GROUPS, occupancy=0.5)
    config = _boundary(None).configuration()
    assert config["n_shrink"] == 500 and config["shrink_widths"] == [0.2, 0.1, 0.025]


def test_the_shrinkage_arm_opens_what_the_kernel_reads():
    """The arm's registry shows the orbit and scale columns and hides existence."""
    arm = br.ARMS["shrinkage"]
    assert arm.kernel == "shrinkage"
    assert set(arm.groups) == set(ph.SHRINK_GROUPS)
    registry = br.bracket_registry("shrinkage", n_slots=2)
    prior = ph.DecisionSummary.zeros(4, 2, 16, 2, 2, n_an=4)
    observer = br.SummaryObserver(prior=prior)
    caps = registry.lower(
        Capacities(n_subjects=4, rows_per_subject=6, refit_lanes=2), observer
    )
    _boundary(caps)
    row = caps.payload_audience[caps.kind_names.index("truth_catalog")]
    need = int(Audience.OBSERVER) | int(Audience.POLICY)
    for col, (field, group) in enumerate(ph.CATALOG_FIELDS):
        open_ = (int(row[col]) & need) == need
        assert open_ == (group in ph.SHRINK_GROUPS), field
