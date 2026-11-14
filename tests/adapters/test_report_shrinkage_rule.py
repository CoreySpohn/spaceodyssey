"""The shrinkage rule's cloud and its registry check, with no channel.

After a detection the slot has 500 draws. Log period has 1.5 times the width
of log albedo: 0.20, 0.10 and 0.025 after one, two and three or more detections.
The radius and all other physical parameters stay at their catalog values.
"""

import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("photomancy")

from hwoutils.constants import Rearth2AU

from spaceodyssey.adapters import photomancy as ph
from spaceodyssey.adapters.measurement import (
    MEASUREMENT_KIND_NAME,
    measurement_template,
)
from spaceodyssey.adapters.parameters import (
    catalog_group_of,
    from_photomancy,
    semimajor_axis_au,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import EVERYONE, Audience
from spaceodyssey.core.records import unpack
from spaceodyssey.studies import bracket as br
from tests.adapters.test_report_refit import make_study

TRUTH = jax.tree.map(
    lambda x: x[0],
    from_photomancy(
        jnp.array([[6.0, 0.2, 0.3, 1.0, 2.0, 0.1, 0.0, np.log(0.3 * Rearth2AU**2)]]),
        jnp.array([[0.3, 0.4]]),
        0,
    ),
)
N_KEYS = 400


def _spread(width):
    """`(a, ln a, Ag)` of `N_KEYS` independent shrinkage clouds, one row per cloud."""
    keys = jax.random.split(jax.random.PRNGKey(0), N_KEYS)
    clouds = jax.jit(
        jax.vmap(lambda k: ph.shrinkage_cloud(TRUTH, k, ph.SHRINK_DRAWS, width))
    )(keys)
    a = np.asarray(semimajor_axis_au(clouds))
    ag = np.asarray(clouds.albedo[..., 0] / TRUTH.albedo[0])
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
    """Fails if a shrinkage draw moves a field other than period and albedo."""
    clouds, _, _ = _spread(0.2)
    for field in (
        "ecc",
        "cos_i",
        "node_rad",
        "argp_rad",
        "tau_d",
        "star_mass_msun",
        "planet_mass_mearth",
        "radius_rearth",
    ):
        assert (np.asarray(getattr(clouds, field)) == getattr(TRUTH, field)).all(), (
            field
        )
    assert clouds.ln_period_d.std() > 0 and clouds.albedo.std() > 0
    corr = np.corrcoef(clouds.ln_period_d[0], jnp.log(clouds.albedo[0, :, 0]))[0, 1]
    assert abs(corr) < 0.2


def test_the_cloud_is_a_function_of_its_key():
    """The same key gives the same cloud; another key gives another."""
    k1, k2 = jax.random.split(jax.random.PRNGKey(0))
    a = ph.shrinkage_cloud(TRUTH, k1, 50, 0.1)
    assert all(
        np.array_equal(x, y)
        for x, y in zip(
            jax.tree.leaves(a),
            jax.tree.leaves(ph.shrinkage_cloud(TRUTH, k1, 50, 0.1)),
            strict=True,
        )
    )
    assert not np.array_equal(
        a.ln_period_d, ph.shrinkage_cloud(TRUTH, k2, 50, 0.1).ln_period_d
    )


def _boundary(caps):
    return ph.ReportRefit(
        kernel="shrinkage",
        truth_groups=ph.SHRINK_GROUPS,
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
    """Fails if shrinkage reads undeclared groups or loses bounded count estimation."""
    with pytest.raises(ValueError, match="exactly"):
        ph.ReportRefit(kernel="shrinkage", truth_groups=("ephemeris",))
    with pytest.raises(ValueError, match="n_estimate_systems"):
        ph.ReportRefit(
            kernel="shrinkage", truth_groups=ph.SHRINK_GROUPS, n_estimate_systems=0
        )
    config = _boundary(None).configuration()
    assert config["n_shrink"] == 500 and config["shrink_widths"] == [0.2, 0.1, 0.025]


def test_the_shrinkage_arm_opens_what_the_kernel_reads():
    """Fails if shrinkage hides measurements or exposes existence or measurement truth.

    The catalog grants orbit and scale; measured leaves are open to all audiences,
    while identity and generating counts and variance stay scorer-only.
    """
    arm = br.ARMS["shrinkage"]
    assert arm.kernel == "shrinkage"
    assert set(arm.groups) == set(ph.SHRINK_GROUPS)
    registry = br.bracket_registry("shrinkage", n_slots=2, n_bands=2)
    prior = ph.DecisionSummary.zeros(4, 2, 16, 2, n_an=4)
    observer = br.SummaryObserver(prior=prior)
    caps = registry.lower(
        Capacities(n_subjects=4, rows_per_subject=6, refit_lanes=2), observer
    )
    _boundary(caps)
    measurement = unpack(
        measurement_template(n_detections=2, n_bands=2),
        caps.payload_audience[caps.kind_names.index(MEASUREMENT_KIND_NAME)],
    )
    for name, grants in vars(measurement).items():
        expected = (
            int(Audience.SCORER)
            if name in ("identity", "true_counts", "true_variance")
            else int(EVERYONE)
        )
        assert (np.asarray(grants) == expected).all(), name
    row = caps.payload_audience[caps.kind_names.index("truth_catalog")]
    need = int(Audience.OBSERVER) | int(Audience.POLICY)
    for col, field in enumerate(
        caps.payload_paths[caps.kind_names.index("truth_catalog")]
    ):
        group = catalog_group_of(field)
        open_ = (int(row[col]) & need) == need
        assert open_ == (group in ph.SHRINK_GROUPS), field
