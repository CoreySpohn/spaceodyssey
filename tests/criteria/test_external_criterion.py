"""A user-defined criterion and radius claim travel through a physical study."""

import dataclasses
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.jaxedith import ObservingModeChannel
from spaceodyssey.adapters.measurement import measurement_template
from spaceodyssey.adapters.modes import ModeTable
from spaceodyssey.adapters.parameters import PlanetParameters
from spaceodyssey.adapters.photomancy import DecisionSummary
from spaceodyssey.adapters.physical_profile import JATIS_EAC1_V1
from spaceodyssey.adapters.skyscapes import Band, ExoVistaWorldSource, PopulationModel
from spaceodyssey.codec import KindResolver, dump_study, load_study
from spaceodyssey.core.enums import F64, I64, RunStatus
from spaceodyssey.core.records import pack, unpack
from spaceodyssey.criteria.base import AbstractCriterion, StarOptions
from spaceodyssey.criteria.claims import AbstractClaim
from spaceodyssey.plugins import installed_kinds
from spaceodyssey.studies import bracket as br
from tests.studies.bracket_helpers import fixture_files, fixture_modes


class SmallRadiusClaim(AbstractClaim):
    name: ClassVar[str] = "small_radius"
    maximum_rearth: ClassVar[float] = 2.0

    def evaluate(self, params, luminosity_lsun):
        return params.radius_rearth < self.maximum_rearth


class BandVisits(eqx.Module):
    visits: jax.Array


class ThreeBandCriterion(AbstractCriterion):
    mode_names: tuple[str, ...] = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    label_values: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, F64))
    claims: tuple[AbstractClaim, ...]

    def state_template(self, n_tracks, n_bands):
        return BandVisits(jax.ShapeDtypeStruct((n_tracks, n_bands), F64))

    def init_state(self, n_tracks, n_bands):
        return BandVisits(jnp.zeros((n_tracks, n_bands), F64))

    def update(self, state, row, modes):
        take = (row.assignment >= 0) & (row.valid_science > 0)
        bands = jnp.asarray([modes.band_index(m.name) for m in modes.modes])
        observed = jnp.arange(modes.n_bands) == bands[row.mode]
        return BandVisits(
            state.visits.at[row.assignment].add(
                take[:, None] & observed[None, :], mode="drop"
            )
        )

    def value(self, state, p_claim, tick):
        met = jnp.all(state.visits > 0, axis=-1)
        value = jnp.where(met, p_claim[:, 0] * self.label_values[0], 0.0)
        return value, jnp.where(value > 0, 0, -1).astype(I64)

    def options(self, state, summary_star, tick, modes, settings):
        channels = jnp.arange(modes.catalog_channel)
        bands = jnp.asarray(
            [
                modes.band_index(modes.modes[modes.mode_of(c)].name)
                for c in range(modes.catalog_channel)
            ]
        )
        missing = state.visits[:, bands] == 0
        gain = jnp.sum(
            summary_star.p_exist[:, None] * summary_star.p_claim[:, :1] * missing,
            axis=0,
        )
        gain = jnp.where(summary_star.available, gain[None, :], 0.0)
        return StarOptions(
            gain,
            jnp.broadcast_to(
                settings.channel_cost[:-1],
                (*gain.shape, settings.channel_cost.shape[1]),
            ),
            jnp.broadcast_to(channels, gain.shape),
        )


def external_criterion():
    return ThreeBandCriterion(
        mode_names=("detection", "characterization", "extra"),
        labels=("small_complete",),
        label_values=[1.0],
        claims=(SmallRadiusClaim(),),
    )


def external_modes():
    modes = fixture_modes()
    return ModeTable(
        (*modes, dataclasses.replace(modes[1], name="extra", band=Band(980.0, 6.7))),
        (0.1,),
    )


EXTERNAL_KINDS = {
    ("criterion", "external.three_band", 1): ThreeBandCriterion,
    ("component", "external.small_radius", 1): SmallRadiusClaim,
}


def test_an_external_criterion_registers_and_folds_all_declared_bands():
    """Fails if external kinds cannot resolve or packing loses the third band's
    credit.
    """
    resolver = KindResolver({**installed_kinds(), **br.BRACKET_KINDS, **EXTERNAL_KINDS})
    criterion, modes = external_criterion(), external_modes()
    assert resolver.ref_of(criterion) == ("criterion", "external.three_band", 1)
    assert resolver.ref_of(criterion.claims[0]) == (
        "component",
        "external.small_radius",
        1,
    )
    params = PlanetParameters(
        **{f.name: jnp.zeros(3) for f in dataclasses.fields(PlanetParameters)}
    )
    params = dataclasses.replace(
        params, radius_rearth=jnp.asarray([1.99, 2.0, 2.01]), albedo=jnp.zeros((3, 3))
    )
    np.testing.assert_array_equal(
        criterion.claims[0].evaluate(params, 1.0), [True, False, False]
    )
    template = criterion.state_template(1, modes.n_bands)
    state = unpack(template, pack(criterion.init_state(1, modes.n_bands)))
    row = jax.tree.map(
        lambda x: jnp.zeros(x.shape, x.dtype), measurement_template(1, modes.n_bands)
    )
    row = dataclasses.replace(
        row,
        assignment=jnp.asarray([0], I64),
        valid_science=jnp.asarray(1.0),
        mode=jnp.asarray(0, I64),
    )
    state = criterion.update(state, row, modes)
    row = dataclasses.replace(row, mode=jnp.asarray(1, I64))
    state = criterion.update(state, row, modes)
    assert float(criterion.value(state, jnp.ones((1, 1)), 0)[0][0]) == 0.0
    row = dataclasses.replace(row, mode=jnp.asarray(2, I64))
    state = unpack(template, pack(criterion.update(state, row, modes)))
    values, labels = criterion.value(state, jnp.asarray([[0.75]]), 0)
    np.testing.assert_array_equal(values, [0.75])
    np.testing.assert_array_equal(labels, [0])


@pytest.mark.slow(
    reason="one-star three-band physical study compiles the chunked refit driver"
)
def test_a_criterion_defined_outside_the_library_runs_unchanged():
    """Fails if the library assumes a band count, a claim or a criterion class
    it ships.
    """
    resolver = KindResolver({**installed_kinds(), **br.BRACKET_KINDS, **EXTERNAL_KINDS})
    modes = external_modes()
    source = ExoVistaWorldSource.from_files(
        [[fixture_files()[0]]],
        n_slots=1,
        bands=modes.bands,
        generator=PopulationModel.exovista(),
    )
    channel = ObservingModeChannel.from_stars(
        source.stars,
        n_slots=1,
        epoch_jd=source.epoch_jd,
        modes=modes.modes,
        durations_d=modes.durations_d,
        profile=JATIS_EAC1_V1,
        n_time_nodes=1,
    )
    prior = DecisionSummary.zeros(
        1, 1, 3, channel.n_modes, n_an=1, n_claims=1, n_variants=modes.catalog_channel
    )
    study = br.bracket_study(
        source,
        channel,
        horizon_d=30.0,
        budget_d=3.0,
        observer=br.SummaryObserver(
            prior=prior,
            criterion=external_criterion(),
            grants=br.GRANT_PRESETS["oracle"],
        ),
        policy=br.PackagePolicy(max_visits=3, max_spectra=3),
    )
    boundary = br.bracket_boundary(
        study,
        lanes=1,
        n_grid=3,
        n_first=8,
        n_estimate_systems=8,
        n_particles=8,
        n_round=8,
        n_rounds=1,
        n_centers=2,
        n_constants=2,
        max_rounds=1,
        forecast_quantiles=(0.5,),
    )
    study = dataclasses.replace(
        study,
        observer=dataclasses.replace(
            study.observer, prior=boundary.initial_summary(study)
        ),
    )
    loaded = load_study(
        dump_study(study, resolver).to_json(),
        resolver,
        registry=br.bracket_registry(
            study.observer.grants, channel.n_slots, channel.n_bands
        ),
    )
    assert loaded.caps.payload_paths == study.caps.payload_paths
    study = loaded
    run = br.BoundExecutor(boundary=boundary).run_worlds(
        study, source.take(jnp.arange(1)), jnp.asarray([jax.random.PRNGKey(0)]), 256
    )
    assert np.all(np.asarray(run.status) == RunStatus.COMPLETE)
    rows = br.row_log(run, study)
    assert rows["counts"].shape[0] > 0
    assert rows["counts"].shape[-1] == 3
    assert set(rows["mode"].tolist()) == {0, 1, 2}
    scored = br.verdict(run, study)
    assert scored.counts["small_complete"][0] > 0
    assert all(np.all(np.isfinite(value)) for value in scored.counts.values())
    assert np.all(np.isfinite(scored.yield_))
