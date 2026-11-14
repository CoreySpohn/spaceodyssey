"""Compatible criterion ladders share one declared union channel table."""

import dataclasses
from importlib.resources import files

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.constants import Mearth2kg, Rearth2AU

from spaceodyssey.adapters.jaxedith import ObservingModeChannel
from spaceodyssey.adapters.modes import load_mode_specs, profile_modes, union_modes
from spaceodyssey.adapters.photomancy import DecisionSummary
from spaceodyssey.adapters.physical_profile import JATIS_EAC1_V1
from spaceodyssey.adapters.skyscapes import (
    ExoVistaWorldSource,
    PopulationModel,
    PopulationTruth,
    StarSpec,
)
from spaceodyssey.codec import KindResolver, dump_study, load_study
from spaceodyssey.core.records import pack
from spaceodyssey.criteria.claims import AbstractClaim, EarthCandidateClaim
from spaceodyssey.criteria.transition import TransitionTable, load_transition_table
from spaceodyssey.plugins import installed_kinds
from spaceodyssey.studies import bracket as br
from spaceodyssey.studies.bracket import ladder_studies
from tests.studies.bracket_helpers import fixture_criterion, needs_mode

EXAMPLE_MODES = files("spaceodyssey.adapters") / "examples/h2o_o2_modes.json"
EXAMPLE = files("spaceodyssey.criteria") / "examples/h2o_then_o2.json"
RESOLVER = KindResolver({**installed_kinds(), **br.BRACKET_KINDS})


def example_table():
    return load_transition_table(
        EXAMPLE, claims={"earth_candidate": EarthCandidateClaim()}
    )


def single_step_910():
    return fixture_criterion()


class VariableWidthTable(TransitionTable):
    extra_bands: int = eqx.field(static=True, default=0)

    def state_template(self, n_tracks, n_bands):
        return super().state_template(n_tracks, n_bands + self.extra_bands)

    def init_state(self, n_tracks, n_bands):
        return super().init_state(n_tracks, n_bands + self.extra_bands)


class RenamedClaim(EarthCandidateClaim):
    name = "renamed"


class RadiusClaim(AbstractClaim):
    name: str
    maximum_rearth: float

    def evaluate(self, params, luminosity_lsun):
        return params.radius_rearth < self.maximum_rearth


@pytest.fixture(scope="module")
def union_study():
    """One hand-built host and planet; no spectrum construction or file loading."""
    modes = union_modes(profile_modes(JATIS_EAC1_V1), load_mode_specs(EXAMPLE_MODES))
    bands = tuple(dict.fromkeys(m.band for m in modes))
    epoch_jd = 2462502.5
    stars = (
        StarSpec(
            "hand host",
            1.0,
            10.0,
            270.0,
            66.56,
            1.0,
            1.0,
            tuple((b.center_nm, 10.0) for b in bands),
        ),
    )
    leaf = jnp.ones((1, 1, 1))
    population = PopulationTruth(
        period_d=365.25 * leaf,
        e=0.0 * leaf,
        cos_i=0.0 * leaf,
        W_rad=0.0 * leaf,
        cos_w=leaf,
        sin_w=0.0 * leaf,
        tp_jd=epoch_jd * leaf,
        Mp_kg=Mearth2kg * leaf,
        Rp_AU=Rearth2AU * leaf,
        present=leaf.astype(bool),
        albedo=jnp.full((1, 1, 1, len(bands)), 0.2),
        bands=bands,
    )
    source = ExoVistaWorldSource(
        population=population,
        dropped=jnp.zeros((1, 1), int),
        generator=PopulationModel.exovista(),
        stars=stars,
        epoch_jd=epoch_jd,
    )
    channel = ObservingModeChannel.from_stars(
        stars,
        n_slots=1,
        epoch_jd=epoch_jd,
        modes=modes,
        durations_d=(0.1,),
        profile=JATIS_EAC1_V1,
        n_time_nodes=1,
    )
    prior = DecisionSummary.zeros(
        1,
        1,
        3,
        channel.n_modes,
        n_an=1,
        n_claims=1,
        n_variants=channel.mode_table.catalog_channel,
    )
    return br.bracket_study(
        source,
        channel,
        horizon_d=1.0,
        budget_d=1.0,
        observer=br.SummaryObserver(prior=prior, criterion=example_table()),
        policy=br.PackagePolicy(max_visits=1, max_spectra=1),
    )


def test_the_union_keeps_each_mode_once_and_refuses_a_conflict():
    """Fails if a shared mode is duplicated, an entry's mode is lost, or two specs
    share a name.
    """
    base, extra = profile_modes(JATIS_EAC1_V1), load_mode_specs(EXAMPLE_MODES)
    u = union_modes(base, extra, base)
    assert [m.name for m in u] == [m.name for m in base] + ["ifs_940", "ifs_760"]
    assert u == (*base, *extra)
    assert union_modes((), base, ()) == base and union_modes() == ()
    assert union_modes(extra, tuple(dataclasses.replace(m) for m in extra)) == extra
    with pytest.raises(ValueError, match="ifs_940"):
        union_modes(extra, (dataclasses.replace(extra[0], frame_time_s=100.0),))


@pytest.mark.medium(reason="builds one union channel and dumps three documents")
def test_ladder_entries_differ_only_in_the_criterion(union_study):
    """Fails if two ladder entries differ anywhere but the criterion field, or
    share a hash.
    """
    a, b = ladder_studies(union_study, (example_table(), single_step_910()))
    assert a.channel_model is b.channel_model and a.channels is b.channels
    assert a.channel_model is union_study.channel_model
    docs = [dump_study(s, RESOLVER) for s in (union_study, a, b)]
    assert docs[0].content_hash == docs[1].content_hash != docs[2].content_hash
    for study, doc in zip((a, b), docs[1:], strict=True):
        for field in dataclasses.fields(study):
            if field.name != "observer":
                assert getattr(study, field.name) is getattr(union_study, field.name)
        for field in dataclasses.fields(study.observer):
            if field.name != "criterion":
                assert getattr(study.observer, field.name) is getattr(
                    union_study.observer, field.name
                )
        assert (
            dump_study(load_study(doc.to_json(), RESOLVER), RESOLVER).content_hash
            == doc.content_hash
        )
    with pytest.raises(ValueError, match="ifs_1100"):
        ladder_studies(union_study, (needs_mode("ifs_1100"),))
    assert ladder_studies(union_study, ()) == ()


@pytest.mark.medium(reason="shares the physical union channel and population prior")
def test_each_entry_initializes_its_own_start_node(union_study):
    """Fails if an entry reuses the base criterion's packed initial state."""
    initial = single_step_910()
    terminal = dataclasses.replace(initial, start=1)
    a, b = ladder_studies(union_study, (initial, terminal))
    for entry, criterion in ((a, initial), (b, terminal)):
        n_bands = entry.channel_model.mode_table.n_bands
        expected = pack(criterion.init_state(1, n_bands))[None]
        np.testing.assert_array_equal(entry.observer.init(entry.caps).state, expected)
        np.testing.assert_array_equal(
            entry.observer.init_summary(entry.caps).state, expected
        )
    assert not np.array_equal(
        a.observer.init(a.caps).state, b.observer.init(b.caps).state
    )


@pytest.mark.medium(reason="shares the physical union channel and population prior")
def test_ladders_refuse_a_different_family_width_or_claim_schema(union_study):
    """Fails if a ladder reuses a prior with another state layout or claim meaning."""
    criterion = single_step_910()
    fields = {f.name: getattr(criterion, f.name) for f in dataclasses.fields(criterion)}
    variable = VariableWidthTable(**fields)
    base = dataclasses.replace(
        union_study,
        observer=dataclasses.replace(union_study.observer, criterion=variable),
    )
    with pytest.raises(ValueError, match="family"):
        ladder_studies(base, (criterion,))
    with pytest.raises(ValueError, match="width"):
        ladder_studies(base, (dataclasses.replace(variable, extra_bands=1),))
    for claims, target_claim in (((), None), ((RenamedClaim(),), "renamed")):
        incompatible = dataclasses.replace(
            criterion, claims=claims, target_claim=target_claim
        )
        with pytest.raises(ValueError, match="claim schema"):
            ladder_studies(union_study, (incompatible,))
    claim = RadiusClaim("earth_candidate", 2.0)
    configured = dataclasses.replace(criterion, claims=(claim,))
    base = dataclasses.replace(
        union_study,
        observer=dataclasses.replace(union_study.observer, criterion=configured),
    )
    assert len(ladder_studies(base, (dataclasses.replace(configured),))) == 1
    for replacement in (
        RadiusClaim("other", 2.0),
        dataclasses.replace(claim, maximum_rearth=3.0),
        EarthCandidateClaim(),
    ):
        incompatible = dataclasses.replace(
            configured, claims=(replacement,), target_claim=replacement.name
        )
        with pytest.raises(ValueError, match="claim schema"):
            ladder_studies(base, (incompatible,))
    claims = (EarthCandidateClaim(), RenamedClaim())
    configured = dataclasses.replace(criterion, claims=claims)
    observer = dataclasses.replace(
        union_study.observer,
        criterion=configured,
        prior=dataclasses.replace(
            union_study.observer.prior,
            p_claim=jnp.repeat(union_study.observer.prior.p_claim, 2, axis=-1),
        ),
    )
    base = dataclasses.replace(union_study, observer=observer)
    with pytest.raises(ValueError, match="claim schema"):
        ladder_studies(base, (dataclasses.replace(configured, claims=claims[::-1]),))
