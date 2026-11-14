"""Named observing modes, bands, channels and resources."""

import dataclasses
import json

import jax
import numpy as np
import pytest
from hwoutils.constants import d2s

from spaceodyssey.adapters.jaxedith import (
    ObservingModeChannel,
    ReportingLaw,
    survey_grid_d,
    survey_table,
)
from spaceodyssey.adapters.modes import (
    ModeSpec,
    ModeTable,
    load_mode_specs,
    profile_modes,
)
from spaceodyssey.adapters.physical_profile import JATIS_EAC1_V1
from spaceodyssey.adapters.skyscapes import Band, star_spec_from_model

EPOCH_JD = 2462502.5
pytestmark = pytest.mark.medium(reason="builds declared reports and host scenes")


@pytest.fixture(scope="module")
def stars():
    bands = (
        *[m.band for m in profile_modes(JATIS_EAC1_V1)],
        Band(760.0, 760.0 / 140.0),
    )
    return (
        star_spec_from_model(
            "sun",
            mass_msun=1.0,
            dist_pc=10.0,
            luminosity_lsun=1.0,
            teff_k=5772.0,
            radius_rsun=1.0,
            bands=bands,
        ),
    )


def channel_with(stars, modes):
    """The fixture channel with the supplied mode declarations."""
    return ObservingModeChannel.from_stars(
        stars,
        n_slots=2,
        epoch_jd=EPOCH_JD,
        modes=modes,
        durations_d=survey_grid_d(survey_table(JATIS_EAC1_V1), 3),
        profile=JATIS_EAC1_V1,
    )


@pytest.fixture(scope="module")
def channel(bracket_channel):
    return bracket_channel


@pytest.fixture(scope="module")
def study(bracket_study):
    return bracket_study


def test_channels_are_found_by_mode_name(channel):
    """Fails if a channel index is derived by adding group sizes (the old n_detect +
    k).
    """
    t = channel.mode_table
    assert [
        t.mode_of(t.channel(m.name, k))
        for m in t.modes
        for k in range(len(t.durations_d))
    ] == [i for i in range(len(t.modes)) for _ in t.durations_d]
    assert t.catalog_channel == channel.n_channels - 1 == t.n_channels - 1
    assert channel.n_modes == channel.n_block_modes == len(t.durations_d) == 3


def test_a_third_mode_needs_no_index_arithmetic(stars):
    """Fails if a new mode moves channels or rebuilds the same pinned response."""
    extra = ModeSpec(
        "ifs_760",
        "spectrograph",
        Band(760.0, 760.0 / 140.0),
        0.29165,
        300.0,
        140.0,
        None,
    )
    base = profile_modes(JATIS_EAC1_V1)
    two, three = channel_with(stars, base), channel_with(stars, (*base, extra))
    assert three.mode_table.channel("detection", 0) == two.mode_table.channel(
        "detection", 0
    )
    assert three.mode_table.bands[three.mode_table.band_index("ifs_760")] == extra.band
    report = three.assets().reports[2]
    assert report.wavelength_nm == 760.0
    assert report.resolving_power == 140.0
    assert report.detector_spec.frame_time_s == 300.0
    backend = two.assets().reports[0].optical_path.coronagraph._backend
    assert all(
        r.optical_path.coronagraph._backend is backend
        for r in (*two.assets().reports, *three.assets().reports)
    )
    with jax.default_matmul_precision("highest"):
        other = channel_with(stars, base).assets()
    assert other.reports[0].optical_path.coronagraph._backend is not backend


def test_the_source_holds_no_characterization_requirement(channel):
    """Fails if a removed block or survey field survives on the channel."""
    assert not any(
        hasattr(channel, n)
        for n in ("survey", "block_rate_divisor", "exposures_s", "block_days")
    )


def test_mode_specs_and_tables_refuse_bad_declarations():
    """Fails if a spectrograph carries a requirement, or one band would need two
    reports.
    """
    with pytest.raises(ValueError, match="spectrograph"):
        ModeSpec("ifs", "spectrograph", Band(760.0, 5.4), 0.29, 300.0, 140.0, 5.0)
    a = ModeSpec("a", "spectrograph", Band(760.0, 5.4), 0.29, 300.0, 140.0, None)
    with pytest.raises(ValueError, match="band"):
        ModeTable((a, dataclasses.replace(a, name="b", frame_time_s=100.0)), (1.0,))
    with pytest.raises(ValueError, match="duplicate"):
        ModeTable((a, a), (1.0,))
    imaging = ModeSpec("image", "imaging", a.band, 0.29, 300.0, None, 7.0)
    with pytest.raises(ValueError, match="imaging"):
        ModeTable(
            (
                imaging,
                dataclasses.replace(imaging, name="blue", band=Band(500.0, 100.0)),
            ),
            (1.0,),
        )


def test_modes_are_required(stars):
    """Fails if a channel inherits its bands from a profile default."""
    with pytest.raises(TypeError, match="modes"):
        ObservingModeChannel.from_stars(
            stars,
            n_slots=2,
            epoch_jd=EPOCH_JD,
            durations_d=(1.0,),
            profile=JATIS_EAC1_V1,
        )


def test_imaging_modes_require_one_reporting_threshold(stars):
    """Fails if imaging declarations and the shared refit law can disagree."""
    image, spectrum = profile_modes(JATIS_EAC1_V1)
    high_threshold = dataclasses.replace(
        image, name="high_threshold", snr_threshold=9.0
    )
    with pytest.raises(ValueError, match=r"detection.*7\.0.*high_threshold.*9\.0"):
        ModeTable((image, high_threshold, spectrum), (1.0,))

    kwargs = dict(
        n_slots=2,
        epoch_jd=EPOCH_JD,
        modes=(dataclasses.replace(image, snr_threshold=9.0), high_threshold, spectrum),
        durations_d=(1.0,),
        profile=JATIS_EAC1_V1,
    )
    with pytest.raises(ValueError, match=r"law.*7\.0.*detection.*9\.0"):
        ObservingModeChannel.from_stars(
            stars, law=ReportingLaw(threshold=7.0), **kwargs
        )

    inferred = ObservingModeChannel.from_stars(stars, **kwargs)
    assert inferred.assets().law.threshold == 9.0
    law = ReportingLaw(threshold=9.0, kappa=2.0, corr=0.25)
    explicit = ObservingModeChannel.from_stars(stars, law=law, **kwargs)
    assert explicit.assets().law == law
    assert all(
        a.optical_path.coronagraph._backend is b.optical_path.coronagraph._backend
        for a, b in zip(
            inferred.assets().reports, explicit.assets().reports, strict=True
        )
    )
    assert inferred.assets().calibration_id != explicit.assets().calibration_id


def test_channel_table_charges_occupancy_to_the_named_resource(study):
    """Fails if a mode charges the wrong resource column."""
    t = study.channel_model.mode_table
    assert study.caps.n_channels == t.n_channels
    cost = np.asarray(study.channels.cost)
    col = t.resource("telescope_days")
    assert (cost[: t.catalog_channel, col] > 0).all() and cost[
        t.catalog_channel
    ].sum() == 0
    assert (np.delete(cost[: t.catalog_channel], col, axis=1) == 0).all()


def test_mode_table_maps_durations_and_loads_specs(tmp_path):
    """Fails if declarations lose band order, channel time, or resource names."""
    modes = profile_modes(JATIS_EAC1_V1)
    table = ModeTable(modes, (0.5, 2.0))
    assert table.imaging_channels == (0, 1)
    assert table.spectrograph_channels == (2, 3)
    assert table.duration_of(3) == 1
    assert table.duration_s(3) == 2.0 * d2s
    assert table.imaging_band == Band(500.0, 100.0)
    assert table.resources == (("telescope_days", 2),)
    path = tmp_path / "modes.json"
    path.write_text(json.dumps([dataclasses.asdict(m) for m in modes]))
    assert load_mode_specs(path) == modes
