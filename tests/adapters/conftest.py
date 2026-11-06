"""Shared physical fixtures: the observatory and the pinned public response."""

import pytest
from hwoutils.conversions import Msun_to_kg
from orbix.observatory import Observatory, ObservatoryL2Halo

from spaceodyssey.adapters.jaxedith import DetectorSpec, SummaryReport, SummaryScene

EPOCH_JD = 2461000.5  # zero of the orbit time axis in the physical fixtures


def uma_scene():
    """47 UMa, general-knowledge values pending a pinned public source."""
    return SummaryScene(
        star_flux_density_jy=42.0,
        dist_pc=13.8,
        star_mass_kg=Msun_to_kg(1.03),
        luminosity_lsun=1.54,
        ra_deg=164.867,
        dec_deg=40.430,
        star_diameter_mas=0.79,
        n_exozodi=3.0,
    )


@pytest.fixture(scope="session")
def observatory():
    return Observatory(orbit=ObservatoryL2Halo.from_default())


@pytest.fixture(scope="session")
def yip_report(observatory):
    """The summary path on the pinned package, built once (about 4 s cold)."""
    return SummaryReport.from_yip(
        DetectorSpec.fixture(),
        uma_scene(),
        orbit_epoch_jd=EPOCH_JD,
        observatory=observatory,
    )
