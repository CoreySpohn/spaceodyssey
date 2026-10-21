"""The habitable-zone claim edges match the skyscapes reference definition.

skyscapes is an optional physics dependency, so the comparison is skipped
when it, or a release that carries ``skyscapes.habitable_zone``, is absent.
"""

import math

import pytest

from spaceodyssey.profiles import HABITABLE_ZONE, OPTIMISTIC_ZONE

habitable_zone = pytest.importorskip("skyscapes.habitable_zone")


def _zone_edges(claim):
    assert claim.name == "habitable_zone"
    by_side = {edge.side: edge for edge in claim.edges}
    assert sorted(by_side) == ["above", "below"]
    assert {edge.quantity for edge in claim.edges} == {"ln_a_eeid"}
    return by_side["above"].threshold, by_side["below"].threshold


def test_conservative_zone_matches_skyscapes_bitwise():
    inner, outer = _zone_edges(HABITABLE_ZONE)
    assert inner == math.log(habitable_zone.HZ_INNER_EEID)
    assert outer == math.log(habitable_zone.HZ_OUTER_EEID)


def test_optimistic_zone_matches_skyscapes_bitwise():
    (claim,) = [c for c in OPTIMISTIC_ZONE.claims if c.name == "habitable_zone"]
    inner, outer = _zone_edges(claim)
    assert inner == math.log(habitable_zone.OPTIMISTIC_HZ_INNER_EEID)
    assert outer == math.log(habitable_zone.OPTIMISTIC_HZ_OUTER_EEID)


@pytest.mark.parametrize("luminosity_lsun", [0.1, 1.0, 3.0])
def test_zone_in_au_matches_skyscapes(luminosity_lsun):
    inner, outer = _zone_edges(HABITABLE_ZONE)
    inner_au, outer_au = habitable_zone.habitable_zone_au(luminosity_lsun)
    eeid_au = math.sqrt(luminosity_lsun)
    assert float(inner_au) == pytest.approx(math.exp(inner) * eeid_au, rel=1e-6)
    assert float(outer_au) == pytest.approx(math.exp(outer) * eeid_au, rel=1e-6)
