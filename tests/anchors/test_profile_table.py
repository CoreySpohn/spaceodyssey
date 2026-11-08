"""The first physical profile: complete table, page agreement, pins (KP5).

The profile table ships as JSON; the readable page lives in
docs/profiles/first-summary-v1.md; the fixture manifest pins the library
revisions and the response package. Each check names the defect it rejects.
"""

import copy
import importlib.metadata
import json
import re
from pathlib import Path

import pytest
from packaging.version import Version

from spaceodyssey.adapters.physical_profile import (
    audience_flags,
    check_profile,
    load_profile,
    payload_column_names,
)
from spaceodyssey.core.enums import Audience

ROOT = Path(__file__).parents[2]
PAGE = ROOT / "docs" / "profiles" / "first-summary-v1.md"
MANIFEST = json.loads((Path(__file__).parent / "fixture_manifest.json").read_text())
SIBLING_PINS = (ROOT / ".github" / "sibling-pins.txt").read_text()
PROFILE = load_profile()

# The record envelope's payload columns, in record order.
RECORD_COLUMNS = (
    "time_d",
    "science_s",
    "occupied_s",
    "detected",
    "counts",
    "limit_counts",
    "snr",
    "ra",
    "dec",
    "ra_err",
    "dec_err",
    "corr",
    "cadence",
    "calibration_id",
    "valid_science",
)


def _mutate(path, value):
    profile = copy.deepcopy(PROFILE)
    *parents, last = path
    node = profile
    for key in parents:
        node = node[key]
    if value is KeyError:
        del node[last]
    else:
        node[last] = value
    return profile


def test_profile_is_complete_with_the_record_columns():
    """Fails if a payload column or boundary lacks a unit, sign, measure or audience.

    The shipped table passes `check_profile` and lists exactly the record
    envelope's columns in record order, each readable by a nonempty audience.
    """
    assert check_profile(PROFILE) == []
    assert payload_column_names(PROFILE) == RECORD_COLUMNS
    for column in PROFILE["payload_columns"]:
        assert audience_flags(column["audience"]) & Audience.SCORER


@pytest.mark.parametrize(
    ("path", "value", "message"),
    [
        (("payload_columns", 4, "unit"), KeyError, "counts.unit"),
        (("payload_columns", 7, "sign"), "  ", "ra.sign"),
        (("payload_columns", 0, "audience"), ["auditor"], "time_d.audience"),
        (("boundaries", 8, "status"), "blocked: ", "detector_variance.status"),
        (("boundaries", 1, "name"), "geometry_position", "repeated name"),
        (("payload_columns",), [], "payload_columns: missing"),
    ],
    ids=["no-unit", "blank-sign", "bad-audience", "blocked-unnamed", "repeat", "empty"],
)
def test_incomplete_profiles_are_named(path, value, message):
    """Fails if `check_profile` accepts a missing, blank or invalid entry."""
    problems = check_profile(_mutate(path, value))
    assert any(message in p for p in problems), problems


def _page_table(heading):
    """Body rows of the first markdown table after `heading`, as lists of cells."""
    section = PAGE.read_text().split(f"## {heading}\n", 1)[1].splitlines()
    first = next(i for i, line in enumerate(section) if line.startswith("|"))
    rows = []
    for line in section[first + 2 :]:
        if not line.startswith("|"):
            break
        rows.append([c.strip() for c in line.strip("|").split("|")])
    return rows


def test_page_tables_match_the_shipped_profile():
    """Fails if the readable page and the shipped table disagree on any cell."""
    columns = [
        [f"`{c['name']}`", c["unit"], c["sign"], c["measure"], ", ".join(c["audience"])]
        for c in PROFILE["payload_columns"]
    ]
    assert _page_table("Payload columns") == columns
    fields = ("observable", "unit", "sign", "measure", "allocation", "status")
    boundaries = [
        [f"`{b['name']}`", *(b[f] for f in fields)] for b in PROFILE["boundaries"]
    ]
    assert _page_table("Error budgets") == boundaries


def test_page_has_no_dashes_or_contractions():
    """Fails if public prose carries em or en dashes or contractions."""
    text = PAGE.read_text()
    assert not re.search("[\u2013\u2014]", text)
    assert not re.search(r"\b\w+n't\b|\b(it|that|there)'s\b", text)


def test_manifest_pins_every_consumed_library_and_the_response():
    """Fails if a consumed library is unpinned, below its floor, or the pins disagree.

    Every library in the manifest has a 40-hex revision and a version floor the
    installed package meets. A floor that is not yet released
    (a development version) is installed from the sibling pin at exactly the
    manifest revision. The manifest's response pin is the profile's and is
    yippy's catalog entry for its release.
    """
    from yippy import datasets

    pins = dict(re.findall(r"^(\S+) @ git\+\S+@([0-9a-f]{40})$", SIBLING_PINS, re.M))
    for name in MANIFEST["libraries"]:
        entry = MANIFEST["libraries"][name]
        assert re.fullmatch("[0-9a-f]{40}", entry["revision"]), name
        installed = Version(importlib.metadata.version(name))
        assert installed >= Version(entry["floor"]), (name, installed)
        if Version(entry["floor"]).is_devrelease:
            assert pins.get(name) == entry["revision"], (name, pins)
    data = MANIFEST["data"]
    response = PROFILE["response"]
    assert (data["package"], data["release"], data["digest"]) == (
        response["package"],
        response["release"],
        response["digest"],
    )
    assert datasets.DATA_RELEASE_TAG == response["release"]
    assert datasets.yip_info(response["package"])["md5"] == response["digest"]


@pytest.mark.medium(reason="hashes the 78 MB package; downloads it on a cold cache")
def test_loader_digest_and_header_equal_the_pins():
    """Fails if the fetched package's digest or header differs from the profile.

    The package fetched through yippy's loader hashes to the pinned digest, and
    its header carries the profile's aperture, obscuration, band, pixel scale,
    grid and centre.
    """
    import pooch
    from astropy.io import fits
    from yippy.datasets import cache_dir, fetch_yip

    response = PROFILE["response"]
    path = Path(fetch_yip(response["package"]))
    archive = cache_dir() / f"{response['package']}.zip"
    assert f"md5:{pooch.file_hash(str(archive), alg='md5')}" == response["digest"]
    header = fits.getheader(path / "stellar_intens.fits")
    assert header["D"] == response["diameter_m"]
    assert header["OBSCURED"] == response["obscured_area_fraction"]
    assert header["PIXSCALE"] == response["pixel_scale_lod"]
    assert header["NAXIS1"] == response["grid_pixels"]
    assert header["XCENTER"] == response["header_center_pix"]
    band_um = [header["MINLAM"], header["MAXLAM"]]
    assert [x * 1000.0 for x in band_um] == pytest.approx(response["band_nm"])
    assert header["LAMBDA"] * 1000.0 == pytest.approx(response["central_wavelength_nm"])
