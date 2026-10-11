"""Claim-axis contracts and weighted claim reports."""

from itertools import product

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.engines.analytic import CHANNELS, certified_yield
from spaceodyssey.metrics import CertifiedYield
from spaceodyssey.profiles import EXOEARTH_CANDIDATE, HABITABLE_ZONE_ONLY
from spaceodyssey.testing import check_outcome
from tests.test_plan import make_outcome

FIELDS = (
    "claim_p_pos",
    "claim_p_neg",
    "claim_p_wrong",
    "claim_p_neg_wrong",
    "claim_truth_positive",
    "claim_in_range",
    "claim_names",
)


def claim_fields(profile=EXOEARTH_CANDIDATE):
    """Nonzero probability channels with a rectangular claim axis."""
    shape = (3, len(profile.claims))
    return dict(
        claim_p_pos=jnp.full(shape, 0.6),
        claim_p_neg=jnp.full(shape, 0.1),
        claim_p_wrong=jnp.full(shape, 0.02),
        claim_p_neg_wrong=jnp.full(shape, 0.03),
        claim_truth_positive=jnp.ones(shape, dtype=bool),
        claim_in_range=jnp.ones(shape, dtype=bool),
        claim_names=profile.claim_names,
        profile_hash=profile.content_hash(),
    )


def test_claim_fields_all_or_none():
    """Every partial combination of the seven claim fields raises."""
    fields = claim_fields()
    for present in product((False, True), repeat=len(FIELDS)):
        kwargs = {
            name: fields[name]
            for name, keep in zip(FIELDS, present, strict=True)
            if keep
        }
        if any(present) and not all(present):
            with pytest.raises(ValueError, match="all or none"):
                make_outcome(**kwargs)
        else:
            make_outcome(**kwargs)


@pytest.mark.parametrize("name", FIELDS[:-1])
@pytest.mark.parametrize("shape", [(3,), (2, 3), (3, 2), (3, 3, 1)])
def test_claim_shapes(name, shape):
    fields = claim_fields()
    fields[name] = jnp.zeros(shape, dtype=bool if "p_" not in name else float)
    with pytest.raises(ValueError, match="shape"):
        make_outcome(**fields)


@pytest.mark.parametrize("name", FIELDS[:4])
@pytest.mark.parametrize("value", [-0.01, 1.01, np.nan, np.inf])
def test_claim_probability_values(name, value):
    fields = claim_fields()
    fields[name] = jnp.full((3, 3), value)
    with pytest.raises(ValueError, match=r"\[0, 1\]"):
        make_outcome(**fields)


@pytest.mark.parametrize("positive", [0.0, 1.0])
def test_claim_probability_endpoints(positive):
    fields = claim_fields()
    fields.update(
        claim_p_pos=jnp.full((3, 3), positive),
        claim_p_neg=jnp.full((3, 3), 1 - positive),
        claim_p_wrong=jnp.full((3, 3), positive),
        claim_p_neg_wrong=jnp.full((3, 3), 1 - positive),
    )
    make_outcome(**fields)


@pytest.mark.parametrize(
    "changes",
    [
        {"claim_p_pos": 0.95},
        {"claim_p_wrong": 0.7},
        {"claim_p_neg_wrong": 0.2},
    ],
)
def test_claim_probability_relationships(changes):
    fields = claim_fields()
    fields.update({name: jnp.full((3, 3), value) for name, value in changes.items()})
    with pytest.raises(ValueError):
        make_outcome(**fields)


@pytest.mark.parametrize("value", ["", "a" * 63, "g" * 64, 123])
def test_hash_format(value):
    with pytest.raises(ValueError, match="profile_hash"):
        make_outcome(profile_hash=value)


@pytest.mark.parametrize(
    "change, message",
    [
        ({"profile_hash": "0" * 64}, "profile_hash"),
        ({"claim_names": ("other", "radius_class", "water")}, "claim_names"),
        ({"p_pos": jnp.full(3, 0.7)}, "p_pos exceeds"),
    ],
)
def test_check_outcome_profile_failures(change, message):
    fields = claim_fields()
    fields.update(change)
    with pytest.raises(AssertionError, match=message):
        check_outcome(make_outcome(**fields), EXOEARTH_CANDIDATE)


def test_either_claim_uses_both_directions():
    fields = claim_fields()
    fields["claim_p_pos"] = fields["claim_p_pos"].at[:, 2].set(0.5)
    outcome = make_outcome(**fields)
    check_outcome(outcome, EXOEARTH_CANDIDATE)
    bad = eqx.tree_at(lambda o: o.p_pos, outcome, jnp.full(3, 0.61))
    with pytest.raises(AssertionError):
        check_outcome(bad, EXOEARTH_CANDIDATE)


def test_one_claim_yields_equal_profile_yields():
    fields = claim_fields(HABITABLE_ZONE_ONLY)
    outcome = make_outcome(
        p_wrong=jnp.full(3, 0.02), p_neg_wrong=jnp.full(3, 0.03), **fields
    )
    check_outcome(outcome, HABITABLE_ZONE_ONLY)
    report = certified_yield(outcome, [0.2, 0.7, 1.3], certificate="example")
    assert report.profile_hash == HABITABLE_ZONE_ONLY.content_hash()
    claim = report.claim_yields[HABITABLE_ZONE_ONLY.claim_names[0]]
    for channel in CHANNELS:
        assert claim[channel] == getattr(report, channel)
    assert claim["y_cert_neg"] == pytest.approx(report.y_neg - report.y_neg_wrong)
    assert CertifiedYield.from_json(report.to_json()) == report


def test_legacy_outcome_has_no_claim_yields():
    report = certified_yield(make_outcome(), [1, 2, 3], certificate="example")
    assert report.profile_hash is None
    assert report.claim_yields == {}
