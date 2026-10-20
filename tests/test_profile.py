import dataclasses
import hashlib
import json
import math
import os
import subprocess
import sys

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.special import ndtri

from spaceodyssey import Claim, Edge, Profile, profiles

REFERENCE = profiles.EXOEARTH_CANDIDATE


def test_hash_stable_across_processes():
    code = (
        "from spaceodyssey.profiles import EXOEARTH_CANDIDATE; "
        "print(EXOEARTH_CANDIDATE.content_hash())"
    )
    hashes = [
        subprocess.check_output(
            [sys.executable, "-c", code],
            env={**os.environ, "PYTHONHASHSEED": seed},
            text=True,
        ).strip()
        for seed in ("1", "987654")
    ]
    assert hashes == [REFERENCE.content_hash()] * 2
    assert len(hashes[0]) == 64
    int(hashes[0], 16)


@pytest.mark.parametrize(
    "changes",
    [
        {"alpha": 0.01},
        {"support_rule": "separate"},
        {"prior_set_id": "alternative@1"},
        {"claims": REFERENCE.claims[::-1]},
        {"claims": REFERENCE.claims[:1]},
        {"claims": (dataclasses.replace(profiles.WATER, name="vapor"),)},
        {"claims": (dataclasses.replace(profiles.WATER, counted="positive"),)},
        {"claims": (Claim("water", (Edge("other", -4.0, "above"),), "either"),)},
        {
            "claims": (
                Claim("water", (Edge("log10_h2o_vmr", -3.0, "above"),), "either"),
            )
        },
        {
            "claims": (
                Claim("water", (Edge("log10_h2o_vmr", -4.0, "below"),), "either"),
            )
        },
    ],
)
def test_content_changes_hash(changes):
    # Compare nested mutations to the same one-claim baseline.
    baseline = REFERENCE
    if "claims" in changes and len(changes["claims"]) == 1:
        if changes["claims"][0].name in ("water", "vapor"):
            baseline = dataclasses.replace(REFERENCE, claims=(profiles.WATER,))
    assert (
        dataclasses.replace(baseline, **changes).content_hash()
        != baseline.content_hash()
    )


def test_name_is_label_and_canonical_hash():
    profile = Profile(
        "label",
        (Claim("c", (Edge("x", 1.5, "above"),), "either"),),
        0.05,
        support_rule="joint",
        prior_set_id="example@1",
        evidence_rule="every_claim",
    )
    renamed = dataclasses.replace(profile, name="another label")
    assert renamed.content_hash() == profile.content_hash()
    payload = {
        "schema": "spaceodyssey.Profile/1",
        "alpha": (0.05).hex(),
        "support_rule": "joint",
        "prior_set_id": "example@1",
        "evidence_rule": "every_claim",
        "claims": [
            {
                "name": "c",
                "counted": "either",
                "edges": [{"quantity": "x", "threshold": (1.5).hex(), "side": "above"}],
            }
        ],
    }
    expected = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    assert profile.content_hash() == expected


INVALID = [
    (Edge, ("", 0.0, "above")),
    (Edge, (1, 0.0, "above")),
    (Edge, ("x", float("nan"), "above")),
    (Edge, ("x", float("inf"), "above")),
    (Edge, ("x", -float("inf"), "above")),
    (Edge, ("x", "0", "above")),
    (Edge, ("x", True, "above")),
    (Edge, ("x", 0.0, "left")),
    (Edge, ("x", 0.0, np.array("above"))),
    (Claim, ("c", (Edge("x", 0.0, "above"),), np.array("either"))),
    (Claim, ("", (Edge("x", 0.0, "above"),), "positive")),
    (Claim, ("c", [], "positive")),
    (Claim, ("c", (), "positive")),
    (Claim, ("c", (Edge("x", 0.0, "above"),) * 3, "positive")),
    (Claim, ("c", ("bad",), "positive")),
    (Claim, ("c", (Edge("x", 0.0, "above"),), "negative")),
    (Claim, ("c", (Edge("x", 0.0, "above"), Edge("x", 1.0, "above")), "positive")),
    (Claim, ("c", (Edge("x", 1.0, "above"), Edge("x", 0.0, "below")), "positive")),
    (Claim, ("c", (Edge("x", 0.0, "above"), Edge("x", 0.0, "below")), "positive")),
]


@pytest.mark.parametrize("constructor,args", INVALID)
def test_invalid_edge_and_claim_constructions(constructor, args):
    with pytest.raises(ValueError):
        constructor(*args)


@pytest.mark.parametrize(
    "changes",
    [
        {"alpha": value}
        for value in (0, 0.5, -0.1, float("nan"), float("inf"), "0.1", True)
    ]
    + [
        {"name": ""},
        {"name": 1},
        {"prior_set_id": ""},
        {"prior_set_id": None},
        {"claims": []},
        {"claims": ()},
        {"claims": ("bad",)},
        {"claims": (profiles.WATER, profiles.WATER)},
        {"support_rule": "unknown"},
        {"support_rule": np.array("joint")},
        {"evidence_rule": np.array("every_claim")},
        {"evidence_rule": "joint"},
        {"evidence_rule": "None"},
    ],
)
def test_invalid_profile_constructions(changes):
    with pytest.raises(ValueError):
        dataclasses.replace(REFERENCE, **changes)


def test_reference_truth_on_ten_thousand_random_planets():
    rng = np.random.default_rng(712)
    a = rng.uniform(0.4, 2.5, 10**4)
    radius = rng.uniform(0.3, 2.0, 10**4)
    water = rng.uniform(-8.0, -1.0, 10**4)
    q = {
        "ln_a_eeid": jnp.asarray(np.log(a)),
        "ln_rp_re": jnp.asarray(np.log(radius)),
        "ln_rp_sqrt_a_eeid": jnp.asarray(np.log(radius * np.sqrt(a))),
        "log10_h2o_vmr": jnp.asarray(water),
    }
    expected = np.stack(
        [
            (a >= 0.95) & (a <= 1.67),
            (radius >= 0.8 / np.sqrt(a)) & (radius <= 1.4),
            water >= -4.0,
        ],
        axis=-1,
    )
    np.testing.assert_array_equal(jax.jit(REFERENCE.claim_truth)(q), expected)
    np.testing.assert_array_equal(
        REFERENCE.truth_positive(q), expected[:, :2].all(axis=-1)
    )


def test_edges_inclusive_and_arbitrary_batch_shape():
    q = {
        "ln_a_eeid": jnp.array([[math.log(0.95), math.log(1.67)]]),
        "ln_rp_sqrt_a_eeid": jnp.full((1, 2), math.log(0.8)),
        "ln_rp_re": jnp.full((1, 2), math.log(1.4)),
        "log10_h2o_vmr": jnp.full((1, 2), -4.0),
    }
    np.testing.assert_array_equal(REFERENCE.claim_truth(q), np.ones((1, 2, 3), bool))
    for claim in REFERENCE.claims:
        for edge in claim.edges:
            values = dict(q)
            direction = -jnp.inf if edge.side == "above" else jnp.inf
            # Test only the boundary of this edge, with the other edge satisfied.
            values[edge.quantity] = jnp.full(
                (1, 2), jnp.nextafter(jnp.asarray(edge.threshold), direction)
            )
            column = REFERENCE.claim_names.index(claim.name)
            assert not bool(jnp.any(REFERENCE.claim_truth(values)[..., column]))
    assert bool(profiles.WATER_ONLY.truth_positive({"log10_h2o_vmr": jnp.array(-8.0)}))


def test_equal_profile_static_field_does_not_retrace():
    class Holder(eqx.Module):
        profile: Profile = eqx.field(static=True)

    traces = []

    @jax.jit
    def evaluate(holder, value):
        traces.append(True)
        return value + holder.profile.alpha

    equal = dataclasses.replace(
        REFERENCE,
        claims=tuple(
            dataclasses.replace(c, edges=tuple(dataclasses.replace(e) for e in c.edges))
            for c in REFERENCE.claims
        ),
    )
    assert equal == REFERENCE and hash(equal) == hash(REFERENCE)
    assert evaluate(Holder(REFERENCE), 1.0) == evaluate(Holder(equal), 1.0)
    assert len(traces) == 1


@pytest.mark.parametrize("rule", ["joint", "separate", "split"])
def test_critical_value_and_error_bound(rule):
    profile = dataclasses.replace(REFERENCE, support_rule=rule)
    if rule == "joint":
        assert profile.critical_value() is None
    else:
        alpha = profile.alpha / (3 if rule == "split" else 1)
        assert profile.critical_value() == pytest.approx(ndtri(1 - alpha))
    assert profile.issued_error_bound() == pytest.approx(
        0.15 if rule == "separate" else 0.05
    )


def test_reference_variants_and_exports():
    assert REFERENCE.support_rule == "joint"
    assert REFERENCE.claim_names == ("habitable_zone", "radius_class", "water")
    assert REFERENCE.quantities == frozenset(
        ("ln_a_eeid", "ln_rp_re", "ln_rp_sqrt_a_eeid", "log10_h2o_vmr")
    )
    assert profiles.HABITABLE_ZONE_ONLY.claims == (profiles.HABITABLE_ZONE,)
    assert profiles.WATER_ONLY.claims == (profiles.WATER,)
    assert profiles.ZONE_WATER.claims == (profiles.HABITABLE_ZONE, profiles.WATER)
    assert profiles.FULL_POSITIVE.claims[-1].counted == "positive"
    assert profiles.OPTIMISTIC_ZONE.claims[0].edges == (
        Edge("ln_a_eeid", math.log(0.75), "above"),
        Edge("ln_a_eeid", math.log(1.77), "below"),
    )
    assert tuple(p.alpha for p in profiles.ALPHA_BAND) == (0.01, 0.05, 0.10)
    assert len({p.content_hash() for p in profiles.ALPHA_BAND}) == 3
    with pytest.raises(dataclasses.FrozenInstanceError):
        REFERENCE.alpha = 0.2


@pytest.mark.parametrize("scalar", [np.float64, np.float32, np.int64])
def test_real_scalars_are_normalized(scalar):
    edge = Edge("x", scalar(1), "above")
    assert type(edge.threshold) is float


def test_alpha_scalar_is_normalized():
    profile = dataclasses.replace(REFERENCE, alpha=np.float64(0.05))
    assert type(profile.alpha) is float
    assert profile == REFERENCE
    assert profile.content_hash() == REFERENCE.content_hash()


def test_signed_zero_and_edge_order_are_canonical():
    def profile(zero, reverse):
        edges = (Edge("x", zero, "above"), Edge("x", 1, "below"))
        claim = Claim("interval", edges[::-1] if reverse else edges, "positive")
        return dataclasses.replace(REFERENCE, claims=(claim,))

    baseline = profile(0.0, False)
    for other in (profile(-0.0, False), profile(0.0, True)):
        assert baseline == other
        assert hash(baseline) == hash(other)
        assert baseline.content_hash() == other.content_hash()
        assert other.claims[0].edges[0].threshold.hex() == (0.0).hex()


@pytest.mark.parametrize("missing", ["support_rule", "prior_set_id", "evidence_rule"])
def test_profile_requires_explicit_keyword_configuration(missing):
    fields = dataclasses.asdict(REFERENCE)
    fields["claims"] = REFERENCE.claims
    del fields[missing]
    with pytest.raises(TypeError, match=missing):
        Profile(**fields)


def test_profile_configuration_is_keyword_only():
    with pytest.raises(TypeError, match="positional"):
        Profile("label", REFERENCE.claims, 0.05, "joint", "example@1")


def test_profile_evidence_configuration_is_keyword_only():
    with pytest.raises(TypeError, match="positional"):
        Profile(
            "label",
            REFERENCE.claims,
            0.05,
            "every_claim",
            support_rule="joint",
            prior_set_id="example@1",
        )


def test_reference_profiles_do_not_require_prior_evidence():
    assert REFERENCE.evidence_rule == "none"
    for name in profiles.__all__:
        if name == "EXOEARTH_CANDIDATE_EVERY_CLAIM":
            continue
        value = getattr(profiles, name)
        candidates = value if isinstance(value, tuple) else (value,)
        for profile in candidates:
            if isinstance(profile, Profile):
                assert profile.evidence_rule == "none", profile.name


def test_every_claim_variant_preserves_reference_configuration():
    strict = profiles.EXOEARTH_CANDIDATE_EVERY_CLAIM
    assert "EXOEARTH_CANDIDATE_EVERY_CLAIM" in profiles.__all__
    assert strict.name == "exoearth_candidate_every_claim"
    assert strict.evidence_rule == "every_claim"
    for item in dataclasses.fields(REFERENCE):
        if item.name not in ("name", "evidence_rule"):
            assert getattr(strict, item.name) == getattr(REFERENCE, item.name)
    assert strict.content_hash() != REFERENCE.content_hash()


def test_reference_prior_identity():
    assert REFERENCE.prior_set_id == "exoearth-reference@1"


def test_float32_rounding_is_preserved_in_content_hash():
    single = dataclasses.replace(REFERENCE, alpha=np.float32(0.1))
    double = dataclasses.replace(REFERENCE, alpha=0.1)
    assert single.content_hash() != double.content_hash()
    claim = Claim("x", (Edge("x", np.float32(0.1), "above"),), "positive")
    rounded = dataclasses.replace(claim, edges=(Edge("x", 0.1, "above"),))
    assert dataclasses.replace(REFERENCE, claims=(claim,)).content_hash() != (
        dataclasses.replace(REFERENCE, claims=(rounded,)).content_hash()
    )


def test_evidence_rule_none_is_valid_and_changes_the_hash():
    strict = dataclasses.replace(REFERENCE, evidence_rule="every_claim")
    free = dataclasses.replace(strict, evidence_rule="none")
    assert free.evidence_rule == "none"
    assert free.content_hash() != strict.content_hash()
