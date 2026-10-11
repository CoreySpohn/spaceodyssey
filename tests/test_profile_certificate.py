"""Profile certificate compatibility and engine identity contracts."""

from dataclasses import replace
from typing import ClassVar

import equinox as eqx
import pytest

import spaceodyssey
from spaceodyssey import AbstractProfileCertificate, Claim, Edge
from spaceodyssey.build import build
from spaceodyssey.engines.analytic import run_integrate
from spaceodyssey.errors import UnsupportedOperator, UnsupportedProfile
from spaceodyssey.profile import Profile
from spaceodyssey.testing import LinearGaussianTruth
from tests.doubles import make_spec
from tests.test_analytic import TestIntegrateEngine as _TestIntegrateEngine
from tests.test_claim_outcome import claim_fields
from tests.test_plan import make_outcome
from tests.test_truth import make_profile


class ProfileCertificateDouble(AbstractProfileCertificate):
    profile: Profile = eqx.field(static=True)
    truth_model: LinearGaussianTruth
    SUPPORT_RULES: ClassVar[frozenset[str]] = frozenset({"joint", "separate"})
    SUPPORTED_QUANTITIES: ClassVar[frozenset[str] | None] = frozenset({"x"})

    def integrate(self, predictive):
        return make_outcome(**claim_fields(self.profile))


@pytest.mark.parametrize("cause", ["model_quantity", "tier_quantity", "rule", "prior"])
def test_unsupported_profile_raises_before_any_computation(monkeypatch, cause):
    model = LinearGaussianTruth()

    def forbidden(*args, **kwargs):
        pytest.fail("model computation must not run for an unsupported profile")

    monkeypatch.setattr(LinearGaussianTruth, "quantities", forbidden)
    monkeypatch.setattr(LinearGaussianTruth, "expected_data", forbidden)
    monkeypatch.setattr(LinearGaussianTruth, "simulate", forbidden)
    profile = make_profile()
    if cause.endswith("quantity"):
        quantity = "unknown" if cause == "model_quantity" else "y"
        profile = replace(
            profile, claims=(Claim("claim", (Edge(quantity, 0, "above"),), "positive"),)
        )
        message = f"quantities.*{quantity}"
    elif cause == "rule":
        profile = replace(profile, support_rule="split")
        message = "support rule.*split"
    else:
        profile = replace(profile, prior_set_id="missing@1")
        message = "prior set.*missing@1"
    with pytest.raises(UnsupportedProfile, match=message):
        ProfileCertificateDouble(profile, model)


@pytest.mark.parametrize("operator", ["initial", "update", "propagate"])
def test_default_state_operators_raise(operator):
    certificate = ProfileCertificateDouble(make_profile(), LinearGaussianTruth())
    args = () if operator == "initial" else (None, None)
    with pytest.raises(UnsupportedOperator, match=operator):
        getattr(certificate, operator)(*args)


def test_unrestricted_quantities_and_public_export():
    class UnrestrictedCertificate(ProfileCertificateDouble):
        SUPPORTED_QUANTITIES: ClassVar[frozenset[str] | None] = None

    profile = make_profile(claims=(Claim("y", (Edge("y", 0, "above"),), "positive"),))
    certificate = UnrestrictedCertificate(profile, LinearGaussianTruth())
    assert certificate.profile is profile
    assert spaceodyssey.AbstractProfileCertificate is AbstractProfileCertificate
    with pytest.raises(TypeError):
        AbstractProfileCertificate()


@pytest.mark.parametrize("wrong_hash", [False, True])
def test_run_integrate_checks_profile_identity(wrong_hash):
    class IdentityCertificate(ProfileCertificateDouble):
        def integrate(self, predictive):
            fields = claim_fields(self.profile)
            if wrong_hash:
                fields["profile_hash"] = "0" * 64
            return make_outcome(**fields)

    profile = make_profile()
    certificate = IdentityCertificate(profile, LinearGaussianTruth())
    mission = build(make_spec(), _TestIntegrateEngine.config())
    mission = eqx.tree_at(lambda m: m.model.certification, mission, certificate)
    if wrong_hash:
        with pytest.raises(AssertionError, match="profile_hash"):
            run_integrate(mission, predictive=None, weights=[1, 1, 1])
    else:
        report = run_integrate(mission, predictive=None, weights=[1, 1, 1])
        assert report.profile_hash == profile.content_hash()
