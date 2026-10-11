"""Profile certificate compatibility and engine identity contracts."""

from dataclasses import replace
from typing import ClassVar

import equinox as eqx
import pytest

import spaceodyssey
from spaceodyssey import AbstractProfileCertificate, Claim, Edge
from spaceodyssey.build import build
from spaceodyssey.engines.analytic import run_integrate
from spaceodyssey.errors import UnsupportedOperator, UnsupportedProfile, WiringError
from spaceodyssey.plan import PlanPredictive
from spaceodyssey.profile import Profile
from spaceodyssey.testing import LinearGaussianTruthModel
from tests.doubles import make_spec
from tests.test_analytic import TestIntegrateEngine as _TestIntegrateEngine
from tests.test_claim_outcome import claim_fields
from tests.test_plan import make_outcome, make_plan
from tests.test_truth import make_profile, make_truth


class ProfileCertificateDouble(AbstractProfileCertificate):
    profile: Profile = eqx.field(static=True)
    truth_model: LinearGaussianTruthModel
    SUPPORT_RULES: ClassVar[frozenset[str]] = frozenset({"joint", "separate"})
    SUPPORTED_QUANTITIES: ClassVar[frozenset[str] | None] = frozenset({"x"})

    def integrate(self, predictive):
        return make_outcome(**claim_fields(self.profile))


@pytest.mark.parametrize("cause", ["model_quantity", "tier_quantity", "rule", "prior"])
@pytest.mark.parametrize("stage", ["init", "check_init"])
def test_unsupported_profile_raises_before_any_computation(monkeypatch, cause, stage):
    class ComputingCertificate(ProfileCertificateDouble):
        def __init__(self, profile, truth_model):
            self.profile = profile
            self.truth_model = truth_model
            if stage == "init":
                truth_model.quantities(make_truth().theta, make_truth().context)

        def __check_init__(self):
            if stage == "check_init":
                self.truth_model.quantities(make_truth().theta, make_truth().context)

    model = LinearGaussianTruthModel()

    def forbidden(*args, **kwargs):
        pytest.fail("model computation must not run for an unsupported profile")

    monkeypatch.setattr(LinearGaussianTruthModel, "quantities", forbidden)
    monkeypatch.setattr(LinearGaussianTruthModel, "expected_data", forbidden)
    monkeypatch.setattr(LinearGaussianTruthModel, "simulate", forbidden)
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
        ComputingCertificate(profile, model)


@pytest.mark.parametrize("operator", ["initial", "update", "propagate"])
def test_default_state_operators_raise(operator):
    certificate = ProfileCertificateDouble(make_profile(), LinearGaussianTruthModel())
    args = () if operator == "initial" else (None, None)
    with pytest.raises(UnsupportedOperator, match=operator):
        getattr(certificate, operator)(*args)


def test_unrestricted_quantities_and_public_export():
    class UnrestrictedCertificate(ProfileCertificateDouble):
        SUPPORTED_QUANTITIES: ClassVar[frozenset[str] | None] = None

    profile = make_profile(claims=(Claim("y", (Edge("y", 0, "above"),), "positive"),))
    certificate = UnrestrictedCertificate(profile, LinearGaussianTruthModel())
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
    certificate = IdentityCertificate(profile, LinearGaussianTruthModel())
    mission = build(make_spec(), _TestIntegrateEngine.config())
    mission = eqx.tree_at(lambda m: m.model.certification, mission, certificate)
    if wrong_hash:
        with pytest.raises(WiringError, match="profile_hash"):
            run_integrate(
                mission,
                predictive=PlanPredictive(make_plan(), make_truth()),
                weights=[1, 1, 1],
            )
    else:
        report = run_integrate(
            mission,
            predictive=PlanPredictive(make_plan(), make_truth()),
            weights=[1, 1, 1],
        )
        assert report.profile_hash == profile.content_hash()


@pytest.mark.parametrize(
    "cause", ["flux", "model", "dimension", "context", "plan_axis"]
)
def test_engine_validates_predictive_before_integrating(cause):
    class ComputingCertificate(ProfileCertificateDouble):
        def integrate(self, predictive):
            pytest.fail("integration must not run for invalid predictive inputs")

    plan, truth = make_plan(), make_truth()
    if cause == "flux":
        plan = make_plan(flux_reference="different")
        error, message = UnsupportedProfile, "flux reference"
    else:
        error = ValueError
        changes, message = {
            "model": ({"model_id": "other"}, "model_id"),
            "dimension": ({"theta": [[1, 2, 3]] * 3}, "wrong D"),
            "context": ({"context": {}}, "missing context"),
            "plan_axis": ({"n": 2}, "planet axis P"),
        }[cause]
        truth = make_truth(**changes)
    certificate = ComputingCertificate(make_profile(), LinearGaussianTruthModel())
    mission = build(make_spec(), _TestIntegrateEngine.config())
    mission = eqx.tree_at(lambda m: m.model.certification, mission, certificate)
    predictive = PlanPredictive(plan, make_truth())
    predictive = eqx.tree_at(lambda p: p.truth, predictive, truth)
    with pytest.raises(error, match=message):
        run_integrate(mission, predictive=predictive, weights=[1, 1, 1])


def test_constructor_cannot_hide_profile_inputs():
    class HiddenInputs(ProfileCertificateDouble):
        def __init__(self, **kwargs):
            pytest.fail("unchecked constructor must not run")

    with pytest.raises(TypeError, match="expose profile and truth_model"):
        HiddenInputs(profile=make_profile(), truth_model=LinearGaussianTruthModel())


def test_unrelated_profile_attribute_is_not_a_profile_certificate():
    class UnrelatedCertificate(spaceodyssey.ToyGaussianCertifier):
        profile: str = eqx.field(static=True, default="display label")

    certificate = UnrelatedCertificate(0.0, 0.05, 1.0)
    mission = build(make_spec(), _TestIntegrateEngine.config())
    mission = eqx.tree_at(lambda m: m.model.certification, mission, certificate)
    report = run_integrate(
        mission,
        predictive=PlanPredictive(make_plan(), make_truth().theta[:, 0]),
        weights=[1, 1, 1],
    )
    assert report.profile_hash is None


def test_signed_zero_profiles_share_jit_cache_and_engine_identity():
    traces = []

    class JittedCertificate(ProfileCertificateDouble):
        @eqx.filter_jit
        def integrate(self, predictive):
            traces.append(True)
            return make_outcome(**claim_fields(self.profile))

    mission = build(make_spec(), _TestIntegrateEngine.config())
    model = LinearGaussianTruthModel()
    hashes = []
    for zero in (0.0, -0.0):
        profile = make_profile(
            claims=(Claim("x", (Edge("x", zero, "above"),), "positive"),)
        )
        certificate = JittedCertificate(profile=profile, truth_model=model)
        configured = eqx.tree_at(lambda m: m.model.certification, mission, certificate)
        report = run_integrate(
            configured,
            predictive=PlanPredictive(make_plan(), make_truth()),
            weights=[1, 1, 1],
        )
        hashes.append(report.profile_hash)
    assert hashes[0] == hashes[1]
    assert len(traces) == 1


@pytest.mark.parametrize("supported", [False, True])
def test_constructor_resolves_factories_once_before_validation(supported):
    made = {}

    def profile_factory():
        assert "profile" not in made
        made["profile"] = make_profile(
            prior_set_id="uniform@1" if supported else "missing@1"
        )
        return made["profile"]

    def model_factory():
        assert "model" not in made
        made["model"] = LinearGaussianTruthModel()
        return made["model"]

    class FactoryCertificate(ProfileCertificateDouble):
        profile: Profile = eqx.field(static=True, default_factory=profile_factory)
        truth_model: LinearGaussianTruthModel = eqx.field(default_factory=model_factory)

        def __check_init__(self):
            if not supported:
                pytest.fail("unsupported factory inputs must fail before subclass work")

    if supported:
        certificate = FactoryCertificate()
        assert certificate.profile is made["profile"]
        assert certificate.truth_model is made["model"]
    else:
        with pytest.raises(UnsupportedProfile, match=r"prior set.*missing@1"):
            FactoryCertificate()
