"""Profile certificate compatibility and engine identity contracts."""

from dataclasses import replace
from typing import ClassVar

import equinox as eqx
import pytest

import spaceodyssey
from spaceodyssey import AbstractProfileCertificate, Claim, Edge
from spaceodyssey.errors import UnsupportedOperator, UnsupportedProfile
from spaceodyssey.profile import Profile
from tests.doubles import LinearGaussianTruthModel
from tests.test_claim_outcome import claim_fields
from tests.test_plan import make_outcome
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
                truth = make_truth()
                truth_model.quantities(truth.theta, truth.context)

        def __check_init__(self):
            if stage == "check_init":
                truth = make_truth()
                self.truth_model.quantities(truth.theta, truth.context)

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


def test_constructor_cannot_hide_profile_inputs():
    class HiddenInputs(ProfileCertificateDouble):
        def __init__(self, **kwargs):
            pytest.fail("unchecked constructor must not run")

    with pytest.raises(TypeError, match="expose profile and truth_model"):
        HiddenInputs(profile=make_profile(), truth_model=LinearGaussianTruthModel())


def test_profile_specific_restriction_runs_before_constructor():
    class RestrictedCertificate(ProfileCertificateDouble):
        @classmethod
        def check_profile_support(cls, profile, truth_model):
            if profile.support_rule == "joint":
                raise UnsupportedProfile("joint requires a different integrator")

        def __init__(self, profile, truth_model):
            pytest.fail("profile restrictions must run before construction")

    with pytest.raises(UnsupportedProfile, match=r"joint.*integrator"):
        RestrictedCertificate(
            replace(make_profile(), support_rule="joint"), LinearGaussianTruthModel()
        )


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


@pytest.mark.parametrize("name", ["profile", "truth_model"])
def test_profile_input_converters_are_unsupported(name):
    def forbidden(value):
        pytest.fail("unsupported converters must not run")

    if name == "profile":

        class ConvertedCertificate(ProfileCertificateDouble):
            profile: Profile = eqx.field(static=True, converter=forbidden)

    else:

        class ConvertedCertificate(ProfileCertificateDouble):
            truth_model: LinearGaussianTruthModel = eqx.field(converter=forbidden)

    with pytest.raises(TypeError, match=f"converters on {name} are unsupported"):
        ConvertedCertificate(make_profile(), LinearGaussianTruthModel())


def test_profile_construction_checks_support_once():
    checks = []

    class CountedCertificate(ProfileCertificateDouble):
        @classmethod
        def check_profile_support(cls, profile, truth_model):
            checks.append((profile, truth_model))

    certificate = CountedCertificate(make_profile(), LinearGaussianTruthModel())
    assert checks == [(certificate.profile, certificate.truth_model)]


def test_factory_field_cannot_hide_profile_parameter():
    class HiddenProfile(ProfileCertificateDouble):
        profile: Profile = eqx.field(static=True, default_factory=make_profile)

        def __init__(self, truth_model):
            pytest.fail("constructor without profile must not run")

    with pytest.raises(TypeError, match="expose profile and truth_model"):
        HiddenProfile(LinearGaussianTruthModel())


def test_custom_constructor_defaults_take_precedence_over_field_factories():
    default_profile, default_model = make_profile(), LinearGaussianTruthModel()

    def forbidden():
        pytest.fail("custom constructor defaults must not invoke field factories")

    class CustomDefaults(ProfileCertificateDouble):
        profile: Profile = eqx.field(static=True, default_factory=forbidden)
        truth_model: LinearGaussianTruthModel = eqx.field(default_factory=forbidden)

        def __init__(self, profile=default_profile, truth_model=default_model):
            self.profile = profile
            self.truth_model = truth_model

    certificate = CustomDefaults()
    assert certificate.profile is default_profile
    assert certificate.truth_model is default_model
