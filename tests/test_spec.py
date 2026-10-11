"""Tests for the declarative layer: refs, config, spec, json, hashing."""

import pytest

from spaceodyssey.spec import (
    FidelityConfig,
    FileRef,
    MissionSpec,
    SeamChoice,
    SeamRef,
    config_from_json,
    content_hash,
    spec_from_json,
    to_json,
)


def make_spec(**overrides) -> MissionSpec:
    """A representative MissionSpec for round-trip tests."""
    spec = MissionSpec.make(
        "survey-smoke",
        1825.0,
        budgets={"time_d": 365.0},
        bands=("500nm", "700nm"),
        modes=("imaging", "ifs"),
        catalog=FileRef(path="catalogs/targets.ecsv", sha256="abc123"),
        target_priors={7: FileRef(path="priors/7.json")},
        exozodi_model="fixed",
    )
    if overrides:
        from dataclasses import replace

        spec = replace(spec, **overrides)
    return spec


def make_config() -> FidelityConfig:
    """A representative FidelityConfig with one model override."""
    return FidelityConfig.make(
        "sample",
        observation=SeamRef.make("observation", "summary_rates", floor_snr=0.5),
        certification=SeamChoice(
            truth=SeamRef.make("certification", "snr_threshold", threshold=7.0),
            model=SeamRef.make("certification", "snr_threshold", threshold=5.0),
        ),
    )


class TestConstruction:
    """Normalization and access helpers."""

    def test_make_sorts_params(self):
        """SeamRef.make sorts keyword parameters into stable pairs."""
        ref = SeamRef.make("observation", "toy", zeta=1, alpha=2)
        assert ref.params == (("alpha", 2), ("zeta", 1))
        assert ref.params_dict == {"alpha": 2, "zeta": 1}

    def test_params_must_be_primitive(self):
        """Non-primitive, non-sequence parameters are rejected with guidance."""
        with pytest.raises(TypeError, match="primitive"):
            SeamRef.make("observation", "toy", grid={"a": 1})

    def test_sequence_params_canonicalize(self):
        """Lists of primitives become tuples and survive the json round trip."""
        from spaceodyssey.spec import config_from_json, to_json

        ref = SeamRef.make("context", "toy", target_ids=[3, 1, 2])
        assert ref.params_dict["target_ids"] == (3, 1, 2)
        config = FidelityConfig.make("sample", context=ref)
        assert config_from_json(to_json(config)) == config

    def test_config_make_normalizes_refs(self):
        """A bare SeamRef becomes a SeamChoice with truth == model."""
        config = make_config()
        obs = config.choice("observation")
        assert isinstance(obs, SeamChoice)
        assert obs.model is None
        assert obs.model_ref is obs.truth

    def test_config_choice_missing_seam(self):
        """Unchosen seams return None."""
        assert make_config().choice("policy") is None

    def test_model_override_kept(self):
        """An explicit model ref survives normalization."""
        cert = make_config().choice("certification")
        assert cert.model is not None
        assert cert.model_ref.params_dict["threshold"] == 5.0

    def test_spec_make_normalizes(self):
        """Budgets, priors, and extra collapse to sorted pairs."""
        spec = make_spec()
        assert spec.budgets_dict == {"time_d": 365.0}
        assert spec.extra == (("exozodi_model", "fixed"),)
        assert spec.target_priors[0][0] == 7


class TestJsonRoundTrip:
    """Canonical serialization preserves equality."""

    def test_spec_round_trip(self):
        """MissionSpec survives to_json/spec_from_json exactly."""
        spec = make_spec()
        assert spec_from_json(to_json(spec)) == spec

    def test_config_round_trip(self):
        """FidelityConfig with nested choice pairs survives the round trip."""
        config = make_config()
        assert config_from_json(to_json(config)) == config

    def test_wrong_type_rejected(self):
        """The typed deserializers check what they decoded."""
        with pytest.raises(TypeError, match="MissionSpec"):
            spec_from_json(to_json(make_config()))


class TestContentHash:
    """Content hashes identify declarative objects."""

    def test_equal_objects_equal_hash(self):
        """Two equal specs hash identically."""
        assert content_hash(make_spec()) == content_hash(make_spec())
        assert len(content_hash(make_spec())) == 12

    def test_param_change_changes_hash(self):
        """Any field change moves the hash."""
        assert content_hash(make_spec()) != content_hash(make_spec(duration_d=1826.0))

    def test_round_trip_preserves_hash(self):
        """Hashing after a json round trip gives the same identity."""
        spec = make_spec()
        assert content_hash(spec_from_json(to_json(spec))) == content_hash(spec)
