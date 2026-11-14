"""Bracket studies carry their grants and criterion in the hashed document."""

import dataclasses

import equinox as eqx
import jax
import numpy as np
import pytest

from spaceodyssey.codec import KindResolver, dump_study, load_study
from spaceodyssey.plugins import installed_kinds
from spaceodyssey.studies.bracket import BRACKET_KINDS, GRANT_PRESETS, Grants
from tests.studies.bracket_helpers import build, fixture_source, with_grants

RESOLVER = KindResolver({**installed_kinds(), **BRACKET_KINDS})


def assert_leaves_equal(actual, expected):
    """Compare the static declarations and every dynamic leaf exactly."""
    assert jax.tree.structure(actual) == jax.tree.structure(expected)
    assert (
        eqx.partition(actual, eqx.is_array)[1]
        == eqx.partition(expected, eqx.is_array)[1]
    )
    for a, b in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert type(a) is type(b)
        if isinstance(a, jax.Array):
            assert (a.shape, a.dtype, a.weak_type) == (b.shape, b.dtype, b.weak_type)
        np.testing.assert_array_equal(a, b)


@pytest.fixture(scope="module")
def fixture_study():
    source = fixture_source()
    source = dataclasses.replace(
        source,
        stars=source.stars[:1],
        population=jax.tree.map(lambda x: x[:, :1], source.population),
        dropped=source.dropped[:, :1],
    )
    return build(arm="replan", run=False, source=source).study


def test_the_bracket_kinds_resolve_beside_the_installed_ones():
    """Fails if a class is registered twice or a reference collides with an
    installed kind.
    """
    installed = installed_kinds()
    assert not installed.keys() & BRACKET_KINDS.keys()
    resolver = KindResolver({**installed, **BRACKET_KINDS})
    for ref, cls in BRACKET_KINDS.items():
        assert resolver.resolve(ref) is cls
    assert {cls.__name__ for cls in BRACKET_KINDS.values()} >= {
        "SummaryObserver",
        "PackagePolicy",
        "ObservingModeChannel",
        "ExoVistaWorldSource",
        "CriterionObjective",
        "TransitionTable",
        "Band",
        "StarSpec",
        "SummaryScene",
        "ModeSpec",
        "ModeTable",
        "SupportRange",
        "PopulationModel",
        "PopulationTruth",
        "DecisionSummary",
        "PolicyState",
        "Grants",
        "EarthCandidateClaim",
        "OccurrenceTable",
        "PlanetTypes",
        "PopulationConfig",
    }
    for ref, name in (
        (("observer", "bracket.summary", 1), "SummaryObserver"),
        (("policy", "bracket.package", 1), "PackagePolicy"),
        (("channel_model", "jaxedith.observing_mode", 2), "ObservingModeChannel"),
        (("world_source", "skyscapes.exovista", 1), "ExoVistaWorldSource"),
        (("objective", "criterion.verdict", 1), "CriterionObjective"),
        (("criterion", "criterion.transition_table", 1), "TransitionTable"),
    ):
        assert resolver.resolve(ref).__name__ == name


@pytest.mark.slow(
    reason="physical fixture construction and document round trips take 18 s cold"
)
def test_a_bracket_study_round_trips_and_its_hash_names_the_grants_and_the_kernel(
    fixture_study,
):
    """Fails if a kind is unregistered, or a grant or the kernel is outside the
    hashed document.
    """
    doc = dump_study(fixture_study, RESOLVER)
    assert_leaves_equal(load_study(doc, RESOLVER), fixture_study)
    oracle = dump_study(with_grants(fixture_study, GRANT_PRESETS["oracle"]), RESOLVER)
    groups = ("ephemeris", "class")
    truth_k = dump_study(
        with_grants(fixture_study, Grants(catalog_groups=groups, kernel="truth")),
        RESOLVER,
    )
    shrink_k = dump_study(
        with_grants(fixture_study, Grants(catalog_groups=groups, kernel="shrinkage")),
        RESOLVER,
    )
    assert (
        oracle.content_hash != doc.content_hash
        and truth_k.content_hash != shrink_k.content_hash
    )
    for grants in (
        GRANT_PRESETS["oracle"],
        Grants(delivery="true_snr"),
        Grants(catalog_groups=groups),
        Grants(policy_reads_measurements=False),
    ):
        changed = with_grants(fixture_study, grants)
        changed_doc = dump_study(changed, RESOLVER)
        assert changed_doc.content_hash != doc.content_hash
        assert_leaves_equal(load_study(changed_doc, RESOLVER), changed)
    criterion = dataclasses.replace(
        fixture_study.observer.criterion, requirement_snr=[6.0]
    )
    ladder = dataclasses.replace(
        fixture_study,
        observer=dataclasses.replace(fixture_study.observer, criterion=criterion),
    )
    ladder_doc = dump_study(ladder, RESOLVER)
    assert ladder_doc.content_hash != doc.content_hash
    assert {k: v for k, v in doc.study.items() if k != "observer"} == {
        k: v for k, v in ladder_doc.study.items() if k != "observer"
    }
    assert {k: v for k, v in doc.study["observer"].items() if k != "criterion"} == {
        k: v for k, v in ladder_doc.study["observer"].items() if k != "criterion"
    }
    assert_leaves_equal(load_study(ladder_doc.to_json(), RESOLVER), ladder)
