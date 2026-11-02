"""Toy source composition preserves the standalone truth draw."""

import jax
import jax.numpy as jnp
import jax.random as jr

from spaceodyssey.studies.toy import ToyTruth, draw_truth, survey_1


def test_world_source_draw_matches_standalone_draw():
    study = survey_1()
    key = jr.PRNGKey(7)
    expected = draw_truth(key, study.caps)
    actual = study.world_source.draw(key, study.caps)
    assert jnp.array_equal(actual.signal, expected.signal)
    assert len(jax.tree_util.tree_leaves(ToyTruth(signal=actual.signal))) == 1
