"""Static generating truth stays behind the released-record boundary."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.adapters.skyscapes import PopulationModel
from spaceodyssey.core.battery.result import worlds_of
from spaceodyssey.core.battery.truth_swap import first_truth_dependent_release
from spaceodyssey.studies import bracket as br
from tests.studies.bracket_helpers import build, with_grants
from tests.studies.ladder_helpers import run


@pytest.fixture(scope="module")
def fixture():
    return build(
        "replan",
        run=False,
        policy_kw={"max_visits": 3, "max_spectra": 3},
    ).study


def with_declared(study, *, planning_model, belief_population):
    return dataclasses.replace(
        with_grants(study, br.GRANT_PRESETS["replan"]),
        planning_model=planning_model,
        belief_population=belief_population,
    )


def perturb(study, *, scenes, stored_worlds):
    """Change generating fluxes and slot order, retaining declared belief inputs."""
    assert stored_worlds == "shuffle_slots"
    channel = dataclasses.replace(
        study.channel_model,
        band_scenes=tuple(
            dataclasses.replace(s, star_flux_density_jy=s.star_flux_density_jy * scenes)
            for s in study.channel_model.band_scenes
        ),
    )
    source = dataclasses.replace(
        study.world_source,
        population=jax.tree.map(
            lambda x: jnp.flip(x, axis=2), study.world_source.population
        ),
    )
    return dataclasses.replace(study, channel_model=channel, world_source=source)


def assert_decisions_equal_before(a, b, step):
    """Compare event order, fingerprints, effects and admissions at the boundary."""
    for name in (
        "handled",
        "kind",
        "tick",
        "stage",
        "micro",
        "tie",
        "sid",
        "channel",
        "subject",
        "acquisition_index",
        "wake",
        "n_dec",
        "snapshot_fp",
        "eff_kind",
        "eff_disruptive",
        "eff_subject",
        "admit_code",
        "pol_u",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(a[1], name))[:step],
            np.asarray(getattr(b[1], name))[:step],
            err_msg=name,
        )
    assert np.count_nonzero(np.asarray(a[1].snapshot_fp)[:step]) > 0
    cutoff_tick = int(a[1].tick[step])
    logs = (a[0].log, b[0].log)
    admitted = [
        np.asarray(log.valid)
        & (np.asarray(log.kind) >= br.KIND_MEASUREMENT)
        & (np.asarray(log.start_tick) < cutoff_tick)
        for log in logs
    ]
    for name in (
        "kind",
        "subject",
        "channel_index",
        "acquisition_index",
        "start_tick",
        "end_tick",
        "cost",
        "charged",
    ):
        np.testing.assert_array_equal(
            np.asarray(getattr(logs[0], name))[admitted[0]],
            np.asarray(getattr(logs[1], name))[admitted[1]],
            err_msg=name,
        )


@pytest.mark.slow(reason="two fixture runs with static truth perturbed")
def test_static_truth_moves_no_bracket_decision_before_a_released_record_differs(
    fixture,
):
    """Fails if a non-oracle decision reads generating scenes, stored worlds
    or slot order.
    """
    a = with_declared(
        fixture,
        planning_model=fixture.channel_model,
        belief_population=PopulationModel.exovista(),
    )
    assert br.grants_in_force(a) == {
        "association": a.channel_model.association,
        "model": "declared",
        "population": "declared",
        "delivery": "measured",
        "catalog": "none",
        "policy_reads": "yes",
        "kernel": "posterior",
    }
    b = perturb(a, scenes=1.1, stored_worlds="shuffle_slots")
    left, right = run(a), run(b)
    for wa, wb in zip(worlds_of(left), worlds_of(right), strict=True):
        steps = first_truth_dependent_release(wa, wb, a.caps)
        permitted = [step for step in steps if step is not None]
        assert permitted, "the static-truth perturbation must change a released record"
        step = min(permitted)
        assert step > 0
        assert_decisions_equal_before(wa, wb, step)
