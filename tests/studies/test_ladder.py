"""Ladder execution shares programs and noise while keeping truth private."""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hwoutils.testbudget.compiles import cache_hit_count, compile_count

from spaceodyssey.adapters.skyscapes import PopulationModel
from spaceodyssey.core.battery.result import tree_bits_equal, worlds_of
from spaceodyssey.core.battery.truth_swap import first_truth_dependent_release
from spaceodyssey.core.enums import RunStatus
from spaceodyssey.studies import bracket as br
from tests.studies import test_ladder_documents as documents
from tests.studies.bracket_helpers import build, with_grants

document_union_study = documents.union_study
single_step_910 = documents.single_step_910


def example_table(requirement=(5.0, 5.0)):
    return dataclasses.replace(
        documents.example_table(), requirement_snr=jnp.asarray(requirement)
    )


@pytest.fixture(scope="module")
def union_study(document_union_study):
    base = document_union_study
    return br.bracket_study(
        base.world_source,
        base.channel_model,
        horizon_d=30.0,
        budget_d=3.0,
        observer=dataclasses.replace(base.observer, grants=br.GRANT_PRESETS["oracle"]),
        policy=br.PackagePolicy(max_visits=3, max_spectra=3),
    )


@pytest.fixture(scope="module")
def fixture():
    return build(
        "replan",
        run=False,
        policy_kw={"max_visits": 3, "max_spectra": 3},
    ).study


def compiles_during(call):
    """Count backend programs, including persistent-cache loads, during one call."""
    before = compile_count() + cache_hit_count()
    jax.block_until_ready(call())
    return compile_count() + cache_hit_count() - before


def run(study):
    """Execute fresh seed-zero worlds with a small, declared refit shape."""
    boundary = br.bracket_boundary(
        study,
        lanes=1,
        n_grid=study.observer.prior.available.shape[1],
        n_first=8,
        n_estimate_systems=8,
        n_particles=8,
        n_round=8,
        n_rounds=2,
        n_centers=2,
        n_constants=2,
        max_rounds=2,
        forecast_quantiles=(0.5,),
    )
    source = study.world_source
    roots = jnp.stack(
        [jax.random.fold_in(jax.random.PRNGKey(0), i) for i in range(source.n_worlds)]
    )
    result = br.BoundExecutor().run_worlds(
        study, source.take(jnp.arange(source.n_worlds)), roots, 3000, refit=boundary
    )
    jax.block_until_ready(result)
    np.testing.assert_array_equal(result.status, RunStatus.COMPLETE)
    return result


def runs_equal(a, b):
    return tree_bits_equal((a.carry, a.trace, a.status), (b.carry, b.trace, b.status))


def row_log(result, study):
    """Attach the executor's band ordinal to the scorer's completed rows."""
    rows = br.row_log(result, study)
    modes, log = study.channel_model.mode_table, result.carry.log
    rows["band"] = np.asarray(
        [modes.band_index(modes.modes[m].name) for m in rows["mode"]], int
    )
    ordinals = []
    for w, s, t, c, b in zip(
        rows["world"],
        rows["star"],
        rows["end_tick"],
        rows["channel"],
        rows["band"],
        strict=True,
    ):
        hit = np.flatnonzero(
            np.asarray(log.valid[w])
            & (np.asarray(log.subject[w]) == s)
            & (np.asarray(log.end_tick[w]) == t)
            & (np.asarray(log.channel_index[w]) == c)
        )
        assert hit.size == 1
        ordinals.append(int(log.group_ordinal[w, hit[0], b]))
    rows["ordinal"] = np.asarray(ordinals, int)
    return rows


def matched_unit_variates(a, b):
    """Join uncensored count residuals by world, host, identity, band and ordinal."""

    def indexed(rows):
        out = {}
        for r, band in enumerate(rows["band"]):
            for slot, identity in enumerate(rows["identity"][r]):
                variance = rows["true_variance"][r, slot, band]
                counts = rows["counts"][r, slot, band]
                truth = rows["true_counts"][r, slot, band]
                if variance <= 0 or not np.isfinite([variance, counts, truth]).all():
                    continue
                key = (
                    rows["world"][r],
                    rows["star"][r],
                    identity,
                    band,
                    rows["ordinal"][r],
                )
                assert key not in out
                out[key] = (counts - truth) / np.sqrt(variance)
        return out

    left, right = indexed(a), indexed(b)
    shared = sorted(left.keys() & right.keys())
    return np.asarray([left[k] for k in shared]), np.asarray([right[k] for k in shared])


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


@pytest.mark.slow(reason="two ladder entries on one compiled program")
def test_a_traced_ladder_compiles_once_and_is_deterministic(union_study):
    """Fails if a traced requirement recompiles or a rerun differs."""
    a, b = br.ladder_studies(
        union_study,
        (example_table(requirement=(5.0, 5.0)), example_table(requirement=(7.0, 5.0))),
    )
    n0 = compiles_during(lambda: run(a))
    n1 = compiles_during(lambda: run(b))
    assert n0 >= 1 and n1 == 0 and runs_equal(run(a), run(a))


@pytest.mark.slow(reason="two ladder entries that diverge in sequence")
def test_ladder_entries_share_the_draw_of_each_observation_identity(union_study):
    """Fails if two entries draw different noise for the same star, slot, band
    and ordinal.
    """
    a, b = (
        row_log(run(s), s)
        for s in br.ladder_studies(union_study, (example_table(), single_step_910()))
    )
    assert a["mode"].tolist() != b["mode"].tolist()
    za, zb = matched_unit_variates(a, b)
    assert za.size > 0
    np.testing.assert_allclose(za, zb, rtol=1e-12)


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
