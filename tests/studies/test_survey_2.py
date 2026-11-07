"""Survey 2, an adaptive revisit search: reductions, refresh and the run document."""

import dataclasses

import jax
import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, HostExecutor, replay_run
from spaceodyssey.core.enums import EffectKind, EventKind
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.studies import toy
from spaceodyssey.studies.survey_2 import (
    SURVEY_PLAN,
    RevisitPolicy,
    SearchMean,
    survey_2,
)
from tests.codec.run_cases import KINDS as REFRESH_KINDS
from tests.codec.run_cases import bindings, boundary
from tests.core.helpers import worlds
from tests.studies.survey_helpers import (
    action_projection,
    redump,
    round_trip,
    run_with_actions,
    same_actions,
    same_histories,
)

KINDS = {
    **REFRESH_KINDS,
    ("policy", "survey2.revisit", 1): RevisitPolicy,
    ("observer", "survey2.search_mean", 1): SearchMean,
}
# Individual reductions fit medium. Shared adaptive run comparisons have cold
# observations around 41 and 32 s; the refresh document round trip is also slow.
N_WORLDS = 3
N_STEPS = 80
EVERY = 100
IGNORING = RevisitPolicy(ignore_belief=True)


@pytest.fixture(scope="module")
def batch():
    return worlds(N_WORLDS)


@pytest.fixture(scope="module")
def survey_1_run(batch):
    truths, roots = batch
    return run_with_actions("host", toy.survey_1(), truths, roots, N_STEPS)


@pytest.fixture(scope="module")
def adaptive_off(batch):
    truths, roots = batch
    return run_with_actions("host", survey_2(), truths, roots, N_STEPS)


@pytest.fixture(scope="module")
def adaptive_on(batch):
    truths, roots = batch
    return run_with_actions(
        "host", survey_2(EVERY), truths, roots, N_STEPS, refit=boundary()
    )


@pytest.mark.slow  # survey 2 runs cold on the host executor
def test_a_belief_ignoring_revisit_policy_with_no_refresh_is_survey_1(
    batch, survey_1_run
):
    """Fails if the revisit policy, told to ignore its belief, changes any step,
    record, belief value or recorded draw of survey 1 (the full canonical history).
    """
    truths, roots = batch
    for executor in (HostExecutor, CompiledExecutor):
        run = executor().run_worlds(survey_2(policy=IGNORING), truths, roots, N_STEPS)
        assert bool(np.all(np.asarray(run.complete)))
        assert same_histories(run, survey_1_run.result, N_WORLDS)


@pytest.mark.slow  # one cold run per executor
@pytest.mark.parametrize("executor", ["host", "compiled"])
def test_with_refresh_the_actions_are_survey_1s_though_the_history_is_not(
    executor, batch, survey_1_run
):
    """Fails if a refresh wakes the policy or changes what it chose or read: the
    action projection must equal survey 1's. The full history differs by the refresh
    checkpoints, so equality of the whole history would be the wrong claim.
    """
    truths, roots = batch
    run = run_with_actions(
        executor,
        survey_2(EVERY, policy=IGNORING),
        truths,
        roots,
        N_STEPS,
        refit=boundary(),
    )
    assert same_actions(run, survey_1_run, N_WORLDS)
    assert not same_histories(run.result, survey_1_run.result, N_WORLDS)


@pytest.mark.slow  # a cold adaptive host run
def test_the_adaptive_policy_acts_on_its_belief(batch, survey_1_run, adaptive_off):
    """Fails if the adaptive policy takes survey 1's plan, which would make the two
    reductions above vacuous: some world must choose a different subject sequence.
    """
    assert not same_actions(adaptive_off, survey_1_run, N_WORLDS)
    subjects = [
        tuple(row[0] for row in action_projection(adaptive_off, w)[1])
        for w in range(N_WORLDS)
    ]
    assert any(s != (0, 1, 0, 1, 0) for s in subjects)


@pytest.mark.slow
def test_a_refresh_does_not_change_the_adaptive_actions(adaptive_off, adaptive_on):
    """Fails if refit checkpoints change the adaptive policy's decisions, subjects
    or payloads.
    """
    assert same_actions(adaptive_on, adaptive_off, N_WORLDS)
    assert not same_histories(adaptive_on.result, adaptive_off.result, N_WORLDS)


@pytest.mark.medium
def test_the_adaptive_rule_matches_an_eager_reference(batch):
    """Fails if a recorded choice differs from the rule recomputed eagerly (NumPy,
    no JAX) from the belief the policy was shown: an unseen subject first, else the
    undetected subject with the highest signal-to-noise ratio, stop when all are
    detected or the plan length is spent.
    """
    truths, roots = batch
    study = survey_2()
    noise = float(study.channel_model.noise[0])
    threshold = float(study.objective.threshold)
    kinds_seen = set()
    for world in range(N_WORLDS):
        views, effects = [], []
        HostExecutor().run(
            study,
            index_world(truths, world),
            roots[world],
            N_STEPS,
            view_log=views,
            effects_log=effects,
        )
        assert views
        for pos, (view, eff) in enumerate(zip(views, effects, strict=True)):
            mean = np.asarray(view.belief.mean)[:2]
            n = np.asarray(view.belief.n)[:2]
            snr = mean * np.sqrt(n) / noise
            detected = (snr > threshold) & (n > 0)
            if detected.all() or pos >= len(SURVEY_PLAN):
                expect = (int(EffectKind.STOP), None)
            elif (n == 0).any():
                expect = (int(EffectKind.OBSERVE), int(np.argmax(n == 0)))
            else:
                expect = (
                    int(EffectKind.OBSERVE),
                    int(np.argmax(np.where(detected, -np.inf, snr))),
                )
            got = int(np.asarray(eff.kind)[0])
            assert got == expect[0]
            if expect[1] is not None:
                assert int(np.asarray(eff.subject)[0]) == expect[1]
            kinds_seen.add(expect[1] if expect[1] is None else "observe")
    assert "observe" in kinds_seen


@pytest.mark.slow
def test_survey_2_agrees_between_executors_with_refresh(adaptive_on, batch):
    """Fails if the compiled executor differs from the host on the adaptive study
    with a refresh grid.
    """
    truths, roots = batch
    compiled = CompiledExecutor().run_worlds(
        survey_2(EVERY), truths, roots, N_STEPS, refit=boundary()
    )
    assert bool(np.all(np.asarray(compiled.complete)))
    assert same_histories(compiled, adaptive_on.result, N_WORLDS)


@pytest.mark.parametrize(
    "every",
    [
        pytest.param(0, marks=pytest.mark.medium),
        pytest.param(EVERY, marks=pytest.mark.slow),
    ],
)
def test_the_run_document_round_trips(every, batch):
    """Fails if a dumped run document, loaded and executed, does not reproduce the
    original history, or its re-dump has another content hash.
    """
    truths, roots = batch
    study = survey_2(every)
    kw = {"refit": boundary()} if every else {}
    bound = bindings() if every else None
    registry = toy.refresh_registry() if every else toy.toy_registry()
    original = HostExecutor().run_worlds(study, truths, roots, N_STEPS, **kw)
    doc, run = round_trip(study, registry, "host", KINDS, bindings=bound, **kw)
    assert redump(run, "host", KINDS, bindings=bound, **kw).content_hash == (
        doc.content_hash
    )
    again = run.execute(truths, roots, N_STEPS)
    assert same_histories(again, original, N_WORLDS)
    other = survey_2(every, policy=IGNORING)
    changed, _ = round_trip(other, registry, "host", KINDS, bindings=bound, **kw)
    assert changed.content_hash != doc.content_hash


@pytest.mark.slow
@pytest.mark.parametrize("executor", ["host", "compiled"])
def test_replay_of_the_adaptive_run(executor, batch):
    """Fails if keyed replay loses an adaptive belief, summary, checkpoint or product,
    or accepts a corrupted product payload on either executor.
    """
    truths, roots = batch
    study = survey_2(EVERY)
    refit = boundary()
    driver = {"host": HostExecutor, "compiled": CompiledExecutor}[executor]()
    live = driver.run_worlds(study, truths, roots, N_STEPS, refit=refit)
    assert bool(np.all(np.asarray(live.complete)))
    carry = index_world(live.carry, 0)
    trace = index_world(live.trace, 0)
    root = live.roots[0]
    out = replay_run(study, root, carry, trace, refit=refit, executor=executor)

    for replayed, original in (
        (out.belief, carry.belief),
        (out.pbelief, carry.pbelief),
    ):
        for got, expected in zip(
            jax.tree.leaves(replayed), jax.tree.leaves(original), strict=True
        ):
            np.testing.assert_array_equal(got, expected)

    products = np.flatnonzero(
        np.asarray(carry.log.valid) & (np.asarray(carry.log.kind) == toy.PRODUCT_KIND)
    )
    refresh_steps = np.asarray(trace.handled) & (
        np.asarray(trace.kind) == EventKind.REFRESH
    )
    # Each completed checkpoint has a hold step followed by its completion step.
    assert int(refresh_steps.sum()) % 2 == 0
    assert out.checkpoints == int(refresh_steps.sum()) // 2 > 0
    assert out.products == len(products) > 0
    assert not out.refused and not out.unfinished

    payload = carry.log.payload.at[products[0], 0].add(1.0)
    tampered = dataclasses.replace(
        carry, log=dataclasses.replace(carry.log, payload=payload)
    )
    with pytest.raises(ValueError, match=r"product row .* differs in payload"):
        replay_run(study, root, tampered, trace, refit=refit, executor=executor)
