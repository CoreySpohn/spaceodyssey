"""Keyed replay with refresh equals the run, intermediates included (slow tier).

Every test runs a refresh study on an executor, so the file is in `SLOW_FILES`: the
compile of the chunked driver and the masked kernel exceeds the unit budget.
"""

import functools

import equinox as eqx
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import (
    EventKind,
    HostExecutor,
    RunStatus,
    index_world,
    replay_run,
)
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import F64, Capacity, EffectKind
from spaceodyssey.core.executors import replay
from spaceodyssey.core.records import pedigree_of
from spaceodyssey.core.refresh import MaskedBatchRefit, RefreshSchedule
from spaceodyssey.studies import oracle, toy
from spaceodyssey.studies.scripted import (
    SCRIPTED_CHANNEL,
    ScriptedRecords,
    scripted_refresh_study,
)
from tests.core.helpers import worlds
from tests.core.kit_helpers import ResampleSample
from tests.core.refresh_helpers import (
    ToyRefitAnswers,
    masked,
    refresh_study,
    run_pair,
)
from tests.core.test_derived_corrections import SUMMARY
from tests.core.test_derived_corrections import _study as derived_study

SERVICE_CHANNEL = 0  # QueryPolicy.service default


def replays(study, result, refit):
    """`replay_run` of every world of a run, as the list of `Replayed`."""
    return [
        replay_run(
            study,
            result.roots[w],
            index_world(result.carry, w),
            index_world(result.trace, w),
            refit=refit,
        )
        for w in range(result.roots.shape[0])
    ]


def product_pedigrees(carry, audience=0, subject=0):
    """`{ordinal: triples}` of the product rows of one audience and subject."""
    log = carry.log
    keep = np.asarray(log.valid) & (np.asarray(log.kind) == toy.PRODUCT_KIND)
    out = {}
    for slot in np.flatnonzero(keep):
        who, _, triples = pedigree_of(log, slot)
        if who == (1, 2)[audience] and int(log.subject[slot]) == subject:
            out[int(log.acquisition_index[slot])] = triples
    return out


@functools.cache
def toy_pair():
    return run_pair(refresh_study(100), 60, 2, 7)


# ---- replay equals the run -------------------------------------------------------


@pytest.mark.parametrize("side", [0, 1], ids=["host", "compiled"])
def test_replay_equals_the_run_on_the_toy_refit(side):
    """Fails if a replay of either executor's run misses a checkpoint, a product row,
    the trace projection or the final belief."""
    result = toy_pair()[side]
    for out in replays(refresh_study(100), result, masked()):
        assert out.checkpoints >= 3 and out.products > 0
        assert float(jnp.sum(out.belief.n)) > 0  # the installs, not an empty belief


# ---- same-tick corrections on both sides of a checkpoint -------------------------


def corrections_study():
    """Scripted corrections at the checkpoint tick and at the tick after it.

    Checkpoints fall at 100 and 200. At 100 (before the checkpoint, release stage) two
    acquisitions are corrected or created together; at 101 both are corrected again.
    """
    return scripted_refresh_study(
        ticks=(50, 100, 100, 100, 101, 101),
        values=(5.0, 8.0, 3.0, 4.0, 7.0, 6.0),
        ids=(7001, 7001, 7002, 7002, 7001, 7002),
        revisions=(0, 1, 0, 1, 2, 2),
        every=100,
    )


@pytest.mark.parametrize("side", [0, 1], ids=["host", "compiled"])
def test_replay_with_same_tick_corrections_around_a_checkpoint(side):
    """Fails if a correction on either side of a checkpoint reaches the wrong
    revision set, or the replay needs the final `superseded_by` to place it."""
    study = corrections_study()
    result = run_pair(study, 60, 2, 7)[side]
    for out in replays(study, result, masked()):
        assert out.checkpoints == 2
        assert float(out.belief.mean[0]) == 6.5  # the second corrections, 7.0 and 6.0
    # control: the sets differ at the two checkpoints, so the order matters
    sets = product_pedigrees(index_world(result.carry, 0))
    ch = SCRIPTED_CHANNEL
    assert sets[1] == ((ch, 7001, 1), (ch, 7002, 1))
    assert sets[2] == ((ch, 7001, 2), (ch, 7002, 2))


# ---- two Query answers on one channel --------------------------------------------


class AskTwice(oracle.QueryPolicy):
    """The plan, asking the host at decisions `query_at` and `second_at`."""

    second_at: int = eqx.field(default=3)

    def decide(self, pstate, view, study, key):
        eff, new_ps, u = self.inner.decide(pstate, view, study, key)
        ask = (view.n_dec == self.query_at) | (view.n_dec == self.second_at)
        kind = eff.kind.at[0].set(jnp.where(ask, EffectKind.QUERY, eff.kind[0]))
        eff = eqx.tree_at(
            lambda e: (e.kind, e.subject, e.channel_index),
            eff,
            (
                kind,
                eff.subject.at[0].set(jnp.where(ask, self.subject, eff.subject[0])),
                eff.channel_index.at[0].set(
                    jnp.where(ask, self.service, eff.channel_index[0])
                ),
            ),
        )
        return eff, jnp.where(ask, pstate, new_ps), u.astype(F64)


def test_replay_with_two_questions_on_one_channel():
    """Fails if two answers on one service channel collide in the replay (identity,
    pedigree or revision set), or a refit that reads them differs from the run."""
    registry = toy.refresh_registry()
    kind = registry.register(oracle.ANSWER_KIND)
    observer = ToyRefitAnswers()
    caps = registry.lower(Capacities(rows_per_subject=8, refit_lanes=2), observer)
    policy = AskTwice(inner=toy.FixedPlanPolicy(plan=oracle.DEFAULT_PLAN), query_at=1)
    schedule = RefreshSchedule(130, 0, *toy.PRODUCT_CHANNELS, toy.PRODUCT_KIND)
    study = toy.survey_1(caps, observer=observer, refresh=schedule, policies=(policy,))
    truths, roots = worlds(1)
    result = HostExecutor().run(
        study,
        index_world(truths, 0),
        roots[0],
        80,
        service=oracle.ToyOracleService(kind),
        refit=masked(),
    )
    log = result.carry.log
    answers = [
        int(log.acquisition_index[i])
        for i in np.flatnonzero(np.asarray(log.valid))
        if int(log.kind[i]) == kind
    ]
    assert answers == [1, 3]  # control: two questions, told apart by decision
    out = replay_run(study, roots[0], result.carry, result.trace, refit=masked())
    assert out.checkpoints >= 2
    # the answers are policy rows: its later pedigrees hold both, told apart by index
    sets = product_pedigrees(result.carry, audience=1).values()
    triples = [t for ts in sets for t in ts]
    assert {(SERVICE_CHANNEL, 1, 0), (SERVICE_CHANNEL, 3, 0)} <= set(triples)


# ---- a run cut by its step budget ------------------------------------------------


@pytest.mark.parametrize("side", [0, 1], ids=["host", "compiled"])
@pytest.mark.parametrize("where", ["on_hold", "after_completion", "mid"])
def test_replay_of_a_truncated_run_keeps_the_stale_belief(side, where):
    """Fails if a hold with no completion installs, or a truncated run's final belief
    differs from the replay's."""
    study = refresh_study(100)
    full = toy_pair()[0]
    kinds = np.asarray(full.trace.kind[0])
    # refresh steps alternate hold, completion, hold, ...
    steps = np.flatnonzero(
        (kinds == EventKind.REFRESH) & np.asarray(full.trace.handled[0])
    )
    budget = {
        "on_hold": int(steps[2]) + 1,  # the second checkpoint holds and is cut
        "after_completion": int(steps[3]) + 1,
        "mid": int(steps[4]) + 5,
    }[where]
    result = run_pair(study, budget, 1, 7)[side]
    assert int(result.status[0]) == RunStatus.TRUNCATED
    n_steps = int(
        (
            (np.asarray(result.trace.kind[0]) == EventKind.REFRESH)
            & np.asarray(result.trace.handled[0])
        ).sum()
    )
    if where == "on_hold":
        assert n_steps % 2 == 1  # control: the run was cut on a hold
    (out,) = replays(study, result, masked())
    assert out.checkpoints == n_steps // 2 >= 1


# ---- mutant replayers ------------------------------------------------------------


def test_a_replayer_that_reads_the_final_superseded_by_is_caught(monkeypatch):
    """Fails if the replay still matches the run when it hides every row the final log
    marks superseded, which is what reading `superseded_by` would do."""
    study = corrections_study()
    result = run_pair(study, 60, 1, 7)[0]
    replays(study, result, masked())  # control: the real replayer passes
    real = replay._visible

    def reads_final_superseded_by(carry, n_log, accepted, released):
        cols = real(carry, n_log, accepted, released)
        final = jnp.asarray(carry.log.superseded_by) < 0
        return {**cols, "log.valid": cols["log.valid"] & final}

    monkeypatch.setattr(replay, "_visible", reads_final_superseded_by)
    with pytest.raises(ValueError, match="checkpoint"):
        replays(study, result, masked())


def test_a_replayer_that_keys_a_refit_by_a_count_is_caught(monkeypatch):
    """Fails if the replay still matches a key-reading refit after its refit key is
    taken from the log length instead of the revision set."""
    study = toy.survey_refresh(100, observer=ResampleSample())
    refit = MaskedBatchRefit()
    result = run_pair(study, 60, 1, 7, refit=refit)[0]
    (out,) = replays(study, result, refit)  # control: the real keys reproduce the run
    assert out.checkpoints >= 3
    real = replay._request

    def keyed_by_log_length(view, root, study):
        request = real(view, root, study)
        count = jnp.asarray(view.n_log, request.keys.dtype)
        keys = jnp.broadcast_to(
            jnp.asarray(root, request.keys.dtype) + count, request.keys.shape
        )
        return eqx.tree_at(lambda r: r.keys, request, keys)

    monkeypatch.setattr(replay, "_request", keyed_by_log_length)
    with pytest.raises(ValueError):
        replays(study, result, refit)


# ---- fold-time row metadata and refusals -----------------------------------------


class FoldTimeMean(toy.ToyRunningMean):
    """The running mean that also sums the lifecycle fields it sees at fold time."""

    def fold(self, belief, rec, caps, key=None):
        out = super().fold(belief, rec, caps, key)
        seen = jnp.stack([rec.accepted_tick, rec.released_tick, rec.superseded_by])
        add = jnp.where(rec.valid, seen.astype(F64), 0.0)
        return eqx.tree_at(lambda b: b.aux, out, out.aux.at[1:4].add(add))


def metadata_study(ids, revisions, ticks):
    process = ScriptedRecords(
        ticks=ticks,
        values=(5.0,) * len(ticks),
        ids=ids,
        revisions=revisions,
        statuses=(0,) * len(ticks),
    )
    return toy.survey_1(
        policies=(toy.NullPolicy(),), process=process, observer=FoldTimeMean()
    )


@pytest.mark.parametrize("side", [0, 1], ids=["host", "compiled"])
def test_incremental_folds_receive_the_row_as_it_was_at_fold_time(side):
    """Fails if a replayed fold reads the final accepted tick, released tick or
    superseded_by of its row instead of the values before the release stamped them."""
    study = metadata_study((7001, 7001, 7002), (0, 1, 0), (1, 2, 3))
    result = run_pair(study, 60, 1, 7)[side]
    (out,) = replays(study, result, masked())
    aux = np.asarray(out.belief.aux)
    # control: two first revisions folded, each with all three fields still unset
    assert list(aux[1:4]) == [-2.0, -2.0, -2.0]
    log = result.carry.log
    assert (np.asarray(log.accepted_tick[0])[np.asarray(log.valid[0])] >= 0).all()


def test_a_refused_derived_correction_replays_as_a_refusal_not_a_cut():
    """Fails if an INVALID_CORRECTION run replays with a refusal or an unfinished
    checkpoint, or loses the run's overflow bits."""
    study = derived_study((SUMMARY, SUMMARY), (0, 1))
    result = run_pair(study, 20, 1, 7)[0]
    assert int(result.carry.overflow_mask[0]) & int(Capacity.INVALID_CORRECTION)
    (out,) = replays(study, result, masked())
    assert out.refused == () and not out.unfinished
    assert out.overflow_mask & int(Capacity.INVALID_CORRECTION)
