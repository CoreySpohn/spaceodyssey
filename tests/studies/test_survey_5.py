"""Survey 5, astrometric candidates: a later release is not read by a decision."""

import dataclasses

import numpy as np
import pytest

from spaceodyssey.codec import Bindings
from spaceodyssey.core import CompiledExecutor, HostExecutor, battery
from spaceodyssey.core.battery.result import tree_bits_equal
from spaceodyssey.core.enums import EffectKind
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.study import Clock
from spaceodyssey.studies import catalog
from spaceodyssey.studies.survey_5 import (
    CANDIDATE_ID_BASE,
    FOLLOW_SUBJECT,
    LAST_CANDIDATE_SLOT,
    ROSTER_COUNT_SLOT,
    AbsolutePlanPolicy,
    FollowUpPolicy,
    RosterObserver,
    survey_5,
    survey_5_registry,
)
from tests.codec.test_document import TOY_KINDS
from tests.core.helpers import worlds
from tests.studies.survey_helpers import redump, round_trip, same_histories
from tests.studies.test_survey_5_static import RELEASE

KINDS = {
    **TOY_KINDS,
    ("observer", "survey5.roster", 1): RosterObserver,
    ("policy", "survey5.follow_up", 1): FollowUpPolicy,
    ("policy", "survey5.absolute_plan", 1): AbsolutePlanPolicy,
}
pytestmark = pytest.mark.slow(reason="every test runs a cold study")

N_STEPS = 60
LATER = 100


class Run:
    """A host run with the decision views and effects it recorded."""

    def __init__(self, study, world=0):
        truths, roots = worlds(1)
        self.study = study
        self.views, self.effects = [], []
        self.result = HostExecutor().run(
            study,
            index_world(truths, world),
            roots[world],
            N_STEPS,
            view_log=self.views,
            effects_log=self.effects,
        )
        log = self.result.carry.log
        self.rows = np.flatnonzero(
            np.asarray(log.valid) & (np.asarray(log.channel) == 13)
        )

    def release_ticks(self):
        """Ticks at which the observer accepted the catalog rows."""
        return np.asarray(self.result.carry.log.accepted_tick)[self.rows].tolist()

    def roster_at_decisions(self):
        """`(tick, roster count)` of every decision the policy made."""
        return [(int(v.tick), int(v.belief.aux[ROSTER_COUNT_SLOT])) for v in self.views]


def premature_reads(run, release):
    """Decisions made before `release` that already read a candidate."""
    return [t for t, count in run.roster_at_decisions() if count and t < release]


@pytest.fixture(scope="module")
def base():
    return Run(survey_5())


@pytest.fixture(scope="module")
def later():
    return Run(survey_5(delay_ticks=catalog.CATALOG_DELAY_TICKS + LATER))


@pytest.fixture(scope="module")
def early():
    return Run(survey_5(delay_ticks=0))


def test_a_decision_before_the_release_tick_reads_an_empty_roster(base, early):
    """Fails if any decision made before the candidate's release tick sees the
    candidate, or the decision at the release tick does not. Negative control: a
    catalog released as soon as it ends is read before the declared release tick, so
    the same check refuses it.
    """
    assert base.release_ticks() == [RELEASE]
    roster = base.roster_at_decisions()
    assert [t for t, _ in roster][:3] == [0, 130, RELEASE]
    for tick, count in roster:
        assert count == (1 if tick >= RELEASE else 0), (tick, count)
    assert bool(base.result.complete)
    assert premature_reads(base, RELEASE) == []
    assert premature_reads(early, RELEASE) != []


def test_a_candidate_released_after_a_decision_does_not_change_it(base, later):
    """Fails if the catalog's later release moves a decision made before it: the
    decisions before the release are the same with the release `LATER` ticks later,
    and each run's first look at the candidate is at its own release tick.
    """
    assert later.release_ticks() == [RELEASE + LATER]
    for run, release in ((base, RELEASE), (later, RELEASE + LATER)):
        assert premature_reads(run, release) == []
        for tick, count in run.roster_at_decisions():
            assert count == int(tick >= release), (release, tick, count)
    before = [i for i, v in enumerate(base.views) if int(v.tick) < RELEASE]
    assert before == [0, 1]
    for i in before:
        assert int(base.views[i].tick) == int(later.views[i].tick)
        assert tree_bits_equal(base.effects[i], later.effects[i])
    # the decision that reads the candidate follows it up, at the release tick
    for run, release in ((base, RELEASE), (later, RELEASE + LATER)):
        assert int(run.views[2].tick) == release
        assert int(run.effects[2].kind[0]) == EffectKind.OBSERVE
        assert int(run.effects[2].subject[0]) == FOLLOW_SUBJECT


def test_candidates_have_their_own_identifiers_and_the_value_stays_private(base):
    """Fails if a candidate id is a subject index, the observer misses the candidate, or
    the policy reads the observer-only catalog value (probes of the policy, observer
    and scorer audiences).
    """
    carry = base.result.carry
    row = base.rows[0]
    assert int(carry.belief.aux[LAST_CANDIDATE_SLOT]) == CANDIDATE_ID_BASE
    assert int(carry.pbelief.aux[LAST_CANDIDATE_SLOT]) == CANDIDATE_ID_BASE
    ids = {int(v.belief.aux[LAST_CANDIDATE_SLOT]) for v in base.views} - {0}
    assert ids == {CANDIDATE_ID_BASE}
    assert not ids & set(range(base.study.caps.n_subjects))
    value = float(np.asarray(carry.log.payload)[row, 0])
    assert value != 0.0
    # observer: sees the value; policy: every decision and the final summary read zero
    assert float(carry.belief.aux[catalog.CATALOG_AUX_SLOT]) == value
    slot = catalog.CATALOG_AUX_SLOT
    assert all(float(v.belief.aux[slot]) == 0.0 for v in base.views)
    assert float(carry.pbelief.aux[slot]) == 0.0
    # scorer: the raw variate column is a scorer column no belief holds
    variate = float(np.asarray(carry.log.payload)[row, 1])
    assert variate not in set(np.asarray(carry.belief.aux).tolist())
    assert variate not in set(np.asarray(carry.pbelief.aux).tolist())


def test_survey_5_agrees_between_executors_and_round_trips():
    """Fails if the compiled executor differs from the host on the survey, or a run
    document does not reproduce it (history and content hash) on either executor.
    """
    truths, roots = worlds(2)
    study = survey_5()
    host = HostExecutor().run_worlds(study, truths, roots, N_STEPS)
    compiled = CompiledExecutor().run_worlds(study, truths, roots, N_STEPS)
    assert bool(np.all(np.asarray(compiled.complete)))
    assert same_histories(host, compiled, 2)
    slot = catalog.CATALOG_AUX_SLOT
    assert float(np.asarray(compiled.carry.pbelief.aux)[0, slot]) == 0.0
    docs = {}
    for executor, expected in (("host", host), ("compiled", compiled)):
        doc, run = round_trip(study, survey_5_registry(), executor, KINDS)
        assert doc.run == {"executor": executor}
        assert isinstance(run.bindings, Bindings)
        assert redump(run, executor, KINDS).content_hash == doc.content_hash
        assert same_histories(run.execute(truths, roots, N_STEPS), expected, 2)
        docs[executor] = doc
    # the executor is part of the document; a different catalog delay is another run
    assert docs["host"].content_hash != docs["compiled"].content_hash
    other, _ = round_trip(survey_5(delay_ticks=0), survey_5_registry(), "host", KINDS)
    assert other.content_hash != docs["host"].content_hash


def _daily(study):
    """One tick is one day, so a two-day delay is two ticks."""
    return dataclasses.replace(study, clock=Clock(86400.0))


def test_a_delayed_release_moves_belief_updates_and_nothing_else():
    """Fails if a two-tick delay of every release (the catalog's included) moves
    anything but the belief updates on an absolute plan: acquisitions, decisions and
    every record column must be unchanged, and the belief must move.
    """
    truths, roots = worlds(2)
    study = _daily(survey_5(policy=AbsolutePlanPolicy()))
    result = battery.latency_shift(study, truths, roots, N_STEPS, delay_s=2 * 86400.0)
    assert result, result.diagnostic
    assert result.detail["delay_ticks"] == 2
    assert result.detail["moved"] >= 2


def test_the_latency_check_rejects_a_policy_that_follows_candidates_up():
    """Negative control: the follow-up policy reacts to the release, so a delay moves
    its acquisitions and the replay-with-latency check fails."""
    truths, roots = worlds(2)
    result = battery.latency_shift(
        _daily(survey_5()), truths, roots, N_STEPS, delay_s=2 * 86400.0
    )
    assert not result
    assert "steps other than releases changed" in result.diagnostic
