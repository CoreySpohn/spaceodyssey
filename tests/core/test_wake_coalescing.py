"""Wakes coalesce per tick: one decision sees every record released at that tick."""

import numpy as np

from spaceodyssey.core.enums import AdmissionCode, EventKind
from spaceodyssey.core.executors.base import canonical_history
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.studies import toy as kinds
from tests.core.corpus import CORPUS_BY_NAME, frozen
from tests.core.helpers import one_world
from tests.core.mutants import PerWakeHostExecutor

N_STEPS = 40


def _run(executor, study):
    truth, root = one_world()
    views = []
    carry, trace = executor.run(study, truth, root, N_STEPS, view_log=views)
    layout = study.observer.layout(study.caps)
    return canonical_history(carry, trace, layout), views, trace


def _decisions_at(trace, tick):
    handled = np.asarray(trace.handled)
    return int(
        ((np.asarray(trace.tick) == tick) & np.asarray(trace.wake) & handled).sum()
    )


def test_two_releases_one_tick_one_decision():
    study = CORPUS_BY_NAME["bundle"].make_study()
    h, views, trace = _run(HostExecutor(), study)
    releases_at_260 = [e for e in h.events if e == (260, 1, EventKind.RELEASE)]
    assert len(releases_at_260) == 2
    assert _decisions_at(trace, 260) == 1
    (view,) = [v for v in views if int(v.tick) == 260]
    # the decision at 260 sees the record released at 130 and both released at 260
    assert int(view.n_rows) == 3


def test_per_wake_rule_would_decide_twice_and_the_first_misses_a_record():
    study = CORPUS_BY_NAME["bundle"].make_study()
    h, views, trace = _run(PerWakeHostExecutor(), study)
    assert _decisions_at(trace, 260) == 2
    first, second = [v for v in views if int(v.tick) == 260]
    assert int(first.n_rows) == 2 and int(second.n_rows) == 3
    # and the second decision spends a plan entry on a lane the first just took
    assert np.any(h.ints["admit_code"] == AdmissionCode.LANE_BUSY)


def test_wakes_at_distinct_ticks_stay_separate():
    # the staggered bundle releases at 185 and 260: two ticks, two decisions, and the
    # decision at 185 still meets the busy lane
    study = kinds.survey_1(policies=(frozen(bundle_at=1),))
    h, _views, trace = _run(HostExecutor(), study)
    assert _decisions_at(trace, 185) == 1 and _decisions_at(trace, 260) == 1
    assert np.any(h.ints["admit_code"] == AdmissionCode.LANE_BUSY)
