"""A later revision of an acquisition supersedes the earlier one; the fold is once."""

import numpy as np
import pytest

from spaceodyssey.core import (
    CompiledExecutor,
    EventKind,
    HostExecutor,
    canonical_history,
    histories_equal,
    index_world,
    replay_belief,
    snapshot_view,
)
from spaceodyssey.core.enums import Audience
from tests.core.corpus import CORPUS_BY_NAME
from tests.core.helpers import N_STEPS_CORPUS, handle_event, one_world, worlds
from tests.core.record_helpers import scripted_study


def _run(revisions, values, audience=None):
    kw = {} if audience is None else {"audience": audience}
    study = scripted_study(
        ticks=(100, 200),
        values=values,
        ids=(7001, 7001),
        revisions=revisions,
        **kw,
    )
    truths, roots = worlds(1)
    host, trace = HostExecutor().run(
        study, index_world(truths, 0), roots[0], N_STEPS_CORPUS
    )
    cs, ts = CompiledExecutor().run(study, truths, roots, N_STEPS_CORPUS)
    return study, host, trace, (index_world(cs, 0), index_world(ts, 0))


def test_revision_supersedes():
    study, carry, _, _ = _run((0, 1), (2.0, 9.0))
    log = carry.log
    assert list(np.asarray(log.revision)[:2]) == [0, 1]
    assert list(np.asarray(log.superseded_by)[:2]) == [1, -1]
    # the observer accepted both rows and folded the first revision only
    assert np.all(np.asarray(log.accepted_tick)[:2] >= 0)
    assert int(carry.belief.n[0]) == 1 and float(carry.belief.mean[0]) == 2.0
    assert not bool(carry.dup_fold)
    # the policy view counts the revisions it can see
    view = snapshot_view(carry, Audience.POLICY, study.caps)
    assert list(np.asarray(view.log.superseded_by)[:2]) == [1, -1]


def test_a_revision_that_arrives_first_is_the_one_folded():
    _study, carry, _, _ = _run((1, 0), (9.0, 2.0))
    log = carry.log
    assert list(np.asarray(log.superseded_by)[:2]) == [-1, 1]
    assert int(carry.belief.n[0]) == 1 and float(carry.belief.mean[0]) == 9.0


def test_a_repeated_scripted_revision_is_refused_at_construction():
    # two rows of one acquisition at one revision are not a correction; the handler's
    # behavior for them is checked in test_derived_corrections
    with pytest.raises(ValueError, match=r"\(7001, 0\)"):
        _run((0, 0), (2.0, 9.0))


def test_observer_only_revisions_do_not_reach_the_policy():
    study, carry, _, _ = _run((0, 1), (2.0, 9.0), audience=Audience.OBSERVER)
    assert np.all(np.asarray(carry.log.released_tick)[:2] == -1)
    assert int(snapshot_view(carry, Audience.POLICY, study.caps).n_rows) == 0
    view = snapshot_view(carry, Audience.OBSERVER, study.caps)
    assert int(view.n_rows) == 2
    assert list(np.asarray(view.log.superseded_by)[:2]) == [1, -1]
    assert int(carry.belief.n[0]) == 1 and int(carry.pbelief.n[0]) == 0


@pytest.mark.parametrize("revisions", [(0, 1), (1, 0)])
def test_revisions_agree_between_executors_and_replay(revisions):
    study, carry, trace, (cc, ct) = _run(revisions, (2.0, 9.0))
    layout = study.observer.layout(study.caps)
    a = canonical_history(carry, trace, layout)
    host_h = a
    comp_h = canonical_history(cc, ct, layout)
    r = histories_equal(host_h, comp_h)
    assert r.structure_equal and r.max_int_diff == 0 and r.max_abs <= 1e-9
    assert list(a.ints["superseded_by"]) == list(comp_h.ints["superseded_by"])
    replayed = replay_belief(study, carry, trace)
    assert (
        np.asarray(replayed.mean).tobytes() == np.asarray(carry.belief.mean).tobytes()
    )
    assert np.asarray(replayed.n).tobytes() == np.asarray(carry.belief.n).tobytes()


@pytest.mark.parametrize(
    "name", ["baseline", "delayed_release", "cancel_live", "bundle", "coincident"]
)
def test_record_folded_once(name):
    study = CORPUS_BY_NAME[name].make_study()
    truths, roots = worlds(2)
    cs, _ = CompiledExecutor().run(study, truths, roots, N_STEPS_CORPUS)
    for w in range(2):
        c = index_world(cs, w)
        assert not bool(c.dup_fold)
        log = c.log
        takes = (
            np.asarray(log.valid)
            & (np.asarray(log.accepted_tick) >= 0)
            & (np.asarray(log.kind) == 1)
            & (np.asarray(log.status) == 0)
        )
        assert int(np.asarray(c.belief.n).sum()) == int(takes.sum()) >= 1


def test_a_second_release_of_one_row_is_counted_as_a_double_fold():
    study = CORPUS_BY_NAME["baseline"].make_study()
    truth, root = one_world()
    carry, _ = HostExecutor().run(study, truth, root, N_STEPS_CORPUS)
    before = carry.belief
    again = handle_event(
        carry, study, truth, root, EventKind.RELEASE, slot=0, audience=3
    )
    assert bool(again.dup_fold)
    assert np.array_equal(np.asarray(again.belief.n), np.asarray(before.n))
