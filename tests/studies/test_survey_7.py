"""Survey 7, bundles: whole admission, one pointing, one shared background.

One module-scoped set of host runs of the toy study with two channels of lane 2 in one
bundle: the pair with a full balance, the pair with a balance short of both members,
the same two channels requested one after the other (the serializing control), and
one member alone on the short balance. All four share one program structure.
"""

import types

import jax
import numpy as np
import pytest

from spaceodyssey.core import HostExecutor
from spaceodyssey.core.enums import AdmissionCode
from spaceodyssey.core.preflight import preflight
from tests.core.helpers import one_world
from tests.studies.scenarios.survey_7 import (
    LANE,
    MEMBERS,
    BundleBackgroundObserver,
    one_background,
    pointing_end,
    short_balance,
    survey_7,
    survey_7_caps,
    two_backgrounds,
)

N_STEPS = 100
OK, NO_FUNDS = int(AdmissionCode.OK), int(AdmissionCode.NO_FUNDS)


def _run(study):
    truth, root = one_world(caps=survey_7_caps())
    run = HostExecutor().run(study, truth, root, N_STEPS)
    trace = run.trace
    first = np.flatnonzero(np.asarray(trace.handled) & np.asarray(trace.wake))[0]
    log = run.carry.log
    rows = np.flatnonzero(np.asarray(log.valid))
    return types.SimpleNamespace(
        study=study,
        carry=run.carry,
        first_codes=np.asarray(trace.admit_code)[first].tolist(),
        rows=rows,
        log=log,
        end=int(run.carry.ledger.lane_busy_until[LANE]),
        debited=np.asarray(run.carry.ledger.debited)[1, 0],
    )


@pytest.fixture(scope="module")
def runs():
    """Host runs of the pair: bundled, short balance, serialized, one member alone."""
    pair = survey_7(True)
    assert preflight(pair) == []
    return types.SimpleNamespace(
        bundled=_run(pair),
        short=_run(short_balance(pair)),
        serial=_run(survey_7(False)),
        alone=_run(short_balance(survey_7(False))),
    )


@pytest.mark.slow(reason="four host runs of one bundle study, 6 s cold")
def test_the_pair_is_admitted_or_refused_whole(runs):
    """Fails if a balance that covers one member of the pair but not both admits one
    of them (a record, or a debit on asset 1), or refuses the pair at a balance that
    covers both. Counted: two members admitted, two refused, and the member that was
    refused with the pair is admitted alone at the same balance.
    """
    assert runs.bundled.first_codes == [OK, OK]
    assert len(runs.bundled.rows) == len(MEMBERS) == 2
    assert runs.short.first_codes == [NO_FUNDS, NO_FUNDS]
    assert len(runs.short.rows) == 0
    assert runs.short.debited == 0.0
    assert runs.alone.first_codes[0] == OK and len(runs.alone.rows) == 1


@pytest.mark.slow(reason="shares the host runs of the whole-admission test")
def test_pointing_is_charged_once(runs):
    """Fails if the second member pays its own transition: the bundle's end must be
    the decision tick plus one transition plus the longest member occupancy, and the
    serialized request of the same channels must end later and break that rule.
    Counted: both members start together, in one bundle, and the serialized records
    are not bundled.
    """
    run, serial = runs.bundled, runs.serial
    starts = np.asarray(run.log.start_tick)[run.rows]
    assert len(set(starts.tolist())) == 1 and len(starts) == 2
    assert len(set(np.asarray(run.log.bundle)[run.rows].tolist())) == 1
    assert run.end == pointing_end(run.study, 0) == 55  # decided at tick 0
    assert int(starts[0]) == int(run.study.assets.transition_ticks[LANE])
    assert (np.asarray(serial.log.bundle)[serial.rows] == -1).all()
    assert len(serial.rows) == 2
    assert serial.end > run.end
    assert serial.end != pointing_end(serial.study, 0)


@pytest.mark.slow(reason="shares the host runs; eager folds of two records")
def test_the_shared_background_is_counted_once(runs):
    """Fails if the observer counts the background once per member: its belief after
    the bundle must equal the one-background reference and differ from the
    two-background one. Counted: two records folded in one bundle, so both
    references apply to the same two records; an observer that ignores `Record.bundle`
    reaches the two-background belief on the same records, and so does the
    serialized run, whose records are not bundled.
    """
    run = runs.bundled
    belief = run.carry.belief
    assert int(belief.n[0]) == int(belief.members[0]) == 2
    one, two = one_background(run.study), two_backgrounds(run.study)
    assert float(belief.prec[0]) == pytest.approx(one, rel=1e-12)
    assert abs(float(belief.prec[0]) - two) > 0.5
    # the same two records through an observer that does not group
    blind = BundleBackgroundObserver(grouped=False)
    state = blind.init(run.study.caps)
    for i in run.rows:
        rec = jax.tree_util.tree_map(lambda a, i=i: a[i], run.log)
        state = blind.fold(state, rec, run.study.caps)
    assert float(state.prec[0]) == pytest.approx(two, rel=1e-12)
    assert float(runs.serial.carry.belief.prec[0]) == pytest.approx(two, rel=1e-12)
