"""Survey 3, known planets: priors with a declared pedigree and window scheduling.

The merge-time guard (medium) is test (d) with replay: two table variants on both
executors, one compiled program per executor. The other run tests are slow: they share
one store of runs (`runs`), the survey on both executors for every table variant
below, all of one shape; `EXCLUDE_ROWS` is a static rule and compiles its own program
(test (c)); the epoch relation wraps the process and compiles another; the controls
replace the observer type. References are computed from the run's log and the table
alone, never from the observer's resolution.
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, HostExecutor, battery
from spaceodyssey.core.battery.result import tree_bits_equal
from spaceodyssey.core.enums import PUBLIC, Audience, RecordStatus
from spaceodyssey.core.executors.base import GO_TOLERANCE, index_world
from spaceodyssey.core.executors.replay import replay_run
from spaceodyssey.core.preflight import FindingCode, preflight
from spaceodyssey.core.snapshot import snapshot_fingerprint, snapshot_view
from spaceodyssey.studies import toy
from spaceodyssey.studies.premission import PAD, PedigreeRule, authored_table
from spaceodyssey.studies.prior_observer import PriorObserver
from spaceodyssey.studies.survey_3 import (
    EXTERNAL,
    FALLBACK_WAIT,
    HI_COLUMN,
    LO_COLUMN,
    N_OBSERVATIONS,
    NO_WINDOW,
    OLDEST,
    PRIOR_CHANNEL,
    RAW_CHANNEL,
    TARGET,
    WINDOW,
    WINDOW_PRIOR_INDEX,
    survey_3,
    survey_3_table,
)
from tests.core.helpers import worlds
from tests.core.prior_helpers import imported_beliefs
from tests.studies.survey_helpers import (
    run_on,
    run_with_actions,
    same_histories,
)

OBS, POL = int(Audience.OBSERVER), int(Audience.POLICY)
MEAS = toy.RECORD_MEASUREMENT
N_STEPS = 60
N_WORLDS = 2
PRIOR_3 = (TARGET, 4.0, 2.0)  # (subject, value, precision) of the windowed prior
PRIOR_4 = (1, -3.0, 1.0)  # the prior summarizing raw row 1
ROW_1 = (RAW_CHANNEL, 1)  # (channel, local index) of raw row 1
EXECUTORS = ("compiled", "host")
VARIANTS = {
    "none": dict(row_2=None, prior_3=False),  # no windowed prior
    "external": dict(row_2=None),  # prior 3 entirely external
    "hidden": dict(),  # row 2 observer-only, named by prior 3
    "raw_only": dict(row_2=None, prior_4=False),  # prior 4 not imported
    "public": dict(row_2=PUBLIC),  # row 2 public: both audiences drop prior 3
    "permuted": dict(order=(4, 2, 0, 3, 1)),  # "hidden", rows permuted
}


def study_of(name, rule=PedigreeRule.EXCLUDE_PRIOR):
    return survey_3(survey_3_table(**VARIANTS[name]), rule)


def run_store(names):
    """`({(variant, executor): ActionRun}, roots)` for `names` on both executors."""
    truths, roots = worlds(N_WORLDS, study_of("hidden").caps)
    out = {}
    for name in names:
        for executor in EXECUTORS:
            run = run_with_actions(executor, study_of(name), truths, roots, N_STEPS)
            assert np.asarray(run.result.complete).all(), (name, executor)
            out[name, executor] = run
    return out, roots


@pytest.fixture(scope="module")
def runs():
    """The run store of every variant (slow tests)."""
    return run_store(VARIANTS)


def carry_of(result, w):
    return index_world(result.carry, w)


def expected_means(carry, audience, priors=(), dropped=(), n_subjects=4):
    """Per subject mean from the run's log and the given priors alone.

    Each retained prior `(subject, value, precision)` adds its precision and precision
    times value; every completed measurement row the audience received (accepted for
    the observer, released to the policy) adds one and its value, except the imported
    rows named in `dropped` by `(channel, local index)`.
    """
    log = jax.tree_util.tree_map(np.asarray, carry.log)
    stamp = log.accepted_tick if audience == OBS else log.released_tick
    keep = log.valid & (stamp >= 0) & (log.kind == MEAS)
    keep &= log.status == RecordStatus.COMPLETED
    for channel, index in dropped:
        keep &= ~((log.channel == channel) & (log.acquisition_index == index))
    total, held = np.zeros(n_subjects), np.zeros(n_subjects)
    for s, value, precision in priors:
        total[s] += precision * value
        held[s] += precision
    np.add.at(total, log.subject[keep], log.payload[keep, 0])
    np.add.at(held, log.subject[keep], 1.0)
    return np.where(held > 0, total / np.where(held > 0, held, 1.0), 0.0)


def mean_failures(result, want, where):
    """Each audience's final mean against `want(carry, audience)`, per world."""
    failures = []
    for w in range(N_WORLDS):
        carry = carry_of(result, w)
        for audience, belief in ((OBS, carry.belief), (POL, carry.pbelief)):
            got = np.asarray(belief.mean)
            if not np.allclose(got, want(carry, audience), rtol=0, atol=GO_TOLERANCE):
                failures.append(f"{where}, world {w}, audience {audience}: {got}")
    return failures


def prior_moves_beliefs(with_prior, without, where):
    """(a): the windowed prior must move both beliefs of its subject, in every world."""
    failures = []
    for w in range(N_WORLDS):
        a, b = carry_of(with_prior, w), carry_of(without, w)
        for name in ("belief", "pbelief"):
            if float(getattr(a, name).mean[TARGET]) == float(
                getattr(b, name).mean[TARGET]
            ):
                failures.append(f"{where}, world {w}: the prior left {name} alone")
    return failures


def belief_failures(result, reference, where):
    """Every belief leaf of `result` against `reference`, in the go band ((b))."""
    failures = []
    for w in range(N_WORLDS):
        a, b = carry_of(result, w), carry_of(reference, w)
        for name in ("belief", "pbelief"):
            for leaf in ("mean", "n", "weight"):
                x = np.asarray(getattr(getattr(a, name), leaf), float)
                y = np.asarray(getattr(getattr(b, name), leaf), float)
                if not np.allclose(x, y, rtol=0, atol=GO_TOLERANCE):
                    failures.append(f"{where}, world {w}: {name}.{leaf} {x} != {y}")
    return failures


def prior_rows(carry, subject=TARGET):
    """`audience -> (window, released tick)` of the copies of `subject`'s prior."""
    log = jax.tree_util.tree_map(np.asarray, carry.log)
    rows = log.valid & (log.kind == WINDOW_PRIOR_INDEX) & (log.subject == subject)
    return {
        int(log.audience[i]): (
            (float(log.payload[i, LO_COLUMN]), float(log.payload[i, HI_COLUMN])),
            int(log.released_tick[i]),
        )
        for i in np.flatnonzero(rows)
    }


# ---- (a) to (c): the prior and the overlap rules -------------------------------------


@pytest.mark.slow  # the run store of every variant (K6)
def test_a_disjoint_external_prior_changes_both_beliefs(runs):
    """B7 row 3 (a). Fails if a prior whose pedigree is entirely external leaves the
    observer belief or the policy summary of its subject equal to the run without
    it, or if either audience's belief is not that prior plus the rows it received
    (prior 4, which summarizes a public raw row, is excluded for both).
    """
    store, _ = runs
    for executor in EXECUTORS:
        external = store["external", executor].result
        none = store["none", executor].result
        assert prior_moves_beliefs(external, none, executor) == []
        assert (
            mean_failures(
                external, lambda c, a: expected_means(c, a, [PRIOR_3]), executor
            )
            == []
        )
        assert mean_failures(none, lambda c, a: expected_means(c, a), executor) == []
        for w in range(N_WORLDS):
            carry = carry_of(external, w)
            assert float(carry.belief.weight[TARGET]) == PRIOR_3[2]
            assert float(carry.pbelief.weight[TARGET]) == PRIOR_3[2]


@pytest.mark.slow  # the run store of every variant (K6)
def test_exclude_prior_with_an_imported_source_equals_the_raw_only_run(runs):
    """B7 row 3 (b). Fails if a prior whose pedigree names an imported raw row
    released to both audiences is installed by either audience under `EXCLUDE_PRIOR`:
    both beliefs must equal those of the run without that prior, in the go band, with
    the named raw row folded by both.
    """
    store, _ = runs
    for executor in EXECUTORS:
        run = store["external", executor].result
        raw_only = store["raw_only", executor].result
        assert belief_failures(run, raw_only, executor) == []
        for w in range(N_WORLDS):
            carry = carry_of(run, w)
            # evidence: the prior was imported for both and its source was folded
            assert set(prior_rows(carry, PRIOR_4[0])) == {OBS, POL}
            assert int(carry.belief.n[1]) >= 1 and int(carry.pbelief.n[1]) >= 1


@pytest.mark.slow  # its own compiled program per executor (static rule; K6)
def test_exclude_rows_matches_independent_sufficient_statistics():
    """B7 row 3 (c). Fails if, under `EXCLUDE_ROWS`, either audience's belief is not
    its retained priors plus the rows it received less the raw row a retained prior
    summarizes, computed from the log; if that raw row is folded; or if the executors
    disagree.
    """
    study = study_of("external", PedigreeRule.EXCLUDE_ROWS)
    truths, roots = worlds(N_WORLDS, study.caps)
    results = {}
    for executor in EXECUTORS:
        run = run_on(executor, study, truths, roots, N_STEPS)
        assert np.asarray(run.complete).all()
        results[executor] = run
        want = lambda c, a: expected_means(c, a, [PRIOR_3, PRIOR_4], [ROW_1])  # noqa: E731
        assert mean_failures(run, want, executor) == []
        for w in range(N_WORLDS):
            carry = index_world(run.carry, w)
            log = jax.tree_util.tree_map(np.asarray, carry.log)
            row = log.valid & (log.channel == ROW_1[0])
            row &= log.acquisition_index == ROW_1[1]
            # the raw row was delivered to both audiences and folded by neither
            assert log.accepted_tick[row].tolist() == [0]
            assert log.released_tick[row].tolist() == [0]
            for belief in (carry.belief, carry.pbelief):
                assert float(belief.weight[1]) == PRIOR_4[2]
                assert int(belief.n[1]) == int(
                    (log.valid & (log.subject == 1) & (log.kind == MEAS)).sum() - 1
                )
    assert same_histories(results["host"], results["compiled"], N_WORLDS)


# ---- (d): audiences ------------------------------------------------------------------


def policy_failures(a, b, where):
    """Everything the policy saw or did, compared bit for bit between two runs."""
    failures = []
    for w in range(N_WORLDS):
        ca, cb = carry_of(a.result, w), carry_of(b.result, w)
        ta, tb = index_world(a.result.trace, w), index_world(b.result.trace, w)
        wa = np.asarray(ta.wake) & np.asarray(ta.handled)
        wb = np.asarray(tb.wake) & np.asarray(tb.handled)
        if (
            np.asarray(ta.snapshot_fp)[wa].tolist()
            != (np.asarray(tb.snapshot_fp)[wb].tolist())
            or np.asarray(ta.tick)[wa].tolist() != np.asarray(tb.tick)[wb].tolist()
        ):
            failures.append(f"{where}, world {w}: decision ticks or fingerprints")
        da, db = a.decisions[w], b.decisions[w]
        if len(da) != len(db) or not all(
            tree_bits_equal(x, y) for x, y in zip(da, db, strict=True)
        ):
            failures.append(f"{where}, world {w}: a decision view or effect differs")
        if not tree_bits_equal(ca.pbelief, cb.pbelief):
            failures.append(f"{where}, world {w}: the policy summary differs")
        study = study_of("hidden")
        va = snapshot_view(ca, POL, study.caps)
        vb = snapshot_view(cb, POL, study.caps)
        layout = study.observer.layout(study.caps)
        if not tree_bits_equal(va, vb) or int(snapshot_fingerprint(va, layout)) != int(
            snapshot_fingerprint(vb, layout)
        ):
            failures.append(f"{where}, world {w}: the final policy view differs")
    return failures


def written(carry):
    """Sorted `(kind, subject, channel, index, revision, audience, payload)` rows."""
    log = jax.tree_util.tree_map(np.asarray, carry.log)
    cols = (log.kind, log.subject, log.channel, log.acquisition_index)
    cols += (log.revision, log.audience)
    rows = [
        (*(int(c[i]) for c in cols), tuple(log.payload[i].tolist()))
        for i in np.flatnonzero(log.valid)
    ]
    return sorted(rows)


def replay_failures(store, roots, names):
    """Keyed replay of each world of `names` on both executors (D2)."""
    failures = []
    for name in names:
        for executor in EXECUTORS:
            run = store[name, executor]
            for w in range(N_WORLDS):
                carry = carry_of(run.result, w)
                # replay raises on any mismatch of the beliefs it rebuilds
                out = replay_run(
                    study_of(name), roots[w], carry, index_world(run.result.trace, w)
                )
                for got, want in (
                    (out.belief.weight, carry.belief.weight),
                    (out.pbelief.weight, carry.pbelief.weight),
                ):
                    if not np.array_equal(np.asarray(got), np.asarray(want)):
                        failures.append(f"{name}, {executor}, world {w}: priors")
                if float(out.pbelief.weight[TARGET]) != PRIOR_3[2]:
                    failures.append(f"{name}, {executor}, world {w}: policy prior")
    return failures


@pytest.mark.slow  # two variants, one cold program per executor
def test_an_observer_only_source_moves_no_policy_view_or_decision():
    """B7 row 3 (d), the hidden-row check of D5 and K3 over a whole run, with replay
    (D2). Fails if a public prior whose pedigree names an observer-only raw row
    changes any policy view, decision, effect, fingerprint or the policy summary
    against the run without that row; if the observer keeps the prior or the policy
    drops it (a raw row the policy never receives must not exclude for it); if the
    exclusion leaks out of the observer's own row (its window); if keyed replay from
    `init` and `init_summary` does not reproduce both runs, installed priors
    included; or if the executors disagree.
    """
    store, roots = run_store(("hidden", "external"))
    assert replay_failures(store, roots, ("hidden", "external")) == []
    for executor in EXECUTORS:
        hidden, external = store["hidden", executor], store["external", executor]
        assert policy_failures(hidden, external, executor) == []
        want = lambda c, a: expected_means(c, a, [PRIOR_3] if a == POL else [])  # noqa: E731
        assert mean_failures(hidden.result, want, executor) == []
        for w in range(N_WORLDS):
            carry = carry_of(hidden.result, w)
            # evidence: the observer folded the hidden row and dropped the prior
            assert float(carry.belief.weight[TARGET]) == 0.0
            assert float(carry.pbelief.weight[TARGET]) == PRIOR_3[2]
            assert int(carry.belief.n[TARGET]) == int(carry.pbelief.n[TARGET]) + 1
            copies = prior_rows(carry)
            assert copies[OBS][0] == NO_WINDOW and copies[POL][0] == WINDOW
    for name in ("hidden", "external"):
        assert same_histories(
            store[name, "host"].result, store[name, "compiled"].result, N_WORLDS
        ), name


@pytest.mark.slow  # the run store of every variant (K6)
def test_permuting_the_table_changes_no_decision_or_written_row(runs):
    """B7 row 3 (d), permutation. Fails if permuting the table's rows changes a
    decision or its effects, any written row (identity, audience, payload), or a
    belief beyond the go band.
    """
    store, _ = runs
    for executor in EXECUTORS:
        hidden = store["hidden", executor]
        permuted = store["permuted", executor]
        for w in range(N_WORLDS):
            assert hidden.decisions[w] and len(hidden.decisions[w]) == len(
                permuted.decisions[w]
            )
            for (_, ea), (_, eb) in zip(
                hidden.decisions[w], permuted.decisions[w], strict=True
            ):
                assert tree_bits_equal(ea, eb)
            assert written(carry_of(hidden.result, w)) == written(
                carry_of(permuted.result, w)
            )
        assert belief_failures(permuted.result, hidden.result, executor) == []


# ---- (e): retained priors are disjoint -----------------------------------------------


def two_summaries(imported):
    """A raw row of the target and two priors summarizing it, imported or not."""
    return authored_table(
        subject=[TARGET] * 3,
        local_index=[0, 0, 1],
        channel=[RAW_CHANNEL, PRIOR_CHANNEL, PRIOR_CHANNEL],
        kind=[MEAS, WINDOW_PRIOR_INDEX, WINDOW_PRIOR_INDEX],
        audience=[PUBLIC] * 3,
        payload=[[1.0, 0.0], [1.0, 1.0], [1.0, 1.0]],
        namespace=[0 if imported else PAD, 0, 0],
        pedigree=[
            (),
            ((0, RAW_CHANNEL, TARGET, 0, not imported),),
            ((0, RAW_CHANNEL, TARGET, 0, not imported), (*EXTERNAL, True)),
        ],
        width=2,
    )


def test_intersecting_priors_are_refused_and_one_datum_survives_once():
    """B7 row 3 (e). Fails if two retained priors whose pedigrees intersect are not
    refused (D008) on the survey, whichever rule retains them; or if, where the rule
    drops both summaries because their datum is imported, the installed beliefs are
    not the datum's own sufficient statistics (one unit of precision, its value).
    Where both are retained, they would install two units of precision for one datum.
    """
    for imported, rule in (
        (False, PedigreeRule.EXCLUDE_PRIOR),
        (False, PedigreeRule.EXCLUDE_ROWS),
        (True, PedigreeRule.EXCLUDE_ROWS),
    ):
        study = survey_3(two_summaries(imported), rule)
        codes = [f.code for f in preflight(study)]
        assert codes.count(FindingCode.D008) >= 1, (imported, rule)
        assert float(study.observer.init(study.caps).weight[TARGET]) == 2.0
    study = survey_3(two_summaries(True), PedigreeRule.EXCLUDE_PRIOR)
    assert FindingCode.D008 not in [f.code for f in preflight(study)]
    beliefs = imported_beliefs(study.observer, study.caps)
    for belief in beliefs:
        assert float(belief.weight[TARGET]) == 0.0
        assert int(belief.n[TARGET]) == 1 and float(belief.mean[TARGET]) == 1.0


# ---- (f): the window -----------------------------------------------------------------


def chosen_start(run, w):
    """The tick of the decision the first one waited for, and the wait it asked."""
    trace = index_world(run.result.trace, w)
    wake = np.asarray(trace.wake) & np.asarray(trace.handled)
    ticks = np.asarray(trace.tick)[wake]
    _, first = run.decisions[w][0]
    return int(ticks[1]), int(np.asarray(first.tick)[1])


@pytest.mark.slow  # the run store of every variant (K6)
def test_the_policy_starts_in_the_window_or_at_the_fallback(runs):
    """B7 row 3 (f). Fails if, with the prior retained for the policy, the policy does
    not start in the prior's window (counted from the row's release); or if, with the
    prior excluded for the policy, the policy's copy still carries the window or the
    policy does not take the fallback start. The chosen ticks are compared, not the
    posterior.
    """
    store, _ = runs
    assert not WINDOW[0] <= FALLBACK_WAIT <= WINDOW[1]
    for executor in EXECUTORS:
        retained, dropped = store["external", executor], store["public", executor]
        for w in range(N_WORLDS):
            window, released = prior_rows(carry_of(retained.result, w))[POL]
            assert window == WINDOW
            tick, asked = chosen_start(retained, w)
            assert tick == asked == released + WINDOW[0]
            assert released + WINDOW[0] <= tick <= released + WINDOW[1]
            carry = carry_of(dropped.result, w)
            assert prior_rows(carry)[POL][0] == NO_WINDOW
            assert float(carry.pbelief.weight[TARGET]) == 0.0
            assert chosen_start(dropped, w) == (FALLBACK_WAIT, FALLBACK_WAIT)


# ---- replay, executors and pre-flight ------------------------------------------------


@pytest.mark.slow  # the run store of every variant (K6)
def test_every_variant_passes_preflight_and_agrees_between_executors(runs):
    """B7 on both executors. Fails if pre-flight refuses a table variant of the
    survey, or the host and compiled executors disagree on any of them.
    """
    store, _ = runs
    for name in VARIANTS:
        assert preflight(study_of(name), executor="compiled") == [], name
        assert same_histories(
            store[name, "host"].result, store[name, "compiled"].result, N_WORLDS
        ), name


# ---- B6: the epoch relation ----------------------------------------------------------


@pytest.mark.slow  # one program per executor, shared by every shift and table (K6)
def test_the_epoch_relation_holds_with_windows_and_hidden_rows():
    """B6 on survey 3. Fails if a shift of the epoch origin, one of them larger than
    the oldest imported time, moves anything but the ticks on either executor, with a
    windowed prior, negative observation times and an observer-only row, or with the
    fallback start; for instance if the window or the fallback were read as absolute
    ticks.
    """
    shifts = (3, OLDEST - 1, 1000)
    assert max(shifts) > OLDEST
    hidden, public = study_of("hidden"), study_of("public")
    truths, roots = worlds(N_WORLDS, hidden.caps)
    for runner in (CompiledExecutor(), HostExecutor()):
        executor = type(runner).__name__
        cases = [(hidden, d) for d in shifts] + [(public, shifts[-1])]
        for study, delta in cases:
            result = battery.shift_epoch(
                study, truths, roots, N_STEPS, delta, executor=runner
            )
            assert result, f"{executor}, shift {delta}: {result.diagnostic}"
            # every imported row (both copies of each prior) is before the start
            assert result.detail["n_before_start"] == N_WORLDS * 7
            assert result.detail["n_views"] == N_WORLDS * (N_OBSERVATIONS + 2)


# ---- (g): controls (slow: each replaces the observer type) ---------------------------


class RuleOff(PriorObserver):
    """Control: every row and prior an audience receives keeps weight one."""

    def weights(self, caps, audience):
        t = self.table
        return (t.present & ((t.audience & audience) != 0)).astype(jnp.int64)


class IgnoresPrior(PriorObserver):
    """Control: an observer that installs no prior, whatever the table holds."""

    def prior_belief(self, caps, audience):
        flat = super().prior_belief(caps, audience)
        zero = jnp.zeros_like(flat.weight)
        return dataclasses.replace(flat, mean=zero, weight=zero)


def _control_run(observer_cls, name, truths, roots):
    table = survey_3_table(**VARIANTS[name])
    observer = observer_cls(table=table, prior_kind=WINDOW_PRIOR_INDEX)
    study = survey_3(table, observer=observer)
    return run_on("compiled", study, truths, roots, N_STEPS)


@pytest.mark.slow  # two observer types: two more compiled programs
def test_the_controls_fail_their_checks():
    """B7 row 3 (g). Fails if an observer with the rule off does not double count
    (it installs the prior that summarizes raw row 1 and folds the row: (b) must
    fail), or if an observer that ignores the installed prior passes (a).
    """
    truths, roots = worlds(N_WORLDS, study_of("hidden").caps)
    raw_only = run_on("compiled", study_of("raw_only"), truths, roots, N_STEPS)
    off = _control_run(RuleOff, "external", truths, roots)
    assert belief_failures(off, raw_only, "rule off") != []
    double = lambda c, a: expected_means(c, a, [PRIOR_3, PRIOR_4])  # noqa: E731
    assert mean_failures(off, double, "rule off") == []
    ignored = _control_run(IgnoresPrior, "external", truths, roots)
    flat = _control_run(IgnoresPrior, "none", truths, roots)
    assert prior_moves_beliefs(ignored, flat, "ignores prior") != []
