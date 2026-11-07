"""Table priors with refits: the audience-prior boundary, D009, D010, replay (B3).

The refit table imports two public raw rows and three priors. Subject 0 has an
observer-only prior (10, precision 2) and a policy-only prior (-4, precision 1): the
two audiences' masked rows and keys are equal there, but their priors differ. Subject
1 has one public prior (3, precision 1) whose pedigree names the raw row of subject 1,
which `EXCLUDE_ROWS` drops for both audiences, so the two audiences resolve alike.

The contract test needs no run. The run tests compile the refresh study on both
executors (medium tier).
"""

import dataclasses

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.core import (
    CompiledExecutor,
    HostExecutor,
    RunStatus,
    index_world,
    replay_run,
)
from spaceodyssey.core.conformance import (
    check_derived_summary,
    check_folded_once,
    check_overlap_declared,
)
from spaceodyssey.core.enums import Audience, RecordStatus
from spaceodyssey.core.preflight import preflight
from spaceodyssey.core.preflight.checks import PreflightContext
from spaceodyssey.core.refresh import (
    HostLoopRefit,
    MaskedBatchRefit,
    _gather,
)
from spaceodyssey.studies import toy
from spaceodyssey.studies.prior_observer import (
    AudiencePriorRefit,
    PriorBelief,
    PriorRefitObserver,
)
from tests.core.helpers import worlds
from tests.core.prior_helpers import IgnoresWeights, refit_study, refit_table
from tests.core.test_premission import _imported

OBS, POL = int(Audience.OBSERVER), int(Audience.POLICY)
MEAS = toy.RECORD_MEASUREMENT
PRIORS = {OBS: {0: (10.0, 2.0), 1: (3.0, 1.0)}, POL: {0: (-4.0, 1.0), 1: (3.0, 1.0)}}


def _request(study):
    """The refresh request of the carry after the imports, every entry dirty."""
    c = _imported(study)
    c = dataclasses.replace(c, refresh_dirty=jnp.ones_like(c.refresh_dirty))
    return _gather(c, jax.random.PRNGKey(0), study)


def _posterior(audience, subject, values):
    """The independent answer: the audience's prior plus each value at weight one."""
    mean, weight = PRIORS[audience][subject]
    return (mean * weight + sum(values)) / (weight + len(values))


# ---- the contract, written first --------------------------------------------------


class SharesAnyway(AudiencePriorRefit):
    """Control: honors the core's sharing whatever the two priors are."""

    def shared(self, study):
        return jnp.ones(study.caps.n_subjects, bool)


class StartsFromInit(AudiencePriorRefit):
    """Control: starts both audiences from `observer.init`."""

    def start(self, study, audience):
        return study.observer.init(study.caps)


class DropsPrior(AudiencePriorRefit):
    """Control: a refresh that starts from a flat belief and drops the prior."""

    def start(self, study, audience):
        caps = study.caps
        flat = study.observer.prior_belief(caps, audience)
        return PriorBelief(
            mean=jnp.zeros_like(flat.mean),
            n=flat.n,
            aux=flat.aux,
            weight=jnp.zeros_like(flat.weight),
        )


class IgnoresWeightsRefit(IgnoresWeights, PriorRefitObserver):
    """Control: a refit-capable observer that folds and installs every row."""


def _contract_failures(boundary, study, req):
    """What the contract finds wrong with `boundary`'s answer to `req`."""
    batch = jax.tree_util.tree_map(lambda x: x[None], req)
    out = jax.tree_util.tree_map(
        lambda x: np.asarray(x[0]), boundary.refit_batch(study, batch)
    )
    mean, n, weight = out.slices
    failures = []
    for audience, a in ((OBS, 0), (POL, 1)):
        want = _posterior(audience, 0, [1.0])
        if (
            abs(float(mean[a, 0]) - want) > 1e-12
            or float(weight[a, 0]) != PRIORS[audience][0][1]
        ):
            failures.append(f"audience {audience}: {float(mean[a, 0])} != {want}")
        if int(n[a, 0]) != 1:
            failures.append(f"audience {audience}: {int(n[a, 0])} rows")
    if not bool(out.computed[1, 0]):
        failures.append("the policy entry of subject 0 copied the observer's")
    return failures, out


@pytest.mark.medium  # five boundary programs and the eager import, about 8 s cold
def test_the_audience_prior_boundary_meets_its_contract():
    """B3 contract. With identical masked rows and keys for both audiences of
    subject 0 (the core shares the entry) but different installed priors, fails if
    either audience's answer is not its own prior plus the accepted row, or if the
    policy entry reuses the observer's. Subject 1, whose audiences resolve alike, may
    share. Controls: a boundary that honors the core's sharing, one that starts from
    `observer.init`, and the observer's own masked kernel all fail.
    """
    study = refit_study()
    req = _request(study)
    rows = jax.tree_util.tree_map(np.asarray, req.rows)
    assert np.array_equal(np.asarray(req.keys)[0, 0], np.asarray(req.keys)[1, 0])
    assert np.array_equal(rows.payload[0, 0], rows.payload[1, 0])
    assert int(req.source[1, 0]) == 0  # the core shares subject 0 across audiences
    failures, out = _contract_failures(AudiencePriorRefit(), study, req)
    assert failures == []
    assert not bool(out.computed[1, 1])  # equal priors: the sharing is honored
    np.testing.assert_array_equal(out.slices[0][1, 1], out.slices[0][0, 1])
    assert float(out.slices[0][0, 1]) == 3.0  # the raw row of subject 1 is dropped
    for control in (SharesAnyway(), StartsFromInit(), MaskedBatchRefit()):
        failures, _ = _contract_failures(control, study, req)
        assert any("audience 2" in f for f in failures), type(control).__name__


# ---- D009 and D010 ----------------------------------------------------------------


def _codes(study, boundary):
    found = preflight(study, context=PreflightContext(boundary=boundary))
    return [(str(f.code), str(f.severity)) for f in found]


def test_d009_refuses_different_priors_on_the_observer_init_boundary():
    """Fails if the boundary that starts both refits from `observer.init` is accepted
    while the audiences resolve to different priors, if it is refused when they
    resolve alike, or if the audience-prior boundary is refused.
    """
    differ = refit_study()
    assert _codes(differ, MaskedBatchRefit()) == [("D009", "error")]
    assert _codes(differ, AudiencePriorRefit()) == []
    alike = refit_study(refit_table(policy_value=10.0))
    alike = dataclasses.replace(
        alike,
        observer=dataclasses.replace(
            alike.observer,
            table=_equal_priors(alike.observer.table),
        ),
    )
    alike = dataclasses.replace(
        alike,
        process=dataclasses.replace(alike.process, table=alike.observer.table),
    )
    assert _codes(alike, MaskedBatchRefit()) == []


def _equal_priors(t):
    """The refit table with both subject-0 priors public and equal (one is padded)."""
    return dataclasses.replace(
        t,
        audience=t.audience.at[2].set(3),
        source=dataclasses.replace(
            t.source, namespace=t.source.namespace.at[3].set(-1)
        ),
    )


def _ofti():
    photomancy = pytest.importorskip("spaceodyssey.adapters.photomancy")
    return photomancy.OftiRefit(stellar_mass_kg=2e30, dist_pc=10.0)


@pytest.mark.parametrize(
    "boundary",
    [
        _ofti,
        lambda: HostLoopRefit(toy.toy_refit_fit),
        lambda: MaskedBatchRefit(toy.toy_mean_kernel),
    ],
    ids=["ofti", "host_fit", "flat_kernel"],
)
def test_d010_refuses_a_table_prior_with_a_boundary_that_ignores_it(boundary):
    """Fails if a table prior with OFTI, or with a boundary whose fit or kernel
    never reads the installed prior (the stand-in for the updater boundary), is
    accepted; or if the same boundary is refused once every prior is dropped.
    """
    study = refit_study()
    assert _codes(study, boundary()) == [("D010", "error")]
    t = study.observer.table
    gone = dataclasses.replace(
        t,
        source=dataclasses.replace(
            t.source, namespace=t.source.namespace.at[2:].set(-1)
        ),
    )
    bare = dataclasses.replace(
        study,
        observer=dataclasses.replace(study.observer, table=gone),
        process=dataclasses.replace(study.process, table=gone),
    )
    assert _codes(bare, boundary()) == []


def test_without_a_boundary_d009_and_d010_are_reported_unverified():
    """Fails if a refit study with a table prior and no boundary in the context is
    reported clean instead of unverified, or the warning is an error.
    """
    found = preflight(refit_study())
    assert [(str(f.code), str(f.severity)) for f in found] == [("D010", "warning")]
    assert "unverified" in found[0].message


# ---- runs: the prior across a refresh, replay ---------------------------------------


def _run(executor, boundary_cls=AudiencePriorRefit, n_steps=120):
    """Two worlds of the refit study on `executor` ("host" or "compiled")."""
    study = refit_study()
    truths, roots = worlds(2, study.caps)
    if executor == "host":
        run = HostExecutor().run_worlds(
            study, truths, roots, n_steps, refit=boundary_cls()
        )
    else:
        run = CompiledExecutor().run(
            study, truths, roots, n_steps, refit=boundary_cls(), chunk=7
        )
    return study, run, truths, roots


def _replay_failures(study, run, roots):
    """Replay every world from `init` and `init_summary` with the boundary."""
    failures = []
    for w in range(2):
        carry = index_world(run.carry, w)
        out = replay_run(
            study,
            roots[w],
            carry,
            index_world(run.trace, w),
            refit=AudiencePriorRefit(),
        )
        if not (out.checkpoints >= 3 and out.products > 0):
            failures.append(f"world {w}: {out.checkpoints} checkpoints")
        if not np.array_equal(
            np.asarray(out.belief.mean), np.asarray(carry.belief.mean)
        ):
            failures.append(f"world {w}: replayed belief differs")
        if not np.array_equal(
            np.asarray(out.pbelief.weight), np.asarray(carry.pbelief.weight)
        ):
            failures.append(f"world {w}: replayed policy prior differs")
    return failures


def _expected_beliefs(carry):
    """Each audience's subject 0 and 1 means from the run's log, independently."""
    log = jax.tree_util.tree_map(np.asarray, carry.log)
    out = {}
    for audience, tick in ((OBS, log.accepted_tick), (POL, log.released_tick)):
        for s in (0, 1):
            keep = (
                log.valid
                & (tick >= 0)
                & (log.kind == MEAS)
                & (log.status == RecordStatus.COMPLETED)
                & (log.subject == s)
            )
            # the imported raw row of subject 1 is dropped by its prior
            keep &= ~((log.channel == 60) & (log.acquisition_index == 1))
            out[audience, s] = _posterior(audience, s, list(log.payload[keep, 0]))
    return out


def _belief_failures(result):
    failures = []
    for w in range(2):
        carry = index_world(result.carry, w)
        want = _expected_beliefs(carry)
        for (audience, s), value in want.items():
            belief = carry.belief if audience == OBS else carry.pbelief
            if abs(float(belief.mean[s]) - value) > 1e-12:
                failures.append(f"world {w} audience {audience} subject {s}")
    return failures


@pytest.mark.medium  # the refresh study on the host executor, and two controls
def test_a_refit_observer_keeps_its_prior_across_a_refresh():
    """B3. Fails if, after several checkpoints, either audience's belief is not its
    own prior plus the rows it received (computed from the log), if the run does not
    replay, if a prior row is counted as a datum, or if the kit's overlap and
    folded-once checks fail on the run. Controls: a refresh that starts from a flat
    belief drops the prior and fails; an observer that folds rows of weight zero fails
    folded-once.
    """
    study, host, truths, roots = _run("host")
    assert all(int(s) == RunStatus.COMPLETE for s in np.asarray(host.status))
    assert (np.asarray(host.carry.n_dec) > 0).all()
    assert _belief_failures(host) == []
    assert _replay_failures(study, host, roots) == []
    assert check_overlap_declared(study)
    assert check_folded_once(study, host, truths, roots)
    # the prior rows in the log are derived: a refit counts each datum once
    summary = check_derived_summary(study, host, roots, summary_kind="prior")
    assert summary, summary.diagnostic
    t = study.observer.table
    ignores = dataclasses.replace(study, observer=IgnoresWeightsRefit(table=t, rule=1))
    late = check_folded_once(ignores, host, truths, roots)
    assert not late and "weight zero" in late.diagnostic
    _, dropped, _, _ = _run("host", DropsPrior)
    assert _belief_failures(dropped)


@pytest.mark.medium  # the refresh study on the compiled executor, replayed
def test_replay_reproduces_prior_run():
    """Named interface check (T4c). Fails if the compiled executor's run of the prior
    study does not hold each audience's prior plus its rows, or if keyed replay from
    `init` and `init_summary` with the audience-prior boundary misses a checkpoint or
    differs from the run (replay raises on any other mismatch). The host run is
    replayed in the test above.
    """
    study, compiled, _, roots = _run("compiled")
    assert np.asarray(compiled.complete).all()
    assert _belief_failures(compiled) == []
    assert _replay_failures(study, compiled, roots) == []
