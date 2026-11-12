"""Survey 18, exozodi-first: the precursor changes the plan only through releases.

The merge-time guard (medium) is (a), (c) and (e) on both executors: every variant is
one table of one shape with array-valued payload, audience, delay and coupling, so all
of them share one compiled program per executor. The other tests are slow (the Slice
2b quick-tier budget, K6): (b), (d), the epoch relation (it wraps the process) and the
truth-reading control (another policy type, one world per run).

A plan is what the policy did: each decision's tick, effect kinds and admission codes,
and each measurement's subject, channel, acquisition index and start tick. Payloads
are left out, since they carry the world. The worlds come in pairs whose target star
levels lie on opposite sides of the policy's threshold, and plans are compared across
every pair of worlds, so a plan that read the truth would differ.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
import pytest

from spaceodyssey.core import CompiledExecutor, HostExecutor, battery
from spaceodyssey.core.battery.result import worlds_of
from spaceodyssey.core.battery.truth_swap import first_truth_dependent_release
from spaceodyssey.core.enums import F64, PUBLIC, Audience, Purpose
from spaceodyssey.core.preflight import preflight
from spaceodyssey.studies import toy
from tests.studies.scenarios.survey_18 import (
    BRIGHT,
    FAINT,
    N_OBSERVATIONS,
    PRIOR_MEAN,
    PRIOR_WEIGHT,
    STAR_LEVEL_INDEX,
    STAR_OF,
    TARGET_STAR,
    WITHHELD,
    Survey18Policy,
    draw_survey_18_truth,
    survey_18,
    survey_18_table,
)
from tests.studies.survey_helpers import run_on, same_histories

OBS = int(Audience.OBSERVER)
MEAS = toy.RECORD_MEASUREMENT
N_STEPS = 40
N_WORLDS = 2
EXECUTORS = ("compiled", "host")
DELAY = 60  # (b): the precursor is released during the first observation
# the levels of the two stars per world: the target star is faint in one world of
# each pair and bright in the other (the threshold is 0.8)
LEVELS = ((0.1, 1.9), (1.9, 0.1))
N_DECISIONS = N_OBSERVATIONS + 1  # the observations and the drain


def posterior(value):
    """The level a summary holds after one precursor row of `value`."""
    return (PRIOR_MEAN * PRIOR_WEIGHT + value) / (PRIOR_WEIGHT + 1.0)


def survey_worlds(seed, levels):
    """`N_WORLDS` worlds drawn from their world keys, with the star levels set."""
    caps = survey_18().caps
    roots = jax.vmap(lambda i: jr.fold_in(jr.PRNGKey(seed), i))(jnp.arange(N_WORLDS))
    truths = jax.vmap(
        lambda r: draw_survey_18_truth(jr.fold_in(r, Purpose.WORLD), caps)
    )(roots)
    return eqx.tree_at(lambda t: t.level, truths, jnp.asarray(levels, F64)), roots


WORLDS = {
    "A": lambda: survey_worlds(0, LEVELS),
    # unrelated worlds: other keys, the levels of each world reversed
    "B": lambda: survey_worlds(1, LEVELS[::-1]),
    # (d): the worlds of A with the group latent swapped between them
    "swapped": lambda: (
        lambda t, r: (eqx.tree_at(lambda x: x.level, t, t.level[::-1]), r)
    )(*survey_worlds(0, LEVELS)),
}


def study_of(value=FAINT, audience=PUBLIC, release_delay=0, coupling=1.0, policy=None):
    return survey_18(
        survey_18_table(value, audience),
        release_delay=release_delay,
        coupling=coupling,
        policy=policy,
    )


def plan_of(world, before=None):
    """The plan of one world `(carry, trace)`, up to tick `before` when given."""
    carry, trace = world
    tick = np.asarray(trace.tick)
    keep = np.asarray(trace.handled) & np.asarray(trace.wake)
    log = carry.log
    rows = np.asarray(log.valid) & (np.asarray(log.kind) == MEAS)
    if before is not None:
        keep &= tick < before
        rows &= np.asarray(log.start_tick) < before
    decisions = tuple(
        (int(t), tuple(map(int, k)), tuple(map(int, c)))
        for t, k, c in zip(
            tick[keep],
            np.asarray(trace.eff_kind)[keep],
            np.asarray(trace.admit_code)[keep],
            strict=True,
        )
    )
    columns = (log.subject, log.channel, log.acquisition_index, log.start_tick)
    acquisitions = tuple(
        tuple(int(x) for x in row)
        for row in zip(*(np.asarray(c)[rows] for c in columns), strict=True)
    )
    return decisions, acquisitions


class Runner:
    """Runs of the survey on one executor, cached by world and variant.

    Returns the `(carry, trace)` of every world of the batch.
    """

    def __init__(self, executor):
        self.executor = executor
        self.runs = {}
        self.worlds = {}

    def world(self, name):
        if name not in self.worlds:
            self.worlds[name] = WORLDS[name]()
        return self.worlds[name]

    def __call__(self, name, **variant):
        key = (name, tuple(sorted(variant.items())))
        if key not in self.runs:
            truths, roots = self.world(name)
            run = run_on(self.executor, study_of(**variant), truths, roots, N_STEPS)
            assert np.asarray(run.complete).all(), key
            self.runs[key] = run
        return worlds_of(self.runs[key])


# ---- the checks of rows (a) to (e); each returns the failures it found ----------


def same_everywhere(plans_a, plans_b, what):
    """Failures where a plan of `plans_a` differs from one of `plans_b`."""
    bad = [
        f"{what}: world {i} differs from world {j}"
        for i, a in enumerate(plans_a)
        for j, b in enumerate(plans_b)
        if a != b
    ]
    return bad


def withheld_check(runner):
    """(a) A withheld precursor: its payload never moves the plan."""
    faint, bright = (
        [plan_of(w) for w in runner("A", value=v, release_delay=WITHHELD)]
        for v in (FAINT, BRIGHT)
    )
    return same_everywhere(faint + bright, faint + bright, "withheld")


def delayed_check(runner):
    """(b) Released at `DELAY`: decisions before it are the withheld run's."""
    withheld = runner("A", release_delay=WITHHELD)
    delayed = runner("A", release_delay=DELAY)
    out = same_everywhere(
        [plan_of(w, DELAY) for w in delayed],
        [plan_of(w, DELAY) for w in withheld],
        "before the delayed release",
    )
    # evidence: the release does move the plan after it
    out += [
        f"world {i}: the delayed release changed no decision"
        for i, (d, h) in enumerate(zip(delayed, withheld, strict=True))
        if plan_of(d) == plan_of(h)
    ]
    return out


def observer_only_check(runner):
    """(c) An observer-only precursor: the plan is the withheld run's."""
    return same_everywhere(
        [plan_of(w) for w in runner("A", audience=OBS)],
        [plan_of(w) for w in runner("A", release_delay=WITHHELD)],
        "observer only",
    )


def swap_check(runner, caps, coupling):
    """(d) The group latent swapped: the plan holds until a released row differs.

    With `coupling` 0 the measurements do not carry the level, so no released row
    differs and the whole plan must hold; otherwise a row must differ (or the swap
    proves nothing) and the plan must hold before the tick it is released at.
    """
    out = []
    a = runner("A", coupling=coupling)
    b = runner("swapped", coupling=coupling)
    for w, (wa, wb) in enumerate(zip(a, b, strict=True)):
        out += [
            f"world {w}: {bad}" for bad in _missing_evidence(wa) + _missing_evidence(wb)
        ]
        step, _ = first_truth_dependent_release(wa, wb, caps)
        if (step is None) != (coupling == 0):
            out.append(f"world {w}: a released row differs at step {step}")
            continue
        until = None if step is None else int(np.asarray(wa[1].tick)[step])
        decisions, _ = plan_of(wa, until)
        if not decisions:
            out.append(f"world {w}: no decision before tick {until}")
        if plan_of(wa, until) != plan_of(wb, until):
            out.append(f"world {w}: the plan changed before tick {until}")
    return out


def _missing_evidence(world):
    """What a full run of the survey lacks: its decisions, acquisitions and folds."""
    decisions, acquisitions = plan_of(world)
    out = []
    if len(decisions) < N_DECISIONS:
        out.append(f"{len(decisions)} decisions, not {N_DECISIONS}")
    if len(acquisitions) != N_OBSERVATIONS:
        out.append(f"{len(acquisitions)} acquisitions, not {N_OBSERVATIONS}")
    folded = int(np.sum(np.asarray(world[0].belief.n)))
    if folded != N_OBSERVATIONS:
        out.append(f"{folded} measurements folded, not {N_OBSERVATIONS}")
    return out


def released_check(runner):
    """(e) Released with two payloads: the plans differ, each the same on worlds B."""
    out = []
    plans = {}
    for v in (FAINT, BRIGHT):
        plans[v] = [plan_of(w) for w in runner("A", value=v)]
        other = [plan_of(w) for w in runner("B", value=v)]
        out += same_everywhere(plans[v], other, f"payload {v} on unrelated worlds")
    out += [
        f"world {i}: both payloads gave one plan"
        for i, (f, b) in enumerate(zip(plans[FAINT], plans[BRIGHT], strict=True))
        if f == b
    ]
    return out


# ---- tests ------------------------------------------------------------------------


def _observed_stars(world):
    _, acquisitions = plan_of(world)
    return {STAR_OF[subject] for subject, *_ in acquisitions}


def _precursor(world):
    """`(released_tick, accepted_tick)` of the one imported precursor row."""
    log = world[0].log
    rows = np.flatnonzero(
        np.asarray(log.valid) & (np.asarray(log.kind) == STAR_LEVEL_INDEX)
    )
    assert rows.size == 1, rows
    return int(log.released_tick[rows[0]]), int(log.accepted_tick[rows[0]])


@pytest.mark.slow(reason="one cold program per executor")
def test_the_precursor_changes_the_plan_only_when_released_to_the_policy():
    """B7 row 18 (a), (c) and (e) on both executors. Fails if a withheld precursor's
    payload, or an observer-only precursor, moves any decision or acquisition in any
    world; if a released precursor does not choose the plan its payload calls for
    (the faint payload the target star's planets, the bright one the other star's);
    if a plan depends on the world it ran in (the worlds of a pair straddle the
    threshold; unrelated worlds give the same plan); or if the executors disagree.
    """
    assert not preflight(study_of(), executor="compiled")
    plans = {}
    for executor in EXECUTORS:
        runner = Runner(executor)
        failures = withheld_check(runner)
        failures += observer_only_check(runner)
        failures += released_check(runner)
        assert not failures, f"{executor}: {failures}"
        # evidence, (a): every decision was made, the precursor came after the last
        for world in runner("A", release_delay=WITHHELD):
            decisions, acquisitions = plan_of(world)
            assert len(decisions) == N_DECISIONS
            assert len(acquisitions) == N_OBSERVATIONS
            assert _precursor(world)[0] > decisions[-1][0]
        # evidence, (c): the observer folded the precursor and the summary did not
        for carry, _ in runner("A", audience=OBS):
            assert int(carry.belief.n_level[TARGET_STAR]) == 1
            assert int(carry.pbelief.n_level[TARGET_STAR]) == 0
            assert float(carry.belief.aux[TARGET_STAR]) == posterior(FAINT)
            assert float(carry.pbelief.aux[TARGET_STAR]) == PRIOR_MEAN
        # evidence, (e): released at tick 0 to the first decision, in the summary
        for value, star in ((FAINT, TARGET_STAR), (BRIGHT, 1 - TARGET_STAR)):
            for world in runner("A", value=value):
                assert _precursor(world) == (0, 0)
                assert float(world[0].pbelief.aux[TARGET_STAR]) == posterior(value)
                assert _observed_stars(world) == {star}
        plans[executor] = {
            key: [plan_of(w) for w in worlds_of(run)]
            for key, run in runner.runs.items()
        }
    assert plans["host"] == plans["compiled"]


@pytest.mark.slow(reason="shares the guard's program; the guard covers the claim")
def test_decisions_before_a_delayed_release_are_the_withheld_runs():
    """B7 row 18 (b), both executors. Fails if a precursor released at tick `DELAY`
    moves a decision or acquisition before that tick in any world, or moves nothing
    after it (the check would then prove nothing).
    """
    for executor in EXECUTORS:
        runner = Runner(executor)
        assert not delayed_check(runner), executor
        for world in runner("A", release_delay=DELAY):
            assert _precursor(world) == (DELAY, DELAY)
            decisions, _ = plan_of(world, DELAY)
            assert decisions, "no decision before the delayed release"


@pytest.mark.slow(reason="shares the guard's program")
def test_a_swapped_group_latent_moves_the_plan_only_after_a_differing_release():
    """B7 row 18 (d), both executors. Fails if swapping the stars' levels between the
    worlds moves the plan before the first released row whose payload differs, or at
    all when the measurements do not carry the level (coupling 0, every released row
    identical); or if the swap changes no released row with the level coupled. The
    platform's truth swap also holds on the compiled executor.
    """
    caps = study_of().caps
    for executor in EXECUTORS:
        runner = Runner(executor)
        for coupling in (1.0, 0.0):
            assert not swap_check(runner, caps, coupling), (executor, coupling)
    truths, roots = WORLDS["A"]()
    swapped, _ = WORLDS["swapped"]()
    result = battery.truth_swap(study_of(), truths, swapped, roots, N_STEPS)
    assert result, result.diagnostic


@pytest.mark.slow(reason="the epoch wrapper compiles its own program per executor")
def test_the_epoch_relation_holds_with_released_and_delayed_precursors():
    """B6 on survey 18. Fails if shifting the epoch origin moves any decision, record
    or belief other than by the shift, for a precursor released at tick 0 and one
    released later; for instance if the delay were counted from tick 0.
    """
    truths, roots = WORLDS["A"]()
    for name, executor in (("compiled", CompiledExecutor()), ("host", HostExecutor())):
        runner = Runner(name)
        for delay in (0, DELAY):
            study = study_of(release_delay=delay)
            result = battery.shift_epoch(
                study, truths, roots, N_STEPS, 1000, executor=executor
            )
            assert result, f"{name}, delay {delay}: {result.diagnostic}"
            # evidence: the unshifted run decided, observed and folded; the relation
            # compared every decision view and record of it
            worlds = runner("A", release_delay=delay)
            assert not [bad for w in worlds for bad in _missing_evidence(w)], delay
            n_decisions = sum(len(plan_of(w)[0]) for w in worlds)
            assert result.detail["n_views"] == n_decisions, (name, delay)
            assert result.detail["n_records"] == N_WORLDS * (1 + N_OBSERVATIONS)
            assert result.detail["n_before_start"] == N_WORLDS


class TruthReadingPolicy(Survey18Policy):
    """Control: the survey policy choosing its plan by the world's true level."""

    truth_level: jax.Array = eqx.field(kw_only=True)

    def level(self, view):
        """The target star's true level, not the summary's."""
        return self.truth_level[self.target]


class TruthReadingRunner(Runner):
    """Runs of the control: one world per run, the policy given that world's level."""

    def __call__(self, name, **variant):
        key = (name, tuple(sorted(variant.items())))
        if key not in self.runs:
            truths, roots = self.world(name)
            worlds = []
            for w in range(N_WORLDS):
                one = jax.tree_util.tree_map(lambda x, w=w: x[w : w + 1], truths)
                policy = TruthReadingPolicy(truth_level=one.level[0])
                study = study_of(**variant, policy=policy)
                run = run_on(self.executor, study, one, roots[w : w + 1], N_STEPS)
                worlds += worlds_of(run)
            self.runs[key] = worlds
        return self.runs[key]


@pytest.mark.slow(reason="another policy type and one world per run")
def test_a_policy_reading_the_truth_latent_fails_rows_a_to_e():
    """B7 row 18 (f), control, both executors. Fails if a policy that chooses its
    plan by the target star's true level passes any of (a) to (e) on either executor:
    each check compares plans across worlds whose levels straddle the threshold, so
    such a policy must differ there.
    """
    caps = study_of().caps
    for executor in EXECUTORS:
        runner = TruthReadingRunner(executor)
        assert withheld_check(runner), executor
        assert delayed_check(runner), executor
        assert observer_only_check(runner), executor
        assert swap_check(runner, caps, 1.0), executor
        assert swap_check(runner, caps, 0.0), executor
        assert released_check(runner), executor


@pytest.mark.slow(reason="every variant on both executors")
def test_every_variant_passes_preflight_and_agrees_between_executors():
    """B7 executors. Fails if a variant of the survey (withheld, delayed, observer
    only, released with either payload, uncoupled) fails pre-flight, or if the two
    executors give different histories for it.
    """
    truths, roots = WORLDS["A"]()
    variants = (
        dict(release_delay=WITHHELD),
        dict(release_delay=DELAY),
        dict(audience=OBS),
        dict(value=FAINT),
        dict(value=BRIGHT),
        dict(coupling=0.0),
    )
    for variant in variants:
        study = study_of(**variant)
        assert not preflight(study, executor="compiled"), variant
        host = run_on("host", study, truths, roots, N_STEPS)
        compiled = run_on("compiled", study, truths, roots, N_STEPS)
        assert same_histories(host, compiled, N_WORLDS), variant
