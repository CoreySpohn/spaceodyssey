"""The survey loop and its score: `step`, `execute`, the policies and `score`.

Milestone 1 is E and L as a scored log: a `TablePolicy` replays the stages of the E
track as rows through `execute`, and `score` reads the information table and the
certificates off the log against the unsealed truth.
"""

import functools
import zlib
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from numpy.testing import assert_allclose

from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.design import Design, Keepout, Mode, Resources, no_keepout
from spaceodyssey.information import POSITIVE, UNRESOLVED
from spaceodyssey.log import execute, grant_rows, start_log, step
from spaceodyssey.policy import AbstractPolicy, TablePolicy, TableState, public_view
from spaceodyssey.score import associate_planets, format_report, score
from spaceodyssey.study import Arm, Study
from spaceodyssey.toy import GaussianGasModel, StepReportModel
from spaceodyssey.types import (
    ACCEPTED,
    OVER_BUDGET,
    PAST_END,
    WAITED,
    AbstractObservationModel,
    Action,
    Grant,
    Refit,
    clock_d,
    spent_d,
)
from spaceodyssey.world import SealedWorld, draw_worlds, row_key, run_keys

KEY = jax.random.key(0)
CAPACITY = 8  # W
END_D = 365.0  # the calendar's end, days
COPIES = 250  # copies of each draw type: 16 x 250 = 4,000 draws for the step rule


class OrbitReport(AbstractObservationModel):
    """A resolved orbit: it reports each planet's `hz` bit exactly and weighs an Earth
    size 7 to 3 against a look-alike, whatever the truth's size (the reading of the
    size sits where its likelihood ratio is 7:3).
    """

    searches: ClassVar[bool] = False
    stage: ClassVar[int] = 0

    def sample(self, key, atoms, action):
        return {"hz": atoms.cloud.samples["hz"]}

    def stats(self, payload, atoms, star):
        hz, size = atoms.cloud.samples["hz"], atoms.cloud.samples["size"]
        like = jnp.where(hz == payload["hz"], jnp.where(size > 0.5, 0.7, 0.3), 0.0)
        return jnp.sum(jnp.log(like), axis=-1)

    def finalize(self, stats, atoms, star):
        return stats, atoms

    def cell(self, payload):
        return (payload["hz"][0] > 0.5).astype(int)

    def cells(self, atoms, action):
        hz = atoms.cloud.samples["hz"][:, 0]
        return jnp.stack([1.0 - hz, hz], axis=-1)


class ExpectedReading(AbstractObservationModel):
    """The Gaussian gas model with every reading at its expected value: the true
    abundance, without noise.
    """

    searches: ClassVar[bool] = False
    stage: ClassVar[int] = 0
    gas: GaussianGasModel

    def sample(self, key, atoms, action):
        snr = self.gas.sample(key, atoms, action)["snr"]
        return {"snr": snr, "reading": jnp.where(snr > 0, atoms.x_mean, 0.0)}

    def stats(self, payload, atoms, star):
        return self.gas.stats(payload, atoms, star)

    def finalize(self, stats, atoms, star):
        return self.gas.finalize(stats, atoms, star)

    def cell(self, payload):
        return self.gas.cell(payload)

    def cells(self, atoms, action):
        return self.gas.cells(atoms, action)


def _no_grant():
    no_int = jnp.zeros(0, int)
    return Grant(Action(no_int, no_int, jnp.zeros(0)))


def _design(models, budget_d=10.0, latency_d=0.0):
    """One mode per model, in model order; no keepout at two stars, a calendar of
    `END_D` days."""
    modes = tuple(Mode(slot, latency_d) for slot in range(len(models)))
    resources = Resources(budget_d, END_D)
    return Design(models, modes, resources, no_keepout(2), CAPACITY)


def _study(toy, cloud, design, n_star, claim):
    """A study whose every star starts from `cloud`, under flat roles on the toy,
    with the design's keepout table resized to its stars."""
    design = eqx.tree_at(lambda d: d.keepout, design, no_keepout(n_star))
    refits = toy.stack(
        *[Refit(cloud, jnp.zeros(CAPACITY, bool), s) for s in range(n_star)]
    )
    pool = toy.stack(*[toy.empty_pool(2, 2)] * n_star)
    prior = toy.GasPrior("toy", n_gas=2)  # flat on point gases at +-1 and on N(0, 1)
    member = toy.GasPrior("toy", mean=[0.0], sd=[1.0], n_gas=2)
    return Study(
        claim, prior, prior, member, refits, pool, (), design,
        jnp.zeros((n_star, 2)), alpha=0.05,
    )  # fmt: skip


def _e_track_claim(toy):
    """Candidate (hz, size), then water and oxygen, conditioned on the candidacy."""
    gases = (GasAbove(0, 0.0), GasAbove(1, 0.0))
    return Claim((toy.InHz(), toy.InSize(), *gases), given=2)


@functools.cache  # built once per session; arrays are immutable
def _el_world(toy):
    """E (star 0): an Earth-sized HZ planet with water and oxygen 1 dex above their
    floors. L (star 1): its look-alike, the same but not Earth-sized.
    """
    e = toy.atoms([[1.0]], [[1.0]], [1.0], [[1.0, 1.0]], 0.0)
    look_alike = toy.atoms([[1.0]], [[0.0]], [1.0], [[1.0, 1.0]], 0.0)
    return SealedWorld(toy.stack(e, look_alike))


def _empty_log(world, design):
    """A run's log before its first action, with no grant."""
    grant = _no_grant()
    _, noise, _ = run_keys(KEY)
    payload = world.reveal(noise, grant, design.models, design.modes)
    return start_log(grant_rows(grant, payload, design.modes), CAPACITY)


def _stage_table():
    """The E track as rows at both stars: the orbit, then water, then oxygen."""
    return TablePolicy(Action([0, 1, 0, 1, 0, 1], [0, 0, 1, 1, 2, 2], [1.0] * 6))


@functools.cache  # built once per session; arrays are immutable
def _step_rule_cloud(toy):
    """The 16 point draws over (hz, size, water, oxygen), each gas 1 dex above or
    below its floor, `COPIES` times each, equal weights.
    """
    bits = np.array(np.meshgrid(*[[0.0, 1.0]] * 4, indexing="ij")).reshape(4, -1).T
    bits = np.repeat(bits, COPIES, axis=0)
    gases = 2.0 * bits[:, 2:] - 1.0
    return toy.atoms(bits[:, :1], bits[:, 1:2], np.ones(len(bits)), gases, 0.0)


@functools.cache  # built once per session; arrays are immutable
def _gaussian_cloud(toy):
    """The 4 draws over (hz, size), 1,000 times each, both gases N(0, 1) dex."""
    bits = np.array(np.meshgrid([0.0, 1.0], [0.0, 1.0], indexing="ij")).reshape(2, -1)
    bits = np.repeat(bits.T, 1000, axis=0)
    return toy.atoms(bits[:, :1], bits[:, 1:], np.ones(len(bits)), 0.0, 1.0)


def _run(toy, cloud, models):
    """E and L through `execute` and `score` with the stage table."""
    design = _design(models)
    claim = _e_track_claim(toy)
    study = _study(toy, cloud, design, 2, claim)
    world = _el_world(toy)
    log = execute(KEY, _no_grant(), world, design, _stage_table())
    return log, score(log, world, study)


def _by_star(report, star):
    info = report.information
    pick = np.asarray(info["star"]) == star
    return {name: np.asarray(column)[pick] for name, column in info.items()}


def test_milestone_1_e_and_l_as_a_scored_log(toy):
    """Milestone 1: E and L as a scored log from a real `execute` and `score`.

    Step rule (S/N 6 reports), a cloud of 4,000 point draws: E's q is 0.0625, 0.175,
    0.35, 0.70 over the stages (then a marked end stage at day 6, after its last
    release at day 5, with no new rows), its information +2.42 in total, split
    0.69 + 0.34 + 0.69 + 0.69 over hz, size, water and oxygen; L, the look-alike
    with the same rows, loses 1.14. E's conditional q (both gases given the
    candidacy) is 1, so E is issued, its lower bound the Wilson bound n / (n + 4) at
    n = 250, the Kish size of the condition-weighted draws: 250 / 254 = 0.9843. L
    fails the condition and is void. Gaussian gas model at S/N 5,
    readings at their expected value, 4,000 draws: q is 0.0625, 0.175, 0.337, 0.649,
    E gains +1.03, +0.66, +0.66 (2.34) and L loses 0.13, 0.22, 0.64 (0.98); E's
    conditional q is 0.649 / 0.70 = 0.927 < 0.95, so E is not issued. Both runs print
    their tables.

    Fails if `execute` drops, reorders or misroutes a row, `score` reads the wrong
    stage, role or sign, the chain split does not sum to the information, the
    certificate reads the joint where the condition belongs or counts more
    conditioning draws than the condition-weighted Kish size, or a void planet is
    certified.
    """
    step_models = (
        OrbitReport(),
        StepReportModel(snr_one_day=[6.0, 0.0]),
        StepReportModel(snr_one_day=[0.0, 6.0]),
    )
    log, report = _run(toy, _step_rule_cloud(toy), step_models)
    assert int(log.count) == 6
    e, look_alike = _by_star(report, 0), _by_star(report, 1)
    # E's last row is released at day 5, the run ends at day 6: a marked end stage.
    assert list(e["end"]) == [False, False, False, True]
    assert list(look_alike["end"]) == [False, False, True]
    assert_allclose(e["information"][3], 0.0, atol=1e-12)
    assert_allclose(e["q_after"], [0.175, 0.35, 0.70, 0.70], rtol=1e-10)
    assert_allclose(e["q_before"][0], 0.0625, rtol=1e-10)
    assert_allclose(look_alike["q_after"], [0.175, 0.35, 0.70], rtol=1e-10)
    assert round(float(e["information"].sum()), 2) == 2.42
    assert round(float(look_alike["information"].sum()), 2) == -1.14
    ln2, ln14 = np.log(2.0), np.log(1.4)
    assert_allclose(e["chain_gain"].sum(axis=0), [ln2, ln14, ln2, ln2], atol=1e-10)
    assert_allclose(e["chain_gain"].sum(), np.log(0.70 / 0.0625), rtol=1e-10)
    assert np.isnan(look_alike["chain_gain"]).all()

    cert = {name: np.asarray(column) for name, column in report.certificates.items()}
    res = {name: np.asarray(column) for name, column in report.resolution.items()}
    assert list(cert["status"][:1]) == [POSITIVE] and not cert["void"][0]
    assert cert["void"][1]
    # 500 draws survive the rows (250 per size bit, weights 0.7 and 0.3): n_eff is
    # 250^2 / (250 x 0.58) = 431.0, but the Wilson n is the Kish size of the
    # condition-weighted draws, the 250 equal Earth-sized ones: q_lo = 250 / 254.
    assert_allclose(res["n_eff"][0], 250 / 0.58, rtol=1e-10)
    assert_allclose(res["n_cond"][0], 250.0, rtol=1e-10)
    assert_allclose(cert["q_lo"][0], 250.0 / 254.0, rtol=1e-10)
    assert round(float(cert["q_lo"][0]), 4) == 0.9843
    text = format_report(report)
    assert "-0.0000" not in text  # chain gains that round to zero print as 0.0000
    print("Step rule, S/N 6 reports, 4,000 draws\n")
    print(text)

    gaussian_models = (
        OrbitReport(),
        ExpectedReading(GaussianGasModel(snr_one_day=[5.0, 0.0])),
        ExpectedReading(GaussianGasModel(snr_one_day=[0.0, 5.0])),
    )
    log, report = _run(toy, _gaussian_cloud(toy), gaussian_models)
    e, look_alike = _by_star(report, 0), _by_star(report, 1)
    assert_allclose(e["q_after"], [0.175, 0.337, 0.649, 0.649], atol=5e-4)
    assert_allclose(e["information"], [1.03, 0.66, 0.66, 0.0], atol=5e-3)
    assert_allclose(look_alike["information"], [-0.13, -0.22, -0.64], atol=5e-3)
    assert round(float(e["information"].sum()), 2) == 2.34
    assert round(float(look_alike["information"].sum()), 2) == -0.98
    cert = {name: np.asarray(column) for name, column in report.certificates.items()}
    res = {name: np.asarray(column) for name, column in report.resolution.items()}
    assert_allclose(res["q"][0], 0.649 / 0.70, atol=1e-3)
    assert cert["status"][0] == UNRESOLVED and cert["void"][1]
    print(
        "\nGaussian gas model, S/N 5, readings at their expected value, 4,000 draws\n"
    )
    print(format_report(report))


def _step_models():
    return (
        OrbitReport(),
        StepReportModel(snr_one_day=[6.0, 0.0]),
        StepReportModel(snr_one_day=[0.0, 6.0]),
    )


def test_two_arms_with_an_empty_log_and_different_grants(toy):
    """Two arms of one study whose policies take nothing: their logs hold only their
    grants, and their scores differ.

    Arm A is granted E's orbit, arm B its water report. Neither log holds an observed
    row, both hold one granted row released and charged at zero, and the rows differ.
    E's conditional q at the end is 0.25 under A (both gases at even odds) and 0.5
    under B (water seen); the information tables are empty (no stage after the
    grant).

    Fails if a grant's rows do not enter the log at time zero, are charged, or are not
    read at stage 0, or if an empty table of actions takes an action.
    """
    design = _design(_step_models())
    study = _study(toy, _step_rule_cloud(toy), design, 1, _e_track_claim(toy))
    world = SealedWorld(
        toy.stack(toy.atoms([[1.0]], [[1.0]], [1.0], [[1.0, 1.0]], 0.0))
    )
    nothing = TablePolicy(Action(jnp.zeros(0, int), jnp.zeros(0, int), jnp.zeros(0)))
    arms = [Arm(Grant(Action([0], [mode], [1.0]))) for mode in (0, 1)]
    logs = [execute(KEY, arm.grant, world, design, nothing) for arm in arms]
    for log in logs:
        assert int(log.count) == 1
        assert_allclose([log.rows.time_d[0], log.rows.charged_d[0]], 0.0)
        assert_allclose(log.rows.release_d[0], 0.0)
    assert [int(log.rows.model_slot[0]) for log in logs] == [0, 1]
    reports = [score(log, world, study) for log in logs]
    for report in reports:
        assert report.information["star"].shape == (0,)
        assert report.information["chain_gain"].shape == (0, 4)
    q = [float(report.resolution["q"][0]) for report in reports]
    assert_allclose(q, [0.25, 0.5], rtol=1e-10)


def test_a_table_replays_to_the_log_it_describes(toy):
    """A fixed table of actions replays to its log, row for row.

    Four actions (stars 1, 0, 1, 0; modes 1, 2, 0, 1; exposures 1.5, 2, 0.5, 3 days)
    under a budget of 4.5 days and a release latency of 0.25 days. Each row starts
    when the one before it ends (0, 1.5, 3.5), is charged its exposure and released
    0.25 days after it; the fourth (3 days) exceeds the 0.5 days left and is refused,
    charged nothing, its payload zeros. Each payload sits in its mode's model slot,
    equal to the world's draw with the row's key, and the other slots hold zeros.
    After the third row (day 4) the public view shows that row, released at day 4.25,
    in flight with its payload hidden, and the two released rows' payloads in full.

    Fails if `step` misplaces a row, its time, charge or release, reads a payload from
    the wrong model or key, charges a refused row, or shows a policy a payload before
    its release.
    """
    models = _step_models()
    design = _design(models, budget_d=4.5, latency_d=0.25)
    world = _el_world(toy)
    table = Action([1, 0, 1, 0], [1, 2, 0, 1], [1.5, 2.0, 0.5, 3.0])
    log = execute(KEY, _no_grant(), world, design, TablePolicy(table))

    rows = log.rows
    assert int(log.count) == 4
    assert_allclose(rows.star[:4], [1, 0, 1, 0])
    assert_allclose(rows.mode[:4], [1, 2, 0, 1])
    assert_allclose(rows.model_slot[:4], [1, 2, 0, 1])
    assert_allclose(rows.exposure_d[:4], [1.5, 2.0, 0.5, 3.0])
    assert_allclose(rows.time_d[:4], [0.0, 1.5, 3.5, 4.0])
    assert_allclose(rows.charged_d[:4], [1.5, 2.0, 0.5, 0.0])
    assert_allclose(rows.release_d[:4], [1.75, 3.75, 4.25, 4.0])
    assert_allclose(rows.refusal[:4], [0, 0, 0, OVER_BUDGET])
    _, noise, _ = run_keys(KEY)
    for i in range(3):
        action = jax.tree.map(lambda leaf, i=i: leaf[i], table)
        slot = int(rows.model_slot[i])
        drawn = world.observe(row_key(noise, action.star, i), models[slot], action)
        toy.assert_ulp(jax.tree.map(lambda x, i=i: x[i], rows.payload[slot]), drawn)
        for other in set(range(3)) - {slot}:
            for leaf in jax.tree.leaves(rows.payload[other]):
                assert_allclose(leaf[i], 0.0)
    for leaf in jax.tree.leaves(rows.payload):
        assert_allclose(leaf[3], 0.0)

    three = eqx.tree_at(lambda x: x.count, log, jnp.asarray(3))
    view = public_view(three, jnp.asarray(0), design.resources, design.keepout)
    assert_allclose(view.time_d, 4.0)
    assert_allclose(view.budget_left_d, 0.5)
    assert list(np.asarray(view.in_flight[:4])) == [False, False, True, False]
    assert_allclose(view.log.rows.payload[1]["bit"][0], rows.payload[1]["bit"][0])
    assert_allclose(view.log.rows.payload[2]["bit"][1], rows.payload[2]["bit"][1])
    assert_allclose(rows.payload[0]["hz"][2], 1.0)  # L is in the HZ
    assert_allclose(view.log.rows.payload[0]["hz"][2], 0.0)  # in flight: hidden


class Blind(AbstractObservationModel):
    """A model whose payload does not depend on the truth: every row reads zeros."""

    searches: ClassVar[bool] = False
    stage: ClassVar[int] = 0

    def sample(self, key, atoms, action):
        return {"value": jnp.zeros(atoms.exists.shape)}

    def stats(self, payload, atoms, star):
        return jnp.zeros(atoms.exists.shape[0])

    def finalize(self, stats, atoms, star):
        return stats, atoms

    def cell(self, payload):
        return jnp.zeros((), int)

    def cells(self, atoms, action):
        return jnp.ones((atoms.exists.shape[0], 1))


class EchoState(eqx.Module):
    """A running checksum of every view an `Echo` has been shown."""

    checksum: jax.Array = eqx.field(converter=jnp.asarray)


class Echo(AbstractPolicy):
    """An adaptive policy that reads every bit it is shown: it folds a checksum of each
    public view into its state and, four times, observes star 0 or 1 by the running
    checksum's parity.
    """

    host_only: ClassVar[bool] = True  # reads concrete values on the host

    def init(self, public):
        return EchoState(0)

    def decide(self, key, state, public):
        leaves = jax.tree.leaves(public)
        view = b"".join(np.asarray(x).tobytes() for x in leaves)
        checksum = zlib.crc32(view, int(state.checksum))
        action = Action(checksum % 2, 0, 1.0)
        return action, public.n_taken >= 4, EchoState(checksum)


@pytest.mark.parametrize(
    "policy",
    [
        pytest.param(Echo(), id="adaptive"),
        pytest.param(TablePolicy(Action([0, 1, 1, 0], [0] * 4, [1.0] * 4)), id="table"),
    ],
)
def test_a_policy_cannot_read_the_truth(toy, policy):
    """Non-interference: two worlds whose rows are forced identical give identical
    keys, states and views to the policy and a bit-identical log.

    The worlds differ in every abundance (E and L against two planets 0.3 dex below
    their floors); every row is read by a model whose payload does not depend on the
    truth, so the only way the logs could differ is a policy reading the truth. The
    same runs with a gas model that reads the truth show the policy different views,
    so the comparison can see a difference. The check is one helper, run here on an
    adaptive policy and on a table.

    Fails if `execute` shows a policy anything drawn from the sealed world beyond the
    released rows: any such bit changes a view's checksum.
    """
    other = toy.atoms([[0.0]], [[0.0]], [1.0], [[-0.3, -0.3]], 0.0)
    worlds = (_el_world(toy), SealedWorld(toy.stack(other, other)))
    blind = _design((Blind(),))
    seeing = _design((GaussianGasModel(snr_one_day=[5.0, 5.0]),))
    toy.assert_non_interference(KEY, policy, worlds, blind, seeing, _no_grant())


class Counted(AbstractPolicy):
    """Defers to `inner`, counting each time its `decide` body runs (traced or
    eager) in `COUNTED`."""

    inner: AbstractPolicy

    def init(self, public):
        return self.inner.init(public)

    def decide(self, key, state, public):
        COUNTED.append(None)
        return self.inner.decide(key, state, public)


COUNTED = []


def test_a_table_decides_with_a_traced_cursor(toy):
    """A table's decision compiles once and reads its position from its state.

    Under one `eqx.filter_jit`, three decisions on a two-action table return its two
    actions in order, then stop, and the cursor counts 1, 2, 3; the three calls trace
    once. An empty table stops at its first decision. `execute` runs the same table
    (three decisions) through one compiled `decide`: its body runs once.

    Fails if the table reads its position as a Python value (`int` of a traced
    counter raises under jit, and a concrete one retraces), skips or repeats an
    action, or stops early or late, or if `execute` calls `decide` uncompiled (its
    body would run at every decision).
    """
    design = _design(_step_models())
    log = _empty_log(_el_world(toy), design)
    view = public_view(log, jnp.asarray(0), design.resources, design.keepout)
    table = TablePolicy(Action([1, 0], [2, 1], [1.5, 0.5]))
    traces = []

    @eqx.filter_jit
    def decide(state):
        traces.append(None)
        return table.decide(KEY, state, view)

    state = table.init(view)
    taken = []
    for _ in range(3):
        action, stop, state = decide(state)
        taken.append((int(action.star), int(action.mode), bool(stop)))
    assert taken == [(1, 2, False), (0, 1, False), (0, 1, True)]
    assert int(state.cursor) == 3
    assert len(traces) == 1
    empty = TablePolicy(Action(jnp.zeros(0, int), jnp.zeros(0, int), jnp.zeros(0)))
    assert bool(empty.decide(KEY, empty.init(view), view)[1])

    COUNTED.clear()
    log = execute(KEY, _no_grant(), _el_world(toy), design, Counted(table))
    assert int(log.count) == 2
    assert len(COUNTED) == 1


class KeyLog(AbstractPolicy):
    """Takes three one-day looks at star 0 and records the key of each decision."""

    host_only: ClassVar[bool] = True  # reads concrete values on the host

    keys: list = eqx.field(default_factory=list)

    def init(self, public):
        return TableState(0)

    def decide(self, key, state, public):
        self.keys.append(np.asarray(jax.random.key_data(key)))
        return Action(0, 0, 1.0), public.n_taken >= 3, state


def test_each_decision_is_keyed_from_the_policy_key(toy):
    """Decision `i` gets the run's policy key folded with `i`, the actions taken so
    far: four decisions (three looks and the stop) get four distinct keys, each equal
    to `fold_in(policy_key, i)`.

    Fails if `execute` hands a policy the noise or world key, one key for every
    decision, or a key that depends on anything but the run key and the decision's
    position.
    """
    policy = KeyLog()
    execute(KEY, _no_grant(), _el_world(toy), _design(_step_models()), policy)
    _, _, policy_key = run_keys(KEY)
    expected = [
        jax.random.key_data(jax.random.fold_in(policy_key, i)) for i in range(4)
    ]
    assert len(policy.keys) == 4
    for got, want in zip(policy.keys, expected, strict=True):
        np.testing.assert_array_equal(got, want)
    assert len({k.tobytes() for k in policy.keys}) == 4


def test_waiting_costs_no_budget_and_releases_follow_the_calendar(toy):
    """A policy waits by asking for a later start: the wait moves the calendar, not the
    budget, and rows are released, shown and scored in calendar order.

    Budget 2 days. Row 0: E's orbit, one day from day 0, in a mode released 20 days
    after it ends (day 21). Row 1: L's orbit, one day not before day 10, in a mode
    released at once (day 11). Row 1 waits 9 days, starts at day 10 and is accepted:
    the two rows charge 2 days, all of the budget, and the clock reads day 11. At day
    11 the view shows row 1 released and row 0 in flight, with the full budget
    charged. `score` stages L at day 11 (the run's end) and E at day 21, after it. A
    third one-day row not before day 30 is refused (no budget left), charged nothing,
    and still moves the clock to day 30, its requested start. A wait, an action of
    zero exposure not before day 40, is written as a `WAITED` row: charged nothing,
    its payload zeros, released at its start, and the clock reads day 40; `score`
    stages it as no observation (the run's end moves to day 40, a stage with no
    information).

    Fails if a row starts at the time charged so far instead of on the calendar (row
    1 would start at day 1), if waiting is charged (row 1 would be refused), if
    releases are read against charged time (row 1 would wait in flight until day 2),
    if a refused row is left out of the calendar clock, or if a wait is charged,
    carries a payload, waits a latency or is scored as an observation.
    """
    models = (OrbitReport(),)
    modes = (Mode(0, 20.0), Mode(0, 0.0))
    design = Design(models, modes, Resources(2.0, END_D), no_keepout(2), CAPACITY)
    table = TablePolicy(Action([0, 1], [0, 1], [1.0, 1.0], [0.0, 10.0]))
    world = _el_world(toy)
    log = execute(KEY, _no_grant(), world, design, table)
    rows = log.rows
    assert int(log.count) == 2
    assert_allclose(rows.time_d[:2], [0.0, 10.0])
    assert_allclose(rows.charged_d[:2], [1.0, 1.0])
    assert_allclose(rows.release_d[:2], [21.0, 11.0])
    assert list(np.asarray(rows.refusal[:2])) == [0, 0]
    view = public_view(log, jnp.asarray(0), design.resources, design.keepout)
    assert_allclose([view.time_d, view.budget_left_d], [11.0, 0.0])
    assert list(np.asarray(view.in_flight[:2])) == [True, False]
    assert_allclose(view.log.rows.payload[0]["hz"][:2, 0], [0.0, 1.0])
    late = Action(0, 0, 1.0, 30.0)
    refused = eqx.filter_jit(step)(
        run_keys(KEY)[1], log, world, models, late, modes[0], design.resources,
        design.keepout,
    )  # fmt: skip
    assert int(refused.rows.refusal[2]) == OVER_BUDGET
    assert_allclose([refused.rows.charged_d[2], spent_d(refused)], [0.0, 2.0])
    assert_allclose(clock_d(refused), 30.0)
    wait = Action(1, 0, 0.0, 40.0)
    waited = eqx.filter_jit(step)(
        run_keys(KEY)[1], refused, world, models, wait, modes[0], design.resources,
        design.keepout,
    )  # fmt: skip
    assert int(waited.rows.refusal[3]) == WAITED
    assert_allclose([waited.rows.charged_d[3], spent_d(waited)], [0.0, 2.0])
    assert_allclose([waited.rows.release_d[3], clock_d(waited)], [40.0, 40.0])
    assert_allclose(waited.rows.payload[0]["hz"][3], 0.0)

    claim = Claim((toy.InHz(),))
    study = _study(toy, _step_rule_cloud(toy), design, 2, claim)
    report = score(log, world, study)
    e, look_alike = _by_star(report, 0), _by_star(report, 1)
    assert_allclose(look_alike["time_d"], [11.0])
    assert list(look_alike["end"]) == [True]
    assert_allclose(e["time_d"], [11.0, 21.0])
    assert list(e["end"]) == [True, False]
    look_alike = _by_star(score(waited, world, study), 1)
    assert_allclose(look_alike["time_d"], [11.0, 40.0])
    assert list(look_alike["end"]) == [False, True]
    assert_allclose(look_alike["information"][1], 0.0, atol=1e-12)


def test_keepout_delays_a_row_and_nothing_starts_after_the_calendar_end(toy):
    """`step` starts a row asked for in keepout when its star opens, and refuses one
    that would start after the calendar's end.

    Keepout bins of 10 days: star 0 is closed over days [0, 20), star 1 always; the
    calendar ends at day 25. A table asks for four one-day rows: star 0 at day 0
    (starts at day 20, when it opens), star 0 again (starts at day 21, when the
    instrument is free), star 1 (never opens: refused `PAST_END` at its ready time,
    day 22), and star 0 not before day 30 (after the end: refused `PAST_END`). The
    refused rows are charged nothing.

    Fails if a row starts in keepout (row 0 at day 0), if keepout is checked at the
    requested time rather than the start (row 1), if a star that never opens or a
    row past the end is accepted, or if a refused row is charged.
    """
    closed = np.array([[True, True, False], [True, True, True]])
    design = Design(
        (OrbitReport(),), (Mode(0, 0.0),), Resources(10.0, 25.0),
        Keepout(closed, 10.0), CAPACITY,
    )  # fmt: skip
    table = TablePolicy(Action([0, 0, 1, 0], [0] * 4, [1.0] * 4, [0, 0, 0, 30.0]))
    log = execute(KEY, _no_grant(), _el_world(toy), design, table)
    rows = log.rows
    assert int(log.count) == 4
    assert_allclose(rows.time_d[:4], [20.0, 21.0, 22.0, 30.0])
    assert list(np.asarray(rows.refusal[:4])) == [
        ACCEPTED, ACCEPTED, PAST_END, PAST_END
    ]  # fmt: skip
    assert_allclose(rows.charged_d[:4], [1.0, 1.0, 0.0, 0.0])


def test_a_second_world_and_star_do_not_retrace_step(toy):
    """`step` compiles once for a run: a second world, a second star, a later row and
    a refused row reuse the first program.

    Fails if a row, log or action field is weakly typed or read as a Python value, so
    that the host loop's next step traces again.
    """
    traces = []

    @eqx.filter_jit
    def counted(key, log, world, models, action, mode, resources, keepout):
        traces.append(None)
        return step(key, log, world, models, action, mode, resources, keepout)

    models = _step_models()
    design = _design(models, budget_d=2.5)
    worlds = (_el_world(toy), SealedWorld(toy.stack(*[toy.atoms(
        [[0.0]], [[1.0]], [1.0], [[-1.0, 1.0]], 0.0)] * 2)))  # fmt: skip
    for world in worlds:
        log = _empty_log(world, design)
        for star in (0, 1, 0):
            action = Action(star, 1, 1.0)
            log = counted(
                KEY, log, world, models, action, design.modes[1], design.resources,
                design.keepout,
            )  # fmt: skip
        assert_allclose(log.rows.refusal[:3], [0, 0, OVER_BUDGET])
    assert len(traces) == 1


def test_two_planets_at_a_star_run_through_execute_and_score(toy):
    """A star with two planets runs through `execute` and `score`.

    Planet 0 is E and planet 1 its look-alike, at one star; every draw holds both, the
    second a copy of the first. The stage table of star 0 (orbit, water, oxygen) gives
    each planet three stages: planet 0 is claim-true and gains, planet 1 is a
    look-alike and loses, with no chain split, and is void; both read the same q,
    since every draw's two planets agree.

    Fails if `unseal`, the pool of the truth or `score` assumes one planet per star, a
    planet slot is scored against another's truth, or the association of the truth's
    planets with the draws' slots is not the identity or accepts a mismatch of slots.
    """
    cloud = _step_rule_cloud(toy)
    cloud = eqx.tree_at(
        lambda a: (a.cloud.samples, a.x_mean, a.x_var, a.exists),
        cloud,
        (
            {k: jnp.repeat(v, 2, axis=1) for k, v in cloud.cloud.samples.items()},
            jnp.repeat(cloud.x_mean, 2, axis=1),
            jnp.repeat(cloud.x_var, 2, axis=1),
            jnp.repeat(cloud.exists, 2, axis=1),
        ),
    )
    models = _step_models()
    design = _design(models)
    study = _study(toy, cloud, design, 1, _e_track_claim(toy))
    truth = toy.atoms([[1.0, 1.0]], [[1.0, 0.0]], [1.0], [[1.0, 1.0]], 0.0)
    world = SealedWorld(toy.stack(truth))
    table = TablePolicy(Action([0, 0, 0], [0, 1, 2], [1.0] * 3))
    report = score(execute(KEY, _no_grant(), world, design, table), world, study)

    info = {name: np.asarray(column) for name, column in report.information.items()}
    assert list(info["planet"]) == [0, 0, 0, 1, 1, 1]
    assert list(info["truth"]) == [True] * 3 + [False] * 3
    assert info["information"][:3].sum() > 0 > info["information"][3:].sum()
    assert np.isnan(info["chain_gain"][3:]).all()
    assert_allclose(info["q_after"][:3], info["q_after"][3:], rtol=1e-12)
    assert list(np.asarray(report.certificates["void"])) == [False, True]

    at_star = jax.tree.map(lambda leaf: leaf[0], world.unseal())
    assert list(np.asarray(associate_planets(at_star, cloud))) == [0, 1]
    with pytest.raises(ValueError, match="planet slots"):
        associate_planets(at_star, _step_rule_cloud(toy))


def test_rows_in_flight_at_the_end_are_scored_after_a_marked_end(toy):
    """A row still in flight when the run ends is scored at its release, after a
    marked end stage, and the totals print the information at the end beside the
    total.

    Two rows at E's star, the orbit and then water, one day each, released half a day
    after they end: the run ends at day 2 with the water row in flight until day 2.5.
    E's stages end at days 1.5, 2 (the end, marked, nothing new) and 2.5, gaining
    ln 2.8, 0 and ln 2; the totals give ln 2.8 at the end and ln 5.6 in all.

    Fails if `score` drops or misdates a row released after the run ends, does not
    mark the end as its own stage, or totals the information at the end wrongly.
    """
    design = _design(_step_models(), latency_d=0.5)
    study = _study(toy, _step_rule_cloud(toy), design, 2, _e_track_claim(toy))
    world = _el_world(toy)
    table = TablePolicy(Action([0, 0], [0, 1], [1.0, 1.0]))
    report = score(execute(KEY, _no_grant(), world, design, table), world, study)

    e = _by_star(report, 0)
    assert_allclose(e["time_d"], [1.5, 2.0, 2.5])
    assert list(e["end"]) == [False, True, False]
    assert_allclose(e["information"], [np.log(2.8), 0.0, np.log(2.0)], atol=1e-10)
    totals = format_report(report).split("Information in total (nats)\n")[1]
    header, first = (line.split() for line in totals.splitlines()[:2])
    row = dict(zip(header, first, strict=True))
    assert (row["star"], row["planet"], row["member"]) == ("0", "0", "1")
    assert row["information_at_end"] == f"{np.log(2.8):.4f}"
    assert row["information"] == f"{np.log(5.6):.4f}"


def test_a_study_refuses_per_star_fields_that_disagree(toy):
    """A study checks its per-star fields against each other and the log's capacity.

    Fails if a study accepts `prior_mass_outside` with another number of stars or of
    roles than its pools, or refits whose row masks are not of the log's capacity.
    """
    design = _design(_step_models())
    good = _study(toy, _step_rule_cloud(toy), design, 2, _e_track_claim(toy))

    def remake(outside, design=design):
        return Study(
            good.claim, good.truth, good.roles.forecast, good.roles.verdict,
            good.refits, good.pool, good.stars, design, outside, good.alpha,
        )  # fmt: skip

    remake(jnp.zeros((2, 2)))
    with pytest.raises(ValueError, match="disagree on S"):
        remake(jnp.zeros((3, 2)))
    with pytest.raises(ValueError, match="disagree on R"):
        remake(jnp.zeros((2, 3)))
    small = Design(design.models, design.modes, design.resources, design.keepout, 4)
    with pytest.raises(ValueError, match="row masks"):
        remake(jnp.zeros((2, 2)), small)


def test_a_grant_in_a_mode_outside_the_table_is_refused(toy):
    """A granted action whose mode is not in the design's mode table is refused.

    Fails if `execute` indexes the mode table with a granted mode it has not checked,
    so that a bad grant raises an `IndexError` or reads another mode.
    """
    design = _design(_step_models())
    grant = Grant(Action([0], [3], [1.0]))
    nothing = TablePolicy(Action(jnp.zeros(0, int), jnp.zeros(0, int), jnp.zeros(0)))
    with pytest.raises(ValueError, match="mode table"):
        execute(KEY, grant, _el_world(toy), design, nothing)


@pytest.mark.medium(reason="scores eight stars of the toy survey with refits, ~8 s")
def test_a_found_pool_planet_is_judged_on_a_refit_and_an_undetected_one_not_at_all(
    toy,
):
    """`score` refits a star whose rows found a planet its cloud holds no track for,
    and judges that planet on the refit; an undetected pool planet has no row.

    The toy survey of eight stars (a known planet each, a pool planet with
    probability 0.9), each searched for three days by a table: the search finds most
    pool planets. The first world in which a claim-true pool planet is found is
    scored. Every pool planet the run detected has a certificate row, judged on a
    cloud that holds it: a claim-true one is positive or unresolved, its `q_hi` above
    0.1, never negative. A pool planet the run did not detect has no certificate,
    resolution or information row.

    Fails if `score` judges a found planet on the time-zero refit, which holds no
    planet in its slot (the found planets then get no certificate row, so the set of
    stars with a pool-planet row is not the set the run detected; a claim-true one
    that did get a row would be read at q = 0, and its status and `q_hi` are checked
    too), or if an undetected pool planet is certified (the draws hold nothing there).
    """
    n_star = 8
    survey = toy.survey(
        jax.random.key(7), n_star, 200, 50, pool_mass=0.9, budget_d=40.0,
        end_d=3000.0, capacity=n_star + 2, n_grant=0,
    )  # fmt: skip
    study = survey.study
    table = TablePolicy(Action(np.arange(n_star), np.zeros(n_star, int), [3.0] * 8))
    for w in range(8):
        key = jax.random.fold_in(KEY, w)
        world = jax.tree.map(
            lambda x: x[0], draw_worlds(key, study.truth, study.stars, 1)
        )
        log = execute(key, survey.grant, world, study.design, table)
        truth = world.unseal()
        detected = np.asarray(log.rows.payload[0]["detected"][:n_star, 1]) > 0.5
        claim_true = np.asarray(truth.cloud.samples["hz"][:, 0, 1]) > 0.5
        claim_true &= np.asarray(truth.x_mean[:, 0, 1, 0]) > 0.0
        if np.any(detected & claim_true):
            break
    report = score(log, world, study)
    cert = {k: np.asarray(v) for k, v in report.certificates.items()}
    pool_rows = cert["planet"] == 1
    assert set(cert["star"][pool_rows]) == set(np.flatnonzero(detected))
    for i in np.flatnonzero(pool_rows):
        s = int(cert["star"][i])
        assert bool(cert["truth"][i]) == bool(claim_true[s])
        if claim_true[s]:
            assert int(cert["status"][i]) in (POSITIVE, UNRESOLVED)
            assert cert["q_hi"][i] > 0.1
    for table_ in (report.resolution, report.information):
        undetected = ~detected[np.asarray(table_["star"])]
        assert not np.any((np.asarray(table_["planet"]) == 1) & undetected)
