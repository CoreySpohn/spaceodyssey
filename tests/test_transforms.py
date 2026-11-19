"""Every public function of `belief` and `information`, the replay and the planner's
two beliefs, the pool functions (their roles, thinning and replay), the stage a
star's rows reached, `frontier` of `value` and the survey `step`, under `jax.jit`,
`jax.vmap` over a leading world axis, and `jax.grad` where it is differentiable
(forward mode only for `absorb`).
"""

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from numpy.testing import assert_allclose

from spaceodyssey.belief import (
    Belief,
    absorb,
    chain,
    count_pmf,
    draw_claim_mass,
    log_weights,
    mark_weights,
    planning_beliefs,
    pool_rate,
    q,
    quadrature_draws,
    replay,
    replay_belief,
    role_belief,
    role_pool,
    weights,
)
from spaceodyssey.claim import Claim, GasAbove
from spaceodyssey.design import Mode, Resources, no_keepout
from spaceodyssey.information import (
    cell_probs,
    certify,
    chain_gain,
    claim_information,
    clip_q,
    diagnose,
    expected_log_score,
    information,
    log_score,
    mask_cells,
    outcome_cells,
    outcome_prob,
    refit_trigger,
    update_by_cell,
    widen_by_mass_outside,
)
from spaceodyssey.log import step
from spaceodyssey.roles import AbstractRolePrior, Roles
from spaceodyssey.toy import DetectionModel, GaussianGasModel
from spaceodyssey.types import Action, ObservationLog, Refit, Row
from spaceodyssey.value import (
    Count,
    Menu,
    completeness,
    frontier,
    reached_stage,
    replay_pool,
    thin,
)
from spaceodyssey.world import SealedWorld

N_ROLES, N_DRAWS, N_PLANETS = 3, 6, 2
CAPACITY = 4  # W of the replayed logs
FORWARD_ONLY = {"absorb"}  # a lax.while_loop: forward-mode derivatives only


class Tilt(AbstractRolePrior):
    """log density `shift * hz` of the first planet: a prior with one traced leaf.

    A tilt against the refit target, read only as a weight, so it has no sampler.
    """

    shift: jax.Array = eqx.field(converter=jnp.asarray)

    def log_prob(self, atoms, star):
        return self.shift * atoms.cloud.samples["hz"][:, 0]

    def sample(self, key, star):
        raise NotImplementedError("a tilt is read only as a weight")


def _worlds(toy):
    """Two worlds of one structure: three roles over six draws of two planets."""
    rng = np.random.default_rng(0)
    worlds = []
    for _ in range(2):
        atoms = toy.atoms(
            rng.uniform(0.0, 1.0, (N_DRAWS, N_PLANETS)),
            rng.uniform(0.0, 1.0, (N_DRAWS, N_PLANETS)),
            rng.uniform(0.2, 1.0, N_DRAWS),
            mean=rng.normal(size=(N_DRAWS, 1)),
            var=rng.uniform(0.5, 2.0, (N_DRAWS, 1)),
            n_gas=1,
        )
        pool = toy.pool(
            rng.uniform(0.0, 1.0, 3),
            rng.uniform(0.2, 1.0, (N_ROLES, 3)),
            rng.normal(),
            n_roles=N_ROLES,
        )
        log_prior = rng.normal(size=(N_ROLES, N_DRAWS))
        log_like = rng.normal(size=(N_ROLES, N_DRAWS))
        worlds.append(Belief(atoms, log_prior, log_like, pool))
    return worlds


def _cases(toy):
    """Per public function: (call, args per world, the argument to differentiate,
    a scalar of the output).

    The differentiated argument is a selector into the args, or None where the output
    is not differentiable. The scalar weights the outputs unevenly, so a derivative
    that sums to zero (the weights of a role) is still checked entry by entry.
    """
    star = toy.star
    claim = Claim((toy.InHz(), GasAbove(0, 0.3)))
    conditional = Claim((GasAbove(0, 0.3), toy.InHz(), toy.InSize()), given=1)
    plain = Claim((toy.InHz(), toy.InSize()))
    b0, b1 = _worlds(toy)
    planets = (jnp.asarray(0), jnp.asarray(1))

    def uneven(x):
        x = jnp.asarray(x)
        return jnp.sum(x * jnp.cos(jnp.arange(x.size).reshape(x.shape)))

    def diag(b, planet):
        return diagnose(b, conditional, star, planet, jnp.full(N_ROLES, 0.01))

    # Inputs are built through the shared compiled kernels; the cases call the raw ones.
    mass = jnp.full(N_ROLES, 0.01)
    d0, d1 = (
        toy.jit.diagnose(b, conditional, star, k, mass)
        for b, k in zip((b0, b1), planets, strict=True)
    )
    pmf = [toy.jit.count_pmf(b, plain, star, 3)[0] for b in (b0, b1)]
    counts = (jnp.asarray(1), jnp.asarray(2))
    roles = Roles(Tilt("toy", 0.0), Tilt("toy", jnp.array([0.5, -0.4])))
    roles1 = Roles(Tilt("toy", 0.2), Tilt("toy", jnp.array([0.1, 0.7])))

    def reweight(roles, b):
        return role_belief(
            b.atoms, roles, b.log_like, star, b.pool, min_n_eff=1.0, max_khat=jnp.inf
        )

    hz_only = Claim((toy.InHz(),))
    model = GaussianGasModel(snr_one_day=[2.0], edges=[0.0, 0.6])
    actions = (Action(0, 0, 1.0), Action(0, 0, 2.5))
    cells = [model.cells(b.atoms, a) for b, a in zip((b0, b1), actions, strict=True)]
    detect = (jnp.array([0.3, 0.9, 0.1]), jnp.array([0.6, 0.2, 0.5]))

    # The planner reads point states: the frontier runs on the blocks' means, the cell
    # update and the claim-bit information on quadrature nodes, and the split is its
    # own case.
    point = [
        eqx.tree_at(lambda b: b.atoms.x_var, b, 0.0 * b.atoms.x_var) for b in (b0, b1)
    ]
    on_nodes = [toy.jit.quadrature_draws(b, claim, 4) for b in (b0, b1)]
    node_cells = [
        toy.jit.outcome_cells(b, model, a)
        for b, a in zip(on_nodes, actions, strict=True)
    ]
    keep = (jnp.arange(9) % 2 == 0, jnp.arange(9) < 5)

    # Replays: two logs of one structure, a gas row per slot, released over time.
    gas = (model,)
    rng = np.random.default_rng(1)

    def log(release_d):
        zero_i, zero_d = jnp.zeros(CAPACITY, int), jnp.zeros(CAPACITY)
        reading = rng.normal(size=(CAPACITY, N_PLANETS, 1))
        payload = {"snr": jnp.full(reading.shape, 2.0), "reading": reading}
        rows = Row(
            zero_i, zero_i, zero_d, zero_d + 1.0, zero_d, release_d, zero_i, zero_i,
            (payload,),
        )  # fmt: skip
        return ObservationLog(rows, jnp.asarray(3))

    logs = (log(jnp.array([0.0, 0.5, 1.5, 3.0])), log(jnp.array([0.0, 2.5, 0.5, 1.0])))
    refits = [Refit(b.atoms, jnp.zeros(CAPACITY, bool), 0) for b in (b0, b1)]
    times = (jnp.asarray(1.0), jnp.asarray(2.0))
    early = [
        toy.jit.replay(r, lg, gas, star, jnp.asarray(0.2))
        for r, lg in zip(refits, logs, strict=True)
    ]
    full = [
        toy.jit.replay(r, lg, gas, star, t)
        for r, lg, t in zip(refits, logs, times, strict=True)
    ]

    def reading_of(a):
        return a[1].rows.payload[0]["reading"]

    def belief_scalar(b):
        out = uneven(weights(b)) + uneven(b.atoms.x_mean) + uneven(b.atoms.x_var)
        if b.atoms.nodes is not None:
            nodes = b.atoms.nodes
            out = out + uneven(nodes.x) + uneven(nodes.weights)
        return out

    def cells_scalar(c):
        return uneven(c.draw) + uneven(c.node)

    # One star per world for `frontier`, valued by `Count` on a search.
    search = (DetectionModel(0.6, 0.1),)
    menus = (
        Menu(jnp.array([[1.0, 2.5]]), jnp.array([[True, True]]), (0, 0)),
        Menu(jnp.array([[0.5, 1.5]]), jnp.array([[True, False]]), (0, 0)),
    )

    def value(b, utility, menu):
        return frontier(
            *toy.split(b), plain, star, utility, search, menu, jnp.asarray(0.0), 1
        )

    # Searches for the pool and the stage: two logs of one structure, each row a
    # search of its own exposure, some of them finding a planet.
    def search_log(exposure_d, release_d, detected):
        zero_i, zero_d = jnp.zeros(CAPACITY, int), jnp.zeros(CAPACITY)
        payload = {
            "exposure_d": exposure_d,
            "p_hz_one_day": zero_d + 0.6,
            "p_other_one_day": zero_d + 0.1,
            "detected": jnp.asarray(detected, float),
        }
        rows = Row(
            zero_i, zero_i, zero_d, exposure_d, exposure_d, release_d, zero_i, zero_i,
            (payload,),
        )  # fmt: skip
        return ObservationLog(rows, jnp.asarray(3))

    searches = (
        search_log(
            jnp.array([1.0, 0.5, 2.0, 1.0]), jnp.array([0.0, 0.5, 1.5, 3.0]),
            [[0, 0], [1, 0], [0, 0], [0, 1]],
        ),
        search_log(
            jnp.array([0.7, 1.5, 0.2, 1.0]), jnp.array([2.5, 0.0, 0.5, 1.0]),
            [[0, 0], [0, 0], [0, 1], [1, 1]],
        ),
    )  # fmt: skip

    # `step` on two sealed worlds of one star each, taking the free slot of a log.
    truths = [
        SealedWorld(toy.stack(toy.atoms([[0.4, 0.8]], [[1.0, 0.0]], [1.0], m, 0.0, 1)))
        for m in (-0.5, 0.9)
    ]
    keys = (jax.random.key(3), jax.random.key(4))

    def take(key, lg, world, action):
        return step(
            key, lg, world, gas, action, Mode(0, 0.5), Resources(10.0, 365.0),
            no_keepout(1),
        )  # fmt: skip

    return {
        "weights": (weights, [(b0,), (b1,)], lambda a: a[0].log_prior_ratio, uneven),
        "log_weights": (log_weights, [(b0,), (b1,)], lambda a: a[0].log_like, uneven),
        "draw_claim_mass": (
            lambda b, k: draw_claim_mass(b, conditional, star, k),
            [(b0, planets[0]), (b1, planets[1])],
            lambda a: a[0].atoms.x_mean,
            uneven,
        ),
        "replay": (
            lambda r, lg, t: replay(r, lg, gas, star, t),
            [(refits[0], logs[0], times[0]), (refits[1], logs[1], times[1])],
            reading_of,
            lambda r: sum(uneven(x) for x in jax.tree.leaves(r.stats)),
        ),
        "absorb": (
            lambda r, lg, t: absorb(r, lg, gas, star, t),
            [(early[0], logs[0], times[0]), (early[1], logs[1], times[1])],
            reading_of,
            lambda r: sum(uneven(x) for x in jax.tree.leaves(r.stats)),
        ),
        "replay_belief": (
            lambda r, roles: replay_belief(r, roles, gas, star, b0.pool, **toy.ungated),
            [(full[0], roles), (full[1], roles1)],
            lambda a: a[0].stats[0]["weighted_sum"],
            belief_scalar,
        ),
        "planning_beliefs": (
            lambda frozen, now: planning_beliefs(
                frozen, now, roles, gas, star, b0.pool, claim, 4, **toy.ungated
            ),
            [(early[0], full[0]), (early[1], full[1])],
            lambda a: a[1].stats[0]["weighted_sum"],
            lambda pair: belief_scalar(pair[0]) + belief_scalar(pair[1]),
        ),
        "role_pool": (
            lambda roles, b: role_pool(
                b.pool.marks,
                roles,
                b.pool.log_like,
                b.pool.log_mass,
                star,
                min_n_eff=1.0,
                max_khat=jnp.inf,
            ),
            [(roles, b0), (roles1, b1)],
            lambda a: a[0].verdict.shift,
            lambda p: uneven(mark_weights(p)),
        ),
        "replay_pool": (
            lambda p, lg, t: replay_pool(p, refits[0], lg, search, t),
            [(b0.pool, searches[0], times[0]), (b1.pool, searches[1], times[1])],
            lambda a: a[1].rows.exposure_d,
            lambda p: uneven(p.log_mass) + uneven(mark_weights(p)),
        ),
        "reached_stage": (
            lambda lg, t: reached_stage(lg, search, jnp.asarray(0), t),
            [(searches[0], times[0]), (searches[1], times[1])],
            None,
            None,
        ),
        "step": (
            take,
            [
                (keys[0], logs[0], truths[0], actions[0]),
                (keys[1], logs[1], truths[1], actions[1]),
            ],
            None,
            None,
        ),
        "quadrature_draws": (
            lambda b: quadrature_draws(b, claim, 4),
            [(b0,), (b1,)],
            lambda a: a[0].atoms.x_mean,
            belief_scalar,
        ),
        "frontier": (
            value,
            [
                (toy.stack(point[0]), Count(), menus[0]),
                (toy.stack(point[1]), Count(), menus[1]),
            ],
            lambda a: a[2].exposure_d,
            lambda f: uneven(f.reward),
        ),
        "q": (
            lambda b, k: q(b, claim, star, k),
            [(b0, planets[0]), (b1, planets[1])],
            lambda a: a[0].log_prior_ratio,
            uneven,
        ),
        "chain": (
            lambda b, k: chain(b, conditional, star, k),
            [(b0, planets[0]), (b1, planets[1])],
            lambda a: a[0].atoms.x_mean,
            uneven,
        ),
        "count_pmf": (
            lambda b: count_pmf(b, plain, star, 3),
            [(b0,), (b1,)],
            lambda a: a[0].pool.log_mass,
            uneven,
        ),
        "pool_rate": (
            lambda p: pool_rate(p, plain, star),
            [(b0.pool,), (b1.pool,)],
            lambda a: a[0].log_like,
            uneven,
        ),
        "role_belief": (
            reweight,
            [(roles, b0), (roles1, b1)],
            lambda a: a[0].verdict.shift,
            lambda b: uneven(weights(b)),
        ),
        "log_score": (
            log_score,
            [(pmf[0], counts[0]), (pmf[1], counts[1])],
            lambda a: a[0],
            uneven,
        ),
        "information": (
            information,
            [(pmf[0], pmf[1], counts[0]), (pmf[1], pmf[0], counts[1])],
            lambda a: a[1],
            uneven,
        ),
        "chain_gain": (
            lambda before, after, k: chain_gain(before, after, plain, star, k, 1),
            [(b0, b1, planets[0]), (b1, b0, planets[1])],
            lambda a: a[1].log_prior_ratio,
            uneven,
        ),
        "diagnose": (
            diag,
            [(b0, planets[0]), (b1, planets[1])],
            lambda a: a[0].log_prior_ratio,
            lambda d: uneven(d.q) + uneven(d.mc_se) + uneven(d.n_eff),
        ),
        "clip_q": (
            clip_q,
            [(d0.q, jnp.asarray(0.01)), (d1.q, jnp.asarray(0.02))],
            lambda a: a[0],
            uneven,
        ),
        "certify": (
            lambda d: certify(d, 0.05),
            [(d0,), (d1,)],
            lambda a: a[0].q,
            lambda c: c.q_lo + 2.0 * c.q_hi,
        ),
        "refit_trigger": (lambda d: refit_trigger(d, 0.05), [(d0,), (d1,)], None, None),
        "widen_by_mass_outside": (
            widen_by_mass_outside,
            [
                (d0.q, d0.q + 0.1, jnp.full(N_ROLES, 0.01)),
                (d1.q, d1.q + 0.2, jnp.full(N_ROLES, 0.03)),
            ],
            lambda a: a[2],
            uneven,
        ),
        "mark_weights": (
            mark_weights,
            [(b0.pool,), (b1.pool,)],
            lambda a: a[0].log_like,
            uneven,
        ),
        "expected_log_score": (
            expected_log_score,
            [(pmf[0], pmf[1]), (pmf[1], pmf[0])],
            lambda a: a[0],
            uneven,
        ),
        "outcome_prob": (
            outcome_prob,
            [(b0, cells[0]), (b1, cells[1])],
            lambda a: a[0].log_prior_ratio,
            uneven,
        ),
        "outcome_cells": (
            lambda b, a: outcome_cells(b, model, a),
            [(on_nodes[0], actions[0]), (on_nodes[1], actions[1])],
            lambda a: a[0].atoms.nodes.x,
            cells_scalar,
        ),
        "cell_probs": (
            cell_probs,
            [(on_nodes[0], node_cells[0]), (on_nodes[1], node_cells[1])],
            lambda a: a[0].atoms.nodes.weights,
            uneven,
        ),
        "mask_cells": (
            mask_cells,
            [(node_cells[0], keep[0]), (node_cells[1], keep[1])],
            lambda a: a[0].node,
            cells_scalar,
        ),
        "update_by_cell": (
            update_by_cell,
            [
                (on_nodes[0], node_cells[0], jnp.asarray(1)),
                (on_nodes[1], node_cells[1], jnp.asarray(5)),
            ],
            lambda a: a[1].node,
            belief_scalar,
        ),
        "claim_information": (
            lambda b, m, a, k: claim_information(*toy.split(b), m, a, claim, star, k),
            [
                (on_nodes[0], model, actions[0], planets[0]),
                (on_nodes[1], model, actions[1], planets[1]),
            ],
            lambda a: a[0].log_prior_ratio,
            uneven,
        ),
        "thin": (
            thin,
            [(b0.pool, detect[0]), (b1.pool, detect[1])],
            lambda a: a[1],
            lambda p: uneven(p.log_mass) + uneven(mark_weights(p)),
        ),
        "completeness": (
            lambda p, a: completeness(p, DetectionModel(0.6, 0.1), a, hz_only, star),
            [(b0.pool, actions[0]), (b1.pool, actions[1])],
            lambda a: a[1].exposure_d,
            uneven,
        ),
    }


CASES = [
    "weights",
    "log_weights",
    "draw_claim_mass",
    "q",
    "chain",
    "count_pmf",
    "pool_rate",
    "role_belief",
    "log_score",
    "information",
    "chain_gain",
    "diagnose",
    "clip_q",
    "certify",
    "refit_trigger",
    "widen_by_mass_outside",
    "mark_weights",
    "expected_log_score",
    "outcome_prob",
    "outcome_cells",
    "cell_probs",
    "mask_cells",
    "update_by_cell",
    "claim_information",
    "thin",
    "completeness",
    "replay",
    "absorb",
    "replay_belief",
    "planning_beliefs",
    "quadrature_draws",
    "role_pool",
    "replay_pool",
    "reached_stage",
    "step",
    "frontier",
]


@pytest.fixture(scope="module")
def cases(toy):
    return _cases(toy)


def _close(actual, expected):
    for a, e in zip(jax.tree.leaves(actual), jax.tree.leaves(expected), strict=True):
        assert_allclose(a, e, rtol=1e-10, atol=1e-12)


# Cases whose compiles alone sit at the unit budget: the lookahead under jit, vmap and
# two derivatives, and the planning beliefs' quadrature under the same.
MEDIUM = {
    "frontier": "compiles the lookahead four ways, about 5 s",
    "planning_beliefs": "compiles two replays and the quadrature four ways, about 5 s",
}


def _case(name):
    if name not in MEDIUM:
        return name
    return pytest.param(name, marks=pytest.mark.medium(reason=MEDIUM[name]))


@pytest.mark.parametrize("name", [_case(n) for n in CASES])
def test_public_functions_jit_vmap_and_grad(cases, name):
    """Each public function gives its eager result under `jax.jit` and, for two
    worlds stacked on a leading axis, under `jax.vmap`; where differentiable, its
    derivative matches finite differences forward and reverse.

    The planet index is traced (and differs between the worlds), the count is traced,
    and the stacked verdict prior of `role_belief` is mapped inside the vmap.

    Fails if a public function branches in Python on a traced value, needs a concrete
    planet index or count, reads a shape from the wrong axis so that it cannot batch
    over worlds, or has a derivative that raises, is NaN or is wrong.
    """
    fn, (args0, args1), wrt, scalar = cases[name]
    modes = ("fwd",) if name in FORWARD_ONLY else ("fwd", "rev")
    out0, out1 = fn(*args0), fn(*args1)
    _close(jax.jit(fn)(*args0), out0)
    stacked = jax.tree.map(lambda *x: jnp.stack(x), args0, args1)
    expected = jax.tree.map(lambda *x: jnp.stack(x), out0, out1)
    _close(jax.jit(jax.vmap(fn))(*stacked), expected)
    if wrt is None:
        return

    @jax.jit
    def f(x):
        return scalar(fn(*eqx.tree_at(wrt, args0, x)))

    x0 = wrt(args0)
    derivative = jax.jacfwd(f) if name in FORWARD_ONLY else jax.grad(f)
    assert np.all(np.isfinite(derivative(x0)))
    check_grads(f, (x0,), order=1, modes=modes, eps=1e-5, atol=1e-6, rtol=1e-6)
