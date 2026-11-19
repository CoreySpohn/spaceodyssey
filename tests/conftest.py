"""Shared test configuration: float64 is set once, here."""

import importlib.util
import os
import zlib
from typing import ClassVar

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest
from hypothesis import settings
from jax.scipy.stats import norm
from jaxtyping import install_import_hook
from numpy.testing import assert_array_max_ulp

# The jaxtyping annotations are checked in tests only, and on by default; the hook must
# be installed before the package is first imported. SPACEODYSSEY_TYPECHECK=0 turns it
# off.
TYPECHECK = os.environ.get("SPACEODYSSEY_TYPECHECK", "1") == "1"
if TYPECHECK:
    install_import_hook("spaceodyssey", "beartype.beartype")

from spaceodyssey import belief, information, value  # noqa: E402
from spaceodyssey._standin import diagnostics  # noqa: E402
from spaceodyssey.belief import Belief, select_roles  # noqa: E402
from spaceodyssey.claim import AbstractConjunct  # noqa: E402
from spaceodyssey.log import execute  # noqa: E402
from spaceodyssey.policy import AbstractPolicy  # noqa: E402
from spaceodyssey.types import (  # noqa: E402
    AbstractObservationModel,
    AbstractRolePrior,
    Atoms,
    Pool,
    WeightedCloud,
)

jax.config.update("jax_enable_x64", True)

# Released hwoutils versions may not include the optional plugins yet.
pytest_plugins = []
if importlib.util.find_spec("hwoutils.testbudget") is not None:
    pytest_plugins.append("hwoutils.testbudget.plugin")
if importlib.util.find_spec("hwoutils.internal_refs") is not None:
    pytest_plugins.append("hwoutils.internal_refs")

settings.register_profile("quick", max_examples=25, deadline=None)
settings.register_profile("nightly", max_examples=200, deadline=None)
settings.load_profile(os.environ.get("HYPOTHESIS_PROFILE", "quick"))


class _Recording(AbstractPolicy):
    """A policy that records a checksum of everything it is handed at each decision
    (the key, its state and the public view) and then defers to `inner`.
    """

    host_only: ClassVar[bool] = True  # reads concrete values on the host

    inner: AbstractPolicy
    seen: list = eqx.field(default_factory=list)

    def init(self, public):
        return self.inner.init(public)

    def decide(self, key, state, public):
        leaves = [jax.random.key_data(key), *jax.tree.leaves((state, public))]
        self.seen.append(zlib.crc32(b"".join(np.asarray(x).tobytes() for x in leaves)))
        return self.inner.decide(key, state, public)


class _Jit:
    """One compiled wrapper per hot kernel, shared by every test file.

    A kernel called outside `jit` compiles each of its operations as a program of its
    own for every new shape (about 100 per call of `diagnose`); through one shared
    `eqx.filter_jit` wrapper it compiles once per shape and structure, and a second
    file calling it at the same shape reuses the program. Static arguments (claims,
    models, `n_max`) are part of the key, arrays are traced.
    """

    diagnose = staticmethod(eqx.filter_jit(information.diagnose))
    certify = staticmethod(eqx.filter_jit(information.certify))
    refit_trigger = staticmethod(eqx.filter_jit(information.refit_trigger))
    pareto_khat = staticmethod(eqx.filter_jit(jax.vmap(diagnostics.pareto_khat)))
    q = staticmethod(eqx.filter_jit(belief.q))
    chain = staticmethod(eqx.filter_jit(belief.chain))
    weights = staticmethod(eqx.filter_jit(belief.weights))
    count_pmf = staticmethod(eqx.filter_jit(belief.count_pmf))
    pool_rate = staticmethod(eqx.filter_jit(belief.pool_rate))
    replay = staticmethod(eqx.filter_jit(belief.replay))
    absorb = staticmethod(eqx.filter_jit(belief.absorb))
    replay_belief = staticmethod(eqx.filter_jit(belief.replay_belief))
    role_belief = staticmethod(eqx.filter_jit(belief.role_belief))
    planning_beliefs = staticmethod(eqx.filter_jit(belief.planning_beliefs))
    quadrature_draws = staticmethod(eqx.filter_jit(belief.quadrature_draws))
    mark_weights = staticmethod(eqx.filter_jit(belief.mark_weights))
    claim_information = staticmethod(eqx.filter_jit(information.claim_information))
    thin = staticmethod(eqx.filter_jit(value.thin))
    completeness = staticmethod(eqx.filter_jit(value.completeness))


class _Flat(AbstractRolePrior):
    """A constant density on `chart`: the refit prior of the toy clouds, so a role's
    prior ratio is its own log prior.
    """

    def log_prob(self, atoms, star):
        return jnp.zeros(atoms.x_mean.shape[0])

    def sample(self, key, star):
        raise NotImplementedError("a flat prior has no sampler")


# Toy state for the layer tests: per-draw, per-planet `hz` and `size` bits plus a gas
# block. Builders, conjuncts, the toy prior and search, and the assertion and stacking
# helpers are exposed through the `toy` fixture, not imported.
class _Toy:
    """Constructed fixtures: conjuncts, a prior and a search on the toy state, belief
    builders, and the helpers the layer tests share.
    """

    star = ()  # the toy's star is an empty pytree, never None
    jit = _Jit  # the shared compiled kernels
    Flat = _Flat
    # Reweighting thresholds that refuse nothing: the tests read the raw beliefs.
    ungated: ClassVar[dict] = {"min_n_eff": 0.0, "max_khat": jnp.inf}

    class InHz(AbstractConjunct):
        """The `hz` bit of the toy state."""

        def prob(self, atoms, star):
            return atoms.cloud.samples["hz"]

    class InSize(AbstractConjunct):
        """The `size` bit of the toy state."""

        def prob(self, atoms, star):
            return atoms.cloud.samples["size"]

    @staticmethod
    def atoms(hz, size, w, mean=0.0, var=1.0, n_gas=2, exists=True, prior=None):
        """Atoms over draws; `hz` and `size` have shape (A, K), `w` is the weight over
        draws.

        `mean` and `var` are the gas block, a scalar, a (G,) vector or an (A, G) array
        shared by the K planets of a draw. `exists` broadcasts to (A, K); every slot
        exists by default. The cloud's prior is `prior`, flat on "toy" by default, and
        its target is a placeholder zero.
        """
        hz, size = jnp.asarray(hz, float), jnp.asarray(size, float)
        n_draw, n_planet = hz.shape
        block = lambda x: jnp.broadcast_to(  # noqa: E731
            jnp.broadcast_to(jnp.asarray(x, float), (n_draw, n_gas))[:, None, :],
            (n_draw, n_planet, n_gas),
        )
        cloud = WeightedCloud(
            {"hz": hz, "size": size},
            jnp.log(jnp.asarray(w, float)),
            jnp.zeros(n_draw),
            jnp.nan,
            _Flat("toy") if prior is None else prior,
        )
        return Atoms(
            cloud,
            block(mean),
            block(var),
            jnp.broadcast_to(jnp.asarray(exists, bool), (n_draw, n_planet)),
        )

    @classmethod
    def pool(
        cls,
        mark_hz,
        mark_weights,
        log_mass,
        n_roles=1,
        like=1.0,
        cloud=1.0,
        ref=1.0,
    ):
        """A pool whose marks have the given claim bit and per-role weights (R, Am).

        `mark_weights` is the marks' role prior and `like` their likelihood, both linear
        and broadcast to (R, Am); `cloud` and `ref` are the marks' cloud weight and the
        density of the cloud's prior, linear, (Am,), so the role prior ratio is
        `mark_weights / ref`.
        """
        mark_hz = jnp.asarray(mark_hz, float)[:, None]
        n_mark = mark_hz.shape[0]
        cloud = jnp.broadcast_to(jnp.asarray(cloud, float), (n_mark,))
        marks = cls.atoms(mark_hz, jnp.zeros_like(mark_hz), cloud)
        log_mass = jnp.broadcast_to(jnp.asarray(log_mass, float), (n_roles,))
        log_ratio = jnp.log(jnp.asarray(mark_weights, float)) - jnp.log(
            jnp.broadcast_to(jnp.asarray(ref, float), (n_mark,))
        )
        log_like = jnp.broadcast_to(jnp.log(jnp.asarray(like, float)), log_ratio.shape)
        return Pool(marks, log_ratio, log_like, log_mass)

    @classmethod
    def empty_pool(cls, n_roles=1, n_gas=2):
        """A pool with no expected planets; its one mark is a point gas state."""
        marks = cls.atoms([[0.0]], [[0.0]], [1.0], 0.0, 0.0, n_gas)
        pool = cls.pool([0.0], jnp.ones((n_roles, 1)), -jnp.inf, n_roles)
        return eqx.tree_at(lambda p: p.marks, pool, marks)

    @classmethod
    def belief(cls, atoms, pool=None, n_roles=1):
        """A belief with flat priors and no likelihood."""
        n_draw = atoms.cloud.log_weights.shape[0]
        flat = jnp.zeros((n_roles, n_draw))
        return Belief(
            atoms, flat, flat, cls.empty_pool(n_roles) if pool is None else pool
        )

    @staticmethod
    def split(b):
        """The planner's two beliefs from one hand-built two-role belief: the forecast
        (role 0) and the verdict member (role 1), sharing the draws. Leading axes (a
        stack of stars) are kept.
        """
        return select_roles(b, slice(0, 1)), select_roles(b, slice(1, 2))

    class GasPrior(AbstractRolePrior):
        """One planet in the toy box (hz = size = 1) whose `n_gas` gases are iid
        N(mean, sd^2); it samples a point state.
        """

        mean: jax.Array = eqx.field(converter=jnp.asarray, default=0.0)
        sd: jax.Array = eqx.field(converter=jnp.asarray, default=1.0)
        n_gas: int = eqx.field(static=True, default=1)

        def log_prob(self, atoms, star):
            return norm.logpdf(atoms.x_mean, self.mean, self.sd).sum(axis=(1, 2))

        def sample(self, key, star):
            x = self.mean + self.sd * jax.random.normal(key, (1, 1, self.n_gas))
            block = (x[0], 0.0, self.n_gas)
            return _Toy.atoms([[1.0]], [[1.0]], [1.0], *block, prior=self)

    class Search(AbstractObservationModel):
        """A test search of planet 0: per day of exposure it detects the planet with
        probability `hit` when its `hz` bit is set and `other` otherwise, so
        `1 - (1 - p)**exposure_d` in all, smooth in the exposure. A slot with no
        planet is always missed. Cell 0 is the miss, cell 1 the detection, which
        reaches stage 1; a payload carries the exposure and whether the planet was
        detected.
        """

        searches: ClassVar[bool] = True
        stage: ClassVar[int] = 1  # a detection opens a spectrum
        hit: jax.Array = eqx.field(converter=jnp.asarray)
        other: jax.Array = eqx.field(converter=jnp.asarray)

        def _detect(self, atoms, exposure_d):
            p = jnp.where(atoms.cloud.samples["hz"][:, 0] > 0.5, self.hit, self.other)
            d = -jnp.expm1(exposure_d * jnp.log1p(-p))
            return jnp.where(atoms.exists[:, 0], d, 0.0)

        def cells(self, atoms, action):
            d = self._detect(atoms, action.exposure_d)
            return jnp.stack([1.0 - d, d], axis=-1)

        def stats(self, payload, atoms, star):
            d = self._detect(atoms, payload["exposure_d"])
            p = jnp.where(payload["detected"] > 0.5, d, 1.0 - d)
            return jnp.log(p)

        def finalize(self, stats, atoms, star):
            return stats, atoms

        def sample(self, key, atoms, action):
            raise NotImplementedError("a planning search: read through cells")

    @staticmethod
    def assert_non_interference(key, policy, worlds, blind, seeing, grant):
        """Swapped-truth non-interference of any policy.

        The two `worlds` hold different truths. Under the design `blind`, whose models
        draw payloads that do not depend on the truth, their rows are forced identical,
        so the policy must be handed bit-identical keys, states and public views at
        every decision (checksummed as it is called) and write bit-identical logs; any
        difference is the policy reading the truth. Under the design `seeing`, whose
        models read the truth, the views must differ, so the comparison can see one.
        """

        def run(world, design):
            recording = _Recording(policy)
            log = execute(key, grant, world, design, recording)
            return recording.seen, log

        (seen_a, log_a), (seen_b, log_b) = (run(world, blind) for world in worlds)
        assert len(seen_a) > 1
        assert seen_a == seen_b
        jax.tree.map(np.testing.assert_array_equal, log_a, log_b)
        assert run(worlds[0], seeing)[0] != run(worlds[1], seeing)[0]

    @staticmethod
    def stack(*trees):
        """Stack pytrees of one structure along a new leading axis."""
        return jax.tree.map(lambda *leaves: jnp.stack(leaves), *trees)

    @staticmethod
    def assert_ulp(actual, expected, maxulp=4):
        """Every leaf agrees to `maxulp` ULPs where finite, with -inf in the same
        places.
        """

        def one(a, b):
            a, b = np.asarray(a), np.asarray(b)
            np.testing.assert_array_equal(np.isneginf(a), np.isneginf(b))
            finite = np.isfinite(a)
            assert_array_max_ulp(a[finite], b[finite], maxulp=maxulp)

        jax.tree.map(one, actual, expected)


@pytest.fixture(scope="session")
def toy():
    return _Toy


@pytest.fixture(scope="session")
def typecheck():
    """Whether the jaxtyping import hook is on."""
    return TYPECHECK
