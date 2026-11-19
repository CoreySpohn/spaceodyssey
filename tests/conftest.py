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

from spaceodyssey.belief import Belief  # noqa: E402
from spaceodyssey.claim import AbstractConjunct  # noqa: E402
from spaceodyssey.log import execute  # noqa: E402
from spaceodyssey.policy import AbstractPolicy  # noqa: E402
from spaceodyssey.roles import AbstractRolePrior  # noqa: E402
from spaceodyssey.types import (  # noqa: E402
    AbstractObservationModel,
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

    inner: AbstractPolicy
    seen: list = eqx.field(default_factory=list)

    def init(self, public):
        return self.inner.init(public)

    def decide(self, key, state, public):
        leaves = [jax.random.key_data(key), *jax.tree.leaves((state, public))]
        self.seen.append(zlib.crc32(b"".join(np.asarray(x).tobytes() for x in leaves)))
        return self.inner.decide(key, state, public)


# Toy state for the layer tests: per-draw, per-planet `hz` and `size` bits plus a gas
# block. Builders, conjuncts, the toy prior and search, and the assertion and stacking
# helpers are exposed through the `toy` fixture, not imported.
class _Toy:
    """Constructed fixtures: conjuncts, a prior and a search on the toy state, belief
    builders, and the helpers the layer tests share.
    """

    star = ()  # the toy's star is an empty pytree, never None

    class InHz(AbstractConjunct):
        """The `hz` bit of the toy state."""

        def prob(self, atoms, star):
            return atoms.cloud.samples["hz"]

    class InSize(AbstractConjunct):
        """The `size` bit of the toy state."""

        def prob(self, atoms, star):
            return atoms.cloud.samples["size"]

    @staticmethod
    def atoms(hz, size, w, mean=0.0, var=1.0, n_gas=2, exists=True):
        """Atoms over draws; `hz` and `size` have shape (A, K), `w` is the weight over
        draws.

        `mean` and `var` are the gas block, a scalar, a (G,) vector or an (A, G) array
        shared by the K planets of a draw. `exists` broadcasts to (A, K); every slot
        exists by default.
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
            jnp.nan,
            jnp.zeros(n_draw),
            chart="toy",
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

        `mark_weights` is the marks' prior and `like` their likelihood, both linear and
        broadcast to (R, Am); `cloud` and `ref` are the marks' cloud weight and
        reference density, linear, (Am,).
        """
        mark_hz = jnp.asarray(mark_hz, float)[:, None]
        n_mark = mark_hz.shape[0]
        cloud = jnp.broadcast_to(jnp.asarray(cloud, float), (n_mark,))
        marks = cls.atoms(mark_hz, jnp.zeros_like(mark_hz), cloud)
        marks = eqx.tree_at(
            lambda a: a.cloud.log_ref_density,
            marks,
            jnp.broadcast_to(jnp.log(jnp.asarray(ref, float)), (n_mark,)),
        )
        log_mass = jnp.broadcast_to(jnp.asarray(log_mass, float), (n_roles,))
        log_prior = jnp.log(jnp.asarray(mark_weights, float))
        log_like = jnp.broadcast_to(jnp.log(jnp.asarray(like, float)), log_prior.shape)
        return Pool(marks, log_prior, log_like, log_mass)

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
            return _Toy.atoms([[1.0]], [[1.0]], [1.0], x[0], 0.0, self.n_gas)

    class Search(AbstractObservationModel):
        """A test search of planet 0: per day of exposure it detects the planet with
        probability `hit` when its `hz` bit is set and `other` otherwise, so
        `1 - (1 - p)**exposure_d` in all, smooth in the exposure. A slot with no
        planet is always missed. Cell 0 is the miss, cell 1 the detection. Read only
        through `cells`.
        """

        searches: ClassVar[bool] = True
        hit: jax.Array = eqx.field(converter=jnp.asarray)
        other: jax.Array = eqx.field(converter=jnp.asarray)

        def cells(self, atoms, action):
            p = jnp.where(atoms.cloud.samples["hz"][:, 0] > 0.5, self.hit, self.other)
            d = -jnp.expm1(action.exposure_d * jnp.log1p(-p))
            d = jnp.where(atoms.exists[:, 0], d, 0.0)
            return jnp.stack([1.0 - d, d], axis=-1)

        def sample(self, key, atoms, action):
            raise NotImplementedError("read only through cells")

        def stats(self, payload, atoms, star):
            raise NotImplementedError("read only through cells")

        def finalize(self, stats, atoms, star):
            raise NotImplementedError("read only through cells")

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
            log = execute(key, recording, world, design, grant)
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
