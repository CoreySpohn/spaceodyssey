"""Shared test configuration: float64 is set once, here."""

import importlib.util
import os

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest
from hypothesis import settings
from jaxtyping import install_import_hook

# The jaxtyping annotations are checked in tests only, and on by default; the hook must
# be installed before the package is first imported. SPACEODYSSEY_TYPECHECK=0 turns it
# off.
TYPECHECK = os.environ.get("SPACEODYSSEY_TYPECHECK", "1") == "1"
if TYPECHECK:
    install_import_hook("spaceodyssey", "beartype.beartype")

from spaceodyssey.belief import Belief  # noqa: E402
from spaceodyssey.claim import AbstractConjunct  # noqa: E402
from spaceodyssey.types import Atoms, Pool, WeightedCloud  # noqa: E402

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


# Toy state for the layer tests: per-draw, per-planet `hz` and `size` bits plus a gas
# block. Builders and conjuncts are exposed through the `toy` fixture, not imported.
class _Toy:
    """Constructed fixtures: conjuncts on the toy state and belief builders."""

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
            jnp.zeros(n_draw),
            "toy",
        )
        return Atoms(
            cloud,
            jnp.zeros((n_draw, n_planet, n_gas), int),
            block(mean),
            block(var),
            jnp.zeros((n_draw, 1), bool),
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
        log_mass0=None,
        like0=None,
    ):
        """A pool whose marks have the given claim bit and per-role weights (R, Am).

        `mark_weights` is the marks' prior and `like` their likelihood, both linear and
        broadcast to (R, Am); `cloud` and `ref` are the marks' cloud weight and
        reference density, linear, (Am,). `log_mass0` and `like0`, the time-zero
        mass and likelihood, default to `log_mass` and `like`.
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
        log_mass0 = log_mass if log_mass0 is None else log_mass0
        log_mass0 = jnp.broadcast_to(jnp.asarray(log_mass0, float), (n_roles,))
        log_prior = jnp.log(jnp.asarray(mark_weights, float))
        log_like = jnp.broadcast_to(jnp.log(jnp.asarray(like, float)), log_prior.shape)
        log_like0 = log_like if like0 is None else jnp.log(jnp.asarray(like0, float))
        log_like0 = jnp.broadcast_to(log_like0, log_prior.shape)
        return Pool(marks, log_prior, log_like, log_mass, log_mass0, log_like0)

    @classmethod
    def empty_pool(cls, n_roles=1):
        """A pool with no expected planets."""
        return cls.pool([0.0], jnp.ones((n_roles, 1)), -jnp.inf, n_roles)

    @classmethod
    def belief(cls, atoms, pool=None, n_roles=1):
        """A belief with flat priors and no likelihood."""
        n_draw = atoms.cloud.log_weight.shape[0]
        flat = jnp.zeros((n_roles, n_draw))
        return Belief(
            atoms, flat, flat, cls.empty_pool(n_roles) if pool is None else pool
        )


@pytest.fixture(scope="session")
def toy():
    return _Toy


@pytest.fixture(scope="session")
def typecheck():
    """Whether the jaxtyping import hook is on."""
    return TYPECHECK
