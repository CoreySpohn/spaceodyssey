"""Leaf types shared by every layer: cloud, atoms, pool, row, action and grant.

This module imports nothing from the package. Axis names used in the annotations:
``A`` draws, ``K`` planets one draw can hold, ``G`` gases, ``R`` roles (forecast is
row 0, the verdict members are rows 1 to ``R - 1``), ``W`` log row capacity and ``Am``
the atoms of a pool.
"""

import equinox as eqx
import jax
import jax.numpy as jnp
from jaxtyping import Array, Bool, Float, Int, PyTree


def _asarrays(tree):
    """Convert every leaf of ``tree`` to a JAX array."""
    return jax.tree.map(jnp.asarray, tree)


class WeightedCloud(eqx.Module):
    """Weighted draws from one refit; a stand-in mirroring the refit cloud's fields.

    ``samples`` is a pytree whose leaves share the leading draw axis ``A``.

    Weight contract: ``log_weight`` is the log of target over proposal, where the
    target is ``p0`` times the rows in ``row_mask``. It is not a log ratio against a
    proposal that equals ``p0``: that reading would count ``-log p0`` twice when the
    role weights are formed. One cloud has one row set, so ``row_mask`` is constant
    across its draws unless a block normalization is defined. The cloud comes from one
    refit.

    Attributes:
        samples: the state pytree, leading axis ``A``.
        log_weight: log of target over proposal, shape ``(A,)``.
        log_ref_density: log of ``p0`` on ``measure``, shape ``(A,)``.
        measure: the coordinates in which densities are taken, a static label.
    """

    samples: PyTree = eqx.field(converter=_asarrays)
    log_weight: Float[Array, " A"] = eqx.field(converter=jnp.asarray)
    log_ref_density: Float[Array, " A"] = eqx.field(converter=jnp.asarray)
    measure: str = eqx.field(static=True)


class Atoms[StateT](eqx.Module):
    """A weighted cloud with the gas block and the rows each draw already represents.

    ``cloud.samples`` carries the state of type ``StateT``. The gas block holds, per
    draw, planet and gas, a Gaussian over the abundance: ``x_mean`` and ``x_var`` (zero
    variance is a point state).

    Attributes:
        cloud: the weighted draws.
        component: gas label into the shared Gaussian dictionary; unused until a role
            has more than one component.
        x_mean: mean abundance, shape ``(A, K, G)``.
        x_var: abundance variance, shape ``(A, K, G)``, zero for a point state.
        row_mask: rows this draw already represents, shape ``(A, W)``; all false here.
    """

    cloud: WeightedCloud
    component: Int[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    x_mean: Float[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    x_var: Float[Array, "A K G"] = eqx.field(converter=jnp.asarray)
    row_mask: Bool[Array, "A W"] = eqx.field(converter=jnp.asarray)


class Pool(eqx.Module):
    """The undetected planets at a star, per role.

    Each mark is one planet (``K = 1`` in ``marks``); the marks carry the conditional
    distribution of an undetected planet and the pool carries how many there are in
    expectation.

    Attributes:
        marks: the marks, leading axis ``Am``.
        log_prior: log prior of each mark under each role, shape ``(R, Am)``.
        log_like: log likelihood of each mark under each role, shape ``(R, Am)``.
        log_mass: log expected number of planets, per role, shape ``(R,)``.
        log_mass0: the same at time zero, shape ``(R,)``.
    """

    marks: Atoms
    log_prior: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)
    log_like: Float[Array, "R Am"] = eqx.field(converter=jnp.asarray)
    log_mass: Float[Array, " R"] = eqx.field(converter=jnp.asarray)
    log_mass0: Float[Array, " R"] = eqx.field(converter=jnp.asarray)


class Row(eqx.Module):
    """One released observation result.

    Attributes:
        star: star index.
        mode: observing mode index.
        start_d: start time in days.
        exposure_d: exposure length in days.
        charged_d: time charged against the budget in days.
        release_d: time at which the row is released, in days.
        refusal: refusal code, zero for an accepted row.
        law: index of the law that inverts the payload.
        payload: one zero-padded array per law; only the array of ``law`` is meaningful.
    """

    star: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    mode: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    start_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    exposure_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    charged_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    release_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)
    refusal: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    law: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    payload: tuple[Array, ...] = eqx.field(
        converter=lambda p: tuple(map(jnp.asarray, p))
    )


class Action(eqx.Module):
    """What a policy asks for: one pytree with an integer kind.

    Attributes:
        kind: integer action kind.
        star: star index.
        mode: observing mode index.
        exposure_d: requested exposure in days.
    """

    kind: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    star: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    mode: Int[Array, ""] = eqx.field(converter=jnp.asarray)
    exposure_d: Float[Array, ""] = eqx.field(converter=jnp.asarray)


class Grant(eqx.Module):
    """Rows delivered at time zero, stacked along a leading axis.

    Attributes:
        rows: a ``Row`` whose leaves have a leading axis over the granted rows.
    """

    rows: Row
