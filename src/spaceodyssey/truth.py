"""Truth parameters, model priors, and simulated plan observations."""

import abc
import hashlib
import json

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np
from photomancy.priors import AbstractPrior

from spaceodyssey.errors import UnsupportedProfile
from spaceodyssey.plan import PlanSummary, _as_bool, _check_values
from spaceodyssey.profile import Profile


class Truth(eqx.Module):
    """Parameters (P, D), known context (P,), and a static model identifier.

    Context keys must be nonempty strings. Required keys and D are checked
    by the consuming model's :meth:`AbstractTruthModel.validate_truth`.
    """

    theta: jax.Array = eqx.field(converter=jnp.asarray)
    context: dict[str, jax.Array] = eqx.field(
        converter=lambda values: {k: jnp.asarray(v) for k, v in values.items()}
    )
    model_id: str = eqx.field(static=True)

    def __check_init__(self):
        """Check parameter rank and context keys and planet axes."""
        if self.theta.ndim != 2:
            raise ValueError("theta must have shape (P, D)")
        for name, value in self.context.items():
            if not isinstance(name, str) or not name:
                raise ValueError("context keys must be nonempty strings")
            if value.shape != self.theta.shape[:1]:
                raise ValueError(f"context {name!r} must have shape (P,)")


class ObservedData(eqx.Module):
    """Plan measurements with zero-filled missing data and boolean masks.

    Spectrum (P, B) and photometry (P, V) use the plan's flux reference.
    Astrometry is (P, V, 2) in mas, RV is (P,) in m/s. The masks have
    shapes (P,) and (P, V); RV is ignored at infinite plan precision.
    """

    spectrum: jax.Array = eqx.field(converter=jnp.asarray)
    has_spectrum: jax.Array = eqx.field(converter=_as_bool)
    photometry: jax.Array = eqx.field(converter=jnp.asarray)
    astrometry_mas: jax.Array = eqx.field(converter=jnp.asarray)
    visit_usable: jax.Array = eqx.field(converter=_as_bool)
    rv_semi_amplitude_ms: jax.Array = eqx.field(converter=jnp.asarray)

    def __check_init__(self):
        """Validate the shared planet and visit axes."""
        if self.spectrum.ndim != 2:
            raise ValueError("spectrum must have shape (P, B)")
        p = self.spectrum.shape[0]
        if self.photometry.ndim != 2 or self.photometry.shape[0] != p:
            raise ValueError("photometry must have shape (P, V)")
        shapes = {
            "has_spectrum": (p,),
            "visit_usable": self.photometry.shape,
            "astrometry_mas": (*self.photometry.shape, 2),
            "rv_semi_amplitude_ms": (p,),
        }
        for name, shape in shapes.items():
            if getattr(self, name).shape != shape:
                raise ValueError(f"{name} must have shape {shape}")


class PriorSet(eqx.Module):
    """Named prior distributions in one parameterization.

    Labels uniquely identify members, including the nominal member. Change
    the version of ``id`` whenever a member distribution changes.
    """

    members: tuple[AbstractPrior, ...]
    id: str = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    nominal_index: int = eqx.field(static=True)

    def __check_init__(self):
        """Validate labels, dimensions, and the nominal member index."""
        if not isinstance(self.members, tuple) or not self.members:
            raise ValueError("members must be a nonempty tuple")
        if not isinstance(self.labels, tuple) or len(self.labels) != len(self.members):
            raise ValueError("labels must be a tuple with one label per member")
        if any(not isinstance(s, str) or not s for s in self.labels):
            raise ValueError("labels must be nonempty strings")
        if len(set(self.labels)) != len(self.labels):
            raise ValueError("labels must be unique")
        if not isinstance(self.id, str) or not self.id:
            raise ValueError("prior set id must be a nonempty string")
        if type(self.nominal_index) is not int or not (
            0 <= self.nominal_index < len(self.members)
        ):
            raise ValueError("nominal_index must index a member")
        if any(p.ndim != self.members[0].ndim for p in self.members):
            raise ValueError("prior set members must have equal ndim")

    def content_hash(self) -> str:
        """Return the SHA-256 digest of canonical, versioned prior-set JSON.

        Member order, types, pytree structure (including static fields), and
        leaf dtypes, shapes, and values are content. Floats use hexadecimal
        strings with signed zero normalized. This host-side operation is
        independent of Python's randomized in-process hash.
        """
        members = []
        for member in self.members:
            leaves = []
            for leaf in jax.tree_util.tree_leaves(member):
                array = np.asarray(leaf)
                values = array.reshape(-1).tolist()
                if jnp.issubdtype(array.dtype, jnp.floating):
                    values = [(float(value) + 0.0).hex() for value in values]
                leaves.append(
                    {"dtype": str(array.dtype), "shape": array.shape, "values": values}
                )
            members.append(
                {
                    "type": f"{type(member).__module__}.{type(member).__qualname__}",
                    "treedef": str(jax.tree_util.tree_structure(member)),
                    "leaves": leaves,
                }
            )
        content = {
            "schema": "spaceodyssey.PriorSet/1",
            "id": self.id,
            "labels": self.labels,
            "nominal_index": self.nominal_index,
            "members": members,
        }
        encoded = json.dumps(content, sort_keys=True, separators=(",", ":")).encode(
            "utf-8"
        )
        return hashlib.sha256(encoded).hexdigest()


def _plan_masks(plan):
    if plan.visit_success.shape[1] == 0:
        return jnp.zeros(plan.n_planets, dtype=bool), plan.visit_success
    index = jnp.maximum(plan.spectrum_visit, 0)
    has_spectrum = (plan.spectrum_visit >= 0) & jnp.take_along_axis(
        plan.visit_success, index[:, None], axis=1
    )[:, 0]
    return has_spectrum, plan.visit_success


class AbstractTruthModel(eqx.Module):
    """A parameterization, its quantities and priors, and its forward model."""

    model_id: eqx.AbstractClassVar[str]
    param_names: eqx.AbstractClassVar[tuple[str, ...]]
    param_units: eqx.AbstractClassVar[tuple[str, ...]]
    context_names: eqx.AbstractClassVar[tuple[str, ...]]
    quantity_names: eqx.AbstractClassVar[tuple[str, ...]]
    flux_reference: eqx.AbstractClassVar[str]
    prior: eqx.AbstractVar[AbstractPrior]
    prior_sets: eqx.AbstractVar[tuple[PriorSet, ...]]

    def __check_init__(self):
        """Validate prior dimensions and nominal members."""
        d = len(self.param_names)
        if len(self.param_units) != d or self.prior.ndim != d:
            raise ValueError("param_names, param_units and prior.ndim must agree on D")
        if len({ps.id for ps in self.prior_sets}) != len(self.prior_sets):
            raise ValueError("prior set ids must be unique")
        for ps in self.prior_sets:
            if any(member.ndim != d for member in ps.members):
                raise ValueError(f"prior set {ps.id!r} has wrong D")
            _check_values(
                eqx.tree_equal(ps.members[ps.nominal_index], self.prior),
                f"prior set {ps.id!r} nominal member differs from prior",
            )

    def validate_truth(self, truth: Truth, plan: PlanSummary | None = None) -> None:
        """Check model identity, parameter dimension, context, and plan axis."""
        if truth.model_id != self.model_id:
            raise ValueError(f"truth model_id must be {self.model_id!r}")
        if truth.theta.shape[1] != len(self.param_names):
            raise ValueError(f"theta has wrong D; expected {len(self.param_names)}")
        missing = set(self.context_names) - truth.context.keys()
        if missing:
            raise ValueError(f"missing context: {sorted(missing)}")
        if plan is not None:
            if truth.theta.shape[0] != plan.n_planets:
                raise ValueError("truth and plan must share the planet axis P")
            _check_flux_reference(self, plan)

    @abc.abstractmethod
    def quantities(self, theta, context) -> dict[str, jax.Array]:
        """Return named scalar quantities, each with shape theta.shape[:-1]."""

    @abc.abstractmethod
    def expected_data(self, truth: Truth, plan: PlanSummary) -> ObservedData:
        """Validate truth and return noise-free, masked plan measurements."""

    def prior_set(self, prior_set_id: str) -> PriorSet:
        """Return a declared prior set or raise UnsupportedProfile naming it."""
        for ps in self.prior_sets:
            if ps.id == prior_set_id:
                return ps
        raise UnsupportedProfile(f"unknown prior set: {prior_set_id!r}")

    def simulate(self, key, truth: Truth, plan: PlanSummary) -> ObservedData:
        """Draw independent Gaussian measurement noise with the plan covariance.

        Args:
            key: One PRNG key for this batch, or P per-planet keys. For stable
                planet and realization identities, supply per-planet keys from
                ``spaceodyssey.rng.stream(root, planet_id, realization_id,
                Purpose.MEASUREMENT)``. A single key uses batch row indices.
            truth: Parameters and known conditions.
            plan: Noise scales, covariance factors, and availability masks.

        Returns:
            Masked observations. Spectral covariance is diag(sigma**2) + F F^T.
            Photometric sigma is abs(expected flux) / SNR; zero SNR contributes
            zero photometry. RV with infinite precision contributes zero.
        """
        if str(jr.key_impl(key)) != "threefry2x32":
            raise ValueError("simulate requires threefry2x32 PRNG keys")
        self.validate_truth(truth, plan)
        mean = self.expected_data(truth, plan)
        has, usable = _plan_masks(plan)
        key_shape = jr.key_data(key).shape
        if key_shape == (2,):
            keys = jax.vmap(lambda i: jr.fold_in(key, i))(jnp.arange(plan.n_planets))
        elif key_shape == (plan.n_planets, 2):
            keys = key
        else:
            raise ValueError("key must be one PRNG key or P per-planet keys")

        def noise(channel, shape):
            return jax.vmap(lambda k: jr.normal(jr.fold_in(k, channel), shape))(keys)

        spectral_noise = plan.noise_sigma * noise(0, mean.spectrum.shape[1:])
        if plan.noise_corr_factors is not None:
            factors = plan.noise_corr_factors
            spectral_noise += jnp.einsum(
                "pbk,pk->pb", factors, noise(1, (factors.shape[-1],))
            )
        snr = plan.visit_photometric_snr
        phot_mask = usable & (snr > 0)
        phot_sigma = jnp.abs(mean.photometry) / jnp.where(phot_mask, snr, 1)
        ast_sigma = jnp.where(usable, plan.astrometric_precision_mas, 0)
        rv_sigma = jnp.where(plan.mass_measured, plan.k_precision_ms, 0)
        return ObservedData(
            spectrum=jnp.where(has[:, None], mean.spectrum + spectral_noise, 0),
            has_spectrum=has,
            photometry=jnp.where(
                phot_mask,
                mean.photometry + phot_sigma * noise(2, mean.photometry.shape[1:]),
                0,
            ),
            astrometry_mas=jnp.where(
                usable[..., None],
                mean.astrometry_mas
                + ast_sigma[..., None] * noise(3, mean.astrometry_mas.shape[1:]),
                0,
            ),
            visit_usable=usable,
            rv_semi_amplitude_ms=jnp.where(
                plan.mass_measured,
                mean.rv_semi_amplitude_ms + rv_sigma * noise(4, ()),
                0,
            ),
        )


def _check_flux_reference(truth_model, plan):
    if (
        plan.flux_reference
        and truth_model.flux_reference
        and plan.flux_reference != truth_model.flux_reference
    ):
        raise UnsupportedProfile(
            f"flux reference mismatch: plan {plan.flux_reference!r}, "
            f"model {truth_model.flux_reference!r}"
        )


def check_compatible(
    profile: Profile, truth_model: AbstractTruthModel, plan: PlanSummary | None = None
) -> None:
    """Raise UnsupportedProfile naming missing quantities, priors, or flux units."""
    missing = profile.quantities - set(truth_model.quantity_names)
    if missing:
        raise UnsupportedProfile(f"missing quantities: {sorted(missing)}")
    truth_model.prior_set(profile.prior_set_id)
    if plan is not None:
        _check_flux_reference(truth_model, plan)
