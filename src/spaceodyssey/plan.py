"""Plan-level contracts for certificates that integrate over a batch of planets.

A profile certificate scores what an observing plan would let it certify,
planet by planet, without drawing a mission. Its input is a
:class:`PlanSummary` (what the plan delivers for each planet) paired with the
truth planets in a :class:`PlanPredictive`; its output is a
:class:`CertificationOutcome` (per-planet certification probabilities). Every
per-planet array carries one canonical leading batch axis P of planets;
wavelengths may also be shared across planets.

Shape and dtype checks run at construction, including during tracing;
value checks run on concrete inputs. A tier outside its validated range reports it
through :attr:`CertificationOutcome.in_range`, which the host-side
:meth:`CertificationOutcome.raise_if_out_of_range` turns into an exception.
"""

from typing import Any, ClassVar, final

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np

from spaceodyssey.errors import OutOfRangeError


def _as_bool(value) -> jax.Array:
    return jnp.asarray(value, dtype=bool)


CONTRACT_VERSION = 3


def _optional_array(value):
    return None if value is None else jnp.asarray(value)


def _check_values(valid, message):
    # Value checks are host-side only; static shape/dtype checks also run in jit.
    if not isinstance(valid, jax.core.Tracer) and not bool(np.all(valid)):
        raise ValueError(message)


@final
class PlanSummary(eqx.Module):
    """Physical data delivered by a plan, with canonical planet axis P.

    Noise is the standard deviation of measured spectral flux divided by
    a fixed reference planet flux in each bin (dimensionless fractional
    planet-flux units, not planet-to-star contrast). The normalization must
    also be applied to any signal model and correlated factors. It is not
    a noise estimate divided by a noisy measured flux.

    Version 2 removes ``snr``: consumers choose their signal model and SNR
    convention. For the toy's unit signal, migrate positive v1 SNR using
    ``noise_sigma = 1 / snr``. Missing data use ``spectrum_visit = -1``.
    ``mass_measured`` is derived from the mass precision.
    Version 3 renames ``rv_precision_ms`` to ``k_precision_ms`` and adds
    ``k_precision_assumed_ms`` to distinguish sampling and reported precision.

    Attributes:
        bin_wavelength_nm: Bin centres in nm, (B,) or (P, B).
        noise_sigma: Finite positive independent noise sigma, (P, B), in
            fractional planet-flux units. Total covariance per planet is
            diag(noise_sigma**2) + F F^T.
        noise_corr_factors: Optional finite F, (P, B, Kc), in the same
            units as noise_sigma. None or Kc = 0 denotes white noise.
        noise_known: Whether the noise covariance is known, bool (P,).
        spectral_resolution: Resolving power lambda / delta-lambda, (P,).
        visit_epochs_d: Mission epochs in days, (P, V).
        visit_success: Usable-data mask, bool (P, V).
        visit_photometric_snr: Nonnegative photometric SNR, (P, V).
        spectrum_visit: Integer visit index, (P,); -1 means no spectrum.
            A spectrum at a failed visit is unusable.
        astrometric_precision_mas: One-sigma precision in mas, (P, V).
        flux_reference: Static name of the reference flux; empty means unspecified.
        k_precision_ms: True one-sigma sampling precision of the
            semi-amplitude (K) estimate from the plan's RV analysis, including
            all RV noise, in m/s, (P,). Real and positive; inf means no mass
            measurement.
        k_precision_assumed_ms: Optional real reported K precision in m/s, (P,).
            None means the analysis reports k_precision_ms. Positive, with
            inf exactly where k_precision_ms is inf. Either precision may
            exceed the other.

    Shapes and index dtype are checked even during tracing. Value checks
    run on concrete construction; validate inputs before passing them to jit.
    """

    CONTRACT_VERSION: ClassVar[int] = CONTRACT_VERSION

    bin_wavelength_nm: jax.Array = eqx.field(converter=jnp.asarray)
    noise_sigma: jax.Array = eqx.field(converter=jnp.asarray)
    noise_known: jax.Array = eqx.field(converter=_as_bool)
    spectral_resolution: jax.Array = eqx.field(converter=jnp.asarray)
    visit_epochs_d: jax.Array = eqx.field(converter=jnp.asarray)
    visit_success: jax.Array = eqx.field(converter=_as_bool)
    visit_photometric_snr: jax.Array = eqx.field(converter=jnp.asarray)
    spectrum_visit: jax.Array = eqx.field(converter=jnp.asarray)
    astrometric_precision_mas: jax.Array = eqx.field(converter=jnp.asarray)
    k_precision_ms: jax.Array = eqx.field(converter=jnp.asarray)
    noise_corr_factors: jax.Array | None = eqx.field(
        default=None, converter=_optional_array
    )

    flux_reference: str = eqx.field(default="", static=True)
    k_precision_assumed_ms: jax.Array | None = eqx.field(
        default=None, converter=_optional_array
    )

    def __check_init__(self):
        """Validate planet, spectral, and visit axes and concrete values."""
        if self.noise_sigma.ndim != 2:
            raise ValueError(
                f"noise_sigma must have shape (P, B), got {self.noise_sigma.shape}"
            )
        p, b = self.noise_sigma.shape
        if self.bin_wavelength_nm.shape not in ((b,), (p, b)):
            raise ValueError("bin_wavelength_nm must have shape (B,) or (P, B)")
        if self.visit_epochs_d.ndim != 2 or self.visit_epochs_d.shape[0] != p:
            raise ValueError(f"visit_epochs_d must have shape ({p}, V)")
        visits = self.visit_epochs_d.shape
        for name in (
            "visit_success",
            "astrometric_precision_mas",
            "visit_photometric_snr",
        ):
            shape = getattr(self, name).shape
            if shape != visits:
                raise ValueError(f"{name} must have shape {visits}, got {shape}")
        for name in (
            "spectral_resolution",
            "k_precision_ms",
            "noise_known",
            "spectrum_visit",
        ):
            shape = getattr(self, name).shape
            if shape != (p,):
                raise ValueError(f"{name} must have shape ({p},), got {shape}")
        for name in ("k_precision_ms", "k_precision_assumed_ms"):
            value = getattr(self, name)
            if value is not None and jnp.issubdtype(value.dtype, jnp.complexfloating):
                raise ValueError(f"{name} must have a real dtype")
        assumed = self.k_precision_assumed_ms
        if assumed is not None:
            if assumed.shape != (p,):
                raise ValueError(
                    f"k_precision_assumed_ms must have shape ({p},), "
                    f"got {assumed.shape}"
                )
            _check_values(
                assumed > 0, "k_precision_assumed_ms must be > 0 (inf allowed)"
            )
            _check_values(
                jnp.isinf(assumed) == jnp.isinf(self.k_precision_ms),
                "k_precision_assumed_ms must be inf exactly where "
                "k_precision_ms is inf",
            )
        if not jnp.issubdtype(self.spectrum_visit.dtype, jnp.integer):
            raise ValueError("spectrum_visit must have integer dtype")
        f = self.noise_corr_factors
        if f is not None:
            if f.ndim != 3 or f.shape[:2] != (p, b):
                raise ValueError("noise_corr_factors must have shape (P, B, Kc)")
            _check_values(jnp.isfinite(f), "noise_corr_factors must be finite")
        _check_values(
            jnp.isfinite(self.noise_sigma) & (self.noise_sigma > 0),
            "noise_sigma must be finite and > 0",
        )
        _check_values(
            self.k_precision_ms > 0,
            "k_precision_ms must be > 0 (inf allowed)",
        )
        _check_values(
            (self.spectrum_visit >= -1) & (self.spectrum_visit < visits[1]),
            "spectrum_visit must be in [-1, V-1]",
        )
        _check_values(
            self.visit_photometric_snr >= 0,
            "visit_photometric_snr must be >= 0",
        )

    @property
    def mass_measured(self) -> jax.Array:
        """Whether a finite K sampling precision supplies a mass measurement, (P,)."""
        return jnp.isfinite(self.k_precision_ms)

    @property
    def k_precision_for_certification_ms(self) -> jax.Array:
        """Reported K precision for a certifier, (P,), in m/s.

        Fall back to the true sampling precision when no assumed precision
        is supplied. Simulation uses k_precision_ms directly.
        """
        if self.k_precision_assumed_ms is None:
            return self.k_precision_ms
        return self.k_precision_assumed_ms

    @property
    def n_planets(self) -> int:
        """The batch size P."""
        return self.noise_sigma.shape[0]


@final
class PlanPredictive(eqx.Module):
    """The data distribution a plan induces on a batch of truth planets.

    The integrate operator evaluates a certificate at fixed truth over the
    noise realizations of the plan's data, so the truth enters here and only
    here. A certificate's decision rule never reads ``truth``; it is used to
    define the data distribution and to score whether a verdict is wrong.

    Attributes:
        plan: The plan summary for the batch.
        truth: The truth planets as a pytree whose array leaves carry the
            leading planet axis P; its structure is set by the certificate
            that consumes it.
    """

    plan: PlanSummary
    truth: Any

    def __check_init__(self):
        """Validate that every truth leaf shares the plan's planet axis."""
        n_planets = self.plan.n_planets
        for leaf in jax.tree_util.tree_leaves(self.truth):
            shape = jnp.shape(leaf)
            if not shape or shape[0] != n_planets:
                raise ValueError(
                    f"truth leaves must lead with the planet axis ({n_planets}), "
                    f"got shape {shape}"
                )


@final
class CertificationOutcome(eqx.Module):
    """Per-planet certification probabilities at fixed truth under a plan.

    The counted direction is the one a yield counts (positive for a
    candidate profile). ``p_wrong`` is the probability of a certification in
    the counted direction whose certified claims are false for the truth
    planet; ``p_neg_wrong`` is the probability of a certified negative
    verdict that is false for it. Closed-form tiers report zero standard
    errors.

    Attributes:
        p_pos: Probability of certification in the counted (positive)
            direction, shape (P,).
        p_neg: Probability of a certified negative verdict, shape (P,).
        p_wrong: Probability of a counted-direction certification that is
            wrong about the truth, shape (P,).
        p_neg_wrong: Probability of a certified negative verdict that is
            wrong about the truth, shape (P,).
        p_pos_se: Monte Carlo standard error of ``p_pos``, shape (P,).
        p_neg_se: Monte Carlo standard error of ``p_neg``, shape (P,).
        p_wrong_se: Monte Carlo standard error of ``p_wrong``, shape (P,).
        p_neg_wrong_se: Monte Carlo standard error of ``p_neg_wrong``,
            shape (P,).
        truth_positive: True where the truth planet satisfies the counted
            claims (a truth candidate), shape (P,).
        in_range: True where the evaluating tier is inside its validated
            range; values outside it are returned but not trusted, shape (P,).
        p_pos_rep: Optional probability estimates, shape (R, P), R >= 2.
        p_neg_rep: Negative probability replicates, shape (R, P).
        p_wrong_rep: Wrong positive probability replicates, shape (R, P).
        p_neg_wrong_rep: Wrong negative probability replicates, shape (R, P).
        replicate_kind: Static batch or bootstrap label. All four replicate
            arrays and both metadata fields must be supplied together.
        claim_p_pos: Stand-alone claim positive probabilities, shape (P, K).
        claim_p_neg: Stand-alone claim negative probabilities, shape (P, K).
        claim_p_wrong: Wrong positive probabilities, shape (P, K).
        claim_p_neg_wrong: Wrong negative probabilities, shape (P, K).
        claim_truth_positive: Claim truth flags, shape (P, K), boolean.
        claim_in_range: Claim validity flags, shape (P, K), boolean.
        claim_names: Ordered names; supplied with all six claim arrays.
        fidelity: Optional free-form method label, such as closed_form or quadrature;
            metadata only, never ordered, as in SeamInfo.
        profile_hash: Optional 64-character lowercase hexadecimal profile content hash.
        replicate_key: Static identifier of the replicate stream. Batch main
            probabilities must match replicate means to rtol 1e-10.
    """

    p_pos: jax.Array = eqx.field(converter=jnp.asarray)
    p_neg: jax.Array = eqx.field(converter=jnp.asarray)
    p_wrong: jax.Array = eqx.field(converter=jnp.asarray)
    p_neg_wrong: jax.Array = eqx.field(converter=jnp.asarray)
    p_pos_se: jax.Array = eqx.field(converter=jnp.asarray)
    p_neg_se: jax.Array = eqx.field(converter=jnp.asarray)
    p_wrong_se: jax.Array = eqx.field(converter=jnp.asarray)
    p_neg_wrong_se: jax.Array = eqx.field(converter=jnp.asarray)
    truth_positive: jax.Array = eqx.field(converter=_as_bool)
    in_range: jax.Array = eqx.field(converter=_as_bool)

    p_pos_rep: jax.Array | None = eqx.field(default=None, converter=_optional_array)
    p_neg_rep: jax.Array | None = eqx.field(default=None, converter=_optional_array)
    p_wrong_rep: jax.Array | None = eqx.field(default=None, converter=_optional_array)
    p_neg_wrong_rep: jax.Array | None = eqx.field(
        default=None, converter=_optional_array
    )
    replicate_kind: str | None = eqx.field(default=None, static=True)
    replicate_key: str | None = eqx.field(default=None, static=True)

    claim_p_pos: jax.Array | None = eqx.field(default=None, converter=_optional_array)
    claim_p_neg: jax.Array | None = eqx.field(default=None, converter=_optional_array)
    claim_p_wrong: jax.Array | None = eqx.field(default=None, converter=_optional_array)
    claim_p_neg_wrong: jax.Array | None = eqx.field(
        default=None, converter=_optional_array
    )
    claim_truth_positive: jax.Array | None = eqx.field(
        default=None, converter=_optional_array
    )
    claim_in_range: jax.Array | None = eqx.field(
        default=None, converter=_optional_array
    )
    claim_names: tuple[str, ...] | None = eqx.field(default=None, static=True)
    profile_hash: str | None = eqx.field(default=None, static=True)
    fidelity: str | None = eqx.field(default=None, static=True)

    def __check_init__(self):
        """Validate planet axes, optional claims, hash, and replicates."""
        shape = self.p_pos.shape
        if len(shape) != 1:
            raise ValueError(f"p_pos must have shape (P,), got {shape}")
        for name in (
            "p_neg",
            "p_wrong",
            "p_neg_wrong",
            "p_pos_se",
            "p_neg_se",
            "p_wrong_se",
            "p_neg_wrong_se",
            "truth_positive",
            "in_range",
        ):
            other = getattr(self, name).shape
            if other != shape:
                raise ValueError(f"{name} must have shape {shape}, got {other}")

        if self.profile_hash is not None:
            if (
                not isinstance(self.profile_hash, str)
                or len(self.profile_hash) != 64
                or any(c not in "0123456789abcdef" for c in self.profile_hash)
            ):
                raise ValueError(
                    "profile_hash must be 64 lowercase hexadecimal characters"
                )
        claim_fields = (
            "claim_p_pos",
            "claim_p_neg",
            "claim_p_wrong",
            "claim_p_neg_wrong",
            "claim_truth_positive",
            "claim_in_range",
        )
        present_claims = [getattr(self, name) is not None for name in claim_fields]
        present_claims.append(self.claim_names is not None)
        if any(present_claims):
            if not all(present_claims):
                raise ValueError("claim fields must be all or none")
            if (
                not isinstance(self.claim_names, tuple)
                or not self.claim_names
                or any(not isinstance(n, str) or not n for n in self.claim_names)
                or len(set(self.claim_names)) != len(self.claim_names)
            ):
                raise ValueError("claim_names must be a nonempty tuple of unique names")
            claim_shape = (shape[0], len(self.claim_names))
            for name in claim_fields:
                value = getattr(self, name)
                if value.shape != claim_shape:
                    raise ValueError(f"{name} must have shape {claim_shape}")
                if name in ("claim_truth_positive", "claim_in_range"):
                    if value.dtype != jnp.bool_:
                        raise ValueError(f"{name} must be boolean")
                else:
                    _check_values(
                        jnp.isfinite(value) & (value >= 0) & (value <= 1),
                        f"{name} probabilities must be finite and in [0, 1]",
                    )
            _check_values(
                self.claim_p_pos + self.claim_p_neg <= 1,
                "claim positive and negative probabilities must sum to at most one",
            )
            _check_values(
                self.claim_p_wrong <= self.claim_p_pos,
                "claim_p_wrong exceeds claim_p_pos",
            )
            _check_values(
                self.claim_p_neg_wrong <= self.claim_p_neg,
                "claim_p_neg_wrong exceeds claim_p_neg",
            )

        names = ("p_pos", "p_neg", "p_wrong", "p_neg_wrong")
        reps = [getattr(self, name + "_rep") for name in names]
        present = [rep is not None for rep in reps]
        if not any(present):
            if self.replicate_kind is not None or self.replicate_key is not None:
                raise ValueError("replicate metadata requires all four arrays")
            return
        if not all(present):
            raise ValueError("replicate arrays must be all four or none")
        if self.replicate_kind not in ("batch", "bootstrap"):
            raise ValueError("replicate_kind must be batch or bootstrap")
        if not isinstance(self.replicate_key, str) or not self.replicate_key:
            raise ValueError("replicate_key must be a nonempty string")
        rep_shape = reps[0].shape
        if len(rep_shape) != 2 or rep_shape[0] < 2 or rep_shape[1:] != shape:
            raise ValueError("replicate arrays must have shape (R, P) with R >= 2")
        for name, rep in zip(names, reps, strict=True):
            if rep.shape != rep_shape:
                raise ValueError(f"{name}_rep must have shape {rep_shape}")
            _check_values(
                jnp.isfinite(rep) & (rep >= 0) & (rep <= 1),
                f"{name}_rep probabilities must be finite and in [0, 1]",
            )
            if self.replicate_kind == "batch":
                _check_values(
                    jnp.isclose(
                        getattr(self, name), rep.mean(axis=0), rtol=1e-10, atol=0
                    ),
                    f"{name} must equal its batch replicate mean",
                )

    def raise_if_out_of_range(self) -> None:
        """Raise when any planet lies outside the tier's validated range.

        Host-side only; never call it under ``jit``.

        Raises:
            OutOfRangeError: Naming the out-of-range planet indices.
        """
        bad = np.flatnonzero(~np.asarray(self.in_range))
        if bad.size:
            raise OutOfRangeError(
                f"{bad.size} of {self.in_range.shape[0]} planets lie outside "
                f"the certificate's validated range: indices {bad.tolist()}"
            )
