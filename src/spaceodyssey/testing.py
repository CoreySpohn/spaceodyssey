"""Conformance suites that ship with the seams, not the implementations.

Every registry entry, in-tree or external, is expected to instantiate its
seam's suite in its own tests. This is what keeps an unordered, open registry
trustworthy: declared operators must be honest (an undeclared operator raises
:class:`spaceodyssey.errors.UnsupportedOperator`, a declared one works), sampled
moments must match declared moments, and outputs must match declared contract
types. Checks raise plain AssertionError so any test framework can host them.
"""

import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.contracts import resolve_contract
from spaceodyssey.errors import UnsupportedOperator


def check_predictive(pred, *, key, n: int = 4096) -> None:
    """Check a predictive distribution's sampling and moment honesty.

    Draws ``n`` samples, verifies they and their log-density are finite, and,
    when the predictive exposes moments, verifies the sample mean lies within
    six standard errors of the declared mean.

    Args:
        pred: The predictive to check.
        key: PRNG key for the draws.
        n: Number of samples.

    Raises:
        AssertionError: On any conformance violation.
    """
    samples = jax.vmap(pred.sample)(jr.split(key, n))
    if not bool(jnp.all(jnp.isfinite(samples))):
        raise AssertionError("predictive produced non-finite samples")
    log_prob = pred.log_prob(samples[0])
    if not bool(jnp.isfinite(log_prob)):
        raise AssertionError("log_prob of a drawn sample is not finite")
    try:
        mean, cov = pred.moments()
    except UnsupportedOperator:
        return
    standard_error = jnp.sqrt(jnp.clip(jnp.diag(cov), min=0.0) / n)
    tolerance = float(6.0 * jnp.max(standard_error)) + 1e-12
    np.testing.assert_allclose(
        np.asarray(samples.mean(axis=0)),
        np.asarray(mean),
        atol=tolerance,
        rtol=0.0,
        err_msg="sample mean disagrees with declared moments",
    )


def check_certificate(cert, info, datasets, *, moments=None, predictive=None):
    """Check a certificate implementation against its declared operators.

    Args:
        cert: The certificate implementation.
        info: Its registered SeamInfo (source of the declared operators).
        datasets: Realized datasets to fold in when "sample" is declared.
        moments: Predictive moments to propagate when "propagate" is declared.
        predictive: Predictive to integrate when "integrate" is declared.

    Returns:
        The final certificate state after any declared-sample updates.

    Raises:
        AssertionError: When a declared operator raises UnsupportedOperator
            or an undeclared one fails to.
    """
    state = cert.initial()
    if "sample" in info.operators:
        for dataset in datasets:
            try:
                state = cert.update(state, dataset)
            except UnsupportedOperator as err:
                raise AssertionError(
                    "certificate declares 'sample' but update() raised"
                ) from err
    else:
        _expect_unsupported(lambda: cert.update(state, datasets[0]), "update")
    if "propagate" in info.operators:
        try:
            cert.propagate(cert.initial(), moments)
        except UnsupportedOperator as err:
            raise AssertionError(
                "certificate declares 'propagate' but propagate() raised"
            ) from err
    else:
        _expect_unsupported(
            lambda: cert.propagate(cert.initial(), moments), "propagate"
        )
    if "integrate" in info.operators:
        try:
            cert.integrate(predictive)
        except UnsupportedOperator as err:
            raise AssertionError(
                "certificate declares 'integrate' but integrate() raised"
            ) from err
    else:
        _expect_unsupported(lambda: cert.integrate(predictive), "integrate")
    return state


def check_ports(instance_output, info) -> None:
    """Check that an implementation's output matches its declared produces.

    Args:
        instance_output: A real output object produced by the implementation.
        info: Its registered SeamInfo.

    Raises:
        AssertionError: When the output is no instance of any declared
            produced contract type.
    """
    if not info.produces:
        return
    produced_types = tuple(resolve_contract(name) for name in info.produces)
    if not isinstance(instance_output, produced_types):
        names = ", ".join(info.produces)
        raise AssertionError(
            f"output of type {type(instance_output).__name__} matches none of "
            f"the declared produced types: {names}"
        )


def _expect_unsupported(call, name: str) -> None:
    try:
        call()
    except UnsupportedOperator:
        return
    except Exception as err:
        raise AssertionError(
            f"undeclared operator '{name}' raised {type(err).__name__} "
            "instead of UnsupportedOperator"
        ) from err
    raise AssertionError(
        f"undeclared operator '{name}' did not raise UnsupportedOperator"
    )
