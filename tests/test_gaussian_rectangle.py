"""Independent checks of deterministic normal rectangle integration."""

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from scipy.integrate import quad
from scipy.stats import norm


@pytest.mark.parametrize("rho", [0.0, 0.5, -0.9])
@pytest.mark.parametrize("upper", [1.2, np.inf])
def test_rectangle_reference_and_monte_carlo(rho, upper):
    from spaceodyssey.gaussian import normal_rectangle_probability

    with jax.enable_x64():
        mean = jnp.array([0.2, -0.1])
        covariance = jnp.array([[1.0, rho], [rho, 1.0]])
        low = jnp.array([-0.8, -0.7])
        high = jnp.array([1.0, upper])
        actual = normal_rectangle_probability(mean, covariance, low, high, nodes=32)
        reference = normal_rectangle_probability(mean, covariance, low, high, nodes=128)
        assert abs(float(actual - reference)) <= 1e-8
        conditional_sd = np.sqrt(1 - rho**2)
        independent, _ = quad(
            lambda x: (
                norm.pdf(x - 0.2)
                * (
                    norm.cdf((upper + 0.1 - rho * (x - 0.2)) / conditional_sd)
                    - norm.cdf((-0.6 - rho * (x - 0.2)) / conditional_sd)
                )
            ),
            -0.8,
            1.0,
            epsabs=1e-12,
        )
        assert float(actual) == pytest.approx(independent, abs=1e-8)
        draws = np.random.default_rng(71).multivariate_normal(
            mean, covariance, size=200_000
        )
        hits = np.all((draws >= low) & (draws <= high), axis=1)
        se = np.sqrt(float(actual * (1 - actual)) / len(hits))
        assert abs(hits.mean() - actual) <= 5 * se


def test_rectangle_jit_and_derivatives_with_infinite_side():
    from spaceodyssey.gaussian import normal_rectangle_probability

    with jax.enable_x64():
        low, high = jnp.array([-1.0, -0.5]), jnp.array([0.7, jnp.inf])

        def probability(mean, rho):
            cov = jnp.array([[1.0, rho], [rho, 1.0]])
            return normal_rectangle_probability(mean, cov, low, high, nodes=32)

        mean = jnp.array([0.1, 0.2])
        compiled = jax.jit(probability)
        assert float(compiled(mean, 0.5)) == float(compiled(mean, 0.5))
        gradient = jax.jit(jax.grad(probability, argnums=(0, 1)))(mean, 0.5)
        for i in range(2):
            step = jnp.eye(2)[i] * 1e-5
            finite_difference = (
                probability(mean + step, 0.5) - probability(mean - step, 0.5)
            ) / 2e-5
            assert float(gradient[0][i]) == pytest.approx(
                float(finite_difference), abs=1e-8
            )
        finite_difference = (
            probability(mean, 0.50001) - probability(mean, 0.49999)
        ) / 2e-5
        assert float(gradient[1]) == pytest.approx(float(finite_difference), abs=1e-8)


@pytest.mark.parametrize("dimension", [1, 3, 4])
def test_independent_rectangle_dimensions(dimension):
    from spaceodyssey.gaussian import normal_rectangle_probability

    with jax.enable_x64():
        actual = normal_rectangle_probability(
            jnp.zeros(dimension),
            jnp.eye(dimension),
            jnp.full(dimension, -1.0),
            jnp.full(dimension, 0.5),
            nodes=8,
        )
        assert float(actual) == pytest.approx(
            (norm.cdf(0.5) - norm.cdf(-1.0)) ** dimension, abs=1e-12
        )
