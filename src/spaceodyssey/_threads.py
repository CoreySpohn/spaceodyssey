"""Carry the caller's numerical JAX settings into host build workers."""

import contextlib

import jax


def _thread_config():
    """The caller's thread-local JAX settings that change what a build computes."""
    return (
        bool(jax.config.jax_enable_x64),
        jax.config.jax_default_device,
        jax.config.jax_default_matmul_precision,
        bool(jax.config.jax_threefry_partitionable),
        jax.config.jax_default_prng_impl,
    )


def _in_thread(builder, declarations, config):
    """`builder(*declarations)` under the JAX settings in `config`."""
    x64, device, precision, partitionable, prng = config
    with contextlib.ExitStack() as stack:
        stack.enter_context(jax.enable_x64(x64))
        stack.enter_context(jax.threefry_partitionable(partitionable))
        stack.enter_context(jax.default_prng_impl(prng))
        if device is not None:
            stack.enter_context(jax.default_device(device))
        if precision is not None:
            stack.enter_context(jax.default_matmul_precision(precision))
        return builder(*declarations)
