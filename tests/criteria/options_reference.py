"""Unoptimized conditional package enumeration, retained as a test oracle."""

import jax
import jax.numpy as jnp

from spaceodyssey.adapters.photomancy import SLOT_OK
from spaceodyssey.core.enums import F64, I64
from spaceodyssey.criteria.base import StarOptions
from spaceodyssey.criteria.transition import pass_probability


def reference_options(self, state, summary_star, tick, modes, settings):
    """Expected incremental gain [G,V] and charged resources [G,V,R].

    Each slot contributes when the first exposure serves its current imaging
    gate or active step. Continuations assume each preceding requirement
    passes; failure ends that proposed package, retaining any achieved label.
    Later exposures use available nodes at or after the preceding node.
    """
    allowed, mode_bands = self._mode_layout(modes)
    n_grid, n_variants = summary_star.available.shape
    duration_count = len(modes.durations_d)
    variant_modes = jnp.arange(n_variants) // duration_count
    bands = mode_bands[variant_modes]
    step_variants = allowed[:, variant_modes]
    imaging_channels = modes.imaging_channels
    imaging = jnp.isin(jnp.arange(n_variants), jnp.asarray(imaging_channels, I64))
    imaging_index = jnp.asarray(
        [
            imaging_channels.index(v) if v in imaging_channels else 0
            for v in range(n_variants)
        ],
        I64,
    )
    costs = settings.channel_cost[:n_variants]
    telescope = dict(settings.resources)["telescope_days"]
    chargeable = jnp.all(jnp.isfinite(costs) & (costs >= 0), axis=-1) & (
        costs[:, telescope] > 0
    )
    available = summary_star.available & chargeable[None, :]
    target = self._target_probability(summary_star.p_claim)
    nodes = jnp.arange(n_grid)[:, None]

    rows = (
        jnp.full(n_variants, jnp.iinfo(I64).max, I64)
        if settings.remaining_rows is None
        else settings.remaining_rows
    )

    def best(probability, node, variants, remaining_rows, *, detection=False):
        valid = (
            available
            & variants[None, :]
            & (nodes >= node)
            & (probability > 0)
            & (remaining_rows > 0)[None, :]
        )
        score = (
            probability
            if detection
            else probability / jnp.where(chargeable, costs[:, telescope], 1.0)[None, :]
        )
        index = jnp.argmax(jnp.where(valid, score, -1.0))
        g, v = index // n_variants, index % n_variants
        ok = valid[g, v]
        return (
            jnp.where(ok, probability[g, v], 0.0),
            jnp.where(ok, costs[v], 0.0),
            g,
            v,
        )

    def slot_package(slot, first_g, first_v, remaining_rows):
        node = state.node[slot]
        step = jnp.minimum(node, self.n_steps - 1)
        candidate = (summary_star.status[slot] == SLOT_OK) & (
            summary_star.scatter[slot]
            < settings.candidate_scatter_lod * settings.imaging_lod_arcsec
        )
        images = jnp.maximum(
            0, settings.candidate_detections - summary_star.n_det[slot]
        )
        images = jnp.maximum(images, (~candidate).astype(I64))
        imaging_step = jnp.any(step_variants[step] & imaging)
        needs_images = (images > 0) & ~imaging_step
        compatible = jnp.where(
            needs_images, imaging[first_v], step_variants[step, first_v]
        )
        compatible &= imaging[first_v] | (images == 0)
        compatible &= (
            (node < self.n_steps) & available[first_g, first_v] & (rows[first_v] > 0)
        )
        zero_cost = jnp.zeros(costs.shape[-1], F64)
        occupancy = jnp.where(
            summary_star.n_det[slot] == 0, summary_star.p_exist[slot], 1.0
        )

        def table_step(i, carry):
            node, reach, time, g, counts, variance, gain, images, started, left = carry
            active = node < self.n_steps
            step = jnp.minimum(node, self.n_steps - 1)
            imaging_step = jnp.any(step_variants[step] & imaging)
            gate = jnp.where(active & ~imaging_step, images, 0)
            if imaging_channels:
                p_det = summary_star.p_det[slot][:, imaging_index]

                def image_attempt(j, attempt):
                    reach, time, g, started, left = attempt
                    p, cost, next_g, v = best(p_det, g, imaging, left, detection=True)
                    p = jnp.where(~started, p_det[first_g, first_v], p)
                    cost = jnp.where(~started, zero_cost, cost)
                    next_g = jnp.where(~started, first_g, next_g)
                    v = jnp.where(~started, first_v, v)
                    p = jnp.where(~started | (left[v] > 0), p, 0.0)
                    consume = started & (reach > 0) & (p > 0)
                    return (
                        reach * p,
                        time + reach * cost,
                        next_g,
                        jnp.asarray(True),
                        left - ((imaging == imaging[v]) & consume),
                    )

                reach, time, g, started, left = jax.lax.fori_loop(
                    0, gate, image_attempt, (reach, time, g, started, left)
                )
            else:
                reach = jnp.where(gate > 0, 0.0, reach)
            images = jnp.maximum(0, images - gate)
            remaining = self.repeat[step] - jnp.where(i == 0, state.passes[slot], 0)

            def step_attempt(j, attempt):
                reach, time, g, counts, variance, images, started, left = attempt
                probability = pass_probability(
                    counts[bands][None, :, None],
                    variance[bands][None, :, None],
                    summary_star.fc_counts[slot],
                    summary_star.fc_variance[slot],
                    self.requirement_snr[step],
                    self.margin[step],
                )
                if imaging_channels:
                    # A null image delivers no counts or variance to the step.
                    probability *= jnp.where(imaging[None, :], p_det, 1.0)
                variants = step_variants[step] & ((images == 0) | imaging)
                p, cost, next_g, v = best(probability, g, variants, left)
                p = jnp.where(~started, probability[first_g, first_v], p)
                cost = jnp.where(~started, zero_cost, cost)
                next_g, v = (
                    jnp.where(~started, first_g, next_g),
                    jnp.where(~started, first_v, v),
                )
                p = jnp.where(~started | (left[v] > 0), p, 0.0)
                consume = started & (reach > 0) & (p > 0)
                return (
                    reach * p,
                    time + reach * cost,
                    next_g,
                    counts.at[bands[v]].set(0.0),
                    variance.at[bands[v]].set(0.0),
                    jnp.maximum(0, images - imaging[v].astype(I64)),
                    jnp.asarray(True),
                    left - ((imaging == imaging[v]) & consume),
                )

            after = jax.lax.fori_loop(
                0,
                jnp.where(active, remaining, 0),
                step_attempt,
                (reach, time, g, counts, variance, images, started, left),
            )
            reach, time, g, counts, variance, images, started, left = after
            successor = jnp.where(active, self.on_pass[step], node)
            gain += reach * (self._values(successor) - self._values(node))
            return (
                successor,
                reach,
                time,
                g,
                counts,
                variance,
                gain,
                images,
                started,
                left,
            )

        result = jax.lax.fori_loop(
            0,
            self.n_steps,
            table_step,
            (
                node,
                compatible.astype(F64) * occupancy,
                zero_cost,
                first_g,
                state.sum_counts[slot],
                state.sum_variance[slot],
                jnp.asarray(0.0, F64),
                images,
                jnp.asarray(False),
                remaining_rows,
            ),
        )
        return target[slot] * result[6], result[2], result[9]

    def option(g, v):
        def track(slot, carry):
            gain, later_time, left, track_gain = carry
            credit, time, left = slot_package(slot, g, v, left)
            return (
                gain + credit,
                later_time + time,
                left,
                track_gain.at[slot].set(credit),
            )

        # The first exposure serves every track; later rows share one allowance.
        gain, later_time, _, track_gain = jax.lax.fori_loop(
            0,
            state.node.size,
            track,
            (
                jnp.asarray(0.0, F64),
                jnp.zeros(costs.shape[-1], F64),
                rows - (imaging == imaging[v]),
                jnp.zeros(state.node.size, F64),
            ),
        )
        return gain, costs[v] + later_time, track_gain

    gain, time, track_gain = jax.vmap(
        jax.vmap(option, in_axes=(None, 0)), in_axes=(0, None)
    )(jnp.arange(n_grid), jnp.arange(n_variants))
    return StarOptions(
        gain,
        time,
        jnp.broadcast_to(jnp.arange(n_variants, dtype=I64), (n_grid, n_variants)),
        track_gain,
    )
