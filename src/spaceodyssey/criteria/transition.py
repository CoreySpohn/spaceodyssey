"""Acyclic measurement requirements and conditional per-star action packages."""

import json
from collections.abc import Mapping
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
from jax.scipy.special import ndtr

from spaceodyssey.adapters.photomancy import SLOT_OK
from spaceodyssey.core.enums import F64, I64
from spaceodyssey.criteria.base import AbstractCriterion, StarOptions, _same_adjacent
from spaceodyssey.criteria.claims import AbstractClaim


def _float_array(value):
    return jnp.asarray(value, F64)


def _index_array(value):
    return jnp.asarray(value, I64)


def pass_probability(n0, v0, n, v, requirement_snr, margin):
    """Mean pass probability over paired count/variance quantiles on the last axis.

    Counts are electrons and variances are electrons squared. Invalid pairs and
    empty accumulated data cannot pass; zero new variance is deterministic.
    """
    finite = (
        jnp.isfinite(n0)
        & jnp.isfinite(v0)
        & jnp.isfinite(n)
        & jnp.isfinite(v)
        & jnp.isfinite(requirement_snr)
        & jnp.isfinite(margin)
    )
    valid = finite & (v0 >= 0) & (v >= 0) & (v0 + v > 0)
    total = jnp.where(valid, v0 + v, 1.0)
    numerator = jnp.where(valid, n0 + n, 0.0)
    threshold = jnp.where(finite, requirement_snr + margin, 0.0) * jnp.sqrt(total)
    stochastic = ndtr(
        (numerator - threshold) / jnp.sqrt(jnp.where(valid & (v > 0), v, 1.0))
    )
    probability = jnp.where(v > 0, stochastic, (numerator >= threshold).astype(F64))
    return jnp.mean(jnp.where(valid, probability, 0.0), axis=-1)


class TransitionState(eqx.Module):
    """Current node and repeat passes [S], and band accumulators [S, B]."""

    node: jax.Array
    passes: jax.Array
    sum_counts: jax.Array
    sum_variance: jax.Array


class TransitionSettings(eqx.Module):
    """Declared candidate gate and lowered channel resource costs.

    channel_cost [V+1, R] includes the catalog row and every charged overhead.
    resources names its columns, including telescope_days. imaging_lod_arcsec
    converts the dimensionless scatter guard to the summary's angular units.
    """

    candidate_detections: jax.Array = eqx.field(converter=_index_array)
    candidate_scatter_lod: jax.Array = eqx.field(converter=_float_array)
    imaging_lod_arcsec: jax.Array = eqx.field(converter=_float_array)
    channel_cost: jax.Array = eqx.field(converter=_float_array)
    resources: tuple[tuple[str, int], ...] = eqx.field(static=True)

    remaining_rows: jax.Array | None = eqx.field(default=None)

    def __check_init__(self):
        """Refuse an unnamed ledger or an invalid candidate guard."""
        if self.channel_cost.ndim != 2:
            raise ValueError("channel_cost must have channel and resource axes")
        resource = dict(self.resources).get("telescope_days")
        if resource is None or not 0 <= resource < self.channel_cost.shape[1]:
            raise ValueError("resources must name a telescope_days column")
        for name in (
            "candidate_detections",
            "candidate_scatter_lod",
            "imaging_lod_arcsec",
        ):
            value = getattr(self, name)
            if value.shape != ():
                raise ValueError(f"{name} must be scalar")
            valid = jnp.isfinite(value) & (value > 0)
            if not isinstance(valid, jax.core.Tracer) and not bool(valid):
                raise ValueError(f"{name} must be finite and positive")


def _detection_starts(probability, available, imaging):
    """Earliest row-major maximizer at or after each forecast node."""
    score = jnp.where(
        available & imaging[None, :] & (probability > 0), probability, -1.0
    )
    variant = jnp.argmax(score, axis=1)
    value = jnp.take_along_axis(score, variant[:, None], axis=1)[:, 0]
    index = jnp.arange(score.shape[0], dtype=I64) * score.shape[1] + variant

    def earlier(later, row):
        choose = row[0] >= later[0]
        best = tuple(jnp.where(choose, a, b) for a, b in zip(row, later, strict=True))
        return best, jnp.where(best[0] > 0, best[1], 0)

    _, starts = jax.lax.scan(
        earlier,
        (jnp.asarray(-1.0, F64), jnp.asarray(0, I64)),
        (value, index),
        reverse=True,
    )
    return starts


class TransitionTable(AbstractCriterion):
    """A DAG of repeated SNR requirements ending at valued labels.

    Nodes below n_steps are steps; n_steps + l is terminal label l. step_labels
    [K] records the achieved label while a step is active (-1 means no credit).
    on_fail is validated but no measurement decides a failure in this family.
    """

    step_names: tuple[str, ...] = eqx.field(static=True)
    labels: tuple[str, ...] = eqx.field(static=True)
    mode_names: tuple[str, ...] = eqx.field(static=True)
    claims: tuple[AbstractClaim, ...] = eqx.field(static=True)
    target_claim: str | None = eqx.field(static=True)
    step_modes: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, jnp.bool_))
    requirement_snr: jax.Array = eqx.field(converter=_float_array)
    margin: jax.Array = eqx.field(converter=_float_array)
    repeat: jax.Array = eqx.field(converter=_index_array)
    on_pass: jax.Array = eqx.field(converter=_index_array)
    on_fail: jax.Array = eqx.field(converter=_index_array)
    label_values: jax.Array = eqx.field(converter=_float_array)
    start: jax.Array = eqx.field(converter=_index_array)
    step_labels: jax.Array = eqx.field(converter=_index_array)

    @property
    def n_steps(self):
        """Number of nonterminal nodes, fixing the bounded package traversal."""
        return len(self.step_names)

    def __check_init__(self):
        """Validate shapes, finite requirements, references and both DAG edges."""
        k, n_labels, m = self.n_steps, len(self.labels), len(self.mode_names)
        if not k or not n_labels or not m:
            raise ValueError("steps, labels and modes must be nonempty")
        for names in (self.step_names, self.labels, self.mode_names):
            if len(set(names)) != len(names):
                raise ValueError("duplicate names")
        for name in (
            "requirement_snr",
            "margin",
            "repeat",
            "on_pass",
            "on_fail",
            "step_labels",
        ):
            if getattr(self, name).shape != (k,):
                raise ValueError(f"{name} must have one entry per step")
        if (
            self.step_modes.shape != (k, m)
            or self.label_values.shape != (n_labels,)
            or self.start.shape != ()
        ):
            raise ValueError("step_modes, label_values or start has the wrong shape")
        claim_names = tuple(c.name for c in self.claims)
        if len(set(claim_names)) != len(claim_names):
            raise ValueError("duplicate claim names")
        if self.target_claim is not None and self.target_claim not in claim_names:
            raise ValueError("target_claim must name one of claims")
        checks = (
            (
                self.requirement_snr,
                jnp.isfinite(self.requirement_snr) & (self.requirement_snr > 0),
                "requirement_snr must be finite and positive",
            ),
            (self.margin, jnp.isfinite(self.margin), "margin must be finite"),
            (self.repeat, self.repeat > 0, "repeat must be positive"),
            (
                self.on_pass,
                (self.on_pass >= 0) & (self.on_pass < k + n_labels),
                "on_pass node out of range",
            ),
            (
                self.on_fail,
                (self.on_fail >= -1) & (self.on_fail < k + n_labels),
                "on_fail node out of range",
            ),
            (
                self.start,
                (self.start >= 0) & (self.start < k + n_labels),
                "start node out of range",
            ),
            (
                self.step_labels,
                (self.step_labels >= -1) & (self.step_labels < n_labels),
                "step label out of range",
            ),
            (
                self.label_values,
                jnp.isfinite(self.label_values)
                & (self.label_values >= 0)
                & (self.label_values <= 1),
                "label_values must lie in [0, 1]",
            ),
            (self.step_modes, self.step_modes.any(-1), "each step must allow a mode"),
        )
        for _value, valid, message in checks:
            if not isinstance(valid, jax.core.Tracer) and not bool(jnp.all(valid)):
                raise ValueError(message)
        if not any(
            isinstance(x, jax.core.Tracer) for x in (self.on_pass, self.on_fail)
        ):
            edges = tuple(
                zip(self.on_pass.tolist(), self.on_fail.tolist(), strict=True)
            )
            visited, active = set(), set()

            def visit(node):
                if node < 0 or node >= k or node in visited:
                    return
                if node in active:
                    raise ValueError("transition table contains a cycle")
                active.add(node)
                for successor in edges[node]:
                    visit(successor)
                active.remove(node)
                visited.add(node)

            for node in range(k):
                visit(node)

    def state_template(self, n_tracks, n_bands):
        """Declare the per-track node, pass count and band-accumulator codec."""
        return TransitionState(
            jax.ShapeDtypeStruct((n_tracks,), I64),
            jax.ShapeDtypeStruct((n_tracks,), I64),
            jax.ShapeDtypeStruct((n_tracks, n_bands), F64),
            jax.ShapeDtypeStruct((n_tracks, n_bands), F64),
        )

    def init_state(self, n_tracks, n_bands):
        """Start every track at the declared node with no accumulated data."""
        return TransitionState(
            jnp.full((n_tracks,), self.start, I64),
            jnp.zeros(n_tracks, I64),
            jnp.zeros((n_tracks, n_bands), F64),
            jnp.zeros((n_tracks, n_bands), F64),
        )

    def _mode_layout(self, modes):
        names = tuple(m.name for m in modes.modes)
        indices = jnp.asarray([names.index(name) for name in self.mode_names], I64)
        allowed = jnp.any(
            self.step_modes[:, :, None]
            & (indices[None, :, None] == jnp.arange(len(names))),
            axis=1,
        )
        bands = jnp.asarray([modes.band_index(name) for name in names], I64)
        return allowed, bands

    def update(self, state, row, modes):
        """Fold assigned measured data, counting at most one pass per row."""
        allowed, bands = self._mode_layout(modes)
        step = jnp.minimum(state.node, self.n_steps - 1)
        mode = jnp.clip(row.mode, 0, len(modes.modes) - 1)
        band = bands[mode]
        take = (
            (jnp.arange(state.node.size)[:, None] == row.assignment[None, :])
            & (row.valid_science > 0)
            & (state.node[:, None] < self.n_steps)
            & allowed[step, mode][:, None]
            & (row.mode == mode)
        )
        counts = state.sum_counts.at[:, band].add(
            jnp.sum(jnp.where(take, row.counts[None, :, band], 0.0), axis=-1)
        )
        variance = state.sum_variance.at[:, band].add(
            jnp.sum(jnp.where(take, row.variance[None, :, band], 0.0), axis=-1)
        )
        n, v = counts[:, band], variance[:, band]
        passed = (
            take.any(-1)
            & (v > 0)
            & jnp.isfinite(n)
            & jnp.isfinite(v)
            & (
                n
                >= (self.requirement_snr[step] + self.margin[step])
                * jnp.sqrt(jnp.maximum(v, 0.0))
            )
        )
        passes = state.passes + passed.astype(I64)
        advance = passed & (passes >= self.repeat[step])
        return TransitionState(
            jnp.where(advance, self.on_pass[step], state.node),
            jnp.where(advance, 0, passes),
            counts.at[:, band].set(jnp.where(passed, 0.0, n)),
            variance.at[:, band].set(jnp.where(passed, 0.0, v)),
        )

    def _labels(self, node):
        return jnp.where(
            node < self.n_steps,
            self.step_labels[jnp.minimum(node, self.n_steps - 1)],
            node - self.n_steps,
        )

    def _values(self, node):
        label = self._labels(node)
        return jnp.where(label >= 0, self.label_values[jnp.maximum(label, 0)], 0.0)

    def _target_probability(self, p_claim):
        if self.target_claim is None:
            return jnp.ones(p_claim.shape[:-1], F64)
        return p_claim[..., tuple(c.name for c in self.claims).index(self.target_claim)]

    def value(self, state, p_claim, tick):
        """Return achieved credit [S] and its label, including active steps."""
        return self._values(state.node) * self._target_probability(
            p_claim
        ), self._labels(state.node)

    def options(self, state, summary_star, tick, modes, settings):
        """Expected incremental gain [G,V] and charged resources [G,V,R].

        Each slot contributes when the first exposure serves its current imaging
        gate or active step. Continuations assume each preceding requirement
        passes; failure ends that proposed package, retaining any achieved label.
        Later exposures use available nodes at or after the preceding node.
        """
        return jax.lax.cond(
            self._can_gain(state, summary_star),
            lambda: self._package_options(state, summary_star, tick, modes, settings),
            lambda: self._zero_options(state.node.size, summary_star, settings),
        )

    def _can_gain(self, state, summary_star):
        """Whether any active track has target credit and a declared start."""
        possible = (state.node < self.n_steps) & (
            self._target_probability(summary_star.p_claim) > 0
        )
        return jnp.any(possible) & jnp.any(summary_star.available)

    def _zero_options(self, n_tracks, summary_star, settings):
        """The same zero-credit table for a joint or individual track view."""
        n_grid, n_variants = summary_star.available.shape
        gain = jnp.zeros((n_grid, n_variants), F64)
        return StarOptions(
            gain,
            jnp.broadcast_to(
                settings.channel_cost[:n_variants],
                (n_grid, n_variants, settings.channel_cost.shape[-1]),
            ),
            jnp.broadcast_to(jnp.arange(n_variants, dtype=I64), gain.shape),
            jnp.zeros((*gain.shape, n_tracks), F64),
        )

    def _package_options(self, state, summary_star, tick, modes, settings):
        """Continuation search for a star with attainable incremental credit."""
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

        candidate = (summary_star.status == SLOT_OK) & (
            summary_star.scatter
            < settings.candidate_scatter_lod * settings.imaging_lod_arcsec
        )
        needed_images = jnp.maximum(
            0, settings.candidate_detections - summary_star.n_det
        )
        needed_images = jnp.maximum(needed_images, (~candidate).astype(I64))
        row_max = jnp.iinfo(I64).max
        repeat_max = jnp.max(self.repeat)
        repeat_limit = row_max // self.n_steps
        repeat_bound = jnp.clip(repeat_max, 0, repeat_limit) * self.n_steps
        safe_repeat = (repeat_max >= 0) & (repeat_max <= repeat_limit)

        def best(
            probability, node, variants, remaining_rows, *, detection=False, starts=None
        ):
            if starts is None:
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
                    else probability
                    / jnp.where(chargeable, costs[:, telescope], 1.0)[None, :]
                )
                index = jnp.argmax(jnp.where(valid, score, -1.0))
            else:
                index = jnp.where(node < n_grid, starts[jnp.maximum(node, 0)], 0)
            g, v = index // n_variants, index % n_variants
            ok = (
                available[g, v]
                & variants[v]
                & (g >= node)
                & (probability[g, v] > 0)
                & (remaining_rows[v] > 0)
            )
            return (
                jnp.where(ok, probability[g, v], 0.0),
                jnp.where(ok, costs[v], 0.0),
                g,
                v,
            )

        def slot_package(slot, first_g, first_v, remaining_rows, detection_starts=None):
            node = state.node[slot]
            step = jnp.minimum(node, self.n_steps - 1)
            images = needed_images[slot]
            imaging_step = jnp.any(step_variants[step] & imaging)
            needs_images = (images > 0) & ~imaging_step
            compatible = jnp.where(
                needs_images, imaging[first_v], step_variants[step, first_v]
            )
            compatible &= imaging[first_v] | (images == 0)
            compatible &= (
                (node < self.n_steps)
                & available[first_g, first_v]
                & (rows[first_v] > 0)
            )
            zero_cost = jnp.zeros(costs.shape[-1], F64)
            occupancy = jnp.where(
                summary_star.n_det[slot] == 0, summary_star.p_exist[slot], 1.0
            )

            def table_step(i, carry):
                node, reach, time, g, counts, variance, gain, images, started, left = (
                    carry
                )
                active = node < self.n_steps
                step = jnp.minimum(node, self.n_steps - 1)
                imaging_step = jnp.any(step_variants[step] & imaging)
                gate = jnp.where(active & ~imaging_step, images, 0)
                if imaging_channels:
                    p_det = summary_star.p_det[slot][:, imaging_index]

                    def image_attempt(j, attempt):
                        reach, time, g, started, left = attempt
                        p, cost, next_g, v = best(
                            p_det,
                            g,
                            imaging,
                            left,
                            detection=True,
                            starts=detection_starts,
                        )
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
            return result[6], result[2], result[9]

        same = _same_adjacent(
            (
                state,
                summary_star.n_det,
                summary_star.status,
                summary_star.scatter,
                summary_star.p_exist,
                summary_star.p_det,
                summary_star.fc_counts,
                summary_star.fc_variance,
                target,
            )
        )
        indices = jnp.arange(n_grid * n_variants)
        first_v = indices % n_variants
        left = jnp.broadcast_to(rows, (indices.size, n_variants)) - (
            imaging[None, :] == imaging[first_v, None]
        )
        zero_gain = jnp.zeros(indices.size, F64)
        zero_time = jnp.zeros((indices.size, costs.shape[-1]), F64)

        def track(slot, carry):
            gain, later, left, credit_by_track, before, after, raw_gain, time = carry
            consumed = before - after
            unchanged = jnp.all(left == before)
            # Remaining rows affect choices only through positivity. Reuse an
            # uncapped path only when every intermediate allowance stays positive.
            uncapped = (
                jnp.all(before > 0)
                & jnp.all(after > 0)
                & jnp.all(left > 0)
                & jnp.all(left - consumed > 0)
            )

            def calculate(remaining):
                def evaluate(starts):
                    return jax.vmap(
                        lambda index, allowance: slot_package(
                            slot,
                            index // n_variants,
                            index % n_variants,
                            allowance,
                            starts,
                        )
                    )(indices, remaining)

                if not imaging_channels:
                    return evaluate(None)
                # While allowances stay positive, their magnitudes cannot change
                # the best detection. Saturate the bound to avoid integer overflow.
                image_bound = (
                    jnp.minimum(needed_images[slot], row_max - repeat_bound)
                    + repeat_bound
                )
                bound = jnp.where(imaging, image_bound, repeat_bound)
                uncapped_search = (
                    safe_repeat & (state.passes[slot] >= 0) & jnp.all(remaining > bound)
                )
                return jax.lax.cond(
                    uncapped_search,
                    lambda: evaluate(
                        _detection_starts(
                            summary_star.p_det[slot][:, imaging_index],
                            available,
                            imaging,
                        )
                    ),
                    lambda: evaluate(None),
                )

            raw_gain, time, next_left = jax.lax.cond(
                same[slot] & (unchanged | uncapped),
                lambda remaining: (raw_gain, time, remaining - consumed),
                calculate,
                left,
            )
            credit = target[slot] * raw_gain
            return (
                gain + credit,
                later + time,
                next_left,
                credit_by_track.at[:, slot].set(credit),
                left,
                next_left,
                raw_gain,
                time,
            )

        # Keep the original track order and add the shared first cost only once.
        result = jax.lax.fori_loop(
            0,
            state.node.size,
            track,
            (
                zero_gain,
                zero_time,
                left,
                jnp.zeros((indices.size, state.node.size), F64),
                left,
                left,
                zero_gain,
                zero_time,
            ),
        )
        return StarOptions(
            result[0].reshape(n_grid, n_variants),
            (costs[first_v] + result[1]).reshape(n_grid, n_variants, -1),
            jnp.broadcast_to(jnp.arange(n_variants, dtype=I64), (n_grid, n_variants)),
            result[3].reshape(n_grid, n_variants, state.node.size),
        )


def load_transition_table(
    path, *, claims: Mapping[str, AbstractClaim]
) -> TransitionTable:
    """Lower named JSON steps and labels using the supplied claim registry."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    steps, labels = data["steps"], data["labels"]
    step_names = tuple(step["name"] for step in steps)
    label_names = tuple(label["name"] for label in labels)
    mode_names = tuple(dict.fromkeys(mode for step in steps for mode in step["modes"]))

    def node(reference):
        if reference in step_names:
            return step_names.index(reference)
        return len(steps) + label_names.index(reference)

    return TransitionTable(
        step_names=step_names,
        labels=label_names,
        mode_names=mode_names,
        claims=tuple(claims[name] for name in data["claims"]),
        target_claim=data["target_claim"],
        step_modes=[[mode in step["modes"] for mode in mode_names] for step in steps],
        requirement_snr=[step["requirement_snr"] for step in steps],
        margin=[step["margin"] for step in steps],
        repeat=[step["repeat"] for step in steps],
        on_pass=[node(step["on_pass"]) for step in steps],
        on_fail=[
            -1 if step["on_fail"] is None else node(step["on_fail"]) for step in steps
        ],
        label_values=[label["value"] for label in labels],
        start=node(data["start"]),
        step_labels=[
            -1
            if step["achieved_label"] is None
            else label_names.index(step["achieved_label"])
            for step in steps
        ],
    )
