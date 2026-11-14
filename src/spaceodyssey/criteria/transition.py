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
from spaceodyssey.criteria.base import AbstractCriterion, StarOptions
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
                else probability
                / jnp.where(chargeable, costs[:, telescope], 1.0)[None, :]
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

        def slot_package(slot, first_g, first_v):
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
                            p_det, g, imaging, left, detection=True
                        )
                        p = jnp.where(~started, p_det[first_g, first_v], p)
                        cost = jnp.where(~started, zero_cost, cost)
                        next_g = jnp.where(~started, first_g, next_g)
                        v = jnp.where(~started, first_v, v)
                        p = jnp.where(left[v] > 0, p, 0.0)
                        return (
                            reach * p,
                            time + reach * cost,
                            next_g,
                            jnp.asarray(True),
                            left - (imaging == imaging[v]),
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
                    variants = step_variants[step] & ((images == 0) | imaging)
                    p, cost, next_g, v = best(probability, g, variants, left)
                    p = jnp.where(~started, probability[first_g, first_v], p)
                    cost = jnp.where(~started, zero_cost, cost)
                    next_g, v = (
                        jnp.where(~started, first_g, next_g),
                        jnp.where(~started, first_v, v),
                    )
                    p = jnp.where(left[v] > 0, p, 0.0)
                    return (
                        reach * p,
                        time + reach * cost,
                        next_g,
                        counts.at[bands[v]].set(0.0),
                        variance.at[bands[v]].set(0.0),
                        jnp.maximum(0, images - imaging[v].astype(I64)),
                        jnp.asarray(True),
                        left - (imaging == imaging[v]),
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
                    rows,
                ),
            )
            return target[slot] * result[6], result[2]

        def option(g, v):
            gain, later_time = jax.vmap(slot_package, in_axes=(0, None, None))(
                jnp.arange(state.node.size), g, v
            )
            return gain.sum(), costs[v] + later_time.sum(0)

        gain, time = jax.vmap(jax.vmap(option, in_axes=(None, 0)), in_axes=(0, None))(
            jnp.arange(n_grid), jnp.arange(n_variants)
        )
        return StarOptions(
            gain,
            time,
            jnp.broadcast_to(jnp.arange(n_variants, dtype=I64), (n_grid, n_variants)),
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
