"""Toy transaction study kinds and builders."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import (
    ALL_EFFECTS,
    F64,
    I64,
    PUBLIC,
    CoincidentOrder,
    Disposition,
    EffectKind,
    EventKind,
    RecordStatus,
    Scope,
    Stage,
)
from spaceodyssey.core.events import enqueue
from spaceodyssey.core.kinds import (
    AbstractChannelModel,
    AbstractDesign,
    AbstractObjective,
    AbstractObserver,
    AbstractPolicy,
    AbstractProcess,
    AbstractWorldSource,
)
from spaceodyssey.core.ledger import AssetTable, ChannelTable
from spaceodyssey.core.records import record_none
from spaceodyssey.core.snapshot import EffectTable, MeanBelief, effects_none
from spaceodyssey.core.study import Agenda, Clock, Study

RECORD_NONE, RECORD_MEASUREMENT, RECORD_AUXILIARY = range(3)


class ToyDesign(AbstractDesign):
    """The continuous design leaf the gradients are taken with respect to."""

    gain: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, F64))


class ToyTruth(eqx.Module):
    """Latent signal vector for the toy channel."""

    signal: jax.Array


def draw_truth(key, caps):
    """Draw independent toy signals uniformly between 0.5 and 1.5."""
    return ToyTruth(signal=jr.uniform(key, (caps.n_subjects,), minval=0.5, maxval=1.5))


class ToyWorldSource(AbstractWorldSource):
    """Source of independent latent signals for the toy study."""

    def draw(self, key, capacities):
        """Draw a toy truth state using the declared subject capacity."""
        return draw_truth(key, capacities)


class ToyPhotometry(AbstractChannelModel):
    """Toy signal with independent Gaussian measurement noise."""

    noise: jax.Array

    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        """Returns (measurement, raw standard normal variate)."""
        z = jr.normal(key, dtype=F64)
        noise = self.noise[channel_index] / jnp.sqrt(jnp.maximum(fraction, 1e-3))
        return design.gain * truth.signal[subject] + noise * z, z


class ToyRunningMean(AbstractObserver):
    """Running mean of completed measurements per subject.

    The record-kind numbers it interprets are fields, so a study can renumber kinds
    without touching the core.
    """

    measurement_kind: int = eqx.field(default=RECORD_MEASUREMENT)
    auxiliary_kind: int = eqx.field(default=RECORD_AUXILIARY)

    def fold(self, belief, rec, caps):
        """Update completed measurement means and auxiliary values."""
        t = rec.subject
        take = (
            rec.valid
            & (rec.kind == self.measurement_kind)
            & (rec.status == RecordStatus.COMPLETED)
        )
        extra = rec.valid & (rec.kind == self.auxiliary_kind)
        n = belief.n[t].astype(F64)
        new = (belief.mean[t] * n + rec.payload[0]) / (n + 1.0)
        return MeanBelief(
            mean=belief.mean.at[t].set(jnp.where(take, new, belief.mean[t])),
            n=belief.n.at[t].add(take.astype(I64)),
            # records of the extra kind only accumulate into the first aux slot
            aux=belief.aux.at[0].add(jnp.where(extra, rec.payload[0], 0.0)),
        )


class ToySolveAnalysis(ToyRunningMean):
    """Analysis stage: a small linear solve written back into `aux`."""

    def analyze(self, belief):
        """Write a small linear solve into the auxiliary state."""
        n_an = belief.aux.shape[0]
        m, n = belief.mean[:n_an], belief.n[:n_an].astype(F64)
        a = jnp.outer(m, m) + jnp.diag(1.0 + n + 0.1 * jnp.arange(n_an))
        return MeanBelief(belief.mean, belief.n, jnp.linalg.solve(a, m + 1.0))


class NullProcess(AbstractProcess):
    """Process that seeds and writes nothing."""

    pass


class HeartbeatProcess(AbstractProcess):
    """A fixed number of periodic process events.

    Seed n_fires events at period_ticks * (i + 1) in the release stage. When writes is
    true, each event logs its fire number in an auxiliary record, released to the
    process audience.
    """

    period_ticks: int
    n_fires: int = eqx.field(static=True)
    audience: int
    writes: bool = eqx.field(default=False)

    def seed(self, carry, caps):
        """Seed a fixed number of periodic process events."""
        for i in range(self.n_fires):
            carry = enqueue(
                carry,
                True,
                kind=EventKind.PROCESS,
                tick=self.period_ticks * (i + 1),
                stage=Stage.RELEASE,
                group=i,
                audience=self.audience,
                caps=caps,
            )
        return carry

    def handle(self, carry, ev, study):
        """Optionally write the current fire number as a record."""
        base = record_none(study.caps)
        if not self.writes:
            return carry, base
        payload = base.payload.at[0].set(ev.group.astype(F64) + 1.0)
        rec = dataclasses.replace(
            base,
            kind=jnp.asarray(RECORD_AUXILIARY, I64),
            start_tick=ev.tick,
            end_tick=ev.tick,
            available_tick=ev.tick,
            status=jnp.asarray(RecordStatus.COMPLETED, I64),
            acquisition_id=ev.sid,
            released_tick=jnp.asarray(-1, I64),
            accepted_tick=jnp.asarray(-1, I64),
            superseded_by=jnp.asarray(-1, I64),
            payload=payload,
            valid=jnp.asarray(True),
        )
        return carry, rec


def _effects(
    caps,
    kind,
    subject=0,
    channel_index=0,
    group=0,
    tick=0,
    lane=0,
    scope=0,
    disposition=0,
    disruptive=False,
):
    def col(v):
        return jnp.asarray(v, I64) * jnp.ones(caps.n_effects, I64)

    return EffectTable(
        kind=jnp.asarray(kind, I64),
        subject=col(subject) if np.ndim(subject) == 0 else jnp.asarray(subject, I64),
        channel_index=col(channel_index)
        if np.ndim(channel_index) == 0
        else jnp.asarray(channel_index, I64),
        group=col(group),
        tick=col(tick) if np.ndim(tick) == 0 else jnp.asarray(tick, I64),
        lane=col(lane) if np.ndim(lane) == 0 else jnp.asarray(lane, I64),
        scope=col(scope) if np.ndim(scope) == 0 else jnp.asarray(scope, I64),
        disposition=col(disposition)
        if np.ndim(disposition) == 0
        else jnp.asarray(disposition, I64),
        disruptive=jnp.asarray(disruptive, jnp.bool_)
        * jnp.ones(caps.n_effects, jnp.bool_),
    )


def _chan_index(study, channel_id):
    return jnp.argmax(study.channels.id == channel_id).astype(I64)


class NullPolicy(AbstractPolicy):
    """Policy that emits no effects."""

    def decide(self, pstate, view, study, key):
        """Return an empty effect table and unchanged policy state."""
        return effects_none(study.caps), pstate, jnp.asarray(0.0, F64)


class FixedPlanPolicy(AbstractPolicy):
    """A fixed plan of (subject, channel id) observations, one entry per decision.

    The decision index is `view.n_dec`; the plan position is `pstate[0]`. At
    `wait_at`, `cancel_at` and `stop_at` the policy emits only that effect and does
    not advance the plan; at `bundle_at` it adds a second Observe of the same
    subject on `bundle_ch`. When the plan is exhausted it stops the campaign with a
    drain. `cancel_gen_offset` shifts the generation named by a Cancel (0 live, -1
    stale). Cancel and Stop act on lane 0.
    """

    plan: tuple[tuple[int, int], ...]
    wait_at: int = eqx.field(default=-1)
    wait_ticks: int = eqx.field(default=50)
    bundle_at: int = eqx.field(default=-1)
    bundle_ch: int = eqx.field(default=12)
    cancel_at: int = eqx.field(default=-1)
    cancel_gen_offset: int = eqx.field(default=0)
    stop_at: int = eqx.field(default=-1)
    stop_scope: int = eqx.field(default=Scope.LANE)
    stop_disp: int = eqx.field(default=Disposition.CLOSE)

    def decide(self, pstate, view, study, key):
        """Execute the next plan entry or a declared special effect."""
        caps = study.caps
        n = view.n_dec
        n_plan = len(self.plan)
        pos = pstate[0].astype(I64)
        subjects = jnp.asarray([p[0] for p in self.plan] or [0], I64)
        cids = jnp.asarray([p[1] for p in self.plan] or [0], I64)
        p = jnp.minimum(pos, max(n_plan - 1, 0))
        is_wait = n == self.wait_at
        is_cancel = n == self.cancel_at
        is_stop = n == self.stop_at
        special = is_wait | is_cancel | is_stop
        exhausted = pos >= n_plan
        observe = ~special & ~exhausted
        bundle = observe & (n == self.bundle_at)

        ch0 = _chan_index(study, cids[p])
        ch1 = _chan_index(study, self.bundle_ch)
        kind0 = jnp.select(
            [is_wait, is_cancel, is_stop, observe],
            [EffectKind.WAIT, EffectKind.CANCEL, EffectKind.STOP, EffectKind.OBSERVE],
            EffectKind.STOP,
        )
        scope0 = jnp.where(is_stop, self.stop_scope, Scope.CAMPAIGN)
        disp0 = jnp.where(is_stop, self.stop_disp, Disposition.DRAIN)
        tick0 = jnp.where(
            is_wait,
            view.tick + self.wait_ticks,
            jnp.where(is_cancel, self.cancel_gen_offset, 0),
        )
        eff = _effects(
            caps,
            kind=jnp.stack(
                [kind0, jnp.where(bundle, EffectKind.OBSERVE, EffectKind.NONE)]
            ),
            subject=jnp.stack([subjects[p], subjects[p]]),
            channel_index=jnp.stack([ch0, ch1]),
            group=n,
            tick=jnp.stack([tick0, jnp.asarray(0, I64)]),
            lane=0,
            scope=jnp.stack([scope0, jnp.asarray(0, I64)]),
            disposition=jnp.stack([disp0, jnp.asarray(0, I64)]),
        )
        new_ps = pstate.at[0].add(observe.astype(F64))
        # the policy's own draw: unused by the plan, but recorded for the draw study
        return eff, new_ps, jr.uniform(key, dtype=F64)

    def decide_py(self, pstate, view_np, study, key):
        """The same rule written with Python control flow on host values."""
        caps = study.caps
        n = int(view_np.n_dec)
        pos = int(pstate[0])
        kinds = [EffectKind.NONE] * caps.n_effects
        subjects = [0] * caps.n_effects
        channels = [0] * caps.n_effects
        ticks = [0] * caps.n_effects
        scopes = [0] * caps.n_effects
        disps = [0] * caps.n_effects
        ids = [int(i) for i in np.asarray(study.channels.id)]
        advance_plan = False
        if n == self.wait_at:
            kinds[0], ticks[0] = EffectKind.WAIT, int(view_np.tick) + self.wait_ticks
        elif n == self.cancel_at:
            kinds[0], ticks[0] = EffectKind.CANCEL, self.cancel_gen_offset
        elif n == self.stop_at:
            kinds[0], scopes[0], disps[0] = (
                EffectKind.STOP,
                self.stop_scope,
                self.stop_disp,
            )
        elif pos >= len(self.plan):
            kinds[0], scopes[0], disps[0] = (
                EffectKind.STOP,
                Scope.CAMPAIGN,
                Disposition.DRAIN,
            )
        else:
            subject, cid = self.plan[pos]
            kinds[0], subjects[0], channels[0] = (
                EffectKind.OBSERVE,
                subject,
                ids.index(cid),
            )
            advance_plan = True
            if n == self.bundle_at:
                kinds[1], subjects[1], channels[1] = (
                    EffectKind.OBSERVE,
                    subject,
                    ids.index(self.bundle_ch),
                )
        eff = EffectTable(
            kind=jnp.asarray(kinds, I64),
            subject=jnp.asarray(subjects, I64),
            channel_index=jnp.asarray(channels, I64),
            group=jnp.full(caps.n_effects, n, I64),
            tick=jnp.asarray(ticks, I64),
            lane=jnp.zeros(caps.n_effects, I64),
            scope=jnp.asarray(scopes, I64),
            disposition=jnp.asarray(disps, I64),
            disruptive=jnp.zeros(caps.n_effects, jnp.bool_),
        )
        new_ps = jnp.asarray(pstate).at[0].add(1.0 if advance_plan else 0.0)
        return eff, new_ps, jr.uniform(key, dtype=F64)


class ReactivePolicy(AbstractPolicy):
    """Follow a plan when the snapshot gains new rows.

    Follows a fixed plan but acts only when its view has gained rows since its last
    decision, so a wake that carries no new information changes nothing. Used where
    extra wakes must not alter the trajectory (leak scenarios).
    """

    plan: tuple[tuple[int, int], ...]

    def decide(self, pstate, view, study, key):
        """Advance the plan when the snapshot gains visible rows."""
        caps = study.caps
        n_plan = len(self.plan)
        pos = pstate[0].astype(I64)
        fresh = (view.n_dec == 0) | (view.n_rows != pstate[1].astype(I64))
        subjects = jnp.asarray([p[0] for p in self.plan], I64)
        cids = jnp.asarray([p[1] for p in self.plan], I64)
        p = jnp.minimum(pos, n_plan - 1)
        exhausted = pos >= n_plan
        observe = fresh & ~exhausted
        stop = fresh & exhausted
        kind0 = jnp.where(
            observe,
            EffectKind.OBSERVE,
            jnp.where(stop, EffectKind.STOP, EffectKind.NONE),
        )
        eff = _effects(
            caps,
            kind=jnp.stack([kind0, jnp.asarray(EffectKind.NONE, I64)]),
            subject=subjects[p],
            channel_index=_chan_index(study, cids[p]),
            group=view.n_dec,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        new_ps = (
            pstate.at[0]
            .add(observe.astype(F64))
            .at[1]
            .set(jnp.where(fresh, view.n_rows.astype(F64), pstate[1]))
        )
        return eff, new_ps, jnp.asarray(0.0, F64)


class SpinWaitPolicy(AbstractPolicy):
    """Observe once, then emit zero-delay waits until the guard stalls.

    Observes subject 0 once, then answers every later wake with a zero-delay Wait, which
    feeds the next wake at the same position until the micro guard stalls the run.
    """

    def decide(self, pstate, view, study, key):
        """Observe once, then wait repeatedly at the same tick."""
        caps = study.caps
        first = view.n_dec == 0
        kind0 = jnp.where(
            first,
            EffectKind.OBSERVE,
            jnp.where(view.n_rows >= 1, EffectKind.WAIT, EffectKind.NONE),
        )
        eff = _effects(
            caps,
            kind=jnp.stack([kind0, jnp.asarray(EffectKind.NONE, I64)]),
            subject=0,
            channel_index=_chan_index(study, 10),
            group=view.n_dec,
            tick=view.tick,
        )
        return eff, pstate, jnp.asarray(0.0, F64)


class ThresholdStopPolicy(AbstractPolicy):
    """Drain after a subject exceeds the signal-to-noise threshold.

    Observes the least-observed of the first two subjects on channel id 10 and drains
    the campaign once any subject's snr exceeds `stop_snr`.
    """

    stop_snr: float = eqx.field(default=3.0)

    def decide(self, pstate, view, study, key):
        """Observe until a subject exceeds the threshold."""
        caps = study.caps
        snr = (
            view.belief.mean
            * jnp.sqrt(view.belief.n.astype(F64))
            / study.channel_model.noise[0]
        )
        stop = jnp.max(snr) > self.stop_snr
        subject = jnp.argmin(view.belief.n[:2]).astype(I64)
        eff = _effects(
            caps,
            kind=jnp.stack(
                [
                    jnp.where(stop, EffectKind.STOP, EffectKind.OBSERVE),
                    jnp.asarray(EffectKind.NONE, I64),
                ]
            ),
            subject=subject,
            channel_index=_chan_index(study, 10),
            group=view.n_dec,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        return eff, pstate, jnp.asarray(0.0, F64)


class StochasticPolicy(AbstractPolicy):
    """Choose observations or waits with a sigmoid probability.

    Gradient study (stochastic decisions): at each of the first `n_obs` decisions
    observe the next subject in turn with probability sigmoid(theta), else wait
    `wait_ticks`; then stop with a drain. `pstate[0]` counts the observes made, so the
    subject of the j-th observe is j mod 2 whatever the decision positions were. Returns
    the uniform it drew.
    """

    theta: jax.Array = eqx.field(converter=lambda x: jnp.asarray(x, F64))
    n_obs: int = eqx.field(default=5)
    wait_ticks: int = eqx.field(default=10)

    def decide(self, pstate, view, study, key):
        """Choose an observation or wait using the policy key."""
        caps = study.caps
        u = jr.uniform(key, dtype=F64)
        stop = view.n_dec >= self.n_obs
        take = ~stop & (u < jax.nn.sigmoid(self.theta))
        kind0 = jnp.where(
            stop, EffectKind.STOP, jnp.where(take, EffectKind.OBSERVE, EffectKind.WAIT)
        )
        eff = _effects(
            caps,
            kind=jnp.stack([kind0, jnp.asarray(EffectKind.NONE, I64)]),
            subject=pstate[0].astype(I64) % 2,
            channel_index=_chan_index(study, 10),
            group=view.n_dec,
            tick=view.tick + self.wait_ticks,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
        )
        return eff, pstate.at[0].add(take.astype(F64)), u


class ReplayPolicy(AbstractPolicy):
    """Emit recorded effects by decision index, repeating the last row.

    Each field of effects has shape (maximum_decisions, maximum_effects). Wait ticks are
    absolute, so replay freezes the discrete path.
    """

    effects: EffectTable

    def decide(self, pstate, view, study, key):
        """Return the recorded effects for the current decision."""
        n_max = self.effects.kind.shape[0]
        i = jnp.minimum(view.n_dec, n_max - 1)
        eff = jax.tree_util.tree_map(lambda a: a[i], self.effects)
        return eff, pstate, jnp.asarray(0.0, F64)


def forced_effects(caps, study, m, n_dec_max=6):
    """Build m alternating-subject observations followed by a campaign drain."""
    rows = []
    for j in range(n_dec_max):
        if j < m:
            k0, subject = EffectKind.OBSERVE, j % 2
        else:
            k0, subject = EffectKind.STOP, 0
        rows.append(
            _effects(
                caps,
                kind=jnp.stack(
                    [jnp.asarray(k0, I64), jnp.asarray(EffectKind.NONE, I64)]
                ),
                subject=subject,
                channel_index=_chan_index(study, 10),
                group=j,
                scope=Scope.CAMPAIGN,
                disposition=Disposition.DRAIN,
            )
        )
    return jax.tree_util.tree_map(lambda *xs: jnp.stack(xs), *rows)


def survey_1(caps=None, **overrides):
    """Build a fixed five-acquisition toy study."""
    caps = Capacities() if caps is None else caps
    c = caps.n_channels
    channels = ChannelTable(
        id=jnp.asarray([10, 11, 12][:c], I64),
        lane=jnp.asarray([0, 1, 2][:c], I64),
        occupancy_ticks=jnp.asarray([100, 150, 50][:c], I64),
        latency_ticks=jnp.asarray([20, 20, 0][:c], I64),
        kind=jnp.full(c, RECORD_MEASUREMENT, I64),
        audience=jnp.full(c, PUBLIC, I64),
        cost=jnp.ones((c, caps.n_resources), F64),
        charge=jnp.ones((c, caps.n_resources), F64),
        coincident_order=jnp.full(c, CoincidentOrder.FAULT_FIRST, I64),
    )
    budget = jnp.concatenate(
        [
            jnp.full((caps.n_assets, caps.n_resources), 10.0, F64),
            jnp.full((caps.n_programs + 1, caps.n_resources), 100.0, F64),
        ]
    )
    base = dict(
        caps=caps,
        channels=channels,
        channel_model=ToyPhotometry(noise=jnp.asarray([0.5, 0.3, 0.8][:c], F64)),
        lane_asset=jnp.asarray([0, 0, 1][: caps.n_lanes], I64),
        transition_ticks=jnp.asarray([10, 10, 5][: caps.n_lanes], I64),
        budget=budget,
        horizon_tick=1000,
        policies=(FixedPlanPolicy(plan=((0, 10), (1, 10), (0, 10), (1, 10), (0, 10))),),
        phase_until=(1000,),
        process=NullProcess(),
        observer=ToyRunningMean(),
        design=ToyDesign(gain=1.0),
    )
    base.update(overrides)
    base["assets"] = AssetTable(
        lane_asset=base.pop("lane_asset"),
        transition_ticks=base.pop("transition_ticks"),
        freeze_ticks=jnp.zeros(caps.n_assets, I64),
        disruptive_resource=jnp.full(caps.n_assets, -1, I64),
    )
    policies = base.pop("policies")
    base["agenda"] = Agenda(
        policies,
        base.pop("phase_until"),
        base.pop("permitted_effects", (ALL_EFFECTS,) * len(policies)),
    )
    base["clock"] = Clock(1.0)
    base["world_source"] = ToyWorldSource()
    base["objective"] = ToyObjective(threshold=base.pop("detect_thr", 3.0))
    return Study(**base)


def d2_study(theta=0.3, **overrides):
    """Build a study with five stochastic observe-or-wait decisions."""
    return survey_1(policies=(StochasticPolicy(theta=theta),), **overrides)


def credit_hard(carry, study):
    """Number of subjects whose snr exceeds the declared threshold."""
    return jnp.sum(_snr(carry, study) > study.objective.threshold)


def credit_soft(carry, study, width):
    """Smooth stand-in for `credit_hard`: sum of sigmoid((snr - thr) / width)."""
    return jnp.sum(
        jax.nn.sigmoid((_snr(carry, study) - study.objective.threshold) / width)
    )


def _snr(carry, study):
    noise_eff = study.channel_model.noise[0]
    b = carry.belief
    return b.mean * jnp.sqrt(b.n.astype(F64)) / noise_eff


class ToyObjective(AbstractObjective):
    """Count subjects exceeding a declared signal-to-noise threshold."""

    threshold: jax.Array = eqx.field(
        default=3.0, converter=lambda x: jnp.asarray(x, F64)
    )

    def evaluate(self, carry, study):
        """Return the hard threshold count."""
        return credit_hard(carry, study)
