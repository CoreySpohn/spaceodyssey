"""Toy transaction study kinds and builders."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import jax.random as jr
import numpy as np

from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.declarations import Declaration
from spaceodyssey.core.enums import (
    ALL_EFFECTS,
    F64,
    I64,
    PUBLIC,
    Audience,
    CoincidentOrder,
    Disposition,
    EffectKind,
    EventKind,
    RecordStatus,
    RefitStatus,
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
    ObserverCapabilities,
)
from spaceodyssey.core.ledger import AssetTable, ChannelTable
from spaceodyssey.core.records import (
    KindRegistry,
    PayloadColumn,
    RecordKind,
    record_none,
)
from spaceodyssey.core.refresh import (
    REFRESH_PRODUCT_KIND,
    HostLoopRefit,
    MaskedBatchRefit,
    RefreshSchedule,
)
from spaceodyssey.core.snapshot import (
    BeliefLayout,
    EffectTable,
    MeanBelief,
    effects_none,
    project_mean,
)
from spaceodyssey.core.study import Agenda, Clock, Study

RECORD_NONE, RECORD_MEASUREMENT, RECORD_AUXILIARY = range(3)

# The measurement layout is what `realize` fills: the value, the raw noise variate and
# the occupancy fraction. The variate lets anyone who holds it recover the true signal
# from the value, so it is a scorer-only column.
MEASUREMENT_KIND = RecordKind(
    "measurement",
    (
        PayloadColumn("value", PUBLIC),
        PayloadColumn("variate", Audience.SCORER),
        PayloadColumn("fraction", PUBLIC),
    ),
)
AUXILIARY_KIND = RecordKind("auxiliary", (PayloadColumn("value", PUBLIC),))


def toy_registry():
    """A fresh registry with the toy kinds at their `RECORD_*` indices."""
    return KindRegistry(
        (RecordKind("none", ()), MEASUREMENT_KIND, AUXILIARY_KIND),
    )


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

    subject_leaves = ("mean", "n")

    measurement_kind: int = eqx.field(default=RECORD_MEASUREMENT)
    auxiliary_kind: int = eqx.field(default=RECORD_AUXILIARY)

    def layout(self, caps):
        """`MeanBelief` for both types: the summary is the belief, counted by `n`."""
        belief = MeanBelief(
            mean=jax.ShapeDtypeStruct((caps.n_subjects,), F64),
            n=jax.ShapeDtypeStruct((caps.n_subjects,), I64),
            aux=jax.ShapeDtypeStruct((caps.n_an,), F64),
        )
        return BeliefLayout(
            belief=belief, summary=belief, count_leaf="n", project=project_mean
        )

    def fold(self, belief, rec, caps, key=None):
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

    def install(self, belief, fitted, subject):
        """Take the subject's mean and count from a refit; keep the auxiliary state."""
        return MeanBelief(
            mean=belief.mean.at[subject].set(fitted.mean[subject]),
            n=belief.n.at[subject].set(fitted.n[subject]),
            aux=belief.aux,
        )


class ToyRefitMean(ToyRunningMean):
    """The running mean that changes only by refresh: refit-capable, never folded.

    Its belief, and its policy summary, hold what the last checkpoint installed.
    """

    def capabilities(self):
        """Refit-capable on measurement rows, not incremental."""
        return ObserverCapabilities(
            incremental=False, refit=True, refit_kinds=("measurement",)
        )


class ToySolveAnalysis(ToyRunningMean):
    """Analysis stage: a small linear solve written back into `aux`."""

    def analyze(self, belief):
        """Write a small linear solve into the auxiliary state."""
        n_an = belief.aux.shape[0]
        m, n = belief.mean[:n_an], belief.n[:n_an].astype(F64)
        a = jnp.outer(m, m) + jnp.diag(1.0 + n + 0.1 * jnp.arange(n_an))
        return MeanBelief(belief.mean, belief.n, jnp.linalg.solve(a, m + 1.0))


class SampleBelief(eqx.Module):
    """A running mean with one posterior draw per subject and the variates drawn.

    `sample` is a draw of each subject's signal from the conjugate Gaussian
    posterior after its latest fold; `draws[subject, acquisition_index]` holds the
    standard-normal variate the fold of that acquisition drew (zero where none was
    drawn, or the index is beyond the table).
    """

    mean: jax.Array
    n: jax.Array
    aux: jax.Array
    sample: jax.Array
    draws: jax.Array


def project_sample(belief):
    """The trace projection of a `SampleBelief`: its posterior draws and `aux`."""
    return belief.sample, belief.aux


class ToySampleObserver(ToyRunningMean):
    """A sampling observer: each fold draws from the record's fold key.

    The full belief is a `SampleBelief`; the policy summary is the `MeanBelief` of
    the running mean, which draws nothing. `noise` is the per-record standard
    deviation the posterior assumes, under a flat prior.
    """

    subject_leaves = ("mean", "n", "sample", "draws")

    noise: jax.Array = eqx.field(default=1.0, converter=lambda x: jnp.asarray(x, F64))
    n_draws: int = eqx.field(default=8, static=True)

    def init(self, caps):
        """Zero means, counts, draws and variates."""
        mean = super().init(caps)
        n = caps.n_subjects
        return SampleBelief(
            mean.mean,
            mean.n,
            mean.aux,
            jnp.zeros(n, F64),
            jnp.zeros((n, self.n_draws), F64),
        )

    def init_summary(self, caps):
        """The running mean's `MeanBelief`."""
        return super().init(caps)

    def layout(self, caps):
        """`SampleBelief` and `MeanBelief`, counted by `n`, projected separately."""
        return BeliefLayout(
            belief=jax.eval_shape(lambda: self.init(caps)),
            summary=jax.eval_shape(lambda: self.init_summary(caps)),
            count_leaf="n",
            project=project_sample,
            project_summary=project_mean,
        )

    def fold(self, belief, rec, caps, key=None):
        """Fold the running mean, then draw the subject's posterior from `key`."""
        if key is None:
            raise ValueError("ToySampleObserver draws from its fold key; pass one")
        mean = super().fold(self.summary(belief), rec, caps)
        t, i = rec.subject, rec.acquisition_index
        took = mean.n[t] > belief.n[t]
        z = jr.normal(key, dtype=F64)
        scale = self.noise / jnp.sqrt(jnp.maximum(mean.n[t], 1).astype(F64))
        draw = mean.mean[t] + scale * z
        return SampleBelief(
            mean.mean,
            mean.n,
            mean.aux,
            belief.sample.at[t].set(jnp.where(took, draw, belief.sample[t])),
            belief.draws.at[t, i].set(
                jnp.where(took, z, belief.draws[t, i]), mode="drop"
            ),
        )

    def fold_summary(self, summary, rec, caps, key=None):
        """Fold the running mean; the summary draws nothing."""
        return super().fold(summary, rec, caps)

    def summary(self, belief):
        """The running mean of a `SampleBelief`."""
        return MeanBelief(belief.mean, belief.n, belief.aux)

    def install(self, belief, fitted, subject):
        """Take every per-subject leaf of the subject from a refit; keep `aux`."""
        mean = super().install(self.summary(belief), self.summary(fitted), subject)
        return SampleBelief(
            mean.mean,
            mean.n,
            mean.aux,
            belief.sample.at[subject].set(fitted.sample[subject]),
            belief.draws.at[subject].set(fitted.draws[subject]),
        )

    def install_summary(self, summary, fitted, subject):
        """Install the running mean of the subject."""
        return super().install(summary, fitted, subject)


def _as_i64(x):
    return jnp.asarray(x, I64)


PLAN_ROWS = 8


def _as_plan(plan):
    """A plan of (subject, channel id) pairs as an integer array of fixed length.

    Unused rows hold -1, so plans of different lengths share one compiled program.
    """
    rows = jnp.asarray(plan, I64).reshape(-1, 2)
    pad = jnp.full((max(PLAN_ROWS - rows.shape[0], 0), 2), -1, I64)
    return jnp.concatenate([rows, pad])


def _plan_length(plan):
    return jnp.sum(plan[:, 0] >= 0).astype(I64)


class NullProcess(AbstractProcess):
    """Process that seeds and writes nothing."""

    pass


class HeartbeatProcess(AbstractProcess):
    """A fixed number of periodic process events.

    Seed n_fires events at period_ticks * (i + 1) in the release stage. When writes is
    true, each event logs its fire number in an auxiliary record, released to the
    process audience.
    """

    period_ticks: jax.Array = eqx.field(converter=_as_i64)
    n_fires: int = eqx.field(static=True)
    audience: jax.Array = eqx.field(converter=_as_i64)
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
    start_tick=None,
):
    n = caps.n_effects

    def fit(v, dtype=I64):
        # a column of fewer rows than the capacity is completed with empty rows
        a = jnp.asarray(v, dtype)
        if a.ndim == 1 and a.shape[0] < n:
            a = jnp.concatenate([a, jnp.zeros(n - a.shape[0], dtype)])
        return a

    def col(v, dtype=I64):
        if np.ndim(v) == 0:
            return jnp.asarray(v, dtype) * jnp.ones(n, dtype)
        return fit(v, dtype)

    return EffectTable(
        kind=fit(kind),
        subject=col(subject),
        channel_index=col(channel_index),
        group=col(group),
        tick=col(tick),
        lane=col(lane),
        scope=col(scope),
        disposition=col(disposition),
        disruptive=col(disruptive, jnp.bool_),
        start_tick=col(0 if start_tick is None else start_tick)
        if caps.commitments
        else None,
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

    plan: jax.Array = eqx.field(converter=_as_plan)
    wait_at: jax.Array = eqx.field(default=-1, converter=_as_i64)
    wait_ticks: jax.Array = eqx.field(default=50, converter=_as_i64)
    bundle_at: jax.Array = eqx.field(default=-1, converter=_as_i64)
    bundle_ch: jax.Array = eqx.field(default=12, converter=_as_i64)
    cancel_at: jax.Array = eqx.field(default=-1, converter=_as_i64)
    cancel_gen_offset: jax.Array = eqx.field(default=0, converter=_as_i64)
    stop_at: jax.Array = eqx.field(default=-1, converter=_as_i64)
    stop_scope: jax.Array = eqx.field(default=Scope.LANE, converter=_as_i64)
    stop_disp: jax.Array = eqx.field(default=Disposition.CLOSE, converter=_as_i64)

    def declare(self):
        """The special stop and the final campaign drain this plan asks for."""
        stops = [(Scope.CAMPAIGN, Disposition.DRAIN)]
        if int(self.stop_at) >= 0:
            stops.append(
                (Scope(int(self.stop_scope)), Disposition(int(self.stop_disp)))
            )
        return Declaration(stops=tuple(stops))

    def decide(self, pstate, view, study, key):
        """Execute the next plan entry or a declared special effect."""
        caps = study.caps
        n = view.n_dec
        n_plan = _plan_length(self.plan)
        pos = pstate[0].astype(I64)
        subjects = jnp.maximum(self.plan[:, 0], 0)
        cids = jnp.maximum(self.plan[:, 1], 0)
        p = jnp.minimum(pos, jnp.maximum(n_plan - 1, 0))
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
        plan = [row for row in np.asarray(self.plan).tolist() if row[0] >= 0]
        wait_at, cancel_at = int(self.wait_at), int(self.cancel_at)
        stop_at, bundle_at = int(self.stop_at), int(self.bundle_at)
        wait_ticks = int(self.wait_ticks)
        cancel_gen_offset = int(self.cancel_gen_offset)
        stop_scope, stop_disp = int(self.stop_scope), int(self.stop_disp)
        bundle_ch = int(self.bundle_ch)
        kinds = [EffectKind.NONE] * caps.n_effects
        subjects = [0] * caps.n_effects
        channels = [0] * caps.n_effects
        ticks = [0] * caps.n_effects
        scopes = [0] * caps.n_effects
        disps = [0] * caps.n_effects
        ids = [int(i) for i in np.asarray(study.channels.id)]
        advance_plan = False
        if n == wait_at:
            kinds[0], ticks[0] = EffectKind.WAIT, int(view_np.tick) + wait_ticks
        elif n == cancel_at:
            kinds[0], ticks[0] = EffectKind.CANCEL, cancel_gen_offset
        elif n == stop_at:
            kinds[0], scopes[0], disps[0] = EffectKind.STOP, stop_scope, stop_disp
        elif pos >= len(plan):
            kinds[0], scopes[0], disps[0] = (
                EffectKind.STOP,
                Scope.CAMPAIGN,
                Disposition.DRAIN,
            )
        else:
            subject, cid = plan[pos]
            kinds[0], subjects[0], channels[0] = (
                EffectKind.OBSERVE,
                subject,
                ids.index(cid),
            )
            advance_plan = True
            if n == bundle_at:
                kinds[1], subjects[1], channels[1] = (
                    EffectKind.OBSERVE,
                    subject,
                    ids.index(bundle_ch),
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

    plan: jax.Array = eqx.field(converter=_as_plan)

    def decide(self, pstate, view, study, key):
        """Advance the plan when the snapshot gains visible rows."""
        caps = study.caps
        n_plan = _plan_length(self.plan)
        pos = pstate[0].astype(I64)
        fresh = (view.n_dec == 0) | (view.n_rows != pstate[1].astype(I64))
        subjects = jnp.maximum(self.plan[:, 0], 0)
        cids = jnp.maximum(self.plan[:, 1], 0)
        p = jnp.minimum(pos, jnp.maximum(n_plan - 1, 0))
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
    if not caps.payload_audience:
        caps = toy_registry().lower(caps)
    c = caps.n_channels
    channels = ChannelTable(
        id=jnp.asarray([10, 11, 12, 13][:c], I64),
        lane=jnp.asarray([0, 1, 2, 2][:c], I64),
        occupancy_ticks=jnp.asarray([100, 150, 50, 40][:c], I64),
        latency_ticks=jnp.asarray([20, 20, 0, 0][:c], I64),
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
        channel_model=ToyPhotometry(noise=jnp.asarray([0.5, 0.3, 0.8, 0.4][:c], F64)),
        lane_asset=jnp.asarray([0, 0, 1][: caps.n_lanes], I64),
        transition_ticks=jnp.asarray([10, 10, 5][: caps.n_lanes], I64),
        budget=budget,
        horizon_tick=jnp.asarray(1000, I64),
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


PRODUCT_CHANNELS = (
    90,
    91,
)  # observer and policy product channels of the refresh studies
PRODUCT_KIND = 3  # index of `REFRESH_PRODUCT_KIND` in `refresh_registry`


def refresh_registry():
    """The toy kinds and the derived refresh product kind, at indices 0 to 3."""
    return KindRegistry(
        (
            RecordKind("none", ()),
            MEASUREMENT_KIND,
            AUXILIARY_KIND,
            REFRESH_PRODUCT_KIND,
        )
    )


def survey_refresh(every_ticks=100, offset_ticks=0, *, observer=None, **overrides):
    """The five-acquisition toy study with a refresh schedule (a checkpoint grid).

    The observer defaults to `ToyRefitMean`, which changes only by refresh. The
    capacities carry the protocol (`rows_per_subject=8`, `refit_lanes=2`) and the
    registry lowered with the observer's refit kinds.
    """
    observer = ToyRefitMean() if observer is None else observer
    caps = overrides.pop("caps", None) or Capacities(rows_per_subject=8, refit_lanes=2)
    caps = refresh_registry().lower(caps, observer)
    schedule = RefreshSchedule(
        every_ticks, offset_ticks, *PRODUCT_CHANNELS, PRODUCT_KIND
    )
    return survey_1(caps, observer=observer, refresh=schedule, **overrides)


def toy_mean_kernel(study, entry):
    """The toy posterior mean in closed form: a masked kernel for `MaskedBatchRefit`.

    A flat-prior Gaussian posterior with known noise has the sample mean of the
    counting measurements as its mean, whatever the order they are summed in. Returns
    the subject slice `(mean, n)`, status `OK` and the row count as diagnostic.
    """
    observer, rows = study.observer, entry.rows
    take = (
        rows.valid
        & (rows.kind == observer.measurement_kind)
        & (rows.status == RecordStatus.COMPLETED)
    )
    n = jnp.sum(take).astype(I64)
    # Sum left to right, as `toy_refit_fit` does, so the two agree bit for bit.
    total, _ = jax.lax.scan(
        lambda t, x: (t + x, None),
        jnp.zeros((), F64),
        jnp.where(take, rows.payload[:, 0], 0.0),
    )
    mean = total / jnp.maximum(n, 1).astype(F64)
    diag = jnp.zeros(1, F64).at[0].set(n.astype(F64))
    return (mean, n), jnp.asarray(RefitStatus.OK, I64), diag


def toy_refit_fit(study, entry):
    """The same posterior as `toy_mean_kernel`, summed left to right on the host."""
    observer, rows = study.observer, entry.rows
    total, n = 0.0, 0
    for i in range(rows.valid.shape[0]):
        counts = (
            rows.valid[i]
            and rows.kind[i] == observer.measurement_kind
            and rows.status[i] == RecordStatus.COMPLETED
        )
        if counts:
            total += float(rows.payload[i, 0])
            n += 1
    mean = total / n if n else 0.0
    return (np.float64(mean), np.int64(n)), RefitStatus.OK, [float(n)]


def toy_masked_refit():
    """The toy refit as an array kernel: a `MaskedBatchRefit` of `toy_mean_kernel`."""
    return MaskedBatchRefit(toy_mean_kernel)


def toy_refit_service():
    """The toy refit as a host loop: a `HostLoopRefit` of `toy_refit_fit`."""
    return HostLoopRefit(toy_refit_fit)


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
