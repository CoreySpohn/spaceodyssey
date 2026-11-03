"""Battery checks reject hidden differences and unexercised runs."""

import dataclasses

import jax.numpy as jnp

from spaceodyssey.core import battery, handlers, preflight
from spaceodyssey.core.declarations import Declaration, Need
from spaceodyssey.core.enums import F64, Disposition, EffectKind, Scope
from spaceodyssey.core.preflight import FindingCode
from spaceodyssey.studies import toy
from tests.core.battery_helpers import (
    PLAN,
    AbsolutePlanPolicy,
    PriorBiasedMean,
    StatedPlan,
    _compiled_with,
    _table,
    batched_truth,
    survey,
    unreleased_view_executor,
)
from tests.core.helpers import worlds


class CancelNeedPolicy(toy.NullPolicy):
    def declare(self):
        return Declaration(
            stops=((Scope.CAMPAIGN, Disposition.DRAIN),), needs=(Need(10, 10, 80),)
        )

    def decide(self, pstate, view, study, key):
        kind = jnp.where(
            view.n_dec == 0,
            EffectKind.OBSERVE,
            jnp.where(view.n_dec == 1, EffectKind.CANCEL, EffectKind.STOP),
        )
        eff = _table(
            study.caps,
            kind,
            jnp.where(view.n_dec == 0, EffectKind.WAIT, EffectKind.NONE),
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
            channel_index=0,
            subject=0,
        )
        return (
            dataclasses.replace(eff, tick=eff.tick.at[1].set(60)),
            pstate,
            jnp.asarray(0.0, F64),
        )


class ReleaseAllocatePolicy(AbsolutePlanPolicy):
    def decide(self, pstate, view, study, key):
        eff, ps, u = super().decide(pstate, view, study, key)
        eff = dataclasses.replace(
            eff,
            kind=eff.kind.at[0].set(
                jnp.where(view.tick % 150 == 132, EffectKind.ALLOCATE, eff.kind[0])
            ),
        )
        return eff, ps, u


class AbortBeforeReleasePolicy(toy.NullPolicy):
    def decide(self, pstate, view, study, key):
        first = view.n_dec == 0
        eff = _table(
            study.caps,
            jnp.where(first, EffectKind.OBSERVE, EffectKind.STOP),
            jnp.where(first, EffectKind.WAIT, EffectKind.NONE),
            scope=Scope.CAMPAIGN,
            disposition=Disposition.ABORT,
        )
        eff = dataclasses.replace(eff, tick=eff.tick.at[1].set(120))
        return eff, pstate, jnp.asarray(0.0, F64)


class HiddenFirstModel(toy.ToyPhotometry):
    def realize(self, truth, subject, channel_index, fraction, key, design, channels):
        value = jnp.where(channel_index == 0, 0.0, truth.signal[subject])
        return value, truth.signal[subject]


class TwoReleasePolicy(toy.NullPolicy):
    def decide(self, pstate, view, study, key):
        scheduled = (view.tick == 0) | (view.tick == 150)
        read = jnp.where(
            view.log.payload[:, 0].sum() > 1.0, EffectKind.NONE, EffectKind.STOP
        )
        kind = jnp.where(
            scheduled,
            EffectKind.OBSERVE,
            jnp.where(
                view.tick == 320,
                read,
                jnp.where(view.tick == 330, EffectKind.STOP, EffectKind.NONE),
            ),
        )
        eff = _table(
            study.caps,
            kind,
            jnp.where(scheduled, EffectKind.WAIT, EffectKind.NONE),
            channel_index=jnp.where(view.tick == 0, 0, 1),
            subject=0,
            scope=Scope.CAMPAIGN,
            disposition=Disposition.DRAIN,
            group=view.n_dec,
        )
        eff = dataclasses.replace(
            eff, tick=eff.tick.at[1].set(jnp.where(view.tick == 0, 150, 320))
        )
        return eff, pstate, jnp.asarray(0.0, F64)


def test_truth_cutoff_ignores_scorer_only_differences():
    study = dataclasses.replace(
        survey(TwoReleasePolicy()),
        channel_model=HiddenFirstModel(noise=jnp.zeros(3, F64)),
    )
    roots = worlds(1)[1]
    a, b = batched_truth(0.0, 1), batched_truth(5.0, 1)
    assert battery.truth_swap(study, a, b, roots, 20)
    result = battery.truth_swap(
        study, a, b, roots, 20, executor=unreleased_view_executor()
    )
    assert not result
    assert "eff_kind" in result.diagnostic


def test_latency_compares_effects_at_release_decisions():
    truths, roots = worlds(1)
    result = battery.latency_shift(
        survey(ReleaseAllocatePolicy()), truths, roots, 40, delay_s=2.0
    )
    assert not result
    assert "requested effects changed" in result.diagnostic
    assert "decision admit_code changed" in result.diagnostic


def test_null_requires_a_finished_run_with_a_fold():
    study = dataclasses.replace(survey(), observer=PriorBiasedMean())
    result = battery.null_reference(
        study, batched_truth(0.0, 1), worlds(1)[1], 2, expected=0.0
    )
    assert not result


def test_finished_null_run_requires_a_folded_record():
    result = battery.null_reference(
        survey(AbortBeforeReleasePolicy()),
        batched_truth(0.0, 1),
        worlds(1)[1],
        20,
        expected=0.0,
    )
    assert not result
    assert result.detail["n_records"] == [1]
    assert result.detail["n_folded"] == [0]
    assert "folded no record" in result.diagnostic
    assert "did not complete" not in result.diagnostic


def test_fork_prefix_compares_record_costs():
    def realize(c, study, truth, root, base, fraction):
        rec = handlers._realize_rec(c, study, truth, root, base, fraction)
        bad = (study.fork.branch != 0) & (
            base.acquisition_index < study.fork.from_index
        )
        return dataclasses.replace(rec, cost=rec.cost + bad.astype(F64))

    truths, roots = worlds(1)
    result = battery.fork_check(
        survey(), truths, roots, 40, executor=_compiled_with(realize=realize)
    )
    assert not result


def test_partial_record_can_meet_a_need_before_full_occupancy_ends():
    study = survey(CancelNeedPolicy())
    truths, roots = worlds(1)
    assert battery.declared_matches_run(study, truths, roots, 15)
    assert not preflight(study)


def test_causality_bound_uses_the_requested_channels_lane():
    decl = Declaration(needs=(Need(10, 100, 80),))
    study = survey(StatedPlan(plan=PLAN, stated=decl))
    study = dataclasses.replace(
        study,
        assets=dataclasses.replace(
            study.assets,
            transition_ticks=study.assets.transition_ticks.at[1].set(500),
        ),
    )
    assert any(f.code == FindingCode.C001 for f in preflight(study))


def test_declared_need_requires_a_decision_in_its_own_phase():
    decl = Declaration(
        stops=((Scope.CAMPAIGN, Disposition.DRAIN),), needs=(Need(10, 10, 130),)
    )
    study = toy.survey_1(
        policies=(StatedPlan(plan=PLAN, stated=decl), StatedPlan(plan=PLAN)),
        phase_until=(100, 1000),
    )
    truths, roots = worlds(1)
    result = battery.declared_matches_run(study, truths, roots, 40)
    assert not result
    assert "phase 0 declares a need" in result.diagnostic
