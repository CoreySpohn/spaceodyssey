"""The compiled program stays small: loops are rolled and shared bodies traced once."""

import dataclasses

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey import core
from spaceodyssey.core import handlers
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import EffectKind, EventKind, Stage
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.studies import toy
from tests.core.helpers import after_decision_1_bundle, one_world, worlds


def _equations(jaxpr):
    total = 0
    for eqn in jaxpr.eqns:
        total += 1
        for value in eqn.params.values():
            for item in value if isinstance(value, (list, tuple)) else [value]:
                inner = getattr(item, "jaxpr", item)
                if hasattr(inner, "eqns"):
                    total += _equations(inner)
    return total


def _admit_equations(n_effects):
    study = toy.survey_1(caps=Capacities(n_effects=n_effects))
    truth, _ = one_world(caps=study.caps)
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    effects = core.effects_none(study.caps)
    return _equations(eqx.filter_make_jaxpr(core.admit)(carry, effects, study)[0].jaxpr)


def test_admission_program_does_not_grow_with_the_effect_rows():
    # unrolled over rows, each added row repeated the whole per-row body
    assert _admit_equations(5) - _admit_equations(2) < 100


def _switch_sizes(jaxpr):
    sizes = []
    for eqn in jaxpr.eqns:
        if eqn.primitive.name == "cond":
            sizes.append(len(eqn.params["branches"]))
        for value in eqn.params.values():
            inner = getattr(value, "jaxpr", None)
            if inner is not None and hasattr(inner, "eqns"):
                sizes += _switch_sizes(inner)
    return sizes


def _switch_branches(awards, commitments=False):
    caps = Capacities(
        n_programs=2 if awards else 1,
        awards=awards,
        n_commitments=2 if commitments else 0,
    )
    study = toy.survey_1(caps=caps)
    truth, root = one_world(caps=caps)
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    ev = jax.tree_util.tree_map(lambda a: a[0], carry.ev)
    jaxpr = eqx.filter_make_jaxpr(handlers.step)(carry, ev, study, truth, root)[0]
    return _switch_sizes(jaxpr.jaxpr)


@pytest.mark.parametrize(
    ("awards", "commitments", "branches"),
    ((False, False, 9), (True, False, 10), (False, True, 10), (True, True, 11)),
)
def test_step_holds_each_distinct_handler_body_once(awards, commitments, branches):
    # eleven kinds, twelve with awards, thirteen with commitments; RESULT and FAULT
    # share a body with RELEASE and CANCEL, and the window kind is inert (the body of
    # NONE) where only commitments are on
    assert _switch_branches(awards, commitments) == [branches]


SHARED = (
    (EventKind.RESULT, handlers.h_release, {"slot": 0, "audience": 3}),
    (EventKind.FAULT, handlers.h_fault, {"lane": 0, "gen": 1}),
    (EventKind.CANCEL, handlers.h_cancel, {"lane": 0, "gen": 1}),
)


@pytest.mark.medium(
    reason="one jitted program runs the shared branches against their handlers"
)
def test_shared_branch_matches_the_handler_of_its_own_kind():
    study, carry = after_decision_1_bundle()
    truth, root = one_world()

    @eqx.filter_jit
    def both(c):
        out = []
        for kind, shared, fields in SHARED:
            c1 = core.enqueue(
                c,
                True,
                kind=kind,
                tick=c.tick,
                stage=Stage.ADMIT,
                caps=study.caps,
                **{k: jnp.asarray(v, jnp.int64) for k, v in fields.items()},
            )
            c1, ev, _ = core.select_next(c1, study.caps)
            out.append(
                (
                    core.step(c1, ev, study, truth, root),
                    shared(c1, ev, study, truth, root),
                )
            )
        return out

    for via_step, direct in both(carry):
        report = core.tree_equal_report(via_step, direct)
        assert report["max_int_diff"] == 0 and report["max_abs"] == 0.0


@pytest.mark.medium(reason="one compiled run per effect table width")
@pytest.mark.parametrize("n_effects", (3, 4, 5))
def test_toy_study_runs_with_a_wider_effect_table(n_effects):
    caps = Capacities(n_effects=n_effects)
    study = toy.survey_1(caps=caps)
    truths, roots = worlds(1, caps)
    result = CompiledExecutor().run(study, truths, roots, 12)
    assert bool(jnp.all(result.trace.eff_kind[..., 2:] == EffectKind.NONE))
    assert bool(jnp.any(result.trace.eff_kind[..., 0] == EffectKind.OBSERVE))


@pytest.mark.parametrize("rows", (1, 2, 4))
def test_admit_refuses_a_table_of_the_wrong_width(rows):
    study = toy.survey_1(caps=Capacities(n_effects=3))
    truth, _ = one_world(caps=study.caps)
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    wrong = core.effects_none(Capacities(n_effects=rows))
    with pytest.raises(ValueError, match=r"n_effects is 3.*exactly n_effects rows"):
        core.admit(carry, wrong, study)


class _OneRow(toy.FixedPlanPolicy):
    def decide(self, pstate, view, study, key):
        eff, ps, u = super().decide(pstate, view, study, key)
        return jax.tree_util.tree_map(lambda a: a[:1], eff), ps, u


def test_transition_names_the_policy_that_emits_a_short_table():
    caps = Capacities(n_effects=3)
    study = toy.survey_1(caps=caps, policies=(_OneRow(plan=((0, 10),)),))
    truths, roots = worlds(1, caps)
    with pytest.raises(ValueError, match=r"n_effects is 3.*policy: .*_OneRow"):
        CompiledExecutor().run(study, truths, roots, 4)


@pytest.mark.parametrize("rows", (1, 2, 4))
def test_admit_refuses_a_program_column_of_the_wrong_width(rows):
    caps = Capacities(n_programs=2, n_effects=3)
    study = toy.survey_1(caps=caps)
    truth, _ = one_world(caps=caps)
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    table = core.effects_none(caps)
    table = dataclasses.replace(table, program=jnp.ones(rows, jnp.int64))
    with pytest.raises(ValueError, match=r"column 'program'.*n_effects is 3"):
        core.admit(carry, table, study)


def test_a_short_policy_is_named_beside_a_correct_one():
    caps = Capacities(n_effects=3)
    good = toy.FixedPlanPolicy(plan=((0, 10),))
    study = toy.survey_1(caps=caps, policies=(good, _OneRow(plan=((0, 10),))))
    truths, roots = worlds(1, caps)
    with pytest.raises(ValueError, match=r"n_effects is 3.*policy: .*_OneRow"):
        CompiledExecutor().run(study, truths, roots, 4)
