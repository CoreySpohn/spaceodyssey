"""The compiled program stays small: loops are rolled and shared bodies traced once."""

import equinox as eqx
import jax
import jax.numpy as jnp
import pytest

from spaceodyssey import core
from spaceodyssey.core import handlers
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import EventKind, Stage
from spaceodyssey.studies import toy
from tests.core.helpers import after_decision_1_bundle, one_world


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


def _switch_branches(awards):
    caps = Capacities(n_programs=2, awards=True) if awards else Capacities()
    study = toy.survey_1(caps=caps)
    truth, root = one_world(caps=caps)
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    ev = jax.tree_util.tree_map(lambda a: a[0], carry.ev)
    jaxpr = eqx.filter_make_jaxpr(handlers.step)(carry, ev, study, truth, root)[0]
    return _switch_sizes(jaxpr.jaxpr)


@pytest.mark.parametrize("awards", (False, True))
def test_step_holds_each_distinct_handler_body_once(awards):
    # eleven kinds, twelve with awards; RESULT and FAULT share a body with RELEASE
    # and CANCEL
    assert _switch_branches(awards) == [10 if awards else 9]


SHARED = (
    (EventKind.RESULT, handlers.h_release, {"slot": 0, "audience": 3}),
    (EventKind.FAULT, handlers.h_fault, {"lane": 0, "gen": 1}),
    (EventKind.CANCEL, handlers.h_cancel, {"lane": 0, "gen": 1}),
)


@pytest.mark.medium
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
