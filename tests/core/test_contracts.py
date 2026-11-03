"""Construction guards and typed composition contracts."""

import dataclasses
import subprocess
import sys
import typing
from collections.abc import Callable
from enum import IntEnum, IntFlag

import jax
import pytest

from spaceodyssey import core
from spaceodyssey.studies.toy import survey_1
from tests.core.contract_helpers import (
    assert_study_typed,
    assert_typed_fields,
    is_untyped,
)


@pytest.fixture(scope="module")
def composed_study():
    return survey_1()


def test_x64_guard(composed_study):
    study = composed_study
    with jax.enable_x64(False):
        with pytest.raises(RuntimeError, match="requires JAX x64"):
            core.Capacities()
        with pytest.raises(RuntimeError, match="requires JAX x64"):
            dataclasses.replace(study)
        for executor in (core.CompiledExecutor(), core.HostExecutor()):
            with pytest.raises(RuntimeError, match="requires JAX x64"):
                executor.run(study, None, None, 1)


def test_import_does_not_enable_x64():
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            'import jax; jax.config.update("jax_enable_x64", False); '
            "import spaceodyssey.core; assert not jax.config.x64_enabled",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr


def test_study_has_no_untyped_fields():
    assert_study_typed()


_ALIAS = getattr(typing, "TypeAliasType", None)
ESCAPES = [
    object,
    typing.Any,
    Callable,
    typing.Callable,
    list[typing.Any],
    Callable[[typing.Any], int],
    typing.TypeVar("Unbounded"),
    typing.TypeVar("Hidden") | None,
    tuple,
    dict,
    list,
    set,
    frozenset,
    tuple,
    dict,
    tuple[dict, ...],
    typing.TypeVar("Loose", bound=object),
    typing.TypeVar("Wide", int, object),
]
if _ALIAS is not None:
    ESCAPES.append(_ALIAS("Alias", object))
    ESCAPES.append(_ALIAS("Nested", list[_ALIAS("Inner", typing.Any)]))


@pytest.mark.parametrize("annotation", ESCAPES, ids=repr)
def test_untyped_fields_are_rejected(annotation):
    assert is_untyped(annotation)
    cls = dataclasses.make_dataclass("AbstractUntyped", [("value", annotation)])
    with pytest.raises(AssertionError, match="untyped field"):
        assert_typed_fields({cls})


@pytest.mark.parametrize(
    "annotation",
    [
        int,
        tuple[int, ...],
        dict[str, float],
        frozenset[int],
        typing.TypeVar("Bounded", bound=int),
        typing.TypeVar("Constrained", int, float),
    ],
    ids=repr,
)
def test_typed_fields_are_accepted(annotation):
    assert not is_untyped(annotation)
    cls = dataclasses.make_dataclass("AbstractTyped", [("value", annotation)])
    assert_typed_fields({cls})


def test_plain_abstract_fields_need_annotations():
    class AbstractMissing:
        value = 0

    with pytest.raises(AssertionError, match="missing annotation"):
        assert_typed_fields({AbstractMissing})


def test_integer_vocabularies():
    for cls in (
        core.Stage,
        core.EventKind,
        core.EffectKind,
        core.Scope,
        core.Disposition,
        core.RecordStatus,
        core.Purpose,
        core.AdmissionCode,
    ):
        assert issubclass(cls, IntEnum)
        assert list(map(int, cls)) == list(range(len(cls)))
    assert issubclass(core.Audience, IntFlag)
    assert int(core.Audience.OBSERVER | core.Audience.POLICY) == 3


def test_collection_preserves_callers_x64_setting():
    code = (
        'import jax, pytest; jax.config.update("jax_enable_x64", False); '
        'result = pytest.main(["--collect-only", "-q", "-p", "no:cacheprovider", '
        '"tests/core/test_keys.py", "tests/studies/test_toy.py"]); '
        "assert result == 0; assert not jax.config.x64_enabled"
    )
    result = subprocess.run(
        [sys.executable, "-c", code],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("ids", [[], [0], [0, 0], [0, 2]])
def test_gate_rejects_missing_duplicate_or_wrong_worlds(monkeypatch, ids):
    from types import SimpleNamespace

    from tests import test_slice1_gate as gate

    returned = [SimpleNamespace(world=world) for world in ids]
    monkeypatch.setattr(gate, "runs", lambda *args: returned)
    checks = [
        SimpleNamespace(name=name, passed=True) for name in gate.EXPECTED_INVARIANTS
    ]
    monkeypatch.setattr(gate, "check_run", lambda run: checks)
    with pytest.raises(AssertionError, match="world"):
        gate.test_corpus_invariants("K3", gate.CORPUS_BY_NAME["baseline"], "host")


@pytest.mark.parametrize("annotation", [object, typing.Any, Callable, int | object])
def test_composed_agenda_escape_is_rejected(monkeypatch, annotation):
    from spaceodyssey.core.study import Agenda

    monkeypatch.setattr(
        Agenda, "__annotations__", {**Agenda.__annotations__, "escape": annotation}
    )
    with pytest.raises(AssertionError, match="untyped field"):
        assert_study_typed()


@pytest.mark.parametrize(
    "container",
    [lambda t: t, lambda t: list[t], lambda t: tuple[t, ...], lambda t: dict[str, t]],
)
def test_concrete_field_graph_is_checked_recursively(container):
    leaf = dataclasses.make_dataclass("ConcreteLeaf", [("escape", object)])
    middle = dataclasses.make_dataclass(
        "ConcreteMiddle", [("children", container(leaf))]
    )
    root = dataclasses.make_dataclass("CompositionRoot", [("nested", middle)])
    with pytest.raises(AssertionError, match="untyped field"):
        assert_typed_fields({root})


@pytest.mark.parametrize("executor", ["host", "scan"])
@pytest.mark.parametrize("mutant", ["measurement", "policy"])
def test_draw_control_rejects_nan_only_corruption(monkeypatch, executor, mutant):
    import equinox as eqx
    import jax.numpy as jnp

    from tests import test_slice1_gate as gate
    from tests.core import mutants as m

    policy = mutant == "policy"
    host, scan = (
        (m.CountKeyHostExecutor, m.CountKeyCompiledExecutor)
        if policy
        else (m.MutantHostExecutor, m.MutantCompiledExecutor)
    )
    if policy:
        monkeypatch.setattr(host, "decide", core.HostExecutor.decide)
    else:
        monkeypatch.setattr(host, "advance", core.HostExecutor.advance)
    monkeypatch.setattr(scan, "transition", core.CompiledExecutor.transition)

    def corrupt(original):
        def run(self, *args, **kwargs):
            result = original(self, *args, **kwargs)
            if policy:
                trace = eqx.tree_at(
                    lambda t: t.pol_u,
                    result.trace,
                    jnp.full_like(result.trace.pol_u, jnp.nan),
                )
                return core.RunResult(result.carry, trace)
            payload = result.carry.log.payload.at[..., 1].set(jnp.nan)
            carry = eqx.tree_at(lambda c: c.log.payload, result.carry, payload)
            return core.RunResult(carry, result.trace)

        return run

    monkeypatch.setattr(host, "run", corrupt(core.HostExecutor.run))
    monkeypatch.setattr(scan, "run", corrupt(core.CompiledExecutor.run))
    with pytest.raises(AssertionError, match="finite"):
        gate.test_negative_decision_count_keys(executor, mutant)


def test_gate_rejects_broken_replay(monkeypatch):
    from tests import test_slice1_gate as gate
    from tests.core import test_replay_log

    check = getattr(gate, "test_replay_acceptance", None)
    assert callable(check), "gate must run A2 replay"
    monkeypatch.setattr(
        test_replay_log,
        "replay_belief",
        lambda study, carry, trace: study.observer.init(study.caps),
    )
    with pytest.raises(AssertionError):
        check("A2", gate.CORPUS_BY_NAME["baseline"])


@pytest.mark.parametrize(
    "name, check_id",
    [("test_fork_acceptance", "A3"), ("test_latency_acceptance", "A4")],
)
def test_gate_runs_fork_and_latency(name, check_id):
    from tests import test_slice1_gate as gate

    check = getattr(gate, name, None)
    assert callable(check), f"gate must run {check_id}"
    assert any(
        mark.name == "parametrize" and check_id in mark.args[1]
        for mark in check.pytestmark
    )


@pytest.mark.parametrize("policy", [False, True])
def test_draw_controls_require_multiple_shared_identities(policy):
    from tests.core import draw_helpers as draws

    compare = draws.compare_policy_draws if policy else draws.compare_draws
    with pytest.raises(AssertionError, match="matched identities"):
        compare([], [])


def test_typed_graph_handles_cycles_and_static_fields():
    leaf = dataclasses.make_dataclass(
        "StaticLeaf", [("escape", object, dataclasses.field(metadata={"static": True}))]
    )
    root = dataclasses.make_dataclass("CyclicRoot", [("value", int)])
    root.__annotations__["cycle"] = root
    root.__annotations__["static_children"] = tuple[leaf, ...]
    with pytest.raises(AssertionError, match="untyped field"):
        assert_typed_fields({root})
