"""Studies that differ only in numbers, or in step count, share one compiled program.

A number a policy or process holds is lifted to an array where the study enters a
compiled call, so a new value reuses the program; the bucketed run takes every step
count of a bucket through one program; names that are never read under trace stay out of
the compile key. None of it changes a history: the lifted and the bucketed runs are
compared with the default scan bit for bit, and the objects and documents are unchanged.
"""

import dataclasses
import hashlib
import math
from decimal import Decimal
from fractions import Fraction

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import DocumentError, dump_study, load_study
from spaceodyssey.codec.records import check_lowering
from spaceodyssey.core.conformance.corpus import make_worlds
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.executors.compiled import CompiledExecutor, _scan_worlds
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.lift import _lifted, lift_traced, lifting_jit
from spaceodyssey.core.transaction import wake_by_audience
from spaceodyssey.studies import oracle, scripted, toy
from tests.codec.test_document import TOY_KINDS
from tests.core.corpus import HAND_STUDIES

pytestmark = pytest.mark.medium  # Each case compiles a scan.

_traces = {"n": 0}


class _CountingExecutor(CompiledExecutor):
    """Counts how often a scan body is traced, which is once per compiled program."""

    def transition(self, carry, i, root, study, truth, *, wake_fn=None):
        _traces["n"] += 1
        kw = {} if wake_fn is None else {"wake_fn": wake_fn}
        return super().transition(carry, i, root, study, truth, **kw)


def _run(study, n_steps, n_worlds=2, **kw):
    truths, roots = make_worlds(study, n_worlds)
    return _CountingExecutor().run(study, truths, roots, n_steps, **kw)


def _same_bits(a, b):
    la, lb = map(jax.tree_util.tree_leaves, (a, b))
    assert len(la) == len(lb)
    for x, y in zip(la, lb, strict=True):
        if isinstance(x, jax.Array):
            assert x.weak_type == y.weak_type
        x, y = np.asarray(x), np.asarray(y)
        assert (x.dtype, x.shape) == (y.dtype, y.shape)
        assert x.tobytes() == y.tobytes()


def _scripted(values=(2.0, 9.0), ticks=(100, 200)):
    return scripted.scripted_study(
        ticks=ticks, values=values, ids=(7001, 7001), revisions=(0, 1)
    )


BUCKET = 16


@pytest.mark.parametrize("name", ["baseline", "bundle"])
@pytest.mark.parametrize("n_steps", [12, 40, 45])
def test_bucketed_run_matches_the_default_scan_bit_for_bit(name, n_steps):
    study = HAND_STUDIES[name]()
    default = _run(study, n_steps)
    bucketed = _run(study, n_steps, bucket=BUCKET)
    _same_bits(default.carry, bucketed.carry)
    _same_bits(default.trace, bucketed.trace)


def test_bucketed_run_matches_on_a_scripted_study():
    study = _scripted()
    default = _run(study, 45)
    bucketed = _run(study, 45, bucket=BUCKET)
    _same_bits((default.carry, default.trace), (bucketed.carry, bucketed.trace))


def test_bucketed_run_shares_one_program_within_a_bucket():
    study = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=2.5),))
    _run(study, 20, bucket=BUCKET)
    before = _traces["n"]
    for n_steps in (17, 25, 32):
        _run(study, n_steps, bucket=BUCKET)
    assert _traces["n"] == before


def test_array_fields_reuse_the_program_and_are_read():
    never = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=math.inf),))
    first = _run(never, 30)
    before = _traces["n"]
    at_once = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=-math.inf),))
    result = _run(at_once, 30)
    assert _traces["n"] == before
    # the second value is read, not the first one baked into the cached program
    assert int(result.carry.n_dec.max()) < int(first.carry.n_dec.max())


def test_marked_numbers_stay_python_numbers_and_are_lifted_for_the_call_only():
    policy = toy.StochasticPolicy(theta=0.3, n_obs=6, wait_ticks=12)
    study = toy.survey_1(policies=(policy,))
    assert type(policy.n_obs) is int and type(policy.wait_ticks) is int
    lifted = lift_traced(study).agenda.policies[0]
    assert lifted.n_obs.dtype == jnp.int64 and lifted.wait_ticks.dtype == jnp.int64
    assert study.agenda.policies[0].n_obs == 6
    stop = lift_traced(toy.ThresholdStopPolicy(stop_snr=3))
    assert stop.stop_snr.dtype == jnp.float64
    script = lift_traced(_scripted().process)
    assert script.ticks.dtype == jnp.int64 and script.values.dtype == jnp.float64
    assert script.kinds.shape == (0,)
    # nothing marked, nothing rebuilt
    plain = toy.survey_1()
    assert lift_traced(plain.agenda) is plain.agenda


_LIFTED_STUDIES = {
    "threshold": lambda: toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=1),)),
    "stochastic": lambda: toy.survey_1(
        policies=(toy.StochasticPolicy(theta=0.3, wait_ticks=12),)
    ),
    "scripted": lambda: scripted.scripted_study(
        ticks=(100, 200), values=(2, 9), ids=(7001, 7001), revisions=(0, 1)
    ),
}


@pytest.mark.parametrize("name", sorted(_LIFTED_STUDIES))
def test_lifting_does_not_change_the_history(name):
    """Marked numbers lifted for the call give the carry and trace of the baked ones."""
    study = _LIFTED_STUDIES[name]()
    truths, roots = make_worlds(study, 2)
    baked = _scan_worlds(CompiledExecutor(), study, truths, roots, wake_by_audience, 40)
    lifted = CompiledExecutor().run(study, truths, roots, 40)
    _same_bits(baked, (lifted.carry, lifted.trace))


def test_scripts_with_other_values_reuse_the_program():
    _run(_scripted(), 30)
    before = _traces["n"]
    _run(_scripted(values=(3.0, 5.0), ticks=(110, 230)), 30)
    assert _traces["n"] == before


def test_kind_names_stay_out_of_the_compile_key():
    study = toy.survey_1()
    renamed = toy.survey_1(
        caps=dataclasses.replace(
            study.caps, kind_names=tuple(n + "x" for n in study.caps.kind_names)
        )
    )
    assert renamed.caps.kind_names != study.caps.kind_names
    assert renamed.caps == study.caps and hash(renamed.caps) == hash(study.caps)
    _run(study, 30)
    before = _traces["n"]
    _run(renamed, 30)
    assert _traces["n"] == before


def test_a_registry_that_names_kinds_otherwise_still_fails_the_lowering_check():
    study = toy.survey_1()
    renamed = dataclasses.replace(
        study,
        caps=dataclasses.replace(
            study.caps, kind_names=tuple(n + "x" for n in study.caps.kind_names)
        ),
    )
    with pytest.raises(DocumentError, match="does not lower"):
        check_lowering(toy.toy_registry(), renamed)


KINDS = {**TOY_KINDS, ("process", "scripted.records", 1): scripted.ScriptedRecords}


def test_documents_are_unchanged():
    """The fields stay Python numbers, so an earlier document keeps its hash."""
    study = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=3),))
    doc = dump_study(study, TOY_KINDS)
    assert doc.content_hash == (
        "2d7d2831f3066b73c0a1741ea3bd19e82ce1375cc3b36b26ed42e2a6b808ea9f"
    )
    assert load_study(doc.to_json(), TOY_KINDS).agenda.policies[0].stop_snr == 3


def test_a_marked_policy_can_be_built_inside_a_transform():
    def gap(snr):
        return toy.ThresholdStopPolicy(stop_snr=snr).stop_snr - 1.0

    assert float(jax.jit(gap)(2.0)) == 1.0


@pytest.mark.parametrize("bucket", [0, -16, 2.5, True, "16"])
def test_a_bucket_must_be_a_positive_int(bucket):
    with pytest.raises(ValueError, match="bucket"):
        _run(toy.survey_1(), 17, bucket=bucket)


def _traces_of(monkeypatch, module, name, calls):
    """Count traces of the jitted function `module.name` over the callables `calls`."""
    original = getattr(module, name).__wrapped__
    count = {"n": 0}

    def counted(*args, **kw):
        count["n"] += 1
        return original(*args, **kw)

    monkeypatch.setattr(module, name, lifting_jit(counted))
    seen = []
    for call in calls:
        call()
        seen.append(count["n"])
    return seen


def _threshold(x):
    return toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=x),))


def test_replay_entry_points_share_one_program_across_values(monkeypatch):
    from spaceodyssey.core.conformance import observer_kit
    from spaceodyssey.core.executors import replay

    study = _threshold(3.0)
    truths, roots = make_worlds(study, 1)
    out = CompiledExecutor().run(study, truths, roots, 40)
    carry = index_world(out.carry, 0)
    trace = index_world(out.trace, 0)
    values = (3.0, 4.0, 5.0)
    seen = _traces_of(
        monkeypatch,
        replay,
        "_fold",
        [
            lambda x=x: replay.replay_belief(_threshold(x), carry, trace, roots[0])
            for x in values
        ],
    )
    assert seen == [1, 1, 1]
    seen = _traces_of(
        monkeypatch,
        observer_kit,
        "_replay",
        [
            lambda x=x: observer_kit.replay_beliefs(_threshold(x), out, roots)
            for x in values
        ],
    )
    assert seen == [1, 1, 1]


def test_masked_refit_shares_one_program_across_values(monkeypatch):
    from spaceodyssey.core import refresh
    from tests.core.refresh_helpers import SpyBoundary

    def study(values):
        return scripted.scripted_refresh_study(
            ticks=(100, 200), values=values, ids=(1, 2)
        )

    base = study((2.0, 9.0))
    truths, roots = make_worlds(base, 1)
    spy = SpyBoundary(toy.toy_masked_refit())
    CompiledExecutor().run(base, truths, roots, 40, refit=spy)
    requests = jax.tree.map(jnp.asarray, spy.calls[0])
    boundary = toy.toy_masked_refit()
    seen = _traces_of(
        monkeypatch,
        refresh,
        "_masked_call",
        [
            lambda v=v: boundary.refit_batch(study(v), requests)
            for v in ((2.0, 9.0), (3.0, 8.0), (4.0, 7.0))
        ],
    )
    assert seen == [1, 1, 1]


class _FloatSubclass(float):
    pass


_I64 = jnp.int64
_F64 = jnp.float64

# What lifting does with each input: a number it can prove exact becomes an array of the
# dtype; every other value comes back as the very object it was, as it was handed on
# before lifting existed.
_LIFTED = [
    (_I64, True, 1),
    (_I64, 2, 2),
    (_I64, 2**63 - 1, 2**63 - 1),
    (_I64, (2, 3), [2, 3]),
    (_I64, [2, 3], [2, 3]),
    (_I64, (2**53 + 1, 2), [2**53 + 1, 2]),
    (_I64, (), []),
    (_F64, 2.5, 2.5),
    (_F64, 2, 2.0),
    (_F64, True, 1.0),
    (_F64, 2**53, 2.0**53),
    (_F64, (2, 3.5), [2.0, 3.5]),
    (_F64, _FloatSubclass(2.0), 2.0),
]
_PASSED_THROUGH = [
    (_I64, 2**63),
    (_I64, -(2**63) - 1),
    (_I64, 2.0),
    (_I64, 1.5),
    (_I64, math.nan),
    (_I64, np.int64(2)),
    (_I64, np.uint64(2)),
    (_I64, np.float32(2)),
    (_I64, Fraction(2)),
    (_I64, Decimal("2")),
    (_I64, ((1, 2), (3, 4))),
    (_I64, [[1], [2, 3]]),
    (_I64, (2, 3.0)),
    (_I64, (2**53 + 1, 2.0, "x")),
    (_I64, np.empty((0, 3))),
    (_I64, np.array([1, 2], object)),
    (_I64, np.array([2**63, 2], np.uint64)),
    (_I64, np.array([1.5, 2], np.float32)),
    (_I64, jnp.asarray(2, jnp.int32)),
    (_I64, jnp.asarray(2.0, jnp.float32)),
    (_F64, 2**53 + 1),
    (_F64, 10**400),
    (_F64, 2**63),
    (_F64, np.float32(1.5)),
    (_F64, Fraction(3, 2)),
    (_F64, Decimal("1.5")),
    (_F64, ((1, 2), (3.0, 4))),
    (_F64, (2**53 + 1, 2.0)),
    (_F64, np.array([np.float16(2), np.float32(3)], object)),
    (_F64, np.empty((2, 0, 3))),
    (_F64, jnp.asarray(2**53 + 1, jnp.int64)),
    (_F64, jnp.asarray(2.0, jnp.float32)),
]


@pytest.mark.parametrize(("dtype", "value", "expected"), _LIFTED)
def test_lifting_converts_a_number_it_can_prove_exact(dtype, value, expected):
    out = _lifted(value, dtype, isinstance(value, tuple | list))
    assert isinstance(out, jax.Array) and out.dtype == dtype
    assert out.tolist() == expected


@pytest.mark.parametrize(("dtype", "value"), _PASSED_THROUGH)
def test_lifting_hands_on_anything_else_untouched(dtype, value):
    assert _lifted(value, dtype, isinstance(value, tuple | list)) is value


@pytest.mark.parametrize(
    ("value", "sequence"), [((2, 3), False), ([2, 3], False), (2, True), (2.5, True)]
)
def test_a_number_field_takes_no_sequence_and_a_sequence_field_no_number(
    value, sequence
):
    assert _lifted(value, _I64, sequence) is value


def test_a_lifted_array_or_tracer_of_the_dtype_is_kept():
    exact = jnp.asarray([2**53 + 1, 2], jnp.int64)
    assert _lifted(exact, _I64) is exact
    seen = jax.jit(lambda q: _lifted(q, _I64))(jnp.asarray(2, jnp.int32))
    assert (
        seen.dtype == jnp.int32
    )  # another dtype is not touched; it is a tracer already


# The history of each lifted input equals the one the same study gave before lifting
# existed (recorded on the base commit; each digest is the sha256 prefix of every
# carry and trace leaf with its dtype and shape).
_BASE_DIGEST = {
    "stop_snr/bool/compiled": "4ae2dc0d9b4946a5",
    "stop_snr/bool/host": "7e928ada7413cada",
    "stop_snr/int/compiled": "fa3501a248069127",
    "stop_snr/int/host": "3c44f99fb468fbfe",
    "stop_snr/int_large_exact/compiled": "69482bb9835e4514",
    "stop_snr/int_large_exact/host": "a3e146106b2eea73",
    "stop_snr/integral/compiled": "fa3501a248069127",
    "stop_snr/integral/host": "3c44f99fb468fbfe",
    "stop_snr/float_sub/compiled": "fa3501a248069127",
    "stop_snr/float_sub/host": "3c44f99fb468fbfe",
    "query_at/bool/compiled": "a6ea3940a101f1a6",
    "query_at/bool/host": "ee391845288a8003",
    "query_at/int/compiled": "68e0ad22293d90ca",
    "query_at/int/host": "ff680283f820bb5c",
    "query_at/int_large_exact/compiled": "37a0e2ffbd98782f",
    "query_at/int_large_exact/host": "f6781172491b172a",
    "values/seq_int/compiled": "90a351073f0941af",
    "values/seq_int/host": "6755ddefb06f6cad",
    "values/seq_mixed/compiled": "90a351073f0941af",
    "values/seq_mixed/host": "6755ddefb06f6cad",
    "values/seq_list/compiled": "90a351073f0941af",
    "values/seq_list/host": "6755ddefb06f6cad",
    "ticks/seq_int/compiled": "2b3598674cbcc5b1",
    "ticks/seq_int/host": "7de904b3749c4cee",
    "ticks/seq_list/compiled": "2b3598674cbcc5b1",
    "ticks/seq_list/host": "7de904b3749c4cee",
}


def _digest(res):
    h = hashlib.sha256()
    for x in jax.tree_util.tree_leaves((res.carry, res.trace)):
        a = np.asarray(x)
        h.update(f"{a.dtype}{a.shape}".encode())
        h.update(a.tobytes())
    return h.hexdigest()[:16]


def _case_study(field, x):
    if field == "stop_snr":
        return _threshold(x), None
    if field == "query_at":
        return oracle.oracle_study(query_at=x)
    if field == "values":
        return scripted.scripted_study(
            ticks=(100, 200), values=x, ids=(7001, 7002)
        ), None
    return scripted.scripted_study(ticks=x, values=(2.0, 9.0), ids=(7001, 7002)), None


_CASE_INPUTS = {
    "bool": True,
    "int": 2,
    "int_large_exact": 2**53,
    "integral": 2.0,
    "seq_int": (2, 3),
    "seq_mixed": (2, 3.0),
    "seq_list": [2, 3],
    "float_sub": _FloatSubclass(2.0),
}


@pytest.mark.parametrize("key", sorted(_BASE_DIGEST))
def test_lifted_histories_equal_the_ones_before_lifting(key):
    field, name, which = key.split("/")
    study, service = _case_study(field, _CASE_INPUTS[name])
    truths, roots = make_worlds(study, 1)
    kw = {} if service is None else {"service": service}
    if which == "compiled":
        res = CompiledExecutor().run(study, truths, roots, 24, **kw)
    else:
        res = HostExecutor().run(study, index_world(truths, 0), roots[0], 24, **kw)
    assert _digest(res) == _BASE_DIGEST[key]


def test_step_and_decision_fingerprint_share_one_program_across_values(monkeypatch):
    import equinox as eqx

    from spaceodyssey import core
    from spaceodyssey.core import handlers, transaction

    jax.clear_caches()  # a program cached by an earlier test would hide the count
    study = _threshold(3.0)
    truths, roots = make_worlds(study, 1)
    truth = core.index_world(truths, 0)
    carry = eqx.filter_jit(core.init_carry)(study, truth)
    carry, ev, _ = core.select_next(carry, study.caps)
    values = (3.0, 4.0, 5.0)

    original = handlers.HANDLERS[0]
    count = {"n": 0}

    def counted(*args, **kw):
        count["n"] += 1
        return original(*args, **kw)

    monkeypatch.setattr(handlers, "HANDLERS", (counted, *handlers.HANDLERS[1:]))
    seen = []
    for x in values:
        core.step(carry, ev, _threshold(x), truth, roots[0])
        seen.append(count["n"])
    assert seen == [1, 1, 1]

    original_fp = transaction._policy_fingerprint
    fp = {"n": 0}

    def counted_fp(*args, **kw):
        fp["n"] += 1
        return original_fp(*args, **kw)

    monkeypatch.setattr(transaction, "_policy_fingerprint", counted_fp)
    seen = []
    for x in values:
        core.decision_fingerprint(carry, _threshold(x))
        seen.append(fp["n"])
    assert seen == [1, 1, 1]
