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

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import DocumentError, dump_study, load_study
from spaceodyssey.codec.records import check_lowering
from spaceodyssey.core.carry import init_carry
from spaceodyssey.core.conformance.corpus import make_worlds
from spaceodyssey.core.executors.base import index_world
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.lift import _lifted, lift_traced
from spaceodyssey.studies import oracle, scripted, toy
from tests.codec.test_document import TOY_KINDS
from tests.core.prior_helpers import refit_study

pytestmark = pytest.mark.medium  # Each case compiles a scan.

N_STEPS = 24  # the length of the recorded base histories
CUT = 6  # past the first bucket boundary, with the study still running (idle from 10)
BUCKET = 4  # step counts 5 to 8 share one program of 8 rows

_traces = {"n": 0}


class _CountingExecutor(CompiledExecutor):
    """Counts how often a scan body is traced, which is once per compiled program."""

    def transition(self, carry, i, root, study, truth, *, wake_fn=None):
        _traces["n"] += 1
        kw = {} if wake_fn is None else {"wake_fn": wake_fn}
        return super().transition(carry, i, root, study, truth, **kw)


def _run(study, n_steps=N_STEPS, n_worlds=1, **kw):
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


def _threshold(x):
    return toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=x),))


def _script(ticks=(100, 200), values=(2.0, 9.0)):
    return scripted.scripted_study(ticks=ticks, values=values, ids=(7001, 7002))


@pytest.fixture(scope="module")
def threshold_run():
    """One compiled scan of the threshold study; every value variant reuses it."""
    return _run(_threshold(3.0))


@pytest.fixture(scope="module")
def cut_run():
    """The default scan stopped while the study is still running."""
    return _run(_threshold(3.0), CUT)


@pytest.fixture(scope="module")
def bucketed_run():
    """The same step count through the 8-row bucketed program."""
    return _run(_threshold(3.0), CUT, bucket=BUCKET)


@pytest.fixture(scope="module")
def scripted_run():
    """One compiled scan of a two-record script; other scripts reuse it."""
    return _run(_script())


def test_bucketed_run_matches_the_default_scan_bit_for_bit(cut_run, bucketed_run):
    """Fails if the bucketed scan steps past n_steps, or changes a bit."""
    _same_bits((cut_run.carry, cut_run.trace), (bucketed_run.carry, bucketed_run.trace))


def test_bucketed_run_shares_one_program_within_a_bucket(bucketed_run):
    """Fails if a step count of the same bucket, its edge included, compiles."""
    before = _traces["n"]
    for n_steps in (BUCKET + 1, 2 * BUCKET):
        _run(_threshold(3.0), n_steps, bucket=BUCKET)
    assert _traces["n"] == before


def test_array_fields_reuse_the_program_and_are_read(threshold_run):
    """Fails if a new value retraces, or the first value stays baked in."""
    before = _traces["n"]
    never = _run(_threshold(math.inf))
    at_once = _run(_threshold(-math.inf))
    assert _traces["n"] == before
    # the second value is read, not the first one baked into the cached program
    assert int(at_once.carry.n_dec.max()) < int(never.carry.n_dec.max())


def test_kind_names_stay_out_of_the_compile_key(threshold_run):
    study = _threshold(3.0)
    renamed = dataclasses.replace(
        study,
        caps=dataclasses.replace(
            study.caps, kind_names=tuple(n + "x" for n in study.caps.kind_names)
        ),
    )
    assert renamed.caps.kind_names != study.caps.kind_names
    assert renamed.caps == study.caps and hash(renamed.caps) == hash(study.caps)
    before = _traces["n"]
    result = _run(renamed)
    assert _traces["n"] == before
    _same_bits((threshold_run.carry, threshold_run.trace), (result.carry, result.trace))


def test_marked_numbers_stay_python_numbers_and_are_lifted_for_the_call_only():
    policy = toy.StochasticPolicy(theta=0.3, n_obs=6, wait_ticks=12)
    study = toy.survey_1(policies=(policy,))
    assert type(policy.n_obs) is int and type(policy.wait_ticks) is int
    lifted = lift_traced(study).agenda.policies[0]
    assert lifted.n_obs.dtype == jnp.int64 and lifted.wait_ticks.dtype == jnp.int64
    assert study.agenda.policies[0].n_obs == 6
    stop = lift_traced(toy.ThresholdStopPolicy(stop_snr=3))
    assert stop.stop_snr.dtype == jnp.float64
    script = lift_traced(_script().process)
    assert script.ticks.dtype == jnp.int64 and script.values.dtype == jnp.float64
    assert script.kinds.shape == (0,)
    # nothing marked, nothing rebuilt
    plain = toy.survey_1()
    assert lift_traced(plain.agenda) is plain.agenda


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


def _jit_of(fn):
    """The Equinox jit an exported entry point calls."""
    if hasattr(fn, "_cached"):
        return fn
    return next(
        c.cell_contents for c in fn.__closure__ if hasattr(c.cell_contents, "_cached")
    )


def _programs(fn):
    return _jit_of(fn)._cached._cache_size()


def _shares_one_program(fn, calls):
    """Warm `fn` with the first call, then count the programs after each variant."""
    sizes = []
    for call in calls:
        call()
        sizes.append(_programs(fn))
    assert sizes[0] >= 1
    assert sizes == sizes[:1] * len(sizes), f"{fn.__name__}: {sizes}"


def _varied(study, x):
    agenda = dataclasses.replace(
        study.agenda, policies=(toy.ThresholdStopPolicy(stop_snr=x),)
    )
    return dataclasses.replace(study, agenda=agenda)


VALUES = (3.0, 4.0, 5.0)


def test_replay_entry_points_share_one_program_across_values(threshold_run):
    """Fails if a policy value retraces a replay: the real exported jits are counted."""
    from spaceodyssey.core.conformance import observer_kit
    from spaceodyssey.core.executors import replay
    from spaceodyssey.core.records import record_none

    out, roots = threshold_run, threshold_run.roots
    carry = index_world(out.carry, 0)
    trace = index_world(out.trace, 0)

    def summary(study):
        return replay._fold_summary(
            study,
            study.observer.init_summary(study.caps),
            record_none(study.caps),
            roots[0],
        )

    _shares_one_program(
        replay._fold,
        [
            lambda x=x: replay.replay_belief(_threshold(x), carry, trace, roots[0])
            for x in VALUES
        ],
    )
    _shares_one_program(
        replay._fold_summary, [lambda x=x: summary(_threshold(x)) for x in VALUES]
    )
    _shares_one_program(
        observer_kit._replay,
        [
            lambda x=x: observer_kit.replay_beliefs(_threshold(x), out, roots)
            for x in VALUES
        ],
    )


def test_masked_refit_shares_one_program_across_values():
    """Fails if a scripted value retraces the real masked refit call."""
    from spaceodyssey.core import refresh

    def study(values):
        return scripted.scripted_refresh_study(
            ticks=(100, 200), values=values, ids=(1, 2)
        )

    base = study((2.0, 9.0))
    truths, roots = make_worlds(base, 1)
    carry = eqx.filter_eval_shape(init_carry, base, index_world(truths, 0))
    shape = eqx.filter_eval_shape(refresh.gather_refit_rows, carry, roots[0], base)
    requests = jax.tree.map(lambda x: jnp.zeros((1, *x.shape), x.dtype), shape)
    boundary = toy.toy_masked_refit()
    _shares_one_program(
        refresh._masked_call,
        [
            lambda v=v: boundary.refit_batch(study(v), requests)
            for v in ((2.0, 9.0), (3.0, 8.0), (4.0, 7.0))
        ],
    )


def test_prior_observer_refit_shares_one_program_across_values():
    """Fails if a policy value retraces the audience-prior refit call."""
    from spaceodyssey.studies import prior_observer
    from tests.core.test_premission_refit import _request

    study = refit_study()
    batch = jax.tree.map(lambda x: x[None], _request(study))
    boundary = prior_observer.AudiencePriorRefit()
    _shares_one_program(
        prior_observer._audience_call,
        [lambda x=x: boundary.refit_batch(_varied(study, x), batch) for x in VALUES],
    )


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
# carry and trace leaf with its dtype and shape). The compiled executor takes the
# spellings of a float and the sequence fields, the host executor the spellings of an
# int; the cases of one study share the program of its first.
_BASE_DIGEST = {
    "stop_snr/bool/compiled": "4ae2dc0d9b4946a5",
    "stop_snr/int/compiled": "fa3501a248069127",
    "stop_snr/int_large_exact/compiled": "69482bb9835e4514",
    "stop_snr/integral/compiled": "fa3501a248069127",
    "stop_snr/float_sub/compiled": "fa3501a248069127",
    "query_at/bool/host": "ee391845288a8003",
    "query_at/int/host": "ff680283f820bb5c",
    "query_at/int_large_exact/host": "f6781172491b172a",
    "stochastic/n_obs_3/compiled": "05b4e1821b033c6d",
    "values/seq_int/compiled": "90a351073f0941af",
    "values/seq_mixed/compiled": "90a351073f0941af",
    "values/seq_list/compiled": "90a351073f0941af",
    "ticks/seq_int/compiled": "2b3598674cbcc5b1",
    "ticks/seq_list/compiled": "2b3598674cbcc5b1",
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
    if field == "stochastic":
        policy = toy.StochasticPolicy(theta=0.3, n_obs=3, wait_ticks=7)
        return toy.survey_1(policies=(policy,)), None
    if field == "query_at":
        return oracle.oracle_study(query_at=x)
    if field == "values":
        return _script(values=x), None
    if field == "ticks":
        return _script(ticks=x), None
    raise KeyError(field)


_CASE_INPUTS = {
    "bool": True,
    "int": 2,
    "int_large_exact": 2**53,
    "integral": 2.0,
    "seq_int": (2, 3),
    "seq_mixed": (2, 3.0),
    "seq_list": [2, 3],
    "float_sub": _FloatSubclass(2.0),
    "n_obs_3": None,
}


def _history_digests(which, before=lambda study: None, after=lambda: None):
    """Digest every recorded case of one executor and return {key: digest}."""
    out = {}
    for key in _BASE_DIGEST:
        field, name, executor = key.split("/")
        if executor != which:
            continue
        study, service = _case_study(field, _CASE_INPUTS[name])
        truths, roots = make_worlds(study, 1)
        before(study)
        kw = {} if service is None else {"service": service}
        if which == "compiled":
            res = _CountingExecutor().run(study, truths, roots, N_STEPS, **kw)
        else:
            res = HostExecutor().run(
                study, index_world(truths, 0), roots[0], N_STEPS, **kw
            )
        out[key] = _digest(res)
        after()
    return out


def test_compiled_histories_equal_the_ones_before_lifting(
    threshold_run, scripted_run
):  # the fixtures hold the programs the cases reuse
    """Fails if lifting a number or a sequence changes a compiled history."""
    got = _history_digests("compiled")
    want = {k: v for k, v in _BASE_DIGEST.items() if k.endswith("/compiled")}
    assert got == want


def test_host_histories_equal_the_ones_before_lifting(monkeypatch):
    """Fails if lifting the query decision changes a host history, or the service
    entry points compile again for another value."""
    from spaceodyssey.core import service
    from spaceodyssey.core.executors import host

    current = {}
    answer = host.answer_pending

    def unlifted(carry, study, svc):  # the exported call, handed the study as written
        return answer(carry, current["study"], svc)

    monkeypatch.setattr(host, "answer_pending", unlifted)
    sizes = []
    got = _history_digests(
        "host",
        lambda study: current.update(study=study),
        lambda: sizes.append(
            (_programs(service._snapshot_for), _programs(service._commit))
        ),
    )
    want = {k: v for k, v in _BASE_DIGEST.items() if k.endswith("/host")}
    assert got == want
    assert min(sizes[0]) >= 1 and sizes == sizes[:1] * len(sizes)


def test_a_changed_script_reuses_the_program(scripted_run):
    """Fails if other ticks or values compile the scripted scan again."""
    before = _traces["n"]
    _run(_script(ticks=(150, 250), values=(3.0, 8.0)))
    assert _traces["n"] == before


def test_step_and_decision_fingerprint_share_one_program_across_values(monkeypatch):
    import equinox as eqx

    from spaceodyssey import core
    from spaceodyssey.core import handlers, transaction

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
    assert seen[0] <= 1 and seen[1:] == seen[:1] * 2

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
    assert seen[0] <= 1 and seen[1:] == seen[:1] * 2
