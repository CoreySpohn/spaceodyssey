"""Studies that differ only in numbers, or in step count, share one compiled program.

A number a policy or process holds is lifted to an array where the study enters a
compiled call, so a new value reuses the program; the bucketed run takes every step
count of a bucket through one program; names that are never read under trace stay out of
the compile key. None of it changes a history: the lifted and the bucketed runs are
compared with the default scan bit for bit, and the objects and documents are unchanged.
"""

import dataclasses
import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import DocumentError, dump_study, load_study
from spaceodyssey.codec.records import check_lowering
from spaceodyssey.core.conformance.corpus import make_worlds
from spaceodyssey.core.executors.compiled import CompiledExecutor, _scan_worlds
from spaceodyssey.core.lift import lift_traced
from spaceodyssey.core.transaction import wake_by_audience
from spaceodyssey.studies import scripted, toy
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
