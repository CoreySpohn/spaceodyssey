"""Studies that differ only in numbers, or in step count, share one compiled program.

A number a policy or process holds as an array is traced, so a new value reuses the
program; the bucketed run takes every step count of a bucket through one program;
names that are never read under trace stay out of the compile key. None of it changes a
history:
the bucketed run is compared with the default scan bit for bit, and the plain document
form of the array fields keeps every earlier document and hash.
"""

import dataclasses
import json

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import DocumentError, dump_study, load_study
from spaceodyssey.codec.records import check_lowering
from spaceodyssey.core.conformance.corpus import make_worlds
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.studies import scripted, toy
from tests.codec.test_document import TOY_KINDS, assert_same_pytree
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


def test_array_fields_reuse_the_program_and_keep_the_history_rule():
    first = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=2.0),))
    _run(first, 30)
    before = _traces["n"]
    second = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=0.5),))
    result = _run(second, 30)
    assert _traces["n"] == before
    # a lower threshold stops sooner, so the value is read, not baked in
    assert int(result.carry.n_dec.max()) <= int(_run(first, 30).carry.n_dec.max())


def test_policy_and_script_numbers_are_arrays_of_the_arithmetic_dtype():
    policy = toy.StochasticPolicy(theta=0.3, n_obs=6, wait_ticks=12)
    assert policy.n_obs.dtype == jnp.int64 and policy.wait_ticks.dtype == jnp.int64
    assert toy.ThresholdStopPolicy(stop_snr=3).stop_snr.dtype == jnp.float64
    script = _scripted().process
    assert script.ticks.dtype == jnp.int64 and script.values.dtype == jnp.float64
    assert script.kinds.shape == (0,)


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


def test_array_fields_are_written_as_the_plain_values_they_were_built_from():
    study = _scripted()
    text = dump_study(study, KINDS).to_json()
    process = json.loads(text)["study"]["process"]
    assert process["ticks"] == [100, 200] and process["values"] == [2.0, 9.0]
    assert process["kinds"] == []
    assert_same_pytree(study, load_study(text, KINDS))


def test_a_plain_value_field_stored_as_an_array_is_refused():
    study = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=2.5),))
    doc = json.loads(dump_study(study, KINDS).to_json())
    doc.pop("content_hash")  # a document without a hash loads unchecked
    policy = doc["study"]["agenda"]["policies"][0]
    assert policy["stop_snr"] == 2.5
    policy["stop_snr"] = {"$array": "float64", "shape": [], "data": 2.5}
    with pytest.raises(DocumentError, match="plain number"):
        load_study(json.dumps(doc), KINDS)
