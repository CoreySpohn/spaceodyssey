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
import math

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import DocumentError, dump_study, load_study
from spaceodyssey.codec.records import check_lowering
from spaceodyssey.core.capacities import OPTIONAL, PLAIN_VALUE
from spaceodyssey.core.conformance.corpus import make_worlds
from spaceodyssey.core.enums import RecordStatus
from spaceodyssey.core.executors.compiled import CompiledExecutor
from spaceodyssey.core.kinds import AbstractDesign
from spaceodyssey.plain import plain_array
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


def test_array_fields_reuse_the_program_and_are_read():
    never = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=math.inf),))
    first = _run(never, 30)
    before = _traces["n"]
    at_once = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=-math.inf),))
    result = _run(at_once, 30)
    assert _traces["n"] == before
    # the second value is read, not the first one baked into the cached program
    assert int(result.carry.n_dec.max()) < int(first.carry.n_dec.max())


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


def test_earlier_documents_dump_byte_for_byte_as_they_did():
    """Fields built from an int, or an enum member, keep that wire form and hash."""
    study = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=3),))
    doc = dump_study(study, KINDS)
    # the hash this study had before the field became an array
    assert doc.content_hash == (
        "2d7d2831f3066b73c0a1741ea3bd19e82ce1375cc3b36b26ed42e2a6b808ea9f"
    )
    again = dump_study(load_study(doc.to_json(), KINDS), KINDS)
    assert again.content_hash == doc.content_hash
    ints = scripted.scripted_study(
        ticks=(100, 200),
        values=(2, 9),
        ids=(7001, 7001),
        revisions=(0, 1),
        statuses=(RecordStatus.COMPLETED, RecordStatus.PARTIAL),
    )
    doc = dump_study(ints, KINDS)
    process = json.loads(doc.to_json())["study"]["process"]
    assert process["values"] == [2, 9] and all(
        type(v) is int for v in process["values"]
    )
    assert all("$enum" in node for node in process["statuses"])
    again = dump_study(load_study(doc.to_json(), KINDS), KINDS)
    assert again.content_hash == doc.content_hash


def test_an_array_made_by_other_means_round_trips_as_plain_numbers():
    study = toy.survey_1(policies=(toy.ThresholdStopPolicy(),))
    for value in (np.asarray(2.5), jnp.asarray(2.5)):
        edited = eqx.tree_at(lambda s: s.agenda.policies[0].stop_snr, study, value)
        text = dump_study(edited, KINDS).to_json()
        assert json.loads(text)["study"]["agenda"]["policies"][0]["stop_snr"] == 2.5
        loaded = load_study(text, KINDS)
        assert float(loaded.agenda.policies[0].stop_snr) == 2.5


@pytest.mark.parametrize(
    ("stored", "match"),
    [
        ([{"$array": "int64", "shape": [], "data": 100}, 200], "not a ArrayImpl"),
        ([100.75, 200], "exactly"),
        ([2**63, 200], "field 'ticks'"),
        (["a", 200], "not a str"),
        ([[100, 200]], "exactly"),
    ],
)
def test_a_plain_field_that_would_be_coerced_is_refused(stored, match):
    doc = json.loads(dump_study(_scripted(), KINDS).to_json())
    doc.pop("content_hash")  # a document without a hash loads unchecked
    doc["study"]["process"]["ticks"] = stored
    with pytest.raises(DocumentError, match=match) as err:
        load_study(json.dumps(doc), KINDS)
    assert "/study/process/ticks" in str(err.value)


class _OptionalGain(AbstractDesign):
    gain: jax.Array = eqx.field(
        default=1.0,
        converter=lambda x: plain_array(jnp.float64, x),
        metadata={**OPTIONAL, **PLAIN_VALUE},
    )


def test_an_optional_plain_field_is_converted_when_omitted_and_omitted_at_default():
    kinds = {**KINDS, ("design", "test.optional_gain", 1): _OptionalGain}
    study = toy.survey_1(design=_OptionalGain())
    doc = dump_study(study, kinds)
    assert "gain" not in json.loads(doc.to_json())["study"]["design"]
    loaded = load_study(doc.to_json(), kinds)
    assert isinstance(loaded.design.gain, np.ndarray)
    assert loaded.design.gain.dtype == np.float64


@pytest.mark.parametrize("bucket", [0, -16, 2.5, True, "16"])
def test_a_bucket_must_be_a_positive_int(bucket):
    with pytest.raises(ValueError, match="bucket"):
        _run(toy.survey_1(), 17, bucket=bucket)


def test_a_plain_value_field_stored_as_an_array_is_refused():
    study = toy.survey_1(policies=(toy.ThresholdStopPolicy(stop_snr=2.5),))
    doc = json.loads(dump_study(study, KINDS).to_json())
    doc.pop("content_hash")  # a document without a hash loads unchecked
    policy = doc["study"]["agenda"]["policies"][0]
    assert policy["stop_snr"] == 2.5
    policy["stop_snr"] = {"$array": "float64", "shape": [], "data": 2.5}
    with pytest.raises(DocumentError, match="plain number"):
        load_study(json.dumps(doc), KINDS)
