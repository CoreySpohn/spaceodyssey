"""Study documents: exact round trip, program reuse, stable hash, and pointed errors."""

import json
import math
import os
import subprocess
import sys
from pathlib import Path

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from spaceodyssey.codec import DocumentError, dump_study, load_study
from spaceodyssey.core.enums import Audience, Purpose
from spaceodyssey.core.executors.base import histories_equal
from spaceodyssey.core.executors.host import HostExecutor
from spaceodyssey.core.keys import Coupling
from spaceodyssey.core.kinds import AbstractDesign
from spaceodyssey.studies import catalog, oracle, scripted, toy
from spaceodyssey.studies.conformance import CORPUS
from tests.core.helpers import one_world


class LabelledDesign(AbstractDesign):
    """A design holding a dict, and a count its `__post_init__` derives from it."""

    labels: dict
    n_labels: int = 0

    def __post_init__(self):
        """Derive the count."""
        object.__setattr__(self, "n_labels", len(self.labels))


# Unit tests pass an explicit mapping; the installed entry points have their own test.
TOY_KINDS = {
    ("observer", "toy.running_mean", 1): toy.ToyRunningMean,
    ("policy", "toy.fixed_plan", 1): toy.FixedPlanPolicy,
    ("policy", "toy.threshold_stop", 1): toy.ThresholdStopPolicy,
    ("process", "toy.null", 1): toy.NullProcess,
    ("process", "toy.heartbeat", 1): toy.HeartbeatProcess,
    ("world_source", "toy.uniform_signal", 1): toy.ToyWorldSource,
    ("channel_model", "toy.photometry", 1): toy.ToyPhotometry,
    ("design", "toy.gain", 1): toy.ToyDesign,
    ("objective", "toy.threshold", 1): toy.ToyObjective,
    ("policy", "toy.spin_wait", 1): toy.SpinWaitPolicy,
    ("policy", "toy.null", 1): toy.NullPolicy,
    ("observer", "catalog.observer", 1): catalog.CatalogObserver,
    ("policy", "oracle.query", 1): oracle.QueryPolicy,
    ("process", "scripted.records", 1): scripted.ScriptedRecords,
    ("design", "test.labelled", 1): LabelledDesign,
}

# The jit trace and XLA compile events; any jit cache miss emits the first.
_COMPILE_EVENTS = (
    "/jax/core/compile/jaxpr_trace_duration",
    "/jax/core/compile/backend_compile_duration",
)
_compiles = {"n": 0}
jax.monitoring.register_event_duration_secs_listener(
    lambda event, _secs, **_kw: (
        _compiles.update(n=_compiles["n"] + 1) if event in _COMPILE_EVENTS else None
    )
)


def rich_study():
    """survey_1 with two phases, a static process, a coupling and Python scalars.

    Between them the fields hold static ints and bools, a Python float, nested
    static tuples with an enum member, weakly typed arrays and a tuple of policies.
    """
    return toy.survey_1(
        policies=(
            toy.FixedPlanPolicy(plan=((0, 10), (1, 11))),
            toy.ThresholdStopPolicy(stop_snr=2.5),
        ),
        phase_until=(300, 1000),
        process=toy.HeartbeatProcess(
            period_ticks=37, n_fires=3, audience=Audience.OBSERVER, writes=True
        ),
        coupling=Coupling(shared=((Purpose.MEASUREMENT, ("asset", "branch")),)),
    )


def assert_same_pytree(a, b):
    """Equal treedef and static part, and every leaf of the same type and bits."""
    assert jax.tree_util.tree_structure(a) == jax.tree_util.tree_structure(b)
    assert eqx.partition(a, eqx.is_array)[1] == eqx.partition(b, eqx.is_array)[1]
    for x, y in zip(*map(jax.tree_util.tree_leaves, (a, b)), strict=True):
        assert type(x) is type(y)
        if isinstance(x, jax.Array):
            assert (x.dtype, x.shape, x.weak_type) == (y.dtype, y.shape, y.weak_type)
            assert np.asarray(x).tobytes() == np.asarray(y).tobytes()
        else:
            assert x == y


def study_hash():
    """Content hash of the rich study; the cross-process test calls it in a child."""
    return dump_study(rich_study(), TOY_KINDS).content_hash


def test_toy_round_trip_is_exact():
    """Fails if a reload changes a Python type (tuple to list, int to float, an enum
    to its int), an array dtype or weak type, a static field, or any leaf bit, or
    if dumping the reloaded study does not give the same canonical text.
    """
    study = rich_study()
    text = dump_study(study, TOY_KINDS).to_json()
    loaded = load_study(text, TOY_KINDS)
    assert_same_pytree(study, loaded)
    assert type(loaded.coupling.shared[0][0]) is Purpose
    assert any(x.weak_type for x in jax.tree_util.tree_leaves(loaded.channels))
    assert dump_study(loaded, TOY_KINDS).to_json() == text


def test_every_corpus_study_round_trips_exactly():
    """Fails if restore validation refuses a valid corpus study (a converter or
    `__post_init__` that does not accept its own output), or if any corpus study
    reloads with a different treedef, static part or leaf, which would recompile.
    """
    for case in CORPUS:
        study = case.make_study()
        assert_same_pytree(study, load_study(dump_study(study, TOY_KINDS), TOY_KINDS))


def test_reloaded_study_runs_and_reuses_the_compiled_program():
    """Fails if a reloaded study misses the host executor's jit cache (a new trace
    or compile) or runs to a different canonical history than the original.
    """
    study = toy.survey_1()
    truth, root = one_world()
    # Run the original here: the suite may have cleared the jit cache since any
    # earlier run of the same study.
    run = HostExecutor().run(study, truth, root, 40)
    loaded = load_study(dump_study(study, TOY_KINDS).to_json(), TOY_KINDS)
    before = _compiles["n"]
    again = HostExecutor().run(loaded, truth, root, 40)
    assert _compiles["n"] == before
    assert histories_equal(run.history(), again.history()).ok


def test_content_hash_is_stable_across_processes():
    """Fails if the hash depends on the process: dict or set order, string hash
    randomization, object ids, or a float formatted differently.
    """
    root = Path(__file__).parents[2]
    script = (
        "import jax; jax.config.update('jax_enable_x64', True); "
        "from tests.codec.test_document import study_hash; print(study_hash())"
    )
    env = dict(os.environ, PYTHONHASHSEED="12345", JAX_PLATFORMS="cpu")
    child = subprocess.run(
        [sys.executable, "-c", script],
        cwd=root,
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert child.stdout.strip().splitlines()[-1] == study_hash()


def test_dtypes_non_finite_floats_and_negative_zero_survive():
    """Fails if NaN or an infinity is written as a bare literal or lost, if -0.0
    comes back as 0.0 (in an array or a Python float), or if a float32 array is
    widened to float64.
    """
    noise = jnp.asarray([math.nan, -math.inf, -0.0], jnp.float32)
    study = toy.survey_1(
        policies=(
            toy.ThresholdStopPolicy(stop_snr=-0.0),
            toy.ThresholdStopPolicy(stop_snr=math.inf),
        ),
        phase_until=(300, 1000),
        channel_model=toy.ToyPhotometry(noise=noise),
        design=toy.ToyDesign(gain=-0.0),
        detect_thr=-math.inf,
    )
    text = dump_study(study, TOY_KINDS).to_json()
    assert "NaN" not in text and "Infinity" not in text
    loaded = load_study(text, TOY_KINDS)
    assert_same_pytree(study, loaded)
    got = np.asarray(loaded.channel_model.noise)
    assert got.dtype == np.float32
    assert np.isnan(got[0]) and got[1] == -np.inf and np.signbit(got[2])
    first, second = loaded.agenda.policies
    assert math.copysign(1.0, first.stop_snr) == -1.0 and second.stop_snr == math.inf
    assert np.signbit(loaded.design.gain) and loaded.objective.threshold == -np.inf


def test_dict_order_is_content():
    """Fails if a dict's insertion order is lost on reload or left out of the hash
    (a policy reading the first key would change behavior unseen), or if a stored
    field that `__post_init__` derives differently is accepted.
    """
    docs = [
        dump_study(toy.survey_1(design=LabelledDesign(labels=labels)), TOY_KINDS)
        for labels in ({"z": 1, "a": 2}, {"a": 2, "z": 1})
    ]
    assert docs[0].content_hash != docs[1].content_hash
    for doc, order in zip(docs, (["z", "a"], ["a", "z"]), strict=True):
        assert list(load_study(doc.to_json(), TOY_KINDS).design.labels) == order
    tampered = json.loads(docs[0].to_json())
    tampered.pop("content_hash")
    tampered["study"]["design"]["n_labels"] = 5
    with pytest.raises(DocumentError) as err:
        load_study(json.dumps(tampered), TOY_KINDS)
    assert err.value.pointer == "/study/design/n_labels"


def _big_endian_noise(study):
    noise = np.array([0.5, 0.3, 0.8], ">f8")
    return eqx.tree_at(lambda s: s.channel_model.noise, study, noise)


def _big_endian_ids(study):
    return eqx.tree_at(lambda s: s.channels.id, study, np.array([10, 11, 12], ">i8"))


@pytest.mark.parametrize(
    ("swap", "where"),
    [
        (_big_endian_noise, "/study/channel_model/noise"),
        (_big_endian_ids, "/study/channels/id"),
    ],
    ids=[">f8", ">i8"],
)
def test_non_native_byte_order_is_refused(swap, where):
    """Fails if a big-endian array is written as if native, losing its byte order
    on reload and in the hash.
    """
    with pytest.raises(DocumentError) as err:
        dump_study(swap(toy.survey_1()), TOY_KINDS)
    assert err.value.pointer == where


def _set_budget_shape(doc):
    doc["study"]["budget"]["shape"] = [3, 4]


def _drop_a_level(doc):
    doc["study"]["agenda"]["levels"].update(shape=[2], data=[3, 3])


def _unknown_observer(doc):
    doc["study"]["observer"]["$kind"] = ["observer", "toy.no_such_kind", 1]


def _edit_hash(doc):
    doc["content_hash"] = "0" * 64


def _newer_schema(doc):
    doc["schema_version"] = 2


def _put(*path_and_value):
    """An edit that sets the value at a path of keys in the study."""
    *path, value = path_and_value

    def edit(doc):
        node = doc["study"]
        for key in path[:-1]:
            node = node[key]
        node[path[-1]] = value

    return edit


@pytest.mark.parametrize(
    ("edit", "where", "code"),
    [
        (_set_budget_shape, "/study/budget/data", None),
        (_drop_a_level, "/study/agenda", None),
        (_unknown_observer, "/study/observer/$kind", "D001"),
        (_newer_schema, "/schema_version", "D002"),
        (_edit_hash, "/content_hash", None),
        (_put("clock", "tick_length_s", "hello"), "/study/clock/tick_length_s", None),
        (
            _put("horizon_tick", {"$array": "float64", "shape": [], "data": 10.75}),
            "/study/horizon_tick",
            None,
        ),
        (_put("agenda", "levels", [3]), "/study/agenda/levels", None),
        (
            _put("observer", {"$enum": ["core", "clock", 1], "value": 1.0}),
            "/study/observer/$enum",
            None,
        ),
        (
            _put("observer", {"$kind": ["core", "audience", 1]}),
            "/study/observer/$kind",
            None,
        ),
        (_put("observer", {"$enum": ["core", "stage", 1]}), "/study/observer", None),
    ],
    ids=[
        "data-shape",
        "module-check",
        "unknown-kind",
        "newer-schema",
        "edited-hash",
        "string-for-float",
        "float-for-int-tick",
        "tuple-for-array",
        "enum-tag-on-module",
        "kind-tag-on-enum",
        "enum-without-value",
    ],
)
def test_bad_document_fails_with_a_pointer(edit, where, code):
    """Fails if a bad document loads, or fails without the pointer of the offending
    value: data that does not match its declared shape, a field that breaks its
    module's check, an unknown kind, a newer schema version, a hand-edited hash, a
    value its converter would coerce (a string clock, a fractional tick), a tuple
    where an array is expected, or a tag naming the wrong category of class.
    """
    doc = json.loads(dump_study(toy.survey_1(), TOY_KINDS).to_json())
    doc.pop("content_hash")  # a document without a hash loads unchecked
    edit(doc)
    with pytest.raises(DocumentError) as err:
        load_study(json.dumps(doc), TOY_KINDS)
    assert (err.value.pointer, err.value.code) == (where, code)
    assert where in str(err.value)


def _token_in_budget(token):
    """Document text with one budget element replaced by a raw JSON token."""
    doc = json.loads(dump_study(toy.survey_1(), TOY_KINDS).to_json())
    doc["study"]["budget"]["data"][0][0] = "@token@"
    return json.dumps(doc).replace('"@token@"', token)


def _repeat_a_nested_key():
    text = dump_study(toy.survey_1(), TOY_KINDS).to_json()
    return text.replace('"budget":{', '"budget":{"shape":[1],', 1)


@pytest.mark.parametrize(
    ("make_text", "where"),
    [
        (lambda: _token_in_budget("1e999"), "/study/budget/data/0/0"),
        (lambda: _token_in_budget("NaN"), "/study/budget/data/0/0"),
        (_repeat_a_nested_key, "/study/budget/shape"),
        (lambda: dump_study(toy.survey_1(), TOY_KINDS).to_json()[:-1], ""),
    ],
    ids=["overflow", "bare-nan", "nested-duplicate-key", "truncated"],
)
def test_malformed_text_fails_with_a_pointer(make_text, where):
    """Fails if a number that overflows to infinity (or a bare NaN) loads, reaches
    the hash check, or fails without its pointer; if a key repeated inside a
    nested object is resolved silently or reported without its path; or if
    malformed JSON escapes as a raw decoder error.
    """
    with pytest.raises(DocumentError) as err:
        load_study(make_text(), TOY_KINDS)
    assert err.value.pointer == where
