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
from spaceodyssey.studies import toy
from tests.core.helpers import one_world

# T5a resolves kinds through an explicit mapping; entry points replace it later.
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


def _set_budget_shape(doc):
    doc["study"]["budget"]["shape"] = [3, 4]


def _drop_a_level(doc):
    doc["study"]["agenda"]["levels"].update(shape=[2], data=[3, 3])


def _unknown_observer(doc):
    doc["study"]["observer"]["$kind"] = ["observer", "toy.no_such_kind", 1]


def _edit_hash(doc):
    doc["content_hash"] = "0" * 64


@pytest.mark.parametrize(
    ("edit", "where", "code"),
    [
        (_set_budget_shape, "/study/budget/data", None),
        (_drop_a_level, "/study/agenda", None),
        (_unknown_observer, "/study/observer/$kind", "D001"),
        (_edit_hash, "/content_hash", None),
    ],
    ids=["data-shape", "module-check", "unknown-kind", "edited-hash"],
)
def test_bad_document_fails_with_a_pointer(edit, where, code):
    """Fails if a declared shape that does not match the data, a field that breaks
    its module's check, an unknown kind or a hand-edited hash loads, or fails
    without the pointer of the offending value.
    """
    doc = json.loads(dump_study(toy.survey_1(), TOY_KINDS).to_json())
    doc.pop("content_hash")  # a document without a hash loads unchecked
    edit(doc)
    with pytest.raises(DocumentError) as err:
        load_study(json.dumps(doc), TOY_KINDS)
    assert (err.value.pointer, err.value.code) == (where, code)
    assert where in str(err.value)
