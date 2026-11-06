"""The OFTI host-loop refit: one real fit, status mapping, keys and model hashes."""

import dataclasses
import types

import equinox as eqx
import jax
import jax.numpy as jnp
import numpy as np
import pytest

pytest.importorskip("photomancy")

from photomancy.orbit.forward import predict_relative_astrometry
from photomancy.orbit.ofti import PARAM_NAMES

from spaceodyssey.adapters.photomancy import (
    COLUMN_NAMES,
    OftiRefit,
    ofti_model,
    photomancy_bindings,
)
from spaceodyssey.codec import DocumentError, dump_run, load_run
from spaceodyssey.core import CompiledExecutor, HostExecutor
from spaceodyssey.core.capacities import Capacities
from spaceodyssey.core.enums import F64, I64, PUBLIC, RefitStatus
from spaceodyssey.core.executors.base import histories_equal, index_world
from spaceodyssey.core.keys import refit_key
from spaceodyssey.core.kinds import AbstractObserver, ObserverCapabilities
from spaceodyssey.core.records import (
    KindRegistry,
    PayloadColumn,
    RecordKind,
)
from spaceodyssey.core.refresh import (
    REFRESH_PRODUCT_KIND,
    RefreshSchedule,
    answer_refresh,
    gather_refit_rows,
    revision_set_digest,
)
from spaceodyssey.core.snapshot import BeliefLayout
from spaceodyssey.studies import toy
from spaceodyssey.studies.scripted import ScriptedRecords
from tests.core.helpers import worlds

MSUN = 1.98892e30
DIST_PC = 10.0
N_PARTICLES = 16
ASTROMETRY = 4  # registry index of the astrometry kind
EPOCHS_D = (0.0, 12.0, 30.0)
SIGMA = 0.02  # arcsec: a loose, short arc so the rejection sampler accepts quickly


class ParticleBelief(eqx.Module):
    samples: jax.Array  # [n_subjects, n_particles, 7]
    n: jax.Array  # epochs fit, per subject
    aux: jax.Array


def project_particles(belief):
    return belief.samples[:, :, 0].mean(axis=1), belief.aux


class OftiObserver(AbstractObserver):
    """Holds OFTI particles per subject; it changes only by refresh."""

    subject_leaves = ("samples", "n")
    n_particles: int = eqx.field(default=N_PARTICLES, static=True)

    def capabilities(self):
        return ObserverCapabilities(
            incremental=False, refit=True, refit_kinds=("astrometry",)
        )

    def init(self, caps):
        return ParticleBelief(
            jnp.zeros((caps.n_subjects, self.n_particles, len(PARAM_NAMES)), F64),
            jnp.zeros(caps.n_subjects, I64),
            jnp.zeros(caps.n_an, F64),
        )

    def layout(self, caps):
        shapes = jax.eval_shape(lambda: self.init(caps))
        return BeliefLayout(
            belief=shapes, summary=shapes, count_leaf="n", project=project_particles
        )

    def install(self, belief, fitted, subject):
        return ParticleBelief(
            belief.samples.at[subject].set(fitted.samples[subject]),
            belief.n.at[subject].set(fitted.n[subject]),
            belief.aux,
        )


class AstrometryScript(ScriptedRecords):
    """Scripted records whose payload is a full astrometry row."""

    table: tuple = eqx.field(default=(), static=True)

    def handle(self, carry, ev, study):
        carry, rec = super().handle(carry, ev, study)
        row = jnp.asarray(self.table, F64)[ev.group]
        return carry, dataclasses.replace(rec, payload=rec.payload.at[:6].set(row))


REGISTRY = KindRegistry(
    (
        RecordKind("none", ()),
        toy.MEASUREMENT_KIND,
        toy.AUXILIARY_KIND,
        REFRESH_PRODUCT_KIND,
        RecordKind("astrometry", tuple(PayloadColumn(n, PUBLIC) for n in COLUMN_NAMES)),
    )
)
KINDS = {
    ("observer", "test.ofti_observer", 1): OftiObserver,
    ("process", "test.astrometry_script", 1): AstrometryScript,
    ("policy", "toy.null", 1): toy.NullPolicy,
    ("world_source", "toy.uniform_signal", 1): toy.ToyWorldSource,
    ("channel_model", "toy.photometry", 1): toy.ToyPhotometry,
    ("design", "toy.gain", 1): toy.ToyDesign,
    ("objective", "toy.threshold", 1): toy.ToyObjective,
}


def astrometry_table():
    """Three noiseless epochs of one orbit, as payload rows (days, arcsec)."""
    a, e, cos_i, w_node = 1.2, 0.1, 0.6, 0.8
    t = jnp.asarray(EPOCHS_D)
    ra, dec = predict_relative_astrometry(
        t, a, e, cos_i, w_node, jnp.cos(0.5), jnp.sin(0.5), 3.0, MSUN, DIST_PC
    )
    rows = np.stack(
        [
            np.asarray(t),
            np.asarray(ra),
            np.asarray(dec),
            np.full(3, SIGMA),
            np.full(3, SIGMA),
            np.zeros(3),
        ],
        axis=1,
    )
    return tuple(tuple(float(x) for x in r) for r in rows)


def ofti_study(ticks=(10, 40, 70), every=100):
    observer = OftiObserver()
    caps = REGISTRY.lower(Capacities(rows_per_subject=8, refit_lanes=2), observer)
    table = astrometry_table()[: len(ticks)]
    n = len(ticks)
    process = AstrometryScript(
        ticks=tuple(ticks),
        values=(0.0,) * n,
        ids=tuple(range(n)),
        revisions=(0,) * n,
        statuses=(0,) * n,
        kinds=(ASTROMETRY,) * n,
        table=table,
    )
    schedule = RefreshSchedule(every, 0, *toy.PRODUCT_CHANNELS, toy.PRODUCT_KIND)
    return toy.survey_1(
        caps,
        observer=observer,
        refresh=schedule,
        policies=(toy.NullPolicy(),),
        process=process,
    )


def refit(**kw):
    base = {
        "stellar_mass_kg": MSUN,
        "dist_pc": DIST_PC,
        "count": "n",
        "batch": 4096,
    }
    return OftiRefit(**{**base, **kw})


@pytest.fixture(scope="module")
def held():
    """The study run to its first checkpoint with no boundary: a frozen world."""
    study = ofti_study()
    truths, roots = worlds(1)
    run = HostExecutor().run(study, index_world(truths, 0), roots[0], 30)
    assert bool(run.carry.refresh.valid)
    return study, run.carry, roots[0]


def entry_of(table, key, n_rows=8):
    """A `RefitEntry`-shaped object over `table` rows (the fit reads rows and key)."""
    payload = np.zeros((n_rows, 6))
    payload[: len(table)] = np.asarray(table, float).reshape(-1, 6)
    valid = np.arange(n_rows) < len(table)
    rows = types.SimpleNamespace(valid=valid, payload=payload)
    return types.SimpleNamespace(rows=rows, key=np.asarray(key))


def stub_study():
    return types.SimpleNamespace(observer=OftiObserver(), caps=Capacities())


KEY = np.asarray(jax.random.PRNGKey(0))


def test_a_failed_or_unsupported_entry_is_a_status_never_an_exception():
    """Exhausted batches, no rows and an unusable row each map to a status.

    Fails if OFTI's RuntimeError escapes the boundary, or an entry that cannot be
    fit returns slices that would install.
    """
    study, table = stub_study(), astrometry_table()
    fit = refit(batch=8, max_batches=1)._fit
    slices, status, diag = fit(study, entry_of(table, KEY))
    assert (slices, status) == (None, RefitStatus.NO_SUPPORT)
    assert diag == [3.0, 0.0]
    assert fit(study, entry_of((), KEY))[:2] == (None, RefitStatus.NO_SUPPORT)
    bad = [list(r) for r in table]
    bad[1][3] = 0.0  # a zero uncertainty, as a masked column would give
    assert fit(study, entry_of(bad, KEY))[:2] == (None, RefitStatus.FAILED)
    wrong = types.SimpleNamespace(observer=toy.ToyRunningMean(), caps=Capacities())
    with pytest.raises(ValueError, match="subject leaves"):
        fit(wrong, entry_of(table, KEY))


def test_the_same_rows_and_key_give_the_same_particles_and_a_new_set_a_new_key():
    """The fit is a pure function of rows and key; a revision moves the key.

    Fails if the boundary reads anything but its entry (a counter, a clock), or if
    a changed revision set leaves the refit key where it was.
    """
    study, table = stub_study(), astrometry_table()[:2]
    fit = refit(batch=1024)._fit
    first, second = (fit(study, entry_of(table, KEY)) for _ in range(2))
    assert first[1] == RefitStatus.OK and first[0][0].shape == (N_PARTICLES, 7)
    assert first[0][0].tobytes() == second[0][0].tobytes()
    other = fit(study, entry_of(table, np.asarray(jax.random.PRNGKey(1))))
    assert other[0][0].tobytes() != first[0][0].tobytes()

    def rows(revision):
        z = jnp.zeros(4, I64)
        return types.SimpleNamespace(
            channel=z + 50,
            acquisition_index=jnp.arange(4, dtype=I64),
            revision=z.at[1].set(revision),
            valid=jnp.arange(4) < 2,
        )

    root = jax.random.PRNGKey(0)
    keys = [refit_key(root, 0, revision_set_digest(rows(r))) for r in (0, 1)]
    assert not np.array_equal(keys[0], keys[1])


def test_a_model_hash_is_stable_and_a_mismatch_between_executors_is_refused():
    """The hash follows the declared target; D003 refuses two hashes in one run.

    Fails if the digest ignores the prior, the mass or the support, or if a run
    document binds host and compiled boundaries of different models.
    """
    base = dict(stellar_mass_kg=MSUN, dist_pc=DIST_PC)
    digest = ofti_model(**base).digest()
    assert digest == ofti_model(**base).digest() == refit().model_hash
    for change in (
        dict(ecc_prior="disk"),
        dict(stellar_mass_kg=0.5 * MSUN),
        dict(log_P_range=(1.0, 5.0)),
        dict(e_max=0.9),
    ):
        assert ofti_model(**{**base, **change}).digest() != digest
    study = ofti_study()
    with pytest.raises(DocumentError) as err:
        dump_run(
            study,
            REGISTRY,
            executor="host",
            bindings=photomancy_bindings(),
            refit={"host": refit(), "compiled": refit(ecc_prior="disk")},
            kinds=KINDS,
        )
    assert err.value.code == "D003"


@pytest.mark.medium
def test_a_checkpoint_fit_installs_for_both_audiences_and_replays(held):
    """One checkpoint installs the OFTI particles; re-answering reproduces them.

    Medium: the study compiles once and OFTI's scorer once (about 5 s cold).
    Fails if the install drops a subject slice, the policy summary misses the
    deduplicated fit, or the same request does not give the same answer.
    """
    study, carry, root = held
    boundary = refit()
    done = answer_refresh(carry, root, study, boundary)
    samples = np.asarray(done.belief.samples[0])
    assert int(done.belief.n[0]) == 3 and samples.shape == (N_PARTICLES, 7)
    assert np.isfinite(samples).all() and len({tuple(r) for r in samples}) > 1
    assert np.asarray(done.pbelief.samples).tobytes() == (
        np.asarray(done.belief.samples).tobytes()
    )
    assert np.asarray(done.refresh_status)[:, 0].tolist() == [RefitStatus.OK] * 2
    assert not np.asarray(done.refresh_dirty).any()
    # replay: the same frozen request, answered again, is the installed belief
    request = eqx.filter_jit(gather_refit_rows)(carry, root, study)
    batch = jax.tree_util.tree_map(lambda x: x[None], request)
    again = boundary.refit_batch(study, batch)
    assert np.asarray(again.slices[0][0, 0, 0]).tobytes() == samples.tobytes()


@pytest.mark.medium
def test_exhausted_batches_leave_the_belief_unchanged(held):
    """A starved sampler records NO_SUPPORT on the checkpoint and installs nothing.

    Fails if the belief moves, the status is not recorded, or the run raises.
    """
    study, carry, root = held
    done = answer_refresh(carry, root, study, refit(batch=8, max_batches=1))
    assert np.asarray(done.belief.samples).tobytes() == (
        np.asarray(carry.belief.samples).tobytes()
    )
    assert int(done.belief.n[0]) == 0
    status = np.asarray(done.refresh_status)[:, 0].tolist()
    assert status == [RefitStatus.NO_SUPPORT] * 2
    assert not np.asarray(done.refresh_dirty).any()


@pytest.mark.medium
def test_a_reloaded_run_document_reproduces_the_host_and_compiled_histories():
    """A document naming the OFTI boundary reloads to the same run on both executors.

    Fails if the binding loses its configuration or model hash, or if the compiled
    chunked driver answers a host boundary differently from the host executor.
    """
    study = ofti_study()
    boundary = refit()
    truths, roots = worlds(1)
    world = index_world(truths, 0)
    mine = HostExecutor().run(study, world, roots[0], 30, refit=boundary)
    assert int(mine.carry.belief.n[0]) == 3
    doc = dump_run(
        study,
        REGISTRY,
        executor="host",
        bindings=photomancy_bindings(),
        refit=boundary,
        kinds=KINDS,
    )
    run = load_run(doc.to_json(), photomancy_bindings(), KINDS)
    again = run.execute(truths, roots, 30, executor="host")
    assert histories_equal(mine.history(), again.history(0)).ok
    compiled = CompiledExecutor().run(study, truths, roots, 30, refit=boundary, chunk=7)
    assert histories_equal(mine.history(), compiled.history(0)).ok
